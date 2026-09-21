/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "time_batch_flow_func.h"
#include "common/udf_log.h"
#include "common/data_utils.h"
#include "securec.h"
#include "flow_func/flow_func_dumper.h"

namespace FlowFunc {
namespace {
    constexpr int64_t kDynamicWindowMode = -1;
    constexpr int64_t kAddDimMode = -1;
}
int32_t TimeBatchFlowFunc::Init() {
    UDF_LOG_DEBUG("Begin init.");
    auto ret = context_->GetAttr("window", window_);
    if (ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("Failed to get attr [window].");
        return ret;
    }
    if ((window_ != kDynamicWindowMode) && (window_ <= 0)) {
        UDF_LOG_ERROR("Attr [window] value should be %ld or in (0, %ld], but got %ld.",
            kDynamicWindowMode, INT64_MAX, window_);
        return FLOW_FUNC_ERR_PARAM_INVALID;
    }
    ret = context_->GetAttr("batch_dim", batch_dim_);
    if (ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("Failed to get attr [batch_dim].");
        return ret;
    }
    if (batch_dim_ < kAddDimMode) {
        UDF_LOG_ERROR("Attr [batch_dim] value should in [%ld, %ld], but got %ld.",
            kAddDimMode, INT64_MAX, batch_dim_);
        return FLOW_FUNC_ERR_PARAM_INVALID;
    }
    ret = context_->GetAttr("drop_remainder", drop_remainder_);
    if (ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("Failed to get attr [drop_remainder].");
        return ret;
    }
    output_num_ = context_->GetOutputNum();
    if (output_num_ == 0U) {
        UDF_LOG_ERROR("Output num need > 0.");
        return FLOW_FUNC_ERR_PARAM_INVALID;
    }
    input_cache_.clear();
    UDF_LOG_DEBUG("End init, window = %ld, batch_dim = %ld, drop_remainder = %d.",
        window_, batch_dim_, drop_remainder_);
    return FLOW_FUNC_SUCCESS;
}

bool TimeBatchFlowFunc::IsEqualFlowInfo(const std::shared_ptr<FlowMsg> &msg0,
    const std::shared_ptr<FlowMsg> &msg1) const {
    if (msg0->GetStartTime() != msg1->GetStartTime()) {
        UDF_LOG_ERROR("The msg0 start time[%lu] and msg1 start time[%lu] is not equal.",
            msg0->GetStartTime(), msg1->GetStartTime());
        return false;
    }
    if (msg0->GetEndTime() != msg1->GetEndTime()) {
        UDF_LOG_ERROR("The msg0 end time[%lu] and msg1 end time[%lu] is not equal.",
            msg0->GetEndTime(), msg1->GetEndTime());
        return false;
    }
    if (msg0->GetFlowFlags() != msg1->GetFlowFlags()) {
        UDF_LOG_ERROR("The msg0 flow flag[%u] and msg1 flow flag[%u] is not equal.",
            msg0->GetFlowFlags(), msg1->GetFlowFlags());
        return false;
    }
    return true;
}

void TimeBatchFlowFunc::ResetState() {
    input_cache_.clear();
    start_time_ = 0U;
    end_time_ = 0U;
    is_time_batch_ok_ = false;
    published_out_num_ = 0U;
    is_empty_msgs_ = false;
    is_eos_ = false;
    UDF_LOG_DEBUG("Reset state success.");
}

int32_t TimeBatchFlowFunc::CheckFlowInfo(const std::vector<std::shared_ptr<FlowMsg>> &input_msgs) const {
    const auto &input_msg0 = input_msgs[0];
    const uint64_t start_time0 = input_msg0->GetStartTime();
    const uint64_t end_time0 = input_msg0->GetEndTime();
    if (start_time0 > end_time0) {
        UDF_LOG_ERROR("The input start time[%lu] can not be greater than end time[%lu].", start_time0, end_time0);
        return FLOW_FUNC_ERR_PARAM_INVALID;
    }
    if ((!input_cache_.empty()) && (start_time0 < end_time_)) {
        UDF_LOG_ERROR("The current input start time[%lu] can not be less than last input end time[%lu].",
            start_time0, end_time_);
        return FLOW_FUNC_ERR_PARAM_INVALID;
    }
    for (size_t i = 1U; i < input_msgs.size(); ++i) {
        if (!IsEqualFlowInfo(input_msg0, input_msgs[i])) {
            UDF_LOG_ERROR("Input[%zu] msg data flow info is not equal input[0] msg data flow info.", i);
            return FLOW_FUNC_ERR_PARAM_INVALID;
        }
    }
    return FLOW_FUNC_SUCCESS;
}

int32_t TimeBatchFlowFunc::IsOkShapeForTimeBatch(const std::vector<int64_t> &base_shape,
    const std::vector<int64_t> &shape) const {
    if (shape.size() != base_shape.size()) {
        UDF_LOG_ERROR("Shape size is not equal.");
        return FLOW_FUNC_ERR_PARAM_INVALID;
    }
    if (batch_dim_ == kAddDimMode) {
        if (shape != base_shape) {
            UDF_LOG_ERROR("Shape is not equal.");
            return FLOW_FUNC_ERR_PARAM_INVALID;
        }
    } else {
        for (size_t i = 0U; i < shape.size(); ++i) {
            if (static_cast<int64_t>(i) == batch_dim_) {
                continue;
            }
            if (shape[i] != base_shape[i]) {
                UDF_LOG_ERROR("Shape dim[%zu] is not equal.", i);
                return FLOW_FUNC_ERR_PARAM_INVALID;
            }
        }
    }
    return FLOW_FUNC_SUCCESS;
}

int32_t TimeBatchFlowFunc::CheckTensorInfo(const std::vector<std::shared_ptr<FlowMsg>> &input_msgs) const {
    for (size_t i = 0U; i < input_msgs.size(); ++i) {
        const auto input_i = input_msgs[i]->GetTensor();
        if (input_i->GetElementCnt() <= 0) {
            UDF_LOG_ERROR("Input[%zu] element cnt[%ld] <= 0.", i, input_i->GetElementCnt());
            return FLOW_FUNC_ERR_PARAM_INVALID;
        }
        const auto &shape_i = input_i->GetShape();
        if (!input_cache_.empty()) {
            const auto ret = IsOkShapeForTimeBatch(input_cache_[i][0]->GetTensor()->GetShape(), shape_i);
            if (ret != FLOW_FUNC_SUCCESS) {
                UDF_LOG_ERROR("Input[%zu] shape is invalid for time batch.", i);
                return FLOW_FUNC_ERR_PARAM_INVALID;
            }
            if (input_i->GetDataType() != input_cache_[i][0]->GetTensor()->GetDataType()) {
                UDF_LOG_ERROR("Input[%zu] data type[%d] is not equal to last input data type[%d].",
                    i, input_i->GetDataType(), input_cache_[i][0]->GetTensor()->GetDataType());
                return FLOW_FUNC_ERR_PARAM_INVALID;
            }
        } else if (batch_dim_ >= static_cast<int64_t>(shape_i.size())) {
            UDF_LOG_ERROR("The batch dim[%ld] need less than input[%zu] shape size[%zu].",
                batch_dim_, i, shape_i.size());
            return FLOW_FUNC_ERR_PARAM_INVALID;
        }
    }
    return FLOW_FUNC_SUCCESS;
}

int32_t TimeBatchFlowFunc::CheckInput(const std::vector<std::shared_ptr<FlowMsg>> &input_msgs) {
    if (input_msgs.size() != output_num_) {
        UDF_LOG_ERROR("Input num [%zu] is not equal output num [%zu].", input_msgs.size(), output_num_);
        return FLOW_FUNC_ERR_PARAM_INVALID;
    }
    for (size_t i = 0U; i < input_msgs.size(); ++i) {
        if (input_msgs[i] == nullptr) {
            UDF_LOG_ERROR("Input[%zu] msg is nullptr.", i);
            return FLOW_FUNC_ERR_PARAM_INVALID;
        }
        const auto ret_code = input_msgs[i]->GetRetCode();
        if (ret_code != 0) {
            UDF_LOG_ERROR("Input[%zu] is invalid, error code[%d].", i, ret_code);
            return ret_code;
        }
        if (is_empty_msgs_ != (input_msgs[i]->GetTensor() == nullptr)) {
            if (i == 0U) {
                is_empty_msgs_ = true;
            } else {
                UDF_LOG_ERROR("Input[%zu] is empty:[%d], but last input is empty:[%d].", i,
                    (input_msgs[i]->GetTensor() == nullptr), is_empty_msgs_);
                return FLOW_FUNC_ERR_PARAM_INVALID;
            }
        }
    }
    if (is_empty_msgs_) {
        UDF_LOG_DEBUG("Current input is empty msg.");
        return FLOW_FUNC_SUCCESS;
    }
    if (!input_cache_.empty() && (input_msgs.size() != input_cache_.size())) {
        UDF_LOG_ERROR("Input current input num is %zu, but first %zu times input num is %zu.",
            input_msgs.size(), input_cache_[0].size(), input_cache_.size());
        return FLOW_FUNC_ERR_PARAM_INVALID;
    }
    if (CheckFlowInfo(input_msgs) != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("Input flow info is invalid.");
        return FLOW_FUNC_ERR_PARAM_INVALID;
    }
    return CheckTensorInfo(input_msgs);
}

int32_t TimeBatchFlowFunc::UpdateState(const std::vector<std::shared_ptr<FlowMsg>> &input_msgs) {
    const auto &input_msg0 = input_msgs[0];
    if (is_empty_msgs_) {
        const auto flow_flags = input_msg0->GetFlowFlags();
        if ((flow_flags & static_cast<uint32_t>(FlowFlag::FLOW_FLAG_EOS)) != 0U) {
            is_time_batch_ok_ = true;
            is_eos_ = true;
            return FLOW_FUNC_SUCCESS;
        } else {
            UDF_LOG_ERROR("The current input is empty msg, but not EOS.");
            return FLOW_FUNC_ERR_PARAM_INVALID;
        }
    }
    if (input_cache_.empty()) {
        start_time_ = input_msg0->GetStartTime();
    }
    end_time_ = input_msg0->GetEndTime();
    uint64_t current_window = end_time_ - start_time_;
    bool check_window = ((window_ > 0) && (current_window > static_cast<uint64_t>(window_)));
    if (check_window) {
        UDF_LOG_ERROR("The current window[%lu] is more than the window[%ld].", current_window, window_);
        return FLOW_FUNC_ERR_PARAM_INVALID;
    }
    check_window = ((window_ > 0) && (current_window == static_cast<uint64_t>(window_)));
    if (check_window) {
        is_time_batch_ok_ = true;
    } else {
        const auto flow_flags = input_msg0->GetFlowFlags();
        if ((flow_flags & static_cast<uint32_t>(FlowFlag::FLOW_FLAG_EOS)) != 0U) {
            is_eos_ = true;
            is_time_batch_ok_ = true;
        }
        if ((flow_flags & static_cast<uint32_t>(FlowFlag::FLOW_FLAG_SEG)) != 0U) {
            is_time_batch_ok_ = true;
        }
    }
    if (input_cache_.empty()) {
        for (size_t i = 0U; i < input_msgs.size(); ++i) {
            std::vector<std::shared_ptr<FlowMsg>> inputs;
            inputs.emplace_back(input_msgs[i]);
            input_cache_.emplace_back(inputs);
        }
    } else {
        for (size_t i = 0U; i < input_msgs.size(); ++i) {
            input_cache_[i].emplace_back(input_msgs[i]);
        }
    }
    return FLOW_FUNC_SUCCESS;
}

void TimeBatchFlowFunc::CalcCopyParams(const std::vector<std::shared_ptr<FlowMsg>> &input,
                                       std::vector<int64_t> &input_copy_sizes,
                                       std::vector<int64_t> &output_shape,
                                       int64_t &output_flat_dim0, uint32_t &max_step) const
{
    const auto input0_tensor = input[0]->GetTensor();
    const auto &input0_shape = input0_tensor->GetShape();
    output_shape = input0_shape;
    max_step = std::dynamic_pointer_cast<MbufFlowMsg>(input[0])->GetStepId();
    if (batch_dim_ == kAddDimMode) {
        output_flat_dim0 = 1;
        output_shape.insert(output_shape.cbegin(), static_cast<int64_t>(input.size()));
        for (size_t i = 0U; i < input.size(); ++i) {
            input_copy_sizes.push_back(input[i]->GetTensor()->GetDataSize());
            const auto step = std::dynamic_pointer_cast<MbufFlowMsg>(input[i])->GetStepId();
            max_step = ((FlowFuncDumpManager::IsInDumpStep(step)) && (step > max_step)) ? step : max_step;
        }
    } else {
        int64_t input_i_copy_num = 1;
        for (size_t i = batch_dim_; i < input0_shape.size(); ++i) {
            input_i_copy_num *= input0_shape[i];
        }
        const auto input0_element_cnt = input0_tensor->GetElementCnt();
        output_flat_dim0 = input0_element_cnt / input_i_copy_num;
        input_copy_sizes.push_back((input0_tensor->GetDataSize() / input0_element_cnt * input_i_copy_num));
        for (size_t i = 1U; i < input.size(); ++i) {
            const auto input_i_tensor = input[i]->GetTensor();
            const auto &input_i_shape = input_i_tensor->GetShape();
            output_shape[batch_dim_] += input_i_shape[batch_dim_];
            input_i_copy_num = 1;
            for (size_t j = batch_dim_; j < input_i_shape.size(); ++j) {
                input_i_copy_num *= input_i_shape[j];
            }
            input_copy_sizes.push_back((input_i_tensor->GetDataSize() / input_i_tensor->GetElementCnt()) *
                input_i_copy_num);
            const auto step = std::dynamic_pointer_cast<MbufFlowMsg>(input[i])->GetStepId();
            max_step = ((FlowFuncDumpManager::IsInDumpStep(step)) && (step > max_step)) ? step : max_step;
        }
    }
}

int32_t TimeBatchFlowFunc::TimeBatch(const std::vector<std::shared_ptr<FlowMsg>> &input, const uint32_t out_index) {
    std::vector<int64_t> input_copy_sizes;
    std::vector<int64_t> output_shape;
    int64_t output_flat_dim0 = 0;
    uint32_t step_id = 0;
    CalcCopyParams(input, input_copy_sizes, output_shape, output_flat_dim0, step_id);
    auto output_msg = context_->AllocTensorMsg(output_shape, input[0]->GetTensor()->GetDataType());
    if (output_msg == nullptr) {
        UDF_LOG_ERROR("Alloc output msg failed.");
        return FLOW_FUNC_FAILED;
    }
    auto output_tensor = output_msg->GetTensor();
    auto output_data = output_tensor->GetData();
    auto out_size = output_tensor->GetDataSize();
    std::vector<int64_t> copyed_sizes(input.size(), 0);
    int64_t output_copyed_size = 0;
    for (int64_t dim = 0; dim < output_flat_dim0; ++dim) {
        for (size_t i = 0U; i < input.size(); ++i) {
            const auto ret = memcpy_s((static_cast<uint8_t *>(output_data) + output_copyed_size), out_size,
                static_cast<uint8_t *>(input[i]->GetTensor()->GetData()) + copyed_sizes[i], input_copy_sizes[i]);
            if (ret != EOK) {
                UDF_LOG_ERROR("The memcpy_s error, out addr[%p], out size[%lu], in addr[%p], in "
                    "size[%ld], ret[%d].", (static_cast<uint8_t *>(output_data) + output_copyed_size), out_size,
                    static_cast<uint8_t *>(input[i]->GetTensor()->GetData()) + copyed_sizes[i],
                    input_copy_sizes[i], ret);
                return FLOW_FUNC_FAILED;
            }
            out_size -= input_copy_sizes[i];
            copyed_sizes[i] += input_copy_sizes[i];
            output_copyed_size += input_copy_sizes[i];
        }
    }
    output_msg->SetStartTime(start_time_);
    output_msg->SetEndTime(end_time_);
    std::dynamic_pointer_cast<MbufFlowMsg>(output_msg)->SetStepId(step_id);
    const auto ret = context_->SetOutput(out_index, output_msg);
    if (ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("Set output[%u] msg failed, ret = %d.", out_index, ret);
        return ret;
    }
    published_out_num_++;
    return FLOW_FUNC_SUCCESS;
}

int32_t TimeBatchFlowFunc::TimeBatchAll() {
    for (size_t i = 0U; i < input_cache_.size(); ++i) {
        const auto ret = TimeBatch(input_cache_[i], i);
        if (ret != FLOW_FUNC_SUCCESS) {
            UDF_LOG_ERROR("Time batch input[%zu] failed, ret = %d.", i, ret);
            return ret;
        }
    }
    return FLOW_FUNC_SUCCESS;
}

int32_t TimeBatchFlowFunc::PublishErrorOut(const int32_t error_code) const {
    // alloc null tensor msg output for error report
    auto error_output_msg = context_->AllocEmptyDataMsg(MsgType::MSG_TYPE_TENSOR_DATA);
    if (error_output_msg != nullptr) {
        error_output_msg->SetRetCode(error_code);
        for (size_t i = published_out_num_; i < output_num_; ++i) {
            const auto ret = context_->SetOutput(i, error_output_msg);
            if (ret != FLOW_FUNC_SUCCESS) {
                UDF_LOG_ERROR("Failed to set error output[%zu], error_code = %d, ret = %d", i, error_code, ret);
                return ret;
            }
        }
    } else {
        UDF_LOG_ERROR("Failed to alloc empty data msg.");
        return FLOW_FUNC_FAILED;
    }
    return FLOW_FUNC_SUCCESS;
}

int32_t TimeBatchFlowFunc::PublishEmptyEosOut() {
    auto empty_data_msg = context_->AllocEmptyDataMsg(MsgType::MSG_TYPE_TENSOR_DATA);
    if (empty_data_msg != nullptr) {
        empty_data_msg->SetFlowFlags(static_cast<uint32_t>(FlowFlag::FLOW_FLAG_EOS));
        for (size_t i = 0U; i < output_num_; ++i) {
            const auto ret = context_->SetOutput(i, empty_data_msg);
            if (ret != FLOW_FUNC_SUCCESS) {
                UDF_LOG_ERROR("Failed to set empty eos output[%zu], ret = %d", i, ret);
                return PublishErrorOut(ret);
            }
            published_out_num_++;
        }
        UDF_LOG_DEBUG("Success to publish empty data eos msg.");
        return FLOW_FUNC_SUCCESS;
    }
    UDF_LOG_ERROR("Failed to alloc empty data msg.");
    return PublishErrorOut(FLOW_FUNC_FAILED);
}

int32_t TimeBatchFlowFunc::Proc(const std::vector<std::shared_ptr<FlowMsg>> &input_msgs) {
    UDF_LOG_DEBUG("Begin proc.");
    auto ret = CheckInput(input_msgs);
    if (ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("The input is invalid.");
        ret = PublishErrorOut(ret);
        ResetState();
        return ret;
    }
    ret = UpdateState(input_msgs);
    if (ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("The input is invalid.");
        ret = PublishErrorOut(ret);
        ResetState();
        return ret;
    }
    uint64_t current_window = end_time_ - start_time_;
    if (is_time_batch_ok_) {
        if (is_empty_msgs_ && input_cache_.empty()) {
            ret = PublishEmptyEosOut();
            ResetState();
            return ret;
        }
        bool check_window = ((window_ > 0) && (current_window < static_cast<uint64_t>(window_)));
        if (check_window && drop_remainder_) {
            UDF_LOG_DEBUG(
                "The current data window[%lu] < time batch window[%ld] and drop flag is true, data will be drop.",
                current_window, window_);
            if (is_eos_) {
                ret = PublishEmptyEosOut();
            }
            ResetState();
            return ret;
        }
        ret = TimeBatchAll();
        if (ret != FLOW_FUNC_SUCCESS) {
            ret = PublishErrorOut(ret);
        }
        ResetState();
        UDF_LOG_DEBUG("End proc, ret[%d].", ret);
        return ret;
    }
    UDF_LOG_INFO("End proc, the current data window[%lu], time batch window[%ld], "
        "will continue to wait for data.", current_window, window_);
    return FLOW_FUNC_SUCCESS;
}

REGISTER_FLOW_FUNC("_BuiltIn_TimeBatch", TimeBatchFlowFunc);
}