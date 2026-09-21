/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flow_func_run_context.h"
#include <memory>
#include "common/udf_log.h"
#include "common/common_define.h"
#include "securec.h"
#include "flow_func_helper.h"
#include "flow_func_config_manager.h"
#include "ascend_hal.h"

namespace FlowFunc {
namespace {
constexpr size_t kMaxUserDataSize = 64U;
bool CheckAlignSizeValid(const uint32_t align) {
    constexpr size_t kMinAlignSize = 32U;
    constexpr size_t kMaxAlignSize = sizeof(RuntimeTensorDesc);
    if ((align < kMinAlignSize) || (align > kMaxAlignSize)) {
        UDF_LOG_ERROR("alloc failed, as align=%u is out of range[%zu, %zu]", align, kMinAlignSize, kMaxAlignSize);
        return false;
    }
    if (kMaxAlignSize % align != 0) {
        UDF_LOG_ERROR("alloc failed, as align=%u must can be divisible by %zu.", align, kMaxAlignSize);
        return false;
    }
    return true;
}
}

std::shared_ptr<FlowMsg> MetaRunContext::AllocTensorMsgWithAlign(const std::vector<int64_t> &shape,
    TensorDataType dataType, uint32_t align) {
    (void)shape;
    (void)dataType;
    (void)align;
    UDF_LOG_ERROR("Not supported, please implement the AllocTensorMsgWithAlign method.");
    return nullptr;
}

std::shared_ptr<FlowMsg> MetaRunContext::AllocTensorListMsg(const std::vector<std::vector<int64_t>> &shapes,
    const std::vector<TensorDataType> &dataTypes, uint32_t align) {
    (void)shapes;
    (void)dataTypes;
    (void)align;
    UDF_LOG_ERROR("Not supported, please implement the AllocTensorMsgWithAlign method.");
    return nullptr;
}

void MetaRunContext::RaiseException(int32_t expCode, uint64_t userContextId) {
    (void)expCode;
    (void)userContextId;
    UDF_LOG_ERROR("Not supported, please implement the RaiseException method.");
}

bool MetaRunContext::GetException(int32_t &expCode, uint64_t &userContextId) {
    (void)expCode;
    (void)userContextId;
    UDF_LOG_ERROR("Not supported, please implement the GetException method.");
    return false;
}

std::shared_ptr<FlowMsg> MetaRunContext::AllocRawDataMsg(int64_t size, uint32_t align) {
    (void)size;
    (void)align;
    UDF_LOG_ERROR("Not supported, please implement the AllocRawDataMsg method.");
    return std::shared_ptr<FlowMsg>(nullptr);
}

std::shared_ptr<FlowMsg> MetaRunContext::ToFlowMsg(std::shared_ptr<Tensor> tensor) {
    (void)tensor;
    UDF_LOG_ERROR("Not supported, please implement the ToFlowMsg method.");
    return std::shared_ptr<FlowMsg>(nullptr);
}

FlowFuncRunContext::FlowFuncRunContext(
    uint32_t device_id, std::shared_ptr<FlowFuncParams> params, WriterCallback writer_call_back, uint32_t processor_id)
    : MetaRunContext(), params_(std::move(params)), writer_call_back_(std::move(writer_call_back)), device_id_(device_id),
      processor_idx_(processor_id) {
}

int32_t FlowFuncRunContext::SetOutput(uint32_t out_idx, std::shared_ptr<FlowMsg> out_msg) {
    if (writer_call_back_ != nullptr) {
        return writer_call_back_(out_idx, out_msg);
    }
    UDF_LOG_ERROR("SetOutput failed for writer_call_back_ is null, instance name[%s]", params_->GetName());
    return FLOW_FUNC_FAILED;
}

std::shared_ptr<FlowMsg> FlowFuncRunContext::AllocTensorMsg(const std::vector<int64_t> &shape, TensorDataType data_type) {
    return MbufFlowMsg::AllocTensorMsg(shape, data_type, device_id_, input_mbuf_head_);
}

std::shared_ptr<FlowMsg> FlowFuncRunContext::AllocRawDataMsg(int64_t size, uint32_t align) {
    return MbufFlowMsg::AllocRawDataMsg(size, device_id_, input_mbuf_head_, align);
}

std::shared_ptr<FlowMsg> FlowFuncRunContext::ToFlowMsg(std::shared_ptr<Tensor> tensor) {
    std::shared_ptr<FlowMsg> flow_msg = nullptr;
    auto mbuf_tensor = std::dynamic_pointer_cast<MbufTensor>(tensor);
    if (mbuf_tensor != nullptr) {
        flow_msg = MbufFlowMsg::ToTensorMsg(mbuf_tensor->GetShape(), mbuf_tensor->GetDataType(),
            mbuf_tensor->SharedMbuf(), input_mbuf_head_);
    }
    return flow_msg;
}

std::shared_ptr<FlowMsg> FlowFuncRunContext::AllocTensorListMsg(const std::vector<std::vector<int64_t>> &shapes,
    const std::vector<TensorDataType> &data_types, uint32_t align) {
    if (!CheckAlignSizeValid(align)) {
        UDF_LOG_ERROR("Align size[%u] is invalid.", align);
        return nullptr;
    }
    return MbufFlowMsg::AllocTensorListMsg(shapes, data_types, device_id_, input_mbuf_head_, align);
}

std::shared_ptr<FlowMsg> FlowFuncRunContext::AllocTensorMsgWithAlign(const std::vector<int64_t> &shape,
    TensorDataType data_type, uint32_t align) {
    if (!CheckAlignSizeValid(align)) {
        UDF_LOG_ERROR("Align size[%u] is invalid.", align);
        return nullptr;
    }
    return MbufFlowMsg::AllocTensorMsg(shape, data_type, device_id_, input_mbuf_head_, align);
}

std::shared_ptr<FlowMsg> FlowFuncRunContext::AllocEmptyDataMsg(MsgType msg_type) {
    return AllocMbufEmptyDataMsg(msg_type);
}

std::shared_ptr<MbufFlowMsg> FlowFuncRunContext::AllocMbufEmptyDataMsg(MsgType msg_type) const {
    if (msg_type == MsgType::MSG_TYPE_TENSOR_DATA) {
        // alloc null tensor msg
        return MbufFlowMsg::AllocEmptyTensorMsg(device_id_, input_mbuf_head_);
    }
    UDF_LOG_ERROR("Unsupported msg type [%u].", static_cast<uint32_t>(msg_type));
    return nullptr;
}

int32_t FlowFuncRunContext::RunFlowModel(const char *model_key, const std::vector<std::shared_ptr<FlowMsg>> &input_msgs,
    std::vector<std::shared_ptr<FlowMsg>> &output_msgs, int32_t timeout) {
    if (model_key == nullptr) {
        UDF_LOG_ERROR("param model_key is null, instance name[%s].", params_->GetName());
        return FLOW_FUNC_ERR_PARAM_INVALID;
    }
    auto flow_model = params_->GetFlowModels(model_key);
    if (flow_model == nullptr) {
        UDF_LOG_ERROR("no flow model found, instance name[%s], model_key=%s.", params_->GetName(), model_key);
        return FLOW_FUNC_ERR_PARAM_INVALID;
    }
    int32_t ret = flow_model->Run(input_msgs, output_msgs, timeout);
    if (ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("flow model run failed, instance name[%s], model_key=%s.", params_->GetName(), model_key);
        return ret;
    }
    UDF_LOG_DEBUG("flow model run end, instance name[%s], model_key=%s.", params_->GetName(), model_key);
    return FLOW_FUNC_SUCCESS;
}

int32_t FlowFuncRunContext::SetInputMbufHead(const MbufInfo &mbuf_info) {
    if (mbuf_info.head_buf == nullptr) {
        UDF_LOG_ERROR("The mbuf_info.head_buf is nullptr, instance name[%s].", params_->GetName());
        return FLOW_FUNC_ERR_PARAM_INVALID;
    }
    if (mbuf_info.head_buf_len > kMaxMbufHeadLen) {
        UDF_LOG_ERROR("The mbuf_info.head_buf_len[%u] > %u, instance name[%s].",
            mbuf_info.head_buf_len, kMaxMbufHeadLen, params_->GetName());
        return FLOW_FUNC_ERR_PARAM_INVALID;
    }
    if (mbuf_info.head_buf_len < sizeof(MbufHeadMsg)) {
        UDF_LOG_ERROR("The mbuf_info.head_buf_len[%u] < headMsgLen[%zu], instance name[%s].",
            mbuf_info.head_buf_len, sizeof(MbufHeadMsg), params_->GetName());
        return FLOW_FUNC_ERR_PARAM_INVALID;
    }
    const auto ret = memcpy_s(input_mbuf_head_.head_buf, static_cast<size_t>(kMaxMbufHeadLen), mbuf_info.head_buf,
                              static_cast<size_t>(mbuf_info.head_buf_len));
    if (ret != EOK) {
        UDF_LOG_ERROR("Failed to memcpy mbuf priv info, ret[%d].", ret);
        return FLOW_FUNC_FAILED;
    }
    input_mbuf_head_.head_buf_len = mbuf_info.head_buf_len;
    return FLOW_FUNC_SUCCESS;
}

int32_t FlowFuncRunContext::CheckParamsForUserData(const void *data, size_t size, size_t offset) const {
    if (data == nullptr) {
        UDF_LOG_ERROR("The data is nullptr.");
        return FLOW_FUNC_ERR_PARAM_INVALID;
    }
    if (size == 0U) {
        UDF_LOG_ERROR("The size is 0, should in (0, 64].");
        return FLOW_FUNC_ERR_PARAM_INVALID;
    }
    if ((offset >= kMaxUserDataSize) || ((kMaxUserDataSize - offset) < size)) {
        UDF_LOG_ERROR("The size + offset need <= %zu, but size = %zu, offset = %zu.",
            kMaxUserDataSize, size, offset);
        return FLOW_FUNC_ERR_PARAM_INVALID;
    }
    if (input_mbuf_head_.head_buf_len == 0U) {
        UDF_LOG_ERROR("The context mbuf head is invalid.");
        return FLOW_FUNC_FAILED;
    }
    if (input_mbuf_head_.head_buf_len < kMaxUserDataSize) {
        UDF_LOG_ERROR("The mbuf head size[%u] < max user data size[%zu].",
            input_mbuf_head_.head_buf_len, kMaxUserDataSize);
        return FLOW_FUNC_ERR_MEM_BUF_ERROR;
    }
    return FLOW_FUNC_SUCCESS;
}

int32_t FlowFuncRunContext::GetUserData(void *data, size_t size, size_t offset) const {
    const auto ret = CheckParamsForUserData(data, size, offset);
    if (ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("Failed to get user data, ret[%d].", ret);
        return ret;
    }
    const auto cpy_ret = memcpy_s(data, size, input_mbuf_head_.head_buf + offset, size);
    if (cpy_ret != EOK) {
        UDF_LOG_ERROR("Failed to get user data, memcpy_s error, size[%zu], offset[%zu], memcpy_s ret[%d].",
                      size, offset, cpy_ret);
        return FLOW_FUNC_FAILED;
    }
    UDF_LOG_DEBUG("Success to get user data, size[%zu], offset[%zu].", size, offset);
    return FLOW_FUNC_SUCCESS;
}

int32_t FlowFuncRunContext::SetOutput(uint32_t out_idx, std::shared_ptr<FlowMsg> out_msg, const OutOptions &options) {
    return SetMultiOutputs(out_idx, {out_msg}, options);
}

int32_t FlowFuncRunContext::SetMultiOutputs(uint32_t out_idx, const std::vector<std::shared_ptr<FlowMsg>> &out_msgs,
    const OutOptions &options) {
    std::vector<std::shared_ptr<FlowMsg>> after_filter_out_msgs;
    int32_t ret = BalanceOptionFilter(options, out_msgs, after_filter_out_msgs);
    if (ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("call balance option filter failed, instance name[%s], out_idx=%u.", params_->GetName(), out_idx);
        return ret;
    }
    UDF_LOG_INFO("after balance option filter out msg size=%zu, instance name[%s]", after_filter_out_msgs.size(),
        params_->GetName());
    for (size_t i = 0; i < after_filter_out_msgs.size(); ++i) {
        int32_t sendRet = SetOutput(out_idx, after_filter_out_msgs[i]);
        if (sendRet != FLOW_FUNC_SUCCESS) {
            ret = sendRet;
            UDF_LOG_ERROR("set output, ret=%d, out_idx=%u, index=%zu", sendRet, out_idx, i);
        }
    }
    return ret;
}

int32_t FlowFuncRunContext::CheckAffinityPolicy(AffinityPolicy policy) const {
    if (params_->IsBalanceScatter()) {
        if (policy != AffinityPolicy::NO_AFFINITY) {
            UDF_LOG_ERROR("balance scatter node only support NO_AFFINITY(%d) policy but (%d), instance name[%s].",
                static_cast<int32_t>(AffinityPolicy::NO_AFFINITY),
                static_cast<int32_t>(policy), params_->GetName());
            return FLOW_FUNC_ERR_PARAM_INVALID;
        }
    } else if (params_->IsBalanceGather()) {
        if (policy == AffinityPolicy::NO_AFFINITY) {
            UDF_LOG_ERROR("balance gather node not support NO_AFFINITY(%d) policy, instance name[%s].",
                static_cast<int32_t>(AffinityPolicy::NO_AFFINITY), params_->GetName());
            return FLOW_FUNC_ERR_PARAM_INVALID;
        }
    } else {
        UDF_LOG_ERROR("Can't use balance config, as node is not balance node, instance name[%s].", params_->GetName());
        return FLOW_FUNC_ERR_PARAM_INVALID;
    }
    return FLOW_FUNC_SUCCESS;
}

int32_t FlowFuncRunContext::BalanceOptionFilter(const OutOptions &options,
    const std::vector<std::shared_ptr<FlowMsg>> &out_msgs,
    std::vector<std::shared_ptr<FlowMsg>> &after_filter_out_msgs) const {
    const auto *balance_config = options.GetBalanceConfig();
    if (balance_config == nullptr) {
        UDF_LOG_INFO("balance config is not exits, instance name[%s].", params_->GetName());
        after_filter_out_msgs = out_msgs;
        return FLOW_FUNC_SUCCESS;
    }
    if (!FlowFuncHelper::IsBalanceConfigValid(*balance_config)) {
        UDF_LOG_ERROR("balance config is invalid, instance name[%s].", params_->GetName());
        return FLOW_FUNC_ERR_PARAM_INVALID;
    }

    int32_t check_ret = CheckAffinityPolicy(balance_config->GetAffinityPolicy());
    if (check_ret != FLOW_FUNC_SUCCESS) {
        UDF_LOG_ERROR("Check AffinityPolicy[%d] failed, instance name[%s].",
            static_cast<int32_t>(AffinityPolicy::NO_AFFINITY), params_->GetName());
        return check_ret;
    }

    const auto &data_pos = balance_config->GetDataPos();
    if (data_pos.size() != out_msgs.size()) {
        UDF_LOG_ERROR("send flow msg but data pos size(%zu) != out_msgs size(%zu), instance name[%s].",
            data_pos.size(), out_msgs.size(), params_->GetName());
        return FLOW_FUNC_ERR_PARAM_INVALID;
    }
    const auto &balance_weight = balance_config->GetBalanceWeight();
    for (size_t i = 0; i < out_msgs.size(); ++i) {
        const auto &pos = data_pos[i];
        const auto &out_msg = out_msgs[i];
        auto mbuf_flow_msg = std::dynamic_pointer_cast<MbufFlowMsg>(out_msg);
        if (mbuf_flow_msg == nullptr) {
            UDF_LOG_ERROR("not support custom define flow msg now, instance name[%s].", params_->GetName());
            return FLOW_FUNC_ERR_PARAM_INVALID;
        }
        uint32_t data_label = 0;
        uint32_t route_label = 0;
        FlowFuncHelper::CalcRouteLabelAndDataLabel(*balance_config, pos, data_label, route_label);
        UDF_LOG_INFO("change out_msg[%zu] on pos[%d, %d] route_label=%u, data_label=%u, "
                     "matrix row_num=%d, col_num=%d, policy=%d",
            i, pos.first, pos.second, route_label, data_label, balance_weight.rowNum, balance_weight.colNum,
            static_cast<int32_t>(balance_config->GetAffinityPolicy()));
        mbuf_flow_msg->SetRouteLabel(route_label);
        mbuf_flow_msg->SetDataLabel(data_label);
    }
    after_filter_out_msgs = out_msgs;
    return FLOW_FUNC_SUCCESS;
}

void FlowFuncRunContext::RaiseException(int32_t exp_code, uint64_t user_context_id) {
    UDF_LOG_INFO("Raise and report exception event begin, exp_code=%d, user_context_id=%lu.",
                 exp_code, user_context_id);
    const MbufHeadMsg *head_msg = reinterpret_cast<MbufHeadMsg *>(input_mbuf_head_.head_buf +
        (input_mbuf_head_.head_buf_len - sizeof(MbufHeadMsg)));
    uint64_t current_trans_id = head_msg->trans_id;
    UDF_LOG_DEBUG("Get trans_id=%lu from input mbuf head.", current_trans_id);
    std::unique_lock<std::mutex> guard(raise_mutex_);
    if (trans_id_to_exception_raised_.find(current_trans_id) != trans_id_to_exception_raised_.end()) {
        UDF_LOG_WARN("TransId[%lu] is already raised. Skip to raise again.", current_trans_id);
        return;
    }
    event_summary event_info_summary = {};
    event_info_summary.pid = getpid();
    event_info_summary.event_id = static_cast<EVENT_ID>(UdfEvent::kEventIdRaiseException);
    event_info_summary.subevent_id = processor_idx_;

    UdfExceptionInfo report_exception_info = {};
    report_exception_info.trans_id = current_trans_id;
    report_exception_info.exp_code = exp_code;
    report_exception_info.user_context_id = user_context_id;
    report_exception_info.exp_context_size = input_mbuf_head_.head_buf_len;
    const auto mem_ret = memcpy_s(report_exception_info.exp_context, kMaxMbufHeadLen,
                                 input_mbuf_head_.head_buf, input_mbuf_head_.head_buf_len);
    if (mem_ret != EOK) {
        UDF_LOG_ERROR("[RaiseException] memcpy_s failed . ret=%d, trans_id[%lu], exceptionCode[%d], user_context_id[%lu]",
                      static_cast<int32_t>(mem_ret), current_trans_id, exp_code, user_context_id);
        return;
    }
    trans_id_to_exception_raised_[current_trans_id] = report_exception_info;
    event_info_summary.msg = reinterpret_cast<char *>(&current_trans_id);
    event_info_summary.msg_len = static_cast<uint32_t>(sizeof(current_trans_id));
    event_info_summary.dst_engine = FlowFuncConfigManager::GetConfig()->IsRunOnAiCpu() ? ACPU_LOCAL : CCPU_LOCAL;
    event_info_summary.grp_id = FlowFuncConfigManager::GetConfig()->GetMainSchedGroupId();
    drvError_t ret = halEschedSubmitEvent(params_->GetDeviceId(), &event_info_summary);
    if (ret != static_cast<int32_t>(DRV_ERROR_NONE)) {
        UDF_LOG_WARN("Send raise exception event failed, ret=%d, trans_id[%lu], exceptionCode[%d], user_context_id[%lu]",
                     static_cast<int32_t>(ret), current_trans_id, exp_code, user_context_id);
    } else {
        UDF_LOG_INFO("Send raise exception success. trans_id[%lu], exceptionCode[%d], user_context_id[%lu]",
                     current_trans_id, exp_code, user_context_id);
    }
}

bool FlowFuncRunContext::GetException(int32_t &exp_code, uint64_t &user_context_id) {
    if (!exception_existed_) {
        UDF_LOG_DEBUG("There is no valid exception during running procedure.");
        return false;
    }
    exp_code = exception_cache_.exp_code;
    user_context_id = exception_cache_.user_context_id;
    exception_existed_ = false;
    return true;
}

int32_t FlowFuncRunContext::GetExceptionByTransId(uint64_t trans_id, UdfExceptionInfo &exception_info) {
    std::unique_lock<std::mutex> guard(raise_mutex_);
    const auto iter = trans_id_to_exception_raised_.find(trans_id);
    if (iter == trans_id_to_exception_raised_.cend()) {
        UDF_LOG_ERROR("There is no exception for trans_id[%lu] waiting to report.", trans_id);
        return FLOW_FUNC_FAILED;
    }
    exception_info = iter->second;
    return FLOW_FUNC_SUCCESS;
}

bool FlowFuncRunContext::SelfRaiseChkAndDel(uint64_t trans_id) {
    std::unique_lock<std::mutex> guard(raise_mutex_);
    const auto iter = trans_id_to_exception_raised_.find(trans_id);
    if (iter == trans_id_to_exception_raised_.cend()) {
        return false;
    }
    trans_id_to_exception_raised_.erase(iter);
    return true;
}

void FlowFuncRunContext::RecordException(const UdfExceptionInfo &exp_info) {
    exception_cache_.exp_code = exp_info.exp_code;
    exception_cache_.user_context_id = exp_info.user_context_id;
    exception_existed_ = true;
}

}  // namespace FlowFunc
