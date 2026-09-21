/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flow_func/meta_flow_func.h"
#include "flow_func/flow_func_log.h"
#include "common/inner_error_codes.h"

namespace FlowFuncSt {
class FlowFuncStub : public FlowFunc::MetaFlowFunc {
 public:
  FlowFuncStub() = default;

  ~FlowFuncStub() override = default;

  int32_t Init() override {
    auto get_ret = context_->GetAttr("out_type", out_data_type_);
    if (get_ret != FlowFunc::FLOW_FUNC_SUCCESS) {
      return get_ret;
    }
    // just support cast to float and int64
    if ((out_data_type_ != FlowFunc::TensorDataType::DT_FLOAT) &&
        (out_data_type_ != FlowFunc::TensorDataType::DT_INT64)) {
      FLOW_FUNC_RUN_LOG_ERROR("outDataType must is invalid.");
      return -1;
    }

    (void)context_->GetAttr("need_re_init_attr", need_re_init_);
    (void)context_->GetAttr("_test_reshape", test_reshape_);
    if (need_re_init_ && init_times_ == 0) {
      ++init_times_;
      return FlowFunc::FLOW_FUNC_ERR_INIT_AGAIN;
    }
    FLOW_FUNC_RUN_LOG_INFO("init success, device id[%d].", context_->GetRunningDeviceId());
    return FlowFunc::FLOW_FUNC_SUCCESS;
  }

  template <typename srcT, typename dstT>
  void Cast(srcT *src, dstT *dst, size_t count) {
    for (size_t i = 0; i < count; ++i) {
      dst[i] = static_cast<dstT>(src[i]);
    }
  }

  template <typename T>
  void Cast(T *src, void *dst, size_t count) {
    if (out_data_type_ == FlowFunc::TensorDataType::DT_FLOAT) {
      Cast(src, static_cast<float *>(dst), count);
    } else if (out_data_type_ == FlowFunc::TensorDataType::DT_INT64) {
      Cast(src, static_cast<int64_t *>(dst), count);
    }
  }

  int32_t CheckUserData() {
    if (context_->GetUserData(nullptr, sizeof(int32_t)) == FlowFunc::FLOW_FUNC_SUCCESS) {
      return FlowFunc::FLOW_FUNC_FAILED;
    }
    int32_t user_data = 1;
    if (context_->GetUserData(&user_data, 0) == FlowFunc::FLOW_FUNC_SUCCESS) {
      return FlowFunc::FLOW_FUNC_FAILED;
    }
    if (context_->GetUserData(&user_data, sizeof(int32_t), 64) == FlowFunc::FLOW_FUNC_SUCCESS) {
      return FlowFunc::FLOW_FUNC_FAILED;
    }
    return FlowFunc::FLOW_FUNC_SUCCESS;
  }

  int32_t Proc(const std::vector<std::shared_ptr<FlowFunc::FlowMsg>> &input_tensors) override {
    int32_t exp_code = 0;
    uint64_t context_id = 0;
    if (context_->GetException(exp_code, context_id)) {
      FLOW_FUNC_LOG_ERROR("Get exception exp_code[%d] context_id[%lu].", exp_code, context_id);
      return FlowFunc::FLOW_FUNC_SUCCESS;
    }
    // cast only have 1 input
    if (input_tensors.size() != 1) {
      FLOW_FUNC_LOG_ERROR("input tensor size is invalid, size=%zu.", input_tensors.size());
      return FlowFunc::FLOW_FUNC_ERR_PARAM_INVALID;
    }

    auto input_tensor = input_tensors[0];
    // invalid input, trans to output
    if (input_tensor->GetRetCode() != 0) {
      auto ret = context_->SetOutput(0, input_tensor);
      return ret;
    }
    auto tensor = input_tensor->GetTensor();
    auto input_data_type = tensor->GetDataType();
    // type same
    if (input_data_type == out_data_type_) {
      auto ret = context_->SetOutput(0, input_tensor);
      return ret;
    }
    if (CheckUserData() != FlowFunc::FLOW_FUNC_SUCCESS) {
      FLOW_FUNC_LOG_ERROR("Failed to CheckUserData.");
      return FlowFunc::FLOW_FUNC_FAILED;
    }
    int32_t user_data = 1;
    if (context_->GetUserData(&user_data, sizeof(int32_t)) != FlowFunc::FLOW_FUNC_SUCCESS) {
      return FlowFunc::FLOW_FUNC_FAILED;
    }
    if (user_data != 0) {
      return FlowFunc::FLOW_FUNC_FAILED;
    }
    auto &input_shape = tensor->GetShape();
    auto output_tensor = context_->AllocTensorMsgWithAlign(input_shape, out_data_type_, 16);
    if (output_tensor != nullptr) {
      FLOW_FUNC_LOG_ERROR("AllocTensorMsg is expected failed.");
      return FlowFunc::FLOW_FUNC_FAILED;
    }
    output_tensor = context_->AllocTensorMsgWithAlign(input_shape, out_data_type_, 65);
    if (output_tensor != nullptr) {
      FLOW_FUNC_LOG_ERROR("AllocTensorMsg is expected failed.");
      return FlowFunc::FLOW_FUNC_FAILED;
    }
    output_tensor = context_->AllocTensorMsgWithAlign(input_shape, out_data_type_, 256);
    if (output_tensor == nullptr) {
      FLOW_FUNC_LOG_ERROR("AllocTensorMsg failed.");
      return FlowFunc::FLOW_FUNC_FAILED;
    }

    auto data_size = tensor->GetDataSize();
    if (data_size == 0) {
      FLOW_FUNC_LOG_WARN("tensor is 0, no need calc.");
      return context_->SetOutput(0, output_tensor);
    }
    auto input_data = tensor->GetData();
    auto output_data = output_tensor->GetTensor()->GetData();
    FLOW_FUNC_LOG_INFO("input_data_type=%d.", static_cast<int32_t>(input_data_type));
    switch (input_data_type) {
      case FlowFunc::TensorDataType::DT_FLOAT:
        Cast(static_cast<float *>(input_data), output_data, data_size / sizeof(float));
        break;
      case FlowFunc::TensorDataType::DT_INT16:
        Cast(static_cast<int16_t *>(input_data), output_data, data_size / sizeof(int16_t));
        break;
      case FlowFunc::TensorDataType::DT_UINT16:
        Cast(static_cast<uint16_t *>(input_data), output_data, data_size / sizeof(uint16_t));
        break;
      case FlowFunc::TensorDataType::DT_UINT32:
        Cast(static_cast<uint32_t *>(input_data), output_data, data_size / sizeof(uint32_t));
        break;
      case FlowFunc::TensorDataType::DT_INT8:
        Cast(static_cast<int8_t *>(input_data), output_data, data_size / sizeof(int8_t));
        break;
      default:
        // not support
        output_tensor->SetRetCode(FlowFunc::FLOW_FUNC_FAILED);
        break;
    }
    if (test_reshape_) {
      int64_t element_cnt = 1;
      for (int64_t dim : input_shape) {
        element_cnt *= dim;
      }
      int32_t ret = output_tensor->GetTensor()->Reshape({element_cnt, 2});
      if (ret == FlowFunc::FLOW_FUNC_SUCCESS) {
        FLOW_FUNC_LOG_ERROR("expect reshape failed, but success.");
        return FlowFunc::FLOW_FUNC_FAILED;
      }

      ret = output_tensor->GetTensor()->Reshape({element_cnt, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
      if (ret == FlowFunc::FLOW_FUNC_SUCCESS) {
        FLOW_FUNC_LOG_ERROR("expect reshape failed as over max dims, but success.");
        return FlowFunc::FLOW_FUNC_FAILED;
      }

      ret = output_tensor->GetTensor()->Reshape({element_cnt});
      if (ret != FlowFunc::FLOW_FUNC_SUCCESS) {
        FLOW_FUNC_LOG_ERROR("expect reshape success, but failed.");
        return FlowFunc::FLOW_FUNC_FAILED;
      }
    }
    FLOW_FUNC_LOG_DEBUG("proc end.");
    FlowFunc::OutOptions options;
    return context_->SetOutput(0, output_tensor, options);
  }

 protected:
  bool need_re_init_ = false;
  bool test_reshape_ = false;
  uint32_t init_times_ = 0;
  FlowFunc::TensorDataType out_data_type_ = FlowFunc::TensorDataType::DT_FLOAT;
};

class FlowFuncStubWithDummyQ : public FlowFuncStub {
 public:
  int32_t Proc(const std::vector<std::shared_ptr<FlowFunc::FlowMsg>> &input_tensors) override {
    // cast only have 1 input
    if (input_tensors.size() != 1) {
      FLOW_FUNC_LOG_ERROR("input tensor size is invalid, size=%zu.", input_tensors.size());
      return FlowFunc::FLOW_FUNC_ERR_PARAM_INVALID;
    }

    auto input_tensor = input_tensors[0];
    // invalid input, trans to output
    if (input_tensor->GetRetCode() != 0) {
      auto ret = context_->SetOutput(0, input_tensor);
      return ret;
    }
    auto tensor = input_tensor->GetTensor();
    auto input_data_type = tensor->GetDataType();
    // type same
    if (input_data_type == out_data_type_) {
      auto ret = context_->SetOutput(0, input_tensor);
      context_->SetOutput(1, input_tensor);
      return ret;
    }
    if (CheckUserData() != FlowFunc::FLOW_FUNC_SUCCESS) {
      FLOW_FUNC_LOG_ERROR("Failed to CheckUserData.");
      return FlowFunc::FLOW_FUNC_FAILED;
    }
    int32_t user_data = 1;
    if (context_->GetUserData(&user_data, sizeof(int32_t)) != FlowFunc::FLOW_FUNC_SUCCESS) {
      return FlowFunc::FLOW_FUNC_FAILED;
    }
    if (user_data != 0) {
      return FlowFunc::FLOW_FUNC_FAILED;
    }
    auto &input_shape = tensor->GetShape();
    auto output_tensor = context_->AllocTensorMsg(input_shape, out_data_type_);
    if (output_tensor == nullptr) {
      FLOW_FUNC_LOG_ERROR("AllocTensorMsg failed.");
      return FlowFunc::FLOW_FUNC_FAILED;
    }

    auto data_size = tensor->GetDataSize();
    if (data_size == 0) {
      FLOW_FUNC_LOG_WARN("tensor is 0, no need calc.");
      return context_->SetOutput(0, output_tensor);
    }
    auto input_data = tensor->GetData();
    auto output_data = output_tensor->GetTensor()->GetData();
    FLOW_FUNC_LOG_INFO("input_data_type=%d.", static_cast<int32_t>(input_data_type));
    switch (input_data_type) {
      case FlowFunc::TensorDataType::DT_FLOAT:
        Cast(static_cast<float *>(input_data), output_data, data_size / sizeof(float));
        break;
      case FlowFunc::TensorDataType::DT_INT16:
        Cast(static_cast<int16_t *>(input_data), output_data, data_size / sizeof(int16_t));
        break;
      case FlowFunc::TensorDataType::DT_UINT16:
        Cast(static_cast<uint16_t *>(input_data), output_data, data_size / sizeof(uint16_t));
        break;
      case FlowFunc::TensorDataType::DT_UINT32:
        Cast(static_cast<uint32_t *>(input_data), output_data, data_size / sizeof(uint32_t));
        break;
      case FlowFunc::TensorDataType::DT_INT8:
        Cast(static_cast<int8_t *>(input_data), output_data, data_size / sizeof(int8_t));
        break;
      default:
        // not support
        output_tensor->SetRetCode(FlowFunc::FLOW_FUNC_FAILED);
        break;
    }
    FLOW_FUNC_LOG_DEBUG("proc end.");
    FlowFunc::OutOptions options;
    context_->SetMultiOutputs(0, {output_tensor}, options);
    return context_->SetOutput(1, output_tensor);
  }
};
REGISTER_FLOW_FUNC("FlowFuncSt", FlowFuncStub);
REGISTER_FLOW_FUNC("FlowFuncStWithDummy", FlowFuncStubWithDummyQ);
}  // namespace FlowFuncSt