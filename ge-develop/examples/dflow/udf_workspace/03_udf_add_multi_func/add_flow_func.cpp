/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <iostream>
#include <mutex>

#include "flow_func/meta_multi_func.h"
#include "flow_func/flow_func_log.h"

namespace FlowFunc {
class AddFlowFunc : public MetaMultiFunc {
 public:
  AddFlowFunc() = default;
  ~AddFlowFunc() override = default;

  int32_t Init(const std::shared_ptr<MetaParams> &params) override {
    const auto get_ret = params->GetAttr("out_type", out_data_type_);
    if (get_ret != FLOW_FUNC_SUCCESS) {
      FLOW_FUNC_LOG_INFO("Attr out_type is not exist.");
      out_data_type_ = TensorDataType::DT_INT32;
    }
    set_output_count_ = 0U;
    return FLOW_FUNC_SUCCESS;
  }

  template <typename srcT, typename dstT>
  void Add1(srcT *src1, srcT *src2, dstT *dst, size_t count) {
    for (size_t i = 0; i < count; i++) {
      dst[i] = src1[i] + src2[i];
    }
  }

  template <typename srcT, typename dstT>
  void Add2(srcT *src1, srcT *src2, dstT *dst, size_t count) {
    for (size_t i = 0; i < count; i++) {
      dst[i] = src1[i] + src1[i] + src2[i];
    }
  }

  int32_t Proc1(const std::shared_ptr<MetaRunContext> &run_context,
                const std::vector<std::shared_ptr<FlowMsg>> &input_flow_msgs) {
    // there should be two input for add func
    const size_t input_num = 2;
    if (input_flow_msgs.size() != input_num) {
      FLOW_FUNC_LOG_ERROR("Input size is not 2.");
      return FLOW_FUNC_FAILED;
    }

    const auto &input1 = input_flow_msgs[0];
    const auto &input2 = input_flow_msgs[1];
    FLOW_FUNC_LOG_INFO("input1 ret code is %d", input1->GetRetCode());
    FLOW_FUNC_LOG_INFO("input2 ret code is %d", input2->GetRetCode());
    if (input1->GetRetCode() != 0 || input2->GetRetCode() != 0) {
      FLOW_FUNC_LOG_ERROR("Input ret code is not 0.");
      return FLOW_FUNC_FAILED;
    }
    MsgType msg_type1 = input1->GetMsgType();
    MsgType msg_type2 = input2->GetMsgType();
    if (msg_type1 != MsgType::MSG_TYPE_TENSOR_DATA || msg_type2 != MsgType::MSG_TYPE_TENSOR_DATA) {
      FLOW_FUNC_LOG_ERROR("Input msg type is not data.");
      return FLOW_FUNC_FAILED;
    }
    auto input_tensor1 = input1->GetTensor();
    auto input_tensor2 = input2->GetTensor();
    auto input_data_type1 = input_tensor1->GetDataType();
    auto input_data_type2 = input_tensor2->GetDataType();
    if (input_data_type1 != input_data_type2) {
      FLOW_FUNC_LOG_ERROR("Input data type is not same.");
      return FLOW_FUNC_FAILED;
    }
    auto input_shape1 = input_tensor1->GetShape();
    auto input_shape2 = input_tensor2->GetShape();
    if (input_shape1 != input_shape2) {
      FLOW_FUNC_LOG_ERROR("Input shape is not same.");
      return FLOW_FUNC_FAILED;
    }

    auto output_msg = run_context->AllocTensorMsg(input_shape1, out_data_type_);
    if (output_msg == nullptr) {
      FLOW_FUNC_LOG_ERROR("Fail to alloc tensor msg.");
      return -1;
    }
    auto output_tensor = output_msg->GetTensor();

    auto data_size1 = input_tensor1->GetDataSize();
    auto data_size2 = input_tensor2->GetDataSize();
    if (data_size1 == 0) {
      return run_context->SetOutput(0, output_msg);
    }
    auto input_data1 = input_tensor1->GetData();
    auto input_data2 = input_tensor2->GetData();
    auto output_data = output_tensor->GetData();

    switch (input_data_type1) {
      case TensorDataType::DT_INT8:
        Add1<int8_t, int8_t>(static_cast<int8_t *>(input_data1), static_cast<int8_t *>(input_data2),
                             static_cast<int8_t *>(output_data), data_size1 / sizeof(float));
        break;
      case TensorDataType::DT_UINT16:
        Add1<uint16_t, uint16_t>(static_cast<uint16_t *>(input_data1), static_cast<uint16_t *>(input_data2),
                                 static_cast<uint16_t *>(output_data), data_size1 / sizeof(uint16_t));
        break;
      case TensorDataType::DT_INT16:
        Add1<int16_t, int16_t>(static_cast<int16_t *>(input_data1), static_cast<int16_t *>(input_data2),
                               static_cast<int16_t *>(output_data), data_size1 / sizeof(int16_t));
        break;
      case TensorDataType::DT_UINT32:
        Add1<uint32_t, uint32_t>(static_cast<uint32_t *>(input_data1), static_cast<uint32_t *>(input_data2),
                                 static_cast<uint32_t *>(output_data), data_size1 / sizeof(uint32_t));
        break;
      case TensorDataType::DT_INT32:
        Add1<int32_t, int32_t>(static_cast<int32_t *>(input_data1), static_cast<int32_t *>(input_data2),
                               static_cast<int32_t *>(output_data), data_size1 / sizeof(int32_t));
        break;
      case TensorDataType::DT_INT64:
        Add1<int64_t, int64_t>(static_cast<int64_t *>(input_data1), static_cast<int64_t *>(input_data2),
                               static_cast<int64_t *>(output_data), data_size1 / sizeof(int64_t));
        break;
      case TensorDataType::DT_FLOAT:
        Add1<float, float>(static_cast<float *>(input_data1), static_cast<float *>(input_data2),
                           static_cast<float *>(output_data), data_size1 / sizeof(float));
        break;
      default:
        FLOW_FUNC_LOG_ERROR("Unsupported data type.");
        const int32_t ret_code = 100;
        output_msg->SetRetCode(ret_code);
        break;
    }
    std::unique_lock<std::mutex> lock(count_mutex_);
    set_output_count_++;
    return run_context->SetOutput(0, output_msg);
  }

  int32_t Proc2(const std::shared_ptr<MetaRunContext> &run_context,
                const std::vector<std::shared_ptr<FlowMsg>> &input_flow_msgs) {
    // there should be two input for add func
    const size_t input_num = 2;
    if (input_flow_msgs.size() != input_num) {
      FLOW_FUNC_LOG_ERROR("Input size is not 2.");
      return FLOW_FUNC_FAILED;
    }

    const auto &input1 = input_flow_msgs[0];
    const auto &input2 = input_flow_msgs[1];
    FLOW_FUNC_LOG_INFO("input1 ret code is ", input1->GetRetCode());
    FLOW_FUNC_LOG_INFO("input2 ret code is ", input2->GetRetCode());
    if (input1->GetRetCode() != 0 || input2->GetRetCode() != 0) {
      FLOW_FUNC_LOG_ERROR("Input ret code is not 0.");
      return FLOW_FUNC_FAILED;
    }
    MsgType msg_type1 = input1->GetMsgType();
    MsgType msg_type2 = input2->GetMsgType();
    if (msg_type1 != MsgType::MSG_TYPE_TENSOR_DATA || msg_type2 != MsgType::MSG_TYPE_TENSOR_DATA) {
      FLOW_FUNC_LOG_ERROR("Input msg type is not data.");
      return FLOW_FUNC_FAILED;
    }
    auto input_tensor1 = input1->GetTensor();
    auto input_tensor2 = input2->GetTensor();
    auto input_data_type1 = input_tensor1->GetDataType();
    auto input_data_type2 = input_tensor2->GetDataType();
    if (input_data_type1 != input_data_type2) {
      FLOW_FUNC_LOG_ERROR("Input data type is not same.");
      return FLOW_FUNC_FAILED;
    }
    auto input_shape1 = input_tensor1->GetShape();
    auto input_shape2 = input_tensor2->GetShape();
    if (input_shape1 != input_shape2) {
      FLOW_FUNC_LOG_ERROR("Input shape is not same.");
      return FLOW_FUNC_FAILED;
    }

    auto output_msg = run_context->AllocTensorMsg(input_shape1, out_data_type_);
    if (output_msg == nullptr) {
      FLOW_FUNC_LOG_ERROR("Fail to alloc tensor msg.");
      return -1;
    }
    auto output_tensor = output_msg->GetTensor();

    auto data_size1 = input_tensor1->GetDataSize();
    auto data_size2 = input_tensor2->GetDataSize();
    if (data_size1 == 0) {
      return run_context->SetOutput(0, output_msg);
    }
    auto input_data1 = input_tensor1->GetData();
    auto input_data2 = input_tensor2->GetData();
    auto output_data = output_tensor->GetData();

    switch (input_data_type1) {
      case TensorDataType::DT_INT8:
        Add2<int8_t, int8_t>(static_cast<int8_t *>(input_data1), static_cast<int8_t *>(input_data2),
                             static_cast<int8_t *>(output_data), data_size1 / sizeof(float));
        break;
      case TensorDataType::DT_UINT16:
        Add2<uint16_t, uint16_t>(static_cast<uint16_t *>(input_data1), static_cast<uint16_t *>(input_data2),
                                 static_cast<uint16_t *>(output_data), data_size1 / sizeof(uint16_t));
        break;
      case TensorDataType::DT_INT16:
        Add2<int16_t, int16_t>(static_cast<int16_t *>(input_data1), static_cast<int16_t *>(input_data2),
                               static_cast<int16_t *>(output_data), data_size1 / sizeof(int16_t));
        break;
      case TensorDataType::DT_UINT32:
        Add2<uint32_t, uint32_t>(static_cast<uint32_t *>(input_data1), static_cast<uint32_t *>(input_data2),
                                 static_cast<uint32_t *>(output_data), data_size1 / sizeof(uint32_t));
        break;
      case TensorDataType::DT_INT32:
        Add2<int32_t, int32_t>(static_cast<int32_t *>(input_data1), static_cast<int32_t *>(input_data2),
                               static_cast<int32_t *>(output_data), data_size1 / sizeof(int32_t));
        break;
      case TensorDataType::DT_INT64:
        Add2<int64_t, int64_t>(static_cast<int64_t *>(input_data1), static_cast<int64_t *>(input_data2),
                               static_cast<int64_t *>(output_data), data_size1 / sizeof(int64_t));
        break;
      case TensorDataType::DT_FLOAT:
        Add2<float, float>(static_cast<float *>(input_data1), static_cast<float *>(input_data2),
                           static_cast<float *>(output_data), data_size1 / sizeof(float));
        break;
      default:
        FLOW_FUNC_LOG_ERROR("Unsupported data type.");
        const int32_t ret_code = 100;
        output_msg->SetRetCode(ret_code);
        break;
    }
    std::unique_lock<std::mutex> lock(count_mutex_);
    set_output_count_++;
    return run_context->SetOutput(1, output_msg);
  }

 private:
  TensorDataType out_data_type_;
  std::mutex count_mutex_;
  uint32_t set_output_count_;
};

FLOW_FUNC_REGISTRAR(AddFlowFunc).RegProcFunc("Proc1", &AddFlowFunc::Proc1).RegProcFunc("Proc2", &AddFlowFunc::Proc2);
}  // namespace FlowFunc