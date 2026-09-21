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
    return FLOW_FUNC_SUCCESS;
  }

  int32_t Proc(const std::shared_ptr<MetaRunContext> &run_context,
               const std::vector<std::shared_ptr<FlowMsg>> &input_flow_msgs) {
    // there should be two input for add func
    if (input_flow_msgs.size() != 1) {
      FLOW_FUNC_LOG_ERROR("Input size is not 1.");
      return FLOW_FUNC_FAILED;
    }

    const auto &input1 = input_flow_msgs[0];
    if (input1->GetRetCode() != 0) {
      FLOW_FUNC_LOG_ERROR("Input ret code is not 0.");
      return FLOW_FUNC_FAILED;
    }
    auto input_tensor1 = input1->GetTensor();
    auto input_data_type1 = input_tensor1->GetDataType();
    auto input_shape1 = input_tensor1->GetShape();
    if (input_shape1.size() != 1) {
      FLOW_FUNC_LOG_ERROR("Input shape is expected as {1}.");
      return FLOW_FUNC_FAILED;
    }
    if (input_data_type1 != TensorDataType::DT_INT32) {
      FLOW_FUNC_LOG_ERROR("Input data type is expected as int32.");
      return FLOW_FUNC_FAILED;
    }
    const uint32_t element_num = 3;
    auto output_msg = run_context->AllocTensorMsg({element_num}, TensorDataType::DT_INT32);
    if (output_msg == nullptr) {
      FLOW_FUNC_LOG_ERROR("Fail to alloc tensor msg.");
      return FLOW_FUNC_FAILED;
    }
    auto output_tensor = output_msg->GetTensor();

    auto dataSize1 = output_tensor->GetDataSize();
    if (dataSize1 != element_num * sizeof(int32_t)) {
      FLOW_FUNC_LOG_ERROR("Alloc tensor data size in invalid.");
      return FLOW_FUNC_FAILED;
    }
    auto *output_data = static_cast<int32_t *>(output_tensor->GetData());
    for (int32_t i = 0; i < element_num; ++i) {
      output_data[i] = i + 1;
      // element value : 1,2,3
    }

    int32_t input_data1 = *(static_cast<int32_t *>(input_tensor1->GetData()));
    if (input_data1 == 0) {
      FLOW_FUNC_LOG_INFO("invoke Proc0");
      return SetOutputByIndex(run_context, {0, 1}, output_msg);
    } else {
      FLOW_FUNC_LOG_INFO("invoke Proc1");
      return SetOutputByIndex(run_context, {2, 3}, output_msg);
    }

    return FLOW_FUNC_SUCCESS;
  }

 private:
  int SetOutputByIndex(const std::shared_ptr<MetaRunContext> &run_context, const std::vector<uint32_t> &ids,
                       std::shared_ptr<FlowMsg> output_msg) const {
    for (uint32_t id : ids) {
      const auto ret = run_context->SetOutput(id, output_msg);
      if (ret != FLOW_FUNC_SUCCESS) {
        FLOW_FUNC_LOG_ERROR("Fail to set output.");
        return FLOW_FUNC_FAILED;
      }
    }
    return FLOW_FUNC_SUCCESS;
  }
};

FLOW_FUNC_REGISTRAR(AddFlowFunc).RegProcFunc("control_func", &AddFlowFunc::Proc);
}  // namespace FlowFunc