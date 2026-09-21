/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flow_func/meta_multi_func.h"
#include "flow_func/flow_func_log.h"
#include "flow_func/mbuf_flow_msg_queue.h"
#include "flow_func/mbuf_flow_msg.h"
#include <mutex>
#include "common/inner_error_codes.h"
namespace FlowFunc {
class MultiFlowFuncStub : public MetaMultiFunc {
 public:
  MultiFlowFuncStub() = default;
  ~MultiFlowFuncStub() override = default;

  int32_t Init(const std::shared_ptr<MetaParams> &params) override {
    auto get_ret = params->GetAttr("out_type", out_data_type_);
    if (get_ret != FLOW_FUNC_SUCCESS) {
      FLOW_FUNC_RUN_LOG_ERROR("GetAttr dType not exist.");
      return get_ret;
    }
    if (out_data_type_ != TensorDataType::DT_UINT32) {
      return -1;
    }
    AscendString attr_val;
    (void)params->GetAttr("need_re_init_attr", attr_val);
    need_re_init_ = (attr_val == "true");
    if (need_re_init_ && init_times_ == 0) {
      ++init_times_;
      return FLOW_FUNC_ERR_INIT_AGAIN;
    }
    return 0;
  }
  // a + b
  void Add(uint32_t *src1, uint32_t *src2, uint32_t *dst, size_t count) {
    for (size_t i = 0; i < count; ++i) {
      dst[i] = src1[i] + src2[i];
      FLOW_FUNC_LOG_DEBUG("input1 = %u, input2 = %u, sum = %u", src1[i], src2[i], dst[i]);
    }
  }
  // 2a + b
  void Add2(uint32_t *src1, uint32_t *src2, uint32_t *dst, size_t count) {
    for (size_t i = 0; i < count; ++i) {
      dst[i] = 2 * src1[i] + src2[i];
      FLOW_FUNC_LOG_DEBUG("input1 = %u, input2 = %u, sum = %u", src1[i], src2[i], dst[i]);
    }
  }

  int32_t Proc1(const std::shared_ptr<MetaRunContext> &run_context,
                const std::vector<std::shared_ptr<FlowMsg>> &input_flow_msgs) {
    int32_t exp_code = 0;
    uint64_t context_id = 0;
    if (run_context->GetException(exp_code, context_id)) {
      FLOW_FUNC_LOG_ERROR("Get exception exp_code[%d] context_id[%lu].", exp_code, context_id);
      return FLOW_FUNC_FAILED;
    }
    if (input_flow_msgs.size() != 2) {
      FLOW_FUNC_LOG_ERROR("add must have 2 input\n");
      return -1;
    }
    auto input_tensor1 = input_flow_msgs[0]->GetTensor();
    auto input_tensor2 = input_flow_msgs[1]->GetTensor();

    auto &input_shape1 = input_tensor1->GetShape();
    auto input_data1 = input_tensor1->GetData();
    auto input_data2 = input_tensor2->GetData();
    auto data_size1 = input_tensor1->GetDataSize();
    if (((*static_cast<uint32_t *>(input_data1)) == 1000) && ((*static_cast<uint32_t *>(input_data2)) == 1000)) {
      FLOW_FUNC_RUN_LOG_INFO("Start raise exception");
      run_context->RaiseException(-100, 100);
      return FLOW_FUNC_SUCCESS;
    }
    // multi proc share init para from Init
    auto output = run_context->AllocTensorMsgWithAlign(input_shape1, out_data_type_, 256);
    auto output_tensor = output->GetTensor();
    auto output_data = output_tensor->GetData();
    Add(static_cast<uint32_t *>(input_data1), static_cast<uint32_t *>(input_data2),
        static_cast<uint32_t *>(output_data), data_size1 / sizeof(uint32_t));
    run_context->SetOutput(0, output);
    std::unique_lock<std::mutex> lock(count_mutex_);
    set_output_count_++;
    FLOW_FUNC_LOG_INFO("Proc1 enter, total proc count is %u", set_output_count_);
    return FLOW_FUNC_SUCCESS;
  }

  int32_t Proc2(const std::shared_ptr<MetaRunContext> &run_context,
                const std::vector<std::shared_ptr<FlowMsg>> &input_flow_msgs) {
    if (input_flow_msgs.size() != 2) {
      FLOW_FUNC_LOG_ERROR("add must have 2 input\n");
      return -1;
    }
    auto input_tensor1 = input_flow_msgs[0]->GetTensor();
    auto input_tensor2 = input_flow_msgs[1]->GetTensor();

    auto &input_shape1 = input_tensor1->GetShape();
    auto input_data1 = input_tensor1->GetData();
    auto input_data2 = input_tensor2->GetData();
    auto data_size1 = input_tensor1->GetDataSize();
    auto output = run_context->AllocTensorMsg(input_shape1, out_data_type_);
    auto output_tensor = output->GetTensor();
    auto output_data = output_tensor->GetData();
    Add2(static_cast<uint32_t *>(input_data1), static_cast<uint32_t *>(input_data2),
         static_cast<uint32_t *>(output_data), data_size1 / sizeof(uint32_t));
    run_context->SetOutput(1, output);
    std::unique_lock<std::mutex> lock(count_mutex_);
    set_output_count_++;
    FLOW_FUNC_LOG_INFO("Proc2 enter, total proc count is %u\n", set_output_count_);
    return FLOW_FUNC_SUCCESS;
  }

 private:
  TensorDataType out_data_type_;
  std::mutex count_mutex_;
  uint32_t set_output_count_;
  bool need_re_init_ = false;
  uint32_t init_times_ = 0;
};

class MultiFlowFuncWithOneProcStub : public MetaMultiFunc {
 public:
  MultiFlowFuncWithOneProcStub() = default;
  ~MultiFlowFuncWithOneProcStub() override = default;

  int32_t Init(const std::shared_ptr<MetaParams> &params) override {
    auto get_ret = params->GetAttr("out_type", out_data_type_);
    if (get_ret != FLOW_FUNC_SUCCESS) {
      FLOW_FUNC_RUN_LOG_ERROR("GetAttr dType not exist.");
      return get_ret;
    }
    if (out_data_type_ != TensorDataType::DT_UINT32) {
      return -1;
    }
    (void)params->GetAttr("need_re_init_attr", need_re_init_);
    if (need_re_init_ && init_times_ == 0) {
      ++init_times_;
      return FLOW_FUNC_ERR_INIT_AGAIN;
    }
    return 0;
  }

  void Add(uint32_t *src1, uint32_t *src2, uint32_t *dst, size_t count) {
    for (size_t i = 0; i < count; ++i) {
      dst[i] = src1[i] + src2[i];
    }
  }

  int32_t SingleProc(const std::shared_ptr<MetaRunContext> &run_context,
                     const std::vector<std::shared_ptr<FlowMsg>> &input_flow_msgs) {
    if (input_flow_msgs.size() != 2) {
      FLOW_FUNC_LOG_ERROR("add must have 2 input!!!");
      return -1;
    }
    auto input_tensor1 = input_flow_msgs[0]->GetTensor();
    auto input_tensor2 = input_flow_msgs[1]->GetTensor();

    auto &input_shape1 = input_tensor1->GetShape();
    auto input_data1 = input_tensor1->GetData();
    auto input_data2 = input_tensor2->GetData();
    auto data_size1 = input_tensor1->GetDataSize();
    auto output = run_context->AllocTensorMsg(input_shape1, out_data_type_);
    auto output_tensor = output->GetTensor();
    auto output_data = output_tensor->GetData();
    Add(static_cast<uint32_t *>(input_data1), static_cast<uint32_t *>(input_data2),
        static_cast<uint32_t *>(output_data), data_size1 / sizeof(uint32_t));
    run_context->SetOutput(0, output);
    return FLOW_FUNC_SUCCESS;
  }

 private:
  TensorDataType out_data_type_;
  bool need_re_init_ = false;
  uint32_t init_times_ = 0;
};

class MultiFlowFuncWithStreamInputStub : public MetaMultiFunc {
 public:
  MultiFlowFuncWithStreamInputStub() = default;
  ~MultiFlowFuncWithStreamInputStub() override = default;

  int32_t Init(const std::shared_ptr<MetaParams> &params) override {
    auto get_ret = params->GetAttr("out_type", out_data_type_);
    if (get_ret != FLOW_FUNC_SUCCESS) {
      FLOW_FUNC_RUN_LOG_ERROR("GetAttr dType not exist.");
      return get_ret;
    }
    if (out_data_type_ != TensorDataType::DT_UINT32) {
      return -1;
    }
    (void)params->GetAttr("need_re_init_attr", need_re_init_);
    if (need_re_init_ && init_times_ == 0) {
      ++init_times_;
      return FLOW_FUNC_ERR_INIT_AGAIN;
    }
    return 0;
  }

  void Add(uint32_t *src1, uint32_t *src2, uint32_t *dst, size_t count) {
    for (size_t i = 0; i < count; ++i) {
      dst[i] = src1[i] + src2[i];
    }
  }

  int32_t StreamInputProc(const std::shared_ptr<MetaRunContext> &run_context,
                          const std::vector<std::shared_ptr<FlowMsgQueue>> &input_msg_queues) {
    if (input_msg_queues.size() != 2) {
      FLOW_FUNC_LOG_ERROR("add must have 2 input\n");
      return -1;
    }
    std::shared_ptr<FlowMsg> input_flow_msg1;
    std::shared_ptr<FlowMsg> input_flow_msg2;
    auto ret1 = input_msg_queues[0]->Dequeue(input_flow_msg1, -1);
    auto ret2 = input_msg_queues[1]->Dequeue(input_flow_msg2, -1);
    if ((ret1 != FLOW_FUNC_SUCCESS) || (ret2 != FLOW_FUNC_SUCCESS)) {
      FLOW_FUNC_LOG_ERROR("dequeue faild, ret1=%d, ret2=%d", ret1, ret2);
      return FLOW_FUNC_FAILED;
    }

    auto input_tensor1 = std::dynamic_pointer_cast<MbufFlowMsg>(input_flow_msg1)->GetTensor();
    auto input_tensor2 = std::dynamic_pointer_cast<MbufFlowMsg>(input_flow_msg2)->GetTensor();

    auto &input_shape1 = input_tensor1->GetShape();
    auto input_data1 = input_tensor1->GetData();
    auto input_data2 = input_tensor2->GetData();
    auto data_size1 = input_tensor1->GetDataSize();
    auto output = run_context->AllocTensorMsg(input_shape1, out_data_type_);
    auto output_tensor = output->GetTensor();
    auto output_data = output_tensor->GetData();
    Add(static_cast<uint32_t *>(input_data1), static_cast<uint32_t *>(input_data2),
        static_cast<uint32_t *>(output_data), data_size1 / sizeof(uint32_t));
    run_context->SetOutput(0, output);
    return FLOW_FUNC_SUCCESS;
  }

 private:
  TensorDataType out_data_type_;
  bool need_re_init_ = false;
  uint32_t init_times_ = 0;
};

class MultiFlowFuncWithCallNnStub : public MetaMultiFunc {
 public:
  MultiFlowFuncWithCallNnStub() = default;
  ~MultiFlowFuncWithCallNnStub() override = default;
  int32_t CallNnProc1(const std::shared_ptr<MetaRunContext> &run_context,
                      const std::vector<std::shared_ptr<FlowMsg>> &input_tensors) {
    if (input_tensors.size() != 2) {
      FLOW_FUNC_LOG_ERROR("input num size=%zu must be 2.", input_tensors.size());
      return FLOW_FUNC_ERR_PARAM_INVALID;
    }
    auto tensor = input_tensors[0]->GetTensor();
    if (tensor == nullptr) {
      FLOW_FUNC_LOG_ERROR("input[0]'s tensor is null.");
      return FLOW_FUNC_ERR_PARAM_INVALID;
    }
    int32_t ret = FLOW_FUNC_SUCCESS;
    std::vector<std::shared_ptr<FlowMsg>> output_msgs;
    if (tensor->GetDataType() == FlowFunc::TensorDataType::DT_FLOAT) {
      ret = run_context->RunFlowModel("float_model_key", input_tensors, output_msgs, 1000);
    } else {
      ret = run_context->RunFlowModel("other_model_key", input_tensors, output_msgs, 1000);
    }

    if (ret != FLOW_FUNC_SUCCESS) {
      FLOW_FUNC_LOG_ERROR("run flow model failed, ret=%d.", ret);
      return ret;
    }

    for (size_t i = 0; i < output_msgs.size(); ++i) {
      ret = run_context->SetOutput(i, output_msgs[i]);
      if (ret != FLOW_FUNC_SUCCESS) {
        FLOW_FUNC_LOG_ERROR("set output failed, output idx=%zu.", i);
        return ret;
      }
      FLOW_FUNC_LOG_INFO("MultiFlowFuncWithCallNnStub:CallNnProc1 SetOutput[%zu] success.", i);
    }
    return FLOW_FUNC_SUCCESS;
  }

  int32_t CallNnProc2(const std::shared_ptr<MetaRunContext> &run_context,
                      const std::vector<std::shared_ptr<FlowMsg>> &input_tensors) {
    if (input_tensors.size() != 2) {
      FLOW_FUNC_LOG_ERROR("input num size=%zu must be 2.", input_tensors.size());
      return FLOW_FUNC_ERR_PARAM_INVALID;
    }
    auto tensor = input_tensors[0]->GetTensor();
    if (tensor == nullptr) {
      FLOW_FUNC_LOG_ERROR("input[0]'s tensor is null.");
      return FLOW_FUNC_ERR_PARAM_INVALID;
    }
    int32_t ret = FLOW_FUNC_SUCCESS;
    std::vector<std::shared_ptr<FlowMsg>> output_msgs;
    if (tensor->GetDataType() == FlowFunc::TensorDataType::DT_FLOAT) {
      ret = run_context->RunFlowModel("float_model_key", input_tensors, output_msgs, 1000);
    } else {
      ret = run_context->RunFlowModel("other_model_key", input_tensors, output_msgs, 1000);
    }

    if (ret != FLOW_FUNC_SUCCESS) {
      FLOW_FUNC_LOG_ERROR("run flow model failed, ret=%d.", ret);
      // must return error, test case depend this.
      return ret;
    }

    for (size_t i = 0; i < output_msgs.size(); ++i) {
      ret = run_context->SetOutput(i, output_msgs[i]);
      if (ret != FLOW_FUNC_SUCCESS) {
        FLOW_FUNC_LOG_ERROR("set output failed, output idx=%zu.", i);
        return ret;
      }
      FLOW_FUNC_LOG_INFO("MultiFlowFuncWithCallNnStub:CallNnProc2 SetOutput[%zu] success.", i);
    }
    return FLOW_FUNC_SUCCESS;
  }
};

class MultiFlowFuncWithRawDataStub : public MetaMultiFunc {
 public:
  MultiFlowFuncWithRawDataStub() = default;
  ~MultiFlowFuncWithRawDataStub() override = default;

  int32_t RawDataProc(const std::shared_ptr<MetaRunContext> &run_context,
                      const std::vector<std::shared_ptr<FlowMsg>> &input_flow_msgs) {
    if (input_flow_msgs.size() != 2) {
      FLOW_FUNC_LOG_ERROR("add must have 2 input!!!");
      return -1;
    }
    auto input_tensor0 = input_flow_msgs[0]->GetTensor();
    auto input_tensor1 = input_flow_msgs[1]->GetTensor();

    auto out0 = FlowBufferFactory::AllocTensor(input_tensor0->GetShape(), input_tensor0->GetDataType());
    auto out_data0 = reinterpret_cast<float *>(out0->GetData());
    auto input_data0 = reinterpret_cast<float *>(input_tensor0->GetData());
    for (int64_t i = 0; i < input_tensor0->GetElementCnt(); ++i) {
      out_data0[i] = input_data0[i];
    }
    auto out_msg0 = run_context->ToFlowMsg(out0);
    run_context->SetOutput(0, out_msg0);

    auto input_size1 = input_tensor1->GetDataSize();
    auto out_msg1 = run_context->AllocRawDataMsg(input_size1);
    void *data_ptr1 = nullptr;
    uint64_t data_size1 = 0UL;
    out_msg1->GetRawData(data_ptr1, data_size1);
    for (uint64_t i = 0; i < input_size1; ++i) {
      reinterpret_cast<uint8_t *>(data_ptr1)[i] = reinterpret_cast<uint8_t *>(input_tensor1->GetData())[i];
    }
    run_context->SetOutput(1, out_msg1);
    return FLOW_FUNC_SUCCESS;
  }
};

FLOW_FUNC_REGISTRAR(MultiFlowFuncStub)
    .RegProcFunc("Proc1", &MultiFlowFuncStub::Proc1)
    .RegProcFunc("Proc2", &MultiFlowFuncStub::Proc2);

FLOW_FUNC_REGISTRAR(MultiFlowFuncWithOneProcStub).RegProcFunc("SingleProc", &MultiFlowFuncWithOneProcStub::SingleProc);
FLOW_FUNC_REGISTRAR(MultiFlowFuncWithStreamInputStub)
    .RegProcFunc("StreamInputProc", &MultiFlowFuncWithStreamInputStub::StreamInputProc);
FLOW_FUNC_REGISTRAR(MultiFlowFuncWithCallNnStub)
    .RegProcFunc("CallNnProc1", &MultiFlowFuncWithCallNnStub::CallNnProc1)
    .RegProcFunc("CallNnProc2", &MultiFlowFuncWithCallNnStub::CallNnProc2);
FLOW_FUNC_REGISTRAR(MultiFlowFuncWithRawDataStub)
    .RegProcFunc("RawDataProc", &MultiFlowFuncWithRawDataStub::RawDataProc);
}  // namespace FlowFunc