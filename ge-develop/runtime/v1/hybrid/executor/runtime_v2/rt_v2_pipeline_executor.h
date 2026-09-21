/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_HYBRID_EXECUTOR_RT_V2_PIPELINE_EXECUTOR_H_
#define GE_HYBRID_EXECUTOR_RT_V2_PIPELINE_EXECUTOR_H_

#include "rt_v2_executor_interface.h"
#include "rt_v2_stage_state.h"

namespace gert {
class RtV2PipelineExecutor : public RtV2ExecutorInterface {
 public:
  static std::unique_ptr<RtV2PipelineExecutor> Create(const ge::GeRootModelPtr &model, RtSession *session = nullptr);
  static std::unique_ptr<RtV2PipelineExecutor> Create(const ge::GeRootModelPtr &model,
                                                      const ge::DevResourceAllocator &allocator,
                                                      RtSession *session = nullptr);

  ge::Status Load(const ModelExecuteArg &arg, const ModelLoadArg &load_arg) override;
  ge::Status Execute(const ModelExecuteArg &arg, Tensor **inputs, size_t input_num, Tensor **outputs,
                     size_t output_num, const RunConfig &config) override;
  ge::Status Unload() override;

  const ModelIoDesc *GetAllInputsDesc(size_t &input_num) const override {
    input_num = inputs_desc_.size();
    return inputs_desc_.data();
  }
  const ModelIoDesc *GetAllOutputsDesc(size_t &output_num) const override {
    output_num = outputs_desc_.size();
    return outputs_desc_.data();
  }

  void AddSubscriber(SubscriberExtendInfo extend_info) const override {
    for (const auto &executor : stage_executors_) {
      executor->AddSubscriber(extend_info);
    }
  }

 private:
  explicit RtV2PipelineExecutor() = default;
  ge::Status Initialize();
  std::vector<ModelIoDesc> inputs_desc_;
  std::vector<ModelIoDesc> outputs_desc_;
  std::vector<std::unique_ptr<StageState>> stage_executors_;
  std::vector<std::shared_ptr<StageNotification>> notifications_;
};
}  // namespace gert

#endif  // GE_HYBRID_EXECUTOR_RT_V2_PIPELINE_EXECUTOR_H_
