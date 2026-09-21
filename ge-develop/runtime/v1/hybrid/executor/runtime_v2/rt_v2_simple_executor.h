/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_HYBRID_EXECUTOR_RT_V2_SIMPLE_EXECUTOR_H_
#define GE_HYBRID_EXECUTOR_RT_V2_SIMPLE_EXECUTOR_H_

#include "rt_v2_executor_interface.h"
#include "hybrid/executor/cann_tracing_profiler.h"
#include "common/model/external_allocator_manager.h"

namespace gert {
class RtV2SimpleExecutor : public RtV2ExecutorInterface {
 public:
  static std::unique_ptr<RtV2SimpleExecutor> Create(const ge::GeRootModelPtr &model, RtSession *session);
  static std::unique_ptr<RtV2SimpleExecutor> Create(const ge::GeRootModelPtr &model,
                                                    ge::DevResourceAllocator &allocator,
                                                    RtSession *session);

  ge::Status Load(const ModelExecuteArg &arg, const ModelLoadArg &load_arg) override {
    if (!loaded_.exchange(true)) {
      GELOGI("Load simple executor for model %s", name_.c_str());
      return executor_->Load(arg, load_arg);
    }
    GELOGI("Executor for model %s has already loaded", name_.c_str());
    return ge::SUCCESS;
  }

  ge::Status Execute(const ModelExecuteArg &arg, Tensor **inputs, size_t input_num, Tensor **outputs,
                     size_t output_num, const RunConfig &config) override {
    GE_ASSERT(loaded_, "Executor for model %s has not been load", name_.c_str());
    for (size_t i = 0U; i < config.iterations_per_loop; ++i) {
      if (config.profiler_collector != nullptr) {
        GE_ASSERT_SUCCESS(config.profiler_collector->RecordStart(arg.stream));
      }
      const auto ret = executor_->Execute(arg, inputs, input_num, outputs, output_num);
      if (ret != ge::GRAPH_SUCCESS) {
        return ret;
      }
      if (config.profiler_collector != nullptr) {
        GE_ASSERT_SUCCESS(config.profiler_collector->RecordEnd(arg.stream));
      }
    }
    return ge::SUCCESS;
  }

  ge::Status Unload() override {
    if (loaded_.exchange(false)) {
      GELOGI("Unload simple executor for model %s", name_.c_str());
      return executor_->UnLoad();
    }
    GELOGI("Executor for model %s has already unloaded", name_.c_str());
    return ge::SUCCESS;
  }

  const ModelIoDesc *GetAllInputsDesc(size_t &input_num) const override {
    return executor_->GetModelDesc().GetAllInputsDesc(input_num);
  }

  const ModelIoDesc *GetAllOutputsDesc(size_t &output_num) const override {
    return executor_->GetModelDesc().GetAllOutputsDesc(output_num);
  }

  void AddSubscriber(SubscriberExtendInfo extend_info) const override {
    extend_info.executor = executor_.get();
    (void)executor_->GetSubscribers().AddSubscriber<CannTracingProfiler>(kMainExeGraph, extend_info);
  }

 private:
  explicit RtV2SimpleExecutor(const std::string &name, std::unique_ptr<ModelV2Executor> &&executor) : name_(name) {
    executor_ = std::move(executor);
  }
  static std::unique_ptr<RtV2SimpleExecutor> Create(const ge::GeRootModelPtr &model,
                                                    StreamAllocator *const stream_allocator,
                                                    EventAllocator *const event_allocator,
                                                    NotifyAllocator *const notify_allocator,
                                                    RtSession *session);
  std::string name_;
  std::atomic_bool loaded_{false};
  std::unique_ptr<ModelV2Executor> executor_;
};
}  // namespace gert

#endif  // GE_HYBRID_EXECUTOR_RT_V2_SIMPLE_EXECUTOR_H_
