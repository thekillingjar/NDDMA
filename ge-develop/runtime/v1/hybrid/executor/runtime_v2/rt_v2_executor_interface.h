/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_HYBRID_EXECUTOR_RT_V2_INTERFACE_H_
#define GE_HYBRID_EXECUTOR_RT_V2_INTERFACE_H_

#include "framework/runtime/model_v2_executor.h"
#include "hybrid/model/hybrid_model.h"
#include "common/profiling/profiling_manager.h"

namespace gert {
class RtV2ExecutorInterface {
 public:
  struct RunConfig {
    explicit RunConfig(size_t iterations_per_loop_tmp, ge::ProfilerCollector *profiler_collector_tmp = nullptr)
        : iterations_per_loop(iterations_per_loop_tmp),
          profiler_collector(profiler_collector_tmp) {}
    size_t iterations_per_loop;
    ge::ProfilerCollector *profiler_collector;
  };
  virtual ge::Status Load(const ModelExecuteArg &arg, const ModelLoadArg &load_arg) = 0;
  virtual ge::Status Execute(const ModelExecuteArg &arg, Tensor **inputs, size_t input_num, Tensor **outputs,
                             size_t output_num, const RunConfig &config = RunConfig(1U, nullptr)) = 0;
  virtual ge::Status Unload() = 0;

  virtual const ModelIoDesc *GetAllInputsDesc(size_t &input_num) const = 0;
  virtual const ModelIoDesc *GetAllOutputsDesc(size_t &output_num) const = 0;
  virtual void AddSubscriber(SubscriberExtendInfo extend_info) const = 0;

  virtual ~RtV2ExecutorInterface() = default;
 protected:
  RtV2ExecutorInterface() = default;
};
}  // namespace gert

#endif  // GE_HYBRID_EXECUTOR_RT_V2_INTERFACE_H_
