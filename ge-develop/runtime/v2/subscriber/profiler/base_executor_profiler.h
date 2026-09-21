/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_SUBSCRIBER_PROFILER_BASE_EXECUTOR_PROFILER_H_
#define AIR_CXX_RUNTIME_V2_SUBSCRIBER_PROFILER_BASE_EXECUTOR_PROFILER_H_
#include <kernel/kernel_log.h>
#include "runtime/subscriber/built_in_subscriber_definitions.h"
#include "runtime/subscriber/global_profiler.h"
#include "exe_graph/runtime/continuous_vector.h"
#include "exe_graph/runtime/context_extend.h"
#include "common/profiling_definitions.h"
#include "common/model/ge_root_model.h"
#include "exe_graph/lowering/graph_frame.h"
#include "core/execution_data.h"
#include "subscriber/subscriber_utils.h"

namespace gert {
struct ProfExtendInfo {
  uint64_t node_name_idx;
  uint64_t node_type_idx;
  uint64_t kernel_type_idx;
  uint32_t engine_type;
  uint32_t block_dim_input_idx = UINT32_MAX;
  uint64_t origin_name_hash_for_hash = UINT64_MAX;
  uint32_t kernel_prof_type = kInvalidProfKernelType;
};

class BaseExecutorProfiler {
 public:
  BaseExecutorProfiler() {
    global_prof_wrapper_ = GlobalProfilingWrapper::GetInstance();
  }
  virtual ~BaseExecutorProfiler() = default;
  bool IsEnabled(ProfilingType profiling_type) const {
    return global_prof_wrapper_->IsEnabled(profiling_type);
  }

  GlobalProfilingWrapper *GetGlobalProf() const {
    return global_prof_wrapper_;
  }

  void InitNameAndTypeWithHash(const ExecutionData &execution_data);
  ge::graphStatus InitNameAndTypeWithHash(const ExecutionData &execution_data,
                                          std::vector<ProfExtendInfo> &prof_extend_infos) const;
 protected:
  std::vector<ProfExtendInfo> prof_extend_infos_{};

 private:
  GlobalProfilingWrapper *global_prof_wrapper_{nullptr};
  bool is_str_to_hash_inited_{false};
};
}  // namespace gert
#endif  // AIR_CXX_RUNTIME_V2_SUBSCRIBER_PROFILER_BASE_EXECUTOR_PROFILER_H_
