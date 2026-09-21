/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_INC_FRAMEWORK_RUNTIME_EXECUTOR_GLOBAL_TRACER_H_
#define AIR_CXX_INC_FRAMEWORK_RUNTIME_EXECUTOR_GLOBAL_TRACER_H_
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_visibility.h"

namespace gert {
class VISIBILITY_EXPORT GlobalTracer {
 public:
  static GlobalTracer *GetInstance() {
    static GlobalTracer global_tracer;
    return &global_tracer;
  }
  uint64_t GetEnableFlags() const {
    return static_cast<uint64_t>(IsLogEnable(GE_MODULE_NAME, DLOG_INFO));
  };
 private:
   GlobalTracer() {};
   ~GlobalTracer() {};
   GlobalTracer(const GlobalTracer &) = delete;
   GlobalTracer(GlobalTracer &&) = delete;
   GlobalTracer &operator=(const GlobalTracer &) = delete;
   GlobalTracer &operator=(GlobalTracer &&) = delete;
};
}
#endif // AIR_CXX_INC_FRAMEWORK_RUNTIME_EXECUTOR_GLOBAL_TRACER_H_
