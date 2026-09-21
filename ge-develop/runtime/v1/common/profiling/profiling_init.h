/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_COMMON_PROFILING_PROFILING_INIT_H_
#define GE_COMMON_PROFILING_PROFILING_INIT_H_

#include <nlohmann/json.hpp>
#include <string>
#include <map>

#include "common/profiling/profiling_properties.h"
#include "framework/common/ge_inner_error_codes.h"
#include "aprof_pub.h"

struct MsprofOptions {
  char jobId[2048];
  char options[2048];
};

namespace ge {
class ProfilingInit {
 public:
  static ProfilingInit &Instance();
  Status Init(const std::map<std::string, std::string> &options);
  Status ProfRegisterCtrlCallback();
  void ShutDownProfiling();
  Status SetDeviceIdByModelId(uint32_t model_id, uint32_t device_id);
  Status UnsetDeviceIdByModelId(uint32_t model_id, uint32_t device_id);

 private:
  ProfilingInit() = default;
  ~ProfilingInit() = default;
  Status InitProfOptions(const std::map<std::string, std::string> &options,
                         MsprofOptions &prof_conf,
                         bool &is_execute_profiling);
  Status ParseOptions(const std::string &options);
};
}  // namespace ge

#endif  // GE_COMMON_PROFILING_PROFILING_INIT_H_
