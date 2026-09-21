/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_kernel_info.h"

#include <mutex>
#include "config/ops_json_file.h"
#include "error_code/error_code.h"
#include "util/constant.h"
#include "util/log.h"
#include "util/util.h"

using namespace std;
namespace aicpu {
KernelInfoPtr AicpuKernelInfo::instance_ = nullptr;

inline KernelInfoPtr AicpuKernelInfo::Instance() {
  static once_flag flag;
  call_once(flag, [&]() { instance_.reset(new (nothrow) AicpuKernelInfo); });
  return instance_;
}

bool AicpuKernelInfo::ReadOpInfoFromJsonFile() {
  std::string real_aicpu_ops_file_path;
  std::string path;
  const char* path_env = nullptr;
  MM_SYS_GET_ENV(MM_ENV_ASCEND_OPP_PATH, path_env);
  std::string env_path;
  if (path_env != nullptr) {
    env_path = std::string(path_env);
    path = env_path + kAicpuOpsFileBasedOnEnvPath;
    // 如果存在新的算子包路径，则不再兼容读取老路径。
    if (IsPathExist(path)) {
      real_aicpu_ops_file_path = path;
    } else {
      real_aicpu_ops_file_path = env_path + kAicpuOpsFileBasedOnEnvPathOld;
    }
  } else {
    std::string file_path = GetOpsPath(reinterpret_cast<void*>(&AicpuKernelInfo::Instance));
    path = file_path + kAicpuOpsFileRelativePathOld;
    if (IsPathExist(path)) {
      real_aicpu_ops_file_path = path;
    } else {
      real_aicpu_ops_file_path = file_path + kAicpuOpsFileRelativePathOld;
    }
  }
  SetJsonPath(real_aicpu_ops_file_path);
  AICPUE_LOGI("AicpuKernelInfo real_aicpu_ops_file_path is %s.", real_aicpu_ops_file_path.c_str());
  return OpsJsonFile::Instance().ParseUnderPath(real_aicpu_ops_file_path,
      op_info_json_file_).state == ge::SUCCESS;
}

FACTORY_KERNELINFO_CLASS_KEY(AicpuKernelInfo, kAicpuKernelInfoChoice)
}  // namespace aicpu
