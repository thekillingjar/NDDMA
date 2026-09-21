/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hostcpu_kernel_info.h"

#include <mutex>
#include "config/ops_json_file.h"
#include "error_code/error_code.h"
#include "util/constant.h"
#include "util/log.h"
#include "util/util.h"

using namespace std;
namespace aicpu {
KernelInfoPtr HostCpuKernelInfo::instance_ = nullptr;

inline KernelInfoPtr HostCpuKernelInfo::Instance() {
  static once_flag flag;
  call_once(flag, []() { instance_.reset(new (nothrow) HostCpuKernelInfo); });
  return instance_;
}

bool HostCpuKernelInfo::ReadOpInfoFromJsonFile() {
  std::string real_hostcpu_ops_file_path;
  std::string path;
  const char* path_env = nullptr;
  MM_SYS_GET_ENV(MM_ENV_ASCEND_OPP_PATH, path_env);
  std::string env_path;
  if (path_env != nullptr) {
    env_path = std::string(path_env);
    path = env_path + kHostCpuOpsFileBasedOnEnvPath;
    // 如果存在新的算子包路径，则不再兼容读取老路径。
    if (IsPathExist(path)) {
      real_hostcpu_ops_file_path = path;
    } else {
      real_hostcpu_ops_file_path = env_path + kHostCpuOpsFileBasedOnEnvPathOld;
    }
  } else {
    std::string file_path = GetOpsPath(reinterpret_cast<void*>(&HostCpuKernelInfo::Instance));
    path = file_path + kHostCpuOpsFileRelativePath;
    if (IsPathExist(path)) {
      real_hostcpu_ops_file_path = path;
    } else {
      real_hostcpu_ops_file_path = file_path + kHostCpuOpsFileRelativePathOld;
    }
  }

  AICPUE_LOGI("HostCpuKernelInfo real_hostcpu_ops_file_path is %s.", real_hostcpu_ops_file_path.c_str());

  return OpsJsonFile::Instance()
      .ParseUnderPath(real_hostcpu_ops_file_path, op_info_json_file_)
      .state == ge::SUCCESS;
}

FACTORY_KERNELINFO_CLASS_KEY(HostCpuKernelInfo, kHostCpuKernelInfoChoice)
}  // namespace aicpu
