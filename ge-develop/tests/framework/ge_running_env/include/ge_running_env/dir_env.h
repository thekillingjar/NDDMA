/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_TESTS_ST_STUBS_UTILS_DIR_ENV_H_
#define AIR_CXX_TESTS_ST_STUBS_UTILS_DIR_ENV_H_
#include "ge_running_env/fake_ns.h"
#include <string>

FAKE_NS_BEGIN
class DirEnv {
 public:
  void InitDir();
  static DirEnv &GetInstance();

 private:
  DirEnv();
  void InitEngineConfJson();
  void InitEngineSo();
  void InitOpsKernelInfoStore();

 private:
  std::string run_root_path_;
  std::string engine_path_;
  std::string ops_kernel_info_store_path_;
  std::string engine_config_path_;
};
FAKE_NS_END
#endif //AIR_CXX_TESTS_ST_STUBS_UTILS_DIR_ENV_H_
