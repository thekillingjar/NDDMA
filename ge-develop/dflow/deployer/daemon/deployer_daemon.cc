/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "framework/common/debug/log.h"
#include "common/config/configurations.h"
#include "daemon/model_deployer_daemon.h"

int main(int argc, char_t *argv[]) {
  if (argv[0] == nullptr) {
    return 1;
  }
  std::string process_name(argv[0]);
  bool is_sub_deployer = (process_name.find("sub_deployer") != std::string::npos);
  if (ge::Configurations::GetInstance().InitInformation() != ge::SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Parse configuration failed.");
    return 1;
  }

  ge::ModelDeployerDaemon daemon(is_sub_deployer);
  auto ret = daemon.Start(argc, argv);
  if (ret != ge::SUCCESS) {
    return 1;  // failure
  }

  return 0;
}
