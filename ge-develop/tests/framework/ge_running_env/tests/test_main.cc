/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include "common/debug/log.h"
#include "ge/ge_api.h"
#include "ge_running_env/ge_running_env_faker.h"

using namespace std;
using namespace ge;

int main(int argc, char **argv) {
  map<AscendString, AscendString> options;
  options.insert({AscendString("ge.exec.opWaitTimeout"), AscendString("11")});
  options.insert({AscendString("ge.exec.opExecuteTimeout"), AscendString("11")});
  ge::GEInitialize(options);
  GeRunningEnvFaker::BackupEnv();
  testing::InitGoogleTest(&argc, argv);
  int ret = RUN_ALL_TESTS();

  return ret;
}
