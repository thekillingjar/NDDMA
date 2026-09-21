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
#include <vector>

#include "macro_utils/dt_public_scope.h"
#include "graph/opsproto_manager.h"
#include "macro_utils/dt_public_unscope.h"

using namespace ge;
using namespace testing;
using namespace std;

class UtestOpsprotoManager : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}

 public:
};

TEST_F(UtestOpsprotoManager, initialize_failure) {
  OpsProtoManager *manager = OpsProtoManager::Instance();
  std::map<std::string, std::string> options;
  options["a"] = "a";
  bool ret = manager->Initialize(options);
  EXPECT_EQ(ret, false);

  options["ge.opsProtoLibPath"] = "";
  ret = manager->Initialize(options);
  EXPECT_EQ(ret, true);

  options["ge.opsProtoLibPath"] = "path1:path2";
  ret = manager->Initialize(options);
  EXPECT_EQ(ret, true);

  options["ge.opsProtoLibPath"] = "/usr/local/HiAI/path1.so:$ASCEND_HOME/path2";
  EXPECT_EQ(ret, true);

  mkdir("test_ops_proto_manager", S_IRUSR);

  options["ge.opsProtoLibPath"] = "test_ops_proto_manager";
  ret = manager->Initialize(options);
  EXPECT_EQ(ret, true);
  rmdir("test_proto_manager");

  manager->Finalize();
}
