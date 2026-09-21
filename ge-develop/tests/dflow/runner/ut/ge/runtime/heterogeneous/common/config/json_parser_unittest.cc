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
#include <string>

#include "ge/ge_error_codes.h"
#include "framework/common/debug/ge_log.h"
#include "common/env_path.h"

#include "macro_utils/dt_public_scope.h"
#include "common/config/json_parser.h"
#include "macro_utils/dt_public_unscope.h"

using namespace std;
namespace ge {
class UtJsonParser : public testing::Test {
 public:
  UtJsonParser() {}
 protected:
  void SetUp() override {}
  void TearDown() override {
  }
  const std::string data_path = PathUtils::Join({EnvPath().GetAirBasePath(), "tests/dflow/runner/ut/ge/runtime/data"});
};

TEST_F(UtJsonParser, read_config_file_invalid) {
  ge::JsonParser jsonParser;
  nlohmann::json js;
  auto ret = jsonParser.ReadConfigFile("", js);
  ASSERT_EQ(ret, ACL_ERROR_GE_PARAM_INVALID);
}
}

