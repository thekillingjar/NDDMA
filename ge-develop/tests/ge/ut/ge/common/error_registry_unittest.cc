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
#include "base/err_msg.h"

namespace {
std::string g_msg = R"(
{
  "error_info_list": [
    {
      "errClass": "GE Errors",
      "errTitle": "Invalid_Argument",
      "ErrCode": "E17777",
      "ErrMessage": "Value [%s] for parameter [%s] is invalid. Reason: %s",
      "Arglist": "value,parameter,reason",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Try again with a valid argument."
      }
    }
  ]
}
)";
REG_FORMAT_ERROR_MSG(g_msg.c_str(), g_msg.size());
}

namespace ge {
class UtestErrorRegistry : public testing::Test {
protected:
  void SetUp() {
  }

  void TearDown() {
  }
};

TEST_F(UtestErrorRegistry, ReportInnerErrMsg) {
  EXPECT_EQ(error_message::ReportPredefinedErrMsg("E10001", {"value", "parameter", "reason"}, {"1", "2", "le 0"}), 0);
  EXPECT_EQ(error_message::ReportPredefinedErrMsg("E16666", {"value", "parameter", "reason"}, {"1", "2", "le 0"}), -1);
  EXPECT_EQ(error_message::ReportPredefinedErrMsg("E17777", {"value", "parameter", "reason"}, {"1", "2", "le 0"}), 0);
}
}