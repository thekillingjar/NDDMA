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
#include "base/registry/opp_package_utils.h"

namespace gert {
class UtestOppPackageUtils : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UtestOppPackageUtils, OppSoDescPathsTestSuccess) {
  std::vector<ge::AscendString> opp_path_vector = {"/path/opp.so"};
  OppSoDesc so_desc(opp_path_vector, "pkg_name");
  EXPECT_EQ(so_desc.GetSoPaths()[0], "/path/opp.so");
  EXPECT_EQ(so_desc.GetPackageName(), "pkg_name");

  auto so_desc2 = so_desc;
  EXPECT_EQ(so_desc2.GetSoPaths()[0], "/path/opp.so");
  EXPECT_EQ(so_desc2.GetPackageName(), "pkg_name");

  OppSoDesc so_desc3(std::move(so_desc2));
  EXPECT_EQ(so_desc3.GetSoPaths()[0], "/path/opp.so");
  EXPECT_EQ(so_desc3.GetPackageName(), "pkg_name");

  so_desc3 = std::move(so_desc);
  EXPECT_EQ(so_desc3.GetSoPaths()[0], "/path/opp.so");
  EXPECT_EQ(so_desc3.GetPackageName(), "pkg_name");

  // 代码覆盖
  so_desc = so_desc;

  OppSoDesc so_desc4(so_desc3);
  EXPECT_EQ(so_desc4.GetSoPaths()[0], "/path/opp.so");
  EXPECT_EQ(so_desc4.GetPackageName(), "pkg_name");

  so_desc = so_desc4;
  EXPECT_EQ(so_desc.GetSoPaths()[0], "/path/opp.so");
  EXPECT_EQ(so_desc.GetPackageName(), "pkg_name");
}

}  // namespace gert
