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

#include "macro_utils/dt_public_scope.h"
#include "common/plugin/tbe_plugin_manager.h"
#include "macro_utils/dt_public_unscope.h"

namespace ge {
class UtestTBEPluginManager: public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestTBEPluginManager, CheckFindParserSo) {
  string path = "";
  vector<string> file_list = {};
  string caffe_parser_path = "";
  EXPECT_NO_THROW(TBEPluginManager::Instance().FindParserUsedSo(path, file_list, caffe_parser_path));
  path = "/lib64";
  EXPECT_NO_THROW(TBEPluginManager::Instance().FindParserUsedSo(path, file_list, caffe_parser_path));
}

TEST_F(UtestTBEPluginManager, ProcessSoFullName) {
  string path = "";
  vector<string> file_list = {};
  string full_name = "";
  string caffe_parser_path = "";
  string caffe_parser_so_suff = "";
  EXPECT_NO_THROW(TBEPluginManager::Instance().ProcessSoFullName(file_list, caffe_parser_path, full_name, caffe_parser_so_suff));
  path = "/lib64";
  EXPECT_NO_THROW(TBEPluginManager::Instance().ProcessSoFullName(file_list, caffe_parser_path, full_name, caffe_parser_so_suff));
}
}  // namespace ge
