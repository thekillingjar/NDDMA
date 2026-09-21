/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <gtest/gtest.h>
#include "graph/op_desc.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"

namespace ge {
class UtestGeAttrDefine : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestGeAttrDefine, GetAttachedAttrDefine) {
  OpDescPtr op_desc = std::make_shared<OpDesc>();
  NamedAttrs attr;
  (void) ge::AttrUtils::SetStr(attr, ATTR_NAME_ATTACHED_RESOURCE_NAME, "tiling");
  (void) ge::AttrUtils::SetStr(attr, ATTR_NAME_ATTACHED_RESOURCE_REUSE_KEY, "tiling_key");
  (void) ge::AttrUtils::SetListInt(attr, ATTR_NAME_ATTACHED_RESOURCE_DEPEND_VALUE_LIST_INT, {0, 1, 2});
  (void) ge::AttrUtils::SetBool(attr, ATTR_NAME_ATTACHED_RESOURCE_REQUIRED_FLAG, true);
  (void) ge::AttrUtils::SetInt(attr, ATTR_NAME_ATTACHED_RESOURCE_ID, 1);
  (void) ge::AttrUtils::SetBool(attr, ATTR_NAME_ATTACHED_RESOURCE_IS_VALID, false);
  std::vector<NamedAttrs> list_name_attr_set;
  list_name_attr_set.emplace_back(attr);
  ge::AttrUtils::SetListNamedAttrs(op_desc, ATTR_NAME_ATTACHED_STREAM_INFO_LIST, list_name_attr_set);

  std::vector<NamedAttrs> list_name_attr_get;
  ge::AttrUtils::GetListNamedAttrs(op_desc, ATTR_NAME_ATTACHED_STREAM_INFO_LIST, list_name_attr_get);
  ASSERT_EQ(list_name_attr_get.size(), 1U);

  std::string ret_str;
  EXPECT_EQ(ge::AttrUtils::GetStr(list_name_attr_get[0], ATTR_NAME_ATTACHED_RESOURCE_NAME, ret_str), true);
  EXPECT_EQ(ret_str, "tiling");

  EXPECT_EQ(ge::AttrUtils::GetStr(list_name_attr_get[0], ATTR_NAME_ATTACHED_RESOURCE_REUSE_KEY, ret_str), true);
  EXPECT_EQ(ret_str, "tiling_key");

  std::vector<int64_t> ret_list;
  EXPECT_EQ(
      ge::AttrUtils::GetListInt(list_name_attr_get[0], ATTR_NAME_ATTACHED_RESOURCE_DEPEND_VALUE_LIST_INT, ret_list),
      true);
  EXPECT_EQ(ret_list.size(), 3);
  EXPECT_EQ(ret_list[2], 2);

  bool ret_bool;
  EXPECT_EQ(ge::AttrUtils::GetBool(list_name_attr_get[0], ATTR_NAME_ATTACHED_RESOURCE_REQUIRED_FLAG, ret_bool), true);
  EXPECT_EQ(ret_bool, true);

  int64_t ret_int;
  EXPECT_EQ(ge::AttrUtils::GetInt(list_name_attr_get[0], ATTR_NAME_ATTACHED_RESOURCE_ID, ret_int), true);
  EXPECT_EQ(ret_int, 1);

  EXPECT_EQ(ge::AttrUtils::GetBool(list_name_attr_get[0], ATTR_NAME_ATTACHED_RESOURCE_IS_VALID, ret_bool), true);
  EXPECT_EQ(ret_bool, false);

  EXPECT_EQ(ge::AttrUtils::HasAttr(list_name_attr_get[0], ATTR_NAME_ATTACHED_RESOURCE_TYPE), false);
}
}
