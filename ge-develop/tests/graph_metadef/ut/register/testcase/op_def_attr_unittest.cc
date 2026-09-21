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
#include "register/op_def_registry.h"

namespace ops {

namespace {

class OpAttrDefUT : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(OpAttrDefUT, AttrTest) {
  OpDef opDef("Test");
  OpAttrDef attr("Test");
  OpAttrDef attr2("Test");
  OpAttrDef attr3("Test1");
  EXPECT_EQ(attr == attr2, true);
  EXPECT_EQ(attr == attr3, false);
  attr.Bool();
  EXPECT_EQ(attr.GetCfgDataType(), "bool");
  EXPECT_EQ(attr.GetProtoDataType(), "Bool");
  opDef.Attr("Test");
  EXPECT_EQ(opDef.GetAttrs().size(), 1);
  opDef.Attr("Test");
  EXPECT_EQ(opDef.GetAttrs().size(), 1);
  opDef.Attr("Test1");
  EXPECT_EQ(opDef.GetAttrs().size(), 2);
  attr.AttrType(Option::OPTIONAL).Bool(true);
  EXPECT_EQ(attr.GetAttrDefaultVal("[]"), "true");
  attr.AttrType(Option::OPTIONAL).Int(10);
  EXPECT_EQ(attr.GetAttrDefaultVal("[]"), "10");
  attr.AttrType(Option::OPTIONAL).String("test");
  EXPECT_EQ(attr.GetAttrDefaultVal("[]"), "test");
  attr.AttrType(Option::OPTIONAL).Float(0.1);
  EXPECT_EQ(attr.GetAttrDefaultVal("[]"), "0.1");
  attr.AttrType(Option::OPTIONAL).ListBool({true, false});
  EXPECT_EQ(attr.GetAttrDefaultVal("[]"), "[true,false]");
  attr.AttrType(Option::OPTIONAL).ListFloat({0.1, 0.1});
  EXPECT_EQ(attr.GetAttrDefaultVal("[]"), "[0.1,0.1]");
  attr.AttrType(Option::OPTIONAL).ListInt({1, 2});
  EXPECT_EQ(attr.GetAttrDefaultVal("[]"), "[1,2]");
  attr.AttrType(Option::OPTIONAL).ListListInt({{1, 2}, {3, 4}});
  EXPECT_EQ(attr.GetAttrDefaultVal("[]"), "[[1,2],[3,4]]");
  attr.Version(1);
  EXPECT_EQ(attr.GetVersion(), 1);
}
TEST_F(OpAttrDefUT, CommentSingleTest) {
  OpAttrDef attr("Test");
  attr.Comment("")
      .Comment("comment of Attr Test");
  EXPECT_EQ(attr.GetComment(), "comment of Attr Test");
}
TEST_F(OpAttrDefUT, CommentCombineTest) {
  OpDef opDef("Test");
  opDef.Attr("Test")
      .Comment("")
      .Comment("comment of Attr Test");
  EXPECT_EQ(opDef.GetAttrs().size(), 1);
  EXPECT_EQ(opDef.GetAttrs().at(0).GetComment(), "comment of Attr Test");
}
}  // namespace
}  // namespace ops
