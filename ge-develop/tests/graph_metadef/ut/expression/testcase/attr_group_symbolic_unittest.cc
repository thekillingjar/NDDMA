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

#include "attribute_group/attr_group_base.h"
#include "attribute_group/attr_group_shape_env.h"
#include "attribute_group/attr_group_symbolic_desc.h"
#include "proto/ge_ir.pb.h"

namespace ge {
namespace {
class AttrGroupsBaseTest : public AttrGroupsBase {
public:
    virtual std::unique_ptr<AttrGroupsBase> Clone() {return nullptr;};
};

class AttributeGroupUt : public testing::Test {};

TEST_F(AttributeGroupUt, Clone) {
  EXPECT_NO_THROW(
    auto symbolicDesc = SymbolicDescAttr().Clone();
    auto shapeEnv = ShapeEnvAttr().Clone();
    proto::AttrGroupDef attr_group_def;
    auto base = AttrGroupsBaseTest();
    EXPECT_EQ(GRAPH_SUCCESS, base.Serialize(attr_group_def));
    EXPECT_EQ(GRAPH_SUCCESS, base.Deserialize(attr_group_def, nullptr));
  );
}

TEST_F(AttributeGroupUt, SymbolicDescAttrSerialize) {
  auto symbolicDesc = SymbolicDescAttr();
  symbolicDesc.symbolic_tensor.MutableOriginSymbolShape().AppendDim(ge::Symbol("s0"));
  symbolicDesc.symbolic_tensor.MutableSymbolicValue()->emplace_back(ge::Symbol("s2"));
  proto::AttrGroupDef attr_group_def;
  auto ret = symbolicDesc.Serialize(attr_group_def);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(AttributeGroupUt, SymbolicDescAttrDeserialize) {
  auto symbolicDesc = SymbolicDescAttr();
  symbolicDesc.symbolic_tensor.MutableOriginSymbolShape().AppendDim(ge::Symbol("s0"));
  symbolicDesc.symbolic_tensor.MutableSymbolicValue()->emplace_back(ge::Symbol("s2"));
  proto::AttrGroupDef attr_group_def;
  EXPECT_EQ(GRAPH_SUCCESS, symbolicDesc.Serialize(attr_group_def));

  auto symbolicDesc1 = SymbolicDescAttr();
  auto ret = symbolicDesc1.Deserialize(attr_group_def, nullptr);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(1, symbolicDesc1.symbolic_tensor.origin_symbol_shape_.GetDimNum());
  EXPECT_EQ(ge::Symbol("s0"), symbolicDesc1.symbolic_tensor.origin_symbol_shape_.GetDim(0));
  EXPECT_EQ(1, symbolicDesc1.symbolic_tensor.GetSymbolicValue()->size());
  EXPECT_EQ(ge::Symbol("s2"), *symbolicDesc1.symbolic_tensor.GetSymbolicValue()->begin());
}
}
}  // namespace ge