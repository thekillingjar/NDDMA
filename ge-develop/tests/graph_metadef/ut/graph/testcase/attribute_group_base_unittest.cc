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
#include "ge_ir.pb.h"

namespace ge {
namespace {
class AttrGroupsBaseTest : public AttrGroupsBase {
public:
    virtual std::unique_ptr<AttrGroupsBase> Clone() {return nullptr;};
};

class AttributeGroupUt : public testing::Test {};

TEST_F(AttributeGroupUt, TypeID) {
  EXPECT_EQ(GetTypeId<AttrGroupsBase>(), (void*)10);
  EXPECT_EQ(GetTypeId<AscTensorAttr>(), (void*)11);
  EXPECT_EQ(GetTypeId<AscNodeAttr>(), (void*)12);
  EXPECT_EQ(GetTypeId<AscGraphAttr>(), (void*)13);
  EXPECT_EQ(GetTypeId<SymbolicDescAttr>(), (void*)14);
  EXPECT_EQ(GetTypeId<ShapeEnvAttr>(), (void*)15);
  EXPECT_EQ(GetTypeId<AutoFuseAttrs>(), (void*)16);
  EXPECT_EQ(GetTypeId<AutoFuseGraphAttrs>(), (void*)17);
}

TEST_F(AttributeGroupUt, Clone) {
  EXPECT_NO_THROW(
    proto::AttrGroupDef attr_group_def;
    auto base = AttrGroupsBaseTest();
    EXPECT_EQ(GRAPH_SUCCESS, base.Serialize(attr_group_def));
    EXPECT_EQ(GRAPH_SUCCESS, base.Deserialize(attr_group_def, nullptr));
  );
}
}
}  // namespace ge
