/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "engine/node_converter_utils.h"

#include <gtest/gtest.h>

#include "ge_graph_dsl/graph_dsl.h"
#include "engine/gelocal/inputs_converter.h"
#include "graph/utils/graph_utils.h"
#include "common/share_graph.h"
#include "faker/global_data_faker.h"
#include "common/bg_test.h"
#include "common/topo_checker.h"
#include "register/op_impl_registry.h"

using namespace ge;
using namespace gert::bg;

namespace gert {
class NodeConverterUtilsUT : public bg::BgTest {};

TEST_F(NodeConverterUtilsUT, CreateOutputShapes) {
  bg::ValueHolder::PushGraphFrame();
  auto op_desc = std::shared_ptr<ge::OpDesc>(new OpDesc("Fake", "Fake"));
  (void)op_desc->AddOutputDesc("fake_output", ge::GeTensorDesc());
  auto output_shape_holders = NodeConverterUtils::CreateOutputShapes(op_desc);
  ASSERT_EQ(output_shape_holders.size(), 1);
  ASSERT_EQ(output_shape_holders[0]->GetType(), ValueHolder::ValueHolderType::kConst);
  bg::ValueHolderPtr tensor_value_holder = NodeConverterUtils::CreateOutputShape(nullptr);
  ASSERT_EQ(tensor_value_holder->GetType(), ValueHolder::ValueHolderType::kConst);
}

TEST_F(NodeConverterUtilsUT, GetShapeFromGeShape) {
  ge::GeShape ge_shape({1,2,3});
  const auto shape = NodeConverterUtils::GetShapeFromGeShape(ge_shape);
  ASSERT_EQ(shape.GetDimNum(), ge_shape.GetDimNum());
  for (size_t i = 0U; i < shape.GetDimNum(); ++i) {
    EXPECT_EQ(shape.GetDim(i), ge_shape.GetDim(i));
  }
}

TEST_F(NodeConverterUtilsUT, GetShapeFromGeShapeForUnknownDimNum) {
  ge::GeShape ge_shape({-2});
  const auto shape = NodeConverterUtils::GetShapeFromGeShape(ge_shape);
  ASSERT_EQ(shape.GetDimNum(), 1);
  ASSERT_EQ(ge_shape.GetDimNum(), 0);
  for (size_t i = 0U; i < shape.GetDimNum(); ++i) {
    EXPECT_EQ(shape.GetDim(i), ge_shape.GetDim(i));
  }
}

} // namespace gert