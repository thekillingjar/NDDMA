/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exe_graph/lowering/generate_exe_graph.h"
#include <gtest/gtest.h>
#include "common/bg_test.h"
#include "faker/node_faker.h"
namespace gert {
class PublicBgRegisterUT : public bg::BgTestAutoCreateFrame {};
TEST_F(PublicBgRegisterUT, SelfRegisterOk) {
  auto shapes = bg::ValueHolder::CreateDataOutput("Shapes", {}, 3);
  auto node = ComputeNodeFaker().IoNum(1, 3).Build();
  auto sizes = bg::GenerateExeGraph::CalcTensorSize(node, shapes);
  ASSERT_EQ(sizes.size(), 3);
}
}  // namespace gert