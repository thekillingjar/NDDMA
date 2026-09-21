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
#include "graph/compute_graph.h"
#include "graph_builder/storage_format.h"

namespace gert {
namespace bg {
class StorageFormatUT : public testing::Test {};

TEST_F(StorageFormatUT, TestStorageFormat01) {
  auto op_desc = std::make_shared<ge::OpDesc>("_tmp_op", "TMP");
  auto tmp_graph = std::make_shared<ge::ComputeGraph>("tmp-graph");
  auto node = tmp_graph->AddNode(op_desc);
  EXPECT_EQ(AnyDiffStorageFormat(node), false);

  ge::GeTensorDesc desc;
  desc.SetFormat(ge::Format(1));
  desc.SetOriginFormat(ge::Format(2));
  op_desc->AddInputDesc("input_1", desc);
  EXPECT_EQ(AnyDiffStorageFormat(node), true);
  op_desc->AddOutputDesc("output_1", desc);
  EXPECT_EQ(AnyDiffStorageFormat(node), true);
}

}  // namespace bg
}  // namespace gert