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
#include "graph/build/memory/var_mem_assign_util.h"
#include "graph/compute_graph.h"
#include "graph/debug/ge_attr_define.h"
#include "common/share_graph.h"
#include "base/graph/manager/graph_var_manager.h"
#include "graph/utils/tensor_utils.h"

using namespace ge;
class UtestVarMemAssignUtil : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestVarMemAssignUtil, GetNameForVarManager_Failed_WithNullptr) {
  ASSERT_EQ(ge::VarMemAssignUtil::GetNameForVarManager(nullptr), "");
}

TEST_F(UtestVarMemAssignUtil, GetNameForVarManager_Success_ReutrnNodeName) {
  auto op_desc = std::make_shared<ge::OpDesc>("name", "Constant");
  ASSERT_NE(op_desc, nullptr);
  ASSERT_EQ(ge::VarMemAssignUtil::GetNameForVarManager(op_desc), "name");
}

TEST_F(UtestVarMemAssignUtil, GetNameForVarManager_Success_ReutrnSrcConstName) {
  auto op_desc = std::make_shared<ge::OpDesc>("name", "Constant");
  ASSERT_NE(op_desc, nullptr);
  ge::AttrUtils::SetStr(op_desc, ge::ATTR_NAME_SRC_CONST_NAME, "real_name");
  ASSERT_EQ(ge::VarMemAssignUtil::GetNameForVarManager(op_desc), "real_name");
}

TEST_F(UtestVarMemAssignUtil, AssignVarMemory_ForFileConstant_DT_STRING_SUCCESS) {
  const int64_t file_constant_size = 100;
  auto graph = gert::ShareGraph::SimpleFileConstantGraph();
  graph->SetSessionID(202311132057);
  auto file_constant = graph->FindFirstNodeMatchType("FileConstant");
  file_constant->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_STRING);
  ge::AttrUtils::SetDataType(file_constant->GetOpDesc(), "dtype", ge::DT_STRING);
  ge::AttrUtils::SetInt(file_constant->GetOpDesc(), "length", file_constant_size);

  VarManager::Instance(graph->GetSessionID())->Init(0, graph->GetSessionID(), 0, 0);
  ASSERT_EQ(ge::VarMemAssignUtil::AssignVarMemory(graph), ge::SUCCESS);

  // 校验给file_constant分的内存长度是按照length属性中的长度
  int64_t size_temp;
  TensorUtils::GetSize(*file_constant->GetOpDesc()->MutableOutputDesc(0), size_temp);
  EXPECT_EQ(size_temp, file_constant_size);
}

TEST_F(UtestVarMemAssignUtil, AssignVarMemory_ForConstant_DT_STRING_SUCCESS) {
  auto graph = gert::ShareGraph::SimpleVariableGraph();
  graph->SetSessionID(202311132057);
  auto constant = graph->FindFirstNodeMatchType("Constant");
  constant->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_STRING);

  VarManager::Instance(graph->GetSessionID())->Init(0, graph->GetSessionID(), 0, 0);
  ASSERT_EQ(ge::VarMemAssignUtil::AssignVarMemory(graph), ge::SUCCESS);

  ConstGeTensorPtr value = std::make_shared<const GeTensor>();
  ASSERT_NE(value, nullptr);
  ASSERT_TRUE(AttrUtils::GetTensor(constant->GetOpDesc(), ATTR_NAME_WEIGHTS, value));
  const auto expect_mem_size = static_cast<int64_t>(value->GetData().size());

  // 校验给constant分的内存长度是按照属性value中的长度
  int64_t size_temp;
  TensorUtils::GetSize(*constant->GetOpDesc()->MutableOutputDesc(0), size_temp);
  EXPECT_EQ(size_temp, expect_mem_size);
}
