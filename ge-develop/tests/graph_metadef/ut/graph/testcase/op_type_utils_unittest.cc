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
#include "graph/utils/op_type_utils.h"
#include "graph/debug/ge_op_types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph/compute_graph.h"

namespace ge {
class UtestOpTypeUtils : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UtestOpTypeUtils, TestDataNodeType) {
  std::string test_node_type = "Data";
  EXPECT_TRUE(OpTypeUtils::IsDataNode(test_node_type));
  EXPECT_FALSE(OpTypeUtils::IsVariableNode(test_node_type));
  EXPECT_FALSE(OpTypeUtils::IsVarLikeNode(test_node_type));

  test_node_type = "AnnData";
  EXPECT_TRUE(OpTypeUtils::IsDataNode(test_node_type));

  test_node_type = "AippData";
  EXPECT_TRUE(OpTypeUtils::IsDataNode(test_node_type));

  test_node_type = "RefData";
  EXPECT_TRUE(OpTypeUtils::IsDataNode(test_node_type));
}

TEST_F(UtestOpTypeUtils, TestVariableNodeType) {
  std::string test_node_type = "Variable";
  EXPECT_TRUE(OpTypeUtils::IsVariableNode(test_node_type));
  EXPECT_TRUE(OpTypeUtils::IsVarLikeNode(test_node_type));

  test_node_type = "VariableV2";
  EXPECT_TRUE(OpTypeUtils::IsVariableNode(test_node_type));
  EXPECT_TRUE(OpTypeUtils::IsVarLikeNode(test_node_type));
}

TEST_F(UtestOpTypeUtils, TestVariableLikeNodeType) {
  std::string test_node_type = "RefData";
  EXPECT_FALSE(OpTypeUtils::IsVariableNode(test_node_type));
  EXPECT_TRUE(OpTypeUtils::IsVarLikeNode(test_node_type));
}

TEST_F(UtestOpTypeUtils, TestGetOriginalTypeFailed) {
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("A", FRAMEWORKOP);
  std::shared_ptr<ge::ComputeGraph> graph = std::make_shared<ge::ComputeGraph>("test1");
  ge::NodePtr node = graph->AddNode(op_desc);

  std::string original_type;
  EXPECT_EQ(OpTypeUtils::GetOriginalType(node->GetOpDesc(), original_type), INTERNAL_ERROR);
}

TEST_F(UtestOpTypeUtils, TestGetOriginalTypeSuccess) {
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("A", FRAMEWORKOP);
  std::shared_ptr<ge::ComputeGraph> graph = std::make_shared<ge::ComputeGraph>("test1");
  ge::NodePtr node = graph->AddNode(op_desc);
  std::string type = "GetNext";
  node->GetOpDesc()->SetType(FRAMEWORKOP);
  ge::AttrUtils::SetStr(node->GetOpDesc(), ge::ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, type);
  std::string original_type;
  EXPECT_EQ(OpTypeUtils::GetOriginalType(node->GetOpDesc(), original_type), GRAPH_SUCCESS);
  EXPECT_EQ(original_type, type);
}

TEST_F(UtestOpTypeUtils, TestIsInputRefData) {
  ge::OpDescPtr ref_data_op_desc = std::make_shared<ge::OpDesc>("RefData", REFDATA);
  ge::OpDescPtr data_op_desc = std::make_shared<ge::OpDesc>("Data", DATA);
  EXPECT_EQ(OpTypeUtils::IsInputRefData(ref_data_op_desc), true);
  (void) AttrUtils::SetStr(ref_data_op_desc, REF_VAR_SRC_VAR_NAME, "1");
  EXPECT_EQ(OpTypeUtils::IsInputRefData(ref_data_op_desc), false);
  EXPECT_EQ(OpTypeUtils::IsInputRefData(data_op_desc), false);
}

TEST_F(UtestOpTypeUtils, TestIsAutofuseNode) {
  ge::OpDescPtr asc_bc = std::make_shared<ge::OpDesc>("asc_bc", ASC_BC);
  ge::OpDescPtr fuse_asc_bc = std::make_shared<ge::OpDesc>("fuse_asc_bc", FUSE_ASC_BC);
  ge::OpDescPtr empty_asc_bc = std::make_shared<ge::OpDesc>("empty_asc_bc", EMPTY_ASC_BC);
  ge::OpDescPtr unknown = std::make_shared<ge::OpDesc>("unknown", "UNKNOWN");
  EXPECT_EQ(OpTypeUtils::IsAutofuseNode(asc_bc), true);
  EXPECT_EQ(OpTypeUtils::IsAutofuseNode(fuse_asc_bc), true);
  EXPECT_EQ(OpTypeUtils::IsAutofuseNode(empty_asc_bc), true);
  EXPECT_EQ(OpTypeUtils::IsEmptyAutofuseNode(empty_asc_bc->GetType()), true);
  EXPECT_EQ(OpTypeUtils::IsAutofuseNode(unknown), false);
}
TEST_F(UtestOpTypeUtils, TestIsAutofuseNodeWithType) {
  EXPECT_EQ(OpTypeUtils::IsAutofuseNode(ASC_BC), true);
  EXPECT_EQ(OpTypeUtils::IsAutofuseNode(FUSE_ASC_BC), true);
  EXPECT_EQ(OpTypeUtils::IsAutofuseNode(EMPTY_ASC_BC), true);
  EXPECT_EQ(OpTypeUtils::IsAutofuseNode("UNKNOWN"), false);
}

TEST_F(UtestOpTypeUtils, TestIsConstNode) {
  ge::OpDescPtr constant = std::make_shared<ge::OpDesc>("constant", CONSTANT);
  ge::OpDescPtr constant_op = std::make_shared<ge::OpDesc>("constant_op", CONSTANTOP);
  ge::OpDescPtr unknown = std::make_shared<ge::OpDesc>("unknown", "UNKNOWN");
  EXPECT_EQ(OpTypeUtils::IsConstNode(constant->GetType()), true);
  EXPECT_EQ(OpTypeUtils::IsConstNode(constant_op->GetType()), true);
  EXPECT_EQ(OpTypeUtils::IsConstNode(unknown->GetType()), false);
}

TEST_F(UtestOpTypeUtils, TestIsGraphInput) {
  ge::OpDescPtr data = std::make_shared<ge::OpDesc>("data", DATA);
  ge::OpDescPtr variable = std::make_shared<ge::OpDesc>("variable", VARIABLE);
  ge::OpDescPtr variable_v2 = std::make_shared<ge::OpDesc>("variable", VARIABLEV2);
  ge::OpDescPtr ref_data = std::make_shared<ge::OpDesc>("ref_data", REFDATA);
  ge::OpDescPtr constant = std::make_shared<ge::OpDesc>("constant", CONSTANT);
  ge::OpDescPtr constant_op = std::make_shared<ge::OpDesc>("constant_op", CONSTANTOP);
  ge::OpDescPtr unknown = std::make_shared<ge::OpDesc>("unknown", "UNKNOWN");
  EXPECT_EQ(OpTypeUtils::IsGraphInputNode(data->GetType()), true);
  EXPECT_EQ(OpTypeUtils::IsGraphInputNode(variable->GetType()), true);
  EXPECT_EQ(OpTypeUtils::IsGraphInputNode(variable_v2->GetType()), true);
  EXPECT_EQ(OpTypeUtils::IsGraphInputNode(ref_data->GetType()), true);
  EXPECT_EQ(OpTypeUtils::IsGraphInputNode(constant->GetType()), true);
  EXPECT_EQ(OpTypeUtils::IsGraphInputNode(constant_op->GetType()), true);
  EXPECT_EQ(OpTypeUtils::IsGraphInputNode(unknown->GetType()), false);
}

TEST_F(UtestOpTypeUtils, TestIsGraphOutput) {
  ge::OpDescPtr data = std::make_shared<ge::OpDesc>("data", DATA);
  ge::OpDescPtr net_output = std::make_shared<ge::OpDesc>("net_output", NETOUTPUT);
  ge::OpDescPtr unknown = std::make_shared<ge::OpDesc>("unknown", "UNKNOWN");
  EXPECT_EQ(OpTypeUtils::IsGraphOutputNode(data->GetType()), false);
  EXPECT_EQ(OpTypeUtils::IsGraphOutputNode(net_output->GetType()), true);
  EXPECT_EQ(OpTypeUtils::IsGraphOutputNode(unknown->GetType()), false);
}

}  // namespace ge
