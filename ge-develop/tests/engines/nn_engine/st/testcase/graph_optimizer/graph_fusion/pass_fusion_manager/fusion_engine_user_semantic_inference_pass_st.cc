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
#include <nlohmann/json.hpp>

#define protected public
#define private public
#include "graph_optimizer/graph_fusion/fusion_pass_manager/builtin_pass/user_semantic_inference.h"
#include "graph_optimizer/fe_graph_optimizer.h"
#include "common/fe_utils.h"
#include "pass_manager.h"
#include "common/util/op_info_util.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "ops_store/ops_kernel_manager.h"

#undef protected
#undef private
 
using namespace std;
using namespace ge;
using namespace fe;
 
class STEST_UserSemanticInferencePass: public testing::Test
{
public:
  FEOpsKernelInfoStorePtr ops_kernel_info_store_ptr;
protected:
  void SetUp()
  {
    cout << "STEST_UserSemanticInferencePass setup" << endl;
  }
 
  void TearDown()
  {
    cout << "STEST_UserSemanticInferencePass TearDown" << endl;
  }

  ge::ComputeGraphPtr CreateGraphForUserSemanticInferUt() {
    std::vector<int64_t> dim_nd = {8, 1, 4096};
    std::vector<int64_t> dim_nz = {8, 32, 448, 16, 16};
    std::vector<int64_t> dim_nz_ed = {8, 64, 448, 16, 64};
    std::vector<int64_t> dim_nd_out = {384, 4096};
    ge::GeShape shape_nd(dim_nd);
    ge::GeShape shape_nz(dim_nz);
    ge::GeShape shape_nz_ed(dim_nz_ed);
    ge::GeShape shape_nd_out(dim_nd_out);
    
    ge::GeTensorDesc tensor_desc_nd(shape_nd, ge::FORMAT_ND, ge::DT_INT32);
    tensor_desc_nd.SetOriginShape(shape_nd);
    tensor_desc_nd.SetOriginDataType(ge::DT_INT32);
    tensor_desc_nd.SetOriginFormat(ge::FORMAT_ND);
    
    ge::GeTensorDesc tensor_desc_nz(shape_nz, ge::FORMAT_FRACTAL_NZ, ge::DT_INT32);
    tensor_desc_nd.SetOriginShape(shape_nz);
    tensor_desc_nd.SetOriginDataType(ge::DT_INT32);
    tensor_desc_nd.SetOriginFormat(ge::FORMAT_FRACTAL_NZ);
    
    ge::GeTensorDesc tensor_desc_nz_ed(shape_nz_ed, ge::FORMAT_FRACTAL_NZ, ge::DT_INT4);
    tensor_desc_nz_ed.SetOriginShape(shape_nz_ed);
    tensor_desc_nz_ed.SetOriginDataType(ge::DT_INT4);
    tensor_desc_nz_ed.SetOriginFormat(ge::FORMAT_FRACTAL_NZ);
    
    ge::GeTensorDesc tensor_desc_nd_out(shape_nd_out, ge::FORMAT_ND, ge::DT_BF16);
    tensor_desc_nd_out.SetOriginShape(shape_nd_out);
    tensor_desc_nd_out.SetOriginDataType(ge::DT_BF16);
    tensor_desc_nd_out.SetOriginFormat(ge::FORMAT_ND);
    
    OpDescPtr data_op = std::make_shared<OpDesc>("data", "Data");
    OpDescPtr bitcast_op = std::make_shared<OpDesc>("bitcast", "Bitcast");
    OpDescPtr GMM_op = std::make_shared<OpDesc>("groupedmatmul", "GroupedMatmul");
    OpDescPtr net_output_op = std::make_shared<OpDesc>("net_output", "NetOutput");
    
    data_op->AddOutputDesc(tensor_desc_nz);
    bitcast_op->AddInputDesc(tensor_desc_nz);
    bitcast_op->AddOutputDesc(tensor_desc_nz_ed);
    GMM_op->AddInputDesc("weight", tensor_desc_nd);
    GMM_op->AddOutputDesc(tensor_desc_nd_out);
    net_output_op->AddInputDesc(tensor_desc_nd_out);

    AttrUtils::SetBool(data_op, "_enable_storage_format_spread", true);
    
    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    NodePtr data_node = graph->AddNode(data_op);
    NodePtr bitcast_node = graph->AddNode(bitcast_op);
    NodePtr GMM_node = graph->AddNode(GMM_op);
    NodePtr net_output_node = graph->AddNode(net_output_op);
    
    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), bitcast_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(bitcast_node->GetOutDataAnchor(0), GMM_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(GMM_node->GetOutDataAnchor(0), net_output_node->GetInDataAnchor(0));
    return graph;
  }

  ge::ComputeGraphPtr CreatePyPTOGraphForUserSemanticInferUt() {
    std::vector<int64_t> dim_nd = {8, 1, 4096};
    std::vector<int64_t> dim_nz = {8, 32, 448, 16, 16};
    std::vector<int64_t> dim_nz_ed = {8, 64, 448, 16, 64};
    std::vector<int64_t> dim_nd_out = {384, 4096};
    ge::GeShape shape_nd(dim_nd);
    ge::GeShape shape_nz(dim_nz);
    
    ge::GeTensorDesc tensor_desc_nd(shape_nd, ge::FORMAT_ND, ge::DT_INT32);
    tensor_desc_nd.SetOriginShape(shape_nd);
    tensor_desc_nd.SetOriginDataType(ge::DT_INT32);
    tensor_desc_nd.SetOriginFormat(ge::FORMAT_ND);
    
    ge::GeTensorDesc tensor_desc_nz(shape_nz, ge::FORMAT_FRACTAL_NZ, ge::DT_INT32);
    tensor_desc_nz.SetOriginShape(shape_nz);
    tensor_desc_nz.SetOriginDataType(ge::DT_INT32);
    tensor_desc_nz.SetOriginFormat(ge::FORMAT_FRACTAL_NZ);
    tensor_desc_nz.SetFormat(ge::FORMAT_FRACTAL_NZ);
    
    OpDescPtr data_op = std::make_shared<OpDesc>("data", "Data");
    OpDescPtr pypto_op = std::make_shared<OpDesc>("pypto", "LightningIndexerPrologPto");
    data_op->AddOutputDesc(tensor_desc_nz);
    pypto_op->AddInputDesc(tensor_desc_nd);

    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    NodePtr data_node = graph->AddNode(data_op);
    NodePtr pypto_node = graph->AddNode(pypto_op);
    
    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), pypto_node->GetInDataAnchor(0));
    return graph;
  }

  ge::ComputeGraphPtr CreatePyPTOGraph2ForUserSemanticInferUt() {
    std::vector<int64_t> dim_nd = {8, 1, 4096};
    ge::GeShape shape_nd(dim_nd);
    
    ge::GeTensorDesc tensor_desc_nd(shape_nd, ge::FORMAT_ND, ge::DT_INT32);
    tensor_desc_nd.SetOriginShape(shape_nd);
    tensor_desc_nd.SetOriginDataType(ge::DT_INT32);
    tensor_desc_nd.SetOriginFormat(ge::FORMAT_ND);
    tensor_desc_nd.SetFormat(ge::FORMAT_ND);
    
    OpDescPtr data_op = std::make_shared<OpDesc>("data", "Data");
    OpDescPtr pypto_op = std::make_shared<OpDesc>("pypto", "LightningIndexerPrologPto");
    data_op->AddOutputDesc(tensor_desc_nd);
    pypto_op->AddInputDesc(tensor_desc_nd);

    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    NodePtr data_node = graph->AddNode(data_op);
    NodePtr pypto_node = graph->AddNode(pypto_op);
    
    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), pypto_node->GetInDataAnchor(0));
    return graph;
  }

  ge::ComputeGraphPtr CreatePyPTOGraph3ForUserSemanticInferUt() {
    std::vector<int64_t> dim_nd = {8, 1, 4096};
    ge::GeShape shape_nd(dim_nd);

    ge::GeTensorDesc tensor_desc_nd(shape_nd, ge::FORMAT_ND, ge::DT_INT32);
    tensor_desc_nd.SetOriginShape(shape_nd);
    tensor_desc_nd.SetOriginDataType(ge::DT_INT32);
    tensor_desc_nd.SetOriginFormat(ge::FORMAT_ND);
    tensor_desc_nd.SetFormat(ge::FORMAT_ND);

    OpDescPtr pypto_op = std::make_shared<OpDesc>("pypto", "LightningIndexerPrologPto");
    pypto_op->AddInputDesc(tensor_desc_nd);
    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    NodePtr pypto_node = graph->AddNode(pypto_op);
    return graph;
  }
};

TEST_F(STEST_UserSemanticInferencePass, optimize_origin_graph_user_semantic)
{
  ComputeGraphPtr graph = CreateGraphForUserSemanticInferUt();
  UserSemanticInferencePass pass;
  vector<GraphPass*> passes = {&pass};
  Status ret = PassManager::Run(*graph, passes, ops_kernel_info_store_ptr);
  EXPECT_EQ(ret, fe::NOT_CHANGED);
}

TEST_F(STEST_UserSemanticInferencePass, optimize_origin_graph_user_semantic_pypto_nz)
{
  ComputeGraphPtr graph = CreatePyPTOGraphForUserSemanticInferUt();
  UserSemanticInferencePass pass;
  vector<GraphPass*> passes = {&pass};
  Status ret = PassManager::Run(*graph, passes, ops_kernel_info_store_ptr);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(STEST_UserSemanticInferencePass, optimize_origin_graph_user_semantic_pypto_nd)
{
  ComputeGraphPtr graph = CreatePyPTOGraph2ForUserSemanticInferUt();
  UserSemanticInferencePass pass;
  vector<GraphPass*> passes = {&pass};
  Status ret = PassManager::Run(*graph, passes, ops_kernel_info_store_ptr);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(STEST_UserSemanticInferencePass, optimize_origin_graph_user_semantic_pypto_nd_2)
{
  ComputeGraphPtr graph = CreatePyPTOGraph3ForUserSemanticInferUt();
  UserSemanticInferencePass pass;
  vector<GraphPass*> passes = {&pass};
  Status ret = PassManager::Run(*graph, passes, ops_kernel_info_store_ptr);
  EXPECT_EQ(ret, fe::SUCCESS);
}