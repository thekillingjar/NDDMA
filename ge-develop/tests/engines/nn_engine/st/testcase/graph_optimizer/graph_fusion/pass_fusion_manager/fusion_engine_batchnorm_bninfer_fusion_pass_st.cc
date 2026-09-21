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
//#include "graph_optimizer/graph_fusion/fusion_pass_manager/builtin_pass/batchnorm_bninfer_fusion_pass.h"
#include "graph/compute_graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "ops_kernel_store/fe_ops_kernel_info_store.h"
#include "ops_store/ops_kernel_manager.h"
#include "adapter/tbe_adapter/tbe_op_store_adapter.h"
#undef protected
#undef private

using namespace std;
using namespace fe;
using namespace ge;
static const string ATTR_DATA_TYPE = "T";

static const char *BNINFER = "BNInfer";
static const char *IS_TRAINING = "is_training";
static const char *EPSILON = "epsilon";

class fusion_engine_batchnorm_bninfer_fusion_pass_st : public testing::Test {
protected:
  void SetUp() {}
  void TearDown() {}

  static NodePtr CreateConstNode(string name, GeTensorDescPtr out_desc_ptr, ComputeGraphPtr graph)
  {
      OpDescPtr constant = std::make_shared<OpDesc>(name, "Const");
      //set OpDesc
      AttrUtils::SetStr(out_desc_ptr, "name", name + "Out0");
      constant->AddOutputDesc(out_desc_ptr->Clone());
      // set attr
      AttrUtils::SetInt(constant, ATTR_DATA_TYPE, DT_FLOAT);
      NodePtr node_const = graph->AddNode(constant);

      return node_const;
  }

  ComputeGraphPtr CreatePadGraph(string opdesc_type_0) {
    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");

    GeTensorDescPtr tensor_desc = std::make_shared<GeTensorDesc>();
    vector<int64_t> dim = {2, 4, 128, 128, 16};
    GeShape shape(dim);
    tensor_desc->SetShape(shape);
    tensor_desc->SetFormat(FORMAT_NC1HWC0);
    tensor_desc->SetDataType(DT_FLOAT);

    GeTensorDescPtr tensor_desc_c = std::make_shared<GeTensorDesc>();
    vector<int64_t> dim_c = {1, 4, 1, 1, 16};
    GeShape shape_c(dim_c);
    tensor_desc_c->SetShape(shape_c);
    tensor_desc_c->SetFormat(FORMAT_NC1HWC0);
    tensor_desc_c->SetDataType(DT_FLOAT);

    NodePtr node_const0 = CreateConstNode("test/const0", tensor_desc, graph);
    NodePtr node_const1 = CreateConstNode("test/const1", tensor_desc_c, graph);
    NodePtr node_const2 = CreateConstNode("test/const2", tensor_desc_c, graph);
    NodePtr node_const3 = CreateConstNode("test/const3", tensor_desc_c, graph);
    NodePtr node_const4 = CreateConstNode("test/const4", tensor_desc_c, graph);

    OpDescPtr opdesc_0 = make_shared<OpDesc>("BatchNorm", opdesc_type_0);
    opdesc_0->AddInputDesc(tensor_desc->Clone());
    opdesc_0->AddInputDesc(tensor_desc_c->Clone());
    opdesc_0->AddInputDesc(tensor_desc_c->Clone());
    opdesc_0->AddInputDesc(tensor_desc_c->Clone());
    opdesc_0->AddInputDesc(tensor_desc_c->Clone());
    opdesc_0->AddOutputDesc(tensor_desc->Clone());
    opdesc_0->AddOutputDesc(tensor_desc_c->Clone());
    opdesc_0->AddOutputDesc(tensor_desc_c->Clone());
    opdesc_0->AddOutputDesc(tensor_desc_c->Clone());
    opdesc_0->AddOutputDesc(tensor_desc_c->Clone());
    AttrUtils::SetBool(opdesc_0, IS_TRAINING, false);
    AttrUtils::SetFloat(opdesc_0, EPSILON, 0.001);
    NodePtr node_0 = graph->AddNode(opdesc_0);

    GraphUtils::AddEdge(node_const0->GetOutDataAnchor(0),
                        node_0->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_const1->GetOutDataAnchor(0),
                        node_0->GetInDataAnchor(1));
    GraphUtils::AddEdge(node_const2->GetOutDataAnchor(0),
                        node_0->GetInDataAnchor(2));
    GraphUtils::AddEdge(node_const3->GetOutDataAnchor(0),
                        node_0->GetInDataAnchor(3));
    GraphUtils::AddEdge(node_const4->GetOutDataAnchor(0),
                        node_0->GetInDataAnchor(4));

    OpDescPtr opdesc_1 = make_shared<OpDesc>("Relu", "Relu");
    opdesc_1->AddInputDesc(tensor_desc->Clone());
    opdesc_1->AddOutputDesc(tensor_desc->Clone());
    NodePtr node_1 = graph->AddNode(opdesc_1);

    GraphUtils::AddEdge(node_0->GetOutDataAnchor(0),
                        node_1->GetInDataAnchor(0));

    return graph;
  }
};

//TEST_F(fusion_engine_batchnorm_bninfer_fusion_pass_st, bn_relu_success) {
//  ComputeGraphPtr graph =
//      CreatePadGraph("BatchNorm");
//
//  Status status = BatchNormBnInferFusionPass().Run(*graph);
//
//  EXPECT_EQ(fe::SUCCESS, status);
//}
