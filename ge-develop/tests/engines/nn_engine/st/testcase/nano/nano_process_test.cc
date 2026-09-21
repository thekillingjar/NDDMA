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
#include "fe_llt_utils.h"
#include "common/fe_op_info_common.h"

#define protected public
#define private public
#include "fusion_manager/fusion_manager.h"
#undef private
#undef protected
#include "itf_handler/itf_handler.h"

using namespace ge;
namespace fe {
class NanoProcessTest: public testing::Test {
protected:
  static void SetUpTestCase() {
    cout << "NanoProcessTest SetUp" << endl;
    InitWithSocVersion("Ascend910B1", "force_fp16");
    FEGraphOptimizerPtr graph_optimizer_ptr = FusionManager::Instance(AI_CORE_NAME).graph_opt_;
    map<string, string> options;
    EXPECT_EQ(graph_optimizer_ptr->Initialize(options, nullptr), SUCCESS);
  }

  static void TearDownTestCase() {
    cout << "NanoProcessTest TearDown" << endl;
    Finalize();
  }
};

// TEST_F(NanoProcessTest, optimize_origin_graph_case1) {
//   FEGraphOptimizerPtr graph_optimizer_ptr = FusionManager::Instance(AI_CORE_NAME).graph_opt_;
//   ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
//   OpDescPtr data = std::make_shared<OpDesc>("data", "Data");
//   OpDescPtr quant = std::make_shared<OpDesc>("quant", "AscendQuant");
//   OpDescPtr weightquant = std::make_shared<OpDesc>("weightquant", "AscendWeightQuant");
//   OpDescPtr mm = std::make_shared<OpDesc>("mm", "MatMulV2");
//   OpDescPtr dequant = std::make_shared<OpDesc>("dequant", "AscendDequant");
//   OpDescPtr relu = std::make_shared<OpDesc>("relu", "Relu");
//   OpDescPtr const1 = std::make_shared<OpDesc>("const1", "Const");
//   OpDescPtr const2 = std::make_shared<OpDesc>("const2", "Const");
//   OpDescPtr const3 = std::make_shared<OpDesc>("const3", "Const");

//   GeShape shape_x1({2, 3});
//   GeTensorDesc x1_desc(shape_x1, ge::FORMAT_ND, ge::DT_INT8);
//   GeShape shape_weight({3, 4});
//   GeTensorDesc weight_desc(shape_weight, ge::FORMAT_ND, ge::DT_INT8);
//   GeShape shape_out({2, 4});
//   GeTensorDesc out_desc(shape_out, ge::FORMAT_ND, ge::DT_INT8);
//   GeShape shape_offset(std::vector<int64_t>{});
//   GeTensorDesc offset_desc(shape_offset, ge::FORMAT_ND, ge::DT_INT8);
//   GeShape shape_deqscale({1});
//   GeTensorDesc deqscale_desc(shape_deqscale, ge::FORMAT_NCHW, ge::DT_UINT64);
//   GeShape shape_deqout({2, 4});
//   GeTensorDesc deqout_desc(shape_deqout, ge::FORMAT_NHWC, ge::DT_FLOAT);
//   data->AddOutputDesc(x1_desc);
//   quant->AddInputDesc(x1_desc);
//   quant->AddOutputDesc(x1_desc);
//   weightquant->AddInputDesc(weight_desc);
//   weightquant->AddInputDesc(offset_desc);
//   weightquant->AddOutputDesc(weight_desc);
//   mm->AddInputDesc(x1_desc);
//   mm->AddInputDesc(weight_desc);
//   mm->AddOutputDesc(out_desc);
//   dequant->AddInputDesc(out_desc);
//   dequant->AddInputDesc(deqscale_desc);
//   dequant->AddOutputDesc(deqout_desc);
//   relu->AddInputDesc(deqout_desc);
//   relu->AddOutputDesc(deqout_desc);
//   const1->AddOutputDesc(weight_desc);
//   const2->AddOutputDesc(offset_desc);
//   const3->AddOutputDesc(deqscale_desc);

//   NodePtr data_node = graph->AddNode(data);
//   NodePtr quant_node = graph->AddNode(quant);
//   NodePtr weightquant_node = graph->AddNode(weightquant);
//   NodePtr mm_node = graph->AddNode(mm);
//   NodePtr dequant_node = graph->AddNode(dequant);
//   NodePtr relu_node = graph->AddNode(relu);
//   NodePtr const1_node = graph->AddNode(const1);
//   NodePtr const2_node = graph->AddNode(const2);
//   NodePtr const3_node = graph->AddNode(const3);
//   AttrUtils::SetFloat(quant, "scale", 1.1);
//   AttrUtils::SetFloat(quant, "offset", 1.2);
//   ge::AttrUtils::SetBool(mm, "transpose_x1", false);
//   ge::AttrUtils::SetBool(mm, "transpose_x2", false);

//   GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), quant_node->GetInDataAnchor(0));
//   GraphUtils::AddEdge(quant_node->GetOutDataAnchor(0), mm_node->GetInDataAnchor(0));
//   GraphUtils::AddEdge(const1_node->GetOutDataAnchor(0), weightquant_node->GetInDataAnchor(0));
//   GraphUtils::AddEdge(const2_node->GetOutDataAnchor(0), weightquant_node->GetInDataAnchor(1));
//   GraphUtils::AddEdge(weightquant_node->GetOutDataAnchor(0), mm_node->GetInDataAnchor(1));
//   GraphUtils::AddEdge(mm_node->GetOutDataAnchor(0), dequant_node->GetInDataAnchor(0));
//   GraphUtils::AddEdge(const3_node->GetOutDataAnchor(0), dequant_node->GetInDataAnchor(1));
//   GraphUtils::AddEdge(dequant_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));
  
//   FillWeightValue(graph);

//   Status ret = graph_optimizer_ptr->OptimizeGraphInit(*graph);
//   EXPECT_EQ(ret, SUCCESS);
//   ret = graph_optimizer_ptr->OptimizeGraphPrepare(*graph);
//   EXPECT_EQ(ret, SUCCESS);
//   ret = graph_optimizer_ptr->OptimizeOriginalGraph(*graph);
//   EXPECT_EQ(ret, SUCCESS);
//   ret = graph_optimizer_ptr->OptimizeOriginalGraphJudgeInsert(*graph);
//   EXPECT_EQ(ret, SUCCESS);
//   ret = graph_optimizer_ptr->OptimizeOriginalGraphJudgeFormatInsert(*graph);
//   EXPECT_EQ(ret, SUCCESS);
//   // EXPECT_EQ(graph->GetDirectNodesSize(), 8);
// }
}
