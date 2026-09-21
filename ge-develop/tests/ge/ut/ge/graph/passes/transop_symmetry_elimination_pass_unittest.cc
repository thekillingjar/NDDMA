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
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/passes/format_optimize/transop_symmetry_elimination_pass.h"
#include "graph/passes/shape_optimize/reshape_remove_pass.h"
#include "graph_builder_utils.h"
#include "utils/op_desc_utils.h"

namespace ge {
class UtestTransopSymmetryEliminationPass : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};
namespace {
/**
 *       data
 *         |
 *      transdata1(NCHW->FZ)
 *         |
 *      transdata2(FZ->NCHW)
 *         |
 *      netoutput
 */
ComputeGraphPtr BuildGraphTransDataNotSymm() {
  auto builder = ut::GraphBuilder("test");
  auto data = builder.AddNode("data", DATA, 0, 1);
  auto transdata1 = builder.AddNode("transdata1", TRANSDATA, 1, 1);
  auto transdata2 = builder.AddNode("transdata2", TRANSDATA, 1, 1);
  auto netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0);
  // set transdata1 format&shape
  auto transdata1_input_desc = transdata1->GetOpDesc()->MutableInputDesc(0);
  transdata1_input_desc->SetFormat(FORMAT_FRACTAL_Z);  //src format
  transdata1_input_desc->SetShape(GeShape({1, 1, 16, 16})); //src shape
  transdata1_input_desc->SetOriginFormat(FORMAT_NCHW); // src origin format
  transdata1_input_desc->SetOriginShape(GeShape({16, 1})); // src orgin shape
  auto transdata1_output_desc = transdata1->GetOpDesc()->MutableOutputDesc(0);
  transdata1_output_desc->SetFormat(FORMAT_NCHW);
  transdata1_output_desc->SetShape(GeShape({1, 16, 1, 1}));
  transdata1_output_desc->SetOriginFormat(FORMAT_NCHW);
  transdata1_output_desc->SetOriginShape(GeShape({16, 1}));

  auto transdata2_input_desc = transdata2->GetOpDesc()->MutableInputDesc(0);
  transdata2_input_desc->SetFormat(FORMAT_NCHW);
  transdata2_input_desc->SetShape(GeShape({16, 1, 1, 1}));
  transdata2_input_desc->SetOriginFormat(FORMAT_NCHW);
  transdata2_input_desc->SetOriginShape(GeShape({16, 1, 1, 1}));
  auto transdata2_output_desc = transdata2->GetOpDesc()->MutableOutputDesc(0);
  transdata2_output_desc->SetFormat(FORMAT_FRACTAL_Z); // dst format
  transdata2_output_desc->SetShape(GeShape({1, 1, 16, 16})); // dst shape
  transdata2_output_desc->SetOriginFormat(FORMAT_NCHW); // dst origin format
  transdata2_output_desc->SetOriginShape(GeShape({16, 1, 1, 1})); //dst origin shape, only orgin shape not symmetry

  builder.AddDataEdge(data, 0, transdata1, 0);
  builder.AddDataEdge(transdata1, 0, transdata2, 0);
  builder.AddDataEdge(transdata2, 0, netoutput, 0);
  return builder.GetGraph();
}
/**
 *       data
 *         |
 *      transdata1(NCHW->FZ)
 *         |
 *      transdata2(FZ->NCHW)
 *         |
 *      netoutput
 */
ComputeGraphPtr BuildGraphTransDataFormatNotContinous() {
  auto builder = ut::GraphBuilder("test");
  auto data = builder.AddNode("data", DATA, 0, 1);
  auto transdata1 = builder.AddNode("transdata1", TRANSDATA, 1, 1);
  auto transdata2 = builder.AddNode("transdata2", TRANSDATA, 1, 1);
  auto netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0);
  // set transdata1 format&shape
  auto transdata1_input_desc = transdata1->GetOpDesc()->MutableInputDesc(0);
  transdata1_input_desc->SetFormat(FORMAT_FRACTAL_Z);  //src format
  transdata1_input_desc->SetShape(GeShape({1, 1, 16, 16})); //src shape
  transdata1_input_desc->SetOriginFormat(FORMAT_NCHW); // src origin format
  transdata1_input_desc->SetOriginShape(GeShape({16, 1})); // src orgin shape
  auto transdata1_output_desc = transdata1->GetOpDesc()->MutableOutputDesc(0);
  transdata1_output_desc->SetFormat(FORMAT_NCHW);
  transdata1_output_desc->SetShape(GeShape({1, 16, 1, 1}));
  transdata1_output_desc->SetOriginFormat(FORMAT_NCHW);
  transdata1_output_desc->SetOriginShape(GeShape({16, 1}));

  auto transdata2_input_desc = transdata2->GetOpDesc()->MutableInputDesc(0);
  transdata2_input_desc->SetFormat(FORMAT_HWCN); // format not continous
  transdata2_input_desc->SetShape(GeShape({16, 1, 1, 1}));
  transdata2_input_desc->SetOriginFormat(FORMAT_HWCN);
  transdata2_input_desc->SetOriginShape(GeShape({16, 1, 1, 1}));
  auto transdata2_output_desc = transdata2->GetOpDesc()->MutableOutputDesc(0);
  transdata2_output_desc->SetFormat(FORMAT_FRACTAL_Z); // dst format
  transdata2_output_desc->SetShape(GeShape({1, 1, 16, 16})); // dst shape
  transdata2_output_desc->SetOriginFormat(FORMAT_NCHW); // dst origin format
  transdata2_output_desc->SetOriginShape(GeShape({16, 1})); //dst origin shape

  builder.AddDataEdge(data, 0, transdata1, 0);
  builder.AddDataEdge(transdata1, 0, transdata2, 0);
  builder.AddDataEdge(transdata2, 0, netoutput, 0);
  return builder.GetGraph();
}
} // namespace

TEST_F(UtestTransopSymmetryEliminationPass, keep_not_symm_transdata) {
  auto graph = BuildGraphTransDataFormatNotContinous();
  TransOpSymmetryEliminationPass transop_symmetry_elimination_pass;
  NamesToPass names_to_pass;
  names_to_pass.emplace_back("TransOpSymmetryEliminationPass", &transop_symmetry_elimination_pass);
  GEPass pass(graph);
  Status status = pass.Run(names_to_pass);
  EXPECT_EQ(SUCCESS, status);
  // no elimination happend
  EXPECT_EQ(graph->GetAllNodesSize(), 4);
}
TEST_F(UtestTransopSymmetryEliminationPass, keep_symm_but_not_continous_transdata) {
  auto graph = BuildGraphTransDataNotSymm();
  TransOpSymmetryEliminationPass transop_symmetry_elimination_pass;
  NamesToPass names_to_pass;
  names_to_pass.emplace_back("TransOpSymmetryEliminationPass", &transop_symmetry_elimination_pass);
  GEPass pass(graph);
  Status status = pass.Run(names_to_pass);
  EXPECT_EQ(SUCCESS, status);
  // no elimination happend
  EXPECT_EQ(graph->GetAllNodesSize(), 4);
}

TEST_F(UtestTransopSymmetryEliminationPass, test_transposed) {
  auto builder = ut::GraphBuilder("test1");
  auto data = builder.AddNode("data", DATA, 0, 1);
  auto transpose1 = builder.AddNode("transpose1", TRANSPOSED, 1, 1);
  auto transpose2 = builder.AddNode("transpose2", TRANSPOSED, 1, 1);
  auto netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0);
  
  transpose1->GetOpDesc()->MutableInputDesc(0)->SetFormat(FORMAT_NHWC);
  transpose1->GetOpDesc()->MutableInputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 224, 224, 3})));
  transpose1->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_NCHW);
  transpose1->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 3, 224, 224})));

  transpose2->GetOpDesc()->MutableInputDesc(0)->SetFormat(FORMAT_NCHW);
  transpose2->GetOpDesc()->MutableInputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 3, 224, 224})));
  transpose2->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_NHWC);
  transpose2->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 224, 224, 3})));

  builder.AddDataEdge(data, 0, transpose1, 0);
  builder.AddDataEdge(transpose1, 0, transpose2, 0);
  builder.AddDataEdge(transpose2, 0, netoutput, 0);

  auto graph = builder.GetGraph();

  TransOpSymmetryEliminationPass transop_symmetry_elimination_pass;
  NamesToPass names_to_pass;
  names_to_pass.emplace_back("TransOpSymmetryEliminationPass", &transop_symmetry_elimination_pass);
  GEPass pass(graph);
  Status status = pass.Run(names_to_pass);
  EXPECT_EQ(SUCCESS, status);
  // no elimination happend
  EXPECT_EQ(graph->GetAllNodesSize(), 2);
}

TEST_F(UtestTransopSymmetryEliminationPass, CheckCanBeEliminated_Test) {
  auto builder = ut::GraphBuilder("test1");
  auto data = builder.AddNode("data", DATA, 0, 1);
  auto transpose1 = builder.AddNode("transpose1", RESHAPE, 1, 1);
  auto transpose2 = builder.AddNode("transpose2", RESHAPE, 1, 1);
  auto netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0);
  
  transpose1->GetOpDesc()->MutableInputDesc(0)->SetFormat(FORMAT_NHWC);
  transpose1->GetOpDesc()->MutableInputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 224, 224, 3})));
  transpose1->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_NCHW);
  transpose1->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 3, 224, 224})));

  transpose2->GetOpDesc()->MutableInputDesc(0)->SetFormat(FORMAT_NCHW);
  transpose2->GetOpDesc()->MutableInputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 3, 224, 224})));
  transpose2->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_NHWC);
  transpose2->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 224, 224, 3})));

  builder.AddDataEdge(data, 0, transpose1, 0);
  builder.AddDataEdge(transpose1, 0, transpose2, 0);
  builder.AddDataEdge(transpose2, 0, netoutput, 0);

  auto graph = builder.GetGraph();

  TransOpSymmetryEliminationPass transop_symmetry_elimination_pass;
  NamesToPass names_to_pass;
  names_to_pass.emplace_back("TransOpSymmetryEliminationPass", &transop_symmetry_elimination_pass);
  GEPass pass(graph);
  Status status = pass.Run(names_to_pass);
  EXPECT_EQ(SUCCESS, status);
  // no elimination happend
  EXPECT_EQ(graph->GetAllNodesSize(), 2);
}

TEST_F(UtestTransopSymmetryEliminationPass, DescAreSymmetry_Test) {
  auto builder = ut::GraphBuilder("test1");
  auto data = builder.AddNode("data", DATA, 0, 1);
  auto transpose1 = builder.AddNode("transpose1", CAST, 1, 1);
  auto transpose2 = builder.AddNode("transpose2", CAST, 1, 1);
  auto netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0);
  
  transpose1->GetOpDesc()->MutableInputDesc(0)->SetFormat(FORMAT_NHWC);
  transpose1->GetOpDesc()->MutableInputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 224, 224, 3})));
  transpose1->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_NCHW);
  transpose1->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 3, 224, 224})));

  transpose2->GetOpDesc()->MutableInputDesc(0)->SetFormat(FORMAT_NCHW);
  transpose2->GetOpDesc()->MutableInputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 3, 224, 224})));
  transpose2->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_NHWC);
  transpose2->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 224, 224, 3})));

  builder.AddDataEdge(data, 0, transpose1, 0);
  builder.AddDataEdge(transpose1, 0, transpose2, 0);
  builder.AddDataEdge(transpose2, 0, netoutput, 0);

  auto graph = builder.GetGraph();

  TransOpSymmetryEliminationPass transop_symmetry_elimination_pass;
  NamesToPass names_to_pass;
  names_to_pass.emplace_back("TransOpSymmetryEliminationPass", &transop_symmetry_elimination_pass);
  GEPass pass(graph);
  Status status = pass.Run(names_to_pass);
  EXPECT_EQ(SUCCESS, status);
  // no elimination happend
  EXPECT_EQ(graph->GetAllNodesSize(), 2);
}

TEST_F(UtestTransopSymmetryEliminationPass, CheckCanBeEliminated_Squeezev2_Test) {
  auto builder = ut::GraphBuilder("test1");
  auto data = builder.AddNode("data", DATA, 0, 1);
  auto squueze1 = builder.AddNode("squueze1", SQUEEZEV2, 1, 1);
  auto unsquueze1 = builder.AddNode("unsquueze1", UNSQUEEZEV2, 1, 1);
  auto netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0);
  std::vector<int64_t> axis = {0};
  AttrUtils::SetListInt(squueze1->GetOpDesc(), "axis", axis);
  AttrUtils::SetListInt(unsquueze1->GetOpDesc(), "axis", axis);
  
  squueze1->GetOpDesc()->MutableInputDesc(0)->SetFormat(FORMAT_NCHW);
  squueze1->GetOpDesc()->MutableInputDesc(0)->SetOriginFormat(FORMAT_NCHW);
  squueze1->GetOpDesc()->MutableInputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 1, -1, -1})));
  squueze1->GetOpDesc()->MutableInputDesc(0)->SetOriginShape(GeShape(std::vector<int64_t>({1, -1, -1})));
  squueze1->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_NCHW);
  squueze1->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(FORMAT_NCHW);
  squueze1->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, -1, -1})));
  squueze1->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape(std::vector<int64_t>({1, -1, -1})));

  unsquueze1->GetOpDesc()->MutableInputDesc(0)->SetFormat(FORMAT_NCHW);
  unsquueze1->GetOpDesc()->MutableInputDesc(0)->SetOriginFormat(FORMAT_NCHW);
  unsquueze1->GetOpDesc()->MutableInputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, -1, -1})));
  unsquueze1->GetOpDesc()->MutableInputDesc(0)->SetOriginShape(GeShape(std::vector<int64_t>({1, -1, -1})));
  unsquueze1->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_NCHW);
  unsquueze1->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(FORMAT_NCHW);
  unsquueze1->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 1, -1, -1})));
  unsquueze1->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape(std::vector<int64_t>({1, -1, -1})));

  builder.AddDataEdge(data, 0, squueze1, 0);
  builder.AddDataEdge(squueze1, 0, unsquueze1, 0);
  builder.AddDataEdge(unsquueze1, 0, netoutput, 0);

  auto graph = builder.GetGraph();
  EXPECT_EQ(graph->GetAllNodesSize(), 4);

  TransOpSymmetryEliminationPass transop_symmetry_elimination_pass;
  NamesToPass names_to_pass;
  names_to_pass.emplace_back("TransOpSymmetryEliminationPass", &transop_symmetry_elimination_pass);
  GEPass pass(graph);
  Status status = pass.Run(names_to_pass);
  EXPECT_EQ(SUCCESS, status);
  // elimination happend
  EXPECT_EQ(graph->GetAllNodesSize(), 2);
}

TEST_F(UtestTransopSymmetryEliminationPass, CheckCanNotBeEliminated_Squeezev2_Aixs_not_equal) {
  auto builder = ut::GraphBuilder("test1");
  auto data = builder.AddNode("data", DATA, 0, 1);
  auto squueze1 = builder.AddNode("squueze1", SQUEEZEV2, 1, 1);
  auto unsquueze1 = builder.AddNode("unsquueze1", UNSQUEEZEV2, 1, 1);
  auto netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0);
  std::vector<int64_t> axis1 = {0};
  std::vector<int64_t> axis2 = {1};
  AttrUtils::SetListInt(squueze1->GetOpDesc(), "axis", axis1);
  AttrUtils::SetListInt(unsquueze1->GetOpDesc(), "axis", axis2);
  
  squueze1->GetOpDesc()->MutableInputDesc(0)->SetFormat(FORMAT_NCHW);
  squueze1->GetOpDesc()->MutableInputDesc(0)->SetOriginFormat(FORMAT_NCHW);
  squueze1->GetOpDesc()->MutableInputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 1, -1, -1})));
  squueze1->GetOpDesc()->MutableInputDesc(0)->SetOriginShape(GeShape(std::vector<int64_t>({1, -1, -1})));
  squueze1->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_NCHW);
  squueze1->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(FORMAT_NCHW);
  squueze1->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, -1, -1})));
  squueze1->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape(std::vector<int64_t>({1, -1, -1})));

  unsquueze1->GetOpDesc()->MutableInputDesc(0)->SetFormat(FORMAT_NCHW);
  unsquueze1->GetOpDesc()->MutableInputDesc(0)->SetOriginFormat(FORMAT_NCHW);
  unsquueze1->GetOpDesc()->MutableInputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, -1, -1})));
  unsquueze1->GetOpDesc()->MutableInputDesc(0)->SetOriginShape(GeShape(std::vector<int64_t>({1, -1, -1})));
  unsquueze1->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_NCHW);
  unsquueze1->GetOpDesc()->MutableOutputDesc(0)->SetOriginFormat(FORMAT_NCHW);
  unsquueze1->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 1, -1, -1})));
  unsquueze1->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape(std::vector<int64_t>({1, -1, -1})));

  builder.AddDataEdge(data, 0, squueze1, 0);
  builder.AddDataEdge(squueze1, 0, unsquueze1, 0);
  builder.AddDataEdge(unsquueze1, 0, netoutput, 0);

  auto graph = builder.GetGraph();
  EXPECT_EQ(graph->GetAllNodesSize(), 4);

  TransOpSymmetryEliminationPass transop_symmetry_elimination_pass;
  NamesToPass names_to_pass;
  names_to_pass.emplace_back("TransOpSymmetryEliminationPass", &transop_symmetry_elimination_pass);
  GEPass pass(graph);
  Status status = pass.Run(names_to_pass);
  EXPECT_EQ(SUCCESS, status);
  // elimination happend
  EXPECT_EQ(graph->GetAllNodesSize(), 4);
}

TEST_F(UtestTransopSymmetryEliminationPass, test_TransdataDescAreSymmetry) {
  auto builder = ut::GraphBuilder("test1");
  auto transA = builder.AddNode("transA", TRANSDATA, 1, 1);
  auto transB = builder.AddNode("transB", TRANSDATA, 1, 1);

  { // 5HD->NCHW  -------------->>>>>>>>>>>> NCHW->5HD
    vector<int64_t> src_in_shape{1,16,100,190,16};
    vector<int64_t> src_out_shape{1,256,100,190};
    vector<int64_t> dst_in_shape{8,32,100,190};
    vector<int64_t> dst_out_shape{8,2,100,190,16};

    transA->GetOpDesc()->UpdateInputDesc(0, GeTensorDesc(GeShape(src_in_shape), FORMAT_NC1HWC0, DT_FLOAT));
    transA->GetOpDesc()->UpdateOutputDesc(0, GeTensorDesc(GeShape(src_out_shape), FORMAT_NCHW, DT_FLOAT));
    transB->GetOpDesc()->UpdateInputDesc(0, GeTensorDesc(GeShape(dst_in_shape), FORMAT_NCHW, DT_FLOAT));
    transB->GetOpDesc()->UpdateOutputDesc(0, GeTensorDesc(GeShape(dst_out_shape), FORMAT_NC1HWC0, DT_FLOAT));
    EXPECT_TRUE(TransOpSymmetryEliminationPass::DescAreSymmetry(transA, transB));
  }

  { // NCHW->5HD -------------->>>>>>>>>>>> 5HD->NCHW
    vector<int64_t> src_in_shape{1,16,100,190,16};
    vector<int64_t> src_out_shape{1,256,100,190};
    vector<int64_t> dst_in_shape{8,32,100,190};
    vector<int64_t> dst_out_shape{8,2,100,190,16};

    transA->GetOpDesc()->UpdateOutputDesc(0, GeTensorDesc(GeShape(src_in_shape), FORMAT_NC1HWC0, DT_FLOAT));
    transA->GetOpDesc()->UpdateInputDesc(0, GeTensorDesc(GeShape(src_out_shape), FORMAT_NCHW, DT_FLOAT));
    transB->GetOpDesc()->UpdateOutputDesc(0, GeTensorDesc(GeShape(dst_in_shape), FORMAT_NCHW, DT_FLOAT));
    transB->GetOpDesc()->UpdateInputDesc(0, GeTensorDesc(GeShape(dst_out_shape), FORMAT_NC1HWC0, DT_FLOAT));
    EXPECT_TRUE(TransOpSymmetryEliminationPass::DescAreSymmetry(transA, transB));
  }

  { // NHWC->5HD -------------->>>>>>>>>>>> 5HD->NHWC  NHWC not support
    vector<int64_t> src_in_shape{1,16,100,190,16};
    vector<int64_t> src_out_shape{1,256,100,190};
    vector<int64_t> dst_in_shape{8,32,100,190};
    vector<int64_t> dst_out_shape{8,2,100,190,16};

    transA->GetOpDesc()->UpdateInputDesc(0, GeTensorDesc(GeShape(src_in_shape), FORMAT_NC1HWC0, DT_FLOAT));
    transA->GetOpDesc()->UpdateOutputDesc(0, GeTensorDesc(GeShape(src_out_shape), FORMAT_NHWC, DT_FLOAT));
    transB->GetOpDesc()->UpdateInputDesc(0, GeTensorDesc(GeShape(dst_in_shape), FORMAT_NHWC, DT_FLOAT));
    transB->GetOpDesc()->UpdateOutputDesc(0, GeTensorDesc(GeShape(dst_out_shape), FORMAT_NC1HWC0, DT_FLOAT));
    EXPECT_FALSE(TransOpSymmetryEliminationPass::DescAreSymmetry(transA, transB));
  }

  { // 5HD->NCHW  -------------->>>>>>>>>>>> NCHW->5HD  size is not equal
    vector<int64_t> src_in_shape{1,1,100,190,16};
    vector<int64_t> src_out_shape{1,16,100,190};
    vector<int64_t> dst_in_shape{8,2,100,190};
    vector<int64_t> dst_out_shape{8,1,100,190,16};

    transA->GetOpDesc()->UpdateInputDesc(0, GeTensorDesc(GeShape(src_in_shape), FORMAT_NC1HWC0, DT_FLOAT));
    transA->GetOpDesc()->UpdateOutputDesc(0, GeTensorDesc(GeShape(src_out_shape), FORMAT_NCHW, DT_FLOAT));
    transB->GetOpDesc()->UpdateInputDesc(0, GeTensorDesc(GeShape(dst_in_shape), FORMAT_NCHW, DT_FLOAT));
    transB->GetOpDesc()->UpdateOutputDesc(0, GeTensorDesc(GeShape(dst_out_shape), FORMAT_NC1HWC0, DT_FLOAT));
    EXPECT_FALSE(TransOpSymmetryEliminationPass::DescAreSymmetry(transA, transB));
  }
}

TEST_F(UtestTransopSymmetryEliminationPass, test_reshape_remove_with_transdata_NCHW_5HD) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", DATA)->EDGE(0, 0)->NODE("trans1", TRANSDATA)->EDGE(0, 0)->NODE("reshape", RESHAPE)
              ->EDGE(0, 0)->NODE("trans2", TRANSDATA)->NODE("net_output", NETOUTPUT));
    CHAIN(NODE("const1", CONSTANT)->EDGE(0, 1)->NODE("reshape", RESHAPE));
  };
  auto compute_graph = ToComputeGraph(g1);
  auto reshape = compute_graph->FindNode("reshape");
  EXPECT_NE(reshape, nullptr);
  GeShape reshape_in({1,16,100,190,16});
  GeShape reshape_out({8,2,100,190,16});
  reshape->GetOpDesc()->MutableInputDesc(0)->SetFormat(FORMAT_NC1HWC0);
  reshape->GetOpDesc()->MutableInputDesc(0)->SetShape(reshape_in);
  reshape->GetOpDesc()->MutableOutputDesc(0)->SetShape(reshape_out);
  reshape->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_NC1HWC0);

  auto trans1 = compute_graph->FindNode("trans1");
  EXPECT_NE(nullptr, trans1);
  GeShape trans_in({1,256,100,190});
  trans1->GetOpDesc()->MutableInputDesc(0)->SetFormat(FORMAT_NCHW);
  trans1->GetOpDesc()->MutableInputDesc(0)->SetShape(trans_in);
  trans1->GetOpDesc()->MutableOutputDesc(0)->SetShape(reshape_in);
  trans1->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_NC1HWC0);

  auto trans2 = compute_graph->FindNode("trans2");
  EXPECT_NE(nullptr, trans2);
  GeShape trans_out({8,32,100,190});
  trans2->GetOpDesc()->MutableInputDesc(0)->SetFormat(FORMAT_NC1HWC0);
  trans2->GetOpDesc()->MutableInputDesc(0)->SetShape(reshape_out);
  trans2->GetOpDesc()->MutableOutputDesc(0)->SetShape(trans_out);
  trans2->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_NCHW);

  NamesToPass names_to_pass;
  ReshapeRemovePass reshape_remove_pass;
  TransOpSymmetryEliminationPass transop_symmetry_elimination_pass;
  names_to_pass.push_back({"ReshapeRemovePass", &reshape_remove_pass});
  names_to_pass.emplace_back("TransOpSymmetryEliminationPass", &transop_symmetry_elimination_pass);
  GEPass pass(compute_graph);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);
  EXPECT_EQ(compute_graph->FindNode("trans1"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("trans2"), nullptr);
}

TEST_F(UtestTransopSymmetryEliminationPass, test_reshape_remove_with_transdata_5HD_NCHW) {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data1", DATA)->EDGE(0, 0)->NODE("trans1", TRANSDATA)->EDGE(0, 0)->NODE("reshape", RESHAPE)
                            ->EDGE(0, 0)->NODE("trans2", TRANSDATA)->NODE("net_output", NETOUTPUT));
                  CHAIN(NODE("const1", CONSTANT)->EDGE(0, 1)->NODE("reshape", RESHAPE));
                };
  auto compute_graph = ToComputeGraph(g1);
  auto reshape = compute_graph->FindNode("reshape");
  EXPECT_NE(reshape, nullptr);
  GeShape reshape_in({1,256,100,190});
  GeShape reshape_out({8,32,100,190});
  reshape->GetOpDesc()->MutableInputDesc(0)->SetFormat(FORMAT_NCHW);
  reshape->GetOpDesc()->MutableInputDesc(0)->SetShape(reshape_in);
  reshape->GetOpDesc()->MutableOutputDesc(0)->SetShape(reshape_out);
  reshape->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_NCHW);

  auto trans1 = compute_graph->FindNode("trans1");
  EXPECT_NE(nullptr, trans1);
  GeShape trans_in({1,16,100,190,16});
  trans1->GetOpDesc()->MutableInputDesc(0)->SetFormat(FORMAT_NC1HWC0);
  trans1->GetOpDesc()->MutableInputDesc(0)->SetShape(trans_in);
  trans1->GetOpDesc()->MutableOutputDesc(0)->SetShape(reshape_in);
  trans1->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_NCHW);

  auto trans2 = compute_graph->FindNode("trans2");
  EXPECT_NE(nullptr, trans2);
  GeShape trans_out({8,2,100,190,16});
  trans2->GetOpDesc()->MutableInputDesc(0)->SetFormat(FORMAT_NCHW);
  trans2->GetOpDesc()->MutableInputDesc(0)->SetShape(reshape_out);
  trans2->GetOpDesc()->MutableOutputDesc(0)->SetShape(trans_out);
  trans2->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_NC1HWC0);

  NamesToPass names_to_pass;
  ReshapeRemovePass reshape_remove_pass;
  TransOpSymmetryEliminationPass transop_symmetry_elimination_pass;
  names_to_pass.push_back({"ReshapeRemovePass", &reshape_remove_pass});
  names_to_pass.emplace_back("TransOpSymmetryEliminationPass", &transop_symmetry_elimination_pass);
  GEPass pass(compute_graph);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);
  EXPECT_EQ(compute_graph->FindNode("trans1"), nullptr);
  EXPECT_EQ(compute_graph->FindNode("trans2"), nullptr);
}
}  // namespace ge
