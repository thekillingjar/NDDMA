
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 #include "graph/ascendc_ir/ascendc_ir_core/ascendc_ir.h"
 #include "graph/attribute_group/attr_group_symbolic_desc.h"
 #include "graph/debug/ge_op_types.h"
 #include "graph/utils/graph_utils_ex.h"
 #include "graph/debug/ge_attr_define.h"
 #include "../../eager_style_graph_builder/all_ops_cpp.h"
 #include "../../eager_style_graph_builder/esb_graph.h"
 #include "lowering/asc_lowerer/loop_api.h"
 #include "lowering/asc_lowerer/asc_overrides.h"
 #include "lowering/lowerings.h"
 #include "utils/autofuse_attrs.h"
 #include "utils/auto_fuse_config.h"
 #include "../../eager_style_graph_builder/compliant_op_desc_builder.h"
 #include "pattern_fusion/flatten_concat_pass.h"
 #include "pattern_fusion/flatten_split_pass.h"
 #include "pattern_fusion/pattern_fusion.h"
 #include "graph_metadef/graph/debug/ge_util.h"
 #include <gtest/gtest.h>
 
 using namespace std;
 using namespace testing;
 namespace ge {
 class PatternFusionBeforeAutoFuseUT : public testing::Test {
    public:
    protected:
    void SetUp() override {
      dlog_setlevel(0, 3, 0);
      ge::autofuse::AutoFuseConfig::MutableLoweringConfig().experimental_lowering_concat = true;
    }
    void TearDown() override {
      dlog_setlevel(0, 3, 0);
    }
   };
   
REG_OP(ConcatV2D)
.DYNAMIC_INPUT(x, TensorType::BasicType())
.OUTPUT(y, TensorType::BasicType())
.REQUIRED_ATTR(concat_dim, Int)
.ATTR(N, Int, 1)
.OP_END_FACTORY_REG(ConcatV2D)

REG_OP(ConcatD)
.DYNAMIC_INPUT(x, TensorType({DT_FLOAT,DT_FLOAT16,DT_INT8,DT_INT16,DT_INT32,DT_INT64,DT_UINT8,DT_UINT16,DT_UINT32,DT_UINT64}))
.OUTPUT(y, TensorType({DT_FLOAT,DT_FLOAT16,DT_INT8,DT_INT16,DT_INT32,DT_INT64,DT_UINT8,DT_UINT16,DT_UINT32,DT_UINT64}))
.REQUIRED_ATTR(concat_dim, Int)
.ATTR(N, Int, 1)
.OP_END_FACTORY_REG(ConcatD)

 TEST_F(PatternFusionBeforeAutoFuseUT, ConcatCombineProcess) {
   ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test1");
   OpDescPtr op_desc_switch1 = std::make_shared<OpDesc>("switch1", "switch");
   OpDescPtr op_desc_cast1 = std::make_shared<OpDesc>("cast1", "Relu");
   OpDescPtr op_desc_cast2 = std::make_shared<OpDesc>("cast2", "Relu");
   OpDescPtr op_desc_cast3 = std::make_shared<OpDesc>("cast3", "Relu");
   OpDescPtr op_desc_cast4 = std::make_shared<OpDesc>("cast4", "Relu");
 
   OpDescPtr op_desc_switch2 = std::make_shared<OpDesc>("switch2", "switch");
   OpDescPtr op_desc_switch3 = std::make_shared<OpDesc>("switch2", "switch");
   OpDescPtr op_desc_switch4 = std::make_shared<OpDesc>("switch3", "switch");
 
   OpDescPtr op_desc_concat1_0 = std::make_shared<OpDesc>("concat1_0", "ConcatV2");
   OpDescPtr op_desc_concat1_1 = std::make_shared<OpDesc>("concat1_1", "ConcatV2D");
   OpDescPtr op_desc_concat2_0 = std::make_shared<OpDesc>("concat2_0", "ConcatV2");
   OpDescPtr op_desc_concat3_0 = std::make_shared<OpDesc>("concat3_0", "ConcatV2D");
 
   vector<int64_t> dim_a = {128, 32};
   GeShape shape_a(dim_a);
   GeTensorDesc tensor_desc_a(shape_a);
   tensor_desc_a.SetFormat(FORMAT_ND);
   tensor_desc_a.SetOriginFormat(FORMAT_ND);
   tensor_desc_a.SetDataType(DT_FLOAT16);
   tensor_desc_a.SetOriginDataType(DT_FLOAT16);
 
   vector<int64_t> dim_b = {128, 16};
   GeShape shape_b(dim_b);
   GeTensorDesc tensor_desc_b(shape_b);
   tensor_desc_b.SetFormat(FORMAT_ND);
   tensor_desc_b.SetOriginFormat(FORMAT_ND);
   tensor_desc_a.SetDataType(DT_FLOAT16);
   tensor_desc_b.SetOriginDataType(DT_FLOAT16);
 
   vector<int64_t> dim_c = {128, 64};
   GeShape shape_c(dim_c);
   GeTensorDesc tensor_desc_c(shape_c);
   tensor_desc_c.SetFormat(FORMAT_ND);
   tensor_desc_c.SetOriginFormat(FORMAT_ND);
   tensor_desc_c.SetDataType(DT_FLOAT16);
   tensor_desc_c.SetOriginDataType(DT_FLOAT16);
 
 
   vector<int64_t> dim_d = {128, 32};
   GeShape shape_d(dim_d);
   GeTensorDesc tensor_desc_d(shape_d);
   tensor_desc_d.SetFormat(FORMAT_ND);
   tensor_desc_d.SetOriginFormat(FORMAT_ND);
   tensor_desc_d.SetDataType(DT_FLOAT16);
   tensor_desc_a.SetOriginDataType(DT_FLOAT16);
 
   op_desc_switch1->AddOutputDesc(tensor_desc_b);
   op_desc_switch2->AddOutputDesc(tensor_desc_b);
   op_desc_switch3->AddOutputDesc(tensor_desc_b);
   op_desc_switch4->AddOutputDesc(tensor_desc_b);
 
   op_desc_cast1->AddOutputDesc(tensor_desc_b);
   op_desc_cast2->AddOutputDesc(tensor_desc_b);
   op_desc_cast3->AddOutputDesc(tensor_desc_b);
   op_desc_cast4->AddOutputDesc(tensor_desc_b);
 
   op_desc_cast1->AddInputDesc(tensor_desc_b);
   op_desc_cast2->AddInputDesc(tensor_desc_b);
   op_desc_cast3->AddInputDesc(tensor_desc_b);
   op_desc_cast4->AddInputDesc(tensor_desc_b);
 
   NodePtr node_switch1 = graph->AddNode(op_desc_switch1);
   NodePtr node_switch2 = graph->AddNode(op_desc_switch2);
   NodePtr node_switch3 = graph->AddNode(op_desc_switch3);
   NodePtr node_switch4 = graph->AddNode(op_desc_switch4);
 
   NodePtr node_cast1 = graph->AddNode(op_desc_cast1);
   NodePtr node_cast2 = graph->AddNode(op_desc_cast2);
   NodePtr node_cast3 = graph->AddNode(op_desc_cast3);
   NodePtr node_cast4 = graph->AddNode(op_desc_cast4);
 
   std::string concatNodeName = "ConcatNode1_1";
   op::ConcatV2D op(concatNodeName.c_str());
   op.BreakConnect();
   op.create_dynamic_input_x(2);
   op.set_attr_concat_dim(1);
   op.set_attr_N(2);
   auto op_desc_concat = ge::OpDescUtils::GetOpDescFromOperator(op);
   NodePtr ConcatNode1_1 = graph->AddNode(op_desc_concat);
 
   std::string concatNodeName1 = "ConcatNode3_1";
   op::ConcatV2D op1(concatNodeName1.c_str());
   op1.BreakConnect();
   op1.create_dynamic_input_x(4);
   op1.set_attr_concat_dim(1);
   op1.set_attr_N(4);
   auto op_desc_concat1 = ge::OpDescUtils::GetOpDescFromOperator(op1);
   NodePtr ConcatNode3_1 = graph->AddNode(op_desc_concat1);
 
   std::string concatNodeName2 = "ConcatNode1_2";
   op::ConcatV2D op2(concatNodeName2.c_str());
   op2.BreakConnect();
   op2.create_dynamic_input_x(2);
   op2.set_attr_concat_dim(1);
   op2.set_attr_N(2);
   auto op_desc_concat2 = ge::OpDescUtils::GetOpDescFromOperator(op2);
   NodePtr ConcatNode1_2 = graph->AddNode(op_desc_concat2);
 
   std::string concatNodeName3 = "ConcatNode2_1";
   ge::op::ConcatV2D op3(concatNodeName3.c_str());
   op3.BreakConnect();
   op3.create_dynamic_input_x(3);
   op3.set_attr_concat_dim(1);
   op3.set_attr_N(3);
   auto op_desc_concat3 = ge::OpDescUtils::GetOpDescFromOperator(op3);
   NodePtr ConcatNode2_1 = graph->AddNode(op_desc_concat3);
 
   GraphUtils::AddEdge(node_switch1->GetOutDataAnchor(0), node_cast1->GetInDataAnchor(0));
   GraphUtils::AddEdge(node_switch2->GetOutDataAnchor(0), node_cast2->GetInDataAnchor(0));
   GraphUtils::AddEdge(node_switch3->GetOutDataAnchor(0), node_cast3->GetInDataAnchor(0));
   GraphUtils::AddEdge(node_switch4->GetOutDataAnchor(0), node_cast4->GetInDataAnchor(0));
 
   GraphUtils::AddEdge(node_switch2->GetOutDataAnchor(0), ConcatNode1_1->GetInDataAnchor(0));
   GraphUtils::AddEdge(node_switch3->GetOutDataAnchor(0), ConcatNode1_1->GetInDataAnchor(1));
 
   GraphUtils::AddEdge(node_switch3->GetOutDataAnchor(0), ConcatNode1_2->GetInDataAnchor(0));
   GraphUtils::AddEdge(node_switch4->GetOutDataAnchor(0), ConcatNode1_2->GetInDataAnchor(1));
 
   GraphUtils::AddEdge(node_cast1->GetOutDataAnchor(0), ConcatNode2_1->GetInDataAnchor(0));
   GraphUtils::AddEdge(ConcatNode1_1->GetOutDataAnchor(0), ConcatNode2_1->GetInDataAnchor(1));
   GraphUtils::AddEdge(node_cast2->GetOutDataAnchor(0), ConcatNode2_1->GetInDataAnchor(2));
 
   GraphUtils::AddEdge(ConcatNode2_1->GetOutDataAnchor(0), ConcatNode3_1->GetInDataAnchor(0));
   GraphUtils::AddEdge(node_cast3->GetOutDataAnchor(0), ConcatNode3_1->GetInDataAnchor(1));
   GraphUtils::AddEdge(ConcatNode1_2->GetOutDataAnchor(0), ConcatNode3_1->GetInDataAnchor(2));
   GraphUtils::AddEdge(node_cast4->GetOutDataAnchor(0), ConcatNode3_1->GetInDataAnchor(3));
 
   ConcatNode1_1->GetOpDesc()->UpdateInputDesc(0, node_switch1->GetOpDesc()->GetOutputDesc(0));
   ConcatNode1_1->GetOpDesc()->UpdateInputDesc(1, node_switch2->GetOpDesc()->GetOutputDesc(0));
   ConcatNode1_1->GetOpDesc()->UpdateOutputDesc(0, tensor_desc_a);
 
   ConcatNode1_2->GetOpDesc()->UpdateInputDesc(0, node_switch3->GetOpDesc()->GetOutputDesc(0));
   ConcatNode1_2->GetOpDesc()->UpdateInputDesc(1, node_switch4->GetOpDesc()->GetOutputDesc(0));
   ConcatNode1_2->GetOpDesc()->UpdateOutputDesc(0, tensor_desc_a);
 
 
   ConcatNode2_1->GetOpDesc()->UpdateInputDesc(0, node_cast1->GetOpDesc()->GetOutputDesc(0));
   ConcatNode2_1->GetOpDesc()->UpdateInputDesc(1, ConcatNode1_1->GetOpDesc()->GetOutputDesc(0));
   ConcatNode2_1->GetOpDesc()->UpdateInputDesc(2, node_cast2->GetOpDesc()->GetOutputDesc(0));
   ConcatNode2_1->GetOpDesc()->UpdateOutputDesc(0, tensor_desc_c);
   ConcatNode2_1->GetOpDesc()->MutableOutputDesc(0)->GetOrCreateAttrsGroup<ge::SymbolicDescAttr>()
       ->symbolic_tensor.MutableOriginSymbolShape() = {Symbol("s0"), Symbol("s1")};
 
 
   ConcatNode3_1->GetOpDesc()->UpdateInputDesc(0, ConcatNode2_1->GetOpDesc()->GetOutputDesc(0));
   ConcatNode3_1->GetOpDesc()->UpdateInputDesc(1, node_cast3->GetOpDesc()->GetOutputDesc(0));
   ConcatNode3_1->GetOpDesc()->UpdateInputDesc(2, ConcatNode1_2->GetOpDesc()->GetOutputDesc(0));
   ConcatNode3_1->GetOpDesc()->UpdateInputDesc(3, node_cast4->GetOpDesc()->GetOutputDesc(0));
   ConcatNode3_1->GetOpDesc()->UpdateOutputDesc(0, tensor_desc_d);
   ConcatNode3_1->GetOpDesc()->MutableOutputDesc(0)->GetOrCreateAttrsGroup<ge::SymbolicDescAttr>()
       ->symbolic_tensor.MutableOriginSymbolShape() = {Symbol("s0"), Symbol("s1")};
   graph->TopologicalSorting();
   FlattenConcatPass multiConcatFusionPro;
   ge::autofuse::AutoFuseConfig::MutableLoweringConfig().experimental_lowering_concat = false;
   EXPECT_EQ(multiConcatFusionPro.Run(graph), ge::GRAPH_SUCCESS);
   ge::autofuse::AutoFuseConfig::MutableLoweringConfig().experimental_lowering_concat = true;
   auto result = multiConcatFusionPro.Run(graph);
 
   for (auto &node : graph->GetDirectNode()) {
     //if ((node->GetType() == CONCATV2) || (node->GetType() == CONCATV2D)) {
       printf("Graph_multisliceConcat_CountFusionOutAnchor:split_out_name is %s start;\n",
         node->GetName().c_str());
       
     //}
   }
 }

 REG_OP(Split)
.INPUT(split_dim, TensorType({DT_INT32,DT_INT64}))
.INPUT(x, TensorType::BasicType())
.DYNAMIC_OUTPUT(y, TensorType::BasicType())
.ATTR(num_split, Int, 1)
.OP_END_FACTORY_REG(Split)

 REG_OP(SplitD)
 .INPUT(x, TensorType::BasicType())
 .DYNAMIC_OUTPUT(y, TensorType::BasicType())
 .REQUIRED_ATTR(split_dim, Int)
 .ATTR(num_split, Int, 1)
 .OP_END_FACTORY_REG(SplitD)

 REG_OP(SplitVD)
.INPUT(x, TensorType::BasicType())
.DYNAMIC_OUTPUT(y, TensorType::BasicType())
.REQUIRED_ATTR(split_dim, Int)
.ATTR(num_split, Int, 1)
.OP_END_FACTORY_REG(SplitVD)

 REG_OP(SplitV)
.INPUT(x, TensorType::BasicType())
.INPUT(size_splits, TensorType({DT_INT32,DT_INT64}))
.INPUT(split_dim, TensorType({DT_INT32,DT_INT64}))
.DYNAMIC_OUTPUT(y, TensorType::BasicType())
.ATTR(num_split, Int, 1)
.OP_END_FACTORY_REG(SplitV)

 REG_OP(Const)
    .OUTPUT(y,
            TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_UINT32,
                        DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .ATTR(value, Tensor, Tensor())

    .OP_END_FACTORY_REG(Const);

 TEST_F(PatternFusionBeforeAutoFuseUT, SplitCombineProcess) {
   ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test1");
   OpDescPtr op_desc_switch1 = std::make_shared<OpDesc>("switch1", "switch");
   OpDescPtr op_desc_cast1 = std::make_shared<OpDesc>("cast1", "Relu");

   // OpDescPtr op_desc_split1_0 = std::make_shared<OpDesc>("split1_0", "SplitD");
   // OpDescPtr op_desc_split2_0 = std::make_shared<OpDesc>("split2_0", "SplitD");
   // OpDescPtr op_desc_split2_1 = std::make_shared<OpDesc>("split2_1", "SplitD");

   vector<int64_t> dim_a = {128, 32};
   GeShape shape_a(dim_a);
   GeTensorDesc tensor_desc_a(shape_a);
   tensor_desc_a.SetFormat(FORMAT_ND);
   tensor_desc_a.SetOriginFormat(FORMAT_ND);
   tensor_desc_a.SetDataType(DT_FLOAT16);
   tensor_desc_a.SetOriginDataType(DT_FLOAT16);

   vector<int64_t> dim_b = {128, 16};
   GeShape shape_b(dim_b);
   GeTensorDesc tensor_desc_b(shape_b);
   tensor_desc_b.SetFormat(FORMAT_ND);
   tensor_desc_b.SetOriginFormat(FORMAT_ND);
   tensor_desc_b.SetDataType(DT_FLOAT16);
   tensor_desc_b.SetOriginDataType(DT_FLOAT16);

   vector<int64_t> dim_c = {128, 2};
   GeShape shape_c(dim_c);
   GeTensorDesc tensor_desc_c(shape_c);
   tensor_desc_c.SetFormat(FORMAT_ND);
   tensor_desc_c.SetOriginFormat(FORMAT_ND);
   tensor_desc_c.SetDataType(DT_FLOAT16);
   tensor_desc_c.SetOriginDataType(DT_FLOAT16);


   vector<int64_t> dim_d = {128, 32};
   GeShape shape_d(dim_d);
   GeTensorDesc tensor_desc_d(shape_d);
   tensor_desc_d.SetFormat(FORMAT_ND);
   tensor_desc_d.SetOriginFormat(FORMAT_ND);
   tensor_desc_d.SetDataType(DT_FLOAT16);
   tensor_desc_d.SetOriginDataType(DT_FLOAT16);

   op_desc_switch1->AddOutputDesc(tensor_desc_a);

   op_desc_cast1->AddOutputDesc(tensor_desc_a);

   op_desc_cast1->AddInputDesc(tensor_desc_a);

   NodePtr node_switch1 = graph->AddNode(op_desc_switch1);

   NodePtr node_cast1 = graph->AddNode(op_desc_cast1);

   std::string splitNodeName = "SplitNode1_1";
   op::SplitD op(splitNodeName.c_str());
   op.BreakConnect();
   op.create_dynamic_output_y(2);
   op.set_attr_split_dim(1);
   op.set_attr_num_split(2);
   auto op_desc_split = ge::OpDescUtils::GetOpDescFromOperator(op);
   NodePtr SplitNode1_1 = graph->AddNode(op_desc_split);

   std::string splitNodeName1 = "SplitNode2_1";
   op::SplitD op1(splitNodeName1.c_str());
   op1.BreakConnect();
   op1.create_dynamic_output_y(4);
   op1.set_attr_split_dim(1);
   op1.set_attr_num_split(4);
   auto op_desc_split1 = ge::OpDescUtils::GetOpDescFromOperator(op1);
   NodePtr SplitNode2_1 = graph->AddNode(op_desc_split1);

   std::string splitNodeName2 = "SplitNode2_2";
   op::SplitD op2(splitNodeName2.c_str());
   op2.BreakConnect();
   op2.create_dynamic_output_y(4);
   op2.set_attr_split_dim(1);
   op2.set_attr_num_split(4);
   auto op_desc_split2 = ge::OpDescUtils::GetOpDescFromOperator(op2);
   NodePtr SplitNode2_2 = graph->AddNode(op_desc_split2);

   GraphUtils::AddEdge(node_switch1->GetOutDataAnchor(0), node_cast1->GetInDataAnchor(0));

   GraphUtils::AddEdge(node_cast1->GetOutDataAnchor(0), SplitNode1_1->GetInDataAnchor(0));

   GraphUtils::AddEdge(SplitNode1_1->GetOutDataAnchor(0), SplitNode2_1->GetInDataAnchor(0));
   GraphUtils::AddEdge(SplitNode1_1->GetOutDataAnchor(1), SplitNode2_2->GetInDataAnchor(0));

   SplitNode1_1->GetOpDesc()->UpdateInputDesc(0, node_switch1->GetOpDesc()->GetOutputDesc(0));
   SplitNode1_1->GetOpDesc()->UpdateOutputDesc(0, tensor_desc_b);
   SplitNode1_1->GetOpDesc()->UpdateOutputDesc(1, tensor_desc_b);
   SplitNode1_1->GetOpDesc()->MutableOutputDesc(0)->GetOrCreateAttrsGroup<ge::SymbolicDescAttr>()
       ->symbolic_tensor.MutableOriginSymbolShape() = {Symbol("s0"), Symbol("s1")};
  SplitNode1_1->GetOpDesc()->MutableOutputDesc(1)->GetOrCreateAttrsGroup<ge::SymbolicDescAttr>()
       ->symbolic_tensor.MutableOriginSymbolShape() = {Symbol("s0"), Symbol("s1")};

   SplitNode2_1->GetOpDesc()->UpdateInputDesc(0, SplitNode1_1->GetOpDesc()->GetOutputDesc(0));
   SplitNode2_1->GetOpDesc()->UpdateOutputDesc(0, tensor_desc_c);
   SplitNode2_1->GetOpDesc()->UpdateOutputDesc(1, tensor_desc_c);
   SplitNode2_1->GetOpDesc()->UpdateOutputDesc(2, tensor_desc_c);
   SplitNode2_1->GetOpDesc()->UpdateOutputDesc(3, tensor_desc_c);

   SplitNode2_2->GetOpDesc()->UpdateInputDesc(0, SplitNode1_1->GetOpDesc()->GetOutputDesc(1));
   SplitNode2_2->GetOpDesc()->UpdateOutputDesc(0, tensor_desc_c);
   SplitNode2_2->GetOpDesc()->UpdateOutputDesc(1, tensor_desc_c);
   SplitNode2_2->GetOpDesc()->UpdateOutputDesc(2, tensor_desc_c);
   SplitNode2_2->GetOpDesc()->UpdateOutputDesc(3, tensor_desc_c);

   graph->TopologicalSorting();
   GE_DUMP(graph, "PatternFusionBeforeAutoFuseUT_BeforeFlattenSplitDPass");
   FlattenSplitPass multiSplitFusionPro;
   auto result = multiSplitFusionPro.Run(graph);
   GE_DUMP(graph, "PatternFusionBeforeAutoFuseUT_AfterFlattenSplitDPass");

   for (auto &node : graph->GetDirectNode()) {
     // if ((node->GetType() == SPLITV) || (node->GetType() == SPLITVD)) {
       printf("Graph_multiSplit_CountFusionOutAnchor:split_out_name is %s start;\n",
         node->GetName().c_str());

     // }
   }
 }

 TEST_F(PatternFusionBeforeAutoFuseUT, SplitVDCombineProcess) {
   ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test1");
   OpDescPtr op_desc_switch1 = std::make_shared<OpDesc>("switch1", "switch");
   OpDescPtr op_desc_cast1 = std::make_shared<OpDesc>("cast1", "Relu");

   // OpDescPtr op_desc_split1_0 = std::make_shared<OpDesc>("split1_0", "SplitD");
   // OpDescPtr op_desc_split2_0 = std::make_shared<OpDesc>("split2_0", "SplitD");
   // OpDescPtr op_desc_split2_1 = std::make_shared<OpDesc>("split2_1", "SplitD");

   vector<int64_t> dim_a = {128, 32};
   GeShape shape_a(dim_a);
   GeTensorDesc tensor_desc_a(shape_a);
   tensor_desc_a.SetFormat(FORMAT_ND);
   tensor_desc_a.SetOriginFormat(FORMAT_ND);
   tensor_desc_a.SetDataType(DT_FLOAT16);
   tensor_desc_a.SetOriginDataType(DT_FLOAT16);

   vector<int64_t> dim_b = {128, 16};
   GeShape shape_b(dim_b);
   GeTensorDesc tensor_desc_b(shape_b);
   tensor_desc_b.SetFormat(FORMAT_ND);
   tensor_desc_b.SetOriginFormat(FORMAT_ND);
   tensor_desc_b.SetDataType(DT_FLOAT16);
   tensor_desc_b.SetOriginDataType(DT_FLOAT16);

   vector<int64_t> dim_c = {128, 2};
   GeShape shape_c(dim_c);
   GeTensorDesc tensor_desc_c(shape_c);
   tensor_desc_c.SetFormat(FORMAT_ND);
   tensor_desc_c.SetOriginFormat(FORMAT_ND);
   tensor_desc_c.SetDataType(DT_FLOAT16);
   tensor_desc_c.SetOriginDataType(DT_FLOAT16);


   vector<int64_t> dim_d = {128, 32};
   GeShape shape_d(dim_d);
   GeTensorDesc tensor_desc_d(shape_d);
   tensor_desc_d.SetFormat(FORMAT_ND);
   tensor_desc_d.SetOriginFormat(FORMAT_ND);
   tensor_desc_d.SetDataType(DT_FLOAT16);
   tensor_desc_d.SetOriginDataType(DT_FLOAT16);

   op_desc_switch1->AddOutputDesc(tensor_desc_a);

   op_desc_cast1->AddOutputDesc(tensor_desc_a);

   op_desc_cast1->AddInputDesc(tensor_desc_a);

   NodePtr node_switch1 = graph->AddNode(op_desc_switch1);

   NodePtr node_cast1 = graph->AddNode(op_desc_cast1);

   std::string splitNodeName = "SplitNode1_1";
   op::SplitVD op(splitNodeName.c_str());
   op.BreakConnect();
   op.create_dynamic_output_y(2);
   op.set_attr_split_dim(1);
   op.set_attr_num_split(2);
   auto op_desc_split = ge::OpDescUtils::GetOpDescFromOperator(op);
   NodePtr SplitNode1_1 = graph->AddNode(op_desc_split);

   std::string splitNodeName1 = "SplitNode2_1";
   op::SplitVD op1(splitNodeName1.c_str());
   op1.BreakConnect();
   op1.create_dynamic_output_y(4);
   op1.set_attr_split_dim(1);
   op1.set_attr_num_split(4);
   auto op_desc_split1 = ge::OpDescUtils::GetOpDescFromOperator(op1);
   NodePtr SplitNode2_1 = graph->AddNode(op_desc_split1);

   std::string splitNodeName2 = "SplitNode2_2";
   op::SplitVD op2(splitNodeName2.c_str());
   op2.BreakConnect();
   op2.create_dynamic_output_y(4);
   op2.set_attr_split_dim(1);
   op2.set_attr_num_split(4);
   auto op_desc_split2 = ge::OpDescUtils::GetOpDescFromOperator(op2);
   NodePtr SplitNode2_2 = graph->AddNode(op_desc_split2);

   GraphUtils::AddEdge(node_switch1->GetOutDataAnchor(0), node_cast1->GetInDataAnchor(0));

   GraphUtils::AddEdge(node_cast1->GetOutDataAnchor(0), SplitNode1_1->GetInDataAnchor(0));

   GraphUtils::AddEdge(SplitNode1_1->GetOutDataAnchor(0), SplitNode2_1->GetInDataAnchor(0));
   GraphUtils::AddEdge(SplitNode1_1->GetOutDataAnchor(1), SplitNode2_2->GetInDataAnchor(0));

   SplitNode1_1->GetOpDesc()->UpdateInputDesc(0, node_switch1->GetOpDesc()->GetOutputDesc(0));
   SplitNode1_1->GetOpDesc()->UpdateOutputDesc(0, tensor_desc_b);
   SplitNode1_1->GetOpDesc()->UpdateOutputDesc(1, tensor_desc_b);
   SplitNode1_1->GetOpDesc()->MutableOutputDesc(0)->GetOrCreateAttrsGroup<ge::SymbolicDescAttr>()
       ->symbolic_tensor.MutableOriginSymbolShape() = {Symbol("s0"), Symbol("s1")};
  SplitNode1_1->GetOpDesc()->MutableOutputDesc(1)->GetOrCreateAttrsGroup<ge::SymbolicDescAttr>()
       ->symbolic_tensor.MutableOriginSymbolShape() = {Symbol("s0"), Symbol("s1")};

   SplitNode2_1->GetOpDesc()->UpdateInputDesc(0, SplitNode1_1->GetOpDesc()->GetOutputDesc(0));
   SplitNode2_1->GetOpDesc()->UpdateOutputDesc(0, tensor_desc_c);
   SplitNode2_1->GetOpDesc()->UpdateOutputDesc(1, tensor_desc_c);
   SplitNode2_1->GetOpDesc()->UpdateOutputDesc(2, tensor_desc_c);
   SplitNode2_1->GetOpDesc()->UpdateOutputDesc(3, tensor_desc_c);

   SplitNode2_2->GetOpDesc()->UpdateInputDesc(0, SplitNode1_1->GetOpDesc()->GetOutputDesc(1));
   SplitNode2_2->GetOpDesc()->UpdateOutputDesc(0, tensor_desc_c);
   SplitNode2_2->GetOpDesc()->UpdateOutputDesc(1, tensor_desc_c);
   SplitNode2_2->GetOpDesc()->UpdateOutputDesc(2, tensor_desc_c);
   SplitNode2_2->GetOpDesc()->UpdateOutputDesc(3, tensor_desc_c);

   graph->TopologicalSorting();
   FlattenSplitPass multiSplitFusionPro;
   auto result = multiSplitFusionPro.Run(graph);

   for (auto &node : graph->GetDirectNode()) {
     // if ((node->GetType() == SPLITV) || (node->GetType() == SPLITVD)) {
       printf("Graph_multiSplit_CountFusionOutAnchor:split_out_name is %s start;\n",
         node->GetName().c_str());

     // }
   }
 }

  TEST_F(PatternFusionBeforeAutoFuseUT, SplitCombineProcessNoSplitD) {
   dlog_setlevel(0, 0, 1);
   setenv("DUMP_GE_GRAPH", "1", 1);
   setenv("DUMP_GRAPH_LEVEL", "1", 1);
   setenv("DUMP_GRAPH_PATH", "./", 1);
   ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test1");
   OpDescPtr op_desc_switch1 = std::make_shared<OpDesc>("switch1", "switch");
   OpDescPtr op_desc_cast1 = std::make_shared<OpDesc>("cast1", "Relu");

   // OpDescPtr op_desc_split1_0 = std::make_shared<OpDesc>("split1_0", "Split");
   // OpDescPtr op_desc_split2_0 = std::make_shared<OpDesc>("split2_0", "Split");
   // OpDescPtr op_desc_split2_1 = std::make_shared<OpDesc>("split2_1", "Split");
   //
   // OpDescPtr op_desc_split_dim0 = std::make_shared<OpDesc>("split_dim0", "Const");
   // OpDescPtr op_desc_split_dim1 = std::make_shared<OpDesc>("split_dim1", "Const");
   // OpDescPtr op_desc_split_dim2 = std::make_shared<OpDesc>("split_dim2", "Const");

   vector<int64_t> dim_a = {128, 32};
   GeShape shape_a(dim_a);
   GeTensorDesc tensor_desc_a(shape_a);
   tensor_desc_a.SetFormat(FORMAT_ND);
   tensor_desc_a.SetOriginFormat(FORMAT_ND);
   tensor_desc_a.SetDataType(DT_FLOAT16);
   tensor_desc_a.SetOriginDataType(DT_FLOAT16);

   vector<int64_t> dim_b = {128, 16};
   GeShape shape_b(dim_b);
   GeTensorDesc tensor_desc_b(shape_b);
   tensor_desc_b.SetFormat(FORMAT_ND);
   tensor_desc_b.SetOriginFormat(FORMAT_ND);
   tensor_desc_b.SetDataType(DT_FLOAT16);
   tensor_desc_b.SetOriginDataType(DT_FLOAT16);

   vector<int64_t> dim_c = {128, 2};
   GeShape shape_c(dim_c);
   GeTensorDesc tensor_desc_c(shape_c);
   tensor_desc_c.SetFormat(FORMAT_ND);
   tensor_desc_c.SetOriginFormat(FORMAT_ND);
   tensor_desc_c.SetDataType(DT_FLOAT16);
   tensor_desc_c.SetOriginDataType(DT_FLOAT16);


   vector<int64_t> dim_d = {128, 32};
   GeShape shape_d(dim_d);
   GeTensorDesc tensor_desc_d(shape_d);
   tensor_desc_d.SetFormat(FORMAT_ND);
   tensor_desc_d.SetOriginFormat(FORMAT_ND);
   tensor_desc_d.SetDataType(DT_FLOAT16);
   tensor_desc_d.SetOriginDataType(DT_FLOAT16);

   op_desc_switch1->AddOutputDesc(tensor_desc_a);

   op_desc_cast1->AddOutputDesc(tensor_desc_a);

   op_desc_cast1->AddInputDesc(tensor_desc_a);

   NodePtr node_switch1 = graph->AddNode(op_desc_switch1);

   NodePtr node_cast1 = graph->AddNode(op_desc_cast1);


   std::string splitNodeName = "SplitNode1_1";
   op::Split op(splitNodeName.c_str());
   op.BreakConnect();
   op.create_dynamic_output_y(2);
   op.set_attr_num_split(2);
   auto op_desc_split = ge::OpDescUtils::GetOpDescFromOperator(op);
   NodePtr SplitNode1_1 = graph->AddNode(op_desc_split);

   std::string constNodeName = "split_dim1_int32";
   op::Const split_dim1_int32(constNodeName.c_str());
   split_dim1_int32.BreakConnect();
   auto op_desc_split_dim1_int32 = ge::OpDescUtils::GetOpDescFromOperator(split_dim1_int32);

   GeTensorDesc td{GeShape(std::vector<int64_t>({1})), FORMAT_ND, DT_INT32};
   GeTensor tensor(td);
   // int32 下的 1 强转成小端 uint8
   tensor.SetData(std::vector<uint8_t>{1,0,0,0});
   AttrUtils::SetTensor(op_desc_split_dim1_int32, "value", tensor);
   NodePtr node_split_dim1_int32 = graph->AddNode(op_desc_split_dim1_int32);




   std::string splitNodeName1 = "SplitNode2_1";
   op::Split op1(splitNodeName1.c_str());
   op1.BreakConnect();
   op1.create_dynamic_output_y(4);
   op1.set_attr_num_split(4);
   auto op_desc_split1 = ge::OpDescUtils::GetOpDescFromOperator(op1);
   NodePtr SplitNode2_1 = graph->AddNode(op_desc_split1);

   std::string constNodeName1 = "split_dim1_int64";
   op::Const split_dim1_int64(constNodeName1.c_str());
   split_dim1_int64.BreakConnect();
   auto op_desc_split_dim1_int64 = ge::OpDescUtils::GetOpDescFromOperator(split_dim1_int64);

   GeTensorDesc td1{GeShape(std::vector<int64_t>({1})), FORMAT_ND, DT_INT64};
   GeTensor tensor1(td1);
   // int64 下的 1 强转成小端 uint8
   tensor1.SetData(std::vector<uint8_t>{1,0,0,0,0,0,0,0});
   AttrUtils::SetTensor(op_desc_split_dim1_int64, "value", tensor1);
   NodePtr node_split_dim1_int64 = graph->AddNode(op_desc_split_dim1_int64);

   std::string splitNodeName2 = "SplitNode2_2";
   op::Split op2(splitNodeName2.c_str());
   op2.BreakConnect();
   op2.create_dynamic_output_y(4);
   op2.set_attr_num_split(4);
   auto op_desc_split2 = ge::OpDescUtils::GetOpDescFromOperator(op2);
   NodePtr SplitNode2_2 = graph->AddNode(op_desc_split2);

   std::string constNodeName2 = "split_dim2_int64";
   op::Const split_dim2_int64(constNodeName2.c_str());
   split_dim1_int64.BreakConnect();
   auto op_desc_split_dim2_int64 = ge::OpDescUtils::GetOpDescFromOperator(split_dim2_int64);

   GeTensorDesc td2{GeShape(std::vector<int64_t>({1})), FORMAT_ND, DT_INT64};
   GeTensor tensor2(td2);
   // int64 下的 2 强转成小端 uint8
   tensor2.SetData(std::vector<uint8_t>{1,0,0,0,0,0,0,0});
   AttrUtils::SetTensor(op_desc_split_dim2_int64, "value", tensor2);
   NodePtr node_split_dim2_int64 = graph->AddNode(op_desc_split_dim2_int64);

   GraphUtils::AddEdge(node_switch1->GetOutDataAnchor(0), node_cast1->GetInDataAnchor(0));

   GraphUtils::AddEdge(node_cast1->GetOutDataAnchor(0), SplitNode1_1->GetInDataAnchor(1));
   GraphUtils::AddEdge(node_split_dim1_int32->GetOutDataAnchor(0), SplitNode1_1->GetInDataAnchor(0));

   GraphUtils::AddEdge(SplitNode1_1->GetOutDataAnchor(0), SplitNode2_1->GetInDataAnchor(1));
   GraphUtils::AddEdge(node_split_dim1_int64->GetOutDataAnchor(0), SplitNode2_1->GetInDataAnchor(0));
   GraphUtils::AddEdge(SplitNode1_1->GetOutDataAnchor(1), SplitNode2_2->GetInDataAnchor(1));
   GraphUtils::AddEdge(node_split_dim2_int64->GetOutDataAnchor(0), SplitNode2_2->GetInDataAnchor(0));

   SplitNode1_1->GetOpDesc()->UpdateInputDesc(0, node_switch1->GetOpDesc()->GetOutputDesc(0));
   SplitNode1_1->GetOpDesc()->UpdateOutputDesc(0, tensor_desc_b);
   SplitNode1_1->GetOpDesc()->UpdateOutputDesc(1, tensor_desc_b);
   SplitNode1_1->GetOpDesc()->MutableOutputDesc(0)->GetOrCreateAttrsGroup<ge::SymbolicDescAttr>()
       ->symbolic_tensor.MutableOriginSymbolShape() = {Symbol("s0"), Symbol("s1")};
  SplitNode1_1->GetOpDesc()->MutableOutputDesc(1)->GetOrCreateAttrsGroup<ge::SymbolicDescAttr>()
       ->symbolic_tensor.MutableOriginSymbolShape() = {Symbol("s0"), Symbol("s1")};

   SplitNode2_1->GetOpDesc()->UpdateInputDesc(0, SplitNode1_1->GetOpDesc()->GetOutputDesc(0));
   SplitNode2_1->GetOpDesc()->UpdateOutputDesc(0, tensor_desc_c);
   SplitNode2_1->GetOpDesc()->UpdateOutputDesc(1, tensor_desc_c);
   SplitNode2_1->GetOpDesc()->UpdateOutputDesc(2, tensor_desc_c);
   SplitNode2_1->GetOpDesc()->UpdateOutputDesc(3, tensor_desc_c);

   SplitNode2_2->GetOpDesc()->UpdateInputDesc(0, SplitNode1_1->GetOpDesc()->GetOutputDesc(1));
   SplitNode2_2->GetOpDesc()->UpdateOutputDesc(0, tensor_desc_c);
   SplitNode2_2->GetOpDesc()->UpdateOutputDesc(1, tensor_desc_c);
   SplitNode2_2->GetOpDesc()->UpdateOutputDesc(2, tensor_desc_c);
   SplitNode2_2->GetOpDesc()->UpdateOutputDesc(3, tensor_desc_c);
   graph->TopologicalSorting();
   GE_DUMP(graph, "PatternFusionBeforeAutoFuseUT_BeforeFlattenSplitPass");
   FlattenSplitPass multiSplitFusionPro;
   auto result = multiSplitFusionPro.Run(graph);
   GE_DUMP(graph, "PatternFusionBeforeAutoFuseUT_AfterFlattenSplitPass");
   for (auto &node : graph->GetDirectNode()) {
     // if ((node->GetType() == SPLITV) || (node->GetType() == SPLITVD)) {
       printf("Graph_multiSplit_CountFusionOutAnchor:split_out_name is %s start;\n",
         node->GetName().c_str());
     // }
   }
 }

  TEST_F(PatternFusionBeforeAutoFuseUT, SplitVCombineProcess) {
   ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test1");
   OpDescPtr op_desc_switch1 = std::make_shared<OpDesc>("switch1", "switch");
   OpDescPtr op_desc_cast1 = std::make_shared<OpDesc>("cast1", "Relu");

   // OpDescPtr op_desc_split1_0 = std::make_shared<OpDesc>("split1_0", "Split");
   // OpDescPtr op_desc_split2_0 = std::make_shared<OpDesc>("split2_0", "Split");
   // OpDescPtr op_desc_split2_1 = std::make_shared<OpDesc>("split2_1", "Split");
   //
   // OpDescPtr op_desc_split_dim0 = std::make_shared<OpDesc>("split_dim0", "Const");
   // OpDescPtr op_desc_split_dim1 = std::make_shared<OpDesc>("split_dim1", "Const");
   // OpDescPtr op_desc_split_dim2 = std::make_shared<OpDesc>("split_dim2", "Const");

   vector<int64_t> dim_a = {128, 32};
   GeShape shape_a(dim_a);
   GeTensorDesc tensor_desc_a(shape_a);
   tensor_desc_a.SetFormat(FORMAT_ND);
   tensor_desc_a.SetOriginFormat(FORMAT_ND);
   tensor_desc_a.SetDataType(DT_FLOAT16);
   tensor_desc_a.SetOriginDataType(DT_FLOAT16);

   vector<int64_t> dim_b = {128, 16};
   GeShape shape_b(dim_b);
   GeTensorDesc tensor_desc_b(shape_b);
   tensor_desc_b.SetFormat(FORMAT_ND);
   tensor_desc_b.SetOriginFormat(FORMAT_ND);
   tensor_desc_b.SetDataType(DT_FLOAT16);
   tensor_desc_b.SetOriginDataType(DT_FLOAT16);

   vector<int64_t> dim_c = {128, 2};
   GeShape shape_c(dim_c);
   GeTensorDesc tensor_desc_c(shape_c);
   tensor_desc_c.SetFormat(FORMAT_ND);
   tensor_desc_c.SetOriginFormat(FORMAT_ND);
   tensor_desc_c.SetDataType(DT_FLOAT16);
   tensor_desc_c.SetOriginDataType(DT_FLOAT16);


   vector<int64_t> dim_d = {128, 32};
   GeShape shape_d(dim_d);
   GeTensorDesc tensor_desc_d(shape_d);
   tensor_desc_d.SetFormat(FORMAT_ND);
   tensor_desc_d.SetOriginFormat(FORMAT_ND);
   tensor_desc_d.SetDataType(DT_FLOAT16);
   tensor_desc_d.SetOriginDataType(DT_FLOAT16);

   op_desc_switch1->AddOutputDesc(tensor_desc_a);

   op_desc_cast1->AddOutputDesc(tensor_desc_a);

   op_desc_cast1->AddInputDesc(tensor_desc_a);

   NodePtr node_switch1 = graph->AddNode(op_desc_switch1);

   NodePtr node_cast1 = graph->AddNode(op_desc_cast1);


   std::string splitNodeName = "SplitNode1_1";
   op::SplitV op(splitNodeName.c_str());
   op.BreakConnect();
   op.create_dynamic_output_y(2);
   op.set_attr_num_split(2);
   auto op_desc_split = ge::OpDescUtils::GetOpDescFromOperator(op);
   NodePtr SplitNode1_1 = graph->AddNode(op_desc_split);

   std::string constNodeName = "split_dim1_int32";
   op::Const split_dim1_int32(constNodeName.c_str());
   split_dim1_int32.BreakConnect();
   auto op_desc_split_dim1_int32 = ge::OpDescUtils::GetOpDescFromOperator(split_dim1_int32);

   GeTensorDesc td{GeShape(std::vector<int64_t>({1})), FORMAT_ND, DT_INT32};
   GeTensor tensor(td);
   // int32 下的 1 强转成小端 uint8
   tensor.SetData(std::vector<uint8_t>{1,0,0,0});
   AttrUtils::SetTensor(op_desc_split_dim1_int32, "value", tensor);
   NodePtr node_split_dim1_int32 = graph->AddNode(op_desc_split_dim1_int32);

   std::string splitNodeName1 = "SplitNode2_1";
   op::SplitV op1(splitNodeName1.c_str());
   op1.BreakConnect();
   op1.create_dynamic_output_y(4);
   op1.set_attr_num_split(4);
   auto op_desc_split1 = ge::OpDescUtils::GetOpDescFromOperator(op1);
   NodePtr SplitNode2_1 = graph->AddNode(op_desc_split1);

   std::string constNodeName1 = "split_dim1_int64";
   op::Const split_dim1_int64(constNodeName1.c_str());
   split_dim1_int64.BreakConnect();
   auto op_desc_split_dim1_int64 = ge::OpDescUtils::GetOpDescFromOperator(split_dim1_int64);

   GeTensorDesc td1{GeShape(std::vector<int64_t>({1})), FORMAT_ND, DT_INT64};
   GeTensor tensor1(td1);
   // int64 下的 1 强转成小端 uint8
   tensor1.SetData(std::vector<uint8_t>{1,0,0,0,0,0,0,0});
   AttrUtils::SetTensor(op_desc_split_dim1_int64, "value", tensor1);
   NodePtr node_split_dim1_int64 = graph->AddNode(op_desc_split_dim1_int64);

   std::string splitNodeName2 = "SplitNode2_2";
   op::SplitV op2(splitNodeName2.c_str());
   op2.BreakConnect();
   op2.create_dynamic_output_y(4);
   op2.set_attr_num_split(4);
   auto op_desc_split2 = ge::OpDescUtils::GetOpDescFromOperator(op2);
   NodePtr SplitNode2_2 = graph->AddNode(op_desc_split2);

   std::string constNodeName2 = "split_dim2_int64";
   op::Const split_dim2_int64(constNodeName2.c_str());
   split_dim1_int64.BreakConnect();
   auto op_desc_split_dim2_int64 = ge::OpDescUtils::GetOpDescFromOperator(split_dim2_int64);

   GeTensorDesc td2{GeShape(std::vector<int64_t>({1})), FORMAT_ND, DT_INT64};
   GeTensor tensor2(td2);
   // int64 下的 2 强转成小端 uint8
   tensor2.SetData(std::vector<uint8_t>{1,0,0,0,0,0,0,0});
   AttrUtils::SetTensor(op_desc_split_dim2_int64, "value", tensor2);
   NodePtr node_split_dim2_int64 = graph->AddNode(op_desc_split_dim2_int64);

   GraphUtils::AddEdge(node_switch1->GetOutDataAnchor(0), node_cast1->GetInDataAnchor(0));

   GraphUtils::AddEdge(node_cast1->GetOutDataAnchor(0), SplitNode1_1->GetInDataAnchor(0));
   GraphUtils::AddEdge(node_split_dim1_int32->GetOutDataAnchor(0), SplitNode1_1->GetInDataAnchor(0));

   GraphUtils::AddEdge(SplitNode1_1->GetOutDataAnchor(0), SplitNode2_1->GetInDataAnchor(0));
   GraphUtils::AddEdge(node_split_dim1_int64->GetOutDataAnchor(0), SplitNode2_1->GetInDataAnchor(0));
   GraphUtils::AddEdge(SplitNode1_1->GetOutDataAnchor(1), SplitNode2_2->GetInDataAnchor(0));
   GraphUtils::AddEdge(node_split_dim2_int64->GetOutDataAnchor(0), SplitNode2_2->GetInDataAnchor(0));

   SplitNode1_1->GetOpDesc()->UpdateInputDesc(0, node_switch1->GetOpDesc()->GetOutputDesc(0));
   SplitNode1_1->GetOpDesc()->UpdateOutputDesc(0, tensor_desc_b);
   SplitNode1_1->GetOpDesc()->UpdateOutputDesc(1, tensor_desc_b);
   SplitNode1_1->GetOpDesc()->MutableOutputDesc(0)->GetOrCreateAttrsGroup<ge::SymbolicDescAttr>()
       ->symbolic_tensor.MutableOriginSymbolShape() = {Symbol("s0"), Symbol("s1")};
  SplitNode1_1->GetOpDesc()->MutableOutputDesc(1)->GetOrCreateAttrsGroup<ge::SymbolicDescAttr>()
       ->symbolic_tensor.MutableOriginSymbolShape() = {Symbol("s0"), Symbol("s1")};

   SplitNode2_1->GetOpDesc()->UpdateInputDesc(0, SplitNode1_1->GetOpDesc()->GetOutputDesc(0));
   SplitNode2_1->GetOpDesc()->UpdateOutputDesc(0, tensor_desc_c);
   SplitNode2_1->GetOpDesc()->UpdateOutputDesc(1, tensor_desc_c);
   SplitNode2_1->GetOpDesc()->UpdateOutputDesc(2, tensor_desc_c);
   SplitNode2_1->GetOpDesc()->UpdateOutputDesc(3, tensor_desc_c);

   SplitNode2_2->GetOpDesc()->UpdateInputDesc(0, SplitNode1_1->GetOpDesc()->GetOutputDesc(1));
   SplitNode2_2->GetOpDesc()->UpdateOutputDesc(0, tensor_desc_c);
   SplitNode2_2->GetOpDesc()->UpdateOutputDesc(1, tensor_desc_c);
   SplitNode2_2->GetOpDesc()->UpdateOutputDesc(2, tensor_desc_c);
   SplitNode2_2->GetOpDesc()->UpdateOutputDesc(3, tensor_desc_c);
   graph->TopologicalSorting();
   FlattenSplitPass multiSplitFusionPro;
   auto result = multiSplitFusionPro.Run(graph);
   for (auto &node : graph->GetDirectNode()) {
     // if ((node->GetType() == SPLITV) || (node->GetType() == SPLITVD)) {
       printf("Graph_multiSplit_CountFusionOutAnchor:split_out_name is %s start;\n",
         node->GetName().c_str());
     // }
   }
 }

 TEST_F(PatternFusionBeforeAutoFuseUT, ConcatCombineProcess_env_not_enabled) {
   ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test1");
   OpDescPtr op_desc_switch1 = std::make_shared<OpDesc>("switch1", "switch");
   OpDescPtr op_desc_cast1 = std::make_shared<OpDesc>("cast1", "Relu");
   OpDescPtr op_desc_cast2 = std::make_shared<OpDesc>("cast2", "Relu");
   OpDescPtr op_desc_cast3 = std::make_shared<OpDesc>("cast3", "Relu");
   OpDescPtr op_desc_cast4 = std::make_shared<OpDesc>("cast4", "Relu");

   OpDescPtr op_desc_switch2 = std::make_shared<OpDesc>("switch2", "switch");
   OpDescPtr op_desc_switch3 = std::make_shared<OpDesc>("switch2", "switch");
   OpDescPtr op_desc_switch4 = std::make_shared<OpDesc>("switch3", "switch");

   OpDescPtr op_desc_concat1_0 = std::make_shared<OpDesc>("concat1_0", "ConcatV2");
   OpDescPtr op_desc_concat1_1 = std::make_shared<OpDesc>("concat1_1", "ConcatV2D");
   OpDescPtr op_desc_concat2_0 = std::make_shared<OpDesc>("concat2_0", "ConcatV2");
   OpDescPtr op_desc_concat3_0 = std::make_shared<OpDesc>("concat3_0", "ConcatV2D");

   vector<int64_t> dim_a = {128, 32};
   GeShape shape_a(dim_a);
   GeTensorDesc tensor_desc_a(shape_a);
   tensor_desc_a.SetFormat(FORMAT_ND);
   tensor_desc_a.SetOriginFormat(FORMAT_ND);
   tensor_desc_a.SetDataType(DT_FLOAT16);
   tensor_desc_a.SetOriginDataType(DT_FLOAT16);

   vector<int64_t> dim_b = {128, 16};
   GeShape shape_b(dim_b);
   GeTensorDesc tensor_desc_b(shape_b);
   tensor_desc_b.SetFormat(FORMAT_ND);
   tensor_desc_b.SetOriginFormat(FORMAT_ND);
   tensor_desc_a.SetDataType(DT_FLOAT16);
   tensor_desc_b.SetOriginDataType(DT_FLOAT16);

   vector<int64_t> dim_c = {128, 64};
   GeShape shape_c(dim_c);
   GeTensorDesc tensor_desc_c(shape_c);
   tensor_desc_c.SetFormat(FORMAT_ND);
   tensor_desc_c.SetOriginFormat(FORMAT_ND);
   tensor_desc_c.SetDataType(DT_FLOAT16);
   tensor_desc_c.SetOriginDataType(DT_FLOAT16);


   vector<int64_t> dim_d = {128, 32};
   GeShape shape_d(dim_d);
   GeTensorDesc tensor_desc_d(shape_d);
   tensor_desc_d.SetFormat(FORMAT_ND);
   tensor_desc_d.SetOriginFormat(FORMAT_ND);
   tensor_desc_d.SetDataType(DT_FLOAT16);
   tensor_desc_a.SetOriginDataType(DT_FLOAT16);

   op_desc_switch1->AddOutputDesc(tensor_desc_b);
   op_desc_switch2->AddOutputDesc(tensor_desc_b);
   op_desc_switch3->AddOutputDesc(tensor_desc_b);
   op_desc_switch4->AddOutputDesc(tensor_desc_b);

   op_desc_cast1->AddOutputDesc(tensor_desc_b);
   op_desc_cast2->AddOutputDesc(tensor_desc_b);
   op_desc_cast3->AddOutputDesc(tensor_desc_b);
   op_desc_cast4->AddOutputDesc(tensor_desc_b);

   op_desc_cast1->AddInputDesc(tensor_desc_b);
   op_desc_cast2->AddInputDesc(tensor_desc_b);
   op_desc_cast3->AddInputDesc(tensor_desc_b);
   op_desc_cast4->AddInputDesc(tensor_desc_b);

   NodePtr node_switch1 = graph->AddNode(op_desc_switch1);
   NodePtr node_switch2 = graph->AddNode(op_desc_switch2);
   NodePtr node_switch3 = graph->AddNode(op_desc_switch3);
   NodePtr node_switch4 = graph->AddNode(op_desc_switch4);

   NodePtr node_cast1 = graph->AddNode(op_desc_cast1);
   NodePtr node_cast2 = graph->AddNode(op_desc_cast2);
   NodePtr node_cast3 = graph->AddNode(op_desc_cast3);
   NodePtr node_cast4 = graph->AddNode(op_desc_cast4);

   std::string concatNodeName = "ConcatNode1_1";
   op::ConcatV2D op(concatNodeName.c_str());
   op.BreakConnect();
   op.create_dynamic_input_x(2);
   op.set_attr_concat_dim(1);
   op.set_attr_N(2);
   auto op_desc_concat = ge::OpDescUtils::GetOpDescFromOperator(op);
   NodePtr ConcatNode1_1 = graph->AddNode(op_desc_concat);

   std::string concatNodeName1 = "ConcatNode3_1";
   op::ConcatV2D op1(concatNodeName1.c_str());
   op1.BreakConnect();
   op1.create_dynamic_input_x(4);
   op1.set_attr_concat_dim(1);
   op1.set_attr_N(4);
   auto op_desc_concat1 = ge::OpDescUtils::GetOpDescFromOperator(op1);
   NodePtr ConcatNode3_1 = graph->AddNode(op_desc_concat1);

   std::string concatNodeName2 = "ConcatNode1_2";
   op::ConcatV2D op2(concatNodeName2.c_str());
   op2.BreakConnect();
   op2.create_dynamic_input_x(2);
   op2.set_attr_concat_dim(1);
   op2.set_attr_N(2);
   auto op_desc_concat2 = ge::OpDescUtils::GetOpDescFromOperator(op2);
   NodePtr ConcatNode1_2 = graph->AddNode(op_desc_concat2);

   std::string concatNodeName3 = "ConcatNode2_1";
   ge::op::ConcatV2D op3(concatNodeName3.c_str());
   op3.BreakConnect();
   op3.create_dynamic_input_x(3);
   op3.set_attr_concat_dim(1);
   op3.set_attr_N(3);
   auto op_desc_concat3 = ge::OpDescUtils::GetOpDescFromOperator(op3);
   NodePtr ConcatNode2_1 = graph->AddNode(op_desc_concat3);

   GraphUtils::AddEdge(node_switch1->GetOutDataAnchor(0), node_cast1->GetInDataAnchor(0));
   GraphUtils::AddEdge(node_switch2->GetOutDataAnchor(0), node_cast2->GetInDataAnchor(0));
   GraphUtils::AddEdge(node_switch3->GetOutDataAnchor(0), node_cast3->GetInDataAnchor(0));
   GraphUtils::AddEdge(node_switch4->GetOutDataAnchor(0), node_cast4->GetInDataAnchor(0));

   GraphUtils::AddEdge(node_switch2->GetOutDataAnchor(0), ConcatNode1_1->GetInDataAnchor(0));
   GraphUtils::AddEdge(node_switch3->GetOutDataAnchor(0), ConcatNode1_1->GetInDataAnchor(1));

   GraphUtils::AddEdge(node_switch3->GetOutDataAnchor(0), ConcatNode1_2->GetInDataAnchor(0));
   GraphUtils::AddEdge(node_switch4->GetOutDataAnchor(0), ConcatNode1_2->GetInDataAnchor(1));

   GraphUtils::AddEdge(node_cast1->GetOutDataAnchor(0), ConcatNode2_1->GetInDataAnchor(0));
   GraphUtils::AddEdge(ConcatNode1_1->GetOutDataAnchor(0), ConcatNode2_1->GetInDataAnchor(1));
   GraphUtils::AddEdge(node_cast2->GetOutDataAnchor(0), ConcatNode2_1->GetInDataAnchor(2));

   GraphUtils::AddEdge(ConcatNode2_1->GetOutDataAnchor(0), ConcatNode3_1->GetInDataAnchor(0));
   GraphUtils::AddEdge(node_cast3->GetOutDataAnchor(0), ConcatNode3_1->GetInDataAnchor(1));
   GraphUtils::AddEdge(ConcatNode1_2->GetOutDataAnchor(0), ConcatNode3_1->GetInDataAnchor(2));
   GraphUtils::AddEdge(node_cast4->GetOutDataAnchor(0), ConcatNode3_1->GetInDataAnchor(3));

   ConcatNode1_1->GetOpDesc()->UpdateInputDesc(0, node_switch1->GetOpDesc()->GetOutputDesc(0));
   ConcatNode1_1->GetOpDesc()->UpdateInputDesc(1, node_switch2->GetOpDesc()->GetOutputDesc(0));
   ConcatNode1_1->GetOpDesc()->UpdateOutputDesc(0, tensor_desc_a);

   ConcatNode1_2->GetOpDesc()->UpdateInputDesc(0, node_switch3->GetOpDesc()->GetOutputDesc(0));
   ConcatNode1_2->GetOpDesc()->UpdateInputDesc(1, node_switch4->GetOpDesc()->GetOutputDesc(0));
   ConcatNode1_2->GetOpDesc()->UpdateOutputDesc(0, tensor_desc_a);


   ConcatNode2_1->GetOpDesc()->UpdateInputDesc(0, node_cast1->GetOpDesc()->GetOutputDesc(0));
   ConcatNode2_1->GetOpDesc()->UpdateInputDesc(1, ConcatNode1_1->GetOpDesc()->GetOutputDesc(0));
   ConcatNode2_1->GetOpDesc()->UpdateInputDesc(2, node_cast2->GetOpDesc()->GetOutputDesc(0));
   ConcatNode2_1->GetOpDesc()->UpdateOutputDesc(0, tensor_desc_c);
   ConcatNode2_1->GetOpDesc()->MutableOutputDesc(0)->GetOrCreateAttrsGroup<ge::SymbolicDescAttr>()
       ->symbolic_tensor.MutableOriginSymbolShape() = {Symbol("s0"), Symbol("s1")};


   ConcatNode3_1->GetOpDesc()->UpdateInputDesc(0, ConcatNode2_1->GetOpDesc()->GetOutputDesc(0));
   ConcatNode3_1->GetOpDesc()->UpdateInputDesc(1, node_cast3->GetOpDesc()->GetOutputDesc(0));
   ConcatNode3_1->GetOpDesc()->UpdateInputDesc(2, ConcatNode1_2->GetOpDesc()->GetOutputDesc(0));
   ConcatNode3_1->GetOpDesc()->UpdateInputDesc(3, node_cast4->GetOpDesc()->GetOutputDesc(0));
   ConcatNode3_1->GetOpDesc()->UpdateOutputDesc(0, tensor_desc_d);
   ConcatNode3_1->GetOpDesc()->MutableOutputDesc(0)->GetOrCreateAttrsGroup<ge::SymbolicDescAttr>()
       ->symbolic_tensor.MutableOriginSymbolShape() = {Symbol("s0"), Symbol("s1")};
   graph->TopologicalSorting();
   FlattenConcatPass multiConcatFusionPro;
   EXPECT_EQ(multiConcatFusionPro.Run(graph), ge::GRAPH_SUCCESS);
 }
 
 TEST_F(PatternFusionBeforeAutoFuseUT, ConcatCombineProcess_2) {
   ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test1");
   OpDescPtr op_desc_switch1 = std::make_shared<OpDesc>("switch1", "switch");
   OpDescPtr op_desc_cast1 = std::make_shared<OpDesc>("cast1", "Relu");
   OpDescPtr op_desc_cast2 = std::make_shared<OpDesc>("cast2", "Relu");
   OpDescPtr op_desc_cast3 = std::make_shared<OpDesc>("cast3", "Relu");
   OpDescPtr op_desc_cast4 = std::make_shared<OpDesc>("cast4", "Relu");
 
   OpDescPtr op_desc_switch2 = std::make_shared<OpDesc>("switch2", "switch");
   OpDescPtr op_desc_switch3 = std::make_shared<OpDesc>("switch2", "switch");
   OpDescPtr op_desc_switch4 = std::make_shared<OpDesc>("switch3", "switch");
 
   OpDescPtr op_desc_concat1_0 = std::make_shared<OpDesc>("concat1_0", "ConcatV2");
   OpDescPtr op_desc_concat1_1 = std::make_shared<OpDesc>("concat1_1", "ConcatV2D");
   OpDescPtr op_desc_concat2_0 = std::make_shared<OpDesc>("concat2_0", "ConcatV2");
   OpDescPtr op_desc_concat3_0 = std::make_shared<OpDesc>("concat3_0", "ConcatV2D");
 
   vector<int64_t> dim_a = {128, 32};
   GeShape shape_a(dim_a);
   GeTensorDesc tensor_desc_a(shape_a);
   tensor_desc_a.SetFormat(FORMAT_ND);
   tensor_desc_a.SetOriginFormat(FORMAT_ND);
   tensor_desc_a.SetDataType(DT_FLOAT16);
   tensor_desc_a.SetOriginDataType(DT_FLOAT16);
 
   vector<int64_t> dim_b = {128, 16};
   GeShape shape_b(dim_b);
   GeTensorDesc tensor_desc_b(shape_b);
   tensor_desc_b.SetFormat(FORMAT_ND);
   tensor_desc_b.SetOriginFormat(FORMAT_ND);
   tensor_desc_a.SetDataType(DT_FLOAT16);
   tensor_desc_b.SetOriginDataType(DT_FLOAT16);
 
   vector<int64_t> dim_c = {128, 64};
   GeShape shape_c(dim_c);
   GeTensorDesc tensor_desc_c(shape_c);
   tensor_desc_c.SetFormat(FORMAT_ND);
   tensor_desc_c.SetOriginFormat(FORMAT_ND);
   tensor_desc_c.SetDataType(DT_FLOAT16);
   tensor_desc_c.SetOriginDataType(DT_FLOAT16);
 
 
   vector<int64_t> dim_d = {128, 160};
   GeShape shape_d(dim_d);
   GeTensorDesc tensor_desc_d(shape_d);
   tensor_desc_d.SetFormat(FORMAT_ND);
   tensor_desc_d.SetOriginFormat(FORMAT_ND);
   tensor_desc_d.SetDataType(DT_FLOAT16);
   tensor_desc_d.SetOriginDataType(DT_FLOAT16);
 
   op_desc_switch1->AddOutputDesc(tensor_desc_b);
   op_desc_switch2->AddOutputDesc(tensor_desc_b);
   op_desc_switch3->AddOutputDesc(tensor_desc_b);
   op_desc_switch4->AddOutputDesc(tensor_desc_b);
 
   op_desc_cast1->AddOutputDesc(tensor_desc_b);
   op_desc_cast2->AddOutputDesc(tensor_desc_b);
   op_desc_cast3->AddOutputDesc(tensor_desc_b);
   op_desc_cast4->AddOutputDesc(tensor_desc_b);
 
   op_desc_cast1->AddInputDesc(tensor_desc_b);
   op_desc_cast2->AddInputDesc(tensor_desc_b);
   op_desc_cast3->AddInputDesc(tensor_desc_b);
   op_desc_cast4->AddInputDesc(tensor_desc_b);
 
   NodePtr node_switch1 = graph->AddNode(op_desc_switch1);
   NodePtr node_switch2 = graph->AddNode(op_desc_switch2);
   NodePtr node_switch3 = graph->AddNode(op_desc_switch3);
   NodePtr node_switch4 = graph->AddNode(op_desc_switch4);
 
   NodePtr node_cast1 = graph->AddNode(op_desc_cast1);
   NodePtr node_cast2 = graph->AddNode(op_desc_cast2);
   NodePtr node_cast3 = graph->AddNode(op_desc_cast3);
   NodePtr node_cast4 = graph->AddNode(op_desc_cast4);
 
   std::string concatNodeName = "ConcatNode1_1";
   op::ConcatV2D op(concatNodeName.c_str());
   op.BreakConnect();
   op.create_dynamic_input_x(2);
   op.set_attr_concat_dim(-1);  
   op.set_attr_N(2);
   auto op_desc_concat = ge::OpDescUtils::GetOpDescFromOperator(op);
   NodePtr ConcatNode1_1 = graph->AddNode(op_desc_concat);
 
   /*vector<int64_t> dim_e = {-1};
   GeShape shape_e(dim_e);
   GeTensorDesc tensor_desc_e(shape_e);
   tensor_desc_e.SetFormat(FORMAT_ND);
   tensor_desc_e.SetOriginFormat(FORMAT_ND);
   tensor_desc_e.SetDataType(DT_INT32);
   tensor_desc_e.SetOriginDataType(DT_INT32);
   ConcatNode1_1->GetOpDesc()->AddInputDesc("concat_dim", tensor_desc_e);*/
 
 
   std::string concatNodeName1 = "ConcatNode3_1";
   op::ConcatV2D op1(concatNodeName1.c_str());
   op1.BreakConnect();
   op1.create_dynamic_input_x(5);
   op1.set_attr_concat_dim(1);
   op1.set_attr_N(5);
   auto op_desc_concat1 = ge::OpDescUtils::GetOpDescFromOperator(op1);
   NodePtr ConcatNode3_1 = graph->AddNode(op_desc_concat1);
 
   std::string concatNodeName2 = "ConcatNode1_2";
   op::ConcatV2D op2(concatNodeName2.c_str());
   op2.BreakConnect();
   op2.create_dynamic_input_x(2);
   op2.set_attr_concat_dim(1);
   op2.set_attr_N(2);
   auto op_desc_concat2 = ge::OpDescUtils::GetOpDescFromOperator(op2);
   NodePtr ConcatNode1_2 = graph->AddNode(op_desc_concat2);
 
   std::string concatNodeName3 = "ConcatNode2_1";
   ge::op::ConcatV2D op3(concatNodeName3.c_str());
   op3.BreakConnect();
   op3.create_dynamic_input_x(3);
   op3.set_attr_concat_dim(1);
   op3.set_attr_N(3);
   auto op_desc_concat3 = ge::OpDescUtils::GetOpDescFromOperator(op3);
   NodePtr ConcatNode2_1 = graph->AddNode(op_desc_concat3);
 
   GraphUtils::AddEdge(node_switch1->GetOutDataAnchor(0), node_cast1->GetInDataAnchor(0));
   GraphUtils::AddEdge(node_switch2->GetOutDataAnchor(0), node_cast2->GetInDataAnchor(0));
   GraphUtils::AddEdge(node_switch3->GetOutDataAnchor(0), node_cast3->GetInDataAnchor(0));
   GraphUtils::AddEdge(node_switch4->GetOutDataAnchor(0), node_cast4->GetInDataAnchor(0));
 
   GraphUtils::AddEdge(node_switch2->GetOutDataAnchor(0), ConcatNode1_1->GetInDataAnchor(0));
   GraphUtils::AddEdge(node_switch3->GetOutDataAnchor(0), ConcatNode1_1->GetInDataAnchor(1));
 
   GraphUtils::AddEdge(node_switch3->GetOutDataAnchor(0), ConcatNode1_2->GetInDataAnchor(0));
   GraphUtils::AddEdge(node_switch4->GetOutDataAnchor(0), ConcatNode1_2->GetInDataAnchor(1));
 
   GraphUtils::AddEdge(node_cast1->GetOutDataAnchor(0), ConcatNode2_1->GetInDataAnchor(0));
   GraphUtils::AddEdge(ConcatNode1_1->GetOutDataAnchor(0), ConcatNode2_1->GetInDataAnchor(1));
   GraphUtils::AddEdge(node_cast2->GetOutDataAnchor(0), ConcatNode2_1->GetInDataAnchor(2));
 
   GraphUtils::AddEdge(ConcatNode2_1->GetOutDataAnchor(0), ConcatNode3_1->GetInDataAnchor(0));
   GraphUtils::AddEdge(node_cast3->GetOutDataAnchor(0), ConcatNode3_1->GetInDataAnchor(1));
   GraphUtils::AddEdge(ConcatNode1_2->GetOutDataAnchor(0), ConcatNode3_1->GetInDataAnchor(2));
   GraphUtils::AddEdge(node_cast4->GetOutDataAnchor(0), ConcatNode3_1->GetInDataAnchor(3));
   GraphUtils::AddEdge(ConcatNode1_1->GetOutDataAnchor(0), ConcatNode3_1->GetInDataAnchor(4));
 
   ConcatNode1_1->GetOpDesc()->UpdateInputDesc(0, node_switch1->GetOpDesc()->GetOutputDesc(0));
   ConcatNode1_1->GetOpDesc()->UpdateInputDesc(1, node_switch2->GetOpDesc()->GetOutputDesc(0));
   ConcatNode1_1->GetOpDesc()->UpdateOutputDesc(0, tensor_desc_a);
 
   ConcatNode1_2->GetOpDesc()->UpdateInputDesc(0, node_switch3->GetOpDesc()->GetOutputDesc(0));
   ConcatNode1_2->GetOpDesc()->UpdateInputDesc(1, node_switch4->GetOpDesc()->GetOutputDesc(0));
   ConcatNode1_2->GetOpDesc()->UpdateOutputDesc(0, tensor_desc_a);
 
 
   ConcatNode2_1->GetOpDesc()->UpdateInputDesc(0, node_cast1->GetOpDesc()->GetOutputDesc(0));
   ConcatNode2_1->GetOpDesc()->UpdateInputDesc(1, ConcatNode1_1->GetOpDesc()->GetOutputDesc(0));
   ConcatNode2_1->GetOpDesc()->UpdateInputDesc(2, node_cast2->GetOpDesc()->GetOutputDesc(0));
   ConcatNode2_1->GetOpDesc()->UpdateOutputDesc(0, tensor_desc_c);
   ConcatNode2_1->GetOpDesc()->MutableOutputDesc(0)->GetOrCreateAttrsGroup<ge::SymbolicDescAttr>()
       ->symbolic_tensor.MutableOriginSymbolShape() = {Symbol("s0"), Symbol("s1")};
 
 
   ConcatNode3_1->GetOpDesc()->UpdateInputDesc(0, ConcatNode2_1->GetOpDesc()->GetOutputDesc(0));
   ConcatNode3_1->GetOpDesc()->UpdateInputDesc(1, node_cast3->GetOpDesc()->GetOutputDesc(0));
   ConcatNode3_1->GetOpDesc()->UpdateInputDesc(2, ConcatNode1_2->GetOpDesc()->GetOutputDesc(0));
   ConcatNode3_1->GetOpDesc()->UpdateInputDesc(3, node_cast4->GetOpDesc()->GetOutputDesc(0));
   ConcatNode3_1->GetOpDesc()->UpdateInputDesc(4, ConcatNode1_1->GetOpDesc()->GetOutputDesc(0));
   ConcatNode3_1->GetOpDesc()->UpdateOutputDesc(0, tensor_desc_d);
   ConcatNode3_1->GetOpDesc()->MutableOutputDesc(0)->GetOrCreateAttrsGroup<ge::SymbolicDescAttr>()
       ->symbolic_tensor.MutableOriginSymbolShape() = {Symbol("s0"), Symbol("s1")};
   graph->TopologicalSorting();
   FlattenConcatPass multiConcatFusionPro;
   auto result = multiConcatFusionPro.Run(graph);
 
   for (auto &node : graph->GetDirectNode()) {
     //if ((node->GetType() == CONCATV2) || (node->GetType() == CONCATV2D)) {
       printf("Graph_multisliceConcat_CountFusionOutAnchor:split_out_name is %s start;\n",
         node->GetName().c_str());
       
     //}
   }
 }

 TEST_F(PatternFusionBeforeAutoFuseUT, ConcatCombineProcess_3) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test1");
  OpDescPtr op_desc_switch1 = std::make_shared<OpDesc>("switch1", "switch");
  OpDescPtr op_desc_cast1 = std::make_shared<OpDesc>("cast1", "Relu");
  OpDescPtr op_desc_cast2 = std::make_shared<OpDesc>("cast2", "Relu");
  OpDescPtr op_desc_cast3 = std::make_shared<OpDesc>("cast3", "Relu");
  OpDescPtr op_desc_cast4 = std::make_shared<OpDesc>("cast4", "Relu");

  OpDescPtr op_desc_switch2 = std::make_shared<OpDesc>("switch2", "switch");
  OpDescPtr op_desc_switch3 = std::make_shared<OpDesc>("switch2", "switch");
  OpDescPtr op_desc_switch4 = std::make_shared<OpDesc>("switch3", "switch");

  OpDescPtr op_desc_concat1_0 = std::make_shared<OpDesc>("concat1_0", "ConcatV2");
  OpDescPtr op_desc_concat1_1 = std::make_shared<OpDesc>("concat1_1", "ConcatV2D");
  OpDescPtr op_desc_concat2_0 = std::make_shared<OpDesc>("concat2_0", "ConcatV2");
  OpDescPtr op_desc_concat3_0 = std::make_shared<OpDesc>("concat3_0", "ConcatV2D");

  vector<int64_t> dim_a = {128, 32};
  GeShape shape_a(dim_a);
  GeTensorDesc tensor_desc_a(shape_a);
  tensor_desc_a.SetFormat(FORMAT_ND);
  tensor_desc_a.SetOriginFormat(FORMAT_ND);
  tensor_desc_a.SetDataType(DT_FLOAT16);
  tensor_desc_a.SetOriginDataType(DT_FLOAT16);

  vector<int64_t> dim_b = {128, 16};
  GeShape shape_b(dim_b);
  GeTensorDesc tensor_desc_b(shape_b);
  tensor_desc_b.SetFormat(FORMAT_ND);
  tensor_desc_b.SetOriginFormat(FORMAT_ND);
  tensor_desc_a.SetDataType(DT_FLOAT16);
  tensor_desc_b.SetOriginDataType(DT_FLOAT16);

  vector<int64_t> dim_c = {128, 64};
  GeShape shape_c(dim_c);
  GeTensorDesc tensor_desc_c(shape_c);
  tensor_desc_c.SetFormat(FORMAT_ND);
  tensor_desc_c.SetOriginFormat(FORMAT_ND);
  tensor_desc_c.SetDataType(DT_FLOAT16);
  tensor_desc_c.SetOriginDataType(DT_FLOAT16);


  vector<int64_t> dim_d = {128, 160};
  GeShape shape_d(dim_d);
  GeTensorDesc tensor_desc_d(shape_d);
  tensor_desc_d.SetFormat(FORMAT_ND);
  tensor_desc_d.SetOriginFormat(FORMAT_ND);
  tensor_desc_d.SetDataType(DT_FLOAT16);
  tensor_desc_d.SetOriginDataType(DT_FLOAT16);

  op_desc_switch1->AddOutputDesc(tensor_desc_b);
  op_desc_switch2->AddOutputDesc(tensor_desc_b);
  op_desc_switch3->AddOutputDesc(tensor_desc_b);
  op_desc_switch4->AddOutputDesc(tensor_desc_b);

  op_desc_cast1->AddOutputDesc(tensor_desc_b);
  op_desc_cast2->AddOutputDesc(tensor_desc_b);
  op_desc_cast3->AddOutputDesc(tensor_desc_b);
  op_desc_cast4->AddOutputDesc(tensor_desc_b);

  op_desc_cast1->AddInputDesc(tensor_desc_b);
  op_desc_cast2->AddInputDesc(tensor_desc_b);
  op_desc_cast3->AddInputDesc(tensor_desc_b);
  op_desc_cast4->AddInputDesc(tensor_desc_b);

  NodePtr node_switch1 = graph->AddNode(op_desc_switch1);
  NodePtr node_switch2 = graph->AddNode(op_desc_switch2);
  NodePtr node_switch3 = graph->AddNode(op_desc_switch3);
  NodePtr node_switch4 = graph->AddNode(op_desc_switch4);

  NodePtr node_cast1 = graph->AddNode(op_desc_cast1);
  NodePtr node_cast2 = graph->AddNode(op_desc_cast2);
  NodePtr node_cast3 = graph->AddNode(op_desc_cast3);
  NodePtr node_cast4 = graph->AddNode(op_desc_cast4);

  std::string concatNodeName = "ConcatNode1_1";
  op::ConcatV2D op(concatNodeName.c_str());
  op.BreakConnect();
  op.create_dynamic_input_x(2);
  op.set_attr_concat_dim(-1);  
  op.set_attr_N(2);
  auto op_desc_concat = ge::OpDescUtils::GetOpDescFromOperator(op);
  NodePtr ConcatNode1_1 = graph->AddNode(op_desc_concat);

  /*vector<int64_t> dim_e = {-1};
  GeShape shape_e(dim_e);
  GeTensorDesc tensor_desc_e(shape_e);
  tensor_desc_e.SetFormat(FORMAT_ND);
  tensor_desc_e.SetOriginFormat(FORMAT_ND);
  tensor_desc_e.SetDataType(DT_INT32);
  tensor_desc_e.SetOriginDataType(DT_INT32);
  ConcatNode1_1->GetOpDesc()->AddInputDesc("concat_dim", tensor_desc_e);*/


  std::string concatNodeName1 = "ConcatNode3_1";
  op::ConcatV2D op1(concatNodeName1.c_str());
  op1.BreakConnect();
  op1.create_dynamic_input_x(5);
  op1.set_attr_concat_dim(1);
  op1.set_attr_N(5);
  auto op_desc_concat1 = ge::OpDescUtils::GetOpDescFromOperator(op1);
  NodePtr ConcatNode3_1 = graph->AddNode(op_desc_concat1);

  std::string concatNodeName2 = "ConcatNode1_2";
  op::ConcatV2D op2(concatNodeName2.c_str());
  op2.BreakConnect();
  op2.create_dynamic_input_x(2);
  op2.set_attr_concat_dim(1);
  op2.set_attr_N(2);
  auto op_desc_concat2 = ge::OpDescUtils::GetOpDescFromOperator(op2);
  NodePtr ConcatNode1_2 = graph->AddNode(op_desc_concat2);

  std::string concatNodeName3 = "ConcatNode2_1";
  ge::op::ConcatV2D op3(concatNodeName3.c_str());
  op3.BreakConnect();
  op3.create_dynamic_input_x(3);
  op3.set_attr_concat_dim(1);
  op3.set_attr_N(3);
  auto op_desc_concat3 = ge::OpDescUtils::GetOpDescFromOperator(op3);
  NodePtr ConcatNode2_1 = graph->AddNode(op_desc_concat3);

  GraphUtils::AddEdge(node_switch1->GetOutDataAnchor(0), node_cast1->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_switch2->GetOutDataAnchor(0), node_cast2->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_switch3->GetOutDataAnchor(0), node_cast3->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_switch4->GetOutDataAnchor(0), node_cast4->GetInDataAnchor(0));

  GraphUtils::AddEdge(node_switch2->GetOutDataAnchor(0), ConcatNode1_1->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_switch3->GetOutDataAnchor(0), ConcatNode1_1->GetInDataAnchor(1));

  GraphUtils::AddEdge(node_switch3->GetOutDataAnchor(0), ConcatNode1_2->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_switch4->GetOutDataAnchor(0), ConcatNode1_2->GetInDataAnchor(1));

  GraphUtils::AddEdge(node_cast1->GetOutDataAnchor(0), ConcatNode2_1->GetInDataAnchor(0));
  GraphUtils::AddEdge(ConcatNode1_1->GetOutDataAnchor(0), ConcatNode2_1->GetInDataAnchor(1));
  GraphUtils::AddEdge(node_cast2->GetOutDataAnchor(0), ConcatNode2_1->GetInDataAnchor(2));

  GraphUtils::AddEdge(ConcatNode2_1->GetOutDataAnchor(0), ConcatNode3_1->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_cast3->GetOutDataAnchor(0), ConcatNode3_1->GetInDataAnchor(1));
  GraphUtils::AddEdge(ConcatNode1_2->GetOutDataAnchor(0), ConcatNode3_1->GetInDataAnchor(2));
  GraphUtils::AddEdge(node_cast4->GetOutDataAnchor(0), ConcatNode3_1->GetInDataAnchor(3));
  GraphUtils::AddEdge(ConcatNode1_1->GetOutDataAnchor(0), ConcatNode3_1->GetInDataAnchor(4));

  ConcatNode1_1->GetOpDesc()->UpdateInputDesc(0, node_switch1->GetOpDesc()->GetOutputDesc(0));
  ConcatNode1_1->GetOpDesc()->UpdateInputDesc(1, node_switch2->GetOpDesc()->GetOutputDesc(0));
  ConcatNode1_1->GetOpDesc()->UpdateOutputDesc(0, tensor_desc_a);

  ConcatNode1_2->GetOpDesc()->UpdateInputDesc(0, node_switch3->GetOpDesc()->GetOutputDesc(0));
  ConcatNode1_2->GetOpDesc()->UpdateInputDesc(1, node_switch4->GetOpDesc()->GetOutputDesc(0));
  ConcatNode1_2->GetOpDesc()->UpdateOutputDesc(0, tensor_desc_a);


  ConcatNode2_1->GetOpDesc()->UpdateInputDesc(0, node_cast1->GetOpDesc()->GetOutputDesc(0));
  ConcatNode2_1->GetOpDesc()->UpdateInputDesc(1, ConcatNode1_1->GetOpDesc()->GetOutputDesc(0));
  ConcatNode2_1->GetOpDesc()->UpdateInputDesc(2, node_cast2->GetOpDesc()->GetOutputDesc(0));
  ConcatNode2_1->GetOpDesc()->UpdateOutputDesc(0, tensor_desc_c);
  ConcatNode2_1->GetOpDesc()->MutableOutputDesc(0)->GetOrCreateAttrsGroup<ge::SymbolicDescAttr>()
      ->symbolic_tensor.MutableOriginSymbolShape() = {Symbol("s0"), Symbol("s1")};
  GraphUtils::AddEdge(node_switch1->GetOutControlAnchor(), ConcatNode2_1->GetInControlAnchor());


  ConcatNode3_1->GetOpDesc()->UpdateInputDesc(0, ConcatNode2_1->GetOpDesc()->GetOutputDesc(0));
  ConcatNode3_1->GetOpDesc()->UpdateInputDesc(1, node_cast3->GetOpDesc()->GetOutputDesc(0));
  ConcatNode3_1->GetOpDesc()->UpdateInputDesc(2, ConcatNode1_2->GetOpDesc()->GetOutputDesc(0));
  ConcatNode3_1->GetOpDesc()->UpdateInputDesc(3, node_cast4->GetOpDesc()->GetOutputDesc(0));
  ConcatNode3_1->GetOpDesc()->UpdateInputDesc(4, ConcatNode1_1->GetOpDesc()->GetOutputDesc(0));
  ConcatNode3_1->GetOpDesc()->UpdateOutputDesc(0, tensor_desc_d);
  ConcatNode3_1->GetOpDesc()->MutableOutputDesc(0)->GetOrCreateAttrsGroup<ge::SymbolicDescAttr>()
      ->symbolic_tensor.MutableOriginSymbolShape() = {Symbol("s0"), Symbol("s1")};
  graph->TopologicalSorting();

  PatternFusion multiconcatFusionpattern;
  auto result = multiconcatFusionpattern.RunAllPatternFusion(graph);

  for (auto &node : graph->GetDirectNode()) {
    //if ((node->GetType() == CONCATV2) || (node->GetType() == CONCATV2D)) {
      printf("Graph_multisliceConcat_CountFusionOutAnchor:split_out_name is %s start;\n",
        node->GetName().c_str());
      
    //}
  }
}

REG_OP(ConcatV2)
    .DYNAMIC_INPUT(x, TensorType::BasicType())
    .INPUT(concat_dim, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::BasicType())
    .ATTR(N, Int, 1)
    .OP_END_FACTORY_REG(ConcatV2)
REG_OP(Constant)
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16,
        DT_UINT8, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .ATTR(value, Tensor, Tensor())
    .OP_END_FACTORY_REG(Constant)

TEST_F(PatternFusionBeforeAutoFuseUT, ConcatCombineProcess_4) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test1");
  OpDescPtr op_desc_switch1 = std::make_shared<OpDesc>("switch1", "switch");
  OpDescPtr op_desc_cast1 = std::make_shared<OpDesc>("cast1", "Relu");
  OpDescPtr op_desc_cast2 = std::make_shared<OpDesc>("cast2", "Relu");
  OpDescPtr op_desc_cast3 = std::make_shared<OpDesc>("cast3", "Relu");
  OpDescPtr op_desc_cast4 = std::make_shared<OpDesc>("cast4", "Relu");

  OpDescPtr op_desc_switch2 = std::make_shared<OpDesc>("switch2", "switch");
  OpDescPtr op_desc_switch3 = std::make_shared<OpDesc>("switch2", "switch");
  OpDescPtr op_desc_switch4 = std::make_shared<OpDesc>("switch3", "switch");

  OpDescPtr op_desc_concat1_0 = std::make_shared<OpDesc>("concat1_0", "ConcatV2");
  OpDescPtr op_desc_concat1_1 = std::make_shared<OpDesc>("concat1_1", "ConcatV2D");
  OpDescPtr op_desc_concat2_0 = std::make_shared<OpDesc>("concat2_0", "ConcatV2");
  OpDescPtr op_desc_concat3_0 = std::make_shared<OpDesc>("concat3_0", "ConcatV2D");

  OpDescPtr op_desc_constant = std::make_shared<OpDesc>("constant1", "Constant");

  vector<int64_t> dim_a = {128, 32};
  GeShape shape_a(dim_a);
  GeTensorDesc tensor_desc_a(shape_a);
  tensor_desc_a.SetFormat(FORMAT_ND);
  tensor_desc_a.SetOriginFormat(FORMAT_ND);
  tensor_desc_a.SetDataType(DT_FLOAT16);
  tensor_desc_a.SetOriginDataType(DT_FLOAT16);

  vector<int64_t> dim_b = {128, 16};
  GeShape shape_b(dim_b);
  GeTensorDesc tensor_desc_b(shape_b);
  tensor_desc_b.SetFormat(FORMAT_ND);
  tensor_desc_b.SetOriginFormat(FORMAT_ND);
  tensor_desc_a.SetDataType(DT_FLOAT16);
  tensor_desc_b.SetOriginDataType(DT_FLOAT16);

  vector<int64_t> dim_c = {128, 64};
  GeShape shape_c(dim_c);
  GeTensorDesc tensor_desc_c(shape_c);
  tensor_desc_c.SetFormat(FORMAT_ND);
  tensor_desc_c.SetOriginFormat(FORMAT_ND);
  tensor_desc_c.SetDataType(DT_FLOAT16);
  tensor_desc_c.SetOriginDataType(DT_FLOAT16);


  vector<int64_t> dim_d = {128, 160};
  GeShape shape_d(dim_d);
  GeTensorDesc tensor_desc_d(shape_d);
  tensor_desc_d.SetFormat(FORMAT_ND);
  tensor_desc_d.SetOriginFormat(FORMAT_ND);
  tensor_desc_d.SetDataType(DT_FLOAT16);
  tensor_desc_d.SetOriginDataType(DT_FLOAT16);

  op_desc_switch1->AddOutputDesc(tensor_desc_b);
  op_desc_switch2->AddOutputDesc(tensor_desc_b);
  op_desc_switch3->AddOutputDesc(tensor_desc_b);
  op_desc_switch4->AddOutputDesc(tensor_desc_b);

  op_desc_cast1->AddOutputDesc(tensor_desc_b);
  op_desc_cast2->AddOutputDesc(tensor_desc_b);
  op_desc_cast3->AddOutputDesc(tensor_desc_b);
  op_desc_cast4->AddOutputDesc(tensor_desc_b);

  op_desc_cast1->AddInputDesc(tensor_desc_b);
  op_desc_cast2->AddInputDesc(tensor_desc_b);
  op_desc_cast3->AddInputDesc(tensor_desc_b);
  op_desc_cast4->AddInputDesc(tensor_desc_b);

  NodePtr node_switch1 = graph->AddNode(op_desc_switch1);
  NodePtr node_switch2 = graph->AddNode(op_desc_switch2);
  NodePtr node_switch3 = graph->AddNode(op_desc_switch3);
  NodePtr node_switch4 = graph->AddNode(op_desc_switch4);

  NodePtr node_cast1 = graph->AddNode(op_desc_cast1);
  NodePtr node_cast2 = graph->AddNode(op_desc_cast2);
  NodePtr node_cast3 = graph->AddNode(op_desc_cast3);
  NodePtr node_cast4 = graph->AddNode(op_desc_cast4);

  std::string concatNodeName = "ConcatNode1_1";
  op::ConcatV2 op(concatNodeName.c_str());
  op.BreakConnect();
  op.create_dynamic_input_x(2);
  //op.set_attr_concat_dim(-1);  
  op.set_attr_N(2);
  auto op_desc_concat = ge::OpDescUtils::GetOpDescFromOperator(op);
  NodePtr ConcatNode1_1 = graph->AddNode(op_desc_concat);

  //OpDescPtr op_desc_constant = std::make_shared<OpDesc>("constant1", "Constant");
  /*vector<int64_t> dim_e = {1};
  GeShape shape_e(dim_e);
  GeTensorDesc tensor_desc_e(shape_e);
  tensor_desc_e.SetFormat(FORMAT_ND);
  tensor_desc_e.SetOriginFormat(FORMAT_ND);
  tensor_desc_e.SetDataType(DT_INT32);
  tensor_desc_e.SetOriginDataType(DT_INT32);
  op_desc_constant->AddInputDesc(tensor_desc_e);
  NodePtr node_constant1 = graph->AddNode(op_desc_constant);*/
  //int count = 1;
  //std::unique_ptr<int32_t[]> addr(ComGraphMakeUnique<int32_t[]>(count));
  //size_t i = 0U;
  //int dimvalue = 1;
  //for (auto &shape : merged_multi_dims_) {
  //  for (int64_t dim : shape) {
      //addr[i] = static_cast<int32_t>(dimvalue);
  //  }
  //}

  //GeTensorDesc const_tensor(GeShape({1}), FORMAT_ND, DT_INT32);
  //GeTensor tensor(const_tensor);
  //(void)tensor.SetData(reinterpret_cast<uint8_t *>(addr.get()), 1 * sizeof(int32_t));
  //AttrUtils::SetTensor(node_constant1->GetOpDesc(), ATTR_NAME_WEIGHTS, tensor);

  std::string constNodeName = "constant1";
  op::Constant opconst(constNodeName.c_str());
  opconst.BreakConnect();
  auto op_desc_constantNew = ge::OpDescUtils::GetOpDescFromOperator(opconst);

  int count = 1;
  std::unique_ptr<int32_t[]> addr(ComGraphMakeUnique<int32_t[]>(count));
  size_t i = 0U;
  int dimvalue = 1;

  addr[i] = static_cast<int32_t>(dimvalue);
  GeTensorDesc const_tensor(GeShape({1}), FORMAT_ND, DT_INT32);
  GeTensor tensor(const_tensor);
  (void)tensor.SetData(reinterpret_cast<uint8_t *>(addr.get()), 1 * sizeof(int32_t));
  AttrUtils::SetTensor(op_desc_constantNew, "value", tensor);
  NodePtr node_constant1 = graph->AddNode(op_desc_constantNew);


  std::string concatNodeName1 = "ConcatNode3_1";
  op::ConcatV2D op1(concatNodeName1.c_str());
  op1.BreakConnect();
  op1.create_dynamic_input_x(5);
  op1.set_attr_concat_dim(1);
  op1.set_attr_N(5);
  auto op_desc_concat1 = ge::OpDescUtils::GetOpDescFromOperator(op1);
  NodePtr ConcatNode3_1 = graph->AddNode(op_desc_concat1);

  std::string concatNodeName2 = "ConcatNode1_2";
  op::ConcatV2D op2(concatNodeName2.c_str());
  op2.BreakConnect();
  op2.create_dynamic_input_x(2);
  op2.set_attr_concat_dim(1);
  op2.set_attr_N(2);
  auto op_desc_concat2 = ge::OpDescUtils::GetOpDescFromOperator(op2);
  NodePtr ConcatNode1_2 = graph->AddNode(op_desc_concat2);

  std::string concatNodeName3 = "ConcatNode2_1";
  ge::op::ConcatV2D op3(concatNodeName3.c_str());
  op3.BreakConnect();
  op3.create_dynamic_input_x(3);
  op3.set_attr_concat_dim(1);
  op3.set_attr_N(3);
  auto op_desc_concat3 = ge::OpDescUtils::GetOpDescFromOperator(op3);
  NodePtr ConcatNode2_1 = graph->AddNode(op_desc_concat3);

  GraphUtils::AddEdge(node_switch1->GetOutDataAnchor(0), node_cast1->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_switch2->GetOutDataAnchor(0), node_cast2->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_switch3->GetOutDataAnchor(0), node_cast3->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_switch4->GetOutDataAnchor(0), node_cast4->GetInDataAnchor(0));

  GraphUtils::AddEdge(node_switch2->GetOutDataAnchor(0), ConcatNode1_1->GetInDataAnchor(1));
  GraphUtils::AddEdge(node_switch3->GetOutDataAnchor(0), ConcatNode1_1->GetInDataAnchor(2));
  GraphUtils::AddEdge(node_constant1->GetOutDataAnchor(0), ConcatNode1_1->GetInDataAnchor(0));

  GraphUtils::AddEdge(node_switch3->GetOutDataAnchor(0), ConcatNode1_2->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_switch4->GetOutDataAnchor(0), ConcatNode1_2->GetInDataAnchor(1));

  GraphUtils::AddEdge(node_cast1->GetOutDataAnchor(0), ConcatNode2_1->GetInDataAnchor(0));
  GraphUtils::AddEdge(ConcatNode1_1->GetOutDataAnchor(0), ConcatNode2_1->GetInDataAnchor(1));
  GraphUtils::AddEdge(node_cast2->GetOutDataAnchor(0), ConcatNode2_1->GetInDataAnchor(2));

  GraphUtils::AddEdge(ConcatNode2_1->GetOutDataAnchor(0), ConcatNode3_1->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_cast3->GetOutDataAnchor(0), ConcatNode3_1->GetInDataAnchor(1));
  GraphUtils::AddEdge(ConcatNode1_2->GetOutDataAnchor(0), ConcatNode3_1->GetInDataAnchor(2));
  GraphUtils::AddEdge(node_cast4->GetOutDataAnchor(0), ConcatNode3_1->GetInDataAnchor(3));
  GraphUtils::AddEdge(ConcatNode1_1->GetOutDataAnchor(0), ConcatNode3_1->GetInDataAnchor(4));

  ConcatNode1_1->GetOpDesc()->UpdateInputDesc(0, node_switch1->GetOpDesc()->GetOutputDesc(0));
  ConcatNode1_1->GetOpDesc()->UpdateInputDesc(1, node_switch2->GetOpDesc()->GetOutputDesc(0));
  ConcatNode1_1->GetOpDesc()->UpdateOutputDesc(0, tensor_desc_a);

  ConcatNode1_2->GetOpDesc()->UpdateInputDesc(0, node_switch3->GetOpDesc()->GetOutputDesc(0));
  ConcatNode1_2->GetOpDesc()->UpdateInputDesc(1, node_switch4->GetOpDesc()->GetOutputDesc(0));
  ConcatNode1_2->GetOpDesc()->UpdateOutputDesc(0, tensor_desc_a);


  ConcatNode2_1->GetOpDesc()->UpdateInputDesc(0, node_cast1->GetOpDesc()->GetOutputDesc(0));
  ConcatNode2_1->GetOpDesc()->UpdateInputDesc(1, ConcatNode1_1->GetOpDesc()->GetOutputDesc(0));
  ConcatNode2_1->GetOpDesc()->UpdateInputDesc(2, node_cast2->GetOpDesc()->GetOutputDesc(0));
  ConcatNode2_1->GetOpDesc()->UpdateOutputDesc(0, tensor_desc_c);
  ConcatNode2_1->GetOpDesc()->MutableOutputDesc(0)->GetOrCreateAttrsGroup<ge::SymbolicDescAttr>()
      ->symbolic_tensor.MutableOriginSymbolShape() = {Symbol("s0"), Symbol("s1")};
  GraphUtils::AddEdge(node_switch1->GetOutControlAnchor(), ConcatNode2_1->GetInControlAnchor());


  ConcatNode3_1->GetOpDesc()->UpdateInputDesc(0, ConcatNode2_1->GetOpDesc()->GetOutputDesc(0));
  ConcatNode3_1->GetOpDesc()->UpdateInputDesc(1, node_cast3->GetOpDesc()->GetOutputDesc(0));
  ConcatNode3_1->GetOpDesc()->UpdateInputDesc(2, ConcatNode1_2->GetOpDesc()->GetOutputDesc(0));
  ConcatNode3_1->GetOpDesc()->UpdateInputDesc(3, node_cast4->GetOpDesc()->GetOutputDesc(0));
  ConcatNode3_1->GetOpDesc()->UpdateInputDesc(4, ConcatNode1_1->GetOpDesc()->GetOutputDesc(0));
  ConcatNode3_1->GetOpDesc()->UpdateOutputDesc(0, tensor_desc_d);
  ConcatNode3_1->GetOpDesc()->MutableOutputDesc(0)->GetOrCreateAttrsGroup<ge::SymbolicDescAttr>()
      ->symbolic_tensor.MutableOriginSymbolShape() = {Symbol("s0"), Symbol("s1")};
  graph->TopologicalSorting();
  FlattenConcatPass multiConcatFusionPro;
  auto result = multiConcatFusionPro.Run(graph);

  for (auto &node : graph->GetDirectNode()) {
    //if ((node->GetType() == CONCATV2) || (node->GetType() == CONCATV2D)) {
      printf("Graph_multisliceConcat_CountFusionOutAnchor:split_out_name is %s start;\n",
        node->GetName().c_str());
      
    //}
  }
}
TEST_F(PatternFusionBeforeAutoFuseUT, ConcatCombineProcess_5) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test1");
  OpDescPtr op_desc_switch1 = std::make_shared<OpDesc>("switch1", "switch");
  OpDescPtr op_desc_cast1 = std::make_shared<OpDesc>("cast1", "Relu");
  OpDescPtr op_desc_cast2 = std::make_shared<OpDesc>("cast2", "Relu");
  OpDescPtr op_desc_cast3 = std::make_shared<OpDesc>("cast3", "Relu");
  OpDescPtr op_desc_cast4 = std::make_shared<OpDesc>("cast4", "Relu");

  OpDescPtr op_desc_switch2 = std::make_shared<OpDesc>("switch2", "switch");
  OpDescPtr op_desc_switch3 = std::make_shared<OpDesc>("switch2", "switch");
  OpDescPtr op_desc_switch4 = std::make_shared<OpDesc>("switch3", "switch");

  OpDescPtr op_desc_concat1_0 = std::make_shared<OpDesc>("concat1_0", "ConcatV2");
  OpDescPtr op_desc_concat1_1 = std::make_shared<OpDesc>("concat1_1", "ConcatV2D");
  OpDescPtr op_desc_concat2_0 = std::make_shared<OpDesc>("concat2_0", "ConcatV2");
  OpDescPtr op_desc_concat3_0 = std::make_shared<OpDesc>("concat3_0", "ConcatV2D");

  OpDescPtr op_desc_constant = std::make_shared<OpDesc>("constant1", "Constant");

  vector<int64_t> dim_a = {128, 32};
  GeShape shape_a(dim_a);
  GeTensorDesc tensor_desc_a(shape_a);
  tensor_desc_a.SetFormat(FORMAT_ND);
  tensor_desc_a.SetOriginFormat(FORMAT_ND);
  tensor_desc_a.SetDataType(DT_FLOAT16);
  tensor_desc_a.SetOriginDataType(DT_FLOAT16);

  vector<int64_t> dim_b = {128, 16};
  GeShape shape_b(dim_b);
  GeTensorDesc tensor_desc_b(shape_b);
  tensor_desc_b.SetFormat(FORMAT_ND);
  tensor_desc_b.SetOriginFormat(FORMAT_ND);
  tensor_desc_a.SetDataType(DT_FLOAT16);
  tensor_desc_b.SetOriginDataType(DT_FLOAT16);

  vector<int64_t> dim_c = {128, 64};
  GeShape shape_c(dim_c);
  GeTensorDesc tensor_desc_c(shape_c);
  tensor_desc_c.SetFormat(FORMAT_ND);
  tensor_desc_c.SetOriginFormat(FORMAT_ND);
  tensor_desc_c.SetDataType(DT_FLOAT16);
  tensor_desc_c.SetOriginDataType(DT_FLOAT16);


  vector<int64_t> dim_d = {128, 160};
  GeShape shape_d(dim_d);
  GeTensorDesc tensor_desc_d(shape_d);
  tensor_desc_d.SetFormat(FORMAT_ND);
  tensor_desc_d.SetOriginFormat(FORMAT_ND);
  tensor_desc_d.SetDataType(DT_FLOAT16);
  tensor_desc_d.SetOriginDataType(DT_FLOAT16);

  op_desc_switch1->AddOutputDesc(tensor_desc_b);
  op_desc_switch2->AddOutputDesc(tensor_desc_b);
  op_desc_switch3->AddOutputDesc(tensor_desc_b);
  op_desc_switch4->AddOutputDesc(tensor_desc_b);

  op_desc_cast1->AddOutputDesc(tensor_desc_b);
  op_desc_cast2->AddOutputDesc(tensor_desc_b);
  op_desc_cast3->AddOutputDesc(tensor_desc_b);
  op_desc_cast4->AddOutputDesc(tensor_desc_b);

  op_desc_cast1->AddInputDesc(tensor_desc_b);
  op_desc_cast2->AddInputDesc(tensor_desc_b);
  op_desc_cast3->AddInputDesc(tensor_desc_b);
  op_desc_cast4->AddInputDesc(tensor_desc_b);

  NodePtr node_switch1 = graph->AddNode(op_desc_switch1);
  NodePtr node_switch2 = graph->AddNode(op_desc_switch2);
  NodePtr node_switch3 = graph->AddNode(op_desc_switch3);
  NodePtr node_switch4 = graph->AddNode(op_desc_switch4);

  NodePtr node_cast1 = graph->AddNode(op_desc_cast1);
  NodePtr node_cast2 = graph->AddNode(op_desc_cast2);
  NodePtr node_cast3 = graph->AddNode(op_desc_cast3);
  NodePtr node_cast4 = graph->AddNode(op_desc_cast4);

  std::string concatNodeName = "ConcatNode1_1";
  op::ConcatV2 op(concatNodeName.c_str());
  op.BreakConnect();
  op.create_dynamic_input_x(2);
  //op.set_attr_concat_dim(-1);  
  op.set_attr_N(2);
  auto op_desc_concat = ge::OpDescUtils::GetOpDescFromOperator(op);
  NodePtr ConcatNode1_1 = graph->AddNode(op_desc_concat);

  //OpDescPtr op_desc_constant = std::make_shared<OpDesc>("constant1", "Constant");
  /*vector<int64_t> dim_e = {1};
  GeShape shape_e(dim_e);
  GeTensorDesc tensor_desc_e(shape_e);
  tensor_desc_e.SetFormat(FORMAT_ND);
  tensor_desc_e.SetOriginFormat(FORMAT_ND);
  tensor_desc_e.SetDataType(DT_INT32);
  tensor_desc_e.SetOriginDataType(DT_INT32);
  op_desc_constant->AddInputDesc(tensor_desc_e);
  NodePtr node_constant1 = graph->AddNode(op_desc_constant);*/
  //int count = 1;
  //std::unique_ptr<int32_t[]> addr(ComGraphMakeUnique<int32_t[]>(count));
  //size_t i = 0U;
  //int dimvalue = 1;
  //for (auto &shape : merged_multi_dims_) {
  //  for (int64_t dim : shape) {
      //addr[i] = static_cast<int32_t>(dimvalue);
  //  }
  //}

  //GeTensorDesc const_tensor(GeShape({1}), FORMAT_ND, DT_INT32);
  //GeTensor tensor(const_tensor);
  //(void)tensor.SetData(reinterpret_cast<uint8_t *>(addr.get()), 1 * sizeof(int32_t));
  //AttrUtils::SetTensor(node_constant1->GetOpDesc(), ATTR_NAME_WEIGHTS, tensor);

  std::string constNodeName = "constant1";
  op::Constant opconst(constNodeName.c_str());
  opconst.BreakConnect();
  auto op_desc_constantNew = ge::OpDescUtils::GetOpDescFromOperator(opconst);

  int count = 1;
  std::unique_ptr<int32_t[]> addr(ComGraphMakeUnique<int32_t[]>(count));
  size_t i = 0U;
  int dimvalue = 1;

  addr[i] = static_cast<int32_t>(dimvalue);
  GeTensorDesc const_tensor(GeShape({1}), FORMAT_ND, DT_INT64);
  GeTensor tensor(const_tensor);
  (void)tensor.SetData(reinterpret_cast<uint8_t *>(addr.get()), 1 * sizeof(int32_t));
  AttrUtils::SetTensor(op_desc_constantNew, "value", tensor);
  NodePtr node_constant1 = graph->AddNode(op_desc_constantNew);


  std::string concatNodeName1 = "ConcatNode3_1";
  op::ConcatV2D op1(concatNodeName1.c_str());
  op1.BreakConnect();
  op1.create_dynamic_input_x(5);
  op1.set_attr_concat_dim(1);
  op1.set_attr_N(5);
  auto op_desc_concat1 = ge::OpDescUtils::GetOpDescFromOperator(op1);
  NodePtr ConcatNode3_1 = graph->AddNode(op_desc_concat1);

  std::string concatNodeName2 = "ConcatNode1_2";
  op::ConcatV2D op2(concatNodeName2.c_str());
  op2.BreakConnect();
  op2.create_dynamic_input_x(2);
  op2.set_attr_concat_dim(1);
  op2.set_attr_N(2);
  auto op_desc_concat2 = ge::OpDescUtils::GetOpDescFromOperator(op2);
  NodePtr ConcatNode1_2 = graph->AddNode(op_desc_concat2);

  std::string concatNodeName3 = "ConcatNode2_1";
  ge::op::ConcatV2D op3(concatNodeName3.c_str());
  op3.BreakConnect();
  op3.create_dynamic_input_x(3);
  op3.set_attr_concat_dim(1);
  op3.set_attr_N(3);
  auto op_desc_concat3 = ge::OpDescUtils::GetOpDescFromOperator(op3);
  NodePtr ConcatNode2_1 = graph->AddNode(op_desc_concat3);

  GraphUtils::AddEdge(node_switch1->GetOutDataAnchor(0), node_cast1->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_switch2->GetOutDataAnchor(0), node_cast2->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_switch3->GetOutDataAnchor(0), node_cast3->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_switch4->GetOutDataAnchor(0), node_cast4->GetInDataAnchor(0));

  GraphUtils::AddEdge(node_switch2->GetOutDataAnchor(0), ConcatNode1_1->GetInDataAnchor(1));
  GraphUtils::AddEdge(node_switch3->GetOutDataAnchor(0), ConcatNode1_1->GetInDataAnchor(2));
  GraphUtils::AddEdge(node_constant1->GetOutDataAnchor(0), ConcatNode1_1->GetInDataAnchor(0));

  GraphUtils::AddEdge(node_switch3->GetOutDataAnchor(0), ConcatNode1_2->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_switch4->GetOutDataAnchor(0), ConcatNode1_2->GetInDataAnchor(1));

  GraphUtils::AddEdge(node_cast1->GetOutDataAnchor(0), ConcatNode2_1->GetInDataAnchor(0));
  GraphUtils::AddEdge(ConcatNode1_1->GetOutDataAnchor(0), ConcatNode2_1->GetInDataAnchor(1));
  GraphUtils::AddEdge(node_cast2->GetOutDataAnchor(0), ConcatNode2_1->GetInDataAnchor(2));

  GraphUtils::AddEdge(ConcatNode2_1->GetOutDataAnchor(0), ConcatNode3_1->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_cast3->GetOutDataAnchor(0), ConcatNode3_1->GetInDataAnchor(1));
  GraphUtils::AddEdge(ConcatNode1_2->GetOutDataAnchor(0), ConcatNode3_1->GetInDataAnchor(2));
  GraphUtils::AddEdge(node_cast4->GetOutDataAnchor(0), ConcatNode3_1->GetInDataAnchor(3));
  GraphUtils::AddEdge(ConcatNode1_1->GetOutDataAnchor(0), ConcatNode3_1->GetInDataAnchor(4));

  ConcatNode1_1->GetOpDesc()->UpdateInputDesc(0, node_switch1->GetOpDesc()->GetOutputDesc(0));
  ConcatNode1_1->GetOpDesc()->UpdateInputDesc(1, node_switch2->GetOpDesc()->GetOutputDesc(0));
  ConcatNode1_1->GetOpDesc()->UpdateOutputDesc(0, tensor_desc_a);

  ConcatNode1_2->GetOpDesc()->UpdateInputDesc(0, node_switch3->GetOpDesc()->GetOutputDesc(0));
  ConcatNode1_2->GetOpDesc()->UpdateInputDesc(1, node_switch4->GetOpDesc()->GetOutputDesc(0));
  ConcatNode1_2->GetOpDesc()->UpdateOutputDesc(0, tensor_desc_a);


  ConcatNode2_1->GetOpDesc()->UpdateInputDesc(0, node_cast1->GetOpDesc()->GetOutputDesc(0));
  ConcatNode2_1->GetOpDesc()->UpdateInputDesc(1, ConcatNode1_1->GetOpDesc()->GetOutputDesc(0));
  ConcatNode2_1->GetOpDesc()->UpdateInputDesc(2, node_cast2->GetOpDesc()->GetOutputDesc(0));
  ConcatNode2_1->GetOpDesc()->UpdateOutputDesc(0, tensor_desc_c);
  ConcatNode2_1->GetOpDesc()->MutableOutputDesc(0)->GetOrCreateAttrsGroup<ge::SymbolicDescAttr>()
      ->symbolic_tensor.MutableOriginSymbolShape() = {Symbol("s0"), Symbol("s1")};
  GraphUtils::AddEdge(node_switch1->GetOutControlAnchor(), ConcatNode2_1->GetInControlAnchor());


  ConcatNode3_1->GetOpDesc()->UpdateInputDesc(0, ConcatNode2_1->GetOpDesc()->GetOutputDesc(0));
  ConcatNode3_1->GetOpDesc()->UpdateInputDesc(1, node_cast3->GetOpDesc()->GetOutputDesc(0));
  ConcatNode3_1->GetOpDesc()->UpdateInputDesc(2, ConcatNode1_2->GetOpDesc()->GetOutputDesc(0));
  ConcatNode3_1->GetOpDesc()->UpdateInputDesc(3, node_cast4->GetOpDesc()->GetOutputDesc(0));
  ConcatNode3_1->GetOpDesc()->UpdateInputDesc(4, ConcatNode1_1->GetOpDesc()->GetOutputDesc(0));
  ConcatNode3_1->GetOpDesc()->UpdateOutputDesc(0, tensor_desc_d);
  ConcatNode2_1->GetOpDesc()->MutableOutputDesc(0)->GetOrCreateAttrsGroup<ge::SymbolicDescAttr>()
      ->symbolic_tensor.MutableOriginSymbolShape() = {Symbol("s0"), Symbol("s1")};
  graph->TopologicalSorting();
  FlattenConcatPass multiConcatFusionPro;
  auto result = multiConcatFusionPro.Run(graph);

  for (auto &node : graph->GetDirectNode()) {
    //if ((node->GetType() == CONCATV2) || (node->GetType() == CONCATV2D)) {
      printf("Graph_multisliceConcat_CountFusionOutAnchor:split_out_name is %s start;\n",
        node->GetName().c_str());
      
    //}
  }
}

TEST_F(PatternFusionBeforeAutoFuseUT, ConcatCombineProcess_6) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test1");
  OpDescPtr op_desc_switch1 = std::make_shared<OpDesc>("switch1", "switch");
  OpDescPtr op_desc_cast1 = std::make_shared<OpDesc>("cast1", "Relu");
  OpDescPtr op_desc_cast2 = std::make_shared<OpDesc>("cast2", "Relu");
  OpDescPtr op_desc_cast3 = std::make_shared<OpDesc>("cast3", "Relu");
  OpDescPtr op_desc_cast4 = std::make_shared<OpDesc>("cast4", "Relu");

  OpDescPtr op_desc_switch2 = std::make_shared<OpDesc>("switch2", "switch");
  OpDescPtr op_desc_switch3 = std::make_shared<OpDesc>("switch2", "switch");
  OpDescPtr op_desc_switch4 = std::make_shared<OpDesc>("switch3", "switch");

  OpDescPtr op_desc_concat1_0 = std::make_shared<OpDesc>("concat1_0", "ConcatV2");
  OpDescPtr op_desc_concat1_1 = std::make_shared<OpDesc>("concat1_1", "ConcatV2D");
  OpDescPtr op_desc_concat2_0 = std::make_shared<OpDesc>("concat2_0", "ConcatV2");
  OpDescPtr op_desc_concat3_0 = std::make_shared<OpDesc>("concat3_0", "ConcatV2D");

  OpDescPtr op_desc_constant = std::make_shared<OpDesc>("constant1", "Constant");

  vector<int64_t> dim_a = {128, 32};
  GeShape shape_a(dim_a);
  GeTensorDesc tensor_desc_a(shape_a);
  tensor_desc_a.SetFormat(FORMAT_ND);
  tensor_desc_a.SetOriginFormat(FORMAT_ND);
  tensor_desc_a.SetDataType(DT_FLOAT16);
  tensor_desc_a.SetOriginDataType(DT_FLOAT16);

  vector<int64_t> dim_b = {128, 16};
  GeShape shape_b(dim_b);
  GeTensorDesc tensor_desc_b(shape_b);
  tensor_desc_b.SetFormat(FORMAT_ND);
  tensor_desc_b.SetOriginFormat(FORMAT_ND);
  tensor_desc_a.SetDataType(DT_FLOAT16);
  tensor_desc_b.SetOriginDataType(DT_FLOAT16);

  vector<int64_t> dim_c = {128, 64};
  GeShape shape_c(dim_c);
  GeTensorDesc tensor_desc_c(shape_c);
  tensor_desc_c.SetFormat(FORMAT_ND);
  tensor_desc_c.SetOriginFormat(FORMAT_ND);
  tensor_desc_c.SetDataType(DT_FLOAT16);
  tensor_desc_c.SetOriginDataType(DT_FLOAT16);


  vector<int64_t> dim_d = {128, 160};
  GeShape shape_d(dim_d);
  GeTensorDesc tensor_desc_d(shape_d);
  tensor_desc_d.SetFormat(FORMAT_ND);
  tensor_desc_d.SetOriginFormat(FORMAT_ND);
  tensor_desc_d.SetDataType(DT_FLOAT16);
  tensor_desc_d.SetOriginDataType(DT_FLOAT16);

  op_desc_switch1->AddOutputDesc(tensor_desc_b);
  op_desc_switch2->AddOutputDesc(tensor_desc_b);
  op_desc_switch3->AddOutputDesc(tensor_desc_b);
  op_desc_switch4->AddOutputDesc(tensor_desc_b);

  op_desc_cast1->AddOutputDesc(tensor_desc_b);
  op_desc_cast2->AddOutputDesc(tensor_desc_b);
  op_desc_cast3->AddOutputDesc(tensor_desc_b);
  op_desc_cast4->AddOutputDesc(tensor_desc_b);

  op_desc_cast1->AddInputDesc(tensor_desc_b);
  op_desc_cast2->AddInputDesc(tensor_desc_b);
  op_desc_cast3->AddInputDesc(tensor_desc_b);
  op_desc_cast4->AddInputDesc(tensor_desc_b);

  NodePtr node_switch1 = graph->AddNode(op_desc_switch1);
  NodePtr node_switch2 = graph->AddNode(op_desc_switch2);
  NodePtr node_switch3 = graph->AddNode(op_desc_switch3);
  NodePtr node_switch4 = graph->AddNode(op_desc_switch4);

  NodePtr node_cast1 = graph->AddNode(op_desc_cast1);
  NodePtr node_cast2 = graph->AddNode(op_desc_cast2);
  NodePtr node_cast3 = graph->AddNode(op_desc_cast3);
  NodePtr node_cast4 = graph->AddNode(op_desc_cast4);

  std::string concatNodeName = "ConcatNode1_1";
  op::ConcatV2 op(concatNodeName.c_str());
  op.BreakConnect();
  op.create_dynamic_input_x(2);
  //op.set_attr_concat_dim(-1);  
  op.set_attr_N(2);
  auto op_desc_concat = ge::OpDescUtils::GetOpDescFromOperator(op);
  NodePtr ConcatNode1_1 = graph->AddNode(op_desc_concat);

  //OpDescPtr op_desc_constant = std::make_shared<OpDesc>("constant1", "Constant");
  /*vector<int64_t> dim_e = {1};
  GeShape shape_e(dim_e);
  GeTensorDesc tensor_desc_e(shape_e);
  tensor_desc_e.SetFormat(FORMAT_ND);
  tensor_desc_e.SetOriginFormat(FORMAT_ND);
  tensor_desc_e.SetDataType(DT_INT32);
  tensor_desc_e.SetOriginDataType(DT_INT32);
  op_desc_constant->AddInputDesc(tensor_desc_e);
  NodePtr node_constant1 = graph->AddNode(op_desc_constant);*/
  //int count = 1;
  //std::unique_ptr<int32_t[]> addr(ComGraphMakeUnique<int32_t[]>(count));
  //size_t i = 0U;
  //int dimvalue = 1;
  //for (auto &shape : merged_multi_dims_) {
  //  for (int64_t dim : shape) {
      //addr[i] = static_cast<int32_t>(dimvalue);
  //  }
  //}

  //GeTensorDesc const_tensor(GeShape({1}), FORMAT_ND, DT_INT32);
  //GeTensor tensor(const_tensor);
  //(void)tensor.SetData(reinterpret_cast<uint8_t *>(addr.get()), 1 * sizeof(int32_t));
  //AttrUtils::SetTensor(node_constant1->GetOpDesc(), ATTR_NAME_WEIGHTS, tensor);

  std::string constNodeName = "constant1";
  op::Constant opconst(constNodeName.c_str());
  opconst.BreakConnect();
  auto op_desc_constantNew = ge::OpDescUtils::GetOpDescFromOperator(opconst);

  int count = 1;
  std::unique_ptr<int32_t[]> addr(ComGraphMakeUnique<int32_t[]>(count));
  size_t i = 0U;
  int dimvalue = 1;

  addr[i] = static_cast<int32_t>(dimvalue);
  GeTensorDesc const_tensor(GeShape({1}), FORMAT_ND, DT_FLOAT16);
  GeTensor tensor(const_tensor);
  (void)tensor.SetData(reinterpret_cast<uint8_t *>(addr.get()), 1 * sizeof(int32_t));
  AttrUtils::SetTensor(op_desc_constantNew, "value", tensor);
  NodePtr node_constant1 = graph->AddNode(op_desc_constantNew);


  std::string concatNodeName1 = "ConcatNode3_1";
  op::ConcatV2D op1(concatNodeName1.c_str());
  op1.BreakConnect();
  op1.create_dynamic_input_x(5);
  op1.set_attr_concat_dim(1);
  op1.set_attr_N(5);
  auto op_desc_concat1 = ge::OpDescUtils::GetOpDescFromOperator(op1);
  NodePtr ConcatNode3_1 = graph->AddNode(op_desc_concat1);

  std::string concatNodeName2 = "ConcatNode1_2";
  op::ConcatV2D op2(concatNodeName2.c_str());
  op2.BreakConnect();
  op2.create_dynamic_input_x(2);
  op2.set_attr_concat_dim(1);
  op2.set_attr_N(2);
  auto op_desc_concat2 = ge::OpDescUtils::GetOpDescFromOperator(op2);
  NodePtr ConcatNode1_2 = graph->AddNode(op_desc_concat2);

  std::string concatNodeName3 = "ConcatNode2_1";
  ge::op::ConcatV2D op3(concatNodeName3.c_str());
  op3.BreakConnect();
  op3.create_dynamic_input_x(3);
  op3.set_attr_concat_dim(1);
  op3.set_attr_N(3);
  auto op_desc_concat3 = ge::OpDescUtils::GetOpDescFromOperator(op3);
  NodePtr ConcatNode2_1 = graph->AddNode(op_desc_concat3);

  GraphUtils::AddEdge(node_switch1->GetOutDataAnchor(0), node_cast1->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_switch2->GetOutDataAnchor(0), node_cast2->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_switch3->GetOutDataAnchor(0), node_cast3->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_switch4->GetOutDataAnchor(0), node_cast4->GetInDataAnchor(0));

  GraphUtils::AddEdge(node_switch2->GetOutDataAnchor(0), ConcatNode1_1->GetInDataAnchor(1));
  GraphUtils::AddEdge(node_switch3->GetOutDataAnchor(0), ConcatNode1_1->GetInDataAnchor(2));
  GraphUtils::AddEdge(node_constant1->GetOutDataAnchor(0), ConcatNode1_1->GetInDataAnchor(0));

  GraphUtils::AddEdge(node_switch3->GetOutDataAnchor(0), ConcatNode1_2->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_switch4->GetOutDataAnchor(0), ConcatNode1_2->GetInDataAnchor(1));

  GraphUtils::AddEdge(node_cast1->GetOutDataAnchor(0), ConcatNode2_1->GetInDataAnchor(0));
  GraphUtils::AddEdge(ConcatNode1_1->GetOutDataAnchor(0), ConcatNode2_1->GetInDataAnchor(1));
  GraphUtils::AddEdge(node_cast2->GetOutDataAnchor(0), ConcatNode2_1->GetInDataAnchor(2));

  GraphUtils::AddEdge(ConcatNode2_1->GetOutDataAnchor(0), ConcatNode3_1->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_cast3->GetOutDataAnchor(0), ConcatNode3_1->GetInDataAnchor(1));
  GraphUtils::AddEdge(ConcatNode1_2->GetOutDataAnchor(0), ConcatNode3_1->GetInDataAnchor(2));
  GraphUtils::AddEdge(node_cast4->GetOutDataAnchor(0), ConcatNode3_1->GetInDataAnchor(3));
  GraphUtils::AddEdge(ConcatNode1_1->GetOutDataAnchor(0), ConcatNode3_1->GetInDataAnchor(4));

  ConcatNode1_1->GetOpDesc()->UpdateInputDesc(0, node_switch1->GetOpDesc()->GetOutputDesc(0));
  ConcatNode1_1->GetOpDesc()->UpdateInputDesc(1, node_switch2->GetOpDesc()->GetOutputDesc(0));
  ConcatNode1_1->GetOpDesc()->UpdateOutputDesc(0, tensor_desc_a);

  ConcatNode1_2->GetOpDesc()->UpdateInputDesc(0, node_switch3->GetOpDesc()->GetOutputDesc(0));
  ConcatNode1_2->GetOpDesc()->UpdateInputDesc(1, node_switch4->GetOpDesc()->GetOutputDesc(0));
  ConcatNode1_2->GetOpDesc()->UpdateOutputDesc(0, tensor_desc_a);


  ConcatNode2_1->GetOpDesc()->UpdateInputDesc(0, node_cast1->GetOpDesc()->GetOutputDesc(0));
  ConcatNode2_1->GetOpDesc()->UpdateInputDesc(1, ConcatNode1_1->GetOpDesc()->GetOutputDesc(0));
  ConcatNode2_1->GetOpDesc()->UpdateInputDesc(2, node_cast2->GetOpDesc()->GetOutputDesc(0));
  ConcatNode2_1->GetOpDesc()->UpdateOutputDesc(0, tensor_desc_c);
  ConcatNode2_1->GetOpDesc()->MutableOutputDesc(0)->GetOrCreateAttrsGroup<ge::SymbolicDescAttr>()
      ->symbolic_tensor.MutableOriginSymbolShape() = {Symbol("s0"), Symbol("s1")};
  GraphUtils::AddEdge(node_switch1->GetOutControlAnchor(), ConcatNode2_1->GetInControlAnchor());


  ConcatNode3_1->GetOpDesc()->UpdateInputDesc(0, ConcatNode2_1->GetOpDesc()->GetOutputDesc(0));
  ConcatNode3_1->GetOpDesc()->UpdateInputDesc(1, node_cast3->GetOpDesc()->GetOutputDesc(0));
  ConcatNode3_1->GetOpDesc()->UpdateInputDesc(2, ConcatNode1_2->GetOpDesc()->GetOutputDesc(0));
  ConcatNode3_1->GetOpDesc()->UpdateInputDesc(3, node_cast4->GetOpDesc()->GetOutputDesc(0));
  ConcatNode3_1->GetOpDesc()->UpdateInputDesc(4, ConcatNode1_1->GetOpDesc()->GetOutputDesc(0));
  ConcatNode3_1->GetOpDesc()->UpdateOutputDesc(0, tensor_desc_d);
  ConcatNode3_1->GetOpDesc()->MutableOutputDesc(0)->GetOrCreateAttrsGroup<ge::SymbolicDescAttr>()
      ->symbolic_tensor.MutableOriginSymbolShape() = {Symbol("s0"), Symbol("s1")};
  graph->TopologicalSorting();
  FlattenConcatPass multiConcatFusionPro;
  auto result = multiConcatFusionPro.Run(graph);

  for (auto &node : graph->GetDirectNode()) {
    //if ((node->GetType() == CONCATV2) || (node->GetType() == CONCATV2D)) {
      printf("Graph_multisliceConcat_CountFusionOutAnchor:split_out_name is %s start;\n",
        node->GetName().c_str());
      
    //}
  }
}

TEST_F(PatternFusionBeforeAutoFuseUT, ConcatCombineProcess_7) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test1");
  OpDescPtr op_desc_switch1 = std::make_shared<OpDesc>("switch1", "switch");
  OpDescPtr op_desc_switch2 = std::make_shared<OpDesc>("switch2", "switch"); 
  OpDescPtr op_desc_constant = std::make_shared<OpDesc>("constant1", "Constant");

  vector<int64_t> dim_a = {-2};
  GeShape shape_a(dim_a);
  GeTensorDesc tensor_desc_a(shape_a);
  tensor_desc_a.SetFormat(FORMAT_ND);
  tensor_desc_a.SetOriginFormat(FORMAT_ND);
  tensor_desc_a.SetDataType(DT_FLOAT16);
  tensor_desc_a.SetOriginDataType(DT_FLOAT16);

  vector<int64_t> dim_b = {128, 16};
  GeShape shape_b(dim_b);
  GeTensorDesc tensor_desc_b(shape_b);
  tensor_desc_b.SetFormat(FORMAT_ND);
  tensor_desc_b.SetOriginFormat(FORMAT_ND);
  tensor_desc_b.SetDataType(DT_FLOAT16);
  tensor_desc_b.SetOriginDataType(DT_FLOAT16);

  op_desc_switch1->AddOutputDesc(tensor_desc_b);
  op_desc_switch2->AddOutputDesc(tensor_desc_b);

  NodePtr node_switch1 = graph->AddNode(op_desc_switch1);
  NodePtr node_switch2 = graph->AddNode(op_desc_switch2);

  std::string concatNodeName = "ConcatNode1_1";
  op::ConcatV2 op(concatNodeName.c_str());
  op.BreakConnect();
  op.create_dynamic_input_x(2);
  op.set_attr_N(2);
  auto op_desc_concat = ge::OpDescUtils::GetOpDescFromOperator(op);
  NodePtr ConcatNode1_1 = graph->AddNode(op_desc_concat);

  std::string constNodeName = "constant1";
  op::Constant opconst(constNodeName.c_str());
  opconst.BreakConnect();
  auto op_desc_constantNew = ge::OpDescUtils::GetOpDescFromOperator(opconst);

  int count = 1;
  std::unique_ptr<int32_t[]> addr(ComGraphMakeUnique<int32_t[]>(count));
  size_t i = 0U;
  int dimvalue = 1;

  addr[i] = static_cast<int32_t>(dimvalue);
  GeTensorDesc const_tensor(GeShape({1}), FORMAT_ND, DT_FLOAT16);
  GeTensor tensor(const_tensor);
  (void)tensor.SetData(reinterpret_cast<uint8_t *>(addr.get()), 1 * sizeof(int32_t));
  AttrUtils::SetTensor(op_desc_constantNew, "value", tensor);
  NodePtr node_constant1 = graph->AddNode(op_desc_constantNew);

  GraphUtils::AddEdge(node_switch1->GetOutDataAnchor(0), ConcatNode1_1->GetInDataAnchor(1));
  GraphUtils::AddEdge(node_switch2->GetOutDataAnchor(0), ConcatNode1_1->GetInDataAnchor(2));
  GraphUtils::AddEdge(node_constant1->GetOutDataAnchor(0), ConcatNode1_1->GetInDataAnchor(0));

  ConcatNode1_1->GetOpDesc()->UpdateInputDesc(0, node_switch1->GetOpDesc()->GetOutputDesc(0));
  ConcatNode1_1->GetOpDesc()->UpdateInputDesc(1, node_switch2->GetOpDesc()->GetOutputDesc(0));
  ConcatNode1_1->GetOpDesc()->UpdateOutputDesc(0, tensor_desc_a);

  graph->TopologicalSorting();
  FlattenConcatPass multiConcatFusionPro;
  auto result = multiConcatFusionPro.Run(graph);

  for (auto &node : graph->GetDirectNode()) {
      printf("Graph_multisliceConcat_CountFusionOutAnchor:split_out_name is %s start;\n",
        node->GetName().c_str());
  }
}
}