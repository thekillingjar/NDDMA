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
#include <string>

#include "macro_utils/dt_public_scope.h"
#include "engines/manager/opskernel_manager/ops_kernel_manager.h"
#include "graph/passes/format_optimize/transop_without_reshape_fusion_pass.h"
#include "macro_utils/dt_public_unscope.h"
#include "common/ge_inner_error_codes.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_builder_utils.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/utils/node_adapter.h"
#include "graph/utils/op_desc_utils_ex.h"
#include "graph/operator_reg.h"
#include "api/gelib/gelib.h"
#include "graph/common/trans_op_creator.h"

namespace ge {
namespace {
// var->cast1->transdata1->cast2->transdata2->conv2d
static ComputeGraphPtr BuildGraph1() {
  auto builder = ut::GraphBuilder("g1");
  auto var = builder.AddNode("var", VARIABLE, 1, 1, FORMAT_NC1HWC0, DT_FLOAT16, std::vector<int64_t>({1, 1, 224, 224, 16}));
  var->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({1, 1, 224, 224, 16}));
  auto cast1 = builder.AddNode("cast1", CAST, 1, 1, FORMAT_NC1HWC0, DT_FLOAT16, std::vector<int64_t>({1, 1, 224, 224, 16}));
  cast1->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT);

  auto transdata1 = builder.AddNode("transdata1", "TransData", 1, 1, FORMAT_NC1HWC0, DT_FLOAT, std::vector<int64_t>({1, 1, 224, 224, 16}));
  transdata1->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_NHWC);
  transdata1->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 224, 224, 3})));

  auto cast2 = builder.AddNode("cast2", CAST, 1, 1, FORMAT_NHWC, DT_FLOAT, std::vector<int64_t>({1, 224, 224, 3}));
  cast2->GetOpDesc()->MutableInputDesc(0)->SetDataType(DT_FLOAT16);

  auto transdata2 = builder.AddNode("transdata2", "TransData", 1, 1, FORMAT_NHWC, DT_FLOAT16, std::vector<int64_t>({1, 224, 224, 3}));
  transdata2->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_NC1HWC0);
  transdata2->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 1, 224, 224, 16})));

  auto conv2d = builder.AddNode("conv2d", "Conv2D", 1, 1, FORMAT_NC1HWC0, DT_FLOAT16, std::vector<int64_t>({1, 1, 224, 224, 16}));
  conv2d->GetOpDesc()->MutableInputDesc(0)->SetOriginShape(GeShape({1, 1, 224, 224, 16}));

  builder.AddDataEdge(var, 0, cast1, 0);
  builder.AddDataEdge(cast1, 0, transdata1, 0);
  builder.AddDataEdge(transdata1, 0, cast2, 0);
  builder.AddDataEdge(cast2, 0, transdata2, 0);
  builder.AddDataEdge(transdata2, 0, conv2d, 0);

  return builder.GetGraph();
}

//           ------>Const
//          |            |
// var->transdata1->transdata2>cast->conv2d
static ComputeGraphPtr BuildGraphWithRing() {
  auto builder = ut::GraphBuilder("g1");
  auto var = builder.AddNode("var", VARIABLE, 1, 1, FORMAT_NC1HWC0, DT_FLOAT16, std::vector<int64_t>({1, 1, 224, 224, 16}));
  var->GetOpDesc()->MutableOutputDesc(0)->SetOriginShape(GeShape({1, 1, 224, 224, 16}));

  auto transdata1 = builder.AddNode("transdata1", "TransData", 1, 1, FORMAT_NC1HWC0, DT_FLOAT16, std::vector<int64_t>({1, 1, 224, 224, 16}));
  transdata1->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_NHWC);
  transdata1->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 224, 224, 16})));
  auto transdata2 = builder.AddNode("transdata2", "TransData", 1, 1, FORMAT_NHWC, DT_FLOAT16, std::vector<int64_t>({1, 224, 224, 16}));
  transdata2->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_NC1HWC0);
  transdata2->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 1, 224, 224, 16})));
  auto cast1 = builder.AddNode("cast1", CAST, 1, 1, FORMAT_NC1HWC0, DT_FLOAT16, std::vector<int64_t>({1, 1, 224, 224, 16}));
  cast1->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT);

  auto conv2d = builder.AddNode("conv2d", "Conv2D", 1, 1, FORMAT_NC1HWC0, DT_FLOAT, std::vector<int64_t>({1, 1, 224, 224, 16}));
  conv2d->GetOpDesc()->MutableInputDesc(0)->SetOriginShape(GeShape({1, 1, 224, 224, 16}));
  auto const1 = builder.AddNode("const1", CONSTANT, 0, 1);

  builder.AddDataEdge(var, 0, transdata1, 0);
  builder.AddDataEdge(transdata1, 0, transdata2, 0);
  builder.AddDataEdge(transdata2, 0, cast1, 0);
  builder.AddDataEdge(cast1, 0, conv2d, 0);
  builder.AddControlEdge(const1, transdata2);
  builder.AddControlEdge(transdata1, const1);
  return builder.GetGraph();
}

REG_OP(Cast)
  .INPUT(x, TensorType({DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT32, DT_UINT8, DT_INT64,
                        DT_UINT64, DT_INT16, DT_UINT16, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128,
                        DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32})) /* input tensor */
  .OUTPUT(y, TensorType({DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT32, DT_UINT8, DT_INT64,
                        DT_UINT64, DT_INT16, DT_UINT16, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128,
                        DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32})) /* output tensor */
  .ATTR(dst_type, Int, 0)
  .ATTR(truncate, Bool, false)
  .OP_END_FACTORY_REG(Cast)

// stub fe opskernel info store
class FEOpsKernelInfoStore : public ge::OpsKernelInfoStore {
public:
    FEOpsKernelInfoStore() {};
    ~FEOpsKernelInfoStore() {};
    FEOpsKernelInfoStore(const FEOpsKernelInfoStore &) = delete;
    FEOpsKernelInfoStore &operator=(const FEOpsKernelInfoStore &) = delete;
    Status Initialize(const map<string, string> &options) {return 0;}
    Status Finalize() { return 0;}
    Status CreateSession(const std::map<std::string, std::string> &sessionOptions) {return 0;}
    Status DestroySession(const std::map<std::string, std::string> &sessionOptions) {return 0;}
    Status CalcOpRunningParam(Node& node){ return 0;}
    Status GenerateTask(const Node &node,
                        RunContext &context,
                        std::vector<domi::TaskDef> &tasks){ return 0;}
    bool CheckSupported(const ge::OpDescPtr &opDescPtr, std::string& unSupportReason) const {return true;}
    bool CheckAccuracySupported(const OpDescPtr &opDescPtr, std::string &un_supported_reason,
                                      const bool realQuery = false) const { return true;}
    void GetAllOpsKernelInfo(std::map<std::string, ge::OpInfo> &infos) const override;
};

void FEOpsKernelInfoStore::GetAllOpsKernelInfo(map<string, ge::OpInfo> &infos) const
{
    map<string, OpInfo> opkInfos{};
    OpInfo aicore_op = {"v100", "FEOpsStore", 0, true, false};
    infos.emplace(std::make_pair("Flatten", aicore_op));
    infos.emplace(std::make_pair("FullConnection", aicore_op));
    infos.emplace(std::make_pair("Permute", aicore_op));
    infos.emplace(std::make_pair("Transpose", aicore_op));
    infos.emplace(std::make_pair("Cast", aicore_op));
    infos.emplace(std::make_pair("TransData", aicore_op));
}

// stub aicpu opskernel info store
class AICPUOpsKernelInfoStore : public ge::OpsKernelInfoStore {
public:
    AICPUOpsKernelInfoStore() {};
    ~AICPUOpsKernelInfoStore() {};
    AICPUOpsKernelInfoStore(const AICPUOpsKernelInfoStore &) = delete;
    AICPUOpsKernelInfoStore &operator=(const AICPUOpsKernelInfoStore &) = delete;
    Status Initialize(const map<string, string> &options) {return 0;}
    Status Finalize() { return 0;}
    Status CreateSession(const std::map<std::string, std::string> &sessionOptions) {return 0;}
    Status DestroySession(const std::map<std::string, std::string> &sessionOptions) {return 0;}
    Status CalcOpRunningParam(Node& node){ return 0;}
    Status GenerateTask(const Node &node,
                                RunContext &context,
                                std::vector<domi::TaskDef> &tasks){ return 0;}
    bool CheckSupported(const ge::OpDescPtr &opDescPtr, std::string& unSupportReason) const {return true;}
    bool CheckAccuracySupported(const OpDescPtr &opDescPtr, std::string &un_supported_reason,
                                      const bool realQuery = false) const { return true;}
    void GetAllOpsKernelInfo(std::map<std::string, ge::OpInfo> &infos) const override;
};

void AICPUOpsKernelInfoStore::GetAllOpsKernelInfo(map<string, ge::OpInfo> &infos) const
{
    map<string, OpInfo> opkInfos{};
    OpInfo aicpu_op = {"DNN_VM_TF", "aicpu_kernel", 0, false, false};
    infos.emplace(std::make_pair("Flatten", aicpu_op));
    infos.emplace(std::make_pair("FullConnection", aicpu_op));
    infos.emplace(std::make_pair("Permute", aicpu_op));
    infos.emplace(std::make_pair("Transpose", aicpu_op));
    infos.emplace(std::make_pair("Cast", aicpu_op));
}

extern "C" void GetDNNEngineObjs(std::map<std::string, DNNEnginePtr> &engines);

void InitOpsKernelInfoStub()
{
    auto &opsKernelInfo = OpsKernelManager::GetInstance().ops_kernel_info_;
    // init opsKernelInfo
    map<string, OpInfo> opkInfos{};
    vector<OpInfo> flatten;
    vector<OpInfo> fullConnection;
    vector<OpInfo> cast;
    OpInfo aicpu_op = {"DNN_VM_TF", "aicpu_kernel", 1, false, false};
    OpInfo aicore_op = {"v100", "FEOpsStore", 0, true, false};
    flatten.push_back(aicpu_op);
    flatten.push_back(aicore_op);
    cast.push_back(aicpu_op);
    opsKernelInfo["FullConnection"] = fullConnection;
    opsKernelInfo["Permute"] = flatten;
    opsKernelInfo["Transpose"] = flatten;
    opsKernelInfo["Flatten"] = flatten;
    opsKernelInfo["Cast"] = cast;
    auto &opsKernelStore = OpsKernelManager::GetInstance().ops_kernel_store_;

    // init opsKernelStore
    auto aicpu_kernel_store = MakeShared<AICPUOpsKernelInfoStore>();
    auto aicore_kernel_store = MakeShared<FEOpsKernelInfoStore>();
    opsKernelStore.emplace(std::pair<string, OpsKernelInfoStorePtr>("aicpu_kernel", aicpu_kernel_store));
    opsKernelStore.emplace(std::pair<string, OpsKernelInfoStorePtr>("FEOpsStore", aicore_kernel_store));
}
} // namespace
class UtestTransopWithoutReshapeFusionPass : public testing::Test {
 protected:
  void SetUp() {
    map<string, string> options;
    ge::GELib::Initialize(options);
    InitOpsKernelInfoStub();
  }
  void TearDown() {
    ge::GELib::GetInstance()->Finalize();
  }
};

TEST_F(UtestTransopWithoutReshapeFusionPass, Run0) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("arg_0", DATA)->NODE("PartitionedCall_0", PARTITIONEDCALL)->NODE("Node_Output", NETOUTPUT));
    CHAIN(NODE("arg_1", DATA)->NODE("PartitionedCall_0"));
  };
  ComputeGraphPtr sgt_graph = ToComputeGraph(g1);
  const auto sgt_node = sgt_graph->FindNode("PartitionedCall_0");
  EXPECT_NE(sgt_node, nullptr);

  DEF_GRAPH(g2) {
    CHAIN(NODE("sgt/arg_0", DATA)->NODE("sgt/conv2d", CONV2D)->NODE("sgt/Node_Output", NETOUTPUT));
    CHAIN(NODE("sgt/arg_1", DATA)->NODE("sgt/conv2d"));
  };
  ComputeGraphPtr sub_graph = ToComputeGraph(g2);
  const auto sub_node = sub_graph->FindNode("sgt/conv2d");
  EXPECT_NE(sub_node, nullptr);

  AttrUtils::SetBool(sgt_node->GetOpDesc(), ATTR_NAME_FFTS_PLUS_SUB_GRAPH, true);
  AttrUtils::SetStr(sub_node->GetOpDesc(), ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIC");

///////////////////////////////////////////////////////////////////////////////////////////////////

  graphStatus retGraphStatus;
  TransOpWithoutReshapeFusionPass transOp;
  // no graph return success
  retGraphStatus = transOp.Run(nullptr);
  EXPECT_EQ(retGraphStatus, GRAPH_SUCCESS);

  retGraphStatus = transOp.Run(sgt_graph);
  EXPECT_EQ(retGraphStatus, GRAPH_SUCCESS);

  retGraphStatus = transOp.Run(sub_graph);
  EXPECT_EQ(retGraphStatus, GRAPH_SUCCESS);

}

TEST_F(UtestTransopWithoutReshapeFusionPass, SetRemainNode) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("arg_0", DATA)->NODE("trans", TRANSPOSE)->NODE("Output", NETOUTPUT));
    CHAIN(NODE("arg_1", DATA)->NODE("trans"));
  };
  ComputeGraphPtr graph = ToComputeGraph(g1);

  TransOpWithoutReshapeFusionPass transOp;
  std::vector<std::pair<OutDataAnchorPtr, InDataAnchorPtr>> nodes_anchor;
  const auto out = graph->FindNode("Output");
  EXPECT_NE(out, nullptr);
  auto op_desc = out->GetOpDesc();
  ge::OpDescUtilsEx::SetType(op_desc, std::string("CAST"));
  OutDataAnchorPtr outptr = std::make_shared<OutDataAnchor>(out, 0);
  const auto in0 = graph->FindNode("arg_0");
  EXPECT_NE(in0, nullptr);
  InDataAnchorPtr inptr0 = std::make_shared<InDataAnchor>(in0, 0);
  const auto in1 = graph->FindNode("arg_1");
  EXPECT_NE(in1, nullptr);
  InDataAnchorPtr inptr1 = std::make_shared<InDataAnchor>(in1, 1);
  const auto trans = graph->FindNode("trans");
  EXPECT_NE(trans, nullptr);
  InDataAnchorPtr transptr = std::make_shared<InDataAnchor>(trans, 2);

  std::pair<OutDataAnchorPtr, InDataAnchorPtr> p0(outptr, inptr0);
  nodes_anchor.push_back(p0);
  std::pair<OutDataAnchorPtr, InDataAnchorPtr> p1(outptr, inptr1);
  nodes_anchor.push_back(p1);
  std::pair<OutDataAnchorPtr, InDataAnchorPtr> p2(nullptr, transptr);
  nodes_anchor.push_back(p2);
  transOp.SetRemainNode(nodes_anchor);

  bool val = false;
  op_desc = trans->GetOpDesc();
  op_desc->TryGetExtAttr(std::string("node_remain"), val);
  EXPECT_TRUE(val == false);
}

TEST_F(UtestTransopWithoutReshapeFusionPass, remove_all_duplicate_trans_op) {
  // build graph
  auto graph = BuildGraph1();
  // run pass
  TransOpWithoutReshapeFusionPass trans_op_fusion_pass;
  auto ret = trans_op_fusion_pass.Run(graph);
  EXPECT_EQ(ret, SUCCESS);
  auto var = graph->FindNode("var");
  EXPECT_EQ(var->GetOutNodes().at(0)->GetType(), "Conv2D");
}

TEST_F(UtestTransopWithoutReshapeFusionPass, remove_all_duplicate_trans_op_with_ring) {
  // build graph
  auto graph = BuildGraphWithRing();
  // run pass
  TransOpWithoutReshapeFusionPass trans_op_fusion_pass;
  auto ret = trans_op_fusion_pass.Run(graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->TopologicalSorting(), SUCCESS); // topo success, no cycle
}

class NodeBuilder {
 public:
  NodeBuilder(const std::string& name, const std::string& type) {
    op_desc_ = std::make_shared<OpDesc>(name, type);
  }

  NodeBuilder& AddInputDesc(std::initializer_list<int64_t> shape,
                            ge::Format format = FORMAT_NCHW,
                            ge::DataType data_type = DT_FLOAT) {
    op_desc_->AddInputDesc(*CreateTensorDesc(shape, format, data_type));
    return *this;
  }

  NodeBuilder& AddOutputDesc(std::initializer_list<int64_t> shape,
                             ge::Format format = FORMAT_NCHW,
                             ge::DataType data_type = DT_FLOAT) {
    op_desc_->AddOutputDesc(*CreateTensorDesc(shape, format, data_type));
    return *this;
  }

  ge::NodePtr Build(const ge::ComputeGraphPtr& graph) {
    return graph->AddNode(op_desc_);
  }
 private:
  ge::GeTensorDescPtr CreateTensorDesc(std::initializer_list<int64_t> shape,
                                       ge::Format format = FORMAT_NCHW,
                                       ge::DataType data_type = DT_FLOAT) {
    GeShape ge_shape{std::vector<int64_t>(shape)};
    ge::GeTensorDescPtr tensor_desc = std::make_shared<ge::GeTensorDesc>();
    tensor_desc->SetShape(ge_shape);
    tensor_desc->SetOriginShape(ge_shape);
    tensor_desc->SetFormat(format);
    tensor_desc->SetDataType(data_type);
    return tensor_desc;
  }

  ge::OpDescPtr op_desc_;
};

/*

    Node---transdata---cast---cast---transdata---A

            ||
            \/
    Node---transdata---cast---cast---transdata---A
*/
TEST_F(UtestTransopWithoutReshapeFusionPass, test_input_output_format_not_support)
{
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("graph_test");
  ge::NodePtr node = NodeBuilder("Data4D", DATA)
      .AddOutputDesc({1,2,3,4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr transdata_5d_2_4d = NodeBuilder("transdata_5d_2_4d", TRANSDATA)
                            .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                            .Build(graph);

  ge::NodePtr transdata_4d_2_nz = NodeBuilder("transdata_4d_2_nz", TRANSDATA)
                            .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                            .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_FRACTAL_NZ, DT_FLOAT16)
                            .Build(graph);

  ge::NodePtr cast_fp16_2_fp32 = NodeBuilder("cast_fp16_2_fp32", CAST)
                                .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                                .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
                                .Build(graph);
  ge::NodePtr cast_fp32_2_fp16 = NodeBuilder("cast_fp32_2_fp16", CAST)
                                .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
                                .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                                .Build(graph);
  ge::NodePtr node_relu_1 = NodeBuilder("relu_1", RELU)
                            .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .Build(graph);
  ge::NodePtr node_relu_b = NodeBuilder("relu_b", RELU)
                            .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .Build(graph);
  ge::NodePtr node_relu_c = NodeBuilder("relu_c", RELU)
                            .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_FRACTAL_NZ, DT_FLOAT16)
                            .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_FRACTAL_NZ, DT_FLOAT16)
                            .Build(graph);

  ge::GraphUtils::AddEdge(node->GetOutDataAnchor(0), transdata_5d_2_4d->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transdata_5d_2_4d->GetOutDataAnchor(0), cast_fp16_2_fp32->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp16_2_fp32->GetOutDataAnchor(0), cast_fp32_2_fp16->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp32_2_fp16->GetOutDataAnchor(0), transdata_4d_2_nz->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transdata_4d_2_nz->GetOutDataAnchor(0), node_relu_1->GetInDataAnchor(0));


  ge::TransOpWithoutReshapeFusionPass pass;
  uint32_t status = pass.Run(graph);
  EXPECT_EQ(domi::SUCCESS, status);
  EXPECT_EQ(node->GetOutDataNodes().size(), 1);
  EXPECT_EQ(node->GetOutDataNodes().at(0)->GetName(), "transdata_5d_2_4d");
}

/*
    Node---transpose---C

            ||
            \/
    Node---transpose---C
*/
TEST_F(UtestTransopWithoutReshapeFusionPass, test_transpose_output_ND)
{
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("graph_test");

  ge::NodePtr node = NodeBuilder("Data4D", DATA)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NHWC, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr transpose_node = NodeBuilder("transpose_node", TRANSPOSE)
                               .AddInputDesc({1, 2, 3, 4}, FORMAT_NHWC, DT_FLOAT16)
                               .AddOutputDesc({1, 2, 3, 4}, FORMAT_ND, DT_FLOAT16)
                               .Build(graph);


  ge::NodePtr node_c = NodeBuilder("node_c", MUL)
                       .AddInputDesc({1, 2, 3, 4}, FORMAT_NHWC, DT_FLOAT16)
                       .AddOutputDesc({1, 2, 3, 4}, FORMAT_NHWC, DT_FLOAT16)
                       .Build(graph);

  // shape different,but format and datatype are equal
  ge::GraphUtils::AddEdge(node->GetOutDataAnchor(0), transpose_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transpose_node->GetOutDataAnchor(0), node_c->GetInDataAnchor(0));

  ge::TransOpWithoutReshapeFusionPass pass;
  uint32_t status = pass.Run(graph);
  EXPECT_EQ(domi::SUCCESS, status);
  EXPECT_EQ(node->GetOutDataNodes().size(), 1);
  if (node->GetOutDataNodes().size() == 1) {
    NodePtr out_node0 = node->GetOutDataNodes().at(0);
    EXPECT_EQ(out_node0->GetOpDesc()->GetType(), TRANSPOSE);
  }
}

/*
    Node---transpose---C

            ||
            \/
    Node---transpose---C
*/
TEST_F(UtestTransopWithoutReshapeFusionPass, test_transpose_input_ND)
{
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("graph_test");

  ge::NodePtr node = NodeBuilder("Data4D", DATA)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_ND, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr transpose_node = NodeBuilder("transpose_node", TRANSPOSE)
                               .AddInputDesc({1, 2, 3, 4}, FORMAT_ND, DT_FLOAT16)
                               .AddOutputDesc({1, 2, 3, 4}, FORMAT_NHWC, DT_FLOAT16)
                               .Build(graph);


  ge::NodePtr node_c = NodeBuilder("node_c", MUL)
                       .AddInputDesc({1, 2, 3, 4}, FORMAT_NHWC, DT_FLOAT16)
                       .AddOutputDesc({1, 2, 3, 4}, FORMAT_NHWC, DT_FLOAT16)
                       .Build(graph);

  // shape different,but format and datatype are equal
  ge::GraphUtils::AddEdge(node->GetOutDataAnchor(0), transpose_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transpose_node->GetOutDataAnchor(0), node_c->GetInDataAnchor(0));

  ge::TransOpWithoutReshapeFusionPass pass;
  uint32_t status = pass.Run(graph);
  EXPECT_EQ(domi::SUCCESS, status);
  EXPECT_EQ(node->GetOutDataNodes().size(), 1);
  if (node->GetOutDataNodes().size() == 1) {
    NodePtr out_node0 = node->GetOutDataNodes().at(0);
    EXPECT_EQ(out_node0->GetOpDesc()->GetType(), TRANSPOSE);
  }
}

/*
    Node---transpose---C

            ||
            \/
    Node---transpose---C
*/
TEST_F(UtestTransopWithoutReshapeFusionPass, test_not_continuous)
{
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("graph_test");

  ge::NodePtr node = NodeBuilder("Data4D", DATA)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NHWC, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr transpose_node = NodeBuilder("transpose_node", TRANSPOSE)
                               .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                               .AddOutputDesc({1, 2, 3, 4}, FORMAT_NHWC, DT_FLOAT16)
                               .Build(graph);


  ge::NodePtr node_c = NodeBuilder("node_c", MUL)
                       .AddInputDesc({1, 2, 3, 4}, FORMAT_NHWC, DT_FLOAT16)
                       .AddOutputDesc({1, 2, 3, 4}, FORMAT_NHWC, DT_FLOAT16)
                       .Build(graph);

  // shape different,but format and datatype are equal
  ge::GraphUtils::AddEdge(node->GetOutDataAnchor(0), transpose_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transpose_node->GetOutDataAnchor(0), node_c->GetInDataAnchor(0));

  ge::TransOpWithoutReshapeFusionPass pass;
  uint32_t status = pass.Run(graph);
  EXPECT_EQ(domi::SUCCESS, status);
  EXPECT_EQ(node->GetOutDataNodes().size(), 1);
  if (node->GetOutDataNodes().size() == 1) {
    NodePtr out_node0 = node->GetOutDataNodes().at(0);
    EXPECT_EQ(out_node0->GetOpDesc()->GetType(), TRANSPOSE);
  }
}
    /*
        Node---transpose---C

                ||
                \/
        Node---transpose---C
    */
TEST_F(UtestTransopWithoutReshapeFusionPass, test_transpose_unknown_shape)
{
    ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("graph_test");

    ge::NodePtr node = NodeBuilder("Data4D", DATA)
        .AddOutputDesc({1, 2, 3, 4}, FORMAT_NHWC, DT_FLOAT16)
        .Build(graph);

    ge::NodePtr transpose_node = NodeBuilder("transpose_node", TRANSPOSE)
                                 .AddInputDesc({1, -1, -1, 4}, FORMAT_NHWC, DT_FLOAT16)
                                 .AddOutputDesc({1, -1, -1, 4}, FORMAT_NHWC, DT_FLOAT16)
                                 .Build(graph);


    ge::NodePtr node_c = NodeBuilder("node_c", MUL)
                         .AddInputDesc({1, -1, -1, 4}, FORMAT_NHWC, DT_FLOAT16)
                         .AddOutputDesc({1, -1, -1, 4}, FORMAT_NHWC, DT_FLOAT16)
                         .Build(graph);

    // shape different,but format and datatype are equal
    ge::GraphUtils::AddEdge(node->GetOutDataAnchor(0), transpose_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(transpose_node->GetOutDataAnchor(0), node_c->GetInDataAnchor(0));

    ge::TransOpWithoutReshapeFusionPass pass;
    dlog_setlevel(GE_MODULE_NAME, 0, 0);
    uint32_t status = pass.Run(graph);
    EXPECT_EQ(domi::SUCCESS, status);
    EXPECT_EQ(node->GetOutDataNodes().size(), 1);
    if (node->GetOutDataNodes().size() == 1) {
        NodePtr out_node0 = node->GetOutDataNodes().at(0);
        EXPECT_EQ(out_node0->GetOpDesc()->GetType(), TRANSPOSE);
    }
}

TEST_F(UtestTransopWithoutReshapeFusionPass, test_add_node_1) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("graph_test");
  TransOpWithoutReshapeFusionPass pass;
  NodePtr cast_node;
  auto ret = pass.AddTransNode(graph, nullptr, cast_node);
  EXPECT_EQ(0, ret);
}

/*

    Node---transdata---cast---cast---transdata---A
                                  |  |
                                  |++|

            ||
            \/

        Node-----A
            |   |
            |+++|


*/
TEST_F(UtestTransopWithoutReshapeFusionPass, test_output_ctrl_in_ctrl)
{
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("graph_test");
  ge::NodePtr node = NodeBuilder("Data4D", DATA)
      .AddOutputDesc({1,2,3,4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr transdata_5d_2_4d = NodeBuilder("transdata_5d_2_4d", TRANSDATA)
                            .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                            .Build(graph);

  ge::NodePtr transdata_4d_2_5d = NodeBuilder("transdata_4d_2_5d", TRANSDATA)
                            .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                            .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .Build(graph);

  ge::NodePtr cast_fp16_2_fp32 = NodeBuilder("cast_fp16_2_fp32", CAST)
                                .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                                .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
                                .Build(graph);
  ge::NodePtr cast_fp32_2_fp16 = NodeBuilder("cast_fp32_2_fp16", CAST)
                                .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
                                .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                                .Build(graph);
  ge::NodePtr node_relu_1 = NodeBuilder("relu_1", RELU)
                            .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .Build(graph);

  ge::GraphUtils::AddEdge(node->GetOutDataAnchor(0), transdata_5d_2_4d->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transdata_5d_2_4d->GetOutDataAnchor(0), cast_fp16_2_fp32->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp16_2_fp32->GetOutDataAnchor(0), cast_fp32_2_fp16->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp32_2_fp16->GetOutDataAnchor(0), transdata_4d_2_5d->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transdata_4d_2_5d->GetOutDataAnchor(0), node_relu_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp32_2_fp16->GetOutControlAnchor(), transdata_4d_2_5d->GetInControlAnchor());

  ge::TransOpWithoutReshapeFusionPass pass;
  uint32_t status = pass.Run(graph);
  EXPECT_EQ(domi::SUCCESS, status);
  EXPECT_EQ(node_relu_1->GetInControlAnchor()->GetPeerOutControlAnchors().size(), 1);
  auto peer_node = node_relu_1->GetInControlAnchor()->GetPeerOutControlAnchors().at(0)->GetOwnerNode();
  EXPECT_EQ(peer_node->GetName(), "Data4D");
  EXPECT_EQ(node->GetOutControlAnchor()->GetPeerInControlAnchors().size(), 1);
  EXPECT_EQ(node->GetOutControlAnchor()->GetPeerInControlAnchors().at(0)->GetOwnerNode()->GetOpDesc()->GetType(), RELU);
}

/*
                  B+++|
                      |
    Node---transdata---cast---cast---transdata---A
                                              |
                                              |++C

            ||
            \/
            B+++|
                |
        Node----|-A
            |
            |+++C

*/
TEST_F(UtestTransopWithoutReshapeFusionPass, test_same_input_output_in_data_ctrl)
{
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("graph_test");
  ge::NodePtr node = NodeBuilder("Data4D", DATA)
      .AddOutputDesc({1,2,3,4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr transdata_5d_2_4d = NodeBuilder("transdata_5d_2_4d", TRANSDATA)
                            .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                            .Build(graph);

  ge::NodePtr transdata_4d_2_5d = NodeBuilder("transdata_4d_2_5d", TRANSDATA)
                            .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                            .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .Build(graph);

  ge::NodePtr cast_fp16_2_fp32 = NodeBuilder("cast_fp16_2_fp32", CAST)
                                .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                                .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
                                .Build(graph);
  ge::NodePtr cast_fp32_2_fp16 = NodeBuilder("cast_fp32_2_fp16", CAST)
                                .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
                                .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                                .Build(graph);
  ge::NodePtr node_relu_1 = NodeBuilder("relu_1", RELU)
                            .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .Build(graph);
  ge::NodePtr node_relu_b = NodeBuilder("relu_b", RELU)
                            .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .Build(graph);
  ge::NodePtr node_relu_c = NodeBuilder("relu_c", RELU)
                            .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .Build(graph);

  ge::GraphUtils::AddEdge(node->GetOutDataAnchor(0), transdata_5d_2_4d->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transdata_5d_2_4d->GetOutDataAnchor(0), cast_fp16_2_fp32->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp16_2_fp32->GetOutDataAnchor(0), cast_fp32_2_fp16->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp32_2_fp16->GetOutDataAnchor(0), transdata_4d_2_5d->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transdata_4d_2_5d->GetOutDataAnchor(0), node_relu_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_relu_b->GetOutControlAnchor(), cast_fp16_2_fp32->GetInControlAnchor());
  ge::GraphUtils::AddEdge(transdata_4d_2_5d->GetOutControlAnchor(), node_relu_c->GetInDataAnchor(0));

  ge::TransOpWithoutReshapeFusionPass pass;
  uint32_t status = pass.Run(graph);
  EXPECT_EQ(domi::SUCCESS, status);
  EXPECT_EQ(node_relu_b->GetOutControlNodes().size(), 1);
  EXPECT_EQ(node_relu_b->GetOutControlNodes().at(0)->GetOpDesc()->GetType(), RELU);
  EXPECT_EQ(node->GetOutControlNodes().size(), 1);
  EXPECT_EQ(node->GetOutControlNodes().at(0)->GetOpDesc()->GetType(), RELU);
}

TEST_F(UtestTransopWithoutReshapeFusionPass, test_run_graph_nullptr)
{
  ge::TransOpWithoutReshapeFusionPass pass;
  uint32_t status = pass.Run(nullptr);
  EXPECT_EQ(domi::SUCCESS, status);
}
/*

    Node---transdata---cast---cast---transdata---A
        |
        |+++B


            ||
            \/

        Node-----A
            |
            |+++B


*/
TEST_F(UtestTransopWithoutReshapeFusionPass, test_output_ctrl_peer_in_ctrl)
{
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("graph_test");
  ge::NodePtr node = NodeBuilder("Data4D", DATA)
      .AddOutputDesc({1,2,3,4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr transdata_5d_2_4d = NodeBuilder("transdata_5d_2_4d", TRANSDATA)
                            .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                            .Build(graph);

  ge::NodePtr transdata_4d_2_5d = NodeBuilder("transdata_4d_2_5d", TRANSDATA)
                            .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                            .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .Build(graph);

  ge::NodePtr cast_fp16_2_fp32 = NodeBuilder("cast_fp16_2_fp32", CAST)
                                .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                                .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
                                .Build(graph);
  ge::NodePtr cast_fp32_2_fp16 = NodeBuilder("cast_fp32_2_fp16", CAST)
                                .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
                                .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                                .Build(graph);
  ge::NodePtr node_relu_1 = NodeBuilder("relu_1", RELU)
                            .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .Build(graph);
  ge::NodePtr node_b = NodeBuilder("node_b", SOFTMAX)
                            .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .Build(graph);

  ge::GraphUtils::AddEdge(node->GetOutDataAnchor(0), transdata_5d_2_4d->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transdata_5d_2_4d->GetOutDataAnchor(0), cast_fp16_2_fp32->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp16_2_fp32->GetOutDataAnchor(0), cast_fp32_2_fp16->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp32_2_fp16->GetOutDataAnchor(0), transdata_4d_2_5d->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transdata_4d_2_5d->GetOutDataAnchor(0), node_relu_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node->GetOutControlAnchor(), node_b->GetInControlAnchor());

  ge::TransOpWithoutReshapeFusionPass pass;
  uint32_t status = pass.Run(graph);
  EXPECT_EQ(domi::SUCCESS, status);
  EXPECT_EQ(node->GetOutControlNodes().size(), 1);
  EXPECT_EQ(node->GetOutControlNodes().at(0)->GetOpDesc()->GetType(), SOFTMAX);
}

/*

    Node---transdata---cast---cast---transdata---A
        |
        |+++B


            ||
            \/

        Node-----A
            |
            |+++B


*/
TEST_F(UtestTransopWithoutReshapeFusionPass, test_output_ctrl_peer_in_data1)
{
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("graph_test");
  ge::NodePtr node = NodeBuilder("Data4D", DATA)
      .AddOutputDesc({1,2,3,4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr transdata_5d_2_4d = NodeBuilder("transdata_5d_2_4d", TRANSDATA)
                            .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                            .Build(graph);

  ge::NodePtr transdata_4d_2_5d = NodeBuilder("transdata_4d_2_5d", TRANSDATA)
                            .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                            .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .Build(graph);

  ge::NodePtr cast_fp16_2_fp32 = NodeBuilder("cast_fp16_2_fp32", CAST)
                                .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                                .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
                                .Build(graph);
  ge::NodePtr cast_fp32_2_fp16 = NodeBuilder("cast_fp32_2_fp16", CAST)
                                .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
                                .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                                .Build(graph);
  ge::NodePtr node_relu_1 = NodeBuilder("relu_1", RELU)
                            .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .Build(graph);
  ge::NodePtr node_b = NodeBuilder("node_b", SOFTMAX)
                            .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .Build(graph);

  ge::GraphUtils::AddEdge(node->GetOutDataAnchor(0), transdata_5d_2_4d->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transdata_5d_2_4d->GetOutDataAnchor(0), cast_fp16_2_fp32->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp16_2_fp32->GetOutDataAnchor(0), cast_fp32_2_fp16->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp32_2_fp16->GetOutDataAnchor(0), transdata_4d_2_5d->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transdata_4d_2_5d->GetOutDataAnchor(0), node_relu_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node->GetOutControlAnchor(), node_b->GetInDataAnchor(0));

  ge::TransOpWithoutReshapeFusionPass pass;
  uint32_t status = pass.Run(graph);
  EXPECT_EQ(domi::SUCCESS, status);
  EXPECT_EQ(node->GetOutControlNodes().size(), 1);
  EXPECT_EQ(node->GetOutControlNodes().at(0)->GetOpDesc()->GetType(), SOFTMAX);
}

/*

    Node---transdata---cast---transpose---cast---transdata---A


            ||
            \/

    Node---transdata---cast---transpose---cast---transdata---A

*/
TEST_F(UtestTransopWithoutReshapeFusionPass, test_can_not_fusion_format)
{
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("graph_test");
  ge::NodePtr node = NodeBuilder("Data4D", DATA)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr transdata_5d_2_4d = NodeBuilder("transdata_5d_2_4d", TRANSDATA)
                            .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                            .Build(graph);

  ge::NodePtr transdata_4d_2_5d = NodeBuilder("transdata_4d_2_5d", TRANSDATA)
                            .AddInputDesc({1, 3, 4, 2}, FORMAT_NHWC, DT_FLOAT16)
                            .AddOutputDesc({1, 3, 4, 5, 2}, FORMAT_FRACTAL_NZ, DT_FLOAT16)
                            .Build(graph);

  ge::NodePtr cast_fp16_2_fp32 = NodeBuilder("cast_fp16_2_fp32", CAST)
                                .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                                .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
                                .Build(graph);
  ge::NodePtr transpose_nchw_2_nhwc = NodeBuilder("transpose_nchw_2_nhwc", TRANSPOSE)
                                      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
                                      .AddOutputDesc({1, 3, 4, 2}, FORMAT_NHWC, DT_FLOAT)
                                      .Build(graph);
  ge::NodePtr cast_fp32_2_fp16 = NodeBuilder("cast_fp32_2_fp16", CAST)
                                .AddInputDesc({1, 3, 4, 2}, FORMAT_NHWC, DT_FLOAT16)
                                .AddOutputDesc({1, 3, 4, 2}, FORMAT_NHWC, DT_FLOAT16)
                                .Build(graph);

  ge::NodePtr node_relu_1 = NodeBuilder("relu_1", RELU)
                            .AddInputDesc({1, 3, 4, 5, 2}, FORMAT_FRACTAL_NZ, DT_FLOAT16)
                            .AddOutputDesc({1, 3, 4, 5, 2}, FORMAT_FRACTAL_NZ, DT_FLOAT16)
                            .Build(graph);

  ge::GraphUtils::AddEdge(node->GetOutDataAnchor(0), transdata_5d_2_4d->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transdata_5d_2_4d->GetOutDataAnchor(0), cast_fp16_2_fp32->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp16_2_fp32->GetOutDataAnchor(0), transpose_nchw_2_nhwc->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transpose_nchw_2_nhwc->GetOutDataAnchor(0), cast_fp32_2_fp16->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp32_2_fp16->GetOutDataAnchor(0), transdata_4d_2_5d->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transdata_4d_2_5d->GetOutDataAnchor(0), node_relu_1->GetInDataAnchor(0));

  ge::TransOpWithoutReshapeFusionPass pass;
  uint32_t status = pass.Run(graph);
  EXPECT_EQ(domi::SUCCESS, status);
  EXPECT_EQ(node->GetOutDataNodes().size(), 1);
  EXPECT_EQ(node->GetOutDataNodes().at(0)->GetOpDesc()->GetType(), TRANSDATA);
}

/*

    Node---transdata---cast---transpose---cast---transdata---A


            ||
            \/

    Node---transdata---cast---transpose---cast---transdata---A

*/
TEST_F(UtestTransopWithoutReshapeFusionPass, test_format_same_shape_different)
{
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("graph_test");
  ge::NodePtr node = NodeBuilder("Data4D", DATA)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr transdata_5d_2_4d = NodeBuilder("transdata_5d_2_4d", TRANSDATA)
                            .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                            .Build(graph);

  ge::NodePtr transdata_4d_2_5d = NodeBuilder("transdata_4d_2_5d", TRANSDATA)
                            .AddInputDesc({1, 3, 4, 2}, FORMAT_NHWC, DT_FLOAT16)
                            .AddOutputDesc({1, 3, 4, 5, 2}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .Build(graph);

  ge::NodePtr cast_fp16_2_fp32 = NodeBuilder("cast_fp16_2_fp32", CAST)
                                .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                                .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
                                .Build(graph);
  ge::NodePtr transpose_nchw_2_nhwc = NodeBuilder("transpose_nchw_2_nhwc", TRANSPOSE)
                                      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
                                      .AddOutputDesc({1, 3, 4, 2}, FORMAT_NHWC, DT_FLOAT)
                                      .Build(graph);
  ge::NodePtr cast_fp32_2_fp16 = NodeBuilder("cast_fp32_2_fp16", CAST)
                                .AddInputDesc({1, 3, 4, 2}, FORMAT_NHWC, DT_FLOAT16)
                                .AddOutputDesc({1, 3, 4, 2}, FORMAT_NHWC, DT_FLOAT16)
                                .Build(graph);

  ge::NodePtr node_relu_1 = NodeBuilder("relu_1", RELU)
                            .AddInputDesc({1, 3, 4, 5, 2}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .AddOutputDesc({1, 3, 4, 5, 2}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .Build(graph);

  ge::GraphUtils::AddEdge(node->GetOutDataAnchor(0), transdata_5d_2_4d->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transdata_5d_2_4d->GetOutDataAnchor(0), cast_fp16_2_fp32->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp16_2_fp32->GetOutDataAnchor(0), transpose_nchw_2_nhwc->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transpose_nchw_2_nhwc->GetOutDataAnchor(0), cast_fp32_2_fp16->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp32_2_fp16->GetOutDataAnchor(0), transdata_4d_2_5d->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transdata_4d_2_5d->GetOutDataAnchor(0), node_relu_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transdata_4d_2_5d->GetOutControlAnchor(), node_relu_1->GetInDataAnchor(0));

  ge::TransOpWithoutReshapeFusionPass pass;
  uint32_t status = pass.Run(graph);
  EXPECT_EQ(domi::SUCCESS, status);
  EXPECT_EQ(node->GetOutDataNodes().size(), 1);
  EXPECT_EQ(node->GetOutDataNodes().at(0)->GetOpDesc()->GetType(), TRANSDATA);
}

/*

    Node---transdata---cast---A

            ||
            \/

    Node---transdata---cast---A

*/
TEST_F(UtestTransopWithoutReshapeFusionPass, test_can_not_fusion)
{
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("graph_test");
  ge::NodePtr node = NodeBuilder("Data4D", DATA)
      .AddOutputDesc({1,2,3,4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr transdata_5d_2_4d = NodeBuilder("transdata_5d_2_4d", TRANSDATA)
                            .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                            .Build(graph);

  ge::NodePtr cast_fp16_2_fp32 = NodeBuilder("cast_fp16_2_fp32", CAST)
                                .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                                .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
                                .Build(graph);

  ge::NodePtr node_a = NodeBuilder("node_a", RELU)
                            .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
                            .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
                            .Build(graph);

  ge::GraphUtils::AddEdge(node->GetOutDataAnchor(0), transdata_5d_2_4d->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transdata_5d_2_4d->GetOutDataAnchor(0), cast_fp16_2_fp32->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp16_2_fp32->GetOutDataAnchor(0), node_a->GetInDataAnchor(0));

  ge::TransOpWithoutReshapeFusionPass pass;
  uint32_t status = pass.Run(graph);
  EXPECT_EQ(domi::SUCCESS, status);
  EXPECT_EQ(node->GetOutDataNodes().size(), 1);
  EXPECT_EQ(node->GetOutDataNodes().at(0)->GetOpDesc()->GetType(), TRANSDATA);
}

/*

    Node---cast---transdata---A

            ||
            \/

    Node---cast---transdata---A

*/
TEST_F(UtestTransopWithoutReshapeFusionPass, test_can_not_fusion_format_not_continue) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("graph_test");
  ge::NodePtr node = NodeBuilder("Data4D", DATA).AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT).Build(graph);

  ge::NodePtr transdata_op = NodeBuilder("transdata_op", TRANSDATA)
                                 .AddInputDesc({1, 2, 3, 4}, FORMAT_NHWC, DT_FLOAT)
                                 .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
                                 .Build(graph);

  ge::NodePtr cast_op = NodeBuilder("cast_op", CAST)
                            .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
                            .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
                            .Build(graph);

  ge::NodePtr node_a = NodeBuilder("node_a", RELU)
                           .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
                           .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
                           .Build(graph);

  ge::GraphUtils::AddEdge(node->GetOutDataAnchor(0), cast_op->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_op->GetOutDataAnchor(0), transdata_op->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transdata_op->GetOutDataAnchor(0), node_a->GetInDataAnchor(0));

  ge::TransOpWithoutReshapeFusionPass pass;
  uint32_t status = pass.Run(graph);
  EXPECT_EQ(domi::SUCCESS, status);
  EXPECT_EQ(node_a->GetInDataNodes().size(), 1);
  EXPECT_EQ(node_a->GetInDataNodes().at(0)->GetOpDesc()->GetType(), TRANSDATA);
}

/*

    Node_a---Node_b---transdata---cast---cast---transdata---A
            ||
            \/
        Node-_a---Node_b---A

*/
TEST_F(UtestTransopWithoutReshapeFusionPass, test_one_output)
{
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("graph_test");
  ge::NodePtr node_a = NodeBuilder("Data4D_a", DATA)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .Build(graph);
  ge::NodePtr node_b = NodeBuilder("Data4D_b", DATA)
      .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr transdata_1 = NodeBuilder("transdata_1", TRANSDATA)
                            .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                            .Build(graph);

  ge::NodePtr transdata_2 = NodeBuilder("transdata_2", TRANSDATA)
                            .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                            .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .Build(graph);

  ge::NodePtr cast_1 = NodeBuilder("cast_1", CAST)
                                .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                                .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
                                .Build(graph);
  ge::NodePtr cast_2 = NodeBuilder("cast_2", CAST)
                                .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
                                .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                                .Build(graph);
  ge::NodePtr node_A = NodeBuilder("node_A", RELU)
                            .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .Build(graph);

  ge::GraphUtils::AddEdge(node_a->GetOutDataAnchor(0), node_b->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_b->GetOutDataAnchor(0), transdata_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transdata_1->GetOutDataAnchor(0), cast_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_1->GetOutDataAnchor(0), cast_2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_2->GetOutDataAnchor(0), transdata_2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transdata_2->GetOutDataAnchor(0), node_A->GetInDataAnchor(0));
  ge::TransOpWithoutReshapeFusionPass pass;
  uint32_t status = pass.Run(graph);
  EXPECT_EQ(domi::SUCCESS, status);
  EXPECT_EQ(node_A->GetInDataNodes().size(), 1);
  auto peer_in_node = node_A->GetInDataNodes().at(0);
  EXPECT_EQ(peer_in_node->GetOpDesc()->GetType(), DATA);
}

/*

    Node---transdata---cast---cast---transdata---A
                                              |
                                              |++B


            ||
            \/

        Node-----A
            |
            |+++B


*/
TEST_F(UtestTransopWithoutReshapeFusionPass, test_output_ctrl_in_data)
{
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("graph_test");
  ge::NodePtr node = NodeBuilder("Data4D", DATA)
      .AddOutputDesc({1,2,3,4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr transdata_5d_2_4d = NodeBuilder("transdata_5d_2_4d", TRANSDATA)
                            .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                            .Build(graph);

  ge::NodePtr transdata_4d_2_5d = NodeBuilder("transdata_4d_2_5d", TRANSDATA)
                            .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                            .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .Build(graph);

  ge::NodePtr cast_fp16_2_fp32 = NodeBuilder("cast_fp16_2_fp32", CAST)
                                .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                                .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
                                .Build(graph);
  ge::NodePtr cast_fp32_2_fp16 = NodeBuilder("cast_fp32_2_fp16", CAST)
                                .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
                                .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                                .Build(graph);
  ge::NodePtr node_relu_1 = NodeBuilder("relu_1", RELU)
                            .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .Build(graph);
  ge::NodePtr node_b = NodeBuilder("node_b", SOFTMAX)
                            .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .Build(graph);

  ge::GraphUtils::AddEdge(node->GetOutDataAnchor(0), transdata_5d_2_4d->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transdata_5d_2_4d->GetOutDataAnchor(0), cast_fp16_2_fp32->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp16_2_fp32->GetOutDataAnchor(0), cast_fp32_2_fp16->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp32_2_fp16->GetOutDataAnchor(0), transdata_4d_2_5d->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transdata_4d_2_5d->GetOutDataAnchor(0), node_relu_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transdata_4d_2_5d->GetOutControlAnchor(), node_b->GetInDataAnchor(0));

  ge::TransOpWithoutReshapeFusionPass pass;
  uint32_t status = pass.Run(graph);
  EXPECT_EQ(domi::SUCCESS, status);
  EXPECT_EQ(node->GetOutControlAnchor()->GetPeerInDataAnchors().size(), 1);
  EXPECT_EQ(node->GetOutControlAnchor()->GetPeerInDataAnchors().at(0)->GetOwnerNode()->GetOpDesc()->GetType(), SOFTMAX);
}

/*

    Node---transdata---cast---cast---transdata---A
            ||
            \/
        Node---A

*/
TEST_F(UtestTransopWithoutReshapeFusionPass, test_same_input_output)
{
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("graph_test");
  ge::NodePtr node = NodeBuilder("Data4D", DATA)
      .AddOutputDesc({1,2,3,4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr transdata_1 = NodeBuilder("transdata_1", TRANSDATA)
                            .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                            .Build(graph);

  ge::NodePtr transdata_2 = NodeBuilder("transdata_2", TRANSDATA)
                            .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                            .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .Build(graph);

  ge::NodePtr cast_1 = NodeBuilder("cast_1", CAST)
                                .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                                .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
                                .Build(graph);
  ge::NodePtr cast_2 = NodeBuilder("cast_2", CAST)
                                .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
                                .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                                .Build(graph);
  ge::NodePtr node_relu_1 = NodeBuilder("relu_1", RELU)
                            .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .Build(graph);

  ge::GraphUtils::AddEdge(node->GetOutDataAnchor(0), transdata_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transdata_1->GetOutDataAnchor(0), cast_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_1->GetOutDataAnchor(0), cast_2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_2->GetOutDataAnchor(0), transdata_2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transdata_2->GetOutDataAnchor(0), node_relu_1->GetInDataAnchor(0));
  ge::TransOpWithoutReshapeFusionPass pass;
  uint32_t status = pass.Run(graph);
  EXPECT_EQ(domi::SUCCESS, status);
}

/*
    Node---|---transdata---cast---cast---transdata---A
           |
           |---transdata--cast--reshape--cast---B

            ||
            \/
    Node---|---A
           |
           |---transdata--cast--reshape--cast--transdata---B

*/
TEST_F(UtestTransopWithoutReshapeFusionPass, test_has_reshpae)
{
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("graph_test");

  ge::NodePtr node = NodeBuilder("Data4D", DATA)
      .AddOutputDesc({1,2,3,4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr transdata_1 = NodeBuilder("transdata_1", TRANSDATA)
                            .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                            .Build(graph);

  ge::NodePtr transdata_2 = NodeBuilder("transdata_2", TRANSDATA)
                            .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                            .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .Build(graph);

  ge::NodePtr transdata_3 = NodeBuilder("transdata_3", TRANSDATA)
                            .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                            .Build(graph);

  ge::NodePtr cast_1 = NodeBuilder("cast_1", CAST)
                                .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                                .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
                                .Build(graph);

  ge::NodePtr cast_2 = NodeBuilder("cast_2", CAST)
                                .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
                                .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                                .Build(graph);

  ge::NodePtr reshape_1 = NodeBuilder("reshape_1", RESHAPE)
                                .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                                .AddOutputDesc({1, 2, 2, 6}, FORMAT_NCHW, DT_FLOAT)
                                .Build(graph);

  ge::NodePtr cast_3 = NodeBuilder("cast_3", CAST)
                                .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                                .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
                                .Build(graph);

  ge::NodePtr cast_4 = NodeBuilder("cast_4", CAST)
                                .AddInputDesc({1, 2, 2, 6}, FORMAT_NCHW, DT_FLOAT)
                                .AddOutputDesc({1, 2, 2, 6}, FORMAT_NCHW, DT_FLOAT16)
                                .Build(graph);

  ge::NodePtr node_relu_1 = NodeBuilder("relu_1", RELU)
                            .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .Build(graph);

  ge::NodePtr node_relu_2 = NodeBuilder("relu_2", RELU)
                            .AddInputDesc({1, 2, 2, 6}, FORMAT_NCHW, DT_FLOAT16)
                            .AddOutputDesc({1, 2, 2, 6}, FORMAT_NCHW, DT_FLOAT16)
                            .Build(graph);

  ge::GraphUtils::AddEdge(node->GetOutDataAnchor(0), transdata_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node->GetOutDataAnchor(0), transdata_3->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transdata_1->GetOutDataAnchor(0), cast_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_1->GetOutDataAnchor(0), cast_2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_2->GetOutDataAnchor(0), transdata_2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transdata_2->GetOutDataAnchor(0), node_relu_1->GetInDataAnchor(0));

  ge::GraphUtils::AddEdge(transdata_3->GetOutDataAnchor(0), cast_3->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_3->GetOutDataAnchor(0), reshape_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(reshape_1->GetOutDataAnchor(0), cast_4->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_4->GetOutDataAnchor(0), node_relu_2->GetInDataAnchor(0));
  ge::TransOpWithoutReshapeFusionPass pass;
  uint32_t status = pass.Run(graph);
  EXPECT_EQ(domi::SUCCESS, status);
  EXPECT_EQ(node->GetOutDataNodes().size(), 2);
}

/*
    Node---cast---transdata---transpose---cast---transdata---C

            ||
            \/
    Node---formattransfer---C

*/
TEST_F(UtestTransopWithoutReshapeFusionPass, test_format_transfer)
{
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("graph_test");

  ge::NodePtr node = NodeBuilder("Data4D", DATA)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr cast_fp16_2_fp32 = NodeBuilder("cast_fp16_2_fp32", CAST)
                                .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                                .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT)
                                .Build(graph);

  ge::NodePtr transdata_5d_2_4d = NodeBuilder("transdata_5d_2_4d", TRANSDATA)
                                  .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT)
                                  .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
                                  .Build(graph);

  ge::NodePtr transpose_nchw_2_nhwc = NodeBuilder("transpose_nchw_2_nhwc", TRANSPOSE)
                                      .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
                                      .AddOutputDesc({1, 3, 4, 2}, FORMAT_NHWC, DT_FLOAT)
                                      .Build(graph);
  ge::NodePtr cast_fp32_2_fp16 = NodeBuilder("cast_fp32_2_fp16", CAST)
                                .AddInputDesc({1, 3, 4, 2}, FORMAT_NHWC, DT_FLOAT)
                                .AddOutputDesc({1, 3, 4, 2}, FORMAT_NHWC, DT_FLOAT16)
                                .Build(graph);
  ge::NodePtr transdata_4d_2_5d = NodeBuilder("transdata_4d_2_5d", TRANSDATA)
                            .AddInputDesc({1, 3, 4, 2}, FORMAT_NHWC, DT_FLOAT16)
                            .AddOutputDesc({1, 3, 4, 2, 5}, FORMAT_FRACTAL_Z, DT_FLOAT16)
                            .Build(graph);

  ge::NodePtr node_relu_3 = NodeBuilder("relu_3", RELU)
                            .AddInputDesc({1, 3, 4, 2, 5}, FORMAT_FRACTAL_Z, DT_FLOAT16)
                            .AddOutputDesc({1, 3, 4, 2, 5}, FORMAT_FRACTAL_Z, DT_FLOAT16)
                            .Build(graph);

  // shape different,but format and datatype are equal
  ge::GraphUtils::AddEdge(node->GetOutDataAnchor(0), cast_fp16_2_fp32->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp16_2_fp32->GetOutDataAnchor(0), transdata_5d_2_4d->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transdata_5d_2_4d->GetOutDataAnchor(0), transpose_nchw_2_nhwc->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transpose_nchw_2_nhwc->GetOutDataAnchor(0), cast_fp32_2_fp16->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp32_2_fp16->GetOutDataAnchor(0), transdata_4d_2_5d->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transdata_4d_2_5d->GetOutDataAnchor(0), node_relu_3->GetInDataAnchor(0));

  ge::TransOpWithoutReshapeFusionPass pass;
  uint32_t status = pass.Run(graph);
  EXPECT_EQ(domi::SUCCESS, status);
  EXPECT_EQ(node->GetOutDataNodes().size(), 1);
  if (node->GetOutDataNodes().size() > 0) {
    NodePtr out_node = node->GetOutDataNodes().at(0);
    EXPECT_EQ(out_node->GetOpDesc()->GetType(), TRANSDATA);
  }
}

/*
    Node---cast---transdata---transpose---C

            ||
            \/
    Node---formattransfer---C

*/
TEST_F(UtestTransopWithoutReshapeFusionPass, test_dynamic_format_transfer)
{
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("graph_test");

  ge::NodePtr node = NodeBuilder("Data4D", DATA)
      .AddOutputDesc({-1, -1, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr cast_fp16_2_fp32 = NodeBuilder("cast_fp16_2_fp32", CAST)
                                .AddInputDesc({-1, -1, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                                .AddOutputDesc({-1, -1, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT)
                                .Build(graph);

  ge::NodePtr transdata_5d_2_4d = NodeBuilder("transdata_5d_2_4d", TRANSDATA)
                                  .AddInputDesc({-1, -1, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT)
                                  .AddOutputDesc({-1, -1, 3, 4}, FORMAT_NCHW, DT_FLOAT)
                                  .Build(graph);

  ge::NodePtr transpose_nchw_2_nhwc = NodeBuilder("transpose_nchw_2_nhwc", TRANSPOSE)
                                      .AddInputDesc({-1, -1, 3, 4}, FORMAT_NCHW, DT_FLOAT)
                                      .AddOutputDesc({-1, -1, 4, 2}, FORMAT_NHWC, DT_FLOAT)
                                      .Build(graph);

  // shape different,but format and datatype are equal
  ge::GraphUtils::AddEdge(node->GetOutDataAnchor(0), cast_fp16_2_fp32->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp16_2_fp32->GetOutDataAnchor(0), transdata_5d_2_4d->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transdata_5d_2_4d->GetOutDataAnchor(0), transpose_nchw_2_nhwc->GetInDataAnchor(0));

  ge::TransOpWithoutReshapeFusionPass pass;
  uint32_t status = pass.Run(graph);
  EXPECT_EQ(domi::SUCCESS, status);
  EXPECT_EQ(cast_fp16_2_fp32->GetOutDataNodes().size(), 1);
  if (node->GetOutDataNodes().size() > 0) {
    NodePtr out_node = cast_fp16_2_fp32->GetOutDataNodes().at(0);
    EXPECT_EQ(out_node->GetOpDesc()->GetType(), TRANSDATA);
  }
}

/*
    Node---|---transdata---cast---cast---transdata---A
           |
           |---transdata--cast--reshape--cast---B
           |
           |---cast---transdata---transpose---cast---transdata---C

            ||
            \/
    Node---|---A
           |
           |---transdata--cast--reshape--cast--transdata---B
           |
           |---formattransfer---C

*/
TEST_F(UtestTransopWithoutReshapeFusionPass, test_mixed_reshape_formattransfer)
{
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("graph_test");

  ge::NodePtr node = NodeBuilder("Data4D", DATA)
      .AddOutputDesc({1,2,3,4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr transdata_1 = NodeBuilder("transdata_1", TRANSDATA)
                            .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                            .Build(graph);

  ge::NodePtr transdata_2 = NodeBuilder("transdata_2", TRANSDATA)
                            .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                            .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .Build(graph);

  ge::NodePtr transdata_3 = NodeBuilder("transdata_3", TRANSDATA)
                            .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                            .Build(graph);


  ge::NodePtr cast_1 = NodeBuilder("cast_1", CAST)
                                .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                                .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
                                .Build(graph);

  ge::NodePtr cast_2 = NodeBuilder("cast_2", CAST)
                                .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
                                .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                                .Build(graph);

  ge::NodePtr reshape_1 = NodeBuilder("reshape_1", RESHAPE)
                                .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                                .AddOutputDesc({1, 2, 2, 6}, FORMAT_NCHW, DT_FLOAT)
                                .Build(graph);

  ge::NodePtr cast_3 = NodeBuilder("cast_3", CAST)
                                .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                                .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
                                .Build(graph);

  ge::NodePtr cast_4 = NodeBuilder("cast_4", CAST)
                                .AddInputDesc({1, 2, 2, 6}, FORMAT_NCHW, DT_FLOAT)
                                .AddOutputDesc({1, 2, 2, 6}, FORMAT_NCHW, DT_FLOAT16)
                                .Build(graph);

  ge::NodePtr cast_fp16_2_fp32_5 = NodeBuilder("cast_fp16_2_fp32_5", CAST)
                                .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                                .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT)
                                .Build(graph);

  ge::NodePtr transdata_5d_2_4d_5 = NodeBuilder("transdata_5d_2_4d_5", TRANSDATA)
                            .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT)
                            .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
                            .Build(graph);

  ge::NodePtr transpose_nchw_2_nhwc_1 = NodeBuilder("transpose_nchw_2_nhwc_1", TRANSPOSE)
                            .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
                            .AddOutputDesc({1, 3, 4, 2}, FORMAT_NHWC, DT_FLOAT)
                            .Build(graph);
  ge::NodePtr cast_fp32_2_fp16_6 = NodeBuilder("cast_fp32_2_fp16_6", CAST)
                                .AddInputDesc({1, 3, 4, 2}, FORMAT_NHWC, DT_FLOAT)
                                .AddOutputDesc({1, 3, 4, 2}, FORMAT_NHWC, DT_FLOAT16)
                                .Build(graph);
  ge::NodePtr transdata_4d_2_5d_6 = NodeBuilder("transdata_4d_2_5d_6", TRANSDATA)
                            .AddInputDesc({1, 3, 4, 2}, FORMAT_NHWC, DT_FLOAT16)
                            .AddOutputDesc({1, 3, 4, 2, 5}, FORMAT_FRACTAL_Z, DT_FLOAT16)
                            .Build(graph);
  ge::NodePtr node_relu_1 = NodeBuilder("relu_1", RELU)
                            .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .Build(graph);

  ge::NodePtr node_relu_2 = NodeBuilder("relu_2", RELU)
                            .AddInputDesc({1, 2, 2, 6}, FORMAT_NCHW, DT_FLOAT16)
                            .AddOutputDesc({1, 2, 2, 6}, FORMAT_NCHW, DT_FLOAT16)
                            .Build(graph);
  ge::NodePtr node_relu_3 = NodeBuilder("relu_3", RELU)
                            .AddInputDesc({1, 3, 4, 2, 5}, FORMAT_FRACTAL_Z, DT_FLOAT16)
                            .AddOutputDesc({1, 3, 4, 2, 5}, FORMAT_FRACTAL_Z, DT_FLOAT16)
                            .Build(graph);

  ge::GraphUtils::AddEdge(node->GetOutDataAnchor(0), transdata_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node->GetOutDataAnchor(0), transdata_3->GetInDataAnchor(0));

  ge::GraphUtils::AddEdge(transdata_1->GetOutDataAnchor(0), cast_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_1->GetOutDataAnchor(0), cast_2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_2->GetOutDataAnchor(0), transdata_2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transdata_2->GetOutDataAnchor(0), node_relu_1->GetInDataAnchor(0));

  ge::GraphUtils::AddEdge(transdata_3->GetOutDataAnchor(0), cast_3->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_3->GetOutDataAnchor(0), reshape_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(reshape_1->GetOutDataAnchor(0), cast_4->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_4->GetOutDataAnchor(0), node_relu_2->GetInDataAnchor(0));

  ge::GraphUtils::AddEdge(node->GetOutDataAnchor(0), cast_fp16_2_fp32_5->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp16_2_fp32_5->GetOutDataAnchor(0), transdata_5d_2_4d_5->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transdata_5d_2_4d_5->GetOutDataAnchor(0), transpose_nchw_2_nhwc_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transpose_nchw_2_nhwc_1->GetOutDataAnchor(0), cast_fp32_2_fp16_6->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp32_2_fp16_6->GetOutDataAnchor(0), transdata_4d_2_5d_6->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transdata_4d_2_5d_6->GetOutDataAnchor(0), node_relu_3->GetInDataAnchor(0));

  ge::TransOpWithoutReshapeFusionPass pass;
  uint32_t status = pass.Run(graph);
  EXPECT_EQ(domi::SUCCESS, status);
  EXPECT_EQ(node->GetOutDataNodes().size(), 3);
}

/*
    Node---cast---transpose---transdata---C

            ||
            \/
    Node---cast---formattransfer---C
*/
TEST_F(UtestTransopWithoutReshapeFusionPass, test_cast_formattransfer)
{
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("graph_test");

  ge::NodePtr node = NodeBuilder("Data4D", DATA)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NHWC, DT_FLOAT)
      .Build(graph);

  ge::NodePtr cast_fp32_2_fp16 = NodeBuilder("cast_fp32_2_fp16", CAST)
                                .AddInputDesc({1, 2, 3, 4}, FORMAT_NHWC, DT_FLOAT)
                                .AddOutputDesc({1, 2, 3, 4}, FORMAT_NHWC, DT_FLOAT16)
                                .Build(graph);

  ge::NodePtr transpose_nhwc_2_nchw = NodeBuilder("transpose_nhwc_2_nchw", TRANSPOSE)
                                      .AddInputDesc({1, 2, 3, 4}, FORMAT_NHWC, DT_FLOAT16)
                                      .AddOutputDesc({1, 3, 4, 2}, FORMAT_NCHW, DT_FLOAT16)
                                      .Build(graph);

  ge::NodePtr transdata_4d_2_5d = NodeBuilder("transdata_4d_2_5d", TRANSDATA)
                            .AddInputDesc({1, 3, 4, 2}, FORMAT_NCHW, DT_FLOAT16)
                            .AddOutputDesc({1, 3, 4, 2, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .Build(graph);

  ge::NodePtr node_relu_3 = NodeBuilder("relu_3", RELU)
                            .AddInputDesc({1, 3, 4, 2, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .AddOutputDesc({1, 3, 4, 2, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .Build(graph);

  // shape different,but format and datatype are equal
  ge::GraphUtils::AddEdge(node->GetOutDataAnchor(0), cast_fp32_2_fp16->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp32_2_fp16->GetOutDataAnchor(0), transpose_nhwc_2_nchw->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transpose_nhwc_2_nchw->GetOutDataAnchor(0), transdata_4d_2_5d->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transdata_4d_2_5d->GetOutDataAnchor(0), node_relu_3->GetInDataAnchor(0));

  ge::TransOpWithoutReshapeFusionPass pass;
  uint32_t status = pass.Run(graph);
  EXPECT_EQ(domi::SUCCESS, status);
  EXPECT_EQ(node->GetOutDataNodes().size(), 1);
  if (node->GetOutDataNodes().size() > 0) {
    NodePtr out_node = node->GetOutDataNodes().at(0);
    EXPECT_EQ(out_node->GetOpDesc()->GetType(), CAST);
  }
}

/*
    Node---|---transdata---cast---cast---transdata---A
           |
           |---transdata--cast--reshape--cast---B
           |
           |---cast---transdata---transpose---transdata---C

            ||
            \/
    Node---|---A
           |
           |---transdata--cast--reshape--cast--transdata---B
           |
           |---formattransfer---cast---C

*/
TEST_F(UtestTransopWithoutReshapeFusionPass, test_mixed_formattransfer_and_cast)
{
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("graph_test");

  ge::NodePtr node = NodeBuilder("Data4D", DATA)
      .AddOutputDesc({1,2,3,4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr transdata_1 = NodeBuilder("transdata_1", TRANSDATA)
                            .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                            .Build(graph);

  ge::NodePtr transdata_2 = NodeBuilder("transdata_2", TRANSDATA)
                            .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                            .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .Build(graph);

  ge::NodePtr transdata_3 = NodeBuilder("transdata_3", TRANSDATA)
                            .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                            .Build(graph);

  ge::NodePtr cast_1 = NodeBuilder("cast_1", CAST)
                                .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                                .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
                                .Build(graph);

  ge::NodePtr cast_2 = NodeBuilder("cast_2", CAST)
                                .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
                                .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                                .Build(graph);

  ge::NodePtr reshape_1 = NodeBuilder("reshape_1", RESHAPE)
                                .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                                .AddOutputDesc({1, 2, 2, 6}, FORMAT_NCHW, DT_FLOAT)
                                .Build(graph);

  ge::NodePtr cast_3 = NodeBuilder("cast_3", CAST)
                                .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                                .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
                                .Build(graph);

  ge::NodePtr cast_4 = NodeBuilder("cast_4", CAST)
                                .AddInputDesc({1, 2, 2, 6}, FORMAT_NCHW, DT_FLOAT)
                                .AddOutputDesc({1, 2, 2, 6}, FORMAT_NCHW, DT_FLOAT16)
                                .Build(graph);

  ge::NodePtr cast_fp16_2_fp32_5 = NodeBuilder("cast_fp16_2_fp32_5", CAST)
                                .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                                .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT)
                                .Build(graph);

  ge::NodePtr transdata_5d_2_4d_5 = NodeBuilder("transdata_5d_2_4d_5", TRANSDATA)
                            .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT)
                            .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
                            .Build(graph);

  ge::NodePtr transpose_nchw_2_nhwc_1 = NodeBuilder("transpose_nchw_2_nhwc_1", TRANSPOSE)
                            .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
                            .AddOutputDesc({1, 3, 4, 2}, FORMAT_NHWC, DT_FLOAT)
                            .Build(graph);

  ge::NodePtr transdata_4d_2_5d_6 = NodeBuilder("transdata_4d_2_5d_6", TRANSDATA)
                            .AddInputDesc({1, 3, 4, 2}, FORMAT_NHWC, DT_FLOAT)
                            .AddOutputDesc({1, 3, 4, 2, 5}, FORMAT_FRACTAL_Z, DT_FLOAT)
                            .Build(graph);
  ge::NodePtr node_relu_1 = NodeBuilder("relu_1", RELU)
                            .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .Build(graph);

  ge::NodePtr node_relu_2 = NodeBuilder("relu_2", RELU)
                            .AddInputDesc({1, 2, 2, 6}, FORMAT_NCHW, DT_FLOAT16)
                            .AddOutputDesc({1, 2, 2, 6}, FORMAT_NCHW, DT_FLOAT16)
                            .Build(graph);
  ge::NodePtr node_relu_3 = NodeBuilder("relu_3", RELU)
                            .AddInputDesc({1, 3, 4, 2, 5}, FORMAT_FRACTAL_Z, DT_FLOAT)
                            .AddOutputDesc({1, 3, 4, 2, 5}, FORMAT_FRACTAL_Z, DT_FLOAT)
                            .Build(graph);

  ge::GraphUtils::AddEdge(node->GetOutDataAnchor(0), transdata_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node->GetOutDataAnchor(0), transdata_3->GetInDataAnchor(0));

  ge::GraphUtils::AddEdge(transdata_1->GetOutDataAnchor(0), cast_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_1->GetOutDataAnchor(0), cast_2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_2->GetOutDataAnchor(0), transdata_2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transdata_2->GetOutDataAnchor(0), node_relu_1->GetInDataAnchor(0));

  ge::GraphUtils::AddEdge(transdata_3->GetOutDataAnchor(0), cast_3->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_3->GetOutDataAnchor(0), reshape_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(reshape_1->GetOutDataAnchor(0), cast_4->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_4->GetOutDataAnchor(0), node_relu_2->GetInDataAnchor(0));

  ge::GraphUtils::AddEdge(node->GetOutDataAnchor(0), cast_fp16_2_fp32_5->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp16_2_fp32_5->GetOutDataAnchor(0), transdata_5d_2_4d_5->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transdata_5d_2_4d_5->GetOutDataAnchor(0), transpose_nchw_2_nhwc_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transpose_nchw_2_nhwc_1->GetOutDataAnchor(0), transdata_4d_2_5d_6->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transdata_4d_2_5d_6->GetOutDataAnchor(0), node_relu_3->GetInDataAnchor(0));

  ge::TransOpWithoutReshapeFusionPass pass;
  uint32_t status = pass.Run(graph);
  EXPECT_EQ(domi::SUCCESS, status);
  EXPECT_EQ(node->GetOutDataNodes().size(), 3);
}

/*

    Node---transdata---cast---cast---transdata---A
                                  |
                                  |+++B

            ||
            \/

        Node-----A
            |
            |+++B

*/
TEST_F(UtestTransopWithoutReshapeFusionPass, test_same_input_output_in_ctrl)
{
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("graph_test");
  ge::NodePtr node = NodeBuilder("Data4D", DATA)
      .AddOutputDesc({1,2,3,4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr transdata_5d_2_4d = NodeBuilder("transdata_5d_2_4d", TRANSDATA)
                            .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                            .Build(graph);

  ge::NodePtr transdata_4d_2_5d = NodeBuilder("transdata_4d_2_5d", TRANSDATA)
                            .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                            .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .Build(graph);

  ge::NodePtr cast_fp16_2_fp32 = NodeBuilder("cast_fp16_2_fp32", CAST)
                                .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                                .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
                                .Build(graph);
  ge::NodePtr cast_fp32_2_fp16 = NodeBuilder("cast_fp32_2_fp16", CAST)
                                .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
                                .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                                .Build(graph);
  ge::NodePtr node_relu_1 = NodeBuilder("relu_1", RELU)
                            .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .Build(graph);
  ge::NodePtr node_b = NodeBuilder("softmax_1", SOFTMAX)
                            .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .Build(graph);

  ge::GraphUtils::AddEdge(node->GetOutDataAnchor(0), transdata_5d_2_4d->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transdata_5d_2_4d->GetOutDataAnchor(0), cast_fp16_2_fp32->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp16_2_fp32->GetOutDataAnchor(0), cast_fp32_2_fp16->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp32_2_fp16->GetOutDataAnchor(0), transdata_4d_2_5d->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transdata_4d_2_5d->GetOutDataAnchor(0), node_relu_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp32_2_fp16->GetOutControlAnchor(), node_b->GetInControlAnchor());

  ge::TransOpWithoutReshapeFusionPass pass;
  uint32_t status = pass.Run(graph);
  EXPECT_EQ(domi::SUCCESS, status);
  EXPECT_EQ(node->GetOutControlAnchor()->GetPeerInControlAnchors().size(), 1);
  EXPECT_EQ(node->GetOutControlAnchor()->GetPeerInControlAnchors().at(0)->GetOwnerNode()->GetOpDesc()->GetType(), SOFTMAX);
}

/*
                  B+++|
                      |
    Node---transdata---cast---cast---transdata---A
                                              |
                                              |++C

            ||
            \/
            B+++|
                |
        Node----|-A
            |
            |+++C

*/
TEST_F(UtestTransopWithoutReshapeFusionPass, test_same_input_output_out_data_ctrl)
{
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("graph_test");
  ge::NodePtr node = NodeBuilder("Data4D", DATA)
      .AddOutputDesc({1,2,3,4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr transdata_5d_2_4d = NodeBuilder("transdata_5d_2_4d", TRANSDATA)
                            .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                            .Build(graph);

  ge::NodePtr transdata_4d_2_5d = NodeBuilder("transdata_4d_2_5d", TRANSDATA)
                            .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                            .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .Build(graph);

  ge::NodePtr cast_fp16_2_fp32 = NodeBuilder("cast_fp16_2_fp32", CAST)
                                .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                                .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
                                .Build(graph);
  ge::NodePtr cast_fp32_2_fp16 = NodeBuilder("cast_fp32_2_fp16", CAST)
                                .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
                                .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                                .Build(graph);
  ge::NodePtr node_relu_1 = NodeBuilder("relu_1", RELU)
                            .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .Build(graph);
  ge::NodePtr node_relu_b = NodeBuilder("relu_b", RELU)
                            .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .Build(graph);
  ge::NodePtr node_relu_c = NodeBuilder("relu_c", RELU)
                            .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .Build(graph);

  ge::GraphUtils::AddEdge(node->GetOutDataAnchor(0), transdata_5d_2_4d->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transdata_5d_2_4d->GetOutDataAnchor(0), cast_fp16_2_fp32->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp16_2_fp32->GetOutDataAnchor(0), cast_fp32_2_fp16->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp32_2_fp16->GetOutDataAnchor(0), transdata_4d_2_5d->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transdata_4d_2_5d->GetOutDataAnchor(0), node_relu_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_relu_b->GetOutControlAnchor(), cast_fp16_2_fp32->GetInControlAnchor());
  ge::GraphUtils::AddEdge(transdata_4d_2_5d->GetOutControlAnchor(), node_relu_c->GetInControlAnchor());
  ge::TransOpWithoutReshapeFusionPass pass;
  uint32_t status = pass.Run(graph);
  EXPECT_EQ(domi::SUCCESS, status);
  EXPECT_EQ(node_relu_c->GetInControlNodes().size(), 2);
  EXPECT_EQ(node_relu_c->GetInControlNodes().at(0)->GetOpDesc()->GetType(), DATA);
  EXPECT_EQ(node_relu_b->GetOutControlAnchor()->GetPeerInControlAnchors().size(), 2);
  EXPECT_EQ(node_relu_b->GetOutControlAnchor()->GetPeerInControlAnchors().at(0)->GetOwnerNode()->GetOpDesc()->GetType(), RELU);
}

/*
                                           |++E
                                           |
           |---cast---transpose---transdata---D
           |
           |
           |
    Node---|---cast---transpose----transdata---C
        |      |                           |
        |++++++|                           |+++A
            ||
            \/
                                  |++E
                                  |
         |---cast---formattransfer---D
         |
         | B+++|+++++++++++++++++++++|+++A
         |     |                     |
    Node-|------cast---formattransfer---C
         |      |
         |++++++|
*/
TEST_F(UtestTransopWithoutReshapeFusionPass, test_cast_formattransfer_out_data_in_ctrl)
{
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("graph_test");

  ge::NodePtr node = NodeBuilder("Data4D", DATA)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NHWC, DT_FLOAT)
      .Build(graph);

  ge::NodePtr cast_fp32_2_fp16 = NodeBuilder("cast_fp32_2_fp16", CAST)
                                .AddInputDesc({1, 2, 3, 4}, FORMAT_NHWC, DT_FLOAT)
                                .AddOutputDesc({1, 2, 3, 4}, FORMAT_NHWC, DT_FLOAT16)
                                .Build(graph);
  ge::NodePtr cast_fp32_2_fp16_2 = NodeBuilder("cast_fp32_2_fp16_2", CAST)
                                .AddInputDesc({1, 2, 3, 4}, FORMAT_NHWC, DT_FLOAT)
                                .AddOutputDesc({1, 2, 3, 4}, FORMAT_NHWC, DT_FLOAT16)
                                .Build(graph);

  ge::NodePtr transpose_nhwc_2_nchw = NodeBuilder("transpose_nhwc_2_nchw", TRANSPOSE)
                                      .AddInputDesc({1, 2, 3, 4}, FORMAT_NHWC, DT_FLOAT16)
                                      .AddOutputDesc({1, 3, 4, 2}, FORMAT_NCHW, DT_FLOAT16)
                                      .Build(graph);

  ge::NodePtr transpose_nhwc_2_nchw_2 = NodeBuilder("transpose_nhwc_2_nchw_2", TRANSPOSE)
                                      .AddInputDesc({1, 2, 3, 4}, FORMAT_NHWC, DT_FLOAT16)
                                      .AddOutputDesc({1, 3, 4, 2}, FORMAT_NCHW, DT_FLOAT16)
                                      .Build(graph);

  ge::NodePtr transdata_4d_2_5d = NodeBuilder("transdata_4d_2_5d", TRANSDATA)
                            .AddInputDesc({1, 3, 4, 2}, FORMAT_NCHW, DT_FLOAT16)
                            .AddOutputDesc({1, 3, 4, 2, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .Build(graph);
  ge::NodePtr transdata_4d_2_5d_2 = NodeBuilder("transdata_4d_2_5d_2", TRANSDATA)
                            .AddInputDesc({1, 3, 4, 2}, FORMAT_NCHW, DT_FLOAT16)
                            .AddOutputDesc({1, 3, 4, 2, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .Build(graph);

  ge::NodePtr node_relu_1 = NodeBuilder("relu_1", RELU)
                            .AddInputDesc({1, 3, 4, 2}, FORMAT_NCHW, DT_FLOAT16)
                            .AddOutputDesc({1, 3, 4, 2}, FORMAT_NCHW, DT_FLOAT16)
                            .Build(graph);

  ge::NodePtr node_relu_2 = NodeBuilder("relu_2", RELU)
                            .AddInputDesc({1, 3, 4, 2}, FORMAT_NCHW, DT_FLOAT16)
                            .AddOutputDesc({1, 3, 4, 2}, FORMAT_NCHW, DT_FLOAT16)
                            .Build(graph);

  ge::NodePtr node_relu_3 = NodeBuilder("relu_3", RELU)
                            .AddInputDesc({1, 3, 4, 2, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .AddOutputDesc({1, 3, 4, 2, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .Build(graph);

  ge::NodePtr node_relu_d = NodeBuilder("relu_d", RELU)
                            .AddInputDesc({1, 3, 4, 2, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .AddOutputDesc({1, 3, 4, 2, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .Build(graph);

 ge::NodePtr node_relu_a = NodeBuilder("relu_a", RELU)
                            .AddInputDesc({1, 3, 4, 2, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .AddOutputDesc({1, 3, 4, 2, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .Build(graph);

  // shape different,but format and datatype are equal
  ge::GraphUtils::AddEdge(node->GetOutDataAnchor(0), cast_fp32_2_fp16->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp32_2_fp16->GetOutDataAnchor(0), transpose_nhwc_2_nchw->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp32_2_fp16->GetOutControlAnchor(), transpose_nhwc_2_nchw->GetInControlAnchor());
  ge::GraphUtils::AddEdge(transpose_nhwc_2_nchw->GetOutDataAnchor(0), transdata_4d_2_5d->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transdata_4d_2_5d->GetOutDataAnchor(0), node_relu_3->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_relu_2->GetOutControlAnchor(), transdata_4d_2_5d->GetInControlAnchor());
  ge::GraphUtils::AddEdge(transdata_4d_2_5d->GetOutControlAnchor(), node_relu_1->GetInControlAnchor());

  ge::GraphUtils::AddEdge(node->GetOutDataAnchor(0), cast_fp32_2_fp16_2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp32_2_fp16_2->GetOutDataAnchor(0), transpose_nhwc_2_nchw_2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transpose_nhwc_2_nchw_2->GetOutDataAnchor(0), transdata_4d_2_5d_2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transdata_4d_2_5d_2->GetOutDataAnchor(0), node_relu_d->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transdata_4d_2_5d_2->GetOutControlAnchor(), node_relu_a->GetInControlAnchor());

  ge::TransOpWithoutReshapeFusionPass pass;
  uint32_t status = pass.Run(graph);
  EXPECT_EQ(domi::SUCCESS, status);
  EXPECT_EQ(node->GetOutDataNodes().size(), 2);
  EXPECT_EQ(node_relu_d->GetInDataNodes().size(), 1);
  if (node_relu_d->GetInDataNodes().size() > 0) {
    NodePtr in_node = node_relu_d->GetInDataNodes().at(0);
    EXPECT_EQ(in_node->GetOpDesc()->GetType(), TRANSDATA);
  }
  EXPECT_EQ(node_relu_a->GetInControlAnchor()->GetPeerOutControlAnchors().size(), 2);
  NodePtr peer_out_data_node = node_relu_a->GetInControlAnchor()->GetPeerOutControlAnchors().at(0)->GetOwnerNode();
  EXPECT_EQ(peer_out_data_node->GetOpDesc()->GetType(), TRANSDATA);
  auto format_transfer_output0 = peer_out_data_node->GetOpDesc()->GetOutputDescPtr(0);
  EXPECT_EQ(format_transfer_output0->GetFormat(), FORMAT_NC1HWC0);
  EXPECT_EQ(format_transfer_output0->GetDataType(), DT_FLOAT16);
  EXPECT_EQ(format_transfer_output0->GetShape().GetDim(0), 1);
  EXPECT_EQ(format_transfer_output0->GetShape().GetDim(1), 3);
  EXPECT_EQ(format_transfer_output0->GetShape().GetDim(2), 4);
  EXPECT_EQ(format_transfer_output0->GetShape().GetDim(3), 2);
  EXPECT_EQ(format_transfer_output0->GetShape().GetDim(4), 5);
}

/*
                          B+++|
                              |
    Node---cast---transpose----transdata---C
              |   |                     |
              |+++|                     |+++A
            ||
            \/
      B+++|+++++++++++++++++++++|+++A
          |                     |
    Node---cast---formattransfer---C
       |   |
       |+++|
*/
TEST_F(UtestTransopWithoutReshapeFusionPass, test_cast_formattransfer_out_ctrl_in_data)
{
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("graph_test");

  ge::NodePtr node = NodeBuilder("Data4D", DATA)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NHWC, DT_FLOAT)
      .Build(graph);

  ge::NodePtr cast_fp32_2_fp16 = NodeBuilder("cast_fp32_2_fp16", CAST)
                                .AddInputDesc({1, 2, 3, 4}, FORMAT_NHWC, DT_FLOAT)
                                .AddOutputDesc({1, 2, 3, 4}, FORMAT_NHWC, DT_FLOAT16)
                                .Build(graph);

  ge::NodePtr transpose_nhwc_2_nchw = NodeBuilder("transpose_nhwc_2_nchw", TRANSPOSE)
                                      .AddInputDesc({1, 2, 3, 4}, FORMAT_NHWC, DT_FLOAT16)
                                      .AddOutputDesc({1, 3, 4, 2}, FORMAT_NCHW, DT_FLOAT16)
                                      .Build(graph);

  ge::NodePtr transdata_4d_2_5d = NodeBuilder("transdata_4d_2_5d", TRANSDATA)
                            .AddInputDesc({1, 3, 4, 2}, FORMAT_NCHW, DT_FLOAT16)
                            .AddOutputDesc({1, 3, 4, 2, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .Build(graph);

  ge::NodePtr node_relu_1 = NodeBuilder("relu_1", RELU)
                            .AddInputDesc({1, 3, 4, 2}, FORMAT_NCHW, DT_FLOAT16)
                            .AddOutputDesc({1, 3, 4, 2}, FORMAT_NCHW, DT_FLOAT16)
                            .Build(graph);

  ge::NodePtr node_relu_2 = NodeBuilder("relu_2", RELU)
                            .AddInputDesc({1, 3, 4, 2}, FORMAT_NCHW, DT_FLOAT16)
                            .AddOutputDesc({1, 3, 4, 2}, FORMAT_NCHW, DT_FLOAT16)
                            .Build(graph);

  ge::NodePtr node_relu_3 = NodeBuilder("relu_3", RELU)
                            .AddInputDesc({1, 3, 4, 2, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .AddOutputDesc({1, 3, 4, 2, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .Build(graph);

  // shape different,but format and datatype are equal
  ge::GraphUtils::AddEdge(node->GetOutDataAnchor(0), cast_fp32_2_fp16->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp32_2_fp16->GetOutDataAnchor(0), transpose_nhwc_2_nchw->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp32_2_fp16->GetOutControlAnchor(), transpose_nhwc_2_nchw->GetInControlAnchor());
  ge::GraphUtils::AddEdge(transpose_nhwc_2_nchw->GetOutDataAnchor(0), transdata_4d_2_5d->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transdata_4d_2_5d->GetOutDataAnchor(0), node_relu_3->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_relu_2->GetOutControlAnchor(), transdata_4d_2_5d->GetInControlAnchor());
  ge::GraphUtils::AddEdge(transdata_4d_2_5d->GetOutControlAnchor(), node_relu_1->GetInDataAnchor(0));

  ge::TransOpWithoutReshapeFusionPass pass;
  uint32_t status = pass.Run(graph);
  EXPECT_EQ(domi::SUCCESS, status);
  EXPECT_EQ(node->GetOutDataNodes().size(), 1);
  if (node->GetOutDataNodes().size() > 0) {
    NodePtr out_node = node->GetOutDataNodes().at(0);
    EXPECT_EQ(out_node->GetOpDesc()->GetType(), CAST);
  }
  EXPECT_EQ(node_relu_3->GetInDataNodes().size(), 1);
  auto peer_in_data_node = node_relu_3->GetInDataNodes().at(0);
  auto format_transfer_output0 = peer_in_data_node->GetOpDesc()->GetOutputDescPtr(0);
  EXPECT_EQ(peer_in_data_node->GetOpDesc()->GetType(), TRANSDATA);
  EXPECT_EQ(format_transfer_output0->GetFormat(), FORMAT_NC1HWC0);
  EXPECT_EQ(format_transfer_output0->GetDataType(), DT_FLOAT16);
  EXPECT_EQ(format_transfer_output0->GetShape().GetDim(0), 1);
  EXPECT_EQ(format_transfer_output0->GetShape().GetDim(1), 3);
  EXPECT_EQ(format_transfer_output0->GetShape().GetDim(2), 4);
  EXPECT_EQ(format_transfer_output0->GetShape().GetDim(3), 2);
  EXPECT_EQ(format_transfer_output0->GetShape().GetDim(4), 5);
}

/*
                          B+++|
                              |
    Node---cast---transpose----transdata---C
              |   |                     |
              |+++|                     |+++A
            ||
            \/
      B+++|+++++++++++++++++++++|+++A
          |                     |
    Node---cast---formattransfer---C
       |   |
       |+++|
*/
TEST_F(UtestTransopWithoutReshapeFusionPass, test_cast_formattransfer_out_and_in_control)
{
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("graph_test");

  ge::NodePtr node = NodeBuilder("Data4D", DATA)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NHWC, DT_FLOAT)
      .Build(graph);

  ge::NodePtr cast_fp32_2_fp16 = NodeBuilder("cast_fp32_2_fp16", CAST)
                                .AddInputDesc({1, 2, 3, 4}, FORMAT_NHWC, DT_FLOAT)
                                .AddOutputDesc({1, 2, 3, 4}, FORMAT_NHWC, DT_FLOAT16)
                                .Build(graph);

  ge::NodePtr transpose_nhwc_2_nchw = NodeBuilder("transpose_nhwc_2_nchw", TRANSPOSE)
                                      .AddInputDesc({1, 2, 3, 4}, FORMAT_NHWC, DT_FLOAT16)
                                      .AddOutputDesc({1, 3, 4, 2}, FORMAT_NCHW, DT_FLOAT16)
                                      .Build(graph);

  ge::NodePtr transdata_4d_2_5d = NodeBuilder("transdata_4d_2_5d", TRANSDATA)
                            .AddInputDesc({1, 3, 4, 2}, FORMAT_NCHW, DT_FLOAT16)
                            .AddOutputDesc({1, 3, 4, 2, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .Build(graph);

  ge::NodePtr node_relu_1 = NodeBuilder("relu_1", RELU)
                            .AddInputDesc({1, 3, 4, 2}, FORMAT_NCHW, DT_FLOAT16)
                            .AddOutputDesc({1, 3, 4, 2}, FORMAT_NCHW, DT_FLOAT16)
                            .Build(graph);

  ge::NodePtr node_relu_2 = NodeBuilder("relu_2", RELU)
                            .AddInputDesc({1, 3, 4, 2}, FORMAT_NCHW, DT_FLOAT16)
                            .AddOutputDesc({1, 3, 4, 2}, FORMAT_NCHW, DT_FLOAT16)
                            .Build(graph);

  ge::NodePtr node_relu_3 = NodeBuilder("relu_3", RELU)
                            .AddInputDesc({1, 3, 4, 2, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .AddOutputDesc({1, 3, 4, 2, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .Build(graph);

  // shape different,but format and datatype are equal
  ge::GraphUtils::AddEdge(node->GetOutDataAnchor(0), cast_fp32_2_fp16->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp32_2_fp16->GetOutDataAnchor(0), transpose_nhwc_2_nchw->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp32_2_fp16->GetOutControlAnchor(), transpose_nhwc_2_nchw->GetInControlAnchor());
  ge::GraphUtils::AddEdge(transpose_nhwc_2_nchw->GetOutDataAnchor(0), transdata_4d_2_5d->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transdata_4d_2_5d->GetOutDataAnchor(0), node_relu_3->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_relu_2->GetOutControlAnchor(), transdata_4d_2_5d->GetInControlAnchor());
  ge::GraphUtils::AddEdge(transdata_4d_2_5d->GetOutControlAnchor(), node_relu_1->GetInControlAnchor());

  ge::TransOpWithoutReshapeFusionPass pass;
  uint32_t status = pass.Run(graph);
  EXPECT_EQ(domi::SUCCESS, status);
  EXPECT_EQ(node->GetOutDataNodes().size(), 1);
  if (node->GetOutDataNodes().size() > 0) {
    NodePtr out_node = node->GetOutDataNodes().at(0);
    EXPECT_EQ(out_node->GetOpDesc()->GetType(), CAST);
    auto cast_output0_desc = out_node->GetOpDesc()->GetOutputDescPtr(0);
    EXPECT_EQ(cast_output0_desc->GetFormat(), FORMAT_NHWC);
    EXPECT_EQ(cast_output0_desc->GetDataType(), DT_FLOAT16);
    EXPECT_EQ(cast_output0_desc->GetShape().GetDim(0), 1);
    EXPECT_EQ(cast_output0_desc->GetShape().GetDim(1), 2);
    EXPECT_EQ(cast_output0_desc->GetShape().GetDim(2), 3);
    EXPECT_EQ(cast_output0_desc->GetShape().GetDim(3), 4);
    auto cast_input0_desc = out_node->GetOpDesc()->GetInputDescPtr(0);
    EXPECT_EQ(cast_input0_desc->GetFormat(), FORMAT_NHWC);
    EXPECT_EQ(cast_input0_desc->GetDataType(), DT_FLOAT);
    EXPECT_EQ(cast_input0_desc->GetShape().GetDim(0), 1);
    EXPECT_EQ(cast_input0_desc->GetShape().GetDim(1), 2);
    EXPECT_EQ(cast_input0_desc->GetShape().GetDim(2), 3);
    EXPECT_EQ(cast_input0_desc->GetShape().GetDim(3), 4);

    NodePtr format_transfer_node = out_node->GetOutDataNodes().at(0);
    EXPECT_EQ(format_transfer_node->GetOpDesc()->GetType(), TRANSDATA);
    auto format_transfer_output0_desc = format_transfer_node->GetOpDesc()->GetOutputDescPtr(0);
    EXPECT_EQ(format_transfer_output0_desc->GetFormat(), FORMAT_NC1HWC0);
    EXPECT_EQ(format_transfer_output0_desc->GetDataType(), DT_FLOAT16);
    EXPECT_EQ(format_transfer_output0_desc->GetShape().GetDim(0), 1);
    EXPECT_EQ(format_transfer_output0_desc->GetShape().GetDim(1), 3);
    EXPECT_EQ(format_transfer_output0_desc->GetShape().GetDim(2), 4);
    EXPECT_EQ(format_transfer_output0_desc->GetShape().GetDim(3), 2);
    EXPECT_EQ(format_transfer_output0_desc->GetShape().GetDim(4), 5);
    auto format_transfer_input0_desc = format_transfer_node->GetOpDesc()->GetInputDescPtr(0);
    EXPECT_EQ(format_transfer_input0_desc->GetFormat(), FORMAT_NHWC);
    EXPECT_EQ(format_transfer_input0_desc->GetDataType(), DT_FLOAT16);
    EXPECT_EQ(format_transfer_input0_desc->GetShape().GetDim(0), 1);
    EXPECT_EQ(format_transfer_input0_desc->GetShape().GetDim(1), 2);
    EXPECT_EQ(format_transfer_input0_desc->GetShape().GetDim(2), 3);
    EXPECT_EQ(format_transfer_input0_desc->GetShape().GetDim(3), 4);
  }
}

/*
                          B+++|
                              |
    Node---cast---transpose----transdata---C
              |   |
              |+++|
            ||
            \/
      B+++|
          |
    Node---cast---formattransfer---C
       |   |
       |+++|
*/
TEST_F(UtestTransopWithoutReshapeFusionPass, test_cast_formattransfer_in_control1)
{
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("graph_test");

  ge::NodePtr node = NodeBuilder("Data4D", DATA)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NHWC, DT_FLOAT)
      .Build(graph);

  ge::NodePtr cast_fp32_2_fp16 = NodeBuilder("cast_fp32_2_fp16", CAST)
                                .AddInputDesc({1, 2, 3, 4}, FORMAT_NHWC, DT_FLOAT)
                                .AddOutputDesc({1, 2, 3, 4}, FORMAT_NHWC, DT_FLOAT16)
                                .Build(graph);

  ge::NodePtr transpose_nhwc_2_nchw = NodeBuilder("transpose_nhwc_2_nchw", TRANSPOSE)
                                      .AddInputDesc({1, 2, 3, 4}, FORMAT_NHWC, DT_FLOAT16)
                                      .AddOutputDesc({1, 3, 4, 2}, FORMAT_NCHW, DT_FLOAT16)
                                      .Build(graph);

  ge::NodePtr transdata_4d_2_5d = NodeBuilder("transdata_4d_2_5d", TRANSDATA)
                            .AddInputDesc({1, 3, 4, 2}, FORMAT_NCHW, DT_FLOAT16)
                            .AddOutputDesc({1, 3, 4, 2, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .Build(graph);

  ge::NodePtr node_relu_2 = NodeBuilder("relu_2", RELU)
                            .AddInputDesc({1, 3, 4, 2}, FORMAT_NCHW, DT_FLOAT16)
                            .AddOutputDesc({1, 3, 4, 2}, FORMAT_NCHW, DT_FLOAT16)
                            .Build(graph);

  ge::NodePtr node_relu_3 = NodeBuilder("relu_3", RELU)
                            .AddInputDesc({1, 3, 4, 2, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .AddOutputDesc({1, 3, 4, 2, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .Build(graph);

  // shape different,but format and datatype are equal
  ge::GraphUtils::AddEdge(node->GetOutDataAnchor(0), cast_fp32_2_fp16->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp32_2_fp16->GetOutDataAnchor(0), transpose_nhwc_2_nchw->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp32_2_fp16->GetOutControlAnchor(), transpose_nhwc_2_nchw->GetInControlAnchor());
  ge::GraphUtils::AddEdge(transpose_nhwc_2_nchw->GetOutDataAnchor(0), transdata_4d_2_5d->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transdata_4d_2_5d->GetOutDataAnchor(0), node_relu_3->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_relu_2->GetOutControlAnchor(), transdata_4d_2_5d->GetInControlAnchor());

  ge::TransOpWithoutReshapeFusionPass pass;
  uint32_t status = pass.Run(graph);
  EXPECT_EQ(domi::SUCCESS, status);
  EXPECT_EQ(node->GetOutDataNodes().size(), 1);
  if (node->GetOutDataNodes().size() > 0) {
    NodePtr out_node = node->GetOutDataNodes().at(0);
    EXPECT_EQ(out_node->GetOpDesc()->GetType(), CAST);
  }
}

/*
                          B+++|
                              |
    Node---cast---transpose----transdata---C

            ||
            \/
      B+++|
          |
    Node---cast---formattransfer-|--C
*/
TEST_F(UtestTransopWithoutReshapeFusionPass, test_cast_formattransfer_in_control2)
{
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("graph_test");

  ge::NodePtr node = NodeBuilder("Data4D", DATA)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NHWC, DT_FLOAT)
      .Build(graph);

  ge::NodePtr cast_fp32_2_fp16 = NodeBuilder("cast_fp32_2_fp16", CAST)
                                .AddInputDesc({1, 2, 3, 4}, FORMAT_NHWC, DT_FLOAT)
                                .AddOutputDesc({1, 2, 3, 4}, FORMAT_NHWC, DT_FLOAT16)
                                .Build(graph);

  ge::NodePtr transpose_nhwc_2_nchw = NodeBuilder("transpose_nhwc_2_nchw", TRANSPOSE)
                                      .AddInputDesc({1, 2, 3, 4}, FORMAT_NHWC, DT_FLOAT16)
                                      .AddOutputDesc({1, 3, 4, 2}, FORMAT_NCHW, DT_FLOAT16)
                                      .Build(graph);

  ge::NodePtr transdata_4d_2_5d = NodeBuilder("transdata_4d_2_5d", TRANSDATA)
                            .AddInputDesc({1, 3, 4, 2}, FORMAT_NCHW, DT_FLOAT16)
                            .AddOutputDesc({1, 3, 4, 2, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .Build(graph);

  ge::NodePtr node_relu_2 = NodeBuilder("relu_2", RELU)
                            .AddInputDesc({1, 3, 4, 2}, FORMAT_NCHW, DT_FLOAT16)
                            .AddOutputDesc({1, 3, 4, 2}, FORMAT_NCHW, DT_FLOAT16)
                            .Build(graph);

  ge::NodePtr node_relu_3 = NodeBuilder("relu_3", RELU)
                            .AddInputDesc({1, 3, 4, 2, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .AddOutputDesc({1, 3, 4, 2, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .Build(graph);

  // shape different,but format and datatype are equal
  ge::GraphUtils::AddEdge(node->GetOutDataAnchor(0), cast_fp32_2_fp16->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp32_2_fp16->GetOutDataAnchor(0), transpose_nhwc_2_nchw->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transpose_nhwc_2_nchw->GetOutDataAnchor(0), transdata_4d_2_5d->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transdata_4d_2_5d->GetOutDataAnchor(0), node_relu_3->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_relu_2->GetOutControlAnchor(), transdata_4d_2_5d->GetInControlAnchor());

  ge::TransOpWithoutReshapeFusionPass pass;
  uint32_t status = pass.Run(graph);
  EXPECT_EQ(domi::SUCCESS, status);
  EXPECT_EQ(node->GetOutDataNodes().size(), 1);
  if (node->GetOutDataNodes().size() > 0) {
    NodePtr out_node = node->GetOutDataNodes().at(0);
    EXPECT_EQ(out_node->GetOpDesc()->GetType(), CAST);
  }
}

/*
                            |+++B
                            |
    Node---cast---transpose-|---transdata---C

            ||
            \/
                                 |+++B
                                 |
    Node---cast---formattransfer-|--C
*/
TEST_F(UtestTransopWithoutReshapeFusionPass, test_cast_formattransfer_out_control)
{
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("graph_test");

  ge::NodePtr node = NodeBuilder("Data4D", DATA)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NHWC, DT_FLOAT)
      .Build(graph);

  ge::NodePtr cast_fp32_2_fp16 = NodeBuilder("cast_fp32_2_fp16", CAST)
                                .AddInputDesc({1, 2, 3, 4}, FORMAT_NHWC, DT_FLOAT)
                                .AddOutputDesc({1, 2, 3, 4}, FORMAT_NHWC, DT_FLOAT16)
                                .Build(graph);

  ge::NodePtr transpose_nhwc_2_nchw = NodeBuilder("transpose_nhwc_2_nchw", TRANSPOSE)
                                      .AddInputDesc({1, 2, 3, 4}, FORMAT_NHWC, DT_FLOAT16)
                                      .AddOutputDesc({1, 3, 4, 2}, FORMAT_NCHW, DT_FLOAT16)
                                      .Build(graph);

  ge::NodePtr transdata_4d_2_5d = NodeBuilder("transdata_4d_2_5d", TRANSDATA)
                            .AddInputDesc({1, 3, 4, 2}, FORMAT_NCHW, DT_FLOAT16)
                            .AddOutputDesc({1, 3, 4, 2, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .Build(graph);

  ge::NodePtr node_relu_2 = NodeBuilder("relu_2", RELU)
                            .AddInputDesc({1, 3, 4, 2}, FORMAT_NCHW, DT_FLOAT16)
                            .AddOutputDesc({1, 3, 4, 2}, FORMAT_NCHW, DT_FLOAT16)
                            .Build(graph);

  ge::NodePtr node_relu_3 = NodeBuilder("relu_3", RELU)
                            .AddInputDesc({1, 3, 4, 2, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .AddOutputDesc({1, 3, 4, 2, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .Build(graph);

  // shape different,but format and datatype are equal
  ge::GraphUtils::AddEdge(node->GetOutDataAnchor(0), cast_fp32_2_fp16->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp32_2_fp16->GetOutDataAnchor(0), transpose_nhwc_2_nchw->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transpose_nhwc_2_nchw->GetOutDataAnchor(0), transdata_4d_2_5d->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transdata_4d_2_5d->GetOutDataAnchor(0), node_relu_3->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transpose_nhwc_2_nchw->GetOutControlAnchor(), node_relu_2->GetInControlAnchor());

  ge::TransOpWithoutReshapeFusionPass pass;
  uint32_t status = pass.Run(graph);
  EXPECT_EQ(domi::SUCCESS, status);
  EXPECT_EQ(node->GetOutDataNodes().size(), 1);
  if (node->GetOutDataNodes().size() > 0) {
    NodePtr out_node = node->GetOutDataNodes().at(0);
    EXPECT_EQ(out_node->GetOpDesc()->GetType(), CAST);
  }
}

/*

    Node---transdata---cast---cast---transdata---A
                                              |+|


            ||
            \/

        Node-----A
            |   |
            |+++|


*/
TEST_F(UtestTransopWithoutReshapeFusionPass, test_output_data_in_ctrl)
{
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("graph_test");
  ge::NodePtr node = NodeBuilder("Data4D", DATA)
      .AddOutputDesc({1,2,3,4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
      .Build(graph);

  ge::NodePtr transdata_5d_2_4d = NodeBuilder("transdata_5d_2_4d", TRANSDATA)
                            .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                            .Build(graph);

  ge::NodePtr transdata_4d_2_5d = NodeBuilder("transdata_4d_2_5d", TRANSDATA)
                            .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                            .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .Build(graph);

  ge::NodePtr cast_fp16_2_fp32 = NodeBuilder("cast_fp16_2_fp32", CAST)
                                .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                                .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
                                .Build(graph);
  ge::NodePtr cast_fp32_2_fp16 = NodeBuilder("cast_fp32_2_fp16", CAST)
                                .AddInputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT)
                                .AddOutputDesc({1, 2, 3, 4}, FORMAT_NCHW, DT_FLOAT16)
                                .Build(graph);
  ge::NodePtr node_relu_1 = NodeBuilder("relu_1", RELU)
                            .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .Build(graph);

  ge::GraphUtils::AddEdge(node->GetOutDataAnchor(0), transdata_5d_2_4d->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transdata_5d_2_4d->GetOutDataAnchor(0), cast_fp16_2_fp32->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp16_2_fp32->GetOutDataAnchor(0), cast_fp32_2_fp16->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp32_2_fp16->GetOutDataAnchor(0), transdata_4d_2_5d->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transdata_4d_2_5d->GetOutDataAnchor(0), node_relu_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transdata_4d_2_5d->GetOutControlAnchor(), node_relu_1->GetInControlAnchor());

  ge::TransOpWithoutReshapeFusionPass pass;
  uint32_t status = pass.Run(graph);
  EXPECT_EQ(domi::SUCCESS, status);
  EXPECT_EQ(node_relu_1->GetInControlAnchor()->GetPeerOutControlAnchors().size(), 1);
  auto peer_node = node_relu_1->GetInControlAnchor()->GetPeerOutControlAnchors().at(0)->GetOwnerNode();
  EXPECT_EQ(peer_node->GetName(), "Data4D");
  EXPECT_EQ(node->GetOutControlAnchor()->GetPeerInControlAnchors().size(), 1);
  EXPECT_EQ(node->GetOutControlAnchor()->GetPeerInControlAnchors().at(0)->GetOwnerNode()->GetOpDesc()->GetType(), RELU);
}

TEST_F(UtestTransopWithoutReshapeFusionPass, test_op_accuracy_ability_check_succ) {
  OpDescPtr cast_op = make_shared<OpDesc>("cast_test", "Cast");
  if (cast_op == nullptr) {
    return;
  }

  TransOpWithoutReshapeFusionPass pass;
  bool is_support = false;
  auto ret = TransOpCreator::CheckAccuracySupported(cast_op, is_support);
  EXPECT_EQ(SUCCESS, ret);
  EXPECT_TRUE(is_support);
}

/*
    Node---cast---transpose----transdata---C
            ||
            \/
    Node---cast---transpose----transdata---C
*/
TEST_F(UtestTransopWithoutReshapeFusionPass, test_fusion_fail_shape_uncontinuous)
{
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("graph_test");

  ge::NodePtr node = NodeBuilder("Data4D", DATA)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NHWC, DT_FLOAT)
      .Build(graph);

  ge::NodePtr cast_fp32_2_fp16 = NodeBuilder("cast_fp32_2_fp16", CAST)
                                .AddInputDesc({1, 2, 3, 4}, FORMAT_NHWC, DT_FLOAT)
                                .AddOutputDesc({1, 2, 3, 4}, FORMAT_NHWC, DT_FLOAT16)
                                .Build(graph);

  ge::NodePtr transpose_nhwc_2_nchw = NodeBuilder("transpose_nhwc_2_nchw", TRANSPOSE)
                                      .AddInputDesc({1, 2, 3, 4}, FORMAT_NHWC, DT_FLOAT16)
                                      .AddOutputDesc({1, 3, 4, 2}, FORMAT_NCHW, DT_FLOAT16)
                                      .Build(graph);

  ge::NodePtr transdata_4d_2_5d = NodeBuilder("transdata_4d_2_5d", TRANSDATA)
                            .AddInputDesc({1, 3, 4, 2}, FORMAT_NCHW, DT_FLOAT16)
                            .AddOutputDesc({1, 3, 4, 2, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .Build(graph);

  ge::NodePtr node_relu_1 = NodeBuilder("relu_1", RELU)
                            .AddInputDesc({1, 3, 4, 2}, FORMAT_NCHW, DT_FLOAT16)
                            .AddOutputDesc({1, 3, 4, 2}, FORMAT_NCHW, DT_FLOAT16)
                            .Build(graph);


  // shape different,but format and datatype are equal
  ge::GraphUtils::AddEdge(node->GetOutDataAnchor(0), cast_fp32_2_fp16->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp32_2_fp16->GetOutDataAnchor(0), transpose_nhwc_2_nchw->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp32_2_fp16->GetOutControlAnchor(), transpose_nhwc_2_nchw->GetInControlAnchor());
  ge::GraphUtils::AddEdge(transpose_nhwc_2_nchw->GetOutDataAnchor(0), transdata_4d_2_5d->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transdata_4d_2_5d->GetOutDataAnchor(0), node_relu_1->GetInDataAnchor(0));

  ge::TransOpWithoutReshapeFusionPass pass;
  uint32_t status = pass.Run(graph);
  EXPECT_EQ(domi::SUCCESS, status);
  EXPECT_EQ(node->GetOutDataNodes().size(), 1);
  if (node->GetOutDataNodes().size() > 0) {
    NodePtr out_node = node->GetOutDataNodes().at(0);
    EXPECT_EQ(out_node->GetOpDesc()->GetType(), CAST);
  }
}

/*
    Node---cast---transpose----transdata---C
            ||
            \/
    Node---cast---transpose----transdata---C
*/
TEST_F(UtestTransopWithoutReshapeFusionPass, test_fusion_fail_format_unsupport)
{
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("graph_test");

  ge::NodePtr node = NodeBuilder("Data4D", DATA)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_FRACTAL_Z_C04, DT_FLOAT)
      .Build(graph);

  ge::NodePtr transdata_5d_2_4d = NodeBuilder("transdata_5d_2_4d", TRANSDATA)
                           .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_FRACTAL_Z_C04, DT_FLOAT16)
                           .AddOutputDesc({1, 3, 4, 2}, FORMAT_NHWC, DT_FLOAT16)
                           .Build(graph);

  ge::NodePtr cast_fp32_2_fp16 = NodeBuilder("cast_fp32_2_fp16", CAST)
                                .AddInputDesc({1, 2, 3, 4}, FORMAT_NHWC, DT_FLOAT)
                                .AddOutputDesc({1, 2, 3, 4}, FORMAT_NHWC, DT_FLOAT16)
                                .Build(graph);

  ge::NodePtr transpose_nhwc_2_nchw = NodeBuilder("transpose_nhwc_2_nchw", TRANSPOSE)
                                      .AddInputDesc({1, 2, 3, 4}, FORMAT_NHWC, DT_FLOAT16)
                                      .AddOutputDesc({1, 3, 4, 2}, FORMAT_NCHW, DT_FLOAT16)
                                      .Build(graph);

  ge::NodePtr node_relu_1 = NodeBuilder("relu_1", RELU)
                            .AddInputDesc({1, 3, 4, 2, 5}, FORMAT_NCHW, DT_FLOAT16)
                            .AddOutputDesc({1, 3, 4, 2, 5}, FORMAT_NCHW, DT_FLOAT16)
                            .Build(graph);


  // shape different,but format and datatype are equal
  ge::GraphUtils::AddEdge(node->GetOutDataAnchor(0), transdata_5d_2_4d->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transdata_5d_2_4d->GetOutDataAnchor(0), cast_fp32_2_fp16->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp32_2_fp16->GetOutDataAnchor(0), transpose_nhwc_2_nchw->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp32_2_fp16->GetOutControlAnchor(), transpose_nhwc_2_nchw->GetInControlAnchor());
  ge::GraphUtils::AddEdge(transpose_nhwc_2_nchw->GetOutDataAnchor(0), node_relu_1->GetInDataAnchor(0));

  EXPECT_EQ(graph->GetDirectNodesSize(), 5);
  ge::TransOpWithoutReshapeFusionPass pass;
  uint32_t status = pass.Run(graph);
  EXPECT_EQ(domi::SUCCESS, status);
  EXPECT_EQ(graph->GetDirectNodesSize(), 5);
}
/*
    Node---cast---transpose----transdata---C
            ||
            \/
    Node---cast---transpose----transdata---C
*/
TEST_F(UtestTransopWithoutReshapeFusionPass, test_fusion_fail_format_nd_unsupport)
{
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("graph_test");

  ge::NodePtr node = NodeBuilder("Data4D", DATA)
      .AddOutputDesc({1, 2, 3, 4, 5}, FORMAT_ND, DT_FLOAT)
      .Build(graph);

  ge::NodePtr transdata_5d_2_4d = NodeBuilder("transdata_5d_2_4d", TRANSDATA)
                           .AddInputDesc({1, 2, 3, 4, 5}, FORMAT_ND, DT_FLOAT16)
                           .AddOutputDesc({1, 3, 4, 2}, FORMAT_NC1HWC0, DT_FLOAT16)
                           .Build(graph);

  ge::NodePtr cast_fp32_2_fp16 = NodeBuilder("cast_fp32_2_fp16", CAST)
                                .AddInputDesc({1, 2, 3, 4}, FORMAT_NC1HWC0, DT_FLOAT)
                                .AddOutputDesc({1, 2, 3, 4}, FORMAT_NC1HWC0, DT_FLOAT16)
                                .Build(graph);

  ge::NodePtr transpose_nhwc_2_nchw = NodeBuilder("transpose_nhwc_2_nchw", TRANSPOSE)
                                      .AddInputDesc({1, 2, 3, 4}, FORMAT_NC1HWC0, DT_FLOAT16)
                                      .AddOutputDesc({1, 3, 4, 2}, FORMAT_NC1HWC0, DT_FLOAT16)
                                      .Build(graph);



  ge::NodePtr node_relu_1 = NodeBuilder("relu_1", RELU)
                            .AddInputDesc({1, 3, 4, 2, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .AddOutputDesc({1, 3, 4, 2, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .Build(graph);


  // shape different,but format and datatype are equal
  ge::GraphUtils::AddEdge(node->GetOutDataAnchor(0), transdata_5d_2_4d->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transdata_5d_2_4d->GetOutDataAnchor(0), cast_fp32_2_fp16->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp32_2_fp16->GetOutDataAnchor(0), transpose_nhwc_2_nchw->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp32_2_fp16->GetOutControlAnchor(), transpose_nhwc_2_nchw->GetInControlAnchor());
  ge::GraphUtils::AddEdge(transpose_nhwc_2_nchw->GetOutDataAnchor(0), node_relu_1->GetInDataAnchor(0));

  EXPECT_EQ(graph->GetDirectNodesSize(), 5);
  ge::TransOpWithoutReshapeFusionPass pass;
  uint32_t status = pass.Run(graph);
  EXPECT_EQ(domi::SUCCESS, status);
  EXPECT_EQ(graph->GetDirectNodesSize(), 5);
}

/*

    Node---cast---transpose----transdata---C
              |   |                     |  |
              |+++|                     |++|
            ||
            \/
                                |+|
          |                     | |
    Node---cast---formattransfer---C
       |   |
       |+++|
*/
TEST_F(UtestTransopWithoutReshapeFusionPass, test_cast_formattransfer_out_data_in_control)
{
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("graph_test");

  ge::NodePtr node = NodeBuilder("Data4D", DATA)
      .AddOutputDesc({1, 2, 3, 4}, FORMAT_NHWC, DT_FLOAT)
      .Build(graph);

  ge::NodePtr cast_fp32_2_fp16 = NodeBuilder("cast_fp32_2_fp16", CAST)
                                .AddInputDesc({1, 2, 3, 4}, FORMAT_NHWC, DT_FLOAT)
                                .AddOutputDesc({1, 2, 3, 4}, FORMAT_NHWC, DT_FLOAT16)
                                .Build(graph);

  ge::NodePtr transpose_nhwc_2_nchw = NodeBuilder("transpose_nhwc_2_nchw", TRANSPOSE)
                                      .AddInputDesc({1, 2, 3, 4}, FORMAT_NHWC, DT_FLOAT16)
                                      .AddOutputDesc({1, 3, 4, 2}, FORMAT_NCHW, DT_FLOAT16)
                                      .Build(graph);

  ge::NodePtr transdata_4d_2_5d = NodeBuilder("transdata_4d_2_5d", TRANSDATA)
                            .AddInputDesc({1, 3, 4, 2}, FORMAT_NCHW, DT_FLOAT16)
                            .AddOutputDesc({1, 3, 4, 2, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .Build(graph);

  ge::NodePtr node_relu_1 = NodeBuilder("relu_1", RELU)
                            .AddInputDesc({1, 3, 4, 2}, FORMAT_NCHW, DT_FLOAT16)
                            .AddOutputDesc({1, 3, 4, 2}, FORMAT_NCHW, DT_FLOAT16)
                            .Build(graph);

  ge::NodePtr node_relu_2 = NodeBuilder("relu_2", RELU)
                            .AddInputDesc({1, 3, 4, 2}, FORMAT_NCHW, DT_FLOAT16)
                            .AddOutputDesc({1, 3, 4, 2}, FORMAT_NCHW, DT_FLOAT16)
                            .Build(graph);

  ge::NodePtr node_relu_3 = NodeBuilder("relu_3", RELU)
                            .AddInputDesc({1, 3, 4, 2, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .AddOutputDesc({1, 3, 4, 2, 5}, FORMAT_NC1HWC0, DT_FLOAT16)
                            .Build(graph);

  // shape different,but format and datatype are equal
  ge::GraphUtils::AddEdge(node->GetOutDataAnchor(0), cast_fp32_2_fp16->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp32_2_fp16->GetOutDataAnchor(0), transpose_nhwc_2_nchw->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp32_2_fp16->GetOutControlAnchor(), transpose_nhwc_2_nchw->GetInControlAnchor());
  ge::GraphUtils::AddEdge(transpose_nhwc_2_nchw->GetOutDataAnchor(0), transdata_4d_2_5d->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(transdata_4d_2_5d->GetOutDataAnchor(0), node_relu_3->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_relu_2->GetOutControlAnchor(), transdata_4d_2_5d->GetInControlAnchor());
  ge::GraphUtils::AddEdge(transdata_4d_2_5d->GetOutControlAnchor(), node_relu_3->GetInControlAnchor());

  ge::TransOpWithoutReshapeFusionPass pass;
  uint32_t status = pass.Run(graph);
  EXPECT_EQ(domi::SUCCESS, status);
  EXPECT_EQ(node->GetOutDataNodes().size(), 1);
  EXPECT_EQ(node_relu_3->GetInControlAnchor()->GetPeerOutControlAnchors().size(), 2);
  if (node->GetOutDataNodes().size() > 0) {
    NodePtr out_node = node->GetOutDataNodes().at(0);
    EXPECT_EQ(out_node->GetOpDesc()->GetType(), CAST);
    auto cast_output0_desc = out_node->GetOpDesc()->GetOutputDescPtr(0);
    EXPECT_EQ(cast_output0_desc->GetFormat(), FORMAT_NHWC);
    EXPECT_EQ(cast_output0_desc->GetDataType(), DT_FLOAT16);
    EXPECT_EQ(cast_output0_desc->GetShape().GetDim(0), 1);
    EXPECT_EQ(cast_output0_desc->GetShape().GetDim(1), 2);
    EXPECT_EQ(cast_output0_desc->GetShape().GetDim(2), 3);
    EXPECT_EQ(cast_output0_desc->GetShape().GetDim(3), 4);
    auto cast_input0_desc = out_node->GetOpDesc()->GetInputDescPtr(0);
    EXPECT_EQ(cast_input0_desc->GetFormat(), FORMAT_NHWC);
    EXPECT_EQ(cast_input0_desc->GetDataType(), DT_FLOAT);
    EXPECT_EQ(cast_input0_desc->GetShape().GetDim(0), 1);
    EXPECT_EQ(cast_input0_desc->GetShape().GetDim(1), 2);
    EXPECT_EQ(cast_input0_desc->GetShape().GetDim(2), 3);
    EXPECT_EQ(cast_input0_desc->GetShape().GetDim(3), 4);

    NodePtr format_transfer_node = out_node->GetOutDataNodes().at(0);
    EXPECT_EQ(format_transfer_node->GetOpDesc()->GetType(), TRANSDATA);
    auto format_transfer_output0_desc = format_transfer_node->GetOpDesc()->GetOutputDescPtr(0);
    EXPECT_EQ(format_transfer_output0_desc->GetFormat(), FORMAT_NC1HWC0);
    EXPECT_EQ(format_transfer_output0_desc->GetDataType(), DT_FLOAT16);
    EXPECT_EQ(format_transfer_output0_desc->GetShape().GetDim(0), 1);
    EXPECT_EQ(format_transfer_output0_desc->GetShape().GetDim(1), 3);
    EXPECT_EQ(format_transfer_output0_desc->GetShape().GetDim(2), 4);
    EXPECT_EQ(format_transfer_output0_desc->GetShape().GetDim(3), 2);
    EXPECT_EQ(format_transfer_output0_desc->GetShape().GetDim(4), 5);
    auto format_transfer_input0_desc = format_transfer_node->GetOpDesc()->GetInputDescPtr(0);
    EXPECT_EQ(format_transfer_input0_desc->GetFormat(), FORMAT_NHWC);
    EXPECT_EQ(format_transfer_input0_desc->GetDataType(), DT_FLOAT16);
    EXPECT_EQ(format_transfer_input0_desc->GetShape().GetDim(0), 1);
    EXPECT_EQ(format_transfer_input0_desc->GetShape().GetDim(1), 2);
    EXPECT_EQ(format_transfer_input0_desc->GetShape().GetDim(2), 3);
    EXPECT_EQ(format_transfer_input0_desc->GetShape().GetDim(3), 4);
  }
}

/**
 *     Data2         Data1
 *         \           |
 *   -------\------TransData1------
 *  |        \      /      `.      |
 *  |         Switch      Const    |
 *  |        /      \              |
 *  |  Identity2   Identity1       |
 *  |      `.           .`         |
 *   --TransData2    Cast1---------
 *         |           |
 *       Relu1     TransData3
 *          \          |
 *           \       Relu2
 *            \       /
 *              Merge
 *                |
 *            NetOutput
 */
TEST_F(UtestTransopWithoutReshapeFusionPass, test_remain_node_control_not_change) {
  DEF_GRAPH(g1) {
    auto data1 = OP_CFG(DATA).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_NHWC, DT_FLOAT, {1, 128, 192, 1})
                 .InCnt(1).OutCnt(1).Build("data1");
    auto transdata1 = OP_CFG(TRANSDATA).TensorDesc(FORMAT_NC1HWC0, DT_FLOAT, {1, 1, 128, 192, 1})
                      .InCnt(1).OutCnt(1).Build("transdata1");
    auto transdata2 = OP_CFG(TRANSDATA).TensorDesc(FORMAT_NHWC, DT_FLOAT, {1, 128, 192, 1})
                      .InCnt(1).OutCnt(1).Build("transdata2");
    auto cast1 = OP_CFG(CAST).TensorDesc(FORMAT_NC1HWC0, DT_FLOAT16, {1, 1, 128, 192, 1})
                 .InCnt(1).OutCnt(1).Build("cast1");
    auto relu1 = OP_CFG(RELU).TensorDesc(FORMAT_NHWC, DT_FLOAT, {1, 128, 192, 1})
                 .InCnt(1).OutCnt(1).Build("relu1");
    auto relu2 = OP_CFG(RELU).TensorDesc(FORMAT_NHWC, DT_FLOAT16, {1, 128, 192, 1})
                 .InCnt(1).OutCnt(1).Build("relu2");

    CHAIN(NODE(data1)->NODE(transdata1)->EDGE(0, 0)->NODE("switch1", SWITCH)->EDGE(0, 0)->NODE("identity1", IDENTITY));
    CHAIN(NODE("data2", DATA)->EDGE(0, 1)->NODE("switch1")->EDGE(1, 0)->NODE("identity2", IDENTITY));
    CHAIN(NODE(transdata1)->EDGE(0, 0)->NODE(cast1)->NODE("transdata3", TRANSDATA)->NODE(relu2)->EDGE(0, 0)->
          NODE("merge1", MERGE)->NODE("output", NETOUTPUT));
    CHAIN(NODE(transdata1)->EDGE(0, 0)->NODE(transdata2)->NODE(relu1)->EDGE(0, 1)->NODE("merge1"));
    CTRL_CHAIN(NODE(transdata1)->NODE("const1", CONSTANT));
    CTRL_CHAIN(NODE("identity1")->NODE(cast1));
    CTRL_CHAIN(NODE("identity2")->NODE(transdata2));
  };
  auto graph = ToComputeGraph(g1);

  GeTensorDesc desc_nhwc(GeShape({1, 128, 192, 1}), FORMAT_NHWC, DT_FLOAT);
  auto transdata1 = graph->FindNode("transdata1");
  transdata1->GetOpDesc()->UpdateInputDesc(0, desc_nhwc);
  GeTensorDesc desc_nc1hwc0(GeShape({1, 1, 128, 192, 1}), FORMAT_NC1HWC0, DT_FLOAT);
  auto transdata2 = graph->FindNode("transdata2");
  transdata2->GetOpDesc()->UpdateInputDesc(0, desc_nc1hwc0);
  auto cast1 = graph->FindNode("cast1");
  cast1->GetOpDesc()->UpdateInputDesc(0, desc_nc1hwc0);
  GeTensorDesc desc_nc1hwc0_fp16(GeShape({1, 1, 128, 192, 1}), FORMAT_NC1HWC0, DT_FLOAT16);
  auto transdata3 = graph->FindNode("transdata3");
  transdata3->GetOpDesc()->UpdateInputDesc(0, desc_nc1hwc0_fp16);
  GeTensorDesc desc_nhwc_fp16(GeShape({1, 128, 192, 1}), FORMAT_NHWC, DT_FLOAT16);
  transdata3->GetOpDesc()->UpdateOutputDesc(0, desc_nhwc_fp16);

  ge::TransOpWithoutReshapeFusionPass pass;
  uint32_t status = pass.Run(graph);
  EXPECT_EQ(domi::SUCCESS, status);
  EXPECT_EQ(graph->GetDirectNode().size(), 12);
  auto const1 = graph->FindNode("const1");
  EXPECT_EQ(const1->GetInControlNodes().size(), 1);
  auto fusion_cast = graph->FindNode("fusion_cast_op_13");
  EXPECT_NE(fusion_cast, nullptr);
}

/*

    Node---cast---cast-----A
                          |  |
                          |++|

            ||
            \/

        Node--cast---A
            |       |
            |+++++++|


*/
TEST_F(UtestTransopWithoutReshapeFusionPass, test_cast_fusion_no_need_check_format_nd)
{
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("graph_test");
  ge::NodePtr node = NodeBuilder("Data4D", DATA)
      .AddOutputDesc({1,2,3,4}, FORMAT_ND, DT_FLOAT16)
      .Build(graph);


  ge::NodePtr cast_bool_2_fp32 = NodeBuilder("cast_bool_2_fp32", CAST)
                                .AddInputDesc({1, 2, 3, 4}, FORMAT_ND, DT_BOOL)
                                .AddOutputDesc({1, 2, 3, 4}, FORMAT_ND, DT_FLOAT)
                                .Build(graph);
  ge::NodePtr cast_fp32_2_fp16 = NodeBuilder("cast_fp32_2_fp16", CAST)
                                .AddInputDesc({1, 2, 3, 4}, FORMAT_ND, DT_FLOAT)
                                .AddOutputDesc({1, 2, 3, 4}, FORMAT_ND, DT_FLOAT16)
                                .Build(graph);
  ge::NodePtr node_relu_1 = NodeBuilder("relu_1", RELU)
                            .AddInputDesc({1, 2, 3, 4}, FORMAT_ND, DT_FLOAT16)
                            .AddOutputDesc({1, 2, 3, 4}, FORMAT_ND, DT_FLOAT16)
                            .Build(graph);

  ge::GraphUtils::AddEdge(node->GetOutDataAnchor(0), cast_bool_2_fp32->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_bool_2_fp32->GetOutDataAnchor(0), cast_fp32_2_fp16->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp32_2_fp16->GetOutDataAnchor(0), node_relu_1->GetInDataAnchor(0));

  EXPECT_EQ(graph->GetDirectNodesSize(), 4);
  ge::TransOpWithoutReshapeFusionPass pass;
  uint32_t status = pass.Run(graph);
  EXPECT_EQ(domi::SUCCESS, status);
  EXPECT_EQ(graph->GetDirectNodesSize(), 3);

  auto fusion_cast = node_relu_1->GetInDataNodes().at(0);
  EXPECT_EQ(fusion_cast->GetType(), CAST);
  EXPECT_EQ(fusion_cast->GetOpDesc()->GetInputDesc(0).GetDataType(), DT_BOOL);
  EXPECT_EQ(fusion_cast->GetOpDesc()->GetOutputDesc(0).GetDataType(), DT_FLOAT16);
}
/*

    Node---cast----A---cast


            ||
            \/

    Node--cast---A---cast


*/
TEST_F(UtestTransopWithoutReshapeFusionPass, test_cast_fusion_has_precision_loss)
{
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("graph_test");
  ge::NodePtr node = NodeBuilder("Data4D", DATA)
      .AddOutputDesc({1,2,3,4}, FORMAT_ND, DT_FLOAT)
      .Build(graph);


  ge::NodePtr cast_fp32_2_int32 = NodeBuilder("cast_fp32_2_int32", CAST)
                                .AddInputDesc({1, 2, 3, 4}, FORMAT_ND, DT_FLOAT)
                                .AddOutputDesc({1, 2, 3, 4}, FORMAT_ND, DT_INT32)
                                .Build(graph);
  ge::NodePtr node_relu_1 = NodeBuilder("relu_1", RELU)
          .AddInputDesc({1, 2, 3, 4}, FORMAT_ND, DT_INT32)
          .AddOutputDesc({1, 2, 3, 4}, FORMAT_ND, DT_INT32)
          .Build(graph);
  ge::NodePtr cast_int32_2_bool = NodeBuilder("cast_fp32_2_int64", CAST)
                                .AddInputDesc({1, 2, 3, 4}, FORMAT_ND, DT_INT32)
                                .AddOutputDesc({1, 2, 3, 4}, FORMAT_ND, DT_BOOL)
                                .Build(graph);
    ge::NodePtr netoutput = NodeBuilder("netoutput", NETOUTPUT)
                                .AddInputDesc({1, 2, 3, 4}, FORMAT_ND, DT_BOOL)
                                .AddOutputDesc({1, 2, 3, 4}, FORMAT_ND, DT_BOOL)
                                .Build(graph);

  ge::GraphUtils::AddEdge(node->GetOutDataAnchor(0), cast_fp32_2_int32->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_fp32_2_int32->GetOutDataAnchor(0), node_relu_1->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_relu_1->GetOutDataAnchor(0), cast_int32_2_bool->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(cast_int32_2_bool->GetOutDataAnchor(0), netoutput->GetInDataAnchor(0));

  EXPECT_EQ(graph->GetDirectNodesSize(), 5);
  ge::TransOpWithoutReshapeFusionPass pass;
  uint32_t status = pass.Run(graph);
  EXPECT_EQ(domi::SUCCESS, status);
  EXPECT_EQ(graph->GetDirectNodesSize(), 5);
}
} // namespace ge
