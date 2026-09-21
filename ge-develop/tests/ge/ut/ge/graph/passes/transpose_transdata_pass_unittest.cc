/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <gtest/gtest.h>

#include "macro_utils/dt_public_scope.h"
#include "graph/passes/format_optimize/transpose_transdata_pass.h"
#include "engines/manager/opskernel_manager/ops_kernel_manager.h"
#include "graph_builder_utils.h"
#include "macro_utils/dt_public_unscope.h"

#include "graph/graph.h"
#include "common/ge_inner_error_codes.h"
#include "common/types.h"
#include "graph/debug/ge_attr_define.h"
#include "api/gelib/gelib.h"

namespace ge {
namespace {
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
    infos.emplace(std::make_pair("TransData", aicore_op));
}

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
    auto aicore_kernel_store = MakeShared<FEOpsKernelInfoStore>();
    opsKernelStore.emplace(std::pair<string, OpsKernelInfoStorePtr>("FEOpsStore", aicore_kernel_store));
}
} // namespace
class UtestGraphPassesTransposeTransdataPass : public testing::Test {
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

static ComputeGraphPtr BuildGraphTransposeD() {
  auto builder = ut::GraphBuilder("g1");
  auto transdata1 = builder.AddNode("transdata1", "TransData", 1, 1, FORMAT_NC1HWC0, DT_FLOAT, std::vector<int64_t>({1, 1, 224, 224, 16}));
  transdata1->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_NHWC);
  transdata1->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 224, 224, 3})));

  auto transpose1 = builder.AddNode("transpose1", "TransposeD", 1, 1, FORMAT_NCHW, DT_FLOAT, std::vector<int64_t>({1, 3, 224, 224}));
  transpose1->GetOpDesc()->MutableInputDesc(0)->SetFormat(FORMAT_NHWC);
  transpose1->GetOpDesc()->MutableInputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 224, 224, 3})));

  auto transdata2 = builder.AddNode("transdata2", "TransData", 1, 1, FORMAT_NCHW, DT_FLOAT, std::vector<int64_t>({1, 3, 224, 224}));
  transdata2->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_NC1HWC0);
  transdata2->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 1, 224, 224, 16})));

  builder.AddDataEdge(transdata1, 0, transpose1, 0);
  builder.AddDataEdge(transpose1, 0, transdata2, 0);

  return builder.GetGraph();
}


/*
 *        transdata1
 *            |
 *        transpose1
 *            |
 *      /          \
 * transdata2     relu
 */
static ComputeGraphPtr BuildGraphTransposeTwo() {
  auto builder = ut::GraphBuilder("g1");
  auto transdata1 = builder.AddNode("transdata1", "TransData", 1, 1, FORMAT_NC1HWC0, DT_FLOAT, std::vector<int64_t>({1,1,224,224,16}));
  transdata1->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_NHWC);
  transdata1->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1,224,224,3})));

  auto transpose1 = builder.AddNode("transpose1", "Transpose", 1, 1, FORMAT_NCHW, DT_FLOAT, std::vector<int64_t>({1,3,224,224}));
  transpose1->GetOpDesc()->MutableInputDesc(0)->SetFormat(FORMAT_NHWC);
  transpose1->GetOpDesc()->MutableInputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1,224,224,3})));

  auto transdata2 = builder.AddNode("transdata2", "TransData", 1, 1, FORMAT_NCHW, DT_FLOAT, std::vector<int64_t>({1,3,224,224}));
  transdata2->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_NC1HWC0);
  transdata2->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1,1,224,224,16})));

  auto relu = builder.AddNode("relu", "Relu", 1, 1, FORMAT_NCHW, DT_FLOAT, std::vector<int64_t>({1,3,224,224}));
  relu->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_NC1HWC0);
  relu->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1,1,224,224,16})));


  builder.AddDataEdge(transdata1, 0, transpose1, 0);
  builder.AddDataEdge(transpose1, 0, transdata2, 0);
  builder.AddDataEdge(transpose1, 0, relu, 0);

  return builder.GetGraph();
}

/*
     transdata1   constant
          \          /
           \        /
            \      /
          transpose1
               |
            transdata2
*/
static ComputeGraphPtr BuildGraphTranspose() {
  auto builder = ut::GraphBuilder("g1");
  auto transdata1 = builder.AddNode("transdata1", "TransData", 1, 1, FORMAT_NC1HWC0, DT_FLOAT, std::vector<int64_t>({1, 1, 224, 224, 16}));
  transdata1->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_NHWC);
  transdata1->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 224, 224, 3})));

  auto transpose1 = builder.AddNode("transpose1", "Transpose", 2, 1, FORMAT_NCHW, DT_FLOAT, std::vector<int64_t>({1, 3, 224, 224}));
  transpose1->GetOpDesc()->MutableInputDesc(0)->SetFormat(FORMAT_NHWC);
  transpose1->GetOpDesc()->MutableInputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 224, 224, 3})));

  auto transdata2 = builder.AddNode("transdata2", "TransData", 1, 1, FORMAT_NCHW, DT_FLOAT, std::vector<int64_t>({1, 3, 224, 224}));
  transdata2->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_NC1HWC0);
  transdata2->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 1, 224, 224, 16})));
  auto constant1 = builder.AddNode("Const", "Constant", 1, 1, FORMAT_ND, DT_INT64, std::vector<int64_t>({4}));
  builder.AddDataEdge(transdata1, 0, transpose1, 0);
  builder.AddDataEdge(transpose1, 0, transdata2, 0);
  builder.AddDataEdge(constant1, 0, transpose1, 1);
  return builder.GetGraph();
}
/*
     transdata1   constant1     constant2
          \          /           /
           \        /           /
            \      /           /
          transpose1  ---------
               |
            transdata2
*/
static ComputeGraphPtr BuildGraphTransposeErrorInput() {
  auto builder = ut::GraphBuilder("g1");
  auto transdata1 = builder.AddNode("transdata1", "TransData", 1, 1, FORMAT_NC1HWC0, DT_FLOAT, std::vector<int64_t>({1, 1, 224, 224, 16}));
  transdata1->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_NHWC);
  transdata1->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 224, 224, 3})));

  auto transpose1 = builder.AddNode("transpose1", "Transpose", 3, 1, FORMAT_NCHW, DT_FLOAT, std::vector<int64_t>({1, 3, 224, 224}));
  transpose1->GetOpDesc()->MutableInputDesc(0)->SetFormat(FORMAT_NHWC);
  transpose1->GetOpDesc()->MutableInputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 224, 224, 3})));

  auto transdata2 = builder.AddNode("transdata2", "TransData", 1, 1, FORMAT_NCHW, DT_FLOAT, std::vector<int64_t>({1, 3, 224, 224}));
  transdata2->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_NC1HWC0);
  transdata2->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 1, 224, 224, 16})));
  auto constant1 = builder.AddNode("Const1", "Constant", 1, 1, FORMAT_ND, DT_INT64, std::vector<int64_t>({4}));
  auto constant2 = builder.AddNode("Const2", "Constant", 1, 1, FORMAT_ND, DT_INT64, std::vector<int64_t>({4}));
  builder.AddDataEdge(transdata1, 0, transpose1, 0);
  builder.AddDataEdge(transpose1, 0, transdata2, 0);
  builder.AddDataEdge(constant1, 0, transpose1, 1);
  builder.AddDataEdge(constant2, 0, transpose1, 2);
  return builder.GetGraph();
}

/*
     transdata1   constant1
          \          /
           \        /
            \      /
          transpose1  ----------relu
               |
            transdata2
*/
static ComputeGraphPtr BuildGraphTransposeErrorOutput() {
  auto builder = ut::GraphBuilder("g1");
  auto transdata1 = builder.AddNode("transdata1", "TransData", 1, 1, FORMAT_NC1HWC0, DT_FLOAT, std::vector<int64_t>({1, 1, 224, 224, 16}));
  transdata1->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_NHWC);
  transdata1->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 224, 224, 3})));

  auto transpose1 = builder.AddNode("transpose1", "Transpose", 2, 2, FORMAT_NCHW, DT_FLOAT, std::vector<int64_t>({1, 3, 224, 224}));
  transpose1->GetOpDesc()->MutableInputDesc(0)->SetFormat(FORMAT_NHWC);
  transpose1->GetOpDesc()->MutableInputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 224, 224, 3})));

  auto transdata2 = builder.AddNode("transdata2", "TransData", 1, 1, FORMAT_NCHW, DT_FLOAT, std::vector<int64_t>({1, 3, 224, 224}));
  transdata2->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_NC1HWC0);
  transdata2->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 1, 224, 224, 16})));
  auto constant1 = builder.AddNode("Const1", "Constant", 1, 1, FORMAT_ND, DT_INT64, std::vector<int64_t>({4}));
  auto relu = builder.AddNode("relu", "Relu", 1, 1, FORMAT_NCHW, DT_FLOAT, std::vector<int64_t>({1, 3, 224, 224}));
  builder.AddDataEdge(transdata1, 0, transpose1, 0);
  builder.AddDataEdge(transpose1, 0, transdata2, 0);
  builder.AddDataEdge(constant1, 0, transpose1, 1);
  builder.AddDataEdge(transpose1, 1, relu, 0);
  return builder.GetGraph();
}

TEST_F(UtestGraphPassesTransposeTransdataPass, test_run) {
  auto compute_graph = BuildGraphTransposeD();
  compute_graph->SetSessionID(0);

  auto transpose = compute_graph->FindNode("transpose1");
  TransposeTransDataPass pass;
  EXPECT_EQ(pass.Run(transpose), SUCCESS);
}


TEST_F(UtestGraphPassesTransposeTransdataPass, run_nd) {
  auto compute_graph = BuildGraphTransposeD();
  compute_graph->SetSessionID(0);

  auto transpose = compute_graph->FindNode("transpose1");
  transpose->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_NHWC);
  TransposeTransDataPass pass;
  Status ret = pass.Run(transpose);

  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(compute_graph->GetDirectNodesSize(), 3);

  transpose->GetOpDesc()->MutableInputDesc(0)->SetFormat(FORMAT_ND);
  ret = pass.Run(transpose);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(compute_graph->GetDirectNodesSize(), 3);
}

TEST_F(UtestGraphPassesTransposeTransdataPass, run_success2) {
  auto compute_graph = BuildGraphTransposeD();
  compute_graph->SetSessionID(0);
  auto transpose = compute_graph->FindNode("transpose1");
  TransposeTransDataPass pass;
  Status ret = pass.Run(transpose);

  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(compute_graph->GetDirectNodesSize(), 2);

  auto node = compute_graph->FindNode("transpose1transdata2");
  bool same_format = node->GetOpDesc()->MutableInputDesc(0)->GetFormat() == FORMAT_NHWC;
  EXPECT_EQ(same_format, true);

  auto transpose_node = compute_graph->FindNode("transpose1");
  EXPECT_EQ(transpose_node, nullptr);
}

TEST_F(UtestGraphPassesTransposeTransdataPass, run_wrong_input) {
  auto compute_graph = BuildGraphTransposeTwo();
  compute_graph->SetSessionID(0);

  auto transpose = compute_graph->FindNode("transpose1");
  TransposeTransDataPass pass;
  Status ret = pass.Run(transpose);

  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestGraphPassesTransposeTransdataPass, run_null) {
  NodePtr node = nullptr;

  TransposeTransDataPass pass;
  Status ret = pass.Run(node);

  EXPECT_EQ(ret, PARAM_INVALID);
}

TEST_F(UtestGraphPassesTransposeTransdataPass, run_op_desc_null) {
  OpDescPtr op_desc = nullptr;
  ComputeGraphPtr owner_graph = nullptr;
  NodePtr node_ptr = shared_ptr<Node>(new (std::nothrow) Node(op_desc, owner_graph));

  TransposeTransDataPass pass;
  Status ret = pass.Run(node_ptr);

  EXPECT_EQ(ret, PARAM_INVALID);
}

TEST_F(UtestGraphPassesTransposeTransdataPass, TransAdjWithUnknowShape) {
  auto compute_graph = BuildGraphTransposeD();
  compute_graph->SetSessionID(0);

  auto transpose = compute_graph->FindNode("transpose1");
  auto op_desc = transpose->GetOpDesc();
  std::vector<int64_t> dims = {-1, 224, 224, 3};
  op_desc->MutableOutputDesc(0)->SetShape(GeShape(dims));
  TransposeTransDataPass pass;
  Status ret = pass.Run(transpose);

  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(compute_graph->GetDirectNodesSize(), 3);
}

TEST_F(UtestGraphPassesTransposeTransdataPass, run_success3) {
  auto compute_graph = BuildGraphTranspose();
  compute_graph->SetSessionID(0);
  auto transpose = compute_graph->FindNode("transpose1");
  TransposeTransDataPass pass;
  Status ret = pass.Run(transpose);

  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(compute_graph->GetDirectNodesSize(), 3);

  auto node = compute_graph->FindNode("transpose1transdata2");
  bool same_format = node->GetOpDesc()->MutableInputDesc(0)->GetFormat() == FORMAT_NHWC;
  EXPECT_EQ(same_format, true);
  auto input_node = node->GetInControlAnchor()->GetPeerOutControlAnchors().at(0)->GetOwnerNode();
  EXPECT_EQ(input_node->GetName(), "Const");
  auto transpose_node = compute_graph->FindNode("transpose1");
  EXPECT_EQ(transpose_node, nullptr);
}

TEST_F(UtestGraphPassesTransposeTransdataPass, run_failed1) {
  auto compute_graph = BuildGraphTransposeErrorInput();
  compute_graph->SetSessionID(0);
  auto transpose = compute_graph->FindNode("transpose1");
  TransposeTransDataPass pass;
  Status ret = pass.Run(transpose);

  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestGraphPassesTransposeTransdataPass, run_failed2) {
  auto compute_graph = BuildGraphTransposeErrorOutput();
  compute_graph->SetSessionID(0);
  auto transpose = compute_graph->FindNode("transpose1");
  TransposeTransDataPass pass;
  Status ret = pass.Run(transpose);

  EXPECT_EQ(ret, FAILED);
}

}  // namespace ge
