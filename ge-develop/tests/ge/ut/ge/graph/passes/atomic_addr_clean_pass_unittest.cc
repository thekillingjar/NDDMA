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
#include "macro_utils/dt_public_scope.h"
#include "graph/passes/memory_conflict/atomic_addr_clean_pass.h"
#include "common/op/ge_op_utils.h"
#include "common/types.h"
#include "graph/anchor.h"
#include "graph/attr_value.h"
#include "graph/compute_graph.h"
#include "graph/op_desc.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/passes/pass_manager.h"
#include "ge/ge_api.h"
#include "api/gelib/gelib.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "engines/manager/opskernel_manager/ops_kernel_manager.h"
#include "graph/debug/ge_attr_define.h"

using namespace testing;
using namespace domi;

namespace ge {
namespace {
class TestOpsKernelInfoStore : public OpsKernelInfoStore {
 public:
  TestOpsKernelInfoStore() = default;

  ~TestOpsKernelInfoStore() override = default;

  Status Initialize(const map<string, string> &options)  { return ge::SUCCESS; }
  Status Finalize() override { return ge::SUCCESS; }
  bool CheckSupported(const OpDescPtr &opDescPtr, std::string &un_supported_reason) const { return true; }
  Status CalcOpRunningParam(Node &node) { return ge::SUCCESS; }
  Status GenerateTask(const Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks)  {
    return ge::SUCCESS; }
  Status CompileOp(vector<ge::NodePtr> &node_vec) { return ge::SUCCESS; }
  void GetAllOpsKernelInfo(map<string, OpInfo> &infos) const { return; }
  std::vector<OpInfo> GetOpsKernelInfoStore(const std::string &name) { return empty_op_info_; }
  OpInfo test_op = {"DNN_VM_TF", "ge_local", 0, false, false, true};
  std::vector<OpInfo> empty_op_info_{test_op};
};
}
class UtestGraphPassesAtomicAddrCleanPass : public Test {
public:
 void SetUp() {
   std::map<AscendString, AscendString> options;
   GEInitialize(options);
 }
 void TearDown() {
   GEFinalize();
 }

  UtestGraphPassesAtomicAddrCleanPass() {
    graph_ = std::make_shared<ComputeGraph>("test");
    GeShape shape = GeShape({1,1,224,224});
    default_tensor_desc_ = std::make_shared<GeTensorDesc>();
    default_tensor_desc_->SetShape(shape);
    default_tensor_desc_->SetFormat(FORMAT_NCHW);
    default_tensor_desc_->SetDataType(DT_FLOAT);
    pass_manager_.AddPass("AtomicAddrCleanPass", new AtomicAddrCleanPass);
  }

  NodePtr NewNode(const string &name, const string &type, int input_cnt, int output_cnt) {
    OpDescPtr op_desc = std::make_shared<OpDesc>(name, type);
    for (int i = 0; i < input_cnt; ++i) {
      op_desc->AddInputDesc(GeTensorDesc());
    }
    for (int i = 0; i < output_cnt; ++i) {
      op_desc->AddOutputDesc(GeTensorDesc());
    }
    NodePtr node = graph_->AddNode(op_desc);
    if (type == CONSTANT) {
      int32_t weight[] = {1};
      GeTensorDesc weight_desc(GeShape({1}), FORMAT_NHWC, DT_INT32);
      GeTensorPtr tensor = std::make_shared<GeTensor>(weight_desc, (uint8_t *)weight, sizeof(weight));
      vector<GeTensorPtr> tensor_vec = {tensor};
      OpDescUtils::SetWeights(node, tensor_vec);
    }
    return node;
  }

  NodePtr NewNode(ComputeGraphPtr &graph, const std::string &name, const std::string &type, int input_cnt,
                  int output_cnt) {
    OpDescPtr op_desc = std::make_shared<OpDesc>(name, type);
    for (int i = 0; i < input_cnt; ++i) {
      op_desc->AddInputDesc(default_tensor_desc_->Clone());
    }

    for (int i = 0; i < output_cnt; ++i) {
      op_desc->AddOutputDesc(default_tensor_desc_->Clone());
    }

    NodePtr node = graph->AddNode(op_desc);
    if (type == CONSTANT) {
      int32_t weight[] = {1};
      GeTensorDesc weight_desc(GeShape({1}), FORMAT_NHWC, DT_INT32);
      GeTensorPtr tensor = std::make_shared<GeTensor>(weight_desc, (uint8_t *)weight, sizeof(weight));
      vector<GeTensorPtr> tensor_vec = {tensor};
      OpDescUtils::SetWeights(node, tensor_vec);
    }

    return node;
  }

  int CountOfAtomicMemsetNode() {
    int node_num = 0;
    for (NodePtr &node : graph_->GetDirectNode()) {
      if (NodeUtils::IsLikeAtomicClean(node)) {
        ++node_num;
      }
    }
    return node_num;
  }

  int CountOfAtomicMemsetNode(ComputeGraphPtr &graph) {
    int node_num = 0;
    for (NodePtr node : graph->GetDirectNode()) {
      if (NodeUtils::IsLikeAtomicClean(node)) {
        node_num++;
      }
    }
    return node_num;
  }

  ComputeGraphPtr graph_;
  GeTensorDescPtr default_tensor_desc_;
  AtomicAddrCleanPass atomic_clean_pass_;
  PassManager pass_manager_;
};

TEST_F(UtestGraphPassesAtomicAddrCleanPass, NullInput) {
  auto ret = atomic_clean_pass_.Run(nullptr);
  EXPECT_EQ(ret, PARAM_INVALID);
}

TEST_F(UtestGraphPassesAtomicAddrCleanPass, EmptyGraph) {
  auto ret = atomic_clean_pass_.Run(graph_);
  EXPECT_EQ(ret, SUCCESS);
}

/*     Op1
 *      |
 *     Op2
 *      |
 *  NetOutput
 */
TEST_F(UtestGraphPassesAtomicAddrCleanPass, GraphWithoutAtomicNode) {
  auto node1 = NewNode("Op1", DATA_TYPE, 0, 1);
  auto node2 = NewNode("Op2", RELU, 1, 1);
  auto net_output = NewNode("NetOutput", NETOUTPUT, 3, 3);

  GraphUtils::AddEdge(node1->GetOutDataAnchor(0), node2->GetInDataAnchor(0));
  GraphUtils::AddEdge(node2->GetOutDataAnchor(0), net_output->GetInDataAnchor(1));
  Status ret = atomic_clean_pass_.Run(graph_);
  EXPECT_EQ(ret, SUCCESS);
}

/*     Op1
 *      |
 *     Op2(atomic)
 *      |
 *  NetOutput
 */
TEST_F(UtestGraphPassesAtomicAddrCleanPass, GraphWithOneTBEAtomicNode) {
  auto node1 = NewNode("Op1", DATA_TYPE, 0, 1);
  auto node2 = NewNode("Op2", RELU, 1, 1);
  OpDescPtr op_desc2 = node2->GetOpDesc();
  op_desc2->SetAttr("atomic_input_index", GeAttrValue::CreateFrom<GeAttrValue::LIST_INT>({123, 456}));
  ge::AttrUtils::SetBool(op_desc2, ge::ATTR_NAME_NOTASK, true);
  auto net_output = NewNode("NetOutput", NETOUTPUT, 3, 3);

  GraphUtils::AddEdge(node1->GetOutDataAnchor(0), node2->GetInDataAnchor(0));
  GraphUtils::AddEdge(node2->GetOutDataAnchor(0), net_output->GetInDataAnchor(1));
  auto ret = atomic_clean_pass_.Run(graph_);

  EXPECT_EQ(ret, SUCCESS);  // todo
  EXPECT_EQ(1, CountOfAtomicMemsetNode());
}

/*           data0
 *            /
 *     Op1  Op0(atomic)
 *      |   /
 *     Op2(no task& atomic)
 *      |
 *  NetOutput
 */
TEST_F(UtestGraphPassesAtomicAddrCleanPass, GraphAtomicMix) {
  auto data0 = NewNode("Data0", DATA_TYPE, 0, 1);
  auto node0 = NewNode("Op0", RELU, 1, 1);
  auto node1 = NewNode("Op1", DATA_TYPE, 0, 1);
  auto node2 = NewNode("Op2", RELU, 2, 1);
  OpDescPtr op_desc0 = node0->GetOpDesc();
  op_desc0->SetAttr("atomic_input_index", GeAttrValue::CreateFrom<GeAttrValue::LIST_INT>({123, 456}));
  op_desc0->SetOpKernelLibName("aicore_kernel");  // op0 need gen atomic task
  OpDescPtr op_desc2 = node2->GetOpDesc();
  op_desc2->SetAttr("atomic_input_index", GeAttrValue::CreateFrom<GeAttrValue::LIST_INT>({123, 456}));
  ge::AttrUtils::SetBool(op_desc2, ge::ATTR_NAME_NOTASK, true);  // op2 need clean by inserted atomic memset node
  auto net_output = NewNode("NetOutput", NETOUTPUT, 1, 1);

  GraphUtils::AddEdge(data0->GetOutDataAnchor(0), node0->GetInDataAnchor(0));
  GraphUtils::AddEdge(node1->GetOutDataAnchor(0), node2->GetInDataAnchor(0));
  GraphUtils::AddEdge(node0->GetOutDataAnchor(0), node2->GetInDataAnchor(1));
  GraphUtils::AddEdge(node2->GetOutDataAnchor(0), net_output->GetInDataAnchor(0));
  std::shared_ptr<GELib> instancePtr = ge::GELib::GetInstance();
  std::shared_ptr<OpsKernelManager> opsKernelManager =
      std::make_shared<OpsKernelManager>(instancePtr->OpsKernelManagerObj());
  auto opkernel = std::make_shared<TestOpsKernelInfoStore>();
  instancePtr->OpsKernelManagerObj().ops_kernel_store_.insert(std::make_pair("aicore_kernel", opkernel));
  auto ret = atomic_clean_pass_.Run(graph_);

  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(1, CountOfAtomicMemsetNode());
  EXPECT_EQ(node0->GetInControlNodes().size(), 0U);  // op0 has no control input from atomic memset node
  EXPECT_EQ(node2->GetInControlNodes().size(), 1U);  // op2 has control input from inserted atomic memset node
}

/*
 *            variable1
 *             /
 *     const transdata1(need clean)
 *      |    /
 *     assign
 *      |
 *    transdata2(need clean, has attr ref_var_src_var_name: varialbe1)
 *      |
 *  NetOutput
 */
TEST_F(UtestGraphPassesAtomicAddrCleanPass, TransDataRefFromVariable_NeedCleanSeparately) {
  auto con = NewNode("const", CONSTANT, 0, 1);
  auto variable1 = NewNode("variable1", VARIABLE, 0, 1);
  auto transdata1 = NewNode("transdata1", TRANSDATA, 1, 1);
  auto transdata2 = NewNode("transdata2", TRANSDATA, 1, 1);
  auto assign = NewNode("assign", ASSIGN, 2, 1);

  transdata1->GetOpDesc()->SetAttr("atomic_output_index", GeAttrValue::CreateFrom<GeAttrValue::LIST_INT>({0}));
  transdata1->GetOpDesc()->SetOpKernelLibName("aicore_kernel");  // op0 need gen atomic task
  transdata2->GetOpDesc()->SetAttr("atomic_output_index", GeAttrValue::CreateFrom<GeAttrValue::LIST_INT>({0}));
  transdata2->GetOpDesc()->SetOpKernelLibName("aicore_kernel");  // op0 need gen atomic task

  ge::AttrUtils::SetStr(transdata2->GetOpDesc()->MutableOutputDesc(0), ge::REF_VAR_SRC_VAR_NAME, "variable1");
  auto net_output = NewNode("NetOutput", NETOUTPUT, 1, 1);

  GraphUtils::AddEdge(con->GetOutDataAnchor(0), assign->GetInDataAnchor(0));
  GraphUtils::AddEdge(assign->GetOutDataAnchor(0), transdata2->GetInDataAnchor(0));
  GraphUtils::AddEdge(variable1->GetOutDataAnchor(0), transdata1->GetInDataAnchor(0));
  GraphUtils::AddEdge(transdata1->GetOutDataAnchor(0), assign->GetInDataAnchor(1));
  GraphUtils::AddEdge(transdata2->GetOutDataAnchor(0), net_output->GetInDataAnchor(0));
  std::shared_ptr<GELib> instancePtr = ge::GELib::GetInstance();
  std::shared_ptr<OpsKernelManager> opsKernelManager =
      std::make_shared<OpsKernelManager>(instancePtr->OpsKernelManagerObj());
  auto opkernel = std::make_shared<TestOpsKernelInfoStore>();
  instancePtr->OpsKernelManagerObj().ops_kernel_store_.insert(std::make_pair("aicore_kernel", opkernel));
  auto ret = atomic_clean_pass_.Run(graph_);

  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(1, CountOfAtomicMemsetNode());

  bool clean_separately = false;
  ge::AttrUtils::GetBool(transdata2->GetOpDesc(), "need_gentask_atomic", clean_separately);
  EXPECT_TRUE(clean_separately);

  bool clean_separately2 = false;
  ge::AttrUtils::GetBool(transdata1->GetOpDesc(), "need_gentask_atomic", clean_separately2);
  EXPECT_FALSE(clean_separately2);
}

TEST_F(UtestGraphPassesAtomicAddrCleanPass, GraphWithOneHcomAllReduceAtomicNode) {
  std::shared_ptr<GELib> instancePtr = ge::GELib::GetInstance();
  OpsKernelManager &ops_kernel_manager = instancePtr->OpsKernelManagerObj();
  OpInfo HCOMALLREDUCE_op = {"DNN_VM_TF", "hccl_kernel", 0, false, false, true};
  OpInfo DATA_op = {"DNN_VM_TF", "ge_local", 0, false, false};
  vector<OpInfo> op_infos;
  op_infos.push_back(HCOMALLREDUCE_op);
  vector<OpInfo> op_infos1;
  op_infos1.push_back(DATA_op);
  ops_kernel_manager.ops_kernel_info_.insert(std::make_pair(HCOMALLREDUCE, op_infos));
  ops_kernel_manager.ops_kernel_info_.insert(std::make_pair(DATA, op_infos1));

  auto node1 = NewNode("Op1", DATA_TYPE, 0, 1);
  auto node2 = NewNode("Op2", HCOMALLREDUCE, 1, 1);
  OpDescPtr op_desc2 = node2->GetOpDesc();
  op_desc2->SetAttr("atomic_input_index", GeAttrValue::CreateFrom<GeAttrValue::LIST_INT>({123, 456}));
  auto net_output = NewNode("NetOutput", NETOUTPUT, 3, 3);

  GraphUtils::AddEdge(node1->GetOutDataAnchor(0), node2->GetInDataAnchor(0));
  GraphUtils::AddEdge(node2->GetOutDataAnchor(0), net_output->GetInDataAnchor(1));
  auto ret = atomic_clean_pass_.Run(graph_);

  EXPECT_EQ(ret, SUCCESS);
  // data+allreduce no atomic clean
  EXPECT_EQ(1, CountOfAtomicMemsetNode());
}

TEST_F(UtestGraphPassesAtomicAddrCleanPass, GraphWithOneHcomAllReduceAtomicNodeRun) {
  std::shared_ptr<GELib> instancePtr = ge::GELib::GetInstance();
  OpsKernelManager &ops_kernel_manager = instancePtr->OpsKernelManagerObj();
  OpInfo HCOMALLREDUCE_op = {"DNN_VM_TF", "hccl_kernel", 0, false, false, true};
  OpInfo DATA_op = {"DNN_VM_TF", "ge_local", 0, false, false};
  vector<OpInfo> op_infos;
  op_infos.push_back(HCOMALLREDUCE_op);
  vector<OpInfo> op_infos1;
  op_infos1.push_back(DATA_op);
  ops_kernel_manager.ops_kernel_info_.insert(std::make_pair(HCOMALLREDUCE, op_infos));
  ops_kernel_manager.ops_kernel_info_.insert(std::make_pair(FRAMEWORK_OP_TYPE, op_infos1));
  ops_kernel_manager.ops_kernel_info_.insert(std::make_pair(DATA, op_infos1));

  auto node1 = NewNode("Op1", DATA_TYPE, 0, 1);
  auto node2 = NewNode("Op2", FRAMEWORK_OP_TYPE, 1, 1);
  auto node3 = NewNode("Op3", HCOMALLREDUCE, 1, 1);
  OpDescPtr op_desc2 = node2->GetOpDesc();
  op_desc2->SetAttr("atomic_input_index", GeAttrValue::CreateFrom<GeAttrValue::LIST_INT>({123, 456}));
  OpDescPtr op_desc3 = node3->GetOpDesc();
  op_desc3->SetAttr("atomic_input_index", GeAttrValue::CreateFrom<GeAttrValue::LIST_INT>({123, 456}));
  auto net_output = NewNode("NetOutput", NETOUTPUT, 3, 3);

  GraphUtils::AddEdge(node1->GetOutDataAnchor(0), node2->GetInDataAnchor(0));
  GraphUtils::AddEdge(node2->GetOutDataAnchor(0), node3->GetInDataAnchor(0));
  GraphUtils::AddEdge(node3->GetOutDataAnchor(0), net_output->GetInDataAnchor(1));
  auto ret = atomic_clean_pass_.Run(graph_);

  EXPECT_EQ(ret, SUCCESS);
  // data+allreduce no atomic clean
  EXPECT_EQ(1, CountOfAtomicMemsetNode());
}

TEST_F(UtestGraphPassesAtomicAddrCleanPass, GraphWithTwoHcomAllReduceAtomicNode) {
  std::shared_ptr<GELib> instancePtr = ge::GELib::GetInstance();
  OpsKernelManager &ops_kernel_manager = instancePtr->OpsKernelManagerObj();
  OpInfo HCOMALLREDUCE_op = {"DNN_VM_TF", "hccl_kernel", 0, false, false, true};
  OpInfo DATA_op = {"DNN_VM_TF", "ge_local", 0, false, false};
  vector<OpInfo> op_infos;
  op_infos.push_back(HCOMALLREDUCE_op);
  vector<OpInfo> op_infos1;
  op_infos1.push_back(DATA_op);
  ops_kernel_manager.ops_kernel_info_.insert(std::make_pair(HCOMALLREDUCE, op_infos));
  ops_kernel_manager.ops_kernel_info_.insert(std::make_pair(DATA, op_infos1));

  auto node1 = NewNode("Op1", DATA_TYPE, 0, 1);
  auto node2 = NewNode("Op2", HCOMALLREDUCE, 1, 1);
  auto node3 = NewNode("Op3", HCOMALLREDUCE, 1, 1);
  OpDescPtr op_desc2 = node2->GetOpDesc();
  op_desc2->SetAttr("atomic_input_index", GeAttrValue::CreateFrom<GeAttrValue::LIST_INT>({123, 456}));
  OpDescPtr op_desc3 = node3->GetOpDesc();
  op_desc3->SetAttr("atomic_input_index", GeAttrValue::CreateFrom<GeAttrValue::LIST_INT>({123, 456}));
  auto net_output = NewNode("NetOutput", NETOUTPUT, 3, 3);

  GraphUtils::AddEdge(node1->GetOutDataAnchor(0), node2->GetInDataAnchor(0));
  GraphUtils::AddEdge(node2->GetOutDataAnchor(0), node3->GetInDataAnchor(0));
  GraphUtils::AddEdge(node3->GetOutDataAnchor(0), net_output->GetInDataAnchor(1));
  auto ret = atomic_clean_pass_.Run(graph_);

  EXPECT_EQ(ret, SUCCESS);
  // data+two allreduce one atomic clean
  EXPECT_EQ(1, CountOfAtomicMemsetNode());
  EXPECT_EQ(node2->GetInControlAnchor()->GetPeerOutControlAnchors().size(), 1); // Op2 only has 1 control edge from atomic node
}

/*
 *     Op1
 *      |
 *     Op2(atomic)
 *      |
 *   HcomALLToALL
 *      |
 *  NetOutput
 */
TEST_F(UtestGraphPassesAtomicAddrCleanPass, GraphWithTwoAtomicNodeLinkALLTOALL) {
  std::shared_ptr<GELib> instancePtr = ge::GELib::GetInstance();
  OpsKernelManager &ops_kernel_manager = instancePtr->OpsKernelManagerObj();
  OpInfo HCOMALLTOALL_op = {"DNN_VM_TF", "hccl_kernel", 0, false, false, false};
  vector<OpInfo> op_infos;
  op_infos.push_back(HCOMALLTOALL_op);
  ops_kernel_manager.ops_kernel_info_.insert(std::make_pair(HCOMALLTOALL, op_infos));

  auto node1 = NewNode("Op1", DATA_TYPE, 0, 1);
  auto node2 = NewNode("Op2", RELU, 1, 1);
  auto node_hcom_all_to_all = NewNode("Op3", HCOMALLTOALL, 1, 1);
  auto net_output = NewNode("NetOutput", NETOUTPUT, 3, 3);

  OpDescPtr op_desc2 = node2->GetOpDesc();
  op_desc2->SetAttr("atomic_input_index", GeAttrValue::CreateFrom<GeAttrValue::LIST_INT>({123, 456}));
  (void)AttrUtils::SetStr(op_desc2, ATTR_NAME_STREAM_LABEL, "label1");

  GraphUtils::AddEdge(node1->GetOutDataAnchor(0), node2->GetInDataAnchor(0));
  GraphUtils::AddEdge(node2->GetOutDataAnchor(0), node_hcom_all_to_all->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_hcom_all_to_all->GetOutDataAnchor(0), net_output->GetInDataAnchor(0));
  graph_->AddInputNode(node1);
  (void)AttrUtils::SetStr(graph_, ATTR_NAME_SESSION_GRAPH_ID, "012");
  auto ret = atomic_clean_pass_.Run(graph_);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(1, CountOfAtomicMemsetNode());
}

/*
 *     Op1
 *      |
 *     Op2(atomic)
 *      |
 *     Op3(atomic)
 *      |
 *  NetOutput
 */
TEST_F(UtestGraphPassesAtomicAddrCleanPass, GraphWithTwoAtomicNode) {
  auto node1 = NewNode("Op1", DATA_TYPE, 0, 1);
  auto node2 = NewNode("Op2", RELU, 1, 1);
  auto node3 = NewNode("Op3", RELU, 1, 1);
  auto net_output = NewNode("NetOutput", NETOUTPUT, 3, 3);

  OpDescPtr op_desc2 = node2->GetOpDesc();
  op_desc2->SetAttr("atomic_input_index", GeAttrValue::CreateFrom<GeAttrValue::LIST_INT>({123, 456}));
  (void)AttrUtils::SetStr(op_desc2, ATTR_NAME_STREAM_LABEL, "label1");

  OpDescPtr op_desc3 = node3->GetOpDesc();
  op_desc3->SetAttr("atomic_input_index", GeAttrValue::CreateFrom<GeAttrValue::LIST_INT>({123, 456}));

  vector<uint32_t> is_connect_netoutput = {1};
  (void)ge::AttrUtils::SetListInt(op_desc3, ATTR_NAME_NODE_CONNECT_OUTPUT, is_connect_netoutput);

  GraphUtils::AddEdge(node1->GetOutDataAnchor(0), node2->GetInDataAnchor(0));
  GraphUtils::AddEdge(node2->GetOutDataAnchor(0), node3->GetInDataAnchor(1));
  GraphUtils::AddEdge(node3->GetOutDataAnchor(0), net_output->GetInDataAnchor(1));
  graph_->AddInputNode(node1);
  (void)AttrUtils::SetStr(graph_, ATTR_NAME_SESSION_GRAPH_ID, "012");
  auto ret = atomic_clean_pass_.Run(graph_);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(2, CountOfAtomicMemsetNode());
}

/*
 *     Op1
 *      |
 *     Op2(atomic)
 *      |
 *     Op3(atomic)
 *      |
 *  NetOutput
 */
TEST_F(UtestGraphPassesAtomicAddrCleanPass, GraphWithLoopCond) {
  auto node1 = NewNode("Op1", DATA_TYPE, 0, 1);
  auto node2 = NewNode("Op2", LOOPCOND, 1, 1);
  auto node3 = NewNode("Op2", RELU, 1, 1);
  auto net_output = NewNode("NetOutput", NETOUTPUT, 3, 3);

  OpDescPtr op_desc2 = node2->GetOpDesc();
  op_desc2->SetAttr("atomic_input_index", GeAttrValue::CreateFrom<GeAttrValue::LIST_INT>({123, 456}));

  OpDescPtr op_desc3 = node3->GetOpDesc();
  op_desc3->SetAttr("atomic_input_index", GeAttrValue::CreateFrom<GeAttrValue::LIST_INT>({123, 456}));

  GraphUtils::AddEdge(node1->GetOutDataAnchor(0), node2->GetInDataAnchor(0));
  GraphUtils::AddEdge(node2->GetOutDataAnchor(0), node3->GetInDataAnchor(1));
  GraphUtils::AddEdge(node3->GetOutDataAnchor(0), net_output->GetInDataAnchor(1));
  graph_->AddInputNode(node1);
  auto ret = atomic_clean_pass_.Run(graph_);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(2, CountOfAtomicMemsetNode());
}

/*     Op1
 *      |
 *     Op2(atomic)
 *      |
 *  NetOutput
 */
TEST_F(UtestGraphPassesAtomicAddrCleanPass, GraphWithOneTBEAtomicNodeWithUnknowShape) {
  auto node1 = NewNode("Op1", DATA_TYPE, 0, 1);
  auto node2 = NewNode("Op2", RELU, 1, 1);
  OpDescPtr op_desc2 = node2->GetOpDesc();
  op_desc2->SetAttr("atomic_input_index", GeAttrValue::CreateFrom<GeAttrValue::LIST_INT>({123, 456}));
  op_desc2->SetOpKernelLibName("aicore_kernel");
  auto net_output = NewNode("NetOutput", NETOUTPUT, 3, 3);

  GraphUtils::AddEdge(node1->GetOutDataAnchor(0), node2->GetInDataAnchor(0));
  GraphUtils::AddEdge(node2->GetOutDataAnchor(0), net_output->GetInDataAnchor(1));

  std::vector<int64_t> dims = {-1, 224, 224, 3};
  op_desc2->MutableOutputDesc(0)->SetShape(GeShape(dims));
  graph_->SetGraphUnknownFlag(true);

  std::shared_ptr<GELib> instancePtr = ge::GELib::GetInstance();
  std::shared_ptr<OpsKernelManager> opsKernelManager =
      std::make_shared<OpsKernelManager>(instancePtr->OpsKernelManagerObj());
  auto opkernel = std::make_shared<TestOpsKernelInfoStore>();
  instancePtr->OpsKernelManagerObj().ops_kernel_store_.insert(std::make_pair("aicore_kernel", opkernel));

  auto ret = atomic_clean_pass_.Run(graph_);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(CountOfAtomicMemsetNode(), 0);
}

/*     Op1                    Op1
 *      |           sub        |
 *   Op2(atomic) -------->  Op2(atomic) known
 *      |                      |
 *  NetOutput              NetOutput
 */
TEST_F(UtestGraphPassesAtomicAddrCleanPass, GraphWithOneTBEAtomicNodeWithSubgraph) {
  // graph
  {
    auto node1 = NewNode("Op1", DATA_TYPE, 0, 1);
    auto node2 = NewNode("Op2", RELU, 1, 1);
    OpDescPtr op_desc2 = node2->GetOpDesc();
    op_desc2->SetAttr("atomic_input_index", GeAttrValue::CreateFrom<GeAttrValue::LIST_INT>({123, 456}));
    auto net_output = NewNode("NetOutput", NETOUTPUT, 3, 3);

    GraphUtils::AddEdge(node1->GetOutDataAnchor(0), node2->GetInDataAnchor(0));
    GraphUtils::AddEdge(node2->GetOutDataAnchor(0), net_output->GetInDataAnchor(1));
    graph_->SetName("root_graph");
  }

  ComputeGraphPtr subgraph = make_shared<ComputeGraph>("sub_graph");
  // sub graph
  {
    auto node1 = NewNode(subgraph, "Op1", DATA_TYPE, 0, 1);
    auto node2 = NewNode(subgraph, "Op2", RELU, 1, 1);
    OpDescPtr op_desc2 = node2->GetOpDesc();
    op_desc2->SetAttr("atomic_input_index", GeAttrValue::CreateFrom<GeAttrValue::LIST_INT>({123, 456}));
    auto net_output = NewNode(subgraph, "NetOutput", NETOUTPUT, 3, 3);

    GraphUtils::AddEdge(node1->GetOutDataAnchor(0), node2->GetInDataAnchor(0));
    GraphUtils::AddEdge(node2->GetOutDataAnchor(0), net_output->GetInDataAnchor(1));
    subgraph->SetName("sub_graph");
  }

  auto Op2 = graph_->FindNode("Op2");
  auto Op2_desc = Op2->GetOpDesc();
  Op2_desc->AddSubgraphName("sub_graph");
  Op2_desc->SetSubgraphInstanceName(0, "sub_graph");
  subgraph->SetParentNode(Op2);
  subgraph->SetParentGraph(graph_);
  graph_->AddSubgraph(subgraph);

  auto ret = atomic_clean_pass_.Run(graph_);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(CountOfAtomicMemsetNode(graph_), 1);

  ret = atomic_clean_pass_.Run(subgraph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(CountOfAtomicMemsetNode(subgraph), 1);
}

/*     Op1                    Op1
 *      |           sub        |
 *   Op2(atomic) -------->  Op2(atomic) unknown
 *      |                      |
 *  NetOutput              NetOutput
 */
TEST_F(UtestGraphPassesAtomicAddrCleanPass, GraphWithOneTBEAtomicNodeWithUnknowSubgraph) {
  // graph
  {
    auto node1 = NewNode("Op1", DATA_TYPE, 0, 1);
    auto node2 = NewNode("Op2", RELU, 1, 1);
    OpDescPtr op_desc2 = node2->GetOpDesc();
    op_desc2->SetAttr("atomic_input_index", GeAttrValue::CreateFrom<GeAttrValue::LIST_INT>({123, 456}));
    op_desc2->SetOpKernelLibName("aicore_kernel");
    auto net_output = NewNode("NetOutput", NETOUTPUT, 3, 3);

    GraphUtils::AddEdge(node1->GetOutDataAnchor(0), node2->GetInDataAnchor(0));
    GraphUtils::AddEdge(node2->GetOutDataAnchor(0), net_output->GetInDataAnchor(1));
    graph_->SetName("root_graph");
    graph_->SetGraphUnknownFlag(true);
  }

  ComputeGraphPtr subgraph = make_shared<ComputeGraph>("sub_graph");
  // sub graph
  {
    auto node1 = NewNode(subgraph, "Op1", DATA_TYPE, 0, 1);
    auto node2 = NewNode(subgraph, "Op2", RELU, 1, 1);
    OpDescPtr op_desc2 = node2->GetOpDesc();
    op_desc2->SetAttr("atomic_input_index", GeAttrValue::CreateFrom<GeAttrValue::LIST_INT>({123, 456}));
    op_desc2->SetOpKernelLibName("aicore_kernel");
    auto net_output = NewNode(subgraph, "NetOutput", NETOUTPUT, 3, 3);

    GraphUtils::AddEdge(node1->GetOutDataAnchor(0), node2->GetInDataAnchor(0));
    GraphUtils::AddEdge(node2->GetOutDataAnchor(0), net_output->GetInDataAnchor(1));
    subgraph->SetName("sub_graph");
  }

  auto Op2 = graph_->FindNode("Op2");
  auto Op2_desc = Op2->GetOpDesc();
  Op2_desc->AddSubgraphName("sub_graph");
  Op2_desc->SetSubgraphInstanceName(0, "sub_graph");
  subgraph->SetParentNode(Op2);
  subgraph->SetParentGraph(graph_);
  subgraph->SetGraphUnknownFlag(true);
  graph_->AddSubgraph(subgraph);

  std::vector<int64_t> dims = {-1, 224, 224, 3};
  auto Op2_subgraph = subgraph->FindNode("Op2");
  auto Op2_subgraph_desc = Op2_subgraph->GetOpDesc();
  Op2_subgraph_desc->MutableOutputDesc(0)->SetShape(GeShape(dims));

  std::shared_ptr<GELib> instancePtr = ge::GELib::GetInstance();
  std::shared_ptr<OpsKernelManager> opsKernelManager =
      std::make_shared<OpsKernelManager>(instancePtr->OpsKernelManagerObj());
  auto opkernel = std::make_shared<TestOpsKernelInfoStore>();
  instancePtr->OpsKernelManagerObj().ops_kernel_store_.insert(std::make_pair("aicore_kernel", opkernel));
  auto ret = atomic_clean_pass_.Run(graph_);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(CountOfAtomicMemsetNode(graph_), 0);

  ret = atomic_clean_pass_.Run(subgraph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(CountOfAtomicMemsetNode(subgraph), 0);
}

TEST_F(UtestGraphPassesAtomicAddrCleanPass, GraphWithTwoAtomicNode_NoCommonAtomic) {
  auto node1 = NewNode("Op1", DATA_TYPE, 0, 1);
  auto node2 = NewNode("Op2", RELU, 1, 1);
  auto node3 = NewNode("Op3", RELU, 1, 1);
  auto net_output = NewNode("NetOutput", NETOUTPUT, 3, 3);

  OpDescPtr op_desc2 = node2->GetOpDesc();
  op_desc2->SetAttr("atomic_input_index", GeAttrValue::CreateFrom<GeAttrValue::LIST_INT>({123, 456}));
  (void)AttrUtils::SetStr(op_desc2, ATTR_NAME_STREAM_LABEL, "label1");

  OpDescPtr op_desc3 = node3->GetOpDesc();
  op_desc3->SetAttr("atomic_input_index", GeAttrValue::CreateFrom<GeAttrValue::LIST_INT>({123, 456}));

  vector<uint32_t> is_connect_netoutput = {1};
  (void)ge::AttrUtils::SetListInt(op_desc2, ATTR_NAME_NODE_CONNECT_OUTPUT, is_connect_netoutput);
  (void)ge::AttrUtils::SetListInt(op_desc3, ATTR_NAME_NODE_CONNECT_OUTPUT, is_connect_netoutput);

  GraphUtils::AddEdge(node1->GetOutDataAnchor(0), node2->GetInDataAnchor(0));
  GraphUtils::AddEdge(node2->GetOutDataAnchor(0), node3->GetInDataAnchor(1));
  GraphUtils::AddEdge(node3->GetOutDataAnchor(0), net_output->GetInDataAnchor(1));
  graph_->AddInputNode(node1);
  (void)AttrUtils::SetStr(graph_, ATTR_NAME_SESSION_GRAPH_ID, "012");
  auto ret = atomic_clean_pass_.Run(graph_);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(2, CountOfAtomicMemsetNode());
}

TEST_F(UtestGraphPassesAtomicAddrCleanPass, GraphWithTwoAtomicNode_Failed) {
  auto node1 = NewNode("Op1", DATA_TYPE, 0, 1);
  auto node2 = NewNode("Op2", RELU, 1, 1);
  auto node3 = NewNode("Op3", RELU, 1, 1);
  auto net_output = NewNode("NetOutput", NETOUTPUT, 3, 3);

  OpDescPtr op_desc2 = node2->GetOpDesc();
  op_desc2->SetAttr("atomic_input_index", GeAttrValue::CreateFrom<GeAttrValue::LIST_INT>({123, 456}));
  (void)AttrUtils::SetStr(op_desc2, ATTR_NAME_STREAM_LABEL, "label1");

  OpDescPtr op_desc3 = node3->GetOpDesc();
  op_desc3->SetAttr("atomic_input_index", GeAttrValue::CreateFrom<GeAttrValue::LIST_INT>({123, 456}));

  vector<uint32_t> is_connect_netoutput = {1};
  (void)ge::AttrUtils::SetListInt(op_desc2, ATTR_NAME_NODE_CONNECT_OUTPUT, is_connect_netoutput);
  (void)ge::AttrUtils::SetListInt(op_desc3, ATTR_NAME_NODE_CONNECT_OUTPUT, is_connect_netoutput);

  GraphUtils::AddEdge(node1->GetOutDataAnchor(0), node2->GetInDataAnchor(0));
  GraphUtils::AddEdge(node2->GetOutDataAnchor(0), node3->GetInDataAnchor(1));
  GraphUtils::AddEdge(node3->GetOutDataAnchor(0), net_output->GetInDataAnchor(1));
  graph_->AddInputNode(node1);
  (void)AttrUtils::SetStr(graph_, ATTR_NAME_SESSION_GRAPH_ID, "012");

  //MOCKER_CPP(&AtomicAddrCleanPass::LinkToAtomicNode).stubs().will(returnValue(FAILED));
  auto ret = atomic_clean_pass_.Run(graph_);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(2, CountOfAtomicMemsetNode());
}

/*
 *     Data                       Data  Atomic_clean
 *      |                           |   /  |
 *     relu                         relu   |
 *      |               ==>           |    |
 *    relu(atomic)               relu(atomic)
 *      |                             |
 *   netoutput                    netoutput
 */
TEST_F(UtestGraphPassesAtomicAddrCleanPass, pass_run_success) {
  auto node1 = NewNode("node1", DATA, 0, 1);

  auto node2 = NewNode("node2", RELU, 1, 1);
  auto node3 = NewNode("node3", RELU, 1, 1);
  auto op_desc = node3->GetOpDesc();
  vector<int64_t> atomic_input_index = {123, 456};
  AttrUtils::SetListInt(op_desc, "atomic_input_index", atomic_input_index);

  auto node4 = NewNode("node4", NETOUTPUT, 1, 0);
  GraphUtils::AddEdge(node1->GetOutDataAnchor(0), node2->GetInDataAnchor(0));
  GraphUtils::AddEdge(node2->GetOutDataAnchor(0), node3->GetInDataAnchor(0));
  GraphUtils::AddEdge(node3->GetOutDataAnchor(0), node4->GetInDataAnchor(0));
  AtomicAddrCleanPass atomi_addr_clean_pass;
  Status ret = atomi_addr_clean_pass.Run(graph_);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(1, CountOfAtomicMemsetNode());

  auto atomic_clean = graph_->FindFirstNodeMatchType(ATOMICADDRCLEAN);
  ASSERT_NE(atomic_clean, nullptr);
  auto out_ctrl_nodes = atomic_clean->GetOutControlNodes();
  EXPECT_EQ(out_ctrl_nodes.size(), 2);
}

TEST_F(UtestGraphPassesAtomicAddrCleanPass, test_skip_insert_fail) {
  graph_ = std::make_shared<ComputeGraph>("test_graph");
  GeTensorDesc tensor_desc(GeShape(), ge::FORMAT_NCHW, ge::DT_FLOAT);
  auto const_desc = std::make_shared<OpDesc>("const", CONSTANTOP);
  const_desc->AddOutputDesc(tensor_desc);
  auto const_node = graph_->AddNode(const_desc);

  GeTensorDesc test_tensor_desc(GeShape(), ge::FORMAT_NCHW, ge::DT_FLOAT);
  auto test_op_desc = std::make_shared<OpDesc>("test", "Div");
  test_op_desc->AddInputDesc(test_tensor_desc);
  test_op_desc->AddOutputDesc(test_tensor_desc);
  std::vector<int32_t> input_indexes = {0};
  // (void)ge::AttrUtils::SetListInt(test_op_desc, ATOMIC_ATTR_INPUT_INDEX, input_indexes);
  (void)ge::AttrUtils::SetListInt(test_op_desc, ATOMIC_ATTR_OUTPUT_INDEX, input_indexes);
  std::map<string, std::map<int64_t, int64_t>> sub_node_workspace_info;
  test_op_desc->SetExtAttr(ge::EXT_ATTR_ATOMIC_WORKSPACE_INFO, sub_node_workspace_info);
  auto test_node = graph_->AddNode(test_op_desc);

  auto out_desc = std::make_shared<OpDesc>("OUT", NETOUTPUT);
  out_desc->AddInputDesc(tensor_desc);
  auto out_node = graph_->AddNode(out_desc);
  EXPECT_EQ(GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), test_node->GetInDataAnchor(0)), SUCCESS);
  EXPECT_EQ(GraphUtils::AddEdge(test_node->GetOutDataAnchor(0), out_node->GetInDataAnchor(0)), SUCCESS);

  AtomicAddrCleanPass atomi_addr_clean_pass;
  EXPECT_EQ(atomi_addr_clean_pass.CheckSkipInsertInLoopGraph(test_node), false);
}

TEST_F(UtestGraphPassesAtomicAddrCleanPass, test_clear_status_succ) {
  AtomicAddrCleanPass atomi_addr_clean_pass;
  EXPECT_EQ(atomi_addr_clean_pass.ClearStatus(), SUCCESS);
}

TEST_F(UtestGraphPassesAtomicAddrCleanPass, test_check_ops_kernel_fail) {
  graph_ = std::make_shared<ComputeGraph>("test_graph");
  GeTensorDesc tensor_desc(GeShape(), ge::FORMAT_NCHW, ge::DT_FLOAT);
  auto const_desc = std::make_shared<OpDesc>("const", CONSTANTOP);
  const_desc->AddOutputDesc(tensor_desc);
  auto const_node = graph_->AddNode(const_desc);

  AtomicAddrCleanPass atomi_addr_clean_pass;
  EXPECT_EQ(atomi_addr_clean_pass.CheckAtomicFromOpsKernel(const_node), false);
}

TEST_F(UtestGraphPassesAtomicAddrCleanPass, test_link_node_fail) {
  graph_ = std::make_shared<ComputeGraph>("test_graph");
  GeTensorDesc tensor_desc(GeShape(), ge::FORMAT_NCHW, ge::DT_FLOAT);
  auto const_desc = std::make_shared<OpDesc>("const", CONSTANTOP);
  auto const_node = graph_->AddNode(const_desc);

  AtomicAddrCleanPass atomi_addr_clean_pass;
  EXPECT_EQ(atomi_addr_clean_pass.LinkToAtomicNode(const_node, const_node), SUCCESS);
}

TEST_F(UtestGraphPassesAtomicAddrCleanPass, test_ge_init_fail) {
  graph_ = std::make_shared<ComputeGraph>("test_graph");
  GeTensorDesc tensor_desc(GeShape(), ge::FORMAT_NCHW, ge::DT_FLOAT);
  auto const_desc = std::make_shared<OpDesc>("const", CONSTANTOP);
  auto const_node = graph_->AddNode(const_desc);
  const std::vector<NodePtr> node_list = {const_node};

  GEFinalize();
  AtomicAddrCleanPass atomi_addr_clean_pass;
  EXPECT_EQ(atomi_addr_clean_pass.CallCompileOp(node_list), ge::GE_CLI_GE_NOT_INITIALIZED);
  std::map<AscendString, AscendString> options;
  GEInitialize(options);
}

}  // namespace ge
