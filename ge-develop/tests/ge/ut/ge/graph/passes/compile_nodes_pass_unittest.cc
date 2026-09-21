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
#include <map>
#include "macro_utils/dt_public_scope.h"
#include "graph/passes/feature/compile_nodes_pass.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/node_adapter.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph_builder_utils.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "engines/manager/opskernel_manager/ops_kernel_manager.h"
#include "ge/ge_api.h"
#include "api/gelib/gelib.h"

using namespace std;
using namespace ge;

namespace {
class TestOpsKernelInfoStore : public OpsKernelInfoStore {
 public:
  TestOpsKernelInfoStore() = default;

  ~TestOpsKernelInfoStore() override = default;

  Status Initialize(const map<string, string> &options)  { return ge::SUCCESS; }
  Status Finalize() override { return ge::SUCCESS; }
  bool CheckSupported(const OpDescPtr &opDescPtr, std::string &un_supported_reason) const { return true; }
  bool CheckAccuracySupported(const ge::NodePtr &node, std::string &un_supported_reason,
                              const bool realQuery = false) const {
    bool is_support = false;
    AttrUtils::GetBool(node->GetOpDesc(), "is_supported", is_support);
    if (is_support) {
     return true;
    }

    if (!un_supported_reason.empty()) {
      bool has_attr = node->GetOpDesc()->HasAttr("add_reason");
      if (has_attr) {
        return false;
      }
      return true;
    }

    un_supported_reason = "not_supported";
    return false;
  }
  Status CalcOpRunningParam(Node &node) { return ge::SUCCESS; }
  Status GenerateTask(const Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks)  {
    return ge::SUCCESS; }
  Status CompileOp(vector<ge::NodePtr> &node_vec) {
    for (const auto &node:node_vec){
      bool has_attr = node->GetOpDesc()->HasAttr("compile_fail");
      if (has_attr) {
        return false;
      }
      return true;
    }
    return ge::SUCCESS; }
  void GetAllOpsKernelInfo(map<string, OpInfo> &infos) const { return; }

  std::vector<OpInfo> GetOpsKernelInfoStore(const std::string &name) {
    return empty_op_info_;
  }
  OpInfo test_op = {"DNN_VM_TF", "ge_local", 0, false, false, true};
  std::vector<OpInfo> empty_op_info_{test_op};
};
}

class UtestCompileNodesPass : public testing::Test {
 protected:
  void SetUp() {
    std::map<AscendString, AscendString> options;
    GEInitialize(options);
    ge::GELib::GetInstance()->OpsKernelManagerObj().ops_kernel_store_.clear();
  }
  void TearDown() {
    GEFinalize();
  }
  void BuildGraph() {
    GeTensorDesc td(GeShape({2,2,2,2}), ge::FORMAT_NCHW, ge::DT_INT16);

    auto pre_op_desc = std::make_shared<OpDesc>("pre", "Constant");
    pre_op_desc->AddOutputDesc(td);
    (void)ge::AttrUtils::SetBool(pre_op_desc, ATTR_NEED_COMPILE, false);
    pre_node_ = graph_->AddNode(pre_op_desc);

    auto cur_op_desc = std::make_shared<OpDesc>("bitcat", "Bitcast");
    cur_op_desc->AddInputDesc(td);
    cur_op_desc->AddOutputDesc(td);
    (void)ge::AttrUtils::SetBool(cur_op_desc, ATTR_NEED_COMPILE, true);
    cur_node_ = graph_->AddNode(cur_op_desc);

    auto peer_op_desc = std::make_shared<OpDesc>("relu", RELU);
    peer_op_desc->AddInputDesc(td);
    peer_op_desc->AddOutputDesc(td);
    peer_node_ = graph_->AddNode(peer_op_desc);

    GraphUtils::AddEdge(pre_node_->GetOutDataAnchor(0), cur_node_->GetInDataAnchor(0));
    GraphUtils::AddEdge(cur_node_->GetOutDataAnchor(0), peer_node_->GetInDataAnchor(0));
  }

  ComputeGraphPtr graph_;
  CompileNodesPass pass_;
  NodePtr cur_node_;
  NodePtr pre_node_;
  NodePtr peer_node_;
};

TEST_F(UtestCompileNodesPass, null_graph_succ) {
  EXPECT_EQ(pass_.Run(nullptr), GRAPH_SUCCESS);
}

TEST_F(UtestCompileNodesPass, not_init_fail) {
  graph_ = std::make_shared<ComputeGraph>("g");
  BuildGraph();
  EXPECT_NE(cur_node_, nullptr);
  EXPECT_NE(pre_node_, nullptr);
  EXPECT_NE(peer_node_, nullptr);

  GEFinalize();
  EXPECT_EQ(pass_.Run(graph_), GE_CLI_GE_NOT_INITIALIZED);
  EXPECT_EQ(graph_->GetDirectNodesSize(), 3);
  std::map<AscendString, AscendString> options;
  GEInitialize(options);
}

TEST_F(UtestCompileNodesPass, no_kernel_fail) {
  graph_ = std::make_shared<ComputeGraph>("g");
  BuildGraph();
  EXPECT_NE(cur_node_, nullptr);
  EXPECT_NE(pre_node_, nullptr);
  EXPECT_NE(peer_node_, nullptr);

  EXPECT_EQ(pass_.Run(graph_), GRAPH_FAILED);
  EXPECT_EQ(graph_->GetDirectNodesSize(), 3);
}

TEST_F(UtestCompileNodesPass, no_kernel_info_fail) {
  graph_ = std::make_shared<ComputeGraph>("g");
  BuildGraph();
  EXPECT_NE(cur_node_, nullptr);
  EXPECT_NE(pre_node_, nullptr);
  EXPECT_NE(peer_node_, nullptr);

  std::shared_ptr<GELib> instancePtr = ge::GELib::GetInstance();
  std::shared_ptr<OpsKernelManager> opsKernelManager =
      std::make_shared<OpsKernelManager>(instancePtr->OpsKernelManagerObj());
  auto opkernel = std::make_shared<TestOpsKernelInfoStore>();
  instancePtr->OpsKernelManagerObj().ops_kernel_store_.insert(std::make_pair("aicore_kernel", opkernel));

  std::string op_kernel_lib_name = "hh";
  AttrUtils::SetStr(cur_node_->GetOpDesc(), "_ge_attr_op_kernel_lib_name", op_kernel_lib_name);

  EXPECT_EQ(pass_.Run(graph_), GRAPH_FAILED);
  EXPECT_EQ(graph_->GetDirectNodesSize(), 3);
}

TEST_F(UtestCompileNodesPass, supported_fail) {
  graph_ = std::make_shared<ComputeGraph>("g");
  BuildGraph();
  EXPECT_NE(cur_node_, nullptr);
  EXPECT_NE(pre_node_, nullptr);
  EXPECT_NE(peer_node_, nullptr);

  std::shared_ptr<GELib> instancePtr = ge::GELib::GetInstance();
  std::shared_ptr<OpsKernelManager> opsKernelManager =
      std::make_shared<OpsKernelManager>(instancePtr->OpsKernelManagerObj());
  auto opkernel = std::make_shared<TestOpsKernelInfoStore>();
  instancePtr->OpsKernelManagerObj().ops_kernel_store_.insert(std::make_pair("aicore_kernel", opkernel));

  std::string op_kernel_lib_name = "aicore_kernel";
  AttrUtils::SetStr(cur_node_->GetOpDesc(), "_ge_attr_op_kernel_lib_name", op_kernel_lib_name);

  AttrUtils::SetBool(cur_node_->GetOpDesc(), "is_supported", true);

  EXPECT_NE(pass_.Run(graph_), GRAPH_SUCCESS);
  EXPECT_EQ(graph_->GetDirectNodesSize(), 3);
}

TEST_F(UtestCompileNodesPass, supported_succ_2) {
  graph_ = std::make_shared<ComputeGraph>("g");
  BuildGraph();
  EXPECT_NE(cur_node_, nullptr);
  EXPECT_NE(pre_node_, nullptr);
  EXPECT_NE(peer_node_, nullptr);

  std::shared_ptr<GELib> instancePtr = ge::GELib::GetInstance();
  std::shared_ptr<OpsKernelManager> opsKernelManager =
      std::make_shared<OpsKernelManager>(instancePtr->OpsKernelManagerObj());
  auto opkernel = std::make_shared<TestOpsKernelInfoStore>();
  instancePtr->OpsKernelManagerObj().ops_kernel_store_.insert(std::make_pair("aicpu_tf_kernel", opkernel));

  std::string op_kernel_lib_name = "aicpu_tf_kernel";
  AttrUtils::SetStr(cur_node_->GetOpDesc(), "_ge_attr_op_kernel_lib_name", op_kernel_lib_name);

  AttrUtils::SetBool(cur_node_->GetOpDesc(), "is_supported", true);

  EXPECT_EQ(pass_.Run(graph_), GRAPH_SUCCESS);
  EXPECT_EQ(graph_->GetDirectNodesSize(), 3);

  // test repass
  EXPECT_EQ(pass_.Run(graph_), GRAPH_SUCCESS);
}

TEST_F(UtestCompileNodesPass, unsupported_reason_fail) {
  graph_ = std::make_shared<ComputeGraph>("g");
  BuildGraph();
  EXPECT_NE(cur_node_, nullptr);
  EXPECT_NE(pre_node_, nullptr);
  EXPECT_NE(peer_node_, nullptr);

  std::shared_ptr<GELib> instancePtr = ge::GELib::GetInstance();
  std::shared_ptr<OpsKernelManager> opsKernelManager =
      std::make_shared<OpsKernelManager>(instancePtr->OpsKernelManagerObj());
  auto opkernel = std::make_shared<TestOpsKernelInfoStore>();
  instancePtr->OpsKernelManagerObj().ops_kernel_store_.insert(std::make_pair("aicore_kernel", opkernel));
  auto opkernel2 = std::make_shared<TestOpsKernelInfoStore>();
  instancePtr->OpsKernelManagerObj().ops_kernel_store_.insert(std::make_pair("aicore_kernel2", opkernel2));
  auto opkernel3 = std::make_shared<TestOpsKernelInfoStore>();
  instancePtr->OpsKernelManagerObj().ops_kernel_store_.insert(std::make_pair("aicore_kernel3", opkernel3));

  std::string op_kernel_lib_name = "aicore_kernel";
  AttrUtils::SetStr(cur_node_->GetOpDesc(), "_ge_attr_op_kernel_lib_name", op_kernel_lib_name);

  EXPECT_NE(pass_.Run(graph_), GRAPH_SUCCESS);
  EXPECT_EQ(graph_->GetDirectNodesSize(), 3);
}

TEST_F(UtestCompileNodesPass, unsupported_reasons_fail) {
  graph_ = std::make_shared<ComputeGraph>("g");
  BuildGraph();
  EXPECT_NE(cur_node_, nullptr);
  EXPECT_NE(pre_node_, nullptr);
  EXPECT_NE(peer_node_, nullptr);

  std::shared_ptr<GELib> instancePtr = ge::GELib::GetInstance();
  std::shared_ptr<OpsKernelManager> opsKernelManager =
      std::make_shared<OpsKernelManager>(instancePtr->OpsKernelManagerObj());
  auto opkernel = std::make_shared<TestOpsKernelInfoStore>();
  instancePtr->OpsKernelManagerObj().ops_kernel_store_.insert(std::make_pair("aicore_kernel", opkernel));
  auto opkernel2 = std::make_shared<TestOpsKernelInfoStore>();
  instancePtr->OpsKernelManagerObj().ops_kernel_store_.insert(std::make_pair("aicore_kernel2", opkernel2));
  auto opkernel3 = std::make_shared<TestOpsKernelInfoStore>();
  instancePtr->OpsKernelManagerObj().ops_kernel_store_.insert(std::make_pair("aicore_kernel3", opkernel3));

  std::string op_kernel_lib_name = "aicore_kernel";
  AttrUtils::SetStr(cur_node_->GetOpDesc(), "_ge_attr_op_kernel_lib_name", op_kernel_lib_name);

  AttrUtils::SetBool(cur_node_->GetOpDesc(), "add_reason", true);

  EXPECT_EQ(pass_.Run(graph_), GRAPH_FAILED);
  EXPECT_EQ(graph_->GetDirectNodesSize(), 3);
}

TEST_F(UtestCompileNodesPass, no_node_succ) {
  graph_ = std::make_shared<ComputeGraph>("g");
  EXPECT_EQ(pass_.Run(graph_), GRAPH_SUCCESS);
}

TEST_F(UtestCompileNodesPass, compile_nodes_succ) {
  std::shared_ptr<GELib> instance = ge::GELib::GetInstance();
  std::unordered_map<std::string, std::vector<NodePtr>> kernel_to_compile_nodes;
  std::vector<NodePtr> node_vec;
  kernel_to_compile_nodes.emplace("hhh", node_vec);
  EXPECT_EQ(pass_.CompileNodes(instance, kernel_to_compile_nodes), GE_GRAPH_PARAM_NULLPTR);

  std::shared_ptr<GELib> instancePtr = ge::GELib::GetInstance();
  std::shared_ptr<OpsKernelManager> opsKernelManager =
      std::make_shared<OpsKernelManager>(instancePtr->OpsKernelManagerObj());
  auto opkernel = std::make_shared<TestOpsKernelInfoStore>();
  instancePtr->OpsKernelManagerObj().ops_kernel_store_.insert(std::make_pair("aicore_kernel", opkernel));

  graph_ = std::make_shared<ComputeGraph>("g");
  GeTensorDesc td(GeShape({2,2,2,2}), ge::FORMAT_NCHW, ge::DT_INT16);
  auto test_op_desc = std::make_shared<OpDesc>("pre", "test_node");
  test_op_desc->AddOutputDesc(td);
  (void)ge::AttrUtils::SetBool(test_op_desc, "compile_fail", false);
  NodePtr test_node = graph_->AddNode(test_op_desc);

  kernel_to_compile_nodes.clear();
  node_vec.push_back(test_node);
  kernel_to_compile_nodes.emplace("aicore_kernel", node_vec);

  EXPECT_EQ(pass_.CompileNodes(instance, kernel_to_compile_nodes), GRAPH_SUCCESS);
}
