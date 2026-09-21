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
#include "graph/fuzz_compile/fuzz_compile_bin_selector.h"
#include "engines/manager/opskernel_manager/ops_kernel_builder_manager.h"
#include "engines/manager/opskernel_manager/ops_kernel_manager.h"
#include "api/gelib/gelib.h"
#include "macro_utils/dt_public_unscope.h"
#include "graph/passes/graph_builder_utils.h"
#include "graph/bin_cache/node_bin_selector_factory.h"
#include "graph/tuning_utils.h"

namespace ge {
using namespace hybrid;
class UtestFuzzCompileBinSelector : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {
    auto instance_ptr = ge::GELib::GetInstance();
    if (instance_ptr != nullptr) {
      instance_ptr->Finalize();
    }
  }
};

namespace {
class FakeOpsKernelInfoStore : public OpsKernelInfoStore {
 private:
  Status Initialize(const std::map<std::string, std::string> &options) override {
    return SUCCESS;
  };
  Status Finalize() override {
    return SUCCESS;
  };
  bool CheckSupported(const OpDescPtr &op_desc, std::string &reason) const override {
    return false;
  };
  void GetAllOpsKernelInfo(std::map<std::string, ge::OpInfo> &infos) const override{};

  Status FuzzCompileOp(std::vector<NodePtr> &node_vec) override {
    return SUCCESS;
  };
};

class FakeOpsKernelBuilder : public OpsKernelBuilder {
 public:
  FakeOpsKernelBuilder() = default;

 private:
  Status Initialize(const map<std::string, std::string> &options) override {
    return SUCCESS;
  };
  Status Finalize() override {
    return SUCCESS;
  };
  Status CalcOpRunningParam(Node &node) override {
    return SUCCESS;
  };
  Status GenerateTask(const Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) override {
    domi::TaskDef task_def;
    tasks.push_back(task_def);
    return SUCCESS;
  };
};
}  // namespace

TEST_F(UtestFuzzCompileBinSelector, do_compile_op_succces) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto node = builder.AddNode("Data", "Data", 1, 1);
  node->GetOpDesc()->SetOpKernelLibName("test");
  OpsKernelBuilderRegistry::GetInstance().kernel_builders_["test"] = std::make_shared<FakeOpsKernelBuilder>();
  FuzzCompileBinSelector bin_selector;
  bin_selector.aicore_kernel_store_ = std::make_shared<FakeOpsKernelInfoStore>();
  EXPECT_EQ(bin_selector.DoCompileOp(node), SUCCESS);
  OpsKernelBuilderRegistry::GetInstance().UnregisterAll();
}

TEST_F(UtestFuzzCompileBinSelector, do_initialize_success) {
  map<std::string, std::string> options;
  options.emplace(BUILD_MODE, BUILD_MODE_TUNING);
  options.emplace(BUILD_STEP, BUILD_STEP_AFTER_BUILDER);
  (void)ge::GELib::Initialize(options);
  OpsKernelInfoStorePtr kernel_ptr = std::make_shared<FakeOpsKernelInfoStore>();
  OpsKernelManager::GetInstance().ops_kernel_store_["AIcoreEngine"] = kernel_ptr;
  FuzzCompileBinSelector bin_selector;
  EXPECT_EQ(bin_selector.Initialize(), SUCCESS);
  OpsKernelManager::GetInstance().ops_kernel_info_.clear();
}

TEST_F(UtestFuzzCompileBinSelector, do_initialize_failed) {
  OpsKernelInfoStorePtr kernel_ptr = std::make_shared<FakeOpsKernelInfoStore>();
  FuzzCompileBinSelector bin_selector;
  map<std::string, std::string> options;
  options.emplace(BUILD_MODE, BUILD_MODE_TUNING);
  options.emplace(BUILD_STEP, BUILD_STEP_AFTER_BUILDER);
  (void)ge::GELib::Initialize(options);
  GELib::GetInstance()->init_flag_ = false;
  EXPECT_EQ(bin_selector.Initialize(), GE_CLI_GE_NOT_INITIALIZED);
}

TEST_F(UtestFuzzCompileBinSelector, do_select_bin_success) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto node = builder.AddNode("Data", "Data", 1, 1);
  FuzzCompileBinSelector bin_selector;
  node->GetOpDesc()->SetOpKernelLibName("AIcoreEngine");
  NodeCompileCacheModule nccm;
  bin_selector.nccm_ = &nccm;
  bin_selector.aicore_kernel_store_ = std::make_shared<FakeOpsKernelInfoStore>();
  OpsKernelBuilderRegistry::GetInstance().kernel_builders_["AIcoreEngine"] = std::make_shared<FakeOpsKernelBuilder>();
  GEThreadLocalContext ge_context;
  std::vector<domi::TaskDef> task_defs;

  // set compile info on node
  ge::AttrUtils::SetStr(node->GetOpDesc(), "compile_info_key", "op_compile_info_key");
  ge::AttrUtils::SetStr(node->GetOpDesc(), "compile_info_json", "op_compile_info_json");
  EXPECT_NE(bin_selector.SelectBin(node, &ge_context, task_defs), nullptr);
}

TEST_F(UtestFuzzCompileBinSelector, do_select_bin_success_without_kernel_miss) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto node = builder.AddNode("Data", "Data", 1, 1);
  FuzzCompileBinSelector bin_selector;
  node->GetOpDesc()->SetOpKernelLibName("AIcoreEngine");
  NodeCompileCacheModule nccm;
  bin_selector.nccm_ = &nccm;
  // add node compile cache
  NodeCompileCacheItem item;
  auto add_item = bin_selector.nccm_->AddCompileCache(node, item);
  ASSERT_NE(add_item, nullptr);
  
  /*
  bin_selector.aicore_kernel_store_ = std::make_shared<FakeOpsKernelInfoStore>();
  OpsKernelBuilderRegistry::GetInstance().kernel_builders_["AIcoreEngine"] = std::make_shared<FakeOpsKernelBuilder>();
  */
  GEThreadLocalContext ge_context;
  std::vector<domi::TaskDef> task_defs;

  // set compile info on node
  ge::AttrUtils::SetStr(node->GetOpDesc(), "compile_info_key", "op_compile_info_key");
  ge::AttrUtils::SetStr(node->GetOpDesc(), "compile_info_json", "op_compile_info_json");
  EXPECT_NE(bin_selector.SelectBin(node, &ge_context, task_defs), nullptr);
}
}  // namespace ge
