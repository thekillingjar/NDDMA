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
#include "graph/compute_graph.h"
#include "common/graph_tuner_errorcode.h"
#include "fe_llt_utils.h"
#define protected public
#define private public
#include "common/configuration.h"
#include "common/platform_utils.h"
#include "graph_optimizer/fe_graph_optimizer.h"
#include "graph_optimizer/op_compiler/op_compiler.h"
#include "graph_optimizer/lx_fusion_optimizer/lx_fusion_optimizer.h"
#include "common/fusion_statistic/fusion_statistic_writer.h"
#include "graph/ge_local_context.h"
#undef private
#undef protected
using namespace ge;
namespace fe {
namespace {
const std::string kFixPipeOpt = "{\"fixpipe_fusion\":true}";
tune::Status L2FusionPreOptimizer1(ge::ComputeGraph &, AOEOption, const std::vector<std::string> &,
                                   std::vector<PassChangeInfo> &pass_vec) {
  PassChangeInfo pass1{"TbeConvStridedwriteFusionPass", 10, 5};
  PassChangeInfo pass3{"MatmulConfusiontransposeUbFusion", 3, 10};
  pass_vec.push_back(pass1);
  pass_vec.push_back(pass3);
  return tune::NO_FUSION_STRATEGY;
}

tune::Status L2FusionPreOptimizer2(ge::ComputeGraph &, AOEOption, const std::vector<std::string> &,
                                   std::vector<PassChangeInfo> &pass_vec) {
  PassChangeInfo pass1{"TbeConvStridedwriteFusionPass", 10, 5};
  PassChangeInfo pass3{"MatmulConfusiontransposeUbFusion", 3, 10};
  pass_vec.push_back(pass1);
  pass_vec.push_back(pass3);
  return tune::SUCCESS;
}

tune::Status L2FusionPreOptimizer3(ge::ComputeGraph &, AOEOption, const std::vector<std::string> &,
                                   std::vector<PassChangeInfo> &pass_vec) {
  vector<int64_t> recovery_scope_ids = {1, 2, 3};
  PassChangeInfo pass1{"TbeConvStridedwriteFusionPass", 10, 5, recovery_scope_ids};
  PassChangeInfo pass3{"MatmulConfusiontransposeUbFusion", 3, 10, recovery_scope_ids};
  pass_vec.push_back(pass1);
  pass_vec.push_back(pass3);
  return tune::HIT_FUSION_STRATEGY;
}

FusionPriorityMgrPtr fusion_priority_mgr;
FEOpsKernelInfoStorePtr ops_kernel_info_store;
BufferOptimize buffer_optimize;
}

class LxFusionOptimizerUT : public testing::Test {
public:
  static void SetUpTestCase() {
    FEOpsKernelInfoStorePtr ops_kernel_info_store = std::make_shared<FEOpsKernelInfoStore>();
    FusionRuleManagerPtr fusion_rule_mgr = std::make_shared<FusionRuleManager>(ops_kernel_info_store);
    fusion_priority_mgr = std::make_shared<FusionPriorityManager>(AI_CORE_NAME, fusion_rule_mgr);
    Configuration::Instance(AI_CORE_NAME).InitLibPath();
    buffer_optimize = Configuration::Instance(fe::AI_CORE_NAME).GetBufferOptimize();
    PlatformUtils::Instance().soc_version_ = "Ascend310P3";
    GetThreadLocalContext().graph_options_.emplace("opt_module.aoe", "");
  }
  static void TearDownTestCase() {
    Configuration::Instance(fe::AI_CORE_NAME).config_param_vec_[static_cast<size_t>(CONFIG_PARAM::BufferOptimize)] =
            static_cast<int64_t>(buffer_optimize);
  }
};

TEST_F(LxFusionOptimizerUT, init_and_finalize) {
  LxFusionOptimizerPtr lx_fusion_optimizer = std::make_shared<LxFusionOptimizer>(fusion_priority_mgr, ops_kernel_info_store);
  EXPECT_EQ(lx_fusion_optimizer->Initialize(), fe::SUCCESS);
  EXPECT_EQ(lx_fusion_optimizer->Initialize(), fe::SUCCESS);
  EXPECT_EQ(lx_fusion_optimizer->Finalize(), fe::SUCCESS);
  EXPECT_EQ(lx_fusion_optimizer->Finalize(), fe::SUCCESS);
  EXPECT_EQ(lx_fusion_optimizer->Initialize(), fe::SUCCESS);
  EXPECT_EQ(lx_fusion_optimizer->Finalize(), fe::SUCCESS);
}

TEST_F(LxFusionOptimizerUT, lx_fusion_Finalize) {
  LxFusionOptimizerPtr lx_fusion_optimizer = std::make_shared<LxFusionOptimizer>(fusion_priority_mgr, ops_kernel_info_store);
  EXPECT_EQ(lx_fusion_optimizer->Initialize(), fe::SUCCESS);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  EXPECT_EQ(lx_fusion_optimizer->LxFusionFinalize(*graph), fe::SUCCESS);
  AttrUtils::SetBool(graph, "_is_lx_fusion_finalize_failed", true);
  EXPECT_EQ(lx_fusion_optimizer->LxFusionFinalize(*graph), fe::FAILED);
}

TEST_F(LxFusionOptimizerUT, l1_fusion_recovery) {
  LxFusionOptimizerPtr lx_fusion_optimizer = std::make_shared<LxFusionOptimizer>(fusion_priority_mgr, ops_kernel_info_store);
  EXPECT_EQ(lx_fusion_optimizer->Initialize(), fe::SUCCESS);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr op_desc = std::make_shared<OpDesc>("relu", "Relu");
  GeTensorDesc output_desc;
  op_desc->AddOutputDesc(output_desc);
  ge::AttrUtils::SetStr(op_desc, kFusionOpBuildOptions, kFixPipeOpt);
  NodePtr node = graph->AddNode(op_desc);
  std::vector<ge::NodePtr> buff_fus_compile_failed_nodes = {node};
  std::vector<ge::NodePtr> buff_fus_rollback_nodes = {node};
  std::vector<ge::NodePtr> buff_fus_to_del_nodes = {node};
  Configuration::Instance(AI_CORE_NAME).config_param_vec_[static_cast<size_t>(CONFIG_PARAM::BufferOptimize)] =
          static_cast<int64_t>(EN_L1_OPTIMIZE);
  Status ret = lx_fusion_optimizer->LxFusionRecovery(*graph, buff_fus_compile_failed_nodes,
                                                     buff_fus_rollback_nodes, buff_fus_to_del_nodes);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(LxFusionOptimizerUT, l2_fusion_recovery) {
  LxFusionOptimizerPtr lx_fusion_optimizer = std::make_shared<LxFusionOptimizer>(fusion_priority_mgr, ops_kernel_info_store);
  EXPECT_EQ(lx_fusion_optimizer->Initialize(), fe::SUCCESS);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr op_desc = std::make_shared<OpDesc>("relu", "Relu");
  GeTensorDesc output_desc;
  op_desc->AddOutputDesc(output_desc);
  ge::AttrUtils::SetStr(op_desc, kFusionOpBuildOptions, kFixPipeOpt);
  NodePtr node = graph->AddNode(op_desc);
  std::vector<ge::NodePtr> buff_fus_compile_failed_nodes = {node};
  std::vector<ge::NodePtr> buff_fus_rollback_nodes = {node};
  std::vector<ge::NodePtr> buff_fus_to_del_nodes = {node};
  Configuration::Instance(AI_CORE_NAME).config_param_vec_[static_cast<size_t>(CONFIG_PARAM::BufferOptimize)] =
          static_cast<int64_t>(EN_L2_OPTIMIZE);
  Status ret = lx_fusion_optimizer->LxFusionRecovery(*graph, buff_fus_compile_failed_nodes,
                                                     buff_fus_rollback_nodes, buff_fus_to_del_nodes);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(LxFusionOptimizerUT, lx_fusion_Func_not_init) {
  LxFusionOptimizerPtr lx_fusion_optimizer = std::make_shared<LxFusionOptimizer>(fusion_priority_mgr, ops_kernel_info_store);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  auto tmp_path = Configuration::Instance(AI_CORE_NAME).lib_path_;
  Configuration::Instance(AI_CORE_NAME).lib_path_ = "test_empty_path";
  EXPECT_EQ(lx_fusion_optimizer->Initialize(), fe::SUCCESS);
  
  OptimizeConfig oCfg;
  lx_fusion_optimizer->L2FusionOptimize(*graph, oCfg);
  LxFusionOptimizeResult res = LxFusionOptimizeResult::NO_FUSION_STRATEGY;
  lx_fusion_optimizer->DoLxFusionOptimize(*graph, res);

  EXPECT_EQ(lx_fusion_optimizer->LxFusionFinalize(*graph), fe::SUCCESS);
  Configuration::Instance(AI_CORE_NAME).lib_path_ = tmp_path;
}

TEST_F(LxFusionOptimizerUT, l1_fusion_Func_not_init_recovery) {
  LxFusionOptimizerPtr lx_fusion_optimizer = std::make_shared<LxFusionOptimizer>(fusion_priority_mgr, ops_kernel_info_store);
  auto tmp_path = Configuration::Instance(AI_CORE_NAME).lib_path_;
  Configuration::Instance(AI_CORE_NAME).lib_path_ = "test_empty_path";
  EXPECT_EQ(lx_fusion_optimizer->Initialize(), fe::SUCCESS);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr op_desc = std::make_shared<OpDesc>("relu", "Relu");
  GeTensorDesc output_desc;
  op_desc->AddOutputDesc(output_desc);
  ge::AttrUtils::SetStr(op_desc, kFusionOpBuildOptions, kFixPipeOpt);
  NodePtr node = graph->AddNode(op_desc);
  std::vector<ge::NodePtr> buff_fus_compile_failed_nodes = {node};
  std::vector<ge::NodePtr> buff_fus_rollback_nodes = {node};
  std::vector<ge::NodePtr> buff_fus_to_del_nodes = {node};
  Configuration::Instance(AI_CORE_NAME).config_param_vec_[static_cast<size_t>(CONFIG_PARAM::BufferOptimize)] =
          static_cast<int64_t>(EN_L1_OPTIMIZE);
  Status ret = lx_fusion_optimizer->LxFusionRecovery(*graph, buff_fus_compile_failed_nodes,
                                                     buff_fus_rollback_nodes, buff_fus_to_del_nodes);

  LxFusionOptimizeResult res = LxFusionOptimizeResult::NO_FUSION_STRATEGY;
  lx_fusion_optimizer->DoLxFusionOptimize(*graph, res);
  Configuration::Instance(AI_CORE_NAME).lib_path_ = tmp_path;
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(LxFusionOptimizerUT, l2_fusion_Func_not_init_recovery) {
  LxFusionOptimizerPtr lx_fusion_optimizer = std::make_shared<LxFusionOptimizer>(fusion_priority_mgr, ops_kernel_info_store);
  auto tmp_path = Configuration::Instance(AI_CORE_NAME).lib_path_;
  Configuration::Instance(AI_CORE_NAME).lib_path_ = "test_empty_path";
  EXPECT_EQ(lx_fusion_optimizer->Initialize(), fe::SUCCESS);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr op_desc = std::make_shared<OpDesc>("relu", "Relu");
  GeTensorDesc output_desc;
  op_desc->AddOutputDesc(output_desc);
  ge::AttrUtils::SetStr(op_desc, kFusionOpBuildOptions, kFixPipeOpt);
  NodePtr node = graph->AddNode(op_desc);
  std::vector<ge::NodePtr> buff_fus_compile_failed_nodes = {node};
  std::vector<ge::NodePtr> buff_fus_rollback_nodes = {node};
  std::vector<ge::NodePtr> buff_fus_to_del_nodes = {node};
  Configuration::Instance(AI_CORE_NAME).config_param_vec_[static_cast<size_t>(CONFIG_PARAM::BufferOptimize)] =
          static_cast<int64_t>(EN_L2_OPTIMIZE);
  Status ret = lx_fusion_optimizer->LxFusionRecovery(*graph, buff_fus_compile_failed_nodes,
                                                     buff_fus_rollback_nodes, buff_fus_to_del_nodes);
  Configuration::Instance(AI_CORE_NAME).lib_path_ = tmp_path;
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(LxFusionOptimizerUT, l1_fusion_failed) {
  LxFusionOptimizerPtr lx_fusion_optimizer = std::make_shared<LxFusionOptimizer>(fusion_priority_mgr, ops_kernel_info_store);
  EXPECT_EQ(lx_fusion_optimizer->Initialize(), fe::SUCCESS);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr op_desc = std::make_shared<OpDesc>("relu", "Relu");
  GeTensorDesc output_desc;
  op_desc->AddOutputDesc(output_desc);
  NodePtr node = graph->AddNode(op_desc);
  AttrUtils::SetInt(graph, "_l1_fusion_ret", tune::FAILED);
  AttrUtils::SetInt(graph, "_l2_fusion_ret", tune::SUCCESS);

  Configuration::Instance(AI_CORE_NAME).config_param_vec_[static_cast<size_t>(CONFIG_PARAM::BufferOptimize)] =
          static_cast<int64_t>(EN_L1_OPTIMIZE);
  LxFusionOptimizeResult lx_ret = LxFusionOptimizeResult::NO_FUSION_STRATEGY;
  bool need_re_compile = false;
  Status ret = lx_fusion_optimizer->LxFusionOptimize(*graph, lx_ret, need_re_compile);
  EXPECT_EQ(ret, fe::FAILED);
}

TEST_F(LxFusionOptimizerUT, l2_fusion_failed) {
  LxFusionOptimizerPtr lx_fusion_optimizer = std::make_shared<LxFusionOptimizer>(fusion_priority_mgr, ops_kernel_info_store);
  EXPECT_EQ(lx_fusion_optimizer->Initialize(), fe::SUCCESS);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr op_desc = std::make_shared<OpDesc>("relu", "Relu");
  GeTensorDesc output_desc;
  op_desc->AddOutputDesc(output_desc);
  NodePtr node = graph->AddNode(op_desc);
  AttrUtils::SetInt(graph, "_l1_fusion_ret", tune::SUCCESS);
  AttrUtils::SetInt(graph, "_l2_fusion_ret", tune::FAILED);

  Configuration::Instance(AI_CORE_NAME).config_param_vec_[static_cast<size_t>(CONFIG_PARAM::BufferOptimize)] =
          static_cast<int64_t>(EN_L1_OPTIMIZE);
  LxFusionOptimizeResult lx_ret = LxFusionOptimizeResult::NO_FUSION_STRATEGY;
  bool need_re_compile = false;
  Status ret = lx_fusion_optimizer->LxFusionOptimize(*graph, lx_ret, need_re_compile);
  EXPECT_EQ(ret, fe::FAILED);
}

TEST_F(LxFusionOptimizerUT, generate_lxfusion_optimize_result_1) {
  Status l1_buffer_ret = tune::SUCCESS;
  Status l2_buffer_ret = tune::HIT_FUSION_STRATEGY;
  LxFusionOptimizeResult opt_ret =
          LxFusionOptimizer::GenerateLxFusionOptimizeResult(tune::SUCCESS, tune::HIT_FUSION_STRATEGY);
  EXPECT_EQ(opt_ret, LxFusionOptimizeResult::BOTH_FUSION_STRATEGY);

  opt_ret = LxFusionOptimizer::GenerateLxFusionOptimizeResult(tune::HIT_FUSION_STRATEGY, tune::SUCCESS);
  EXPECT_EQ(opt_ret, LxFusionOptimizeResult::BOTH_FUSION_STRATEGY);

  opt_ret = LxFusionOptimizer::GenerateLxFusionOptimizeResult(tune::SUCCESS, tune::NO_FUSION_STRATEGY);
  EXPECT_EQ(opt_ret, LxFusionOptimizeResult::L1_FUSION_STRATEGY);

  opt_ret = LxFusionOptimizer::GenerateLxFusionOptimizeResult(tune::NO_FUSION_STRATEGY, tune::SUCCESS);
  EXPECT_EQ(opt_ret, LxFusionOptimizeResult::L2_FUSION_STRATEGY);

  opt_ret = LxFusionOptimizer::GenerateLxFusionOptimizeResult(tune::NO_FUSION_STRATEGY, tune::NO_FUSION_STRATEGY);
  EXPECT_EQ(opt_ret, LxFusionOptimizeResult::NO_FUSION_STRATEGY);
}

TEST_F(LxFusionOptimizerUT, lx_fusion_optimize_case_1) {
  LxFusionOptimizerPtr lx_fusion_optimizer =
      std::make_shared<LxFusionOptimizer>(fusion_priority_mgr, ops_kernel_info_store);
  EXPECT_EQ(lx_fusion_optimizer->Initialize(), fe::SUCCESS);
  Configuration::Instance(AI_CORE_NAME).config_param_vec_[static_cast<size_t>(CONFIG_PARAM::BufferOptimize)] =
      static_cast<int64_t>(EN_L1_OPTIMIZE);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr op_desc = std::make_shared<OpDesc>("relu", "Relu");
  GeTensorDesc output_desc;
  op_desc->AddOutputDesc(output_desc);
  NodePtr node = graph->AddNode(op_desc);
  AttrUtils::SetInt(graph, "_l1_fusion_ret", tune::HIT_FUSION_STRATEGY);
  AttrUtils::SetInt(graph, "_l2_fusion_ret", tune::HIT_FUSION_STRATEGY);

  graph->SetGraphUnknownFlag(true);
  LxFusionOptimizeResult buffer_ret = LxFusionOptimizeResult::NO_FUSION_STRATEGY;
  bool need_re_compile = false;
  Status status = lx_fusion_optimizer->LxFusionOptimize(*graph, buffer_ret, need_re_compile);
  EXPECT_EQ(status, fe::SUCCESS);
  auto op_compiler = std::make_shared<OpCompiler>("compiler_name", AI_CORE_NAME, lx_fusion_optimizer);
  status = op_compiler->GenerateFormatTuneResult(*graph, buffer_ret, need_re_compile);
  EXPECT_EQ(status, fe::SUCCESS);
  EXPECT_TRUE(buffer_ret == LxFusionOptimizeResult::NO_FUSION_STRATEGY);

  graph->DelAttr(ge::ATTR_NAME_GRAPH_UNKNOWN_FLAG);
  status = lx_fusion_optimizer->LxFusionOptimize(*graph, buffer_ret, need_re_compile);
  EXPECT_EQ(status, fe::SUCCESS);
  status = op_compiler->GenerateFormatTuneResult(*graph, buffer_ret, need_re_compile);
  EXPECT_EQ(status, fe::SUCCESS);
  EXPECT_EQ(buffer_ret, LxFusionOptimizeResult::BOTH_FUSION_STRATEGY);
}

TEST_F(LxFusionOptimizerUT, lx_fusion_optimize_case_2) {
  LxFusionOptimizerPtr lx_fusion_optimizer =
      std::make_shared<LxFusionOptimizer>(fusion_priority_mgr, ops_kernel_info_store);
  EXPECT_EQ(lx_fusion_optimizer->Initialize(), fe::SUCCESS);

  Configuration::Instance(AI_CORE_NAME).config_param_vec_[static_cast<size_t>(CONFIG_PARAM::BufferOptimize)] =
      static_cast<int64_t>(EN_L1_OPTIMIZE);

  LxFusionOptimizeResult buffer_ret = LxFusionOptimizeResult::NO_FUSION_STRATEGY;
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr op_desc = std::make_shared<OpDesc>("relu", "Relu");
  GeTensorDesc output_desc;
  op_desc->AddOutputDesc(output_desc);
  NodePtr node = graph->AddNode(op_desc);

  bool need_re_compile = false;
  Status status = lx_fusion_optimizer->LxFusionOptimize(*graph, buffer_ret, need_re_compile);
  EXPECT_EQ(status, fe::SUCCESS);
  auto op_compiler = std::make_shared<OpCompiler>("compiler_name", AI_CORE_NAME, lx_fusion_optimizer);
  status = op_compiler->GenerateFormatTuneResult(*graph, buffer_ret, need_re_compile);
  EXPECT_EQ(status, fe::SUCCESS);
  EXPECT_TRUE(buffer_ret == LxFusionOptimizeResult::NO_FUSION_STRATEGY);

  AttrUtils::SetInt(graph, "_l1_fusion_ret", tune::HIT_FUSION_STRATEGY);
  status = lx_fusion_optimizer->LxFusionOptimize(*graph, buffer_ret, need_re_compile);
  EXPECT_EQ(status, fe::SUCCESS);
  status = op_compiler->GenerateFormatTuneResult(*graph, buffer_ret, need_re_compile);
  EXPECT_EQ(status, fe::SUCCESS);
  EXPECT_EQ(buffer_ret, LxFusionOptimizeResult::L1_FUSION_STRATEGY);
}

TEST_F(LxFusionOptimizerUT, lx_fusion_optimize_case_3) {
  LxFusionOptimizerPtr lx_fusion_optimizer =
      std::make_shared<LxFusionOptimizer>(fusion_priority_mgr, ops_kernel_info_store);
  EXPECT_EQ(lx_fusion_optimizer->Initialize(), fe::SUCCESS);

  Configuration::Instance(AI_CORE_NAME).config_param_vec_[static_cast<size_t>(CONFIG_PARAM::BufferOptimize)] =
      static_cast<int64_t>(EN_L2_OPTIMIZE);

  LxFusionOptimizeResult buffer_ret = LxFusionOptimizeResult::NO_FUSION_STRATEGY;
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr op_desc = std::make_shared<OpDesc>("relu", "Relu");
  GeTensorDesc output_desc;
  op_desc->AddOutputDesc(output_desc);
  NodePtr node = graph->AddNode(op_desc);

  bool need_re_compile = false;
  Status status = lx_fusion_optimizer->LxFusionOptimize(*graph, buffer_ret, need_re_compile);
  EXPECT_EQ(status, fe::SUCCESS);
  auto op_compiler = std::make_shared<OpCompiler>("compiler_name", AI_CORE_NAME, lx_fusion_optimizer);
  status = op_compiler->GenerateFormatTuneResult(*graph, buffer_ret, need_re_compile);
  EXPECT_EQ(status, fe::SUCCESS);
  EXPECT_TRUE(buffer_ret == LxFusionOptimizeResult::NO_FUSION_STRATEGY);

  AttrUtils::SetInt(graph, "_l2_fusion_ret", tune::HIT_FUSION_STRATEGY);
  status = lx_fusion_optimizer->LxFusionOptimize(*graph, buffer_ret, need_re_compile);
  EXPECT_EQ(status, fe::SUCCESS);
  status = op_compiler->GenerateFormatTuneResult(*graph, buffer_ret, need_re_compile);
  EXPECT_EQ(status, fe::SUCCESS);
  EXPECT_EQ(buffer_ret, LxFusionOptimizeResult::L2_FUSION_STRATEGY);
}

TEST_F(LxFusionOptimizerUT, l1_fusion_pass_changed_case_4) {
  LxFusionOptimizerPtr lx_fusion_optimizer =
      std::make_shared<LxFusionOptimizer>(fusion_priority_mgr, ops_kernel_info_store);
  EXPECT_EQ(lx_fusion_optimizer->Initialize(), fe::SUCCESS);
  lx_fusion_optimizer->l2_fusion_pre_optimizer_func_ = L2FusionPreOptimizer1;

  Configuration::Instance(AI_CORE_NAME).config_param_vec_[static_cast<size_t>(CONFIG_PARAM::BufferOptimize)] =
      static_cast<int64_t>(EN_L2_OPTIMIZE);
  FusionStatisticRecorder::Instance().buffer_fusion_info_map_.clear();

  LxFusionOptimizeResult buffer_ret = LxFusionOptimizeResult::NO_FUSION_STRATEGY;
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr op_desc = std::make_shared<OpDesc>("relu", "Relu");
  GeTensorDesc output_desc;
  op_desc->AddOutputDesc(output_desc);
  NodePtr node = graph->AddNode(op_desc);

  AttrUtils::SetInt(graph, "_l2_fusion_ret", tune::HIT_FUSION_STRATEGY);
  FusionInfo info1{graph->GetSessionID(), std::to_string(graph->GetGraphID()), "TbeConvStridedwriteFusionPass", 8};
  FusionStatisticRecorder::Instance().UpdateBufferFusionMatchTimes(info1);
  FusionInfo info2{graph->GetSessionID(), std::to_string(graph->GetGraphID()), "MatmulConfusiontransposeUbFusion", 2};
  FusionStatisticRecorder::Instance().UpdateBufferFusionMatchTimes(info2);
  bool need_re_compile = false;
  Status status = lx_fusion_optimizer->LxFusionOptimize(*graph, buffer_ret, need_re_compile);
  EXPECT_EQ(status, fe::SUCCESS);
  auto op_compiler = std::make_shared<OpCompiler>("compiler_name", AI_CORE_NAME, lx_fusion_optimizer);
  status = op_compiler->GenerateFormatTuneResult(*graph, buffer_ret, need_re_compile);
  EXPECT_EQ(status, fe::SUCCESS);
  EXPECT_TRUE(buffer_ret == LxFusionOptimizeResult::NO_FUSION_STRATEGY);
  std::string session_and_graph_id = std::to_string(graph->GetSessionID()) + "_" + std::to_string(graph->GetGraphID());
  auto iter = FusionStatisticRecorder::Instance().buffer_fusion_info_map_.find(session_and_graph_id);
  if (iter != FusionStatisticRecorder::Instance().buffer_fusion_info_map_.end()) {
    for (auto iter1 = iter->second.begin(); iter1 != iter->second.end(); iter1++) {
      if (iter1->first == "TbeConvStridedwriteFusionPass") {
        EXPECT_EQ(iter1->second.GetMatchTimes(), 8);
        EXPECT_EQ(iter1->second.GetRepoHitTimes(), 0);
      }
      if (iter1->first == "TbeBnupdateEltwiseFusionPass") {
        EXPECT_EQ(iter1->second.GetMatchTimes(), 0);
        EXPECT_EQ(iter1->second.GetRepoHitTimes(), 0);
      }
      if (iter1->first == "MatmulConfusiontransposeUbFusion") {
        EXPECT_EQ(iter1->second.GetMatchTimes(), 2);
        EXPECT_EQ(iter1->second.GetRepoHitTimes(), 0);
      }
    }
  }
}

TEST_F(LxFusionOptimizerUT, l1_fusion_pass_changed_case_5) {
  LxFusionOptimizerPtr lx_fusion_optimizer = std::make_shared<LxFusionOptimizer>(fusion_priority_mgr, ops_kernel_info_store);
  EXPECT_EQ(lx_fusion_optimizer->Initialize(), fe::SUCCESS);
  lx_fusion_optimizer->l2_fusion_pre_optimizer_func_ = L2FusionPreOptimizer2;

  Configuration::Instance(AI_CORE_NAME).config_param_vec_[static_cast<size_t>(CONFIG_PARAM::BufferOptimize)] =
          static_cast<int64_t>(EN_L2_OPTIMIZE);
  FusionStatisticRecorder::Instance().buffer_fusion_info_map_.clear();

  LxFusionOptimizeResult buffer_ret = LxFusionOptimizeResult::NO_FUSION_STRATEGY;
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr op_desc = std::make_shared<OpDesc>("relu", "Relu");
  GeTensorDesc output_desc;
  op_desc->AddOutputDesc(output_desc);
  NodePtr node = graph->AddNode(op_desc);

  AttrUtils::SetInt(graph, "_l2_fusion_ret", tune::HIT_FUSION_STRATEGY);
  FusionInfo info1{graph->GetSessionID(), std::to_string(graph->GetGraphID()), "TbeConvStridedwriteFusionPass", 8};
  FusionStatisticRecorder::Instance().UpdateBufferFusionMatchTimes(info1);
  FusionInfo info2{graph->GetSessionID(), std::to_string(graph->GetGraphID()), "MatmulConfusiontransposeUbFusion", 2};
  FusionStatisticRecorder::Instance().UpdateBufferFusionMatchTimes(info2);
  bool need_re_compile = false;
  Status status = lx_fusion_optimizer->LxFusionOptimize(*graph, buffer_ret, need_re_compile);
  EXPECT_EQ(status, fe::SUCCESS);
  auto op_compiler = std::make_shared<OpCompiler>("compiler_name", AI_CORE_NAME, lx_fusion_optimizer);
  status = op_compiler->GenerateFormatTuneResult(*graph, buffer_ret, need_re_compile);
  EXPECT_EQ(status, fe::SUCCESS);
  EXPECT_EQ(buffer_ret, LxFusionOptimizeResult::L2_FUSION_STRATEGY);
  std::string session_and_graph_id = std::to_string(graph->GetSessionID()) + "_" + std::to_string(graph->GetGraphID());
  auto iter = FusionStatisticRecorder::Instance().buffer_fusion_info_map_.find(session_and_graph_id);
  if (iter != FusionStatisticRecorder::Instance().buffer_fusion_info_map_.end()) {
    for (auto iter1 = iter->second.begin(); iter1 != iter->second.end(); iter1++) {
      if (iter1->first == "TbeConvStridedwriteFusionPass") {
        EXPECT_EQ(iter1->second.GetMatchTimes(), 13);
        EXPECT_EQ(iter1->second.GetRepoHitTimes(), 10);
      }
      if (iter1->first == "TbeBnupdateEltwiseFusionPass") {
        EXPECT_EQ(iter1->second.GetMatchTimes(), 3);
        EXPECT_EQ(iter1->second.GetRepoHitTimes(), 8);
      }
      if (iter1->first == "MatmulConfusiontransposeUbFusion") {
        EXPECT_EQ(iter1->second.GetMatchTimes(), 2);
        EXPECT_EQ(iter1->second.GetRepoHitTimes(), 3);
      }
    }
  }
}

TEST_F(LxFusionOptimizerUT, l1_fusion_pass_changed_case_6) {
  LxFusionOptimizerPtr lx_fusion_optimizer = std::make_shared<LxFusionOptimizer>(fusion_priority_mgr, ops_kernel_info_store);
  EXPECT_EQ(lx_fusion_optimizer->Initialize(), fe::SUCCESS);
  lx_fusion_optimizer->l2_fusion_pre_optimizer_func_ = L2FusionPreOptimizer3;

  Configuration::Instance(AI_CORE_NAME).config_param_vec_[static_cast<size_t>(CONFIG_PARAM::BufferOptimize)] =
          static_cast<int64_t>(EN_L2_OPTIMIZE);
  FusionStatisticRecorder::Instance().buffer_fusion_info_map_.clear();

  LxFusionOptimizeResult buffer_ret = LxFusionOptimizeResult::NO_FUSION_STRATEGY;
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr op_desc = std::make_shared<OpDesc>("relu", "Relu");
  GeTensorDesc output_desc;
  op_desc->AddOutputDesc(output_desc);
  NodePtr node = graph->AddNode(op_desc);

  FusionInfo info1{graph->GetSessionID(), std::to_string(graph->GetGraphID()), "TbeConvStridedwriteFusionPass", 8};
  FusionStatisticRecorder::Instance().UpdateBufferFusionMatchTimes(info1);
  FusionInfo info2{graph->GetSessionID(), std::to_string(graph->GetGraphID()), "MatmulConfusiontransposeUbFusion", 2};
  FusionStatisticRecorder::Instance().UpdateBufferFusionMatchTimes(info2);
  AttrUtils::SetInt(graph, "_l2_fusion_ret", tune::HIT_FUSION_STRATEGY);
  bool need_re_compile = false;
  Status status = lx_fusion_optimizer->LxFusionOptimize(*graph, buffer_ret, need_re_compile);
  EXPECT_EQ(status, fe::SUCCESS);
  auto op_compiler = std::make_shared<OpCompiler>("compiler_name", AI_CORE_NAME, lx_fusion_optimizer);
  status = op_compiler->GenerateFormatTuneResult(*graph, buffer_ret, need_re_compile);
  EXPECT_EQ(status, fe::SUCCESS);
  EXPECT_EQ(buffer_ret, LxFusionOptimizeResult::L2_FUSION_STRATEGY);
  std::string session_and_graph_id = std::to_string(graph->GetSessionID()) + "_" + std::to_string(graph->GetGraphID());
  auto iter = FusionStatisticRecorder::Instance().buffer_fusion_info_map_.find(session_and_graph_id);
  if (iter != FusionStatisticRecorder::Instance().buffer_fusion_info_map_.end()) {
    for (auto iter1 = iter->second.begin(); iter1 != iter->second.end(); iter1++) {
      if (iter1->first == "TbeConvStridedwriteFusionPass") {
        EXPECT_EQ(iter1->second.GetMatchTimes(), 13);
        EXPECT_EQ(iter1->second.GetRepoHitTimes(), 10);
      }
      if (iter1->first == "TbeBnupdateEltwiseFusionPass") {
        EXPECT_EQ(iter1->second.GetMatchTimes(), 3);
        EXPECT_EQ(iter1->second.GetRepoHitTimes(), 8);
      }
      if (iter1->first == "MatmulConfusiontransposeUbFusion") {
        EXPECT_EQ(iter1->second.GetMatchTimes(), 2);
        EXPECT_EQ(iter1->second.GetRepoHitTimes(), 3);
      }
    }
  }
}

class ConcatCOptimizeFusionPassTest : public PatternFusionBasePass {
 protected:
  vector<FusionPattern *> DefinePatterns() {
    vector<FusionPattern *> patterns;
    FusionPattern *pattern = new (std::nothrow) FusionPattern("ConcatCOptimizeFusionPassTest");
    FE_CHECK(pattern == nullptr, REPORT_FE_ERROR("[GraphOpt][ConCatCOptFus][DfnPtn] Fail to new an object."),
             return patterns);
    pattern->AddOpDesc("concat", {"ConcatD"})
        .SetOutput("concat");
    patterns.push_back(pattern);
    return patterns;
  }
  Status Fusion(ge::ComputeGraph &graph, Mapping &mapping, vector<ge::NodePtr> &fusion_nodes) {
    return FAILED;
  }
};

TEST_F(LxFusionOptimizerUT, lx_fusion_optimize_case_7) {
  mmSetEnv("ASCEND_HOME_PATH", "/home/jenkins/Ascend/ascend-toolkit/latest", 0);
  FEOpsKernelInfoStorePtr ops_kernel_info_store_ptr_;
  std::make_shared<FEOpsKernelInfoStore>();
  auto fe_graph_optimizer_ptr = std::make_shared<FEGraphOptimizer>(ops_kernel_info_store_ptr_);
  fe_graph_optimizer_ptr->format_dtype_setter_ptr_ =
  std::make_shared<FormatDtypeSetter>(AI_CORE_NAME);
  fe_graph_optimizer_ptr->op_impl_type_judge_ptr_ =
  std::make_shared<OpImplTypeJudge>(AI_CORE_NAME, ops_kernel_info_store_ptr_);
  fe_graph_optimizer_ptr->op_axis_update_desc_ptr_ =
  std::make_shared<OpAxisUpdateDesc>(AI_CORE_NAME);
  FusionRuleManagerPtr fusion_rule_mgr_ptr_ = std::make_shared<FusionRuleManager>(ops_kernel_info_store_ptr_);
  fe_graph_optimizer_ptr->fusion_priority_mgr_ptr_ = std::make_shared<FusionPriorityManager>(
      fe::AI_CORE_NAME, fusion_rule_mgr_ptr_);
  fe_graph_optimizer_ptr->fusion_priority_mgr_ptr_->Initialize();
  std::vector<FusionPassOrRule> pass_vec;
  auto create_func = []() -> ::fe::GraphPass * { return new (std::nothrow) ConcatCOptimizeFusionPassTest(); };
  FusionPassRegistry::PassDesc pass_desc = {0, create_func};
  fe::FusionPassOrRule pass_or_rule("ConcatCOptimizeFusionPassTest", BUILT_IN_AFTER_BUFFER_OPTIMIZE,
                                    PASS_METHOD, 4000, pass_desc);
  pass_vec.emplace_back(pass_or_rule);
  fe_graph_optimizer_ptr->fusion_priority_mgr_ptr_->sorted_graph_fusion_map_[FusionPriorityManager::GetCurrentHashedKey()] = pass_vec;
  Configuration::Instance(fe::AI_CORE_NAME).content_map_["fusion.config.built-in.file"] = "fusion_config.json";
  Configuration::Instance(fe::AI_CORE_NAME).ascend_ops_path_ =
          GetCodeDir() + "/tests/engines/nn_engine/st/testcase/fusion_config_manager/builtin_config/";
  ge::GetThreadLocalContext().graph_options_[ge::FUSION_SWITCH_FILE] =
          GetCodeDir() + "/tests/engines/nn_engine/st/testcase/fusion_config_manager/builtin_config/fusion_config.json";
  std::string allStr = "ALL";
  Configuration::Instance(fe::AI_CORE_NAME).config_str_param_vec_[static_cast<size_t>(CONFIG_STR_PARAM::FusionLicense)] = allStr;
  fe_graph_optimizer_ptr->fusion_priority_mgr_ptr_->Initialize();

  fe_graph_optimizer_ptr->ops_kernel_info_store_ptr_ =
  std::make_shared<FEOpsKernelInfoStore>(fe::AI_CORE_NAME);

  fe_graph_optimizer_ptr->graph_fusion_ptr_ = std::make_shared<GraphFusion>(fusion_rule_mgr_ptr_,
      ops_kernel_info_store_ptr_, fe_graph_optimizer_ptr->fusion_priority_mgr_ptr_);
  fe_graph_optimizer_ptr->space_size_calculator_ptr_ = std::make_shared<SpaceSizeCalculator>();
  fe_graph_optimizer_ptr->op_setter_ptr_ = std::make_shared<OpSetter>(AI_CORE_NAME);

  std::map<std::string, std::string> context_maps;
  std::string fusion_switch_file_path = GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/graph_optimizer/fusion_switch_file.json";
  if (RealPath(fusion_switch_file_path).empty()) {
    fusion_switch_file_path = "../../../../../tests/engines/nn_engine/ut/testcase/fusion_engine/graph_optimizer/fusion_switch_file.json";
  }
  context_maps.insert(std::make_pair("ge.fusionSwitchFile", fusion_switch_file_path));
  context_maps.insert(std::make_pair("ge.build_inner_model", "false"));
  ge::GetThreadLocalContext().SetGraphOption(context_maps);
  fe_graph_optimizer_ptr->fusion_priority_mgr_ptr_->sorted_graph_fusion_map_[FusionPriorityManager::GetCurrentHashedKey()] = pass_vec;

  fe_graph_optimizer_ptr->fusion_priority_mgr_ptr_->Initialize();

  LxFusionOptimizerPtr lx_fusion_optimizer = std::make_shared<LxFusionOptimizer>(fe_graph_optimizer_ptr->fusion_priority_mgr_ptr_, ops_kernel_info_store_ptr_);
  EXPECT_EQ(lx_fusion_optimizer->Initialize(), fe::SUCCESS);
  Configuration::Instance(AI_CORE_NAME).config_param_vec_[static_cast<size_t>(CONFIG_PARAM::BufferOptimize)] =
          static_cast<int64_t>(EN_L1_OPTIMIZE);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr op_desc = std::make_shared<OpDesc>("concat", "ConcatD");
  GeTensorDesc output_desc;
  op_desc->AddOutputDesc(output_desc);
  NodePtr node = graph->AddNode(op_desc);

  LxFusionOptimizeResult buffer_ret = LxFusionOptimizeResult::NO_FUSION_STRATEGY;
  bool need_re_compile = false;
  Status status = lx_fusion_optimizer->LxFusionOptimize(*graph, buffer_ret, need_re_compile);
  EXPECT_EQ(status, fe::FAILED);
}
}
