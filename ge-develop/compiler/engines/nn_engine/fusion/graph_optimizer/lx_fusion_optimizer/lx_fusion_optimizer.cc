/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/lx_fusion_optimizer/lx_fusion_optimizer.h"
#include "common/fe_log.h"
#include "common/string_utils.h"
#include "common/aicore_util_constants.h"
#include "common/aicore_util_attr_define.h"
#include "common/fe_context_utils.h"
#include "common/configuration.h"
#include "common/fe_graph_common.h"
#include "common/calc_slice_utils.h"
#include "graph/tuning_utils.h"
#include "graph/ge_context.h"
#include "register/graph_optimizer/fusion_common/fusion_statistic_recorder.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/builtin_pass/concat_c_optimize_fusion_pass.h"
#include "graph_optimizer/graph_fusion/graph_fusion.h"

namespace fe {
namespace {
constexpr char const *LX_FUSION_PLUGIN = "libgraph_tuner.so";
constexpr char const *L1_FUSION_OPTIMIZER_FUNC_NAME = "L1FusionOptimize";
constexpr char const *L1_RECOVERY_FUNC_NAME = "L1Recovery";
constexpr char const *L2_FUSION_PRE_OPTIMIZER_FUNC_NAME = "L2FusionPreOptimize";
constexpr char const *L2_FUSION_OPTIMIZER_FUNC_NAME = "L2FusionOptimize";
constexpr char const *L2_RECOVERY_FUNC_NAME = "L2Recovery";
constexpr char const *LX_FUSION_FINALIZE_FUNC_NAME = "LxFusionFinalize";
}
LxFusionOptimizer::LxFusionOptimizer(const FusionPriorityMgrPtr &fusion_priority_mgr,
                                     const OpsKernelInfoStorePtr &ops_kernel_info_store)
                                     : init_flag_(false), fusion_priority_mgr_ptr_(fusion_priority_mgr),
                                       ops_kernel_info_store_ptr_(ops_kernel_info_store) {}
LxFusionOptimizer::~LxFusionOptimizer() {}

Status LxFusionOptimizer::Initialize() {
  if (init_flag_) {
    return SUCCESS;
  }
  // 1. open the lx fusion plugin
  std::string plugin_path = Configuration::Instance(AI_CORE_NAME).GetFeLibPath() + LX_FUSION_PLUGIN;
  FE_LOGD("Begin to load lxfusion so [%s].", plugin_path.c_str());
  FE_MAKE_SHARED(lx_fusion_plugin_manager_ = std::make_shared<PluginManager>(plugin_path), return FAILED);
  FE_CHECK(lx_fusion_plugin_manager_ == nullptr,
           REPORT_FE_ERROR("[GraphOpt][Init][InitLxFusPlg] Create lx fusion plugin manager ptr failed."),
           return FAILED);
  if (lx_fusion_plugin_manager_->OpenPlugin(plugin_path) != SUCCESS) {
    FE_LOGW("Failed to open %s.", plugin_path.c_str());
    lx_fusion_plugin_manager_.reset();
    return SUCCESS;
  }

  if (InitFunctions() != SUCCESS) {
    FE_LOGW("Failed to get functions from plugin[%s].", plugin_path.c_str());
    ClosePlugin();
    return FAILED;
  }
  init_flag_ = true;
  return SUCCESS;
}

Status LxFusionOptimizer::InitFunctions() {
  FE_CHECK_NOTNULL(lx_fusion_plugin_manager_);
  // 2. get the functions of l1 fusion
  Status ret = lx_fusion_plugin_manager_->GetFunction<tune::Status, ge::ComputeGraph &, AOEOption>(
          L1_FUSION_OPTIMIZER_FUNC_NAME, l1_fusion_optimizer_func_);
  if (ret != SUCCESS) {
    FE_LOGW("Failed to retrieve the function %s from the plugin.", L1_FUSION_OPTIMIZER_FUNC_NAME);
    return FAILED;
  }

  ret = lx_fusion_plugin_manager_->GetFunction<tune::Status, ge::ComputeGraph &, const std::vector<ge::NodePtr> &,
          std::vector<ge::NodePtr> *, std::vector<ge::NodePtr> *>(
          L1_RECOVERY_FUNC_NAME, l1_recovery_func_);
  if (ret != SUCCESS) {
    FE_LOGW("Failed to retrieve the function %s from the plugin.", L1_RECOVERY_FUNC_NAME);
    return FAILED;
  }

  // 3. get the functions of l2 pre optimize function
  ret = lx_fusion_plugin_manager_
          ->GetFunction<tune::Status, ge::ComputeGraph &, AOEOption, const std::vector<std::string> &,
                  std::vector<PassChangeInfo> &>(L2_FUSION_PRE_OPTIMIZER_FUNC_NAME, l2_fusion_pre_optimizer_func_);
  if (ret != SUCCESS) {
    FE_LOGW("Failed to retrieve the function %s from the plugin.", L2_FUSION_PRE_OPTIMIZER_FUNC_NAME);
    return FAILED;
  }

  // 4. get the functions of l2 optimize function
  ret = lx_fusion_plugin_manager_->GetFunction<tune::Status, ge::ComputeGraph &, AOEOption>(
          L2_FUSION_OPTIMIZER_FUNC_NAME, l2_fusion_optimizer_func_);
  if (ret != SUCCESS) {
    FE_LOGW("Failed to retrieve the function %s from the plugin.", L2_FUSION_OPTIMIZER_FUNC_NAME);
    return FAILED;
  }

  ret = lx_fusion_plugin_manager_->GetFunction<tune::Status, ge::ComputeGraph &, const std::vector<ge::NodePtr> &,
          std::vector<ge::NodePtr> *, std::vector<ge::NodePtr> *>(
          L2_RECOVERY_FUNC_NAME, l2_recovery_func_);
  if (ret != SUCCESS) {
    FE_LOGW("Failed to retrieve the function %s from the plugin.", L2_RECOVERY_FUNC_NAME);
    return FAILED;
  }

  ret = lx_fusion_plugin_manager_->GetFunction<tune::Status, ge::ComputeGraph &>(
          LX_FUSION_FINALIZE_FUNC_NAME, lx_fusion_finalize_func_);
  if (ret != SUCCESS) {
    FE_LOGW("Failed to retrieve the function %s from the plugin.", LX_FUSION_FINALIZE_FUNC_NAME);
    return FAILED;
  }
  return SUCCESS;
}

void LxFusionOptimizer::ClosePlugin() {
  if (lx_fusion_plugin_manager_ == nullptr) {
    return;
  }
  if (lx_fusion_plugin_manager_->CloseHandle() != SUCCESS) {
    FE_LOGW("Failed to close the handle for the plugin [%s].", lx_fusion_plugin_manager_->GetSoName().c_str());
  }
}

Status LxFusionOptimizer::Finalize() {
  if (!init_flag_) {
    return SUCCESS;
  }
  ClosePlugin();
  init_flag_ = false;
  return SUCCESS;
}

Status LxFusionOptimizer::LxFusionOptimize(ge::ComputeGraph& graph, LxFusionOptimizeResult& buffer_ret,
                                           bool &need_re_compile) {
  if (DoLxFusionOptimize(graph, buffer_ret) != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][BufFusProc][LxFusOpt] Failed to perform lx fusion for graph %s.", graph.GetName().c_str());
    return FAILED;
  }
  
  if (OptimizeConcatCAxis(graph, need_re_compile) != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][BufFusProc][LxFusOpt] Failed to do concat c optimize for graph %s.",
                    graph.GetName().c_str());
    return FAILED;
  }
  return SUCCESS;
}

LxFusionOptimizeResult LxFusionOptimizer::GenerateLxFusionOptimizeResult(const Status &l1_buffer_ret,
                                                                         const Status &l2_buffer_ret) {
  bool is_l1_optimize = (l1_buffer_ret == tune::SUCCESS || l1_buffer_ret == tune::HIT_FUSION_STRATEGY);
  bool is_l2_optimize = (l2_buffer_ret == tune::SUCCESS || l2_buffer_ret == tune::HIT_FUSION_STRATEGY);
  if (is_l1_optimize && !is_l2_optimize) {
    return LxFusionOptimizeResult::L1_FUSION_STRATEGY;
  }
  if (!is_l1_optimize && is_l2_optimize) {
    return LxFusionOptimizeResult::L2_FUSION_STRATEGY;
  }
  if (is_l1_optimize && is_l2_optimize) {
    return LxFusionOptimizeResult::BOTH_FUSION_STRATEGY;
  }
  return LxFusionOptimizeResult::NO_FUSION_STRATEGY;
}

Status LxFusionOptimizer::DoLxFusionOptimize(ge::ComputeGraph& graph, LxFusionOptimizeResult &buffer_ret) {
  // 1. if find any unknown shape node, return success
  if (FeGraphCommon::IsUnknownGraph(graph.shared_from_this())) {
    FE_LOGI("Graph [%s] contains unknown shape op, can not do lx fusion optimize.", graph.GetName().c_str());
    return SUCCESS;
  }

  // 2. get the l1_fusion_flag and l2_fusion_flag
  bool l1_fusion_flag = Configuration::Instance(AI_CORE_NAME).EnableL1Fusion();
  bool l2_fusion_flag = Configuration::Instance(AI_CORE_NAME).EnableL2Fusion();
  bool none_l1_and_l2_fusion_flag = !l1_fusion_flag && !l2_fusion_flag;
  if (none_l1_and_l2_fusion_flag) {
    FE_LOGD("Both L1 fusion and L2 fusion are not enabled.");
    return SUCCESS;
  }

  OptimizeConfig optimize_config;
  GetLxfusionOptimizeConfig(optimize_config);

  Status l2_buffer_ret = tune::NO_FUSION_STRATEGY;
  if (l2_fusion_flag) {
    l2_buffer_ret = L2FusionOptimize(graph, optimize_config);
    if (l2_buffer_ret == FAILED) {
      return FAILED;
    }
  }
  // kb_hit log can not be deleted, other components are already in use.
  FE_LOGI("[sg_kb_hit][%s][%u]", graph.GetName().c_str(), l2_buffer_ret);

  Status l1_buffer_ret = tune::NO_FUSION_STRATEGY;
  if (l1_fusion_flag) {
    if (l1_fusion_optimizer_func_ == nullptr) {
      FE_LOGW("Parameter[l1_fusion_optimizer_func_] must not be null.");
      return SUCCESS;
    }
    l1_buffer_ret = l1_fusion_optimizer_func_(graph, optimize_config.aoe_option);
    bool condition = (l1_buffer_ret == tune::SUCCESS || l1_buffer_ret == tune::HIT_FUSION_STRATEGY);
    if (condition) {
      FE_LOGI("L1 FUSION SUCCESS or HIT_FUSION_STRATEGY");
    } else if (l1_buffer_ret == tune::NO_FUSION_STRATEGY) {
      FE_LOGI("Lx fusion strategy is invalid.");
    } else {
      REPORT_INNER_ERR_MSG("EF9999", "[SubGraphOpt][BufFusProc][fuc] L1FusionOptimizer failed.");
      return FAILED;
    }
  }
  buffer_ret = GenerateLxFusionOptimizeResult(l1_buffer_ret, l2_buffer_ret);
  return SUCCESS;
}

Status LxFusionOptimizer::L2FusionOptimize(ge::ComputeGraph& graph, const OptimizeConfig &optimize_config) {
  if (l2_fusion_pre_optimizer_func_ == nullptr) {
    FE_LOGW("Parameter[l2_fusion_pre_optimizer_func_] must not be null.");
    return tune::NO_FUSION_STRATEGY;
  }
  std::vector<std::string> close_pass_vec;
  fusion_priority_mgr_ptr_->GetFusionPassNameBySwitch(UB_FUSION, false, close_pass_vec);
  FE_LOGD("The closed ub fusion passes are: %s.", StringUtils::StrVecToString(close_pass_vec).c_str());
  std::vector<PassChangeInfo> pass_info_vec;
  Status l2_pre_buffer_ret = l2_fusion_pre_optimizer_func_(graph, optimize_config.aoe_option,
                                                           close_pass_vec, pass_info_vec);
  if (l2_pre_buffer_ret == tune::SUCCESS || l2_pre_buffer_ret == tune::HIT_FUSION_STRATEGY) {
    UpdateFusionTimesAndSliceInfo(graph, pass_info_vec);
    if (l2_fusion_optimizer_func_ == nullptr) {
      FE_LOGW("Parameter[l2_fusion_optimizer_func_] must not be null.");
      return tune::NO_FUSION_STRATEGY;
    }
    Status l2_buffer_ret = l2_fusion_optimizer_func_(graph, optimize_config.aoe_option);
    if (l2_buffer_ret == tune::SUCCESS || l2_buffer_ret == tune::HIT_FUSION_STRATEGY) {
      FE_LOGI("L2 FUSION SUCCESS or HIT_FUSION_STRATEGY");
    } else if (l2_buffer_ret == tune::NO_FUSION_STRATEGY) {
      FE_LOGI("Lx fusion strategy is invalid, go to l2 buffer.");
    } else {
      REPORT_INNER_ERR_MSG("EF9999", "[SubGraphOpt][BufFusProc][fuc] L2FusionOptimizer failed.");
      return FAILED;
    }
    return l2_buffer_ret;
  } else if (l2_pre_buffer_ret == tune::NO_FUSION_STRATEGY) {
    FE_LOGI("The result of l2 fusion pre optimize is no fusion strategy.");
    return l2_pre_buffer_ret;
  } else {
    REPORT_INNER_ERR_MSG("EF9999", "[SubGraphOpt][BufFusProc][fuc] L2FusionPreOptimizer failed.");
    return FAILED;
  }
}

void LxFusionOptimizer::UpdateFusionTimesAndSliceInfo(const ge::ComputeGraph& graph,
                                                      const std::vector<PassChangeInfo> &pass_info_vec) const {
  if (pass_info_vec.empty()) {
    return;
  }
  uint64_t session_id = graph.GetSessionID();
  std::string graph_id = std::to_string(graph.GetGraphID());
  for (const PassChangeInfo &pass_info : pass_info_vec) {
    // update fusion times
    int32_t alter_times = pass_info.recovery_times - pass_info.rollback_times;
    FusionInfo fusion_info(session_id, graph_id, pass_info.pass_name, alter_times, 0, pass_info.recovery_times);
    FusionStatisticRecorder::Instance().UpdateBufferFusionMatchTimes(fusion_info);

    // update slice info
    const std::vector<int64_t> &scope_ids = pass_info.recovery_scope_ids;
    if (scope_ids.empty()) {
      continue;
    }
    std::map<int64_t, std::vector<ge::NodePtr>> scope_nodes_map;
    int64_t scope_id = -1;
    for (const ge::NodePtr &node : graph.GetDirectNode()) {
      if (ge::AttrUtils::GetInt(node->GetOpDesc(), SCOPE_ID_ATTR, scope_id) &&
          std::find(scope_ids.begin(), scope_ids.end(), scope_id) != scope_ids.end()) {
        scope_nodes_map[scope_id].push_back(node);
        // clear op slice info, otherwise op slice info will not be gernerated in following code
        (void)node->GetOpDesc()->DelAttr(FUSION_OP_SLICE_INFO);
      }
    }
    BufferFusionPassBasePtr buffer_fusion_pass_ptr =
            fusion_priority_mgr_ptr_->GetBufferFusionByPassName(pass_info.pass_name);
    for (std::pair<const int64_t, std::vector<ge::NodePtr>> &nodes_pair : scope_nodes_map) {
      CalcSliceUtils::CalcSliceInfo(buffer_fusion_pass_ptr, nodes_pair.second);
    }
  }
}

void LxFusionOptimizer::GetLxfusionOptimizeConfig(OptimizeConfig &optimize_config) {
  optimize_config.enable_superkernel_plus =
          Configuration::Instance(AI_CORE_NAME).IsEnableSuperkernelPlus();
  FE_LOGD("lxfusion enable super kernel plus is %d.", optimize_config.enable_superkernel_plus);
  optimize_config.aoe_option = AOE_OPT_USE_KB;
  std::string license_aoe_val;
  ge::graphStatus status = ge::GetContext().GetOption("opt_module.aoe", license_aoe_val);
  bool status_flag = (status == ge::GRAPH_SUCCESS && license_aoe_val.empty());
  if (status_flag) {
    optimize_config.aoe_option = AOE_OPT_NOT_USE_KB;
  }

  std::string build_mode_value = FEContextUtils::GetBuildMode();
  if (build_mode_value == ge::BUILD_MODE_BASELINE) {
    optimize_config.aoe_option = AOE_OPT_ONLY_USE_KB;
  }
  FE_LOGD("lxfusion license aoe: %d.", optimize_config.aoe_option);
}

Status LxFusionOptimizer::OptimizeConcatCAxis(ge::ComputeGraph& graph, bool &need_re_compile) const {
  GraphFusionPtr graph_fusion_ptr = nullptr;
  FusionRuleManagerPtr fusion_rule_mgr_ptr_ = nullptr;
  FE_MAKE_SHARED(graph_fusion_ptr = std::make_shared<GraphFusion>(fusion_rule_mgr_ptr_, ops_kernel_info_store_ptr_,
      fusion_priority_mgr_ptr_), return FAILED);
  Status ret = graph_fusion_ptr->RunGraphFusionPassByType("OptimizeConcatCAxis", graph, BUILT_IN_AFTER_BUFFER_OPTIMIZE);
  if (ret != SUCCESS) {
    return ret;
  }
  (void)ge::AttrUtils::GetBool(graph, NEED_RE_PRECOMPILE, need_re_compile);
  return SUCCESS;
}

Status LxFusionOptimizer::LxFusionRecovery(ge::ComputeGraph& graph,
                                           const std::vector<ge::NodePtr>& buff_fus_compile_failed_nodes,
                                           std::vector<ge::NodePtr> &buff_fus_rollback_nodes,
                                           std::vector<ge::NodePtr> &buff_fus_to_del_nodes) {
  for (const auto &node : buff_fus_compile_failed_nodes) {
    if (node->GetOpDesc()->HasAttr(kFusionOpBuildOptions)) {
      (void)node->GetOpDesc()->DelAttr(kFusionOpBuildOptions);
      FE_LOGD("Del the fixpipe_fusion attr from unsuccess UB fusion node %s", node->GetName().c_str());
    }
  }
  tune::Status ret_pass = tune::SUCCESS;
  std::string build_mode_value = FEContextUtils::GetBuildMode();
  if (Configuration::Instance(AI_CORE_NAME).EnableL1Fusion()) {
    // ADD L1 block rollback
    if (l1_recovery_func_ == nullptr) {
      FE_LOGW("Parameter[l1_recovery_func_] must not be null.");
      return SUCCESS;
    }
    ret_pass = l1_recovery_func_(graph, buff_fus_compile_failed_nodes,
                                 &buff_fus_rollback_nodes, &buff_fus_to_del_nodes);
  } else if (Configuration::Instance(AI_CORE_NAME).EnableL2Fusion()) {
    // ADD L2 block rollback
    if (l2_recovery_func_ == nullptr) {
      FE_LOGW("Parameter[l2_recovery_func_] must not be null.");
      return SUCCESS;
    }
    ret_pass = l2_recovery_func_(graph, buff_fus_compile_failed_nodes,
                                 &buff_fus_rollback_nodes, &buff_fus_to_del_nodes);
  } else {
    return SUCCESS;
  }

  bool recovery_status =
          (ret_pass != tune::SUCCESS || buff_fus_rollback_nodes.empty() || buff_fus_to_del_nodes.empty());
  if (recovery_status) {
    REPORT_FE_ERROR("[SubGraphOpt][Compile][Recovery] Failed to recover graph %s. Result is %d.",
                    graph.GetName().c_str(), recovery_status);
    return FAILED;
  }
  return SUCCESS;
}

Status LxFusionOptimizer::LxFusionFinalize(ge::ComputeGraph &graph) {
  if (lx_fusion_finalize_func_ == nullptr) {
    FE_LOGW("Parameter[lx_fusion_finalize_func_] must not be null.");
    return SUCCESS;
  }
  tune::Status ret_pass = lx_fusion_finalize_func_(graph);
  if (ret_pass != tune::SUCCESS) {
    FE_LOGW("Failed to recover graph %s, Result is %u.", graph.GetName().c_str(), ret_pass);
    return FAILED;
  }
  return SUCCESS;
}
}
