/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fftsplus_ops_kernel_builder.h"
#include <list>
#include "inc/ffts_utils.h"
#include "framework/common/ge_types.h"
#include "graph/utils/attr_utils.h"
#include "engine/engine_manager.h"
#include "graph/debug/ge_attr_define.h"
#include "register/ops_kernel_builder_registry.h"
#include "fftsplus_task_builder.h"
#include "inc/ffts_param_calculator.h"
#include "inc/ffts_configuration.h"
#include "common/string_utils.h"
#include "common/fe_gentask_utils.h"
#include "register/graph_optimizer/fusion_common/unknown_shape_utils.h"
#include "register/op_ext_gentask_registry.h"
namespace ffts {
namespace{
void PrintCtxPathContent(uint32_t type, FftsPlusContextPath &ctx_path) {
  FFTS_LOGD("Ctx_path content: ctx_id:%u, type:%u, pre_cnt:%u, policy_pri:%hu, max_pre_index:%d,"
            "pre_list size:%zu, cmo_list size:%zu, label_list size:%zu, succ_list size:%zu.",
            ctx_path.ctx_id, type, ctx_path.pre_cnt, ctx_path.policy_pri, ctx_path.max_pre_index,
            ctx_path.pre_list.size(), ctx_path.cmo_list.size(), ctx_path.label_list.size(), ctx_path.succ_list.size());
  LoopPrintIntergerVec(ctx_path.pre_list, "PrintCtxPathContent Ctx_path content: ctx_id:%u, pre_list:",
                       ctx_path.ctx_id);
  LoopPrintIntergerVec(ctx_path.cmo_list, "PrintCtxPathContent Ctx_path content: ctx_id:%u, cmo_list:",
                       ctx_path.ctx_id);
  LoopPrintIntergerVec(ctx_path.label_list, "PrintCtxPathContent Ctx_path content: ctx_id:%u, label_list:",
                       ctx_path.ctx_id);
  LoopPrintIntergerVec(ctx_path.succ_list, "PrintCtxPathContent Ctx_path content: ctx_id:%u, succ_list:",
                       ctx_path.ctx_id);
}
} // namespace

const std::string kFFTSPlusCoreName = "ffts_plus";
const uint64_t max_preload_context_num = 1000;
constexpr uint32_t kUInt8Max = 255;

const std::unordered_set<uint32_t> SUPPORT_CTX_PATH_TYPE {
  RT_CTX_TYPE_AICORE,
  RT_CTX_TYPE_AIV,
  RT_CTX_TYPE_MIX_AIC,
  RT_CTX_TYPE_MIX_AIV
};

static const std::unordered_map<rtFftsPlusContextType_t, std::string> kCtxTypeStrMap = {
    {RT_CTX_TYPE_AICORE, "aicore"},
    {RT_CTX_TYPE_AIV, "aiv"},
    {RT_CTX_TYPE_NOTIFY_WAIT, "notify_wait"},
    {RT_CTX_TYPE_NOTIFY_RECORD, "notify_record"},
    {RT_CTX_TYPE_WRITE_VALUE, "write_value"},
    {RT_CTX_TYPE_MIX_AIC, "mix_aic"},
    {RT_CTX_TYPE_MIX_AIV, "mix_aiv"},
    {RT_CTX_TYPE_SDMA, "sdma"},
    {RT_CTX_TYPE_FLUSH_DATA, "flush_data"},
    {RT_CTX_TYPE_INVALIDATE_DATA, "invalidate_data"},
    {RT_CTX_TYPE_WRITEBACK_DATA, "writeback_data"},
    {RT_CTX_TYPE_AICPU, "aicpu"},
    {RT_CTX_TYPE_COND_SWITCH, "cond_switch"},
    {RT_CTX_TYPE_CASE_SWITCH, "case_switch"},
    {RT_CTX_TYPE_AT_START, "at_start"},
    {RT_CTX_TYPE_AT_END, "at_end"},
    {RT_CTX_TYPE_LABEL, "label"},
    {RT_CTX_TYPE_PERSISTENT_CACHE, "persistent_cache"},
    {RT_CTX_TYPE_DSA, "dsa"}
};

REGISTER_OPS_KERNEL_BUILDER(kFFTSPlusCoreName, FFTSPlusOpsKernelBuilder);

FFTSPlusOpsKernelBuilder::FFTSPlusOpsKernelBuilder() {}

FFTSPlusOpsKernelBuilder::~FFTSPlusOpsKernelBuilder() {}

Status FFTSPlusOpsKernelBuilder::Initialize(const std::map<std::string, std::string> &options) {
  (void)options;
  FFTS_LOGI("Begin to init FFTSPlusOpsKernelBuilder.");
  if (GetPlatformFFTSMode() != FFTSMode::FFTS_MODE_FFTS_PLUS) {
    FFTS_LOGW("FFTSPlusOpsKernelBuilder ffts_plus flag is 0.");
    return SUCCESS;
  }
  if (init_flag_) {
    FFTS_LOGW("FFTSPlusOpsKernelBuilder has been initialized.");
    return SUCCESS;
  }

  Status ret = InitLibPath();
  if (ret != SUCCESS) {
    FFTS_LOGW("Failed to get the so path.");
    return FAILED;
  }
  const string SCHECULE_POLICY_PASS_FUNC_NAME = "ScheculePolicyPass";
  const std::string SCHECULE_POLICY_PASS_PLUGIN = "libascend_sch_policy_pass.so";

  string plugin_path = lib_path_ + SCHECULE_POLICY_PASS_PLUGIN;
  FFTS_MAKE_SHARED(sch_policy_pass_plugin_ = std::make_shared<PluginManager>(plugin_path), return FAILED);
  FFTS_CHECK(sch_policy_pass_plugin_ == nullptr, REPORT_FFTS_ERROR(
      "[FFTSPlusOpsKernelBuilder][Init] [InitSchPolicyPassPlg] Failed to create schedule policy pass plugin manager pointer."),
  return FAILED);
  if (sch_policy_pass_plugin_->OpenPlugin(plugin_path) != SUCCESS) {
    skip_schecule_policy_pass_ = true;
    FFTS_LOGW("Failed to open %s.", plugin_path.c_str());
  } else {
    ret = sch_policy_pass_plugin_->GetFunction<Status, domi::TaskDef &, std::vector<ffts::FftsPlusContextPath> &>(
    SCHECULE_POLICY_PASS_FUNC_NAME, schecule_policy_pass_);
    if (ret != SUCCESS) {
      FFTS_LOGW("Failed to get the function %s in the plugin %s.", SCHECULE_POLICY_PASS_FUNC_NAME.c_str(),
                plugin_path.c_str());
      (void)sch_policy_pass_plugin_->CloseHandle();
      return FAILED;
    }
  }
  FFTS_MAKE_SHARED(ffts_plus_task_builder_ptr_ = std::make_shared<FFTSPlusTaskBuilder>(), return FAILED);
  init_flag_ = true;
  FFTS_LOGI("Initialize FFTSPlusOpsKernelBuilder success.");
  return SUCCESS;
}

Status FFTSPlusOpsKernelBuilder::Finalize() {
  if (!init_flag_) {
    FFTS_LOGW("FFTSPlusOpsKernelBuilder finalize is not allowed, initialize first is necessary.");
    return SUCCESS;
  }
  if (!skip_schecule_policy_pass_) {
    FFTS_CHECK_NOTNULL(sch_policy_pass_plugin_);
    (void)sch_policy_pass_plugin_->CloseHandle();
  }
  init_flag_ = false;
  FFTS_LOGD("Finalized FFTSPlusOpsKernelBuilder success.");
  return SUCCESS;
}

Status FFTSPlusOpsKernelBuilder::CalcOpRunningParam(ge::Node &node) {
  (void)node;
  return SUCCESS;
}

TheadTaskBuilderPtr FFTSPlusOpsKernelBuilder::GetNormBuilder(const ge::Node &node, ge::ComputeGraphPtr &sgt_graph,
                                                             domi::TaskDef &task_def, uint64_t &ready_num,
                                                             uint64_t &total_num) {
  ge::OpDescPtr op_desc = node.GetOpDesc();
  TheadTaskBuilderPtr base_mode_ptr = nullptr;
  std::string sub_graph_name = op_desc->GetSubgraphInstanceName(0);
  if (sub_graph_name.empty()) {
    return base_mode_ptr;
  }
  auto ai_graph = node.GetOwnerComputeGraph();
  if (ai_graph == nullptr) {
    return base_mode_ptr;
  }
  auto root_graph = ge::GraphUtils::FindRootGraph(ai_graph);
  if (root_graph == nullptr) {
    return base_mode_ptr;
  }
  sgt_graph = root_graph->GetSubgraph(sub_graph_name);
  if (sgt_graph == nullptr) {
    return base_mode_ptr;
  }
  Status status = GenPersistentContext(node, ready_num, total_num, task_def);
  if (status != SUCCESS) {
    return base_mode_ptr;
  }
  base_mode_ptr = GetFftsPlusMode(node, *sgt_graph);
  return base_mode_ptr;
}

Status FFTSPlusOpsKernelBuilder::ChooseGenFftsPlusContextId(TheadTaskBuilderPtr base_mode_ptr,
                                                            ge::ComputeGraphPtr sgt_graph,
                                                            std::vector<ge::NodePtr> &sub_graph_nodes,
                                                            const ge::Node &node,
                                                            std::pair<uint64_t, uint64_t> &context_num) const {
  FFTS_CHECK_NOTNULL(base_mode_ptr);
  (void)base_mode_ptr->Initialize();
  uint64_t ready_context_num = context_num.first;
  uint64_t total_context_number = context_num.second;
  Status status;
  std::vector<ge::NodePtr> atomic_nodes;
  if (CONTROL_OP_V2_TYPE.find(node.GetType()) == CONTROL_OP_V2_TYPE.end()) {
    status = base_mode_ptr->GenFftsPlusContextId(*sgt_graph, sub_graph_nodes, ready_context_num, total_context_number,
                                                 atomic_nodes);
  } else {
    FFTS_LOGD("ChooseGenFftsPlusContextId RTSCONTROL");
    auto owngraph = node.GetOwnerComputeGraph();
    ge::Node &temp_node = const_cast<ge::Node&>(node);
    ge::NodePtr node_ptr = temp_node.shared_from_this();
    FFTS_CHECK_NOTNULL(node_ptr);
    sub_graph_nodes.emplace_back(node_ptr);
    status = base_mode_ptr->GenFftsPlusContextId(*owngraph, sub_graph_nodes, ready_context_num, total_context_number,
                                                 atomic_nodes);
  }
  context_num.first = ready_context_num;
  context_num.second = total_context_number;
  return status;
}

Status FFTSPlusOpsKernelBuilder::InitLibPath() {
  Dl_info dl_info;
  EngineManager &(*instance_ptr)(const std::string &) = &EngineManager::Instance;
  if (dladdr(reinterpret_cast<void *>(instance_ptr), &dl_info) == 0) {
    REPORT_FFTS_ERROR("[FFTSPlusOpsKernelBuilder][Init][InitLibPath] Failed to get so file path.");
    return FAILED;
  } else {
    std::string so_path = dl_info.dli_fname;
    FFTS_LOGD("Library SO file path is: %s.", so_path.c_str());

    if (so_path.empty()) {
      REPORT_FFTS_ERROR("[FFTSPlusOpsKernelBuilder][Init][InitLibPath] So file path is empty.");
      return FAILED;
    }

    lib_path_ = RealPath(so_path);
    int32_t pos = lib_path_.rfind('/');
    if (pos < 0) {
      REPORT_FFTS_ERROR("[FFTSPlusOpsKernelBuilder][Init][InitLibPath] The path for the .so file does not contain /.");
      return FAILED;
    }

    lib_path_ = lib_path_.substr(0, pos + 1);
  }
  FFTS_LOGD("The real path of lib is %s.", lib_path_.c_str());
  return SUCCESS;
}

Status FFTSPlusOpsKernelBuilder::SetSingleCtxPolicyPri(uint32_t type, domi::FftsPlusCtxDef *ffts_plus_ctx_def,
                                                       FftsPlusContextPath &ctx_path) const {
  switch (type) {
    case RT_CTX_TYPE_AICORE:
    case RT_CTX_TYPE_AIV:
      return FFTSPlusTaskBuilder::set_policy_pri(ctx_path.ctx_id, ctx_path.policy_pri,
                                                 ffts_plus_ctx_def->mutable_aic_aiv_ctx());
    case RT_CTX_TYPE_MIX_AIC:
    case RT_CTX_TYPE_MIX_AIV:
      return FFTSPlusTaskBuilder::set_policy_pri(ctx_path.ctx_id, ctx_path.policy_pri,
                                                 ffts_plus_ctx_def->mutable_mix_aic_aiv_ctx());
    default:
      // other type ctx no need to process
      return SUCCESS;
  }
  return SUCCESS;
}

Status FFTSPlusOpsKernelBuilder::SetCtxsPolicyPri(
    uint64_t ready_context_num, domi::TaskDef &task_def, TimeLineOptimizerContext &timeCtx) const
{
  FFTS_LOGD("Set policy priority in context.");
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  FFTS_CHECK_NOTNULL(ffts_plus_task_def);
  auto &ctx_path_vector = timeCtx.ctx_path_vector_;
  auto &cmo_id_map = timeCtx.cmo_id_map_;
  for (auto it = ctx_path_vector.begin(); it != ctx_path_vector.end(); ++it) {
    FftsPlusContextPath &ctx_path = *it;
    domi::FftsPlusCtxDef *ffts_plus_ctx_def =
        ffts_plus_task_def->mutable_ffts_plus_ctx(static_cast<int>(ctx_path.ctx_id));
    FFTS_CHECK_NOTNULL(ffts_plus_ctx_def);

    uint32_t type = ffts_plus_ctx_def->context_type();
    PrintCtxPathContent(type, ctx_path);
    for (auto cmo_id : ctx_path.cmo_list) {
      if (cmo_id_map.find(cmo_id) == cmo_id_map.end()) {
        cmo_id_map[cmo_id] = {ctx_path.ctx_id};
      } else {
        cmo_id_map[cmo_id].emplace_back(ctx_path.ctx_id);
      }
    }

    // check precnt is 0 ctx invalid after ctx created, citx id must smaller than ready_context_num
    if (ctx_path.pre_list.empty() && ctx_path.ctx_id >= ready_context_num) {
      FFTS_LOGW("Precnt is 0 ctx %u invalid not successfully, ready_context_num: %lu.", ctx_path.ctx_id, ready_context_num);
    }

    if (SUPPORT_CTX_PATH_TYPE.count(type) == 0) {
      continue;
    }

    if (SetSingleCtxPolicyPri(type, ffts_plus_ctx_def, ctx_path) != SUCCESS) {
      return FAILED;
    }
  }
  return SUCCESS;
}

inline Status SubStractPrecnt(uint32_t ctx_id, domi::FftsPlusTaskDef *ffts_plus_task_def) {
  domi::FftsPlusCtxDef *ffts_plus_ctx = ffts_plus_task_def->mutable_ffts_plus_ctx(static_cast<int>(ctx_id));
  FFTS_CHECK_NOTNULL(ffts_plus_ctx);
  uint32_t type = ffts_plus_ctx->context_type();
  FFTS_LOGD("Current type: %u for subtracting pre-cnt in context %u.", type, ctx_id);
  switch (type) {
    case RT_CTX_TYPE_AICORE:
    case RT_CTX_TYPE_AIV:
      FFTS_LOGD("Update pre_cnt for aic/aiv context: %u.", ctx_id);
      return FFTSPlusTaskBuilder::UpdateCtxPredCnt(ffts_plus_ctx->mutable_aic_aiv_ctx(), -1);
    case RT_CTX_TYPE_MIX_AIC:
    case RT_CTX_TYPE_MIX_AIV:
      FFTS_LOGD("Update pre_cnt for mix aic/aiv context %u.", ctx_id);
      return FFTSPlusTaskBuilder::UpdateCtxPredCnt(ffts_plus_ctx->mutable_mix_aic_aiv_ctx(), -1);
    case RT_CTX_TYPE_AICPU:
      FFTS_LOGD("Update pre_cnt for aicpu context %u.", ctx_id);
      return FFTSPlusTaskBuilder::UpdateCtxPredCnt(ffts_plus_ctx->mutable_aicpu_ctx(), -1);
    case RT_CTX_TYPE_SDMA:
      return FFTSPlusTaskBuilder::UpdateCtxPredCnt(ffts_plus_ctx->mutable_sdma_ctx(), -1);
    case RT_CTX_TYPE_NOTIFY_WAIT:
    case RT_CTX_TYPE_NOTIFY_RECORD:
      return FFTSPlusTaskBuilder::UpdateCtxPredCnt(ffts_plus_ctx->mutable_notify_ctx(), -1);
    case RT_CTX_TYPE_WRITE_VALUE:
      return FFTSPlusTaskBuilder::UpdateCtxPredCnt(ffts_plus_ctx->mutable_write_value_ctx(), -1);
    case RT_CTX_TYPE_CASE_SWITCH:
      return FFTSPlusTaskBuilder::UpdateCtxPredCnt(ffts_plus_ctx->mutable_case_switch_ctx(), -1);
    case RT_CTX_TYPE_COND_SWITCH:
      return FFTSPlusTaskBuilder::UpdateCtxPredCnt(ffts_plus_ctx->mutable_cond_switch_ctx(), -1);
    case RT_CTX_TYPE_DSA:
      return FFTSPlusTaskBuilder::UpdateCtxPredCnt(ffts_plus_ctx->mutable_dsa_ctx(), -1);
    case RT_CTX_TYPE_LABEL:
      return FFTSPlusTaskBuilder::UpdateCtxPredCnt(ffts_plus_ctx->mutable_label_ctx(), -1);
    case RT_CTX_TYPE_INVALIDATE_DATA:
      return FFTSPlusTaskBuilder::UpdateDataPredCnt(ffts_plus_ctx->mutable_data_ctx(), -1);
    default:
      FFTS_LOGI("Type %u does not require an update of pre_cnt.", type);
      return FAILED;
  }
}

inline Status SetSuccList(domi::FftsPlusTaskDef *ffts_plus_task_def, uint32_t ctx_id,
                          const vector<uint32_t> &reserve_ctx_list, const vector<uint32_t> &label_list) {
  domi::FftsPlusCtxDef *ffts_plus_ctx = ffts_plus_task_def->mutable_ffts_plus_ctx(static_cast<int>(ctx_id));
  FFTS_CHECK_NOTNULL(ffts_plus_ctx);
  uint32_t type = ffts_plus_ctx->context_type();
  FFTS_LOGD("Current type: %u for update success list.", type);
  switch (type) {
    case RT_CTX_TYPE_AICORE:
    case RT_CTX_TYPE_AIV:
      FFTS_LOGD("Update succ_list for aic/aiv context: %u.", ctx_id);
      return FFTSPlusTaskBuilder::UpdateSuccList(ffts_plus_task_def, ffts_plus_ctx->mutable_aic_aiv_ctx(),
                                                 reserve_ctx_list, label_list);
    case RT_CTX_TYPE_MIX_AIC:
    case RT_CTX_TYPE_MIX_AIV:
      FFTS_LOGD("Update succ_list for mixed AIC/AIV context: %u.", ctx_id);
      return FFTSPlusTaskBuilder::UpdateSuccList(ffts_plus_task_def, ffts_plus_ctx->mutable_mix_aic_aiv_ctx(),
                                                 reserve_ctx_list, label_list);
    case RT_CTX_TYPE_AICPU:
      FFTS_LOGD("Update succ_list for aicpu context %u.", ctx_id);
      return FFTSPlusTaskBuilder::UpdateSuccList(ffts_plus_task_def, ffts_plus_ctx->mutable_aicpu_ctx(),
                                                 reserve_ctx_list, label_list);
    case RT_CTX_TYPE_SDMA:
      return FFTSPlusTaskBuilder::UpdateSuccList(ffts_plus_task_def, ffts_plus_ctx->mutable_sdma_ctx(),
                                                 reserve_ctx_list, label_list);
    case RT_CTX_TYPE_NOTIFY_WAIT:
    case RT_CTX_TYPE_NOTIFY_RECORD:
      return FFTSPlusTaskBuilder::UpdateSuccList(ffts_plus_task_def, ffts_plus_ctx->mutable_notify_ctx(),
                                                 reserve_ctx_list, label_list);
    case RT_CTX_TYPE_WRITE_VALUE:
      return FFTSPlusTaskBuilder::UpdateSuccList(ffts_plus_task_def, ffts_plus_ctx->mutable_write_value_ctx(),
                                                 reserve_ctx_list, label_list);
    case RT_CTX_TYPE_CASE_SWITCH:
      return FFTSPlusTaskBuilder::UpdateSuccList(ffts_plus_task_def, ffts_plus_ctx->mutable_case_switch_ctx(),
                                                 reserve_ctx_list, label_list);
    case RT_CTX_TYPE_DSA:
      return FFTSPlusTaskBuilder::UpdateSuccList(ffts_plus_task_def, ffts_plus_ctx->mutable_dsa_ctx(),
                                                 reserve_ctx_list, label_list);
    case RT_CTX_TYPE_LABEL:
      return FFTSPlusTaskBuilder::UpdateSuccList(ffts_plus_task_def, ffts_plus_ctx->mutable_label_ctx(),
                                                 reserve_ctx_list, label_list);
    case RT_CTX_TYPE_INVALIDATE_DATA:
      return FFTSPlusTaskBuilder::UpdateSuccList(ffts_plus_task_def, ffts_plus_ctx->mutable_data_ctx(),
                                                 reserve_ctx_list, label_list);
    default:
      FFTS_LOGI("type %u does not require an update to its successor list.", type);
      return FAILED;
  }
}

Status FFTSPlusOpsKernelBuilder::RemoveFromPreList(const uint32_t ctx_id, const uint32_t next_ctx_id,
                                                   domi::FftsPlusTaskDef *ffts_plus_task_def,
                                                   TimeLineOptimizerContext &timeCtx,
                                                   const std::unordered_map<uint32_t, size_t> &ctx_index_map) const {
  domi::FftsPlusCtxDef *ffts_plus_ctx = ffts_plus_task_def->mutable_ffts_plus_ctx(static_cast<int>(next_ctx_id));
  FFTS_CHECK_NOTNULL(ffts_plus_ctx);
  uint32_t type = ffts_plus_ctx->context_type();
  if (type == RT_CTX_TYPE_WRITEBACK_DATA) {
    return SUCCESS;
  }
  auto &context_paths = timeCtx.ctx_path_vector_;
  size_t path_index = 0;

  Status ret = GetIndexByCtxId(next_ctx_id, path_index, ctx_index_map);
  if (ret != SUCCESS) {
    FFTS_LOGD("Context id %u can not find in context_vector_path, try to remove prelist in cmo_id_map.", next_ctx_id);
    auto cmo_prelist_iter = timeCtx.cmo_id_map_.find(next_ctx_id);
    if (cmo_prelist_iter != timeCtx.cmo_id_map_.end()) {
      auto iter = std::find(cmo_prelist_iter->second.begin(), cmo_prelist_iter->second.end(), ctx_id);
      if (iter != cmo_prelist_iter->second.end()) {
        cmo_prelist_iter->second.erase(iter);
      }
    }
    return SUCCESS;
  }
  if (GetIndexByCtxId(next_ctx_id, path_index, ctx_index_map) != SUCCESS) {
    return SUCCESS;
  }
  FftsPlusContextPath &context_path = context_paths[path_index];
  FFTS_LOGD("Before removing context %u from pre_list: %s.", ctx_id,
            fe::StringUtils::IntegerVecToString(context_path.pre_list).c_str());
  auto iter = std::find(context_path.pre_list.begin(), context_path.pre_list.end(), ctx_id);
  if (iter != context_path.pre_list.end()) {
    context_path.pre_list.erase(iter);
  }
  FFTS_LOGD("After removing context %u from pre_list: %s.", ctx_id,
            fe::StringUtils::IntegerVecToString(context_path.pre_list).c_str());
  return SUCCESS;
}

Status FFTSPlusOpsKernelBuilder::UpdateContextForRemoveDuplicate(domi::FftsPlusTaskDef *ffts_plus_task_def,
    const ContextParam &ctx_param,
    const std::unordered_map<uint32_t, size_t> &ctx_index_map,
    TimeLineOptimizerContext &timeCtx) const {
  FFTS_LOGD("UpdateContext ctx_id: %u, label_list size: %zu, remove_ctx_list size: %u, reserve_ctx_list size: %zu.",
            ctx_param.ctx_id_, ctx_param.context_path_.label_list.size(), ctx_param.remove_ctx_list_.size(),
            ctx_param.reserve_ctx_list_.size());
  domi::FftsPlusCtxDef *ffts_plus_ctx = ffts_plus_task_def->mutable_ffts_plus_ctx(static_cast<int>(ctx_param.ctx_id_));
  FFTS_CHECK_NOTNULL(ffts_plus_ctx);
  FFTS_LOGD("Before updateContextForRemoveDuplicate, context: %s.", ffts_plus_ctx->DebugString().c_str());

  // update succ_list's pre_cnt
  for (size_t i = 0; i < ctx_param.remove_ctx_list_.size(); ++i) {
    if (SubStractPrecnt(ctx_param.remove_ctx_list_[i], ffts_plus_task_def) != SUCCESS) {
      return FAILED;
    }
    if (RemoveFromPreList(ctx_param.ctx_id_, ctx_param.remove_ctx_list_[i], ffts_plus_task_def,
        timeCtx, ctx_index_map) != SUCCESS) {
      return FAILED;
    }
  }

  if (SetSuccList(ffts_plus_task_def, ctx_param.ctx_id_,
      ctx_param.reserve_ctx_list_, ctx_param.context_path_.label_list) != SUCCESS) {
    return FAILED;
  }
  FFTS_LOGD("After updateContextForRemoveDuplicate context:%s.", ffts_plus_ctx->DebugString().c_str());
  return SUCCESS;
}

Status FFTSPlusOpsKernelBuilder::GenLabelForPreList(domi::FftsPlusTaskDef *ffts_plus_task_def, const uint32_t ctx_id,
                                                    const vector<uint32_t> &pre_list) const {
  if (pre_list.empty()) {
    return SUCCESS;
  }
  uint32_t pred_cnt = ffts_plus_task_builder_ptr_->GetPreCnt(ctx_id, ffts_plus_task_def);
  FFTS_LOGD("Ctx %u, precnt size: %u, prelist size: %zu.", ctx_id, pred_cnt, pre_list.size());
  if (pred_cnt <= kUInt8Max) {
    return SUCCESS;
  }
  if (pred_cnt != static_cast<uint32_t>(pre_list.size())) {
    return SUCCESS;
  }

  uint32_t label_count = pred_cnt / kUInt8Max;
  for (uint32_t label_index = 0; label_index < label_count; ++label_index) {
    uint32_t new_label_ctx_id = ffts_plus_task_def->ffts_plus_ctx_size();
    domi::FftsPlusCtxDef *new_ctx = ffts_plus_task_def->add_ffts_plus_ctx();
    FFTS_CHECK_NOTNULL(new_ctx);
    new_ctx->set_context_type(RT_CTX_TYPE_LABEL);
    domi::FftsPlusLabelCtxDef *new_label = new_ctx->mutable_label_ctx();
    // remove current ctx_id from prelist id succ list, and then add label ctx_id to prelist id succ list
    // update current ctx_id to lable succ_list
    // set label ctx and current ctx precnt
    int32_t success_count = 0;
    for (uint32_t index = 0; index < kUInt8Max; ++index) {
      uint32_t pre_id = pre_list[label_index * kUInt8Max + index];
      if (ffts_plus_task_builder_ptr_->ReplaceSuccList(ctx_id, new_label_ctx_id, pre_id, ffts_plus_task_def) ==
          SUCCESS) {
        ++success_count;
      }
    }
    FFTS_LOGD("Ctx %u successfully removed precnt: %d.", ctx_id, success_count);
    ffts_plus_task_builder_ptr_->UpdateSuccList(ctx_id, new_label_ctx_id, ffts_plus_task_def);
    new_label->set_pred_cnt(success_count);
    new_label->set_pred_cnt_init(success_count);
    ffts_plus_task_builder_ptr_->UpdatePreCnt(ctx_id, ffts_plus_task_def, 1 - success_count);
  }
  return SUCCESS;
}

void FFTSPlusOpsKernelBuilder::PrintTaskDefContent(domi::FftsPlusTaskDef *ffts_plus_task_def) const {
  uint64_t gen_ctx_size = ffts_plus_task_def->ffts_plus_ctx_size();
  for (size_t i = 0; i < gen_ctx_size; i++) {
    FFTS_LOGD("TaskDefContent before optimize dependencies, context_id: %zu, context: %s.", i,
              ffts_plus_task_def->mutable_ffts_plus_ctx(static_cast<int>(i))->DebugString().c_str());
  }
}

void FFTSPlusOpsKernelBuilder::SortContextPathByMaxPreIndex(const vector<FftsPlusContextPath> &context_paths,
                                                            vector<FftsPlusContextPath> &sort_context_paths) const {
  std::list<size_t> ctx_id_sort;
  for (size_t i = 0; i < context_paths.size(); ++i) {
    if (ctx_id_sort.empty()) {
      ctx_id_sort.emplace_back(i);
      continue;
    }
    auto it = ctx_id_sort.begin();
    for (; it != ctx_id_sort.end(); ++it) {
      if (context_paths[*it].max_pre_index > context_paths[i].max_pre_index) {
        ctx_id_sort.insert(it, i);
        break;
      }
    }
    if (it == ctx_id_sort.end()) {
      ctx_id_sort.insert(it, i);
    }
  }

  size_t idx = 0;
  for (auto it : ctx_id_sort) {
    sort_context_paths.emplace_back(context_paths[it]);
    FFTS_LOGD("Sort_context_paths: i=%zu, ctx_id=%u, max_pre_index=%d.", idx,
              sort_context_paths[idx].ctx_id, sort_context_paths[idx].max_pre_index);
    ++idx;
  }
}

inline void SaveDirectPathAndFlag(int j, const FftsPlusContextPath &i_path, const FftsPlusContextPath &j_path,
                                  vector<bool> &flag_topo, map<uint32_t, vector<uint32_t>> &real_ctx_succ_list,
                                  const std::unordered_map<uint32_t, size_t> &ctxid_flag_index_map) {
  if (flag_topo[j] == false) {
    auto iter = real_ctx_succ_list.find(j_path.ctx_id);
    if (iter != real_ctx_succ_list.end()) {
      iter->second.emplace_back(i_path.ctx_id);
    } else {
      real_ctx_succ_list.insert({j_path.ctx_id, {i_path.ctx_id}});
    }

    flag_topo[j] = true;
  }
  for (size_t j_pre = 0; j_pre < j_path.pre_list.size(); j_pre++) {
    auto iter = ctxid_flag_index_map.find(j_path.pre_list[j_pre]);
    if (iter != ctxid_flag_index_map.end()) {
      flag_topo[iter->second] = true;
    } else {
      FFTS_LOGD("Failed to find ctx_id:%u in ctxid_flag_index_map.", j_path.pre_list[j_pre]);
    }
  }
}

inline bool IsConnected(const FftsPlusContextPath &i_path, uint32_t j_ctx_id) {
  for (size_t z = 0; z < i_path.pre_list.size(); ++z) {
    if (j_ctx_id == i_path.pre_list[z]) {
      return true;
    }
  }
  return false;
}

void FFTSPlusOpsKernelBuilder::TransitiveReductionContextPath(const vector<FftsPlusContextPath> &sort_context_paths,
                                                              map<uint32_t, vector<uint32_t>> &real_ctx_succ_list)
                                                              const {
  if (sort_context_paths.empty()) {
    return;
  }
  size_t aicaiv_ctx_size = sort_context_paths.size();
  std::unordered_map<uint32_t, size_t> ctxid_flag_index_map;
  ctxid_flag_index_map.emplace(sort_context_paths[0].ctx_id, 0);
  for (size_t i = 1; i < aicaiv_ctx_size; ++i) {
    ctxid_flag_index_map.emplace(sort_context_paths[i].ctx_id, i);
    vector<bool> flag_topo(i + 1, false);
    const FftsPlusContextPath &i_path = sort_context_paths[i];
    for (int j = i - 1; j >= 0; --j) {
      const FftsPlusContextPath &j_path = sort_context_paths[j];
      uint32_t j_ctx_id = j_path.ctx_id;
      if (!IsConnected(i_path, j_ctx_id) && !flag_topo[j]) {
        continue;
      }
      SaveDirectPathAndFlag(j, i_path, j_path, flag_topo, real_ctx_succ_list, ctxid_flag_index_map);
    }
  }
}

bool FFTSPlusOpsKernelBuilder::GenerateCtxIdIdxMap(const vector<FftsPlusContextPath> &context_paths,
                                                   std::unordered_map<uint32_t, size_t> &ctx_index_map) const {
  for (size_t i = 0; i < context_paths.size(); ++i) {
    const uint32_t &ctx_id = context_paths[i].ctx_id;
    if (ctx_index_map.find(ctx_id) != ctx_index_map.end()) {
      FFTS_LOGE("Unexpected context path, context id [%u] is not unique.", ctx_id);
      return false;
    }
    ctx_index_map.insert(std::pair<uint32_t, size_t>(ctx_id, i));
  }
  return true;
}

Status FFTSPlusOpsKernelBuilder::GetIndexByCtxId(uint32_t ctx_id, size_t &path_index,
                                                 const std::unordered_map<uint32_t, size_t> &ctx_index_map) const {
  const auto &iter = ctx_index_map.find(ctx_id);
  if (iter != ctx_index_map.end()) {
    path_index = iter->second;
    return SUCCESS;
  }
  FFTS_LOGI("Cannot find ctx_id %u in context_paths.", ctx_id);
  return FAILED;
}

inline void RemoveDuplicate(vector<uint32_t> &reserve_ctx_list, vector<uint32_t> &remove_ctx_list) {
  //  A     A
  // | | -> |
  //  B     B
  for (auto it_rsv = reserve_ctx_list.begin(); it_rsv != reserve_ctx_list.end();) {
    if (find(reserve_ctx_list.begin(), it_rsv, *it_rsv) != it_rsv) {
      remove_ctx_list.emplace_back(*it_rsv);
      it_rsv = reserve_ctx_list.erase(it_rsv);
    } else {
      it_rsv++;
    }
  }
}

Status FFTSPlusOpsKernelBuilder::InsertCxtListByPriority(const vector<FftsPlusContextPath> &context_paths,
    uint32_t insert_ctx_id, vector<uint32_t> &ctx_list,
    const std::unordered_map<uint32_t, size_t> &ctx_index_map) const {
  if (ctx_list.empty()) {
    ctx_list.emplace_back(insert_ctx_id);
    return SUCCESS;
  }

  size_t insert_path_index = 0;
  if (GetIndexByCtxId(insert_ctx_id, insert_path_index, ctx_index_map) != SUCCESS) {
    return FAILED;
  }
  auto it = ctx_list.begin();
  for (; it != ctx_list.end(); ++it) {
    size_t path_index = 0;
    if (GetIndexByCtxId(*it, path_index, ctx_index_map) != SUCCESS) {
      return FAILED;
    }
    if (context_paths[path_index].policy_pri < context_paths[insert_path_index].policy_pri) {
      ctx_list.insert(it, insert_ctx_id);
      break;
    }
  }
  if (it == ctx_list.end()) {
    ctx_list.insert(it, insert_ctx_id);
  }
  return SUCCESS;
}

Status FFTSPlusOpsKernelBuilder::InsertCmoCtxToSucclist(domi::FftsPlusTaskDef *ffts_plus_task_def,
                                                        uint32_t cmo_id, ContextParam &ctx_param,
                                                        std::unordered_map<uint32_t, size_t> &ctx_index_map) const {
  domi::FftsPlusCtxDef *ffts_plus_ctx = ffts_plus_task_def->mutable_ffts_plus_ctx(static_cast<int>(cmo_id));
  FFTS_CHECK_NOTNULL(ffts_plus_ctx);
  uint32_t type = ffts_plus_ctx->context_type();
  if (type != RT_CTX_TYPE_INVALIDATE_DATA) {
    ctx_param.reserve_ctx_list_.emplace_back(cmo_id);
    return SUCCESS;
  }
  size_t path_index = 0;
  if (GetIndexByCtxId(cmo_id, path_index, ctx_index_map) == SUCCESS) {
    FFTS_LOGD("Invalid type data for ctx %u in context_paths.", cmo_id);
  } else {
    FFTS_LOGD("Invalid type data context %u inserted into reserve_ctx_list.", cmo_id);
    ctx_param.reserve_ctx_list_.emplace_back(cmo_id);
  }
  return SUCCESS;
}

Status FFTSPlusOpsKernelBuilder::UpdateContexts(domi::FftsPlusTaskDef *ffts_plus_task_def,
                                                map<uint32_t, vector<uint32_t>> &real_ctx_succ_list,
                                                TimeLineOptimizerContext &timeCtx) const {
  auto &context_paths = timeCtx.ctx_path_vector_;
  std::unordered_map<uint32_t, size_t> ctx_index_map;
  if (!GenerateCtxIdIdxMap(context_paths, ctx_index_map)) {
    return FAILED;
  }
  for (auto iter_ctx_succ = real_ctx_succ_list.begin(); iter_ctx_succ != real_ctx_succ_list.end(); ++iter_ctx_succ) {
    ContextParam ctx_param;
    ctx_param.ctx_id_ = iter_ctx_succ->first;

    size_t path_index = 0;
    if (GetIndexByCtxId(ctx_param.ctx_id_, path_index, ctx_index_map) != SUCCESS) {
      return FAILED;
    }
    ctx_param.context_path_ = context_paths[path_index];
    for (size_t i = 0; i < ctx_param.context_path_.succ_list.size(); i++) {
      if (find(iter_ctx_succ->second.begin(), iter_ctx_succ->second.end(), ctx_param.context_path_.succ_list[i]) !=
          iter_ctx_succ->second.end()) {
        // sort reserve_ctx_list by priority
        (void)InsertCxtListByPriority(context_paths, ctx_param.context_path_.succ_list[i],
                                      ctx_param.reserve_ctx_list_, ctx_index_map);
      } else {
        ctx_param.remove_ctx_list_.emplace_back(ctx_param.context_path_.succ_list[i]);
      }
    }

    FFTS_LOGD("Ctx_id: %u, before CMO reserve_ctx_list size: %zu, remove_ctx_list size: %zu.",
        ctx_param.context_path_.ctx_id, ctx_param.reserve_ctx_list_.size(), ctx_param.remove_ctx_list_.size());
    LoopPrintIntergerVec(ctx_param.remove_ctx_list_, "Ctx_id:%u, before cmo remove_ctx_list:",
                         ctx_param.context_path_.ctx_id);
    LoopPrintIntergerVec(ctx_param.reserve_ctx_list_, "Ctx_id:%u, before cmo reserve_ctx_list:",
                         ctx_param.context_path_.ctx_id);

    for (auto cmo : ctx_param.context_path_.cmo_list) {
      // support invalid data ctx for memory reuse
      (void)InsertCmoCtxToSucclist(ffts_plus_task_def, cmo, ctx_param, ctx_index_map);
    }

    RemoveDuplicate(ctx_param.reserve_ctx_list_, ctx_param.remove_ctx_list_);

    FFTS_LOGD("Ctx_id: %u, after CMO and removing duplicates, reservectxlist size: %zu, remove_ctx_list size: %zu.",
        ctx_param.context_path_.ctx_id, ctx_param.reserve_ctx_list_.size(), ctx_param.remove_ctx_list_.size());
    LoopPrintIntergerVec(ctx_param.remove_ctx_list_, "Ctx_id:%u, after cmo and remove same remove_ctx_list:",
                         ctx_param.context_path_.ctx_id);
    LoopPrintIntergerVec(ctx_param.reserve_ctx_list_, "Ctx_id:%u, after cmo and remove same reserve_ctx_list:",
                         ctx_param.context_path_.ctx_id);

    FFTS_LOGD("Before update content ctx_id:%u, succlist size:%zu.", ctx_param.context_path_.ctx_id,
              ctx_param.context_path_.succ_list.size());
    Status status = UpdateContextForRemoveDuplicate(ffts_plus_task_def, ctx_param, ctx_index_map, timeCtx);
    if (status != SUCCESS) {
      FFTS_LOGD("UpdateContextForRemoveDuplicate unsuccessful.");
      return status;
    }
  }
  return SUCCESS;
}

Status FFTSPlusOpsKernelBuilder::UpdateContextsPreList(domi::TaskDef &task_def,
                                                       const vector<FftsPlusContextPath> &context_paths,
                                                       std::unordered_map<uint32_t, vector<uint32_t>> &cmo_id_map)
                                                       const {
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  FFTS_CHECK_NOTNULL(ffts_plus_task_def);
  for (const auto &context_path : context_paths) {
    if (GenLabelForPreList(ffts_plus_task_def, context_path.ctx_id, context_path.pre_list) != SUCCESS) {
      return FAILED;
    }
  }

  for (auto &it : cmo_id_map) {
    if (GenLabelForPreList(ffts_plus_task_def, it.first, it.second) != SUCCESS) {
      return FAILED;
    }
  }
  return SUCCESS;
}

Status FFTSPlusOpsKernelBuilder::RemoveDuplicateDependencies(
    domi::TaskDef &task_def, TimeLineOptimizerContext &timeCtx) const
{
  auto &context_paths = timeCtx.ctx_path_vector_;
  FFTS_LOGD("Start removing duplicate dependencies, context path size is %zu.", context_paths.size());
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  FFTS_CHECK_NOTNULL(ffts_plus_task_def);

  PrintTaskDefContent(ffts_plus_task_def);

  // topo sort
  vector<FftsPlusContextPath> sort_context_paths;
  SortContextPathByMaxPreIndex(context_paths, sort_context_paths);

  // Transitive Reduction
  map<uint32_t, vector<uint32_t>> real_ctx_succ_list;  // ctxid --- finally succlist
  TransitiveReductionContextPath(sort_context_paths, real_ctx_succ_list);

  FFTS_LOGD("Real_ctx_succ_list size: %zu, context_paths size: %zu.", real_ctx_succ_list.size(), context_paths.size());

  // update context's succ_list and succ_num, update succ_list's pre_cnt;
  return UpdateContexts(ffts_plus_task_def, real_ctx_succ_list, timeCtx);
}

Status FFTSPlusOpsKernelBuilder::TimelineLayoutOptimize(uint64_t ready_context_num,
                                                        const ge::Node &node, domi::TaskDef &task_def) const {
  if (skip_schecule_policy_pass_) {
    FFTS_LOGD("Node[%s] skip_schecule_policy_pass_ is true", node.GetNamePtr());
    return SUCCESS;
  }                                                      
  // scan ctx to compute index and inverse_index, the order ctx insert to vector is important, must be correct
  FFTS_CHECK_NOTNULL(schecule_policy_pass_);
  FFTS_TIMECOST_START(schecule_policy_pass);
  TimeLineOptimizerContext timeCtx;
  Status status = schecule_policy_pass_(task_def, timeCtx.ctx_path_vector_);
  FFTS_TIMECOST_END_LOGI(schecule_policy_pass, "SGT.schecule_policy_pass");
  if (status != SUCCESS) {
    FFTS_LOGI("SchedulePolicyPass did not succeed, node name: %s, node type: %s.",
              node.GetName().c_str(), node.GetType().c_str());
    return SUCCESS; // continue to execute task run without TimelineLayoutOptimize
  }

  status = SetCtxsPolicyPri(ready_context_num, task_def, timeCtx);
  if (status != SUCCESS) {
    FFTS_LOGI("SetCtxsPolicyPri not successfully, node name:%s, node type:%s",
              node.GetName().c_str(), node.GetType().c_str());
    return status;
  }
  bool is_dynamic = false;
  (void)ge::AttrUtils::GetBool(node.GetOpDesc(), kFFTSPlusInDynamic, is_dynamic);
  if (is_dynamic) {
    FFTS_LOGD("Partition node [%s] optimized in dynamic jump contexts.", node.GetNamePtr());
    return SUCCESS;
  }
  // remove ctx succlist duplicate dependence
  status = RemoveDuplicateDependencies(task_def, timeCtx);
  if (status != SUCCESS) {
    FFTS_LOGI("RemoveDuplicateDependencies not successfully, node name: %s, node type: %s.", node.GetName().c_str(),
              node.GetType().c_str());
    return status;
  }

  if (UpdateContextsPreList(task_def, timeCtx.ctx_path_vector_, timeCtx.cmo_id_map_) != SUCCESS) {
    return FAILED;
  }
  return SUCCESS;
}
/*
 *  old ctxid list is 0,1,2,3,4,5
 *  label null ctxid is 0 2 4
 *  old ctxlist         new ctxlist
 *    0          x
 *    1                    0
 *    2          x
 *    3                    1
 *    4          x
 *    5                    2
 *    so the relation new_old map will be {(0, 1),(1, 3),(2, 5)}
 */
Status FFTSPlusOpsKernelBuilder::ReBuildCtxIdsRelation(domi::TaskDef &task_def,
                                                       std::unordered_map<uint32_t, uint32_t> &new_old_map,
                                                       std::unordered_map<uint32_t, uint32_t> &old_new_map) const {
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  FFTS_CHECK_NOTNULL(ffts_plus_task_def);
  uint32_t gen_ctx_size = ffts_plus_task_def->ffts_plus_ctx_size();
  uint32_t start_idx = 0;
  bool need_rebuild = false;
  for (uint32_t i = 0; i < gen_ctx_size; ++i) {
    domi::FftsPlusCtxDef *ffts_plus_ctx = ffts_plus_task_def->mutable_ffts_plus_ctx(i);
    FFTS_CHECK_NOTNULL(ffts_plus_ctx);
    uint32_t type = ffts_plus_ctx->context_type();
    if (type == RT_CTX_TYPE_LABEL) {
      domi::FftsPlusLabelCtxDef *label_ctx_def = ffts_plus_ctx->mutable_label_ctx();
      FFTS_CHECK_NOTNULL(label_ctx_def);
      if (label_ctx_def->successor_num() == 0 && label_ctx_def->pred_cnt() == 0) {
        FFTS_LOGD("ReBuildCtxIdsRelation old ctxid:%u is null label, we need rebuild ctx relation.", i);
        need_rebuild = true;
        continue;
      }
    }
    FFTS_LOGD("ReBuildCtxIdsRelation create map old ctxid:%zu, new ctxid:%u.", i, start_idx);
    new_old_map.emplace(std::make_pair(start_idx, i));
    old_new_map.emplace(std::make_pair(i, start_idx));
    ++start_idx;
  }
  if (need_rebuild) {
    FFTS_LOGD("We should rebuild the context relation; old ctx size: %u, new ctx size: %zu.", gen_ctx_size,
              new_old_map.size());
    return SUCCESS;
  }
  return FAILED;
}

Status FFTSPlusOpsKernelBuilder::UpdateCtxSuccList(domi::FftsPlusCtxDef *ctx_def_old,
                                                   domi::FftsPlusCtxDef *ctx_def_new,
                                                   std::unordered_map<uint32_t, uint32_t> &old_new_map) const {
  if (ctx_def_old->context_type() == RT_CTX_TYPE_AT_START) {
    return UpdateNewCtxSuccList(ctx_def_old->mutable_at_start_ctx(), ctx_def_new->mutable_at_start_ctx(), old_new_map);
  }
  if (ctx_def_old->context_type() == RT_CTX_TYPE_AICORE || ctx_def_old->context_type() == RT_CTX_TYPE_AIV) {
    return UpdateNewCtxSuccList(ctx_def_old->mutable_aic_aiv_ctx(), ctx_def_new->mutable_aic_aiv_ctx(), old_new_map);
  }
  if (ctx_def_old->context_type() == RT_CTX_TYPE_MIX_AIC || ctx_def_old->context_type() == RT_CTX_TYPE_MIX_AIV) {
    return UpdateNewCtxSuccList(ctx_def_old->mutable_mix_aic_aiv_ctx(),
                                ctx_def_new->mutable_mix_aic_aiv_ctx(), old_new_map);
  }
  if (ctx_def_old->context_type() == RT_CTX_TYPE_AICPU) {
    return UpdateNewCtxSuccList(ctx_def_old->mutable_aicpu_ctx(), ctx_def_new->mutable_aicpu_ctx(), old_new_map);
  }
  if (ctx_def_old->context_type() == RT_CTX_TYPE_SDMA) {
    return UpdateNewCtxSuccList(ctx_def_old->mutable_sdma_ctx(), ctx_def_new->mutable_sdma_ctx(), old_new_map);
  }
  if (ctx_def_old->context_type() == RT_CTX_TYPE_NOTIFY_WAIT ||
      ctx_def_old->context_type() == RT_CTX_TYPE_NOTIFY_RECORD) {
    return UpdateNewCtxSuccList(ctx_def_old->mutable_notify_ctx(), ctx_def_new->mutable_notify_ctx(), old_new_map);
  }
  if (ctx_def_old->context_type() == RT_CTX_TYPE_WRITE_VALUE) {
    return UpdateNewCtxSuccList(ctx_def_old->mutable_write_value_ctx(),
                                ctx_def_new->mutable_write_value_ctx(), old_new_map);
  }
  if (ctx_def_old->context_type() == RT_CTX_TYPE_CASE_SWITCH) {
    return UpdateNewCtxSuccList(ctx_def_old->mutable_case_switch_ctx(),
                                ctx_def_new->mutable_case_switch_ctx(), old_new_map);
  }
  if (ctx_def_old->context_type() == RT_CTX_TYPE_LABEL) {
    return UpdateNewCtxSuccList(ctx_def_old->mutable_label_ctx(), ctx_def_new->mutable_label_ctx(), old_new_map);
  }
  if (ctx_def_old->context_type() == RT_CTX_TYPE_DSA) {
    return UpdateNewCtxSuccList(ctx_def_old->mutable_dsa_ctx(), ctx_def_new->mutable_dsa_ctx(), old_new_map);
  }
  if (ctx_def_old->context_type() == RT_CTX_TYPE_INVALIDATE_DATA ||
      ctx_def_old->context_type() == RT_CTX_TYPE_WRITEBACK_DATA) {
    return UpdateNewCtxSuccList(ctx_def_old->mutable_data_ctx(), ctx_def_new->mutable_data_ctx(), old_new_map);
  }
  return SUCCESS;
}

Status FFTSPlusOpsKernelBuilder::UpdateCtxSrcSlotId(domi::FftsPlusCtxDef *ctx_def_old,
                                                    domi::FftsPlusCtxDef *ctx_def_new,
                                                    std::unordered_map<uint32_t, uint32_t> &old_new_map) const {
  if (ctx_def_old->context_type() == RT_CTX_TYPE_MIX_AIC || ctx_def_old->context_type() == RT_CTX_TYPE_MIX_AIV) {
    return UpdateNewCtxSrcSlotId(ctx_def_old->mutable_aic_aiv_ctx(), ctx_def_new->mutable_aic_aiv_ctx(), old_new_map);
  }
  if (ctx_def_old->context_type() == RT_CTX_TYPE_MIX_AIC || ctx_def_old->context_type() == RT_CTX_TYPE_MIX_AIV) {
    return UpdateNewCtxSrcSlotId(ctx_def_old->mutable_aic_aiv_ctx(), ctx_def_new->mutable_aic_aiv_ctx(), old_new_map);
  }
  return SUCCESS;
}

Status FFTSPlusOpsKernelBuilder::ConvertOldCtxInfoToNewCtx(domi::FftsPlusCtxDef *ctx_def_old,
                                                           domi::FftsPlusCtxDef *ctx_def_new,
                                                           std::unordered_map<uint32_t, uint32_t> &old_new_map) const {
  /*
   * The proto construct will make sure the mem inside can be allocated
   */
  *ctx_def_new = *ctx_def_old;
  ctx_def_new->set_context_id(old_new_map[ctx_def_old->context_id()]);
  if (kCtxTypeStrMap.find(static_cast<rtFftsPlusContextType_t>(ctx_def_old->context_type())) == kCtxTypeStrMap.end()) {
    FFTS_LOGW("Type %u does not support conversion from old ctxid to new ctxid.", ctx_def_old->context_type());
    return FAILED;
  }
  Status ret = UpdateCtxSuccList(ctx_def_old, ctx_def_new, old_new_map);
  (void)UpdateCtxSrcSlotId(ctx_def_old, ctx_def_new, old_new_map);
  return ret;
}

Status FFTSPlusOpsKernelBuilder::GenNewSubGraphTaskDef(const ge::Node &node,
                                                       domi::TaskDef &task_def,
                                                       domi::TaskDef &task_def_new,
                                                       std::unordered_map<uint32_t, uint32_t> &new_old_map,
                                                       std::unordered_map<uint32_t, uint32_t> &old_new_map) const {
  FFTS_LOGD("FFTSPlusOpsKernelBuilder GenNewSubGraphTaskDef node name:%s, node type:%s, new ctx size:%zu.",
            node.GetName().c_str(), node.GetType().c_str(), new_old_map.size());
  Status ret = SUCCESS;
  for (size_t i = 0; i < new_old_map.size(); ++i) {
    domi::FftsPlusTaskDef *ffts_plus_task_def_new = task_def_new.mutable_ffts_plus_task();
    FFTS_CHECK_NOTNULL(ffts_plus_task_def_new);
    domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
    FFTS_CHECK_NOTNULL(ffts_plus_task_def);
    ffts_plus_task_def_new->set_op_index(ffts_plus_task_def->op_index());
    ffts_plus_task_def_new->set_addr_size(ffts_plus_task_def->addr_size());

    for (unsigned int j = 0; j < static_cast<unsigned int>(ffts_plus_task_def->additional_data_size()); ++j) {
      domi::AdditionalDataDef *additional_data_def = ffts_plus_task_def_new->add_additional_data();
      FFTS_CHECK_NOTNULL(additional_data_def);
      *additional_data_def = ffts_plus_task_def->additional_data(j);
    }
    domi::FftsPlusCtxDef *ffts_plus_ctx_def_new = ffts_plus_task_def_new->add_ffts_plus_ctx();
    auto old_ctxid = new_old_map[i];
    domi::FftsPlusCtxDef *ffts_plus_ctx_def =
        ffts_plus_task_def->mutable_ffts_plus_ctx(static_cast<int>(old_ctxid));
    FFTS_CHECK_NOTNULL(ffts_plus_ctx_def);
    FFTS_LOGD("FFTSPlusOpsKernelBuilder GenNewSubGraphTaskDef node name:%s, node type:%s, old ctxid:%u, new ctxid:%zu.",
              node.GetName().c_str(), node.GetType().c_str(), old_ctxid, i);
    ret = ConvertOldCtxInfoToNewCtx(ffts_plus_ctx_def, ffts_plus_ctx_def_new, old_new_map);
    if (ret != SUCCESS) {
      return ret;
    }
  }
  return SUCCESS;
}

Status FFTSPlusOpsKernelBuilder::GenerateExtTask(const ge::Node &node, ge::RunContext &context,
                                                 std::vector<domi::TaskDef> &task_defs) {
  if (!node.GetOpDesc()->HasAttr(ATTR_NAME_ALIAS_ENGINE_NAME)) {
    return SUCCESS;
  }
  bool reg_flag = false;
  FFTS_TIMECOST_START(GenerateOpExtTask);
  if (fe::GenerateOpExtTask(node, fe::CheckTilingSink(node), task_defs, reg_flag) != SUCCESS) {
    REPORT_FFTS_ERROR("[FFTSPlusOpsKernelBuilder][GenerateExtTask] Op [%s][%s] failed to generate extra task.",
                    node.GetNamePtr(), node.GetTypePtr());
    return FAILED;
  }
  if (reg_flag) {
    FFTS_TIMECOST_END(GenerateOpExtTask, "GenerateOpExtTask.");
    return SUCCESS;
  }
  auto func = fe::OpExtGenTaskRegistry::GetInstance().FindRegisterFunc(node.GetType());
  if (func == nullptr) {
    return SUCCESS;
  }
  FFTS_LOGD("Node [%s] of type [%s] generated an extra task.", node.GetNamePtr(), node.GetTypePtr());
  auto ret = func(node, context, task_defs);
  if (ret != ge::SUCCESS) {
    REPORT_FFTS_ERROR("[FFTSPlusOpsKernelBuilder][GenerateExtTask] Op [%s][%s] failed to generate extra task.",
                      node.GetNamePtr(), node.GetTypePtr());
    return FAILED;
  }
  return SUCCESS;
}

Status FFTSPlusOpsKernelBuilder::GenerateTask(const ge::Node &node, ge::RunContext &context,
                                              std::vector<domi::TaskDef> &task_defs) {
  FFTS_LOGD("FFTSPlusOpsKernelBuilder GenerateTask node name:%s, node type:%s",
            node.GetName().c_str(), node.GetType().c_str());
  ge::OpDescPtr op_desc = node.GetOpDesc();
  Status status;
  uint64_t ready_context_num = 0;
  uint64_t total_context_number = 0;
  domi::TaskDef task_def;
  std::vector<ge::NodePtr> sub_graph_nodes;
  ge::ComputeGraphPtr sgt_graph = nullptr;
  TheadTaskBuilderPtr  base_mode_ptr = nullptr;
  if (op_desc->HasAttr(ATTR_NAME_ALIAS_ENGINE_NAME)) {
    FFTS_LOGI("[GenerateTask][MIXL2Task] GenerateTask for node [%s].", node.GetName().c_str());
    (void)ge::AttrUtils::SetStr(op_desc, "_ge_attr_lowering_func", "ffts_mix_l2_lower_func");
    (void)ge::AttrUtils::SetStr(op_desc, "_ge_attr_calculate_func", "ffts_mix_l2_calc_func");
    ge::Node &temp_node = const_cast<ge::Node&>(node);
    ge::NodePtr node_ptr = temp_node.shared_from_this();
    sub_graph_nodes.emplace_back(node_ptr);
    FFTS_MAKE_SHARED(sgt_graph = std::make_shared<ge::ComputeGraph>("MIX_L2"), return FAILED);
    FFTS_CHECK_NOTNULL(sgt_graph);
    Mixl2ModeTaskBuilderPtr mixl2_mode_task_builder_ptr;
    FFTS_MAKE_SHARED(mixl2_mode_task_builder_ptr = std::make_shared<Mixl2ModeTaskBuilder>(), return FAILED);
    base_mode_ptr = mixl2_mode_task_builder_ptr;
  } else {
    base_mode_ptr = GetNormBuilder(node, sgt_graph, task_def, ready_context_num, total_context_number);
    FFTS_CHECK_NOTNULL(sgt_graph);
    status = GenSerialDependency(sgt_graph);
    if (status != SUCCESS) {
      return status;
    }
  }
  FFTS_CHECK_NOTNULL(base_mode_ptr);
  (void)base_mode_ptr->Initialize();

  RunContextPtr contextptr = nullptr;
  FFTS_MAKE_SHARED(contextptr = std::make_shared<ge::RunContext>(context), return FAILED);
  FFTS_CHECK_NOTNULL(contextptr);
  bool ret = sgt_graph->SetExtAttr(kRuntimeContentx, contextptr);
  FFTS_LOGD("FFTSPlusOpsKernelBuilder GenerateTask status is %d for setting addr attr in sgt_graph[%s].", ret,
            sgt_graph->GetName().c_str());

  std::vector<ge::NodePtr> memset_nodes;
  status = base_mode_ptr->GenFftsPlusContextId(*sgt_graph, sub_graph_nodes, ready_context_num, total_context_number,
                                               memset_nodes);
  if (status != SUCCESS) {
    return status;
  }
  FFTS_LOGD("FFTSPlusOpsKernelBuilder GenerateTask node name: %s, node type: %s, ready num: %lu, total number: %lu.",
            node.GetName().c_str(), node.GetType().c_str(), ready_context_num, total_context_number);
  status = base_mode_ptr->GenSubGraphTaskDef(memset_nodes, sub_graph_nodes, task_def);
  if (status != SUCCESS) {
    FFTS_LOGD("GenSubGraphTaskDef unsuccess, node name:%s, node type:%s, errno:%u.",
              node.GetName().c_str(), node.GetType().c_str(), status);
    return status;
  }
  FFTS_LOGD("FFTSPlusOpsKernelBuilder After GenSubGraphTaskDef node name:%s, node type:%s, readynum:%lu, "
      "totalnumber:%lu.", node.GetName().c_str(), node.GetType().c_str(), ready_context_num, total_context_number);

  status = TimelineLayoutOptimize(ready_context_num, node, task_def);
  if (status != SUCCESS) {
    FFTS_LOGD("TimelineLayoutOptimize failed, node name: %s, node type: %s, error code: %u.",
              node.GetName().c_str(), node.GetType().c_str(), status);
    return status;
  }
  domi::TaskDef task_def_new;
  domi::TaskDef *task_def_real = &task_def_new;
  /*
   * Remove null label ctxid which neither pre nor succlist, and rebuild relation map between old ctxid and new ctxid
   */
  std::unordered_map<uint32_t, uint32_t> new_old_ctx_id_map;
  std::unordered_map<uint32_t, uint32_t> old_new_ctx_id_map;
  Status res = ReBuildCtxIdsRelation(task_def, new_old_ctx_id_map, old_new_ctx_id_map);
  if (res == SUCCESS) {
    res = GenNewSubGraphTaskDef(node, task_def, task_def_new, new_old_ctx_id_map, old_new_ctx_id_map);
    if (res != SUCCESS) {
      task_def_real = &task_def;
    }
  } else {
    task_def_real = &task_def;
  }

  if (GenSubGraphSqeDef(*task_def_real, ready_context_num, *sgt_graph) != SUCCESS) {
    FFTS_LOGD("GenSubGraphSqeDef unsuccess, node name:%s, node type:%s, errno:%u.",
              node.GetName().c_str(), node.GetType().c_str(), status);
    return FAILED;
  }
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def_real->mutable_ffts_plus_task();
  task_def_real->set_type(RT_MODEL_TASK_FFTS_PLUS_TASK);
  ffts_plus_task_def->set_op_index(op_desc->GetId());
  task_def_real->set_stream_id(op_desc->GetStreamId());
  task_defs.push_back(*task_def_real);
  FFTS_LOGD("FFTSPlusOpsKernelBuilder GenerateTask node name:%s, type:%s, id:%ld, readynum:%lu, stream:%ld success.",
      node.GetName().c_str(), node.GetType().c_str(), op_desc->GetId(), ready_context_num, op_desc->GetStreamId());
  if (GenerateExtTask(node, context, task_defs) != SUCCESS) {
    return FAILED;
  }
  return SUCCESS;
}

Status FFTSPlusOpsKernelBuilder::GenPersistentContext(const ge::Node &node, uint64_t &ready_context_num,
                                                      uint64_t &total_context_number, domi::TaskDef &task_def) const {
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  CachePersistTaskBuilder cp;
  if (cp.GenContextDef(node, ffts_plus_task_def) == SUCCESS) {
    total_context_number++;
    ready_context_num++;
  }
  return SUCCESS;
}

TheadTaskBuilderPtr FFTSPlusOpsKernelBuilder::GetFftsPlusMode(const ge::Node &part_node,
                                                              const ge::ComputeGraph &sgt_graph) {
  ModeType mode_type = ModeType::MANUAL_MODE_TYPE;

  for (const auto &node : sgt_graph.GetDirectNode()) {
    if (node == nullptr) {
      continue;
    }
    if (fe::UnknownShapeUtils::IsUnknownShapeOp(*node->GetOpDesc())) {
      // Dynamic shape, auto thread
      mode_type = ModeType::DYNAMIC_MODE_TYPE;
      break;
    } else {
      ThreadSliceMapPtr slice_info_ptr = nullptr;
      slice_info_ptr = node->GetOpDesc()->TryGetExtAttr(kAttrSgtStructInfo, slice_info_ptr);
      if (slice_info_ptr != nullptr && slice_info_ptr->thread_mode == kAutoMode) {
        mode_type = ModeType::AUTO_MODE_TYPE;
        break;
      }
    }
  }
  if (mode_type != ModeType::DYNAMIC_MODE_TYPE) {
    FFTS_LOGD("Partitioncall[%s]'s graph [%s] mode type is not dynamic mode.", part_node.GetName().c_str(),
              sgt_graph.GetName().c_str());
    (void)ge::AttrUtils::SetStr(part_node.GetOpDesc(), ge::kAttrLowingFunc, ge::kFFTSStaticGraphLowerFunc);
  }
  switch (mode_type) {
    case ModeType::MANUAL_MODE_TYPE: {
      ManualTheadTaskBuilderPtr manual_thread_task_builder_ptr;
      FFTS_MAKE_SHARED(manual_thread_task_builder_ptr = std::make_shared<ManualTheadTaskBuilder>(), return nullptr);
      return manual_thread_task_builder_ptr;
    }
    case ModeType::AUTO_MODE_TYPE: {
      AutoTheadTaskBuilderPtr auto_thread_task_builder_ptr;
      FFTS_MAKE_SHARED(auto_thread_task_builder_ptr = std::make_shared<AutoTheadTaskBuilder>(), return nullptr);
      return auto_thread_task_builder_ptr;
    }
    case ModeType::DYNAMIC_MODE_TYPE: {
      AutoTheadTaskBuilderPtr auto_thread_task_builder_ptr;
      FFTS_MAKE_SHARED(auto_thread_task_builder_ptr = std::make_shared<AutoTheadTaskBuilder>(), return nullptr);
      (void)ge::AttrUtils::SetStr(part_node.GetOpDesc(), ge::kAttrLowingFunc, ge::kFFTSGraphLowerFunc);
      auto_thread_task_builder_ptr->SetModeType(ModeType::DYNAMIC_MODE_TYPE);
      return auto_thread_task_builder_ptr;
    }

    default:
      return nullptr;
  }
}

std::string FFTSPlusOpsKernelBuilder::ConvSqeTypeToStr(uint32_t context_type) const {
  auto iter = kCtxTypeStrMap.find(static_cast<rtFftsPlusContextType_t>(context_type));
  if (iter != kCtxTypeStrMap.end()) {
    return iter->second;
  }
  return "unknown";
}

Status FFTSPlusOpsKernelBuilder::GenSubGraphSqeDef(domi::TaskDef &task_def, const uint64_t &ready_context_num,
                                                   const ge::ComputeGraph &sgt_graph) const {
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  FFTS_CHECK_NOTNULL(ffts_plus_task_def);
  uint64_t gen_ctx_size = ffts_plus_task_def->ffts_plus_ctx_size();
  FFTS_LOGD("This SGT subgraph named %s has %lu tasks.", sgt_graph.GetName().c_str(), gen_ctx_size);

  domi::FftsPlusSqeDef* ffts_plus_sqe = ffts_plus_task_def->mutable_ffts_plus_sqe();
  FFTS_CHECK_NOTNULL(ffts_plus_sqe);
  for (size_t i = 0; i < gen_ctx_size; i++) {
    FFTS_LOGD("Gen subGraph [%s] sqe def, context_id: %zu, context: %s.",
        ConvSqeTypeToStr(ffts_plus_task_def->mutable_ffts_plus_ctx(static_cast<int>(i))->context_type()).c_str(),
        i, ffts_plus_task_def->mutable_ffts_plus_ctx(static_cast<int>(i))->DebugString().c_str());
  }

  uint64_t preload_context_num;
  if (ready_context_num > max_preload_context_num) {
    preload_context_num = max_preload_context_num;
  } else {
    preload_context_num = ready_context_num;
  }

  ffts_plus_sqe->set_ready_context_num(ready_context_num);
  ffts_plus_sqe->set_preload_context_num(preload_context_num);
  ffts_plus_sqe->set_total_context_num(gen_ctx_size);
  ffts_plus_sqe->set_prefetch_ost_num(Configuration::Instance().GetPrefetchOstNum());
  ffts_plus_sqe->set_cmaint_ost_num(Configuration::Instance().GetCmaintOstNum());
  ffts_plus_sqe->set_aic_prefetch_lower(Configuration::Instance().GetAicPrefetchLower());
  ffts_plus_sqe->set_aic_prefetch_upper(Configuration::Instance().GetAicPrefetchUpper());
  ffts_plus_sqe->set_aiv_prefetch_lower(Configuration::Instance().GetAivPrefetchLower());
  ffts_plus_sqe->set_aiv_prefetch_upper(Configuration::Instance().GetAivPrefetchUpper());
  ffts_plus_sqe->set_data_split_unit(Configuration::Instance().GetDataSplitUnit());
  return SUCCESS;
}

const std::unordered_set<std::string> FFTS_NO_NEED_GEN_TASK_OP_TYPE = {"Data", "NetOutput", "Variable", "Const",
                                                                  "Constant", "PhonyConcat"};
const std::unordered_set<std::string> FFTS_CONTROL_OP_V2_TYPE = {"If", "While", "Case"};

bool FFTSPlusOpsKernelBuilder::IsNoCtx(const ge::NodePtr &node) const {
  ge::OpDescPtr op_desc = node->GetOpDesc();
  if (FFTS_NO_NEED_GEN_TASK_OP_TYPE.count(op_desc->GetType()) != 0) {
    return true;
  }
  if (FFTS_CONTROL_OP_V2_TYPE.count(op_desc->GetType()) != 0) {
    return true;
  }
  bool no_task = false;
  (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_NOTASK, no_task);
  if (no_task) {
    return true;
  }
  return false;
}

Status FFTSPlusOpsKernelBuilder::GenSerialDependency(const ge::ComputeGraphPtr &sub_graph) const {
  uint64_t sub_stream_id;
  map<uint64_t, ge::NodePtr> dependencies;
  for (auto &node : sub_graph->GetDirectNode()) {
    FFTS_CHECK_NOTNULL(node);
    if (!ge::AttrUtils::GetInt(node->GetOpDesc(), ge::ATTR_NAME_SUB_STREAM_ID, sub_stream_id)) {
      continue;
    }

    if (IsNoCtx(node)) {
      continue;
    }
    auto iter = dependencies.find(sub_stream_id);
    if (iter == dependencies.end()) {
      dependencies[sub_stream_id] = node;
      continue;
    }

    if (!IsNoEdge(iter->second, node)) {
      dependencies[sub_stream_id] = node;
      continue;
    }
    // set post node
    std::shared_ptr<std::vector<ge::NodePtr>> non_edge_succlist  = nullptr;
    FFTS_MAKE_SHARED(non_edge_succlist = std::make_shared<std::vector<ge::NodePtr>>(), return FAILED);
    if (non_edge_succlist == nullptr) {
      return FAILED;
    }
    non_edge_succlist->emplace_back(node);
    FFTS_CHECK_NOTNULL(iter->second);
    iter->second->GetOpDesc()->SetExtAttr(kNonEdgeSuccList, non_edge_succlist);

    // set pre node non_edge_succlist
    std::shared_ptr<std::vector<ge::NodePtr>> non_edge_pre_node  = nullptr;
    FFTS_MAKE_SHARED(non_edge_pre_node = std::make_shared<std::vector<ge::NodePtr>>(), return FAILED);
    if (non_edge_pre_node == nullptr) {
      return FAILED;
    }
    non_edge_pre_node->emplace_back(iter->second);
    node->GetOpDesc()->SetExtAttr(kNonEdgePreNodes, non_edge_pre_node);

    dependencies[sub_stream_id] = node;
  }
  return SUCCESS;
}
}  // namespace ffts
