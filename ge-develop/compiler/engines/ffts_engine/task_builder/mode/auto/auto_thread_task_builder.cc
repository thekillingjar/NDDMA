/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "auto_thread_task_builder.h"
#include "common/aicore_util_constants.h"
#include "inc/ffts_utils.h"
#include "common/aicore_util_attr_define.h"
namespace ffts {
AutoTheadTaskBuilder::AutoTheadTaskBuilder() {
  mode_type_ = ModeType::AUTO_MODE_TYPE;
}
AutoTheadTaskBuilder::~AutoTheadTaskBuilder() {}


Status AutoTheadTaskBuilder::Initialize() {
  FFTS_MAKE_SHARED(aic_aiv_dynamic_task_builder_ptr_ = std::make_shared<AICAIVDynamicTaskBuilder>(), return FAILED);
  FFTS_MAKE_SHARED(mix_aic_aiv_dynamic_task_builder_ptr_ = std::make_shared<MixAICAIVDynamicTaskBuilder>(),
      return FAILED);
  FFTS_MAKE_SHARED(aic_aiv_auto_task_builder_ptr_ = std::make_shared<AICAIVAutoTaskBuilder>(), return FAILED);
  FFTS_MAKE_SHARED(mix_aic_aiv_auto_task_builder_ptr_ = std::make_shared<MixAICAIVAutoTaskBuilder>(), return FAILED);
  FFTS_MAKE_SHARED(aicpu_auto_task_builder_ptr_ = std::make_shared<AicpuAutoTaskBuilder>(), return FAILED);
  FFTS_MAKE_SHARED(runtime_ops_auto_task_builder_ptr_ =
                   std::make_shared<RuntimeOpsAutoTaskBuilder>(), return FAILED);
  return SUCCESS;
}

void AutoTheadTaskBuilder::SetCtxIdList(ge::NodePtr &node, uint32_t &context_id, const uint32_t &window_size) const {
  // for node self
  vector<uint32_t> context_id_list;
  for (size_t i = 0; i < window_size;  i++) {
    context_id_list.push_back(context_id++);
  }
  (void)ge::AttrUtils::SetListInt(node->GetOpDesc(), kAutoCtxIdList, context_id_list);
}

void AutoTheadTaskBuilder::GetStartFlag(const ge::NodePtr &node, bool &conn_start) const {
  for (auto up_node : node->GetInAllNodes()) {
    if (up_node == nullptr) {
      continue;
    }
    ge::OpDescPtr up_op_desc = up_node->GetOpDesc();
    if (up_op_desc->GetType() == CONSTANT || up_op_desc->GetType() == CONSTANTOP) {
      continue;
    }
    // may have ge_local_op in graph
    if (IsNoCtx(up_node) && (!IsSubGraphData(up_op_desc))) {
      GetStartFlag(up_node, conn_start);
      if (!conn_start) {
        break;
      }
      FFTS_LOGD("Node [%s]'s upstream node [%s] has no context.", node->GetName().c_str(), up_node->GetName().c_str());
    } else if (!IsSubGraphData(up_op_desc)) {
      FFTS_LOGD("Node [%s]'s upstream node [%s] has no data.", node->GetName().c_str(), up_node->GetName().c_str());
      conn_start = false;
      break;
    }
  }
  FFTS_LOGD("Node[%s]'s start flag is %d.", node->GetName().c_str(), conn_start);
  return;
}

void AutoTheadTaskBuilder::GetEndFlag(const ge::NodePtr &node, bool &conn_end) const {
  for (auto next_node : node->GetOutAllNodes()) {
    if (next_node == nullptr) {
      continue;
    }
    ge::OpDescPtr next_op_desc = next_node->GetOpDesc();
    if (IsNoCtx(next_node) && (!IsSubGraphNetOutput(next_op_desc))) {
      FFTS_LOGD("Node [%s]'s next node [%s] has no context.", node->GetName().c_str(), next_node->GetName().c_str());
      GetEndFlag(next_node, conn_end);
      if (conn_end) {
        break;
      }
    } else if (IsSubGraphNetOutput(next_op_desc)) {
      FFTS_LOGD("Node [%s]'s next node is the end node [%s].", node->GetName().c_str(), next_node->GetName().c_str());
      conn_end = true;
      break;
    }
  }
  FFTS_LOGD("Node[%s]'s end flag is %d.", node->GetName().c_str(), conn_end);
  return;
}

void AutoTheadTaskBuilder::SetAllAttrInFirstNode(ge::ComputeGraph &sgt_graph,
                                                 const vector<uint32_t> &at_start_ctx_id_list,
                                                 const uint32_t &out_label_ctx_id) const {
  // Record in_label at_start at_end out_label and nodeself in first node.
  vector<uint32_t> all_ctx_id_list;
  all_ctx_id_list.push_back(0);
  for (size_t i = 0; i < at_start_ctx_id_list.size(); i++) {
    all_ctx_id_list.push_back(at_start_ctx_id_list[i]);
  }
  all_ctx_id_list.push_back(out_label_ctx_id);
  ge::AttrUtils::SetListInt(&sgt_graph, "_all_ctx_id_list", all_ctx_id_list);
}

void AutoTheadTaskBuilder::SetAttrExceptCtxIdList(ge::ComputeGraph &sgt_graph,
                                                  const vector<uint32_t> &at_start_ctx_id_list,
                                                  const vector<uint32_t> &at_end_ctx_id_list, int &count_node_conn_end,
                                                  const uint32_t &out_label_ctx_id,
                                                  std::vector<ge::NodePtr> &sub_graph_nodes,
                                                  uint64_t &total_context_number) const {
  uint32_t at_end_pre_cnt = count_node_conn_end;
  bool first_node_conn_start = true;
  // set context id to node
  for (auto node : sgt_graph.GetDirectNode()) {
    ge::OpDescPtr op_desc = node->GetOpDesc();
    if (IsNoCtx(node)) {
      continue;
    }

    // deal nodes connect at_start or at_end
    bool conn_start = true;
    bool conn_end = false;
    GetStartFlag(node, conn_start);
    GetEndFlag(node, conn_end);

    FFTS_LOGD("Dealing with node: %s.", op_desc->GetName().c_str());

    if (conn_start) {
      (void)ge::AttrUtils::SetStr(&sgt_graph, kFftsFirstOpName, op_desc->GetName());
      // at_start connext node which is first node has this attribute
      FFTS_LOGD("Start node_name: %s, first_node_conn_start: %d (1).",
                op_desc->GetName().c_str(), first_node_conn_start);
      if (first_node_conn_start) {
        ge::AttrUtils::SetInt(op_desc, kAutoInlabelCtxId, total_context_number);
        first_node_conn_start = false;

        // Record in_label at_start at_end out_label and nodeself in first node.
        SetAllAttrInFirstNode(sgt_graph, at_start_ctx_id_list, out_label_ctx_id);
      }
      ge::AttrUtils::SetListInt(op_desc, kAutoAtStartCtxIdList, at_start_ctx_id_list);
    }
    if (conn_end) {
      // node connect at_end which is last node has this attribute
      count_node_conn_end--;
      FFTS_LOGD("End node_name: %s, count_node_conn_end: %d (0).", op_desc->GetName().c_str(), count_node_conn_end);
      if (count_node_conn_end == 0) {
        ge::AttrUtils::SetInt(op_desc, kAutoOutlabelCtxId, out_label_ctx_id);
        ge::AttrUtils::SetInt(op_desc, kAutoAtEndPreCnt, at_end_pre_cnt);
      }
      ge::AttrUtils::SetListInt(op_desc, kAutoAtEndCtxIdList, at_end_ctx_id_list);
    }
    sub_graph_nodes.push_back(node);
  }
}

Status AutoTheadTaskBuilder::GenFftsPlusContextId(ge::ComputeGraph &sgt_graph,
                                                  std::vector<ge::NodePtr> &sub_graph_nodes,
                                                  uint64_t &ready_context_num, uint64_t &total_context_number,
                                                  std::vector<ge::NodePtr> &memset_nodes) {
  uint32_t contextId = total_context_number + 1;
  std::vector<ge::NodePtr> pre_sub_graph_nodes;
  if (GenFftsPlusContextIdWithMemSet(pre_sub_graph_nodes, memset_nodes, sgt_graph) != SUCCESS) {
    return FAILED;
  }
  uint32_t window_size = 0xFFFFF;
  for (const auto &node : sgt_graph.GetDirectNode()) {
    ThreadSliceMapPtr slice_info_ptr = nullptr;
    FFTS_CHECK_NOTNULL(node);
    slice_info_ptr = node->GetOpDesc()->TryGetExtAttr(kAttrSgtStructInfo, slice_info_ptr);
    if (slice_info_ptr != nullptr && slice_info_ptr->parallel_window_size > 0) {
      window_size = slice_info_ptr->parallel_window_size;
      break;
    }
  }
  window_size = window_size > 0xFFFF ? kDefaultWindowSize : window_size;
  FFTS_LOGD("Auto mode, current window size: %u.", window_size);
  ready_context_num = 1;
  // auto theading
  vector<uint32_t> at_start_ctx_id_list;
  int count_node_conn_end = 0;

  // generate at_start context id
  for (size_t i = 0; i < window_size; i++) {
    at_start_ctx_id_list.push_back(contextId++);
  }

  FFTS_LOGD("auto threading at start: ctx id list size %zu.", at_start_ctx_id_list.size());

  for (auto &node : pre_sub_graph_nodes) {
    if (node == nullptr) {
      continue;
    }
    if (!node->GetOpDesc()->HasAttr(kAutoCtxIdList)) {
      SetCtxIdList(node, contextId, window_size);
      bool is_only_add_contextid = false;
      (void)ge::AttrUtils::GetBool(node->GetOpDesc(), kOnlyAddContext, is_only_add_contextid);
      if (!is_only_add_contextid) {
        sub_graph_nodes.push_back(node);
      }
    }
  }
  // generate node context id
  for (auto node : sgt_graph.GetDirectNode()) {
    if (node == nullptr) {
      continue;
    }
    if (IsNoCtx(node)) {
      continue;
    }
    SetCtxIdList(node, contextId, window_size);
    // node->at-end when node has output from out sgt_graph
    bool conn_end = false;
    GetEndFlag(node, conn_end);
    if (conn_end) {
      // generate at_end(context id, pre_cnt) and label(context id).
      count_node_conn_end++;
    }
  }

  // generate at_end and output_label context id
  vector<uint32_t> at_end_ctx_id_list;
  for (size_t i = 0; i < window_size; i++) {
    at_end_ctx_id_list.push_back(contextId++);
  }
  uint32_t out_label_ctx_id = contextId++;
  FFTS_LOGD("auto threading at end, pre cnt: %d, out_label_ctx_id: %u.", count_node_conn_end, out_label_ctx_id);

  SetAttrExceptCtxIdList(sgt_graph, at_start_ctx_id_list, at_end_ctx_id_list, count_node_conn_end, out_label_ctx_id,
                         sub_graph_nodes, total_context_number);
  total_context_number = contextId;
  return SUCCESS;
}

Status AutoTheadTaskBuilder::GenInLabelAtStartCtxDef(const ge::NodePtr &node,
                                                     domi::FftsPlusTaskDef *ffts_plus_task_def) const {
  ge::OpDescPtr op_desc = node->GetOpDesc();

  ThreadSliceMapPtr slice_info_ptr = nullptr;
  slice_info_ptr = op_desc->TryGetExtAttr(kAttrSgtStructInfo, slice_info_ptr);

  FFTS_LOGD("GenInLabelAtStartCtxDef node's name is: %s.", node->GetName().c_str());
  FFTS_CHECK_NOTNULL(slice_info_ptr);
  if (slice_info_ptr->thread_mode == kManualMode) {
    FFTS_LOGD("Manual node [%s] does not generate inlabel and at_start.", node->GetOpDesc()->GetName().c_str());
    return SUCCESS;
  }

  // check
  uint32_t in_label_ctx_id;
  vector<uint32_t> at_start_ctx_id_list;
  (void)ge::AttrUtils::GetListInt(op_desc, kAutoAtStartCtxIdList, at_start_ctx_id_list);
  if (!ge::AttrUtils::GetInt(op_desc, kAutoInlabelCtxId, in_label_ctx_id) || (at_start_ctx_id_list.empty())) {
    REPORT_FFTS_ERROR("[GenerateTask][GenSubGraphTaskDef][GenInLabelAtStartCtxDef] Node %s has no in_label or at_start"
                      " context id.", node->GetName().c_str());
    return FAILED;
  }
  FFTS_CHECK_NOTNULL(ffts_plus_task_def);
  // generate in_label context def
  domi::FftsPlusCtxDef *ffts_plus_ctx_def = ffts_plus_task_def->add_ffts_plus_ctx();
  FFTS_CHECK_NOTNULL(ffts_plus_ctx_def);
  ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_LABEL);
  domi::FftsPlusLabelCtxDef *in_label_ctx_def = ffts_plus_ctx_def->mutable_label_ctx();
  FFTS_CHECK_NOTNULL(in_label_ctx_def);
  in_label_ctx_def->set_pred_cnt(0);
  in_label_ctx_def->set_pred_cnt_init(0);
  in_label_ctx_def->set_successor_num(at_start_ctx_id_list.size());
  for (size_t i = 0; i < at_start_ctx_id_list.size(); i++) {
    in_label_ctx_def->add_successor_list(i + 1);
  }

  // generate at_start_list context def
  for (size_t i = 0; i < at_start_ctx_id_list.size(); i++) {
    ffts_plus_ctx_def = ffts_plus_task_def->add_ffts_plus_ctx();
    ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_AT_START);
    domi::FftsPlusAtStartCtxDef *at_start_ctx_def = ffts_plus_ctx_def->mutable_at_start_ctx();
    FFTS_CHECK_NOTNULL(at_start_ctx_def);
    at_start_ctx_def->set_aten(1);
    at_start_ctx_def->set_pred_cnt(1);
    at_start_ctx_def->set_pred_cnt_init(1);
    at_start_ctx_def->set_thread_id(i);
    at_start_ctx_def->set_thread_id_init(i);
    at_start_ctx_def->set_thread_dim(slice_info_ptr->slice_instance_num);
    at_start_ctx_def->set_thread_window_size(slice_info_ptr->parallel_window_size);
    at_start_ctx_def->set_successor_num(0);
  }
  return SUCCESS;
}

Status AutoTheadTaskBuilder::GenOutLabelAtEndCtxDef(const ge::NodePtr &node,
                                                    domi::FftsPlusTaskDef *ffts_plus_task_def) const {
  ge::OpDescPtr op_desc = node->GetOpDesc();
  vector<uint32_t> at_end_ctx_id_list;
  (void)ge::AttrUtils::GetListInt(op_desc, kAutoAtEndCtxIdList, at_end_ctx_id_list);
  uint32_t at_end_pre_cnt = 0;
  (void)ge::AttrUtils::GetInt(op_desc, kAutoAtEndPreCnt, at_end_pre_cnt);
  uint32_t out_label_ctx_id = 0;
  (void)ge::AttrUtils::GetInt(op_desc, kAutoOutlabelCtxId, out_label_ctx_id);

  if (at_end_ctx_id_list.empty() || (at_end_pre_cnt == 0) || (out_label_ctx_id == 0)) {
    REPORT_FFTS_ERROR("[GenerateTask][GenSubGraphTaskDef][GenOutLabelAtEndCtxDef] Node %s has no out_label, at_end "
                      "context id and at_end_pre_cnt.", node->GetName().c_str());
    return FAILED;
  }
  FFTS_CHECK_NOTNULL(ffts_plus_task_def);
  // generate at_end context def
  for (size_t i = 0; i < at_end_ctx_id_list.size(); i++) {
    domi::FftsPlusCtxDef *ffts_plus_ctx_def = ffts_plus_task_def->add_ffts_plus_ctx();
    FFTS_CHECK_NOTNULL(ffts_plus_ctx_def);
    ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_AT_END);
    domi::FftsPlusAtEndCtxDef *at_end_ctx_def = ffts_plus_ctx_def->mutable_at_end_ctx();
    FFTS_CHECK_NOTNULL(at_end_ctx_def);
    at_end_ctx_def->set_aten(1);  // auto thread
    at_end_ctx_def->set_pred_cnt(at_end_pre_cnt);
    at_end_ctx_def->set_pred_cnt_init(at_end_pre_cnt);
    at_end_ctx_def->set_at_start_slot_num(1);
    at_end_ctx_def->add_succ_at_start_slot(i + 1);
    at_end_ctx_def->set_out_label_slot_num(1);
    at_end_ctx_def->add_succ_out_label_slot(out_label_ctx_id);
  }

  // generate out_label context def
  domi::FftsPlusCtxDef *ffts_plus_ctx_def = ffts_plus_task_def->add_ffts_plus_ctx();
  FFTS_CHECK_NOTNULL(ffts_plus_ctx_def);
  ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_LABEL);
  domi::FftsPlusLabelCtxDef *out_label_ctx_def = ffts_plus_ctx_def->mutable_label_ctx();
  ThreadSliceMapPtr slice_info_ptr = nullptr;
  slice_info_ptr = op_desc->TryGetExtAttr(kAttrSgtStructInfo, slice_info_ptr);
  FFTS_CHECK_NOTNULL(slice_info_ptr);
  out_label_ctx_def->set_pred_cnt(slice_info_ptr->slice_instance_num);
  out_label_ctx_def->set_pred_cnt_init(slice_info_ptr->slice_instance_num);
  out_label_ctx_def->set_successor_num(0);

  return SUCCESS;
}

Status AutoTheadTaskBuilder::FillSerialDependency(const ge::NodePtr &sub_node,
                                                  domi::FftsPlusTaskDef *ffts_plus_task_def,
                                                  const FFTSPlusTaskBuilderPtr &task_builder,
                                                  const vector<uint32_t> &context_id_list) const {
  auto op_desc = sub_node->GetOpDesc();
  std::shared_ptr<std::vector<ge::NodePtr>> suc_nodes = nullptr;
  suc_nodes = op_desc->TryGetExtAttr(kNonEdgeSuccList, suc_nodes);
  if (suc_nodes == nullptr || (*suc_nodes).size() == 0) {
    return SUCCESS;
  }
  for (const auto &suc_node : (*suc_nodes)) {
    FFTS_CHECK_NOTNULL(suc_node);
    auto suc_node_desc = suc_node->GetOpDesc();
    vector<uint32_t> suc_context_id_list;
    if (!ge::AttrUtils::GetListInt(suc_node_desc, kAutoCtxIdList, suc_context_id_list)) {
      REPORT_FFTS_ERROR("[GenSubTask][FillSerDep][Get] Node[name: %s, type: %s] does not have a ctx_id_list.",
                        suc_node_desc->GetName().c_str(), suc_node_desc->GetType().c_str());
      return FAILED;
    }

    if (context_id_list.size() != suc_context_id_list.size()) {
      REPORT_FFTS_ERROR("[GenSubTask][FillSerDep][Judge] Node[%s] and suc_node[%s] do not have the same ctx_id_list.",
                        sub_node->GetName().c_str(), suc_node->GetName().c_str());
      return FAILED;
    }
    FFTS_LOGD("The current node is %s, followed by %s.", sub_node->GetName().c_str(), suc_node->GetName().c_str());
    for (size_t i = 0; i < context_id_list.size(); ++i) {
      Status status = task_builder->UpdateSuccList(suc_context_id_list[i], context_id_list[i], ffts_plus_task_def, i,
                                                   true);
      if (status != SUCCESS) {
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

Status AutoTheadTaskBuilder::UpdateAtomicSuccList(const ge::NodePtr &node, const vector<uint32_t> &context_id_list,
    const FFTSPlusTaskBuilderPtr &task_builder, domi::FftsPlusTaskDef *ffts_plus_task_def) const {
  if (!node->GetOpDesc()->HasAttr(kAtomicCtxIdList)) {
    return SUCCESS;
  }
  vector<uint32_t> atomic_context_id_list;
  (void)ge::AttrUtils::GetListInt(node->GetOpDesc(), kAtomicCtxIdList, atomic_context_id_list);
  if (atomic_context_id_list.size() != context_id_list.size()) {
    FFTS_LOGE("Atomic context size [%zu] does not match original node context size [%zu].", atomic_context_id_list.size(),
              context_id_list.size());
    return FAILED;
  }
  size_t window_size = context_id_list.size();
  for (size_t i = 0; i < window_size; ++i) {
    FFTS_LOGD("Update atomic id [%u] successor id to [%u].", atomic_context_id_list[i], context_id_list[i]);
    Status status = task_builder->UpdateSuccList(context_id_list[i], atomic_context_id_list[i], ffts_plus_task_def,
                                                 i, true);
    if (status != SUCCESS) {
      return FAILED;
    }
    status = task_builder->UpdatePreCnt(context_id_list[i], ffts_plus_task_def, 1);
    if (status != SUCCESS) {
      return FAILED;
    }
  }
  return SUCCESS;
}

Status AutoTheadTaskBuilder::AddSuccListInCtx(domi::FftsPlusTaskDef *ffts_plus_task_def,
                                              const FFTSPlusTaskBuilderPtr &task_builder,
                                              const vector<uint32_t> &context_id_list,
                                              const vector<uint32_t> &output_context_id_list,
                                              const ge::NodePtr &out_node) const {
  FFTS_CHECK_NOTNULL(task_builder);
  if (output_context_id_list.empty() || context_id_list.empty() ||
      output_context_id_list.size() != context_id_list.size()) {
    return FAILED;
  }
  FFTS_LOGD("Current node does not need to write_back; starting to add the first at_end[%u] to the first context's[%u] "
            "successor_list.", output_context_id_list[0], context_id_list[0]);
  std::vector<uint32_t> atomic_ctx_id_vec;
  (void)ge::AttrUtils::GetListInt(out_node->GetOpDesc(), kAtomicCtxIdList, atomic_ctx_id_vec);
  size_t window_size = output_context_id_list.size();
  for (size_t i = 0; i < window_size; i++) {
    Status status = task_builder->UpdateSuccList(output_context_id_list[i], context_id_list[i], ffts_plus_task_def,
                                                 i, true);
    if (status != SUCCESS) {
      return FAILED;
    }
    if (atomic_ctx_id_vec.empty()) {
      continue;
    }
    if (context_id_list.size() != atomic_ctx_id_vec.size()) {
      FFTS_LOGE("Node ctx size[%zu] not equal atomic size[%zu].", context_id_list.size(), atomic_ctx_id_vec.size());
      return FAILED;
    }
    FFTS_LOGD("Update context id[%u]'s successor(atomic) id[%u].", context_id_list[i], atomic_ctx_id_vec[i]);
    status = task_builder->UpdateSuccList(atomic_ctx_id_vec[i], context_id_list[i], ffts_plus_task_def,
                                          i, true);
    if (status != SUCCESS) {
      return FAILED;
    }
    status = task_builder->UpdatePreCnt(atomic_ctx_id_vec[i], ffts_plus_task_def, 1);
    if (status != SUCCESS) {
      return FAILED;
    }
  }
  return SUCCESS;
}

bool AutoTheadTaskBuilder::AddAtEndToWriteBackSuccList(const vector<uint32_t> &at_end_ctx_id_list,
                                                       const vector<uint32_t> &context_id_list) const {
  bool already_add = false;
  for (size_t i = 0; i < context_id_list.size(); i++) {
    domi::FftsPlusCtxDef* ffts_plus_ctx =
        ffts_plus_task_def_->mutable_ffts_plus_ctx(static_cast<int>(context_id_list[i]));
    switch (ffts_plus_ctx->context_type()) {
      case RT_CTX_TYPE_AIV:
        already_add = FFTSPlusTaskBuilder::add_at_end_to_write_back_succ_list(at_end_ctx_id_list[i],
            ffts_plus_ctx->mutable_aic_aiv_ctx(), ffts_plus_task_def_);
        break;
      case RT_CTX_TYPE_MIX_AIV:
        already_add = FFTSPlusTaskBuilder::add_at_end_to_write_back_succ_list(at_end_ctx_id_list[i],
            ffts_plus_ctx->mutable_mix_aic_aiv_ctx(), ffts_plus_task_def_);
        break;
      case RT_CTX_TYPE_SDMA:
        already_add = FFTSPlusTaskBuilder::add_at_end_to_write_back_succ_list(at_end_ctx_id_list[i],
            ffts_plus_ctx->mutable_sdma_ctx(), ffts_plus_task_def_);
        break;
      case RT_CTX_TYPE_NOTIFY_RECORD:
        already_add = FFTSPlusTaskBuilder::add_at_end_to_write_back_succ_list(at_end_ctx_id_list[i],
            ffts_plus_ctx->mutable_notify_ctx(), ffts_plus_task_def_);
        break;
      case RT_CTX_TYPE_WRITE_VALUE:
        already_add = FFTSPlusTaskBuilder::add_at_end_to_write_back_succ_list(at_end_ctx_id_list[i],
            ffts_plus_ctx->mutable_write_value_ctx(), ffts_plus_task_def_);
      default:
        break;
    }
  }
  return already_add;
}

Status AutoTheadTaskBuilder::FillContextSuccList(const ge::NodePtr &sub_node,
                                                 const FFTSPlusTaskBuilderPtr &task_builder,
                                                 const vector<uint32_t> &context_id_list,
                                                 const vector<uint32_t> &at_end_ctx_id_list,
                                                 bool &netoutput_flag) const {
  for (auto output_node : sub_node->GetOutAllNodes()) {
    if (netoutput_flag && output_node->GetType() == "NetOutput") {
      continue;
    }
    if (output_node->GetType() == "NetOutput") {
      netoutput_flag = true;
    } else if (IsNoCtx(output_node)) {
      FillContextSuccList(output_node, task_builder, context_id_list, at_end_ctx_id_list, netoutput_flag);
      continue;
    }
    vector<uint32_t> out_ctx_id_list;
    (void)ge::AttrUtils::GetListInt(output_node->GetOpDesc(), kAutoCtxIdList, out_ctx_id_list);
    FFTS_LOGD("The current output_node name is: %s.", output_node->GetName().c_str());
    bool flag_add_write_back = false;
    if (output_node->GetType() == "NetOutput") {
      out_ctx_id_list = at_end_ctx_id_list;
      // if context need to write back, add at_end_ctx_id to data_write_back's succ_list
      if (AddAtEndToWriteBackSuccList(at_end_ctx_id_list, context_id_list)) {
        flag_add_write_back = true;
        continue;
      }
    }
    if (flag_add_write_back) {
      continue;
    }
    if (AddSuccListInCtx(ffts_plus_task_def_, task_builder, context_id_list, out_ctx_id_list, output_node) != SUCCESS) {
      REPORT_FFTS_ERROR("[GenerateTask][AutoThreadTaskBuilder][AddSuccListInCtx] Failed to add succ_list in context.");
      return FAILED;
    }
  }
  return SUCCESS;
}

Status AutoTheadTaskBuilder::GenerateAtomicCtx(std::vector<ge::NodePtr> &atomic_nodes,
                                               domi::FftsPlusTaskDef *ffts_plus_task_def) const {
  if (atomic_nodes.empty()) {
    return SUCCESS;
  }
  FFTS_LOGD("Gen %zu atomic node ctx with mode:%d.", atomic_nodes.size(), static_cast<int32_t>(mode_type_));
  for (auto &atomic_node : atomic_nodes) {
    auto atomic_desc = atomic_node->GetOpDesc();
    FFTS_CHECK_NOTNULL(atomic_desc);
    int64_t block_dim = 1;
    (void)ge::AttrUtils::GetInt(atomic_desc, ge::TVM_ATTR_NAME_BLOCKDIM, block_dim);
    vector<uint32_t> context_id_list;
    if (!ge::AttrUtils::GetListInt(atomic_desc, kAutoCtxIdList, context_id_list) || context_id_list.empty()) {
      REPORT_FFTS_ERROR("Atomic node[%s] do not have context id.", atomic_desc->GetName().c_str());
      return FAILED;
    }
    vector<string> thread_kernel_name;
    (void)ge::AttrUtils::GetListStr(atomic_desc, fe::kThreadKernelName, thread_kernel_name);
    for (auto context_id : context_id_list) {
      domi::FftsPlusCtxDef *ffts_plus_ctx_def = ffts_plus_task_def->add_ffts_plus_ctx();
      ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_AIV);
      ffts_plus_ctx_def->set_op_index(atomic_desc->GetId());
      ffts_plus_ctx_def->set_op_type(domi::FftsPlusCtxDef::ATOMIC);
      ffts_plus_ctx_def->set_context_id(context_id);
      FFTS_LOGD("Fill atomic context [%u].", context_id);
      domi::FftsPlusAicAivCtxDef *aic_aiv_ctx_def = ffts_plus_ctx_def->mutable_aic_aiv_ctx();
      for (const auto &kernel_name : thread_kernel_name) {
        aic_aiv_ctx_def->add_kernel_name(kernel_name);
        FFTS_LOGD("Add kernel name: %s.", kernel_name.c_str());
      }
      aic_aiv_ctx_def->set_aten(1);
      aic_aiv_ctx_def->set_atm(1);
      aic_aiv_ctx_def->set_successor_num(0);
      if (mode_type_ == ModeType::DYNAMIC_MODE_TYPE) {
        aic_aiv_ctx_def->set_thread_dim(0);
        continue;
      }
      aic_aiv_ctx_def->set_thread_dim(1);
      aic_aiv_ctx_def->set_non_tail_block_dim(block_dim);
      aic_aiv_ctx_def->set_tail_block_dim(block_dim);
      std::vector<int64_t> work_spaces = atomic_desc->GetWorkspace();
      for (auto &work_space : work_spaces) {
        if (work_space != 0) {
          aic_aiv_ctx_def->add_task_addr(work_space);
        }
      }
      ffts_plus_task_def->set_addr_size(ffts_plus_task_def->addr_size() + aic_aiv_ctx_def->task_addr_size());
    }
  }
  return SUCCESS;
}

Status AutoTheadTaskBuilder::GenSubGraphTaskDef(std::vector<ge::NodePtr> &memset_nodes,
                                                std::vector<ge::NodePtr> &sub_graph_nodes,
                                                domi::TaskDef &task_def) {
  ffts_plus_task_def_ = task_def.mutable_ffts_plus_task();
  FFTS_CHECK_NOTNULL(ffts_plus_task_def_);

  if (sub_graph_nodes.empty()) {
    return FAILED;
  }

  Status status = GenInLabelAtStartCtxDef(sub_graph_nodes[0], ffts_plus_task_def_);
  if (status != SUCCESS) {
    return FAILED;
  }

  if (GenerateAtomicCtx(memset_nodes, ffts_plus_task_def_) != SUCCESS) {
    return FAILED;
  }
  for (auto &sub_node : sub_graph_nodes) {
    TaskBuilderType task_builder_type;
    status = GetNodeContextTypeByNode(sub_node, task_builder_type);
    if (status != SUCCESS) {
      return FAILED;
    }
    FFTSPlusTaskBuilderPtr task_builder = GetTaskBuilder(task_builder_type);
    FFTS_CHECK_NOTNULL(task_builder);
    status = task_builder->GenerateTaskDef(sub_node, ffts_plus_task_def_);
    if (status != SUCCESS) {
      return status;
    }
    ge::NodePtr atomic_node = nullptr;
    FFTS_CHECK_NOTNULL(sub_node);
    atomic_node = sub_node->GetOpDesc()->TryGetExtAttr(fe::ATTR_NAME_MEMSET_NODE, atomic_node);
    if (atomic_node != nullptr) {
      vector<uint32_t> atomic_context_id_list;
      (void)ge::AttrUtils::GetListInt(atomic_node->GetOpDesc(), kAutoCtxIdList, atomic_context_id_list);
      FFTS_LOGD("Set node [%s]'s atomic ctx id to num: %zu.", sub_node->GetName().c_str(), atomic_context_id_list.size());
      (void)ge::AttrUtils::SetListInt(sub_node->GetOpDesc(), kAtomicCtxIdList, atomic_context_id_list);
    }
  }

  status = GenOutLabelAtEndCtxDef(sub_graph_nodes[sub_graph_nodes.size() - 1], ffts_plus_task_def_);
  if (status != SUCCESS) {
    return status;
  }
  FFTS_LOGD("Current ffts_plus_task_def size is: %d.", ffts_plus_task_def_->ffts_plus_ctx_size());

  for (auto &sub_node : sub_graph_nodes) {
    ge::OpDescPtr op_desc = sub_node->GetOpDesc();
    vector<uint32_t> at_start_ctx_id_list;
    ge::AttrUtils::GetListInt(op_desc, kAutoAtStartCtxIdList, at_start_ctx_id_list);
    vector<uint32_t> context_id_list;
    ge::AttrUtils::GetListInt(op_desc, kAutoCtxIdList, context_id_list);
    vector<uint32_t> at_end_ctx_id_list;
    ge::AttrUtils::GetListInt(op_desc, kAutoAtEndCtxIdList, at_end_ctx_id_list);

    FFTS_LOGD("Current sub_node name is: %s, at_start_ctx_id_list size: %zu, context_id_list size: %zu, "
              "at_end_ctx_id_list size: %zu.", sub_node->GetName().c_str(), at_start_ctx_id_list.size(),
              context_id_list.size(), at_end_ctx_id_list.size());
    // fill at_start's succ_list
    FFTSPlusTaskBuilderPtr task_builder = nullptr;
    FFTS_MAKE_SHARED(task_builder = std::make_shared<FFTSPlusTaskBuilder>(), return FAILED);
    if (!at_start_ctx_id_list.empty()) {
      std::vector<uint32_t> atomic_ctx_id_vec;
      (void)ge::AttrUtils::GetListInt(sub_node->GetOpDesc(), kAtomicCtxIdList, atomic_ctx_id_vec);
      size_t window_size = at_start_ctx_id_list.size();
      for (size_t i = 0; i < window_size; i++) {
        FFTS_LOGD("index: %zu, context_id_list: %u, at_start_ctx_id_list: %u, ffts_plus_task_def_size: %d.", i,
                  context_id_list[i], at_start_ctx_id_list[i], ffts_plus_task_def_->ffts_plus_ctx_size());
        status = task_builder->UpdateSuccList(context_id_list[i], at_start_ctx_id_list[i], ffts_plus_task_def_,
                                              i, true);
        if (status != SUCCESS) {
          return status;
        }
        if (atomic_ctx_id_vec.empty()) {
          continue;
        }
        if (context_id_list.size() != atomic_ctx_id_vec.size()) {
          FFTS_LOGE("AtStart size [%zu] does not match atomic size [%zu].", at_start_ctx_id_list.size(),
                    atomic_ctx_id_vec.size());
          return FAILED;
        }
        FFTS_LOGD("Update the atomic successor [%u] of ctx[%u] at start.", at_start_ctx_id_list[i], atomic_ctx_id_vec[i]);
        status = task_builder->UpdateSuccList(atomic_ctx_id_vec[i], at_start_ctx_id_list[i], ffts_plus_task_def_,
                                              i, true);
        if (status != SUCCESS) {
          return status;
        }
        status = task_builder->UpdatePreCnt(atomic_ctx_id_vec[i], ffts_plus_task_def_, 1);
        if (status != SUCCESS) {
          return status;
        }
      }
    }
    if (GenerateDataTaskDef(sub_node, ffts_plus_task_def_, mode_type_) != SUCCESS) {
      return status;
    }

    // fill node context's succ_list
    bool netoutput_flag = false;
    status = FillContextSuccList(sub_node, task_builder, context_id_list, at_end_ctx_id_list, netoutput_flag);
    if (status != SUCCESS) {
      REPORT_FFTS_ERROR("[GenTask][GenSubTask][FillCtxSuccList] Node [%s] failed to fill in context succlist.",
                        op_desc->GetName().c_str());
      return status;
    }

    status = FillSerialDependency(sub_node, ffts_plus_task_def_, task_builder, context_id_list);
    if (status != SUCCESS) {
      REPORT_FFTS_ERROR("[GenTask][GenSubTask][FillSerDep] Node[%s] failed to fill in serial dependency.",
                        op_desc->GetName().c_str());
      return status;
    }

    status = UpdateAtomicSuccList(sub_node, context_id_list, task_builder, ffts_plus_task_def_);
    if (status != SUCCESS) {
      REPORT_FFTS_ERROR("[GenTask][GenSubTask][UpdAtomicSuc] Node [%s] failed to update the atomic succlist.",
                        op_desc->GetName().c_str());
      return status;
    }
  }

  return SUCCESS;
}
}  // namespace ffts
