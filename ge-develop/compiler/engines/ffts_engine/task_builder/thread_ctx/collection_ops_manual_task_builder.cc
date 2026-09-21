/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "collection_ops_manual_task_builder.h"
#include "inc/ffts_utils.h"
#include "common/string_utils.h"

namespace ffts {
namespace {
Status FillHCCLCtxDef(const domi::FftsPlusCtxDef &hccl_sub_task, domi::FftsPlusCtxDef *ffts_plus_ctx_def,
                      std::map<int64_t, int32_t> &subtask_input_cnt, int32_t &index,
                      vector<FftsPlusComCtx_t> &sub_ffts_plus_context)
{
  if (sub_ffts_plus_context.empty()) {
    return FAILED;
  }
  if (subtask_input_cnt.count(index) == 0) {
    subtask_input_cnt[index] = 0;
  }
  if (hccl_sub_task.context_type() == RT_CTX_TYPE_SDMA) {
    domi::FftsPlusSdmaCtxDef *sdma_ctx_def = ffts_plus_ctx_def->mutable_sdma_ctx();
    FFTS_CHECK_NOTNULL(sdma_ctx_def);
    sdma_ctx_def->set_aten(0);
    sdma_ctx_def->set_pred_cnt(subtask_input_cnt[index]);
    sdma_ctx_def->set_pred_cnt_init(subtask_input_cnt[index]);

    if (subtask_input_cnt[index] == 0) {
      sdma_ctx_def->set_pred_cnt(sub_ffts_plus_context[0].pred_cnt);
      sdma_ctx_def->set_pred_cnt_init(sub_ffts_plus_context[0].pred_cnt);
    }
  } else if (hccl_sub_task.context_type() == RT_CTX_TYPE_NOTIFY_WAIT ||
             hccl_sub_task.context_type() == RT_CTX_TYPE_NOTIFY_RECORD) {
    domi::FftsPlusNotifyCtxDef *notify_ctx_def = ffts_plus_ctx_def->mutable_notify_ctx();
    FFTS_CHECK_NOTNULL(notify_ctx_def);
    notify_ctx_def->set_aten(0);
    notify_ctx_def->set_pred_cnt(subtask_input_cnt[index]);
    notify_ctx_def->set_pred_cnt_init(subtask_input_cnt[index]);

    if (subtask_input_cnt[index] == 0) {
      notify_ctx_def->set_pred_cnt(sub_ffts_plus_context[0].pred_cnt);
      notify_ctx_def->set_pred_cnt_init(sub_ffts_plus_context[0].pred_cnt);
    }
  } else if (hccl_sub_task.context_type() == RT_CTX_TYPE_WRITE_VALUE) {
    domi::FftsPlusWriteValueCtxDef *writevalue_ctx_def = ffts_plus_ctx_def->mutable_write_value_ctx();
    FFTS_CHECK_NOTNULL(writevalue_ctx_def);
    writevalue_ctx_def->set_aten(0);
    writevalue_ctx_def->set_pred_cnt(subtask_input_cnt[index]);
    writevalue_ctx_def->set_pred_cnt_init(subtask_input_cnt[index]);

    if (subtask_input_cnt[index] == 0) {
      writevalue_ctx_def->set_pred_cnt(sub_ffts_plus_context[0].pred_cnt);
      writevalue_ctx_def->set_pred_cnt_init(sub_ffts_plus_context[0].pred_cnt);
    }
  }
  return SUCCESS;
}

Status ModifyHCCLInputNodeSuccList(const ge::NodePtr &node, std::map<int64_t, int32_t> &subtask_input_cnt,
                                   const std::vector<uint32_t> &ctx_id_list, int32_t &index)
{
  /* when hccl inner'predcnt equal zero, we need add the succlist of the subgraph input node */
  if (subtask_input_cnt[index] == 0) {
    for (const auto &up_node : node->GetInAllNodes()) {
      FFTS_CHECK_NOTNULL(up_node);
      ge::OpDescPtr up_op_desc = up_node->GetOpDesc();
      if (kHCCLOpType.count(up_op_desc->GetType()) > 0) {
        continue;
      }
      if ((up_op_desc->HasAttr(kTypeFFTSPlus) && !IsPhonyOp(up_op_desc)) || up_op_desc->GetType() == kAtomicAddrClean) {
        vector <uint32_t> succ_lists;
        (void)ge::AttrUtils::GetListInt(up_op_desc, kSuccList, succ_lists);
        succ_lists.push_back(ctx_id_list[index]);
        (void)ge::AttrUtils::SetListInt(up_op_desc, kSuccList, succ_lists);
        FFTS_LOGD("hccl node name:%s, succ_lists:%s", up_op_desc->GetName().c_str(),
                  fe::StringUtils::IntegerVecToString(succ_lists).c_str());
      }
    }
  }
  return SUCCESS;
}
}  // namespace

CollectionOpsTaskBuilder::CollectionOpsTaskBuilder() {}

CollectionOpsTaskBuilder::~CollectionOpsTaskBuilder() {}

void CollectionOpsTaskBuilder::GetSubtaskInputCnt(const std::vector<std::vector<int64_t>> &adjacency_list,
                                                  std::map<int64_t, int32_t> &subtask_input_cnt) const {
  for (size_t i = 0; i < adjacency_list.size(); i++) {
    auto &sub_task_idx = adjacency_list.at(i);
    for (size_t j = 0; j < sub_task_idx.size(); j++) {
      if (sub_task_idx.at(j) != -1) {
        subtask_input_cnt[sub_task_idx.at(j)]++;
      }
    }
  }
}

Status CollectionOpsTaskBuilder::GenContextDef(const ge::NodePtr &node,
                                               domi::FftsPlusTaskDef *ffts_plus_task_def) {
  FFTS_LOGD("CollectionOpsTaskBuilder::GenContextDef begin, node name:%s, node type:%s.", node->GetName().c_str(),
            node->GetType().c_str());
  auto op_desc = node->GetOpDesc();
  vector<FftsPlusComCtx_t> sub_ffts_plus_context;
  GenFftsPlusTaskCommonInfo(node, sub_ffts_plus_context);
  std::vector<domi::FftsPlusCtxDef> hccl_sub_tasks;
  std::vector<uint32_t> ctx_id_list;
  std::vector<std::vector<int64_t>> adjacency_list;
  hccl_sub_tasks = op_desc->TryGetExtAttr(kHcclSubTasks, hccl_sub_tasks);
  (void)ge::AttrUtils::GetListInt(op_desc, kCtxIdList, ctx_id_list);
  if (hccl_sub_tasks.size() != ctx_id_list.size()) {
    FFTS_LOGE("HCCL GenContextDef fail, Invalid param");
  }
  (void)ge::AttrUtils::GetListListInt(op_desc, kAdjacencyList, adjacency_list);
  std::map<int64_t, int32_t> subtask_input_cnt;
  GetSubtaskInputCnt(adjacency_list, subtask_input_cnt);
  int32_t i = 0;
  int32_t hccl_output_degree_0_num = 0;
  vector<vector<int64_t>> succ_list_list;
  for (const auto &hccl_sub_task : hccl_sub_tasks) {
    domi::FftsPlusCtxDef *ffts_plus_ctx_def = ffts_plus_task_def->add_ffts_plus_ctx();
    FFTS_CHECK_NOTNULL(ffts_plus_ctx_def);
    ffts_plus_ctx_def->set_context_type(hccl_sub_task.context_type());
    ffts_plus_ctx_def->set_op_index(op_desc->GetId());

    if (sub_ffts_plus_context.empty()) {
      return FAILED;
    }
    FillHCCLCtxDef(hccl_sub_task, ffts_plus_ctx_def, subtask_input_cnt, i, sub_ffts_plus_context);
    ModifyHCCLInputNodeSuccList(node, subtask_input_cnt, ctx_id_list, i);
    vector<int64_t> succ_list;
    for (size_t j = 0; j < adjacency_list.at(i).size(); j++) {
      succ_list.push_back(ctx_id_list[adjacency_list.at(i).at(j)]);
    }
    /* when hccl inner'succlist equal zero, we need link the succlist of this node with subgraph output */
    if (adjacency_list.at(i).size() == 0) {
      succ_list.clear();
      for (const auto ele : sub_ffts_plus_context[0].succ_list) {
        succ_list.push_back(ele);
      }
      hccl_output_degree_0_num++;
    }
    i++;
    succ_list_list.push_back(succ_list);
  }
  (void)ge::AttrUtils::SetListListInt(op_desc, kSuccListList, succ_list_list);
  (void)ge::AttrUtils::SetInt(op_desc, kHcclOutDegree0Num, hccl_output_degree_0_num);
  return SUCCESS;
}
}  // namespace ffts
