/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mixl2_mode_task_builder.h"
#include "common/aicore_util_attr_define.h"
namespace ffts {
Mixl2ModeTaskBuilder::Mixl2ModeTaskBuilder() {}
Mixl2ModeTaskBuilder::~Mixl2ModeTaskBuilder() {}

Status Mixl2ModeTaskBuilder::Initialize() {
  mode_type_ = ModeType::MIX_L2_MODE_TYPE;
  FFTS_MAKE_SHARED(mix_L2_task_builder_ptr_ = std::make_shared<MixL2TaskBuilder>(), return FAILED);
  return SUCCESS;
}

Status Mixl2ModeTaskBuilder::GenFftsPlusContextId(ge::ComputeGraph &sgt_graph,
                                                  std::vector<ge::NodePtr> &sub_graph_nodes,
                                                  uint64_t &ready_context_num, uint64_t &total_context_number,
                                                  std::vector<ge::NodePtr> &memset_nodes) {
  (void)sgt_graph;
  if (sub_graph_nodes.empty()) {
    REPORT_FFTS_ERROR("[FFTSPlusMixL2Mode] [GenFftsPlusContextId] No node to generate task.");
    return FAILED;
  }
  ge::NodePtr node = sub_graph_nodes[0];
  uint32_t contextId = 0;
  FFTS_CHECK_NOTNULL(node);
  ge::OpDescPtr op_desc = node->GetOpDesc();
  ge::NodePtr memset_node = nullptr;
  memset_node = op_desc->TryGetExtAttr(fe::ATTR_NAME_MEMSET_NODE, memset_node);
  if (memset_node != nullptr) {
    FFTS_LOGD("Mixl2 node:%s has memset node:%s.", op_desc->GetNamePtr(), memset_node->GetNamePtr());
    memset_nodes.emplace_back(memset_node);
    vector<uint32_t> atomic_context_id_list = {contextId};
    contextId++;
    (void)ge::AttrUtils::SetListInt(op_desc, kAtomicCtxIdList, atomic_context_id_list);
  }
  FFTS_LOGD("Gen mixl2 task with type:%s, name:%s.", op_desc->GetTypePtr(), op_desc->GetNamePtr());
  (void)ge::AttrUtils::SetInt(op_desc, kContextId, contextId++);

  ready_context_num = 1;
  total_context_number = contextId;
  return SUCCESS;
}

static void GenerateAtomicCtx(const ge::NodePtr node, const std::vector<ge::NodePtr> &memset_nodes,
                       domi::FftsPlusTaskDef *ffts_plus_task_def) {
  (void)node;
  if (memset_nodes.empty()) {
    return;
  }

  auto memset_node_desc = memset_nodes[0]->GetOpDesc();
  int64_t block_dim = 1;
  (void)ge::AttrUtils::GetInt(memset_node_desc, ge::TVM_ATTR_NAME_BLOCKDIM, block_dim);

  std::string kernel_name;
  (void)ge::AttrUtils::GetStr(memset_node_desc, fe::kKernelName, kernel_name);

  domi::FftsPlusCtxDef *ffts_plus_ctx_def = ffts_plus_task_def->add_ffts_plus_ctx();
  ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_AIV);
  ffts_plus_ctx_def->set_op_index(memset_node_desc->GetId());
  ffts_plus_ctx_def->set_op_type(domi::FftsPlusCtxDef::ATOMIC);
  ffts_plus_ctx_def->set_context_id(0);

  // This uniq_ctx_name for profiling parser
  ffts_plus_ctx_def->set_uniq_ctx_name(memset_node_desc->GetName() + "_" + fe::MEMSET_OP_TYPE);

  domi::FftsPlusAicAivCtxDef *aic_aiv_ctx_def = ffts_plus_ctx_def->mutable_aic_aiv_ctx();
  FFTS_CHECK(aic_aiv_ctx_def == nullptr, FFTS_LOGW("aic_aiv_ctx_def is nullptr."), return);
  std::vector<int64_t> work_spaces = memset_node_desc->GetWorkspace();
  for (auto &work_space : work_spaces) {
    if (work_space != 0) {
      aic_aiv_ctx_def->add_task_addr(work_space);
    }
  }
  aic_aiv_ctx_def->add_kernel_name(kernel_name);
  aic_aiv_ctx_def->set_aten(0);
  aic_aiv_ctx_def->set_successor_num(1);
  aic_aiv_ctx_def->add_successor_list(1);
  aic_aiv_ctx_def->set_thread_dim(1);
  aic_aiv_ctx_def->set_non_tail_block_dim(block_dim);
  aic_aiv_ctx_def->set_tail_block_dim(block_dim);
  uint32_t addr_size = aic_aiv_ctx_def->task_addr_size();
  uint32_t cur_addr_size = ffts_plus_task_def->addr_size();
  ffts_plus_task_def->set_addr_size(cur_addr_size + addr_size);
}

Status Mixl2ModeTaskBuilder::GenSubGraphTaskDef(std::vector<ge::NodePtr> &memset_nodes,
                                                std::vector<ge::NodePtr> &sub_graph_nodes,
                                                domi::TaskDef &task_def) {
  if (sub_graph_nodes.empty()) {
    REPORT_FFTS_ERROR("[FFTSPlusMixL2Mode] [GenSubGraphTaskDef] No node to generate taskdef.");
    return FAILED;
  }
  ge::NodePtr node = sub_graph_nodes[0];
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  FFTS_CHECK_NOTNULL(ffts_plus_task_def);
  GenerateAtomicCtx(node, memset_nodes, ffts_plus_task_def);

  FFTS_LOGD("Generated mxl2 node name: %s.", node->GetOpDesc()->GetNamePtr());
  TaskBuilderType task_builder_type = TaskBuilderType::EN_TASK_TYPE_MIX_L2_AIC_AIV;
  FFTSPlusTaskBuilderPtr task_builder = GetTaskBuilder(task_builder_type);
  FFTS_CHECK_NOTNULL(task_builder);
  Status status = task_builder->GenerateTaskDef(node, ffts_plus_task_def);
  return status;
}
}  // namespace ffts
