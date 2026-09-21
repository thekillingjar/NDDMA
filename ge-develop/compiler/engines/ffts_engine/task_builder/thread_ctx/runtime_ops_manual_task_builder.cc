/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "runtime_ops_manual_task_builder.h"
#include "inc/ffts_utils.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "runtime/base.h"
#include "common/resource_def.h"

namespace ffts {
RuntimeOpsTaskBuilder::RuntimeOpsTaskBuilder() {}
RuntimeOpsTaskBuilder::~RuntimeOpsTaskBuilder() {}
Status RuntimeOpsTaskBuilder::JudgeContextType(const ge::NodePtr &node, rtFftsPlusContextType_t &context_type) {
  auto op_type = node->GetType();
  for (auto &type_context : RTS_CONTEXT_TYPE_MAP) {
    if (type_context.second.find(op_type) != type_context.second.end()) {
      context_type = type_context.first;
      return SUCCESS;
    }
  }
  FFTS_LOGE("[JudgeContextType]:the node[%s] type[%s] is not supported, Judge Context type failed.",
            node->GetName().c_str(), node->GetType().c_str());
  return FAILED;
}

Status RuntimeOpsTaskBuilder::GenContextDef(const ge::NodePtr &node, domi::FftsPlusTaskDef *ffts_plus_task_def) {
  FFTS_LOGD("RuntimeOpsTaskBuilder::GenContextDef begin, node name:%s, node type:%s.", node->GetName().c_str(),
            node->GetType().c_str());
  auto op_desc = node->GetOpDesc();
  Status status = FAILED;
  vector<FftsPlusComCtx_t> sub_ffts_plus_context;
  GenFftsPlusTaskCommonInfo(node, sub_ffts_plus_context);

  FftsPlusCtxDefPtr ctx_def_ptr =  nullptr;
  ctx_def_ptr = op_desc->TryGetExtAttr("FFTS_PLUS_TASK_DEF", ctx_def_ptr);
  FFTS_CHECK_NOTNULL(ctx_def_ptr);
  domi::FftsPlusCtxDef *ffts_plus_ctx_def = ffts_plus_task_def->add_ffts_plus_ctx();
  FFTS_CHECK_NOTNULL(ffts_plus_ctx_def);
  ffts_plus_ctx_def->set_op_index(op_desc->GetId());

  uint32_t context_id = 0;
  if (ge::AttrUtils::GetInt(op_desc, kContextId, context_id)) {
    ffts_plus_ctx_def->set_context_id(context_id);
  }

  rtFftsPlusContextType_t context_type = RT_CTX_TYPE_AICORE;
  if (sub_ffts_plus_context.empty()) {
    return FAILED;
  }
  if (JudgeContextType(node, context_type) != SUCCESS) {
    return FAILED;
  }
  ffts_plus_ctx_def->set_context_type(context_type);

  switch (context_type) {
    case RT_CTX_TYPE_CASE_SWITCH:
      status = FillCaseSwitchContext(node, ffts_plus_ctx_def, ctx_def_ptr, sub_ffts_plus_context, 0);
      break;
    case RT_CTX_TYPE_LABEL:
      status = FillLabelContext(op_desc, ffts_plus_ctx_def, sub_ffts_plus_context, 0);
      break;
    case RT_CTX_TYPE_SDMA:
      status = FillSdmaContext(op_desc, ffts_plus_ctx_def, ctx_def_ptr, sub_ffts_plus_context, 0);
      break;
    case RT_CTX_TYPE_COND_SWITCH:
      status = FillCondSwitchContext(node, ffts_plus_ctx_def, ctx_def_ptr, sub_ffts_plus_context, 0);
      break;
    default:
      status = FAILED;
      break;
  }
  if (status != SUCCESS) {
    FFTS_LOGE("GenContextTaskDef failed. Op[%s, optype[%s]]",
              op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return status;
  }
  return SUCCESS;
}

Status RuntimeOpsTaskBuilder::FillLabelContext(const ge::OpDescPtr &op_desc, domi::FftsPlusCtxDef *ffts_plus_ctx_def,
    vector<FftsPlusComCtx_t> &sub_ffts_plus_context, size_t thread_id) const {
  if (sub_ffts_plus_context.empty()) {
    return FAILED;
  }
  domi::FftsPlusLabelCtxDef *label_ctx_def = ffts_plus_ctx_def->mutable_label_ctx();
  FFTS_CHECK_NOTNULL(label_ctx_def);
  // label do not need to fill context data.
  label_ctx_def->set_pred_cnt(sub_ffts_plus_context[thread_id].pred_cnt);
  label_ctx_def->set_pred_cnt_init(sub_ffts_plus_context[thread_id].pred_cnt);
  label_ctx_def->set_successor_num(0);
  FFTS_LOGD("FillLabelContext Op[%s], type[%s]", op_desc->GetName().c_str(), op_desc->GetType().c_str());
  FFTS_LOGD("FillLabelContext pred_cnt[%u], success_num[%u]", sub_ffts_plus_context[0].pred_cnt,
            sub_ffts_plus_context[0].successorNum);
  if (sub_ffts_plus_context.size() == 1) {
    (void)ge::AttrUtils::SetListInt(op_desc, kSuccList, sub_ffts_plus_context[0].succ_list);
    for (uint32_t i = 0; i < sub_ffts_plus_context[0].succ_list.size(); i++) {
      label_ctx_def->add_successor_list(sub_ffts_plus_context[0].succ_list[i]);
      FFTS_LOGD("FillLabelContext succ_list[%u] has occurred",
                sub_ffts_plus_context[0].succ_list[i]);
    }
  }
  return SUCCESS;
}

Status RuntimeOpsTaskBuilder::FillSdmaContext(const ge::OpDescPtr &op_desc, domi::FftsPlusCtxDef *ffts_plus_ctx_def,
    FftsPlusCtxDefPtr &ctx_def_ptr, vector<FftsPlusComCtx_t> &sub_ffts_plus_context, size_t thread_id) const {
  if (sub_ffts_plus_context.empty()) {
    return FAILED;
  }
  domi::FftsPlusSdmaCtxDef *sdma_ctx_def_ptr = ctx_def_ptr->mutable_sdma_ctx();
  FFTS_CHECK_NOTNULL(sdma_ctx_def_ptr);
  ffts_plus_ctx_def->set_context_type(RT_CTX_TYPE_SDMA);
  domi::FftsPlusSdmaCtxDef *sdma_ctx_def = ffts_plus_ctx_def->mutable_sdma_ctx();
  FFTS_CHECK_NOTNULL(sdma_ctx_def);
  Status status = FillSdmaContextData(sdma_ctx_def_ptr, sdma_ctx_def);
  if (status != SUCCESS) {
    FFTS_LOGE("FillSdmaContextData failed. Op[%s, optype[%s]]", op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return status;
  }
  sdma_ctx_def->set_pred_cnt(sub_ffts_plus_context[thread_id].pred_cnt);
  sdma_ctx_def->set_pred_cnt_init(sub_ffts_plus_context[thread_id].pred_cnt);
  sdma_ctx_def->set_successor_num(0);
  FFTS_LOGD("Fill Sdma context with size: %zu", sub_ffts_plus_context.size());
  sdma_ctx_def->set_thread_id(thread_id);
  if (sub_ffts_plus_context.size() == 1) { // manual
    sdma_ctx_def->set_aten(0);
    sdma_ctx_def->set_atm(0);
    (void)ge::AttrUtils::SetListInt(op_desc, kSuccList, sub_ffts_plus_context[0].succ_list);
  } else {
    sdma_ctx_def->set_aten(1);
    sdma_ctx_def->set_atm(1);
  }
  return status;
}

Status RuntimeOpsTaskBuilder::FillSdmaContextData(const domi::FftsPlusSdmaCtxDef *sdma_ctx_def_ptr,
                                                  domi::FftsPlusSdmaCtxDef *sdma_ctx_def) const {
  FFTS_CHECK_NOTNULL(sdma_ctx_def_ptr);
  FFTS_CHECK_NOTNULL(sdma_ctx_def);
  sdma_ctx_def->set_pmg(sdma_ctx_def_ptr->pmg());
  sdma_ctx_def->set_ns(sdma_ctx_def_ptr->ns());
  sdma_ctx_def->set_part_id(sdma_ctx_def_ptr->part_id());
  sdma_ctx_def->set_qos(sdma_ctx_def_ptr->qos());
  sdma_ctx_def->set_thread_dim(sdma_ctx_def_ptr->thread_dim());
  sdma_ctx_def->set_sdma_sqe_header(sdma_ctx_def_ptr->sdma_sqe_header());
  sdma_ctx_def->set_src_stream_id(sdma_ctx_def_ptr->src_stream_id());
  sdma_ctx_def->set_src_sub_stream_id(sdma_ctx_def_ptr->src_sub_stream_id());
  sdma_ctx_def->set_dst_stream_id(sdma_ctx_def_ptr->dst_stream_id());
  sdma_ctx_def->set_dst_sub_stream_id(sdma_ctx_def_ptr->dst_sub_stream_id());
  sdma_ctx_def->set_src_addr_base(sdma_ctx_def_ptr->src_addr_base());
  sdma_ctx_def->set_src_addr_offset(sdma_ctx_def_ptr->src_addr_offset());
  sdma_ctx_def->set_dst_addr_base(sdma_ctx_def_ptr->dst_addr_base());
  sdma_ctx_def->set_dst_addr_offset(sdma_ctx_def_ptr->dst_addr_offset());
  sdma_ctx_def->set_non_tail_data_len(sdma_ctx_def_ptr->non_tail_data_len());
  sdma_ctx_def->set_tail_data_len(sdma_ctx_def_ptr->tail_data_len());
  return SUCCESS;
}


Status RuntimeOpsTaskBuilder::FillCaseSwitchContext(const ge::NodePtr &node, domi::FftsPlusCtxDef *ffts_plus_ctx_def,
    FftsPlusCtxDefPtr &ctx_def_ptr, vector<FftsPlusComCtx_t> &sub_ffts_plus_context, size_t thread_id) const {
  auto op_desc = node->GetOpDesc();
  if (sub_ffts_plus_context.empty()) {
    return FAILED;
  }
  FFTS_LOGD("FillCaseSwitchContext Op[%s, optype[%s]].", op_desc->GetName().c_str(), op_desc->GetType().c_str());

  domi::FftsPlusCaseSwitchCtxDef *case_switch_ctx_def_ptr = ctx_def_ptr->mutable_case_switch_ctx();
  FFTS_CHECK_NOTNULL(case_switch_ctx_def_ptr);
  domi::FftsPlusCaseSwitchCtxDef *case_switch_ctx_def = ffts_plus_ctx_def->mutable_case_switch_ctx();
  FFTS_CHECK_NOTNULL(case_switch_ctx_def);
  Status status = FillCaseSwitchContextData(case_switch_ctx_def_ptr, case_switch_ctx_def);
  if (status != SUCCESS) {
    FFTS_LOGE("FillCaseSwitchContextData failed. Op[%s, optype[%s]]",
              op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return status;
  }
  
  FFTS_LOGD("FillCaseSwitchContext pred_cnt[%u], successnum[%u]", sub_ffts_plus_context[0].pred_cnt,
            sub_ffts_plus_context[0].successorNum);
  case_switch_ctx_def->set_pred_cnt(sub_ffts_plus_context[thread_id].pred_cnt);
  case_switch_ctx_def->set_pred_cnt_init(sub_ffts_plus_context[thread_id].pred_cnt);
  case_switch_ctx_def->set_successor_num(0);
  case_switch_ctx_def->set_start_label_id(0);
  case_switch_ctx_def->set_label_list_len(sub_ffts_plus_context[0].successorNum);
  case_switch_ctx_def->set_thread_id(thread_id);
  if (sub_ffts_plus_context.size() == 1) {
    case_switch_ctx_def->set_aten(0);
    case_switch_ctx_def->set_atm(0);
    (void)ge::AttrUtils::SetListInt(op_desc, kSuccList, sub_ffts_plus_context[0].succ_list);
  } else {
    case_switch_ctx_def->set_aten(1);
    case_switch_ctx_def->set_atm(1);
  }
  if (node->GetType() == "LabelSwitchByIndex") {
    uint64_t data_base = 0;
    status = GetSwitchInputDataAddr(node, data_base);
    if (status != SUCCESS) {
      return FAILED;
    }
    case_switch_ctx_def->set_load_addr0_base(data_base);
    FFTS_LOGD("FillCaseSwitchContext data_base[%llu].", data_base);
  }
  return status;
}

Status RuntimeOpsTaskBuilder::FillCaseSwitchContextData(const domi::FftsPlusCaseSwitchCtxDef *case_switch_ctx_def_ptr,
                                                        domi::FftsPlusCaseSwitchCtxDef *case_switch_ctx_def) const {
  FFTS_CHECK_NOTNULL(case_switch_ctx_def_ptr);
  FFTS_CHECK_NOTNULL(case_switch_ctx_def);
  case_switch_ctx_def->set_thread_dim(case_switch_ctx_def_ptr->thread_dim());
  case_switch_ctx_def->set_ar_size(case_switch_ctx_def_ptr->ar_size());
  case_switch_ctx_def->set_snoop(case_switch_ctx_def_ptr->snoop());
  case_switch_ctx_def->set_ar_cache(case_switch_ctx_def_ptr->ar_cache());
  case_switch_ctx_def->set_ar_prot(case_switch_ctx_def_ptr->ar_prot());
  case_switch_ctx_def->set_va(case_switch_ctx_def_ptr->va());
  case_switch_ctx_def->set_load_addr0_base(case_switch_ctx_def_ptr->load_addr0_base());
  FFTS_LOGD("FillCaseSwitchContext load_addr0_base[%llu].",
            case_switch_ctx_def_ptr->load_addr0_base());
  case_switch_ctx_def->set_ld0_en(case_switch_ctx_def_ptr->ld0_en());
  case_switch_ctx_def->set_load_addr0_offset(case_switch_ctx_def_ptr->load_addr0_offset());
  case_switch_ctx_def->set_load_addr1_base(case_switch_ctx_def_ptr->load_addr1_base());
  case_switch_ctx_def->set_ld1_en(case_switch_ctx_def_ptr->ld1_en());
  case_switch_ctx_def->set_load_addr1_offset(case_switch_ctx_def_ptr->load_addr1_offset());
  return SUCCESS;
}

Status RuntimeOpsTaskBuilder::FillCondSwitchContext(const ge::NodePtr &node, domi::FftsPlusCtxDef *ffts_plus_ctx_def,
    FftsPlusCtxDefPtr &ctx_def_ptr, vector<FftsPlusComCtx_t> &sub_ffts_plus_context, size_t thread_id) const {
  auto op_desc = node->GetOpDesc();
  if (sub_ffts_plus_context.empty()) {
    return FAILED;
  }
  FFTS_LOGD("FillCondSwitchContext Op[%s, optype[%s]].", op_desc->GetName().c_str(), op_desc->GetType().c_str());

  domi::FftsPlusCondSwitchCtxDef *cond_switch_ctx_def_ptr = ctx_def_ptr->mutable_cond_switch_ctx();
  FFTS_CHECK_NOTNULL(cond_switch_ctx_def_ptr);
  domi::FftsPlusCondSwitchCtxDef *cond_switch_ctx_def = ffts_plus_ctx_def->mutable_cond_switch_ctx();
  FFTS_CHECK_NOTNULL(cond_switch_ctx_def);
  Status status = FillCondSwitchContextData(cond_switch_ctx_def_ptr, cond_switch_ctx_def);
  if (status != SUCCESS) {
    FFTS_LOGE("FillCondSwitchContextData failed. Op[%s, optype[%s]]",
              op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return status;
  }
  
  FFTS_LOGD("FillCondSwitchContext pred_cnt[%u], success_num[%u]", sub_ffts_plus_context[0].pred_cnt,
            sub_ffts_plus_context[0].successorNum);
  cond_switch_ctx_def->set_pred_cnt(sub_ffts_plus_context[0].pred_cnt);
  cond_switch_ctx_def->set_pred_cnt_init(sub_ffts_plus_context[0].pred_cnt);
  cond_switch_ctx_def->set_true_successor_num(0);
  auto iter = RTS_COND_TYPE_VALUE_MAP.find(node->GetType());
  if (iter == RTS_COND_TYPE_VALUE_MAP.end()) {
    return FAILED;
  }
  cond_switch_ctx_def->set_condition(iter->second);
  cond_switch_ctx_def->set_cmp_value_1(0);
  cond_switch_ctx_def->set_cmp_value_2(0);
  cond_switch_ctx_def->set_thread_id(thread_id);
  if (sub_ffts_plus_context.size() == 1) {
    cond_switch_ctx_def->set_aten(0);
    cond_switch_ctx_def->set_atm(0);
    (void)ge::AttrUtils::SetListInt(op_desc, kSuccList, sub_ffts_plus_context[0].succ_list);
  } else {
    cond_switch_ctx_def->set_aten(1);
    cond_switch_ctx_def->set_atm(1);
  }
  return status;
}

Status RuntimeOpsTaskBuilder::FillCondSwitchContextData(const domi::FftsPlusCondSwitchCtxDef *cond_switch_ctx_def_ptr,
                                                        domi::FftsPlusCondSwitchCtxDef *cond_switch_ctx_def) const {
  FFTS_CHECK_NOTNULL(cond_switch_ctx_def_ptr);
  FFTS_CHECK_NOTNULL(cond_switch_ctx_def);
  cond_switch_ctx_def->set_thread_dim(cond_switch_ctx_def_ptr->thread_dim());
  cond_switch_ctx_def->set_ar_size(cond_switch_ctx_def_ptr->ar_size());
  cond_switch_ctx_def->set_snoop(cond_switch_ctx_def_ptr->snoop());
  cond_switch_ctx_def->set_ar_cache(cond_switch_ctx_def_ptr->ar_cache());
  cond_switch_ctx_def->set_ar_prot(cond_switch_ctx_def_ptr->ar_prot());
  cond_switch_ctx_def->set_va(cond_switch_ctx_def_ptr->va());
  cond_switch_ctx_def->set_ld0_en(cond_switch_ctx_def_ptr->ld0_en());
  cond_switch_ctx_def->set_ld1_en(cond_switch_ctx_def_ptr->ld1_en());
  return SUCCESS;
}

Status RuntimeOpsTaskBuilder::GetSwitchInputDataAddr(const ge::NodePtr &node, uint64_t &data_base) const {
  std::shared_ptr<ge::RunContext> contextptr = nullptr;
  auto op_desc = node->GetOpDesc();
  contextptr = op_desc->TryGetExtAttr(kRuntimeContentx, contextptr);
  if (contextptr == nullptr) {
    FFTS_LOGD("RuntimeOpsTaskBuilder contextPtr is null.");
    return FAILED;
} else {
    FFTS_LOGD("RuntimeOpsTaskBuilder: contextptr is not null, weightbase = %p, datamembase = %p.",
              contextptr->weightMemBase, contextptr->dataMemBase);
  }
  auto op_type = op_desc->GetType();
  auto op_name = op_desc->GetName();

  if (node->GetAllInDataAnchors().size() == 0) {
    FFTS_LOGD("When labelswitch inputdataanchor is invoked, it should return 1.");
    return FAILED;
  }
  auto anchor = node->GetAllInDataAnchors().at(0);
  FFTS_LOGD("GetSwitchInputDataAddr anchor = %x", anchor);
  if (anchor == nullptr) {
    FFTS_LOGD("labelswitch: inputdataanchor anchor is null");
    return FAILED;
  }
  if (ge::AnchorUtils::GetStatus(anchor) == ge::ANCHOR_SUSPEND) {
      FFTS_LOGD("Node[type=%s,name=%s]: status is suspend.", op_type.c_str(), op_name.c_str());
      return FAILED;
  }
  if (op_desc->GetInputOffset().size() == 0) {
    FFTS_LOGD("Node[type=%s, name=%s]: Status is suspended. GetInputOffset size = 0.",
              op_type.c_str(), op_name.c_str());
    return FAILED;
  }
  const uint64_t virtual_addr_offset = op_desc->GetInputOffset().at(0);
  FFTS_LOGD("virtual_addr_offset = %llu", virtual_addr_offset);
  if (ge::AnchorUtils::GetStatus(anchor) != ge::ANCHOR_DATA) {
    data_base = reinterpret_cast<uint64_t>(contextptr->weightMemBase + virtual_addr_offset);
  } else {
    data_base = reinterpret_cast<uint64_t>(contextptr->dataMemBase + virtual_addr_offset);
  }
  return SUCCESS;
}
}  // namespace ffts
