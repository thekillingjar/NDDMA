/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "data_task_builder.h"

namespace ffts {
namespace {
const int64_t kMaxBurstLen = 201326592;
static const std::map<CACHE_OPERATION, std::tuple<string, ContextType, bool, string>> kDataOptInfo = {
  std::make_pair(CACHE_OPERATION::PREFETCH,
                 std::make_tuple(kPrefetchEnableBm, RT_CTX_TYPE_FLUSH_DATA, true, "prefetch")),
  std::make_pair(CACHE_OPERATION::INVALIDATE,
                 std::make_tuple(kInvalidateBm, RT_CTX_TYPE_INVALIDATE_DATA, false, "invalidate")),
  std::make_pair(CACHE_OPERATION::WRITE_BACK,
                 std::make_tuple(kWriteBackBm, RT_CTX_TYPE_WRITEBACK_DATA, false, "write back"))
};

Status SetDataContextId(size_t anchor_index, const ge::NodePtr &node, bool is_input,
                        std::vector<uint32_t> &ctx_id_vec) {
  ge::GeTensorDescPtr desc_ptr = nullptr;
  FFTS_LOGD("Node [%s] set context ID attribute with size [%zu].", node->GetName().c_str(), ctx_id_vec.size());
  if (is_input) {
    desc_ptr = node->GetOpDesc()->MutableInputDesc(anchor_index);
  } else {
    desc_ptr = node->GetOpDesc()->MutableOutputDesc(anchor_index);
  }
  FFTS_CHECK_NOTNULL(desc_ptr);
  (void)ge::AttrUtils::SetListInt(desc_ptr, kTensorCtxId, ctx_id_vec);
  return SUCCESS;
}
}  // namespace

DataTaskBuilder::DataTaskBuilder() : operation_(CACHE_OPERATION::CACHE_OPERATION_BOTTOM) {}

DataTaskBuilder::DataTaskBuilder(CACHE_OPERATION operation) : operation_(operation), burst_len_(kMaxBurstLen) {}

DataTaskBuilder::~DataTaskBuilder() {}

Status DataTaskBuilder::GenManualDataCtxDef(const ge::NodePtr &node, domi::FftsPlusTaskDef *ffts_plus_task_def) {
  FFTS_LOGD("DataTaskBuilder::GenManualContextDef begin, node name:%s, node type:%s.", node->GetName().c_str(),
            node->GetType().c_str());
  auto op_desc = node->GetOpDesc();

  const auto &operation = kDataOptInfo.at(operation_);
  string bm_name = std::get<0>(operation);
  rtFftsPlusContextType_t context_type = std::get<1>(operation);
  bool is_input = std::get<2>(operation);
  string operation_name = std::get<3>(operation);
  int64_t bm = 0;
  if (!ge::AttrUtils::GetInt(op_desc, bm_name, bm) || bm == 0) {
    return SUCCESS;
  }
  const auto &lib_name = op_desc->GetOpKernelLibName();
  FFTS_LOGD("Node %s needs %s context, lib_name: %s.", node->GetName().c_str(), operation_name.c_str(), lib_name.c_str());
  if (lib_name == kRtsFftsPlusOpStoreName) {
    return SUCCESS;
  }

  auto indices = GetIndices(node);
  size_t curr_num = 0;
  std::vector<int32_t> cmo_idx;
  // for static shape
  for (auto index : indices) {
    uint64_t curr_bm = 0;
    SetBitOne(static_cast<uint32_t>(index), curr_bm);
    if ((static_cast<uint64_t>(bm) & curr_bm) == 0) {
      FFTS_LOGD("bm is %lld and current BM is %llu.", bm, curr_bm);
      continue;
    }

    FFTS_LOGD("Node %s's tensor %d needs %s.", node->GetName().c_str(), index, operation_name.c_str());

    std::vector<DataContextParam> data_ctx_params;
    Status ret = MemorySlice::GenerateManualDataCtxParam(node, index, is_input, burst_len_, data_ctx_params);
    if (ret != SUCCESS) {
      continue;
    }

    if (ExceedMaxCtxNum(curr_num, data_ctx_params.size())) {
      FFTS_LOGI("Exceeded the upper limit of prefetch context. Current total is %zu, pending total is %zu.",
                curr_num, data_ctx_params.size());
      continue;
    }

    curr_num += data_ctx_params.size();
    FFTS_LOGD("Size of context is %zu, curr_num is %zu.", data_ctx_params.size(), curr_num);
    size_t anchor_index = static_cast<size_t>(index);
    std::vector<uint32_t> ctx_id_vec;
    for (auto &param : data_ctx_params) {
      domi::FftsPlusCtxDef *ffts_plus_ctx_def = ffts_plus_task_def->add_ffts_plus_ctx();
      FFTS_CHECK_NOTNULL(ffts_plus_ctx_def);
      ffts_plus_ctx_def->set_context_type(context_type);
      ffts_plus_ctx_def->set_context_id(ffts_plus_task_def->ffts_plus_ctx_size() - 1);

      // This uniq_ctx_name for profiling parser
      ffts_plus_ctx_def->set_uniq_ctx_name(op_desc->GetName() + "_" + operation_name + "_" + to_string(index));

      domi::FftsPlusDataCtxDef *data_ctx_def = ffts_plus_ctx_def->mutable_data_ctx();
      FFTS_CHECK_NOTNULL(data_ctx_def);
      uint32_t ctx_id = static_cast<uint32_t>(ffts_plus_task_def->ffts_plus_ctx_size() - 1);
      ctx_id_vec.emplace_back(ctx_id);
      FFTS_LOGD("Filling one %s context[%u] for tensor %zu in node %s.", operation_name.c_str(), ctx_id,
                anchor_index, node->GetName().c_str());
      Status status = FillManualDataCtx(anchor_index, node, param, ffts_plus_task_def, data_ctx_def);
      if (status != SUCCESS) {
        REPORT_FFTS_ERROR("[GenTask][InvldTsk][GenCtxDef]Fill context %s %zu failed. Op[%s], optype[%s]",
                          operation_name.c_str(), anchor_index, op_desc->GetName().c_str(), op_desc->GetType().c_str());
        return status;
      }
      break;
    }
    (void)SetDataContextId(anchor_index, node, is_input, ctx_id_vec);
    cmo_idx.emplace_back(index);
  }
  (void)ge::AttrUtils::SetListInt(node->GetOpDesc(), operation_name + "_idx", cmo_idx);
  return SUCCESS;
}

Status DataTaskBuilder::GenAutoDataCtxDef(const ge::NodePtr &node, domi::FftsPlusTaskDef *ffts_plus_task_def) {
  FFTS_LOGD("DataTaskBuilder::GenAutoContextDef begin, node name:%s, node type:%s", node->GetName().c_str(),
            node->GetType().c_str());
  auto op_desc = node->GetOpDesc();

  const auto &operation = kDataOptInfo.at(operation_);
  string bm_name = std::get<0>(operation);
  rtFftsPlusContextType_t context_type = std::get<1>(operation);
  bool is_input = std::get<2>(operation);
  string operation_name = std::get<3>(operation);
  int64_t bm = 0;
  if (!ge::AttrUtils::GetInt(op_desc, bm_name, bm) || bm == 0) {
    return SUCCESS;
  }
  FFTS_LOGD("Node %s needs %s context", node->GetName().c_str(), operation_name.c_str());
  auto indices = GetIndices(node);
  size_t curr_num = 0;
  std::vector<int32_t> cmo_idx;
  // for static shape
  for (auto index : indices) {
    uint64_t curr_bm = 0;
    SetBitOne(static_cast<uint32_t>(index), curr_bm);
    if ((static_cast<uint64_t>(bm) & curr_bm) == 0) {
      FFTS_LOGD("bm is %lld and current BM is %llu.", bm, curr_bm);
      continue;
    }

    FFTS_LOGD("Node %s's tensor %d needs %s.", node->GetName().c_str(), index, operation_name.c_str());

    // one context has nontail and tail params
    std::vector<DataContextParam> param_nontail_tail;
    Status ret = MemorySlice::GenerateAutoDataCtxParam(node, index, is_input, burst_len_, param_nontail_tail);
    if (ret != SUCCESS) {
      continue;
    }

    if (ExceedMaxCtxNum(curr_num, param_nontail_tail.size())) {
      FFTS_LOGI("Exceeded the upper limit of prefetch context. Current total is %zu, pending total is %zu.",
                curr_num, param_nontail_tail.size());
      continue;
    }

    curr_num += param_nontail_tail.size();
    FFTS_LOGD("curr_num is %zu, param_nontail_tail size is %zu", curr_num, param_nontail_tail.size());

    std::vector<std::vector<DataContextParam>> data_ctx_params;
    ThreadSliceMapPtr slice_info_ptr = nullptr;
    slice_info_ptr = op_desc->TryGetExtAttr(kAttrSgtStructInfo, slice_info_ptr);
    FFTS_CHECK_NOTNULL(slice_info_ptr);
    for (size_t i = 0; i < static_cast<size_t>(slice_info_ptr->parallel_window_size); i++) {
      data_ctx_params.push_back(param_nontail_tail);
    }
    size_t anchor_index = static_cast<size_t>(index);
    std::vector<uint32_t> ctx_id_vec;
    for (size_t i = 0; i < data_ctx_params.size(); i++) {
      domi::FftsPlusCtxDef *ffts_plus_ctx_def = ffts_plus_task_def->add_ffts_plus_ctx();
      FFTS_CHECK_NOTNULL(ffts_plus_ctx_def);
      uint32_t ctx_id = static_cast<uint32_t>(ffts_plus_task_def->ffts_plus_ctx_size() - 1);
      ctx_id_vec.emplace_back(ctx_id);
      ffts_plus_ctx_def->set_context_type(context_type);
      ffts_plus_ctx_def->set_context_id(ffts_plus_task_def->ffts_plus_ctx_size() - 1);

      // This uniq_ctx_name for profiling parser
      ffts_plus_ctx_def->set_uniq_ctx_name(op_desc->GetName() + "_" + operation_name + "_" + to_string(i) + "_" +
                                           to_string(index));

      domi::FftsPlusDataCtxDef *data_ctx_def = ffts_plus_ctx_def->mutable_data_ctx();
      FFTS_CHECK_NOTNULL(data_ctx_def);
      FFTS_LOGD("Filling one %s context for tensor %zu of node %s, window_id: %zu.", operation_name.c_str(),
                anchor_index, node->GetName().c_str(), i);
      Status status = FillAutoDataCtx(anchor_index, node, data_ctx_params[i], ffts_plus_task_def, data_ctx_def, i);
      if (status != SUCCESS) {
        REPORT_FFTS_ERROR("[GenTask][InvldTsk][GenCtxDef]Fill context %s %zu failed. Op[%s], optype[%s]",
                          operation_name.c_str(), anchor_index, op_desc->GetName().c_str(), op_desc->GetType().c_str());
        return status;
      }
    }
    (void)SetDataContextId(anchor_index, node, is_input, ctx_id_vec);
    cmo_idx.emplace_back(index);
  }
  (void)ge::AttrUtils::SetListInt(node->GetOpDesc(), operation_name + "_idx", cmo_idx);
  return SUCCESS;
}

Status DataTaskBuilder::GenDynamicDataCtxDef(const ge::NodePtr &node, domi::FftsPlusTaskDef *ffts_plus_task_def) {
  FFTS_LOGD("DataTaskBuilder::GenDynamicDataCtxDef begin, node name:%s, node type:%s", node->GetName().c_str(),
            node->GetType().c_str());
  auto op_desc = node->GetOpDesc();

  const auto &operation = kDataOptInfo.at(operation_);
  string bm_name = std::get<0>(operation);
  rtFftsPlusContextType_t context_type = std::get<1>(operation);
  string operation_name = std::get<3>(operation);
  int64_t bm = 0;
  if (!ge::AttrUtils::GetInt(op_desc, bm_name, bm) || bm == 0) {
    return SUCCESS;
  }
  FFTS_LOGD("Node %s needs %s context", node->GetName().c_str(), operation_name.c_str());
  auto indices = GetIndices(node);

  vector<uint32_t> context_id_list;
  (void)ge::AttrUtils::GetListInt(node->GetOpDesc(), kAutoCtxIdList, context_id_list);
  std::vector<int32_t> cmo_idx;
  size_t count = 0;
  for (auto index : indices) {
    uint64_t curr_bm = 0;
    SetBitOne(static_cast<uint32_t>(index), curr_bm);
    if ((static_cast<uint64_t>(bm) & curr_bm) == 0) {
      FFTS_LOGD("bm is %lld and current BM is %llu.", bm, curr_bm);
      continue;
    }
    FFTS_LOGD("Node %s's tensor %d needs %s.", node->GetName().c_str(), index, operation_name.c_str());
    cmo_idx.emplace_back(index);
    Status ret = FillDynamicDataCtx(static_cast<size_t>(index), node, ffts_plus_task_def, context_type,
                                    context_id_list);
    if (ret == FAILED) {
      return FAILED;
    }
    count++;
    if (count == kMaxPretchNum) {
      break;
    }
  }
  (void)ge::AttrUtils::SetListInt(node->GetOpDesc(), operation_name + "_idx", cmo_idx);
  FFTS_LOGD("Generated dynamic data context size: %zu bytes", cmo_idx.size());
  return SUCCESS;
}

std::vector<int> DataTaskBuilder::GetIndices(const ge::NodePtr &node) const {
  vector<int> indices;
  if (operation_ == CACHE_OPERATION::PREFETCH) {
    for (const auto &in_anchor : node->GetAllInDataAnchors()) {
      if (!in_anchor) {
        continue;
      }
      uint32_t idx = in_anchor->GetIdx();
      auto desc_ptr = node->GetOpDesc()->GetInputDescPtr(idx);
      if (desc_ptr == nullptr) {
        continue;
      }
      if (desc_ptr->GetShape().IsScalar()) {
        FFTS_LOGD("Node [%s] in tensor %u is a scalar.", node->GetName().c_str(), idx);
        continue;
      }
      if (idx < kMaxIdx) {
        indices.emplace_back(in_anchor->GetIdx());
      }
    }
  } else {
    uint32_t idx = 0;
    for (size_t i = 0U; i < node->GetOpDesc()->GetOutputsSize(); ++i) {
      auto desc_ptr = node->GetOpDesc()->GetOutputDescPtr(i);
      if (desc_ptr == nullptr) {
        continue;
      }
      if (desc_ptr->GetShape().IsScalar()) {
        FFTS_LOGD("Node[%s] output tensor at index %zu is a scalar.", node->GetName().c_str(), i);
        continue;
      }
      if (IsMemoryEmpty(*desc_ptr.get())) {
        FFTS_LOGD("Node[%s] out tensor:%zu is memory empty.", node->GetName().c_str(), i);
        continue;
      }
      if (idx < kMaxIdx) {
        indices.emplace_back(idx);
        idx++;
      }
    }
  }
  return indices;
}

bool DataTaskBuilder::ExceedMaxCtxNum(size_t curr_num, size_t pending_num) const {
  if (operation_ == CACHE_OPERATION::PREFETCH) {
    return (curr_num + pending_num) > kMaxPretchNum;
  }
  return false;
}

/* In manual slicing, the memory of one thread node may not be continuous.
 * we need to calculate the following params based on the memory slicing
 * info. */
void DataTaskBuilder::FillManualThreadingParam(const DataContextParam &param,
                                               domi::FftsPlusDataCtxDef *data_ctx_def) const {
  FFTS_LOGD("start to fill Manual threading param.");
  data_ctx_def->set_non_tail_len_inner(param.len_inner);
  data_ctx_def->set_non_tail_num_inner(param.num_inner);
  data_ctx_def->set_non_tail_num_outter(param.num_outter);
  data_ctx_def->set_non_tail_stride_inner(param.stride_inner);
  data_ctx_def->set_non_tail_stride_outter(param.stride_outter);

  data_ctx_def->set_tail_len_inner(param.len_inner);
  data_ctx_def->set_tail_num_inner(param.num_inner);
  data_ctx_def->set_tail_num_outter(param.num_outter);
  data_ctx_def->set_tail_stride_inner(param.stride_inner);
  data_ctx_def->set_tail_stride_outter(param.stride_outter);
}

/* We assume the data address is continuous. So there is no gap
 * between two sgt thread and the offset is equal to size of thread.
 * All auto threads use the same data context and hardware use
 * the thread id to differenciate them.
 *
 * For one non-tail thread the base offset is equal to:
 * non-tail thread dim size * thread_id.
 * For the tail thread the base offset is equal to:
 * non-tail thread dim size * (thread_dim - 1) */
void DataTaskBuilder::FillAutoThreadingParam(const vector<DataContextParam> &params,
                                             domi::FftsPlusDataCtxDef *data_ctx_def, const uint32_t &slice_num) const {
  if (params.size() <= 1 || slice_num < 1) {
    return ;
  }
  if (data_ctx_def == nullptr) {
    return;
  }
  auto no_tail_num = (slice_num == 1) ? 1 : (slice_num - 1);
  FFTS_LOGD("start to fill auto threading param, params[1].base_addr_offset: %ld, slice_num: %u, addr_offset: %ld.",
            params[1].base_addr_offset, slice_num, params[1].base_addr_offset / no_tail_num);
  data_ctx_def->set_addr_offset(params[1].base_addr_offset / no_tail_num);
  data_ctx_def->set_non_tail_len_inner(params[0].len_inner);
  data_ctx_def->set_non_tail_num_inner(params[0].num_inner);
  data_ctx_def->set_non_tail_num_outter(params[0].num_outter);
  data_ctx_def->set_non_tail_stride_inner(params[0].stride_inner);
  data_ctx_def->set_non_tail_stride_outter(params[0].stride_outter);

  data_ctx_def->set_tail_len_inner(params[1].len_inner);
  data_ctx_def->set_tail_num_inner(params[1].num_inner);
  data_ctx_def->set_tail_num_outter(params[1].num_outter);
  data_ctx_def->set_tail_stride_inner(params[1].stride_inner);
  data_ctx_def->set_tail_stride_outter(params[1].stride_outter);
}

Status DataTaskBuilder::GetAddrBase(size_t in_anchor_index, const ge::NodePtr &node, uint64_t &addr_base) const {
  vector<int64_t> input_addrs;
  if (!ge::AttrUtils::GetListInt(node->GetOpDesc(), "input_addrs", input_addrs)) {
    FFTS_LOGW("[GenTsk][PrefetchTsk][FillCtxt][Node %s, Type %s] Attribute input_addrs is empty.",
              node->GetName().c_str(), node->GetType().c_str());
    return SUCCESS;
  }

  if (in_anchor_index >= input_addrs.size()) {
    FFTS_LOGW("[GenTsk][PrefetchTsk][FillCtxt][node %s, type %s] In anchor %zu, the value is greater than or equal to the size of input_addrs %zu.",
              node->GetName().c_str(), node->GetType().c_str(), in_anchor_index, input_addrs.size());
    return SUCCESS;
  }
  addr_base = static_cast<uint64_t>(input_addrs[in_anchor_index]);
  return SUCCESS;
}


Status DataTaskBuilder::UpdateSrcSlotAndPfBm(domi::FftsPlusTaskDef *ffts_plus_task_def, uint32_t context_id) const {
  FFTS_LOGD("Update src slot and pf bm for context %u", context_id);
  FFTS_CHECK_NOTNULL(ffts_plus_task_def);
  domi::FftsPlusCtxDef *ctx = ffts_plus_task_def->mutable_ffts_plus_ctx(static_cast<int>(context_id));
  FFTS_CHECK_NOTNULL(ctx);
  uint32_t context_type = ctx->context_type();
  uint32_t prefetch_ctx_id = ffts_plus_task_def->ffts_plus_ctx_size() - 1;
  if (context_type == RT_CTX_TYPE_AICORE || context_type == RT_CTX_TYPE_AIV) {
    auto aic_aiv_ctx = ctx->mutable_aic_aiv_ctx();
    return AddSrcSlotAndBmToCtx(prefetch_ctx_id, aic_aiv_ctx);
  } else if (context_type == RT_CTX_TYPE_MIX_AIC || context_type == RT_CTX_TYPE_MIX_AIV) {
    auto mix_aic_aiv_ctx = ctx->mutable_mix_aic_aiv_ctx();
    return AddSrcSlotAndBmToCtx(prefetch_ctx_id, mix_aic_aiv_ctx);
  } else {
    REPORT_FFTS_ERROR("[DataTaskBuilder][UpdateSrcSlotAndPfBm] Context type %u, with ID %u, does not require prefetching.",
                      context_type, context_id);
    return FAILED;
  }
}


/*
 * Just for auto_mode and dynamic_mode
 * Manual_mode will override it.
 *
 * Just record first context_id generate by node(B and C) the reason is that you can update all window context's
 * (generate by A) succ_list according this first record.
 *
 *   For example: a_1's succ_list(5, 9), window size: 4.
 *                     A (1, 2, 3, 4)
 *                      /          \
 *                     /            \
 *             B(5, 6, 7, 8)     C(9, 10, 11, 12)
 *
 *   You can get the reset of context's succ_list generate by A:
 *                a_(1+1)'s succ_list(5+1, 9+1)
 *                a_(1+2)'s succ_list(5+2, 9+2)
 *                a_(1+3)'s succ_list(5+3, 9+3)
 */
Status DataTaskBuilder::GetSuccessorContextId(uint32_t out_anchor_index, const ge::NodePtr &node,
                                              std::vector<uint32_t> &succ_list, uint32_t &cons_cnt) {
  cons_cnt = 0;
  auto anchors = node->GetAllOutDataAnchors();
  auto output_size = anchors.size();
  if (out_anchor_index >= output_size) {
    REPORT_FFTS_ERROR("[GenTask][DataTskBuilder][GetSuccList]Output anchor index %u >= output size %zu of %s.",
                      out_anchor_index, output_size, node->GetName().c_str());
    return FAILED;
  }

  auto output_anchor = anchors.at(out_anchor_index);
  if (output_anchor == nullptr) {
    return SUCCESS;
  }
  for (const auto &peer_in_anchor : output_anchor->GetPeerInDataAnchors()) {
    FFTS_CHECK_NOTNULL(peer_in_anchor);
    auto peer_in_node = peer_in_anchor->GetOwnerNode();
    FFTS_CHECK_NOTNULL(peer_in_node);
    vector<uint32_t> peer_in_context_id;
    uint32_t ctx_id_tmp = 0;
    auto peer_op = peer_in_node->GetOpDesc();
    FFTS_CHECK_NOTNULL(peer_op);
    /* PhonyConcat is no-task op. We need to penetrate it
     * and find its successors. */
    if (IsPhonyOp(peer_op)) {
      FFTS_LOGD("Peer input op for output %d of %s is PhonyConcat %s.", peer_in_anchor->GetIdx(),
                node->GetName().c_str(), peer_op->GetName().c_str());
      for (const auto &peer_node_of_pc : peer_in_node->GetOutAllNodes()) {
        auto peer_op_of_pc = peer_node_of_pc->GetOpDesc();
        vector<uint32_t> peer_in_context_id_list;
        (void)ge::AttrUtils::GetListInt(peer_op_of_pc, kAutoCtxIdList, peer_in_context_id_list);
        if (peer_in_context_id_list.empty()) {
          FFTS_LOGI("PhonyConcat [%s]: peer operation [%s] needs successor list but it does not have a context ID.",
                    peer_op->GetName().c_str(), peer_op_of_pc->GetName().c_str());
          continue;
        }
        ctx_id_tmp = peer_in_context_id_list[0];
        FFTS_LOGD("Peer input op for PhonyConcat is %s, context id is %u.",
                  peer_op_of_pc->GetName().c_str(), ctx_id_tmp);
        peer_in_context_id.emplace_back(ctx_id_tmp);
      }
    } else {
      vector<uint32_t> peer_in_context_id_list;
      (void)ge::AttrUtils::GetListInt(peer_op, kAutoCtxIdList, peer_in_context_id_list);
      if (peer_in_context_id_list.empty()) {
        FFTS_LOGI("Node %s needs successor list but it has a successor %s which do not have a context id.",
                  node->GetName().c_str(), peer_op->GetName().c_str());
        continue;
      }
      ctx_id_tmp = peer_in_context_id_list[0];
      FFTS_LOGD("Peer input op of %s is %s, context id is %u.",
                node->GetName().c_str(), peer_op->GetName().c_str(), ctx_id_tmp);
      peer_in_context_id.emplace_back(ctx_id_tmp);
    }

    for (auto ele : peer_in_context_id) {
      succ_list.emplace_back(ele);
      cons_cnt++;
    }
    FFTS_LOGD("Total successors(%zu) for node %s is %s.", succ_list.size(), node->GetName().c_str(),
              fe::StringUtils::IntegerVecToString(succ_list).c_str());
  }
  return SUCCESS;
}

void DataTaskBuilder::SetOperation(CACHE_OPERATION operation) {
  operation_ = operation;
}

void DataTaskBuilder::SetBurstLen(int64_t burst_len) {
  burst_len_ = burst_len;
}

void DataTaskBuilder::UpdateRedundantNodes(const ge::NodePtr &node, vector<ge::NodePtr> &redundant_nodes) {
  auto op_desc = node->GetOpDesc();
  map<string, vector<ge::MemReuseInfo>> mem_reuse_infos{};
  mem_reuse_infos = op_desc->TryGetExtAttr(ge::ATTR_NAME_MEMORY_REUSE_INFO, mem_reuse_infos);
  if (mem_reuse_infos.empty()) {
    FFTS_LOGD("[GenTsk][DataTsk][MemReuse] The [node %s, type %s] has no mem_reuse_info to Redundant.",
              node->GetName().c_str(), node->GetType().c_str());
    return;
  }

  for (auto &reuse_infos : mem_reuse_infos) {
    if (reuse_infos.second.empty()) {
      continue;
    }
    for (auto &info : reuse_infos.second) {
      if (info.node == nullptr) {
        continue;
      }
      redundant_nodes.emplace_back(info.node);
    }
  }
}

Status DataTaskBuilder::UpdateSuccListWithMemReuse(const ge::NodePtr &node,
                                                   vector<ge::MemReuseInfo> &mem_reuse_infos,
                                                   domi::FftsPlusTaskDef *ffts_plus_task_def,
                                                   int &data_ctx_id,
                                                   const size_t &window_id) {
  if (mem_reuse_infos.empty()) {
    REPORT_FFTS_ERROR("[GenTsk][DataTsk][FillCtxt][node %s, type %s] mem_reuse_infos value is empty.",
                      node->GetName().c_str(), node->GetType().c_str());
    return FAILED;
  }

  auto sub_graph = node->GetOwnerComputeGraph();
  if (sub_graph == nullptr) {
    FFTS_LOGD("[GenTsk][DataTsk][FillCtxt][node %s, type %s] can't get owner compute sub graph.",
              node->GetName().c_str(), node->GetType().c_str());
    return SUCCESS;
  }

  vector<ge::NodePtr> redundant_nodes;
  for (auto &reuse_info : mem_reuse_infos) {
    if (reuse_info.node == nullptr) {
      FFTS_LOGD("[GenTsk][DataTsk][FillCtxt][node %s, type %s] mem reuse info has node nullptr.",
                node->GetName().c_str(), node->GetType().c_str());
      continue;
    }
    auto op_desc = reuse_info.node->GetOpDesc();
    FFTS_CHECK_NOTNULL(op_desc);
    if (IsPhonyOp(op_desc)) {
      FFTS_LOGD("[GenTsk][DataTsk][FillCtxt][node %s, type %s] mem reuse node is phonyop.",
                node->GetName().c_str(), node->GetType().c_str());
      continue;
    }

    auto ai_graph = reuse_info.node->GetOwnerComputeGraph();
    if (ai_graph == nullptr) {
      FFTS_LOGD("[GenTsk][DataTsk][FillCtxt][node %s, type %s] can't get owner compute graph.",
                node->GetName().c_str(), node->GetType().c_str());
      continue;
    }
    // check is in the same ffts+ sub graph or not
    if (ai_graph != sub_graph) {
      FFTS_LOGD("[GenTsk][DataTsk][FillCtxt] nodes [%s, %s] and [%s, %s] are in different compute graphs.",
                node->GetName().c_str(), node->GetType().c_str(), reuse_info.node->GetName().c_str(),
                reuse_info.node->GetType().c_str());
      continue;
    }
    if (find(redundant_nodes.begin(), redundant_nodes.end(), reuse_info.node) != redundant_nodes.end()) {
      continue;
    }
    redundant_nodes.emplace_back(reuse_info.node);
    UpdateRedundantNodes(reuse_info.node, redundant_nodes);

    FFTS_LOGI("[GenTsk][DataTsk][FillCtxt][node %s, type %s] by [node %s, type %s] reuse mem.",
              node->GetName().c_str(), node->GetType().c_str(), reuse_info.node->GetName().c_str(),
              reuse_info.node->GetType().c_str());
    UpdateInvalidCtxWithMemReuse(reuse_info.node, data_ctx_id, window_id, ffts_plus_task_def);
  }
  return SUCCESS;
}

Status DataTaskBuilder::GenInvalidSuccListWithMemReuse(const ge::NodePtr &node, size_t out_anchor_index,
                                                       domi::FftsPlusTaskDef *ffts_plus_task_def,
                                                       int &data_ctx_id, const size_t &window_id) {
  FFTS_LOGD("[GenTsk][DataTsk][MemReuse] Node %s of type %s is ready to get out_anchor_index: %zu mem reuse info.",
            node->GetName().c_str(), node->GetType().c_str(), out_anchor_index);

  auto op_desc = node->GetOpDesc();
  map<string, vector<ge::MemReuseInfo>> mem_reuse_infos{};
  mem_reuse_infos = op_desc->TryGetExtAttr(ge::ATTR_NAME_MEMORY_REUSE_INFO, mem_reuse_infos);
  if (mem_reuse_infos.empty()) {
    FFTS_LOGD("[GenTsk][DataTsk][MemReuse][node %s, type %s] has no mem_reuse_info.",
              node->GetName().c_str(), node->GetType().c_str());
    return SUCCESS;
  }

  auto anchors = node->GetAllOutDataAnchors();
  auto output_size = anchors.size();
  if (out_anchor_index >= output_size) {
    REPORT_FFTS_ERROR("[GenTask][DataTskBuilder][MemReuse] Output anchor index %zu >= output size %zu of %s.",
                      out_anchor_index, output_size, node->GetName().c_str());
    return FAILED;
  }

  auto output_anchor = anchors.at(out_anchor_index);
  FFTS_CHECK(output_anchor == nullptr, FFTS_LOGD("The output_anchor is a nullptr."), return SUCCESS);
  bool is_exist_successor = false;
  for (const auto &peer_in_anchor : output_anchor->GetPeerInDataAnchors()) {
    if (peer_in_anchor == nullptr) {
      continue;
    }
    auto peer_in_node = peer_in_anchor->GetOwnerNode();
    if (peer_in_node == nullptr) {
      continue;
    }
    is_exist_successor = true;
    break;
  }

  if (is_exist_successor != true) {
    FFTS_LOGD("[GenTsk][DataTsk][MemReuse][node %s, type %s]'s out anchor index: %zu has no successor.",
              node->GetName().c_str(), node->GetType().c_str(), out_anchor_index);
    return SUCCESS;
  }

  std::string mem_info_key = "output";
  mem_info_key.append(std::to_string(out_anchor_index));

  if (mem_reuse_infos.find(mem_info_key) == mem_reuse_infos.end() || mem_reuse_infos[mem_info_key].empty()) {
    FFTS_LOGD("[GenTsk][DataTsk][MemReuse] Node %s, type %s, has no reuse info for out anchor index: %zu.",
              node->GetName().c_str(), node->GetType().c_str(), out_anchor_index);
    return SUCCESS;
  }

  if (UpdateSuccListWithMemReuse(node, mem_reuse_infos[mem_info_key], ffts_plus_task_def,
                                 data_ctx_id, window_id) != SUCCESS) {
    return FAILED;
  }

  return SUCCESS;
}
}  // namespace ffts
