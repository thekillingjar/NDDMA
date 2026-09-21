/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FFTS_ENGINE_TASK_BUILDER_FFTSPLUS_TASK_BUILDER_H_
#define FFTS_ENGINE_TASK_BUILDER_FFTSPLUS_TASK_BUILDER_H_
#include <map>
#include <memory>
#include <vector>
#include "securec.h"
#include "graph/compute_graph.h"
#include "graph/utils/graph_utils.h"
#include "proto/task.pb.h"
#include "inc/ffts_type.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "runtime/rt.h"
#include "common/sgt_slice_type.h"
#include "inc/ffts_log.h"

namespace ffts {

using FftsPlusComCtx_t = struct tagFftsPlusComCtx {
    uint16_t contextType;
    uint8_t successorNum;
    uint32_t pred_cnt;
    vector<uint32_t> succ_list;
};

using FftsPlusCtxDefPtr = std::shared_ptr<domi::FftsPlusCtxDef>;

class FFTSPlusTaskBuilder {
 public:
  FFTSPlusTaskBuilder();
  virtual ~FFTSPlusTaskBuilder();

  Status GenFftsPlusTaskCommonInfo(const ge::NodePtr &node, vector<FftsPlusComCtx_t> &sub_ffts_plus_context) const;


  Status GenFftsPlusDependencyInfo(const ge::NodePtr &node, vector<FftsPlusComCtx_t> &sub_ffts_plus_context) const;
  Status FillProducersInfoForLabelX(const ge::NodePtr &node,
                                    FftsPlusComCtx_t &ffts_plus_context,
                                    uint32_t &pred_cnt,
                                    const std::string &node_type,
                                    const ge::OpDescPtr &op_desc) const;
  void FillManualCustomersInfoForLabelX(const ge::NodePtr &node,
                                        FftsPlusComCtx_t &sub_ffts_plus_context_elem,
                                        uint32_t &jumplabel_context_id) const;
  void FillManualCustomersInfoForLabelSet(const ge::NodePtr &node,
                                          FftsPlusComCtx_t &sub_ffts_plus_context_elem) const;
  void FillManualCustomersInfoForLabelSwitch(const ge::NodePtr &node,
                                             FftsPlusComCtx_t &sub_ffts_plus_context_elem) const;
  void FillManualCustomersInfoForLabelGoto(const ge::NodePtr &node,
                                           FftsPlusComCtx_t &sub_ffts_plus_context_elem,
                                           uint32_t &jumplabel_context_id) const;
  Status FillProducersInfo(const ge::NodePtr &node, FftsPlusComCtx_t &ffts_plus_context) const;
  Status FillCommonProducersInfo(const ge::NodePtr &node, uint32_t &pred_cnt,
                                 FftsPlusComCtx_t &ffts_plus_context) const;
  void FillSingleProducersInfo(const ge::NodePtr &pre_node, uint32_t &pred_cnt, uint32_t recurise_cnt) const;
  void JudgeAutoStratCtxIdListInfo(const ge::NodePtr &node, uint32_t &pred_cnt) const;
  Status FillManualCustomersInfo(const ge::NodePtr &node, FftsPlusComCtx_t &sub_ffts_plus_context_elem) const;
  Status FillCustomersInfo(const ge::NodePtr &node, FftsPlusComCtx_t &sub_ffts_plus_context_elem,
                           vector<FftsPlusComCtx_t> &sub_ffts_plus_context) const;
  bool GetJumpLabelSetContextid(const ge::NodePtr &node,
                                uint32_t &jumplabel_context_id,
                                bool &has_jumpnode) const;
  void FillManualCustomersInfoCommon(const ge::NodePtr &up_node,
                                     ge::OpDescPtr up_op_desc,
                                     FftsPlusComCtx_t &sub_ffts_plus_context_elem) const;
  /*
   * @ingroup ffts
   * @brief   Generate tasks
   * @param   [in] node Node of compute graph
   * @param   [in] context Context for generate tasks
   * @param   [out] task_defs Save the generated tasks.
   * @return  SUCCESS or FAILED
   */
  Status GenerateTaskDef(const ge::NodePtr &node, domi::FftsPlusTaskDef *ffts_plus_task_def);

  virtual Status GenContextDef(const ge::NodePtr &node, domi::FftsPlusTaskDef *ffts_plus_task_def) {
    (void)node;
    (void)ffts_plus_task_def;
    return SUCCESS;
  };

  Status UpdateSuccList(uint32_t succ_id, uint32_t curr_id, domi::FftsPlusTaskDef *ffts_plus_task_def,
                        size_t thread_id = 0, bool is_auto = false) const;

  Status UpdatePreCnt(uint32_t curr_id, domi::FftsPlusTaskDef *ffts_plus_task_def, const int32_t gradient) const;

  uint32_t GetPreCnt(uint32_t curr_id, domi::FftsPlusTaskDef *ffts_plus_task_def) const;

  Status ReplaceSuccList(uint32_t succ_id, uint32_t new_succ_id, uint32_t curr_id,
                         domi::FftsPlusTaskDef *ffts_plus_task_def) const;
  /* Create a label context and move the last succ id to this context. */
  template <typename T>
  static Status GenerateNewLabelCtx(domi::FftsPlusTaskDef *ffts_plus_task_def,
                                    uint32_t last_succ_id,
                                    T *pred_ctx,
                                    domi::FftsPlusLabelCtxDef **new_label) {
    uint32_t new_ctx_id = ffts_plus_task_def->ffts_plus_ctx_size();
    FFTS_LOGD("Generate a new label context. last_succ_id %u, new context id %u.", last_succ_id, new_ctx_id);
    domi::FftsPlusCtxDef* new_ctx = ffts_plus_task_def->add_ffts_plus_ctx();
    FFTS_CHECK_NOTNULL(new_ctx);
    new_ctx->set_context_type(RT_CTX_TYPE_LABEL);
    *new_label = new_ctx->mutable_label_ctx();
    FFTS_CHECK_NOTNULL(*new_label);
    /* We just create a new label. The caller of this function will put the
     * pending context id into the second position of this new label. */
    (*new_label)->add_successor_list(last_succ_id);
    (*new_label)->set_successor_num(1);
    (*new_label)->set_pred_cnt(1);
    (*new_label)->set_pred_cnt_init(1);

    pred_ctx->set_successor_list(RT_CTX_SUCCESSOR_NUM - 1, new_ctx_id);
    return SUCCESS;
  }

  static Status GetFirstAvailableLabel(domi::FftsPlusTaskDef *ffts_plus_task_def,
                                       domi::FftsPlusLabelCtxDef *pred_label_ctx,
                                       domi::FftsPlusLabelCtxDef **avl_label_context,
                                       uint32_t &recursion_count);

  /* If the ctx have 25 or less successor, we just add the successsor's id into it's
   * successor list.
   * And if the ctx have exactly 26 successors, we need to :
   * If the 26th context is label context:
   *   1. Get the next label context and check whether the next
   *      one is also full. If full, keep searching util we get a
   *      label context with less than 26 successor.
   *   2. put the succ_id into the final label context.
   *
   *
   * Else if the 26th is a normal context:
   *   1. Generate a new label context into whole sqe.
   *   2. Move the 26th successor id into the first
   *      position of the new label context.
   *   3. Put the succ_id into the second position of the new
   *      label context.
   *   4. Put the context id of the new label context(which
   *      is the size of all context - 1) into the 26th position of
   *      the normal context. */
  template <typename T>
  static Status AddOneId(domi::FftsPlusTaskDef *ffts_plus_task_def, uint32_t succ_id, T *ctx,
                         size_t thread_id, bool is_auto) {
    uint32_t succ_num = ctx->successor_num();
    if (succ_num == RT_CTX_SUCCESSOR_NUM) {
      uint32_t last_succ_id = ctx->successor_list(RT_CTX_SUCCESSOR_NUM - 1);
      uint32_t ctx_size = ffts_plus_task_def->ffts_plus_ctx_size();
      if (last_succ_id >= ctx_size) {
        REPORT_FFTS_ERROR("[FFTSPLUS][AddOneId] last_succ_id %u, ctx_size:%u", last_succ_id, ctx_size);
        return FAILED;
      }
      domi::FftsPlusCtxDef* last_succ_ctx =
          ffts_plus_task_def->mutable_ffts_plus_ctx(static_cast<int>(last_succ_id));
      FFTS_CHECK_NOTNULL(last_succ_ctx);
      domi::FftsPlusLabelCtxDef* avl_label_ctx = nullptr;
      if (last_succ_ctx->context_type() == RT_CTX_TYPE_LABEL) {
        FFTS_LOGD("last context is label, keep seaching its succesorrs.");
        domi::FftsPlusLabelCtxDef* pre_label = last_succ_ctx->mutable_label_ctx();
        uint32_t recursion_count = 0;
        if (GetFirstAvailableLabel(ffts_plus_task_def, pre_label, &avl_label_ctx, recursion_count) != SUCCESS ||
            avl_label_ctx == nullptr) {
          REPORT_FFTS_ERROR("[FFTSPLUS][AddOneId] Cannot find any available label context for succ_id %u.", succ_id);
          return FAILED;
        }
      } else {
        /* Just generate a label context and move the last successor into the
         * new generated label and return it. */
        if (GenerateNewLabelCtx(ffts_plus_task_def, last_succ_id, ctx, &avl_label_ctx) != SUCCESS) {
          return FAILED;
        }
      }
      FFTS_CHECK_NOTNULL(avl_label_ctx);
      if (is_auto) {
        avl_label_ctx->set_thread_id(thread_id);
        avl_label_ctx->set_aten(kAutoMode);
        FFTS_LOGD("Set auto label thread_id[%u].", thread_id);
      }
      FFTS_LOGD("Add one successor %u.", succ_id);
      avl_label_ctx->add_successor_list(succ_id);
      succ_num = static_cast<uint32_t>(avl_label_ctx->successor_list_size());
      avl_label_ctx->set_successor_num(succ_num);
    } else {
      ++succ_num;
      FFTS_LOGD("Add one successor %u. successor num %u", succ_id, succ_num);
      ctx->set_successor_num(succ_num);
      ctx->add_successor_list(succ_id);
    }
    return SUCCESS;
  }

  template <typename T>
  static Status ReplaceOneId(domi::FftsPlusTaskDef *ffts_plus_task_def, const uint32_t succ_id,
                             const uint32_t new_succ_id, T *ctx) {
    FFTS_CHECK_NOTNULL(ctx);
    FFTS_LOGD("try to replace one successor[%u] to [%u] from ctx", succ_id, new_succ_id);
    for (int32_t index = 0; index < ctx->successor_list_size(); ++index) {
      uint32_t id = ctx->successor_list(index);
      if (id == succ_id) {
        ctx->set_successor_list(index, new_succ_id);
        FFTS_LOGD("success replace one successor[%u] to [%u] from ctx", succ_id, new_succ_id);
        return SUCCESS;
      }
    }
    uint32_t succ_num = ctx->successor_num();
    if (succ_num == RT_CTX_SUCCESSOR_NUM) {
      uint32_t last_succ_id = ctx->successor_list(RT_CTX_SUCCESSOR_NUM - 1);
      domi::FftsPlusCtxDef* last_succ_ctx = ffts_plus_task_def->mutable_ffts_plus_ctx(static_cast<int>(last_succ_id));
      FFTS_CHECK_NOTNULL(last_succ_ctx);
      if (last_succ_ctx->context_type() == RT_CTX_TYPE_LABEL) {
        return ReplaceOneId(ffts_plus_task_def, succ_id, new_succ_id, last_succ_ctx->mutable_label_ctx());
      }
    }
    return FAILED;
  }

  template <typename T>
  static uint32_t GetCtxPredCnt(T *ctx) {
    return ctx->pred_cnt();
  }

  template <typename T>
  static uint32_t GetDataPredCnt(T *ctx) {
    return ctx->cnt();
  }

  template <typename T>
  static Status UpdateCtxPredCnt(T *ctx, const int32_t gradient) {
    FFTS_CHECK_NOTNULL(ctx);
    uint32_t pred_cnt = ctx->pred_cnt();
    int32_t new_pred_cnt = static_cast<int32_t>(pred_cnt) + gradient;
    if (new_pred_cnt < 0) {
      FFTS_LOGE("Update pred_cnt from [%u] to [%d] failed.", pred_cnt, new_pred_cnt);
      return FAILED;
    }
    FFTS_LOGD("Update pred_cnt from [%u] to [%d].", pred_cnt, new_pred_cnt);
    ctx->set_pred_cnt(static_cast<uint32_t>(new_pred_cnt));
    ctx->set_pred_cnt_init(static_cast<uint32_t>(new_pred_cnt));
    return SUCCESS;
  }

  template <typename T>
  static Status UpdateDataPredCnt(T *ctx, const int32_t gradient) {
    FFTS_CHECK_NOTNULL(ctx);
    uint32_t pred_cnt = ctx->cnt();
    int32_t new_pred_cnt = static_cast<int32_t>(pred_cnt) + gradient;
    if (new_pred_cnt < 0) {
      FFTS_LOGE("Update pred_cnt from [%u] to [%d] failed.", pred_cnt, new_pred_cnt);
      return FAILED;
    }
    ctx->set_cnt(static_cast<uint32_t>(new_pred_cnt));
    ctx->set_cnt_init(static_cast<uint32_t>(new_pred_cnt));
    return SUCCESS;
  }

  static Status ClearLabelList(domi::FftsPlusTaskDef* ffts_plus_task_def, const vector<uint32_t> &label_list,
                               size_t iLabel) {
    for (size_t i = iLabel; i < label_list.size(); i++) {
      if (label_list[i] >= static_cast<uint32_t>(ffts_plus_task_def->ffts_plus_ctx_size())) {
        FFTS_LOGD("Label list ctxid %u bigger then ffts ctx size.", label_list[i]);
        return FAILED;
      }
      domi::FftsPlusCtxDef *ffts_plus_ctx = ffts_plus_task_def->mutable_ffts_plus_ctx(static_cast<int>(label_list[i]));
      FFTS_CHECK_NOTNULL(ffts_plus_ctx);
      if (ffts_plus_ctx->context_type() != RT_CTX_TYPE_LABEL) {
        FFTS_LOGD("Label list ctxid %u is not label type.", label_list[i]);
        return FAILED;
      }
      domi::FftsPlusLabelCtxDef *in_label_ctx_def = ffts_plus_ctx->mutable_label_ctx();
      FFTS_CHECK_NOTNULL(in_label_ctx_def);
      in_label_ctx_def->set_pred_cnt(0);
      in_label_ctx_def->set_pred_cnt_init(0);
      in_label_ctx_def->set_successor_num(0);
      in_label_ctx_def->clear_successor_list();
    }
    return SUCCESS;
  }

  template <typename T>
  static Status UpdateIncludeLabelSuccList(domi::FftsPlusTaskDef* ffts_plus_task_def, T *ctx,
                                           const vector<uint32_t> &reserve_ctx_list,
                                           const vector<uint32_t> &label_list, size_t &iLabel) {
    ctx->set_successor_num(RT_CTX_SUCCESSOR_NUM);
    ctx->clear_successor_list();
    FFTS_CHECK(label_list.empty(), FFTS_LOGD("Label list is empty."), return FAILED);
    size_t label_id = RT_CTX_SUCCESSOR_NUM - 1;
    for (size_t i = 0; i < label_id; i++) {
      ctx->add_successor_list(reserve_ctx_list[i]);
    }
    ctx->add_successor_list(label_list[0]);

    if (label_list[0] >= static_cast<uint32_t>(ffts_plus_task_def->ffts_plus_ctx_size())) {
      FFTS_LOGD("Label list 0 ctxid %u bigger then ffts ctx size.", label_list[0]);
      return FAILED;
    }

    domi::FftsPlusCtxDef *ffts_plus_ctx = ffts_plus_task_def->mutable_ffts_plus_ctx(static_cast<int>(label_list[0]));
    FFTS_CHECK_NOTNULL(ffts_plus_ctx);
    if (ffts_plus_ctx->context_type() != RT_CTX_TYPE_LABEL) {
      FFTS_LOGD("Label list 0 ctxid %u is not label type.", label_list[0]);
      return FAILED;
    }
    domi::FftsPlusLabelCtxDef *in_label_ctx_def = ffts_plus_ctx->mutable_label_ctx();
    in_label_ctx_def->clear_successor_list();
    in_label_ctx_def->set_successor_num(0);
    for (size_t i = label_id; i < reserve_ctx_list.size(); i++) {
      FFTS_LOGD("i: %zu, label_id: %zu, iLabel: %zu.", i, label_id, iLabel);
      if ((i % label_id == 0) && (i != label_id) && (i != (reserve_ctx_list.size() - 1))) {
        iLabel++;
        if (iLabel >= label_list.size() ||
            label_list[iLabel] >= static_cast<uint32_t>(ffts_plus_task_def->ffts_plus_ctx_size())) {
          FFTS_LOGD("Label list %zu is bigger than label_list size %zu, or ctxid is bigger than fftsctx size.",
                    iLabel, label_list.size());
          return FAILED;
        }
        in_label_ctx_def->add_successor_list(label_list[iLabel]);
        in_label_ctx_def->set_successor_num(in_label_ctx_def->successor_num() + 1);
        ffts_plus_ctx = ffts_plus_task_def->mutable_ffts_plus_ctx(static_cast<int>(label_list[iLabel]));
        FFTS_CHECK_NOTNULL(ffts_plus_ctx);
        if (ffts_plus_ctx->context_type() != RT_CTX_TYPE_LABEL) {
          FFTS_LOGD("Label list ctxid %u is not label type.", label_list[iLabel]);
          return FAILED;
        }
        FFTS_LOGD("Ctxid %u, update succlist context:%s.", label_list[iLabel], ffts_plus_ctx->DebugString().c_str());
        in_label_ctx_def = ffts_plus_ctx->mutable_label_ctx();
        in_label_ctx_def->clear_successor_list();
        in_label_ctx_def->set_successor_num(0);
      }
      FFTS_LOGD("Label index %zu, label context:%u, succlist:%u.", iLabel, label_list[iLabel], reserve_ctx_list[i]);
      in_label_ctx_def->add_successor_list(reserve_ctx_list[i]);
      in_label_ctx_def->set_successor_num(in_label_ctx_def->successor_num() + 1);
    }
    return SUCCESS;
  }

  template <typename T>
  static Status UpdateSuccList(domi::FftsPlusTaskDef* ffts_plus_task_def, T *ctx,
                               const vector<uint32_t> &reserve_ctx_list, const vector<uint32_t> &label_list) {
    FFTS_CHECK_NOTNULL(ctx);
    size_t level = (reserve_ctx_list.empty()) ? 0 : ((reserve_ctx_list.size() - 1) / (RT_CTX_SUCCESSOR_NUM - 1));
    FFTS_LOGD("UpdateSuccList level %zu.", level);
    for (size_t i = 0; i < reserve_ctx_list.size(); ++i) {
      FFTS_LOGD("Reserve_ctx_list: %u.", reserve_ctx_list[i]);
    }
    for (size_t i = 0; i < label_list.size(); i++) {
      FFTS_LOGD("label_list i%zu: %u.", i, label_list[i]);
    }
    if ((level < 1) || (reserve_ctx_list.size() == RT_CTX_SUCCESSOR_NUM)) {
      ctx->set_successor_num(static_cast<uint32_t>(reserve_ctx_list.size()));
      ctx->clear_successor_list();
      for (size_t i = 0; i < reserve_ctx_list.size(); i++) {
        ctx->add_successor_list(reserve_ctx_list[i]);
      }
      return ClearLabelList(ffts_plus_task_def, label_list, 0);
    } else {
      size_t iLabel = 0;
      if (UpdateIncludeLabelSuccList(ffts_plus_task_def, ctx, reserve_ctx_list, label_list, iLabel) != SUCCESS) {
        return FAILED;
      }
      return ClearLabelList(ffts_plus_task_def, label_list, iLabel + 1);
    }
  }

  template <typename T>
  static bool add_at_end_to_write_back_succ_list(const uint32_t &at_end_ctx_id, T *ctx,
                                                 domi::FftsPlusTaskDef *ffts_plus_task_def) {
    bool already_add = false;
    uint32_t succ_num = ctx->successor_num();
    for (size_t i = 0; i < static_cast<size_t>(succ_num); i++) {
      uint32_t cur_succ_id = ctx->successor_list(i);
      domi::FftsPlusCtxDef* cur_succ_ctx = ffts_plus_task_def->mutable_ffts_plus_ctx(static_cast<int>(cur_succ_id));
      FFTS_CHECK_NOTNULL(cur_succ_ctx);
      auto type = cur_succ_ctx->context_type();
      if (type == RT_HW_CTX_TYPE_WRITEBACK_DATA) {
        domi::FftsPlusDataCtxDef* write_back_ctx = cur_succ_ctx->mutable_data_ctx();
        write_back_ctx->add_successor_list(at_end_ctx_id);
        write_back_ctx->set_successor_num(1);
        if (already_add) {
          domi::FftsPlusCtxDef* common_ctx = ffts_plus_task_def->mutable_ffts_plus_ctx(static_cast<int>(at_end_ctx_id));
          FFTS_CHECK_NOTNULL(common_ctx);
          domi::FftsPlusAtEndCtxDef* at_end_ctx = common_ctx->mutable_at_end_ctx();
          at_end_ctx->set_pred_cnt(at_end_ctx->pred_cnt() + 1);
          at_end_ctx->set_pred_cnt_init(at_end_ctx->pred_cnt_init() + 1);
        }
        already_add = true;
      }
    }
    return already_add;
  }

  template <typename T>
  static Status set_policy_pri(uint32_t ctx_id, uint16_t policy_pri, T *ctx) {
    FFTS_CHECK_NOTNULL(ctx);
    FFTS_LOGD("Set context id %u policy pri %hu.", ctx_id, policy_pri);
    ctx->set_policy_pri(static_cast<uint32_t>(policy_pri));
    return SUCCESS;
  }

 protected:
  void FillContextData(const domi::FftsPlusMixAicAivCtxDef *aicore_ctx_def,
                       domi::FftsPlusMixAicAivCtxDef *mix_aic_aiv_ctx_def) const;

  void FillContextData(const domi::FftsPlusAicAivCtxDef *aicore_ctx_def,
                       domi::FftsPlusAicAivCtxDef *aic_aiv_ctx_def) const;

  Status FillContextData(const domi::FftsPlusAicpuCtxDef *aicpu_ctx_def_ptr,
                         domi::FftsPlusAicpuCtxDef *aicpu_ctx_def) const;
  const std::unordered_set<std::string> LABELX_NODE_TYPE = {"LabelSet", "LabelGotoEx",
                                                            "LabelGoto", "LabelSwitchByIndex"};
 private:
  FFTSPlusTaskBuilder(const FFTSPlusTaskBuilder &builder) = delete;

  FFTSPlusTaskBuilder &operator=(const FFTSPlusTaskBuilder &builder) = delete;
};

}  // namespace ffts
#endif  // FUSION_ENGINE_UTILS_TASK_BUILDER_TASK_BUILDER_H_
