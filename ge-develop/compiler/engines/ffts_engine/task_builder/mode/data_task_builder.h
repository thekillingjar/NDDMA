/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FFTS_ENGINE_TASK_BUILDER_MODE_DATA_TASK_BUILDER_H_
#define FFTS_ENGINE_TASK_BUILDER_MODE_DATA_TASK_BUILDER_H_

#include "task_builder/fftsplus_task_builder.h"
#include "inc/memory_slice.h"
#include "inc/ffts_type.h"
#include "inc/ffts_utils.h"
#include "graph/debug/ge_attr_define.h"
namespace ffts {
using ContextType = rtFftsPlusContextType_t;

class DataTaskBuilder : public FFTSPlusTaskBuilder {
 public:
  DataTaskBuilder();
  explicit DataTaskBuilder(CACHE_OPERATION operation);
  ~DataTaskBuilder() override;

  /*
   * @ingroup ffts
   * @brief   Generate tasks
   * @param   [in] node Node of compute graph
   * @param   [in] context Context for generate tasks
   * @param   [out] task_defs Save the generated tasks.
   * @return  SUCCESS or FAILED
   */
  Status GenManualDataCtxDef(const ge::NodePtr &node, domi::FftsPlusTaskDef *ffts_plus_task_def);

  virtual Status FillManualDataCtx(size_t out_anchor_index, const ge::NodePtr &node, const DataContextParam &param,
                                   domi::FftsPlusTaskDef *ffts_plus_task_def, domi::FftsPlusDataCtxDef *data_ctx_def) {
    (void)out_anchor_index;
    (void)node;
    (void)param;
    (void)ffts_plus_task_def;
    (void)data_ctx_def;
    return SUCCESS;
  };

  Status GenAutoDataCtxDef(const ge::NodePtr &node, domi::FftsPlusTaskDef *ffts_plus_task_def);

  virtual Status FillAutoDataCtx(size_t out_anchor_index, const ge::NodePtr &node,
                                 const std::vector<DataContextParam> &params, domi::FftsPlusTaskDef *ffts_plus_task_def,
                                 domi::FftsPlusDataCtxDef *data_ctx_def, const size_t &window_id) {
    (void)out_anchor_index;
    (void)node;
    (void)params;
    (void)ffts_plus_task_def;
    (void)data_ctx_def;
    (void)window_id;
    return SUCCESS;
  };

  virtual Status UpdateInvalidCtxWithMemReuse(const ge::NodePtr &node, int &data_ctx_id, const size_t &window_id,
                                              domi::FftsPlusTaskDef *ffts_plus_task_def) {
    (void)node;
    (void)data_ctx_id;
    (void)window_id;
    (void)ffts_plus_task_def;
    return SUCCESS;
  }
  Status GenDynamicDataCtxDef(const ge::NodePtr &node, domi::FftsPlusTaskDef *ffts_plus_task_def);

  virtual Status FillDynamicDataCtx(const size_t &out_anchor_index, const ge::NodePtr &node,
                                    domi::FftsPlusTaskDef *ffts_plus_task_def,
                                    const rtFftsPlusContextType_t &context_type,
                                    const vector<uint32_t> &context_id_list) {
    (void)out_anchor_index;
    (void)node;
    (void)ffts_plus_task_def;
    (void)context_type;
    (void)context_id_list;
    return SUCCESS;
  };

  void SetOperation(CACHE_OPERATION operation);

  void SetBurstLen(int64_t burst_len);

  DataTaskBuilder(const DataTaskBuilder &builder) = delete;
  DataTaskBuilder &operator=(const DataTaskBuilder &builder) = delete;

 protected:
  /*
   * Prefetch, invalid and write_back will use the following method.
   */
  std::vector<int> GetIndices(const ge::NodePtr &node) const;

  bool ExceedMaxCtxNum(size_t curr_num, size_t pending_num) const;

  void FillAutoThreadingParam(const vector<DataContextParam> &params, domi::FftsPlusDataCtxDef *data_ctx_def,
                              const uint32_t &slice_num) const;

  void FillManualThreadingParam(const DataContextParam &param, domi::FftsPlusDataCtxDef *data_ctx_def) const;


  /*
   * Prefetch will use the following method.
   */
  Status GetAddrBase(size_t in_anchor_index, const ge::NodePtr &node, uint64_t &addr_base) const;  // manual and auto

  Status UpdateSrcSlotAndPfBm(domi::FftsPlusTaskDef *ffts_plus_task_def, uint32_t context_id) const;

  template<typename T>
  Status AddSrcSlotAndBmToCtx(uint32_t prefetch_ctx_id, T *ctx) const {
    FFTS_CHECK_NOTNULL(ctx);
    size_t src_slot_size = static_cast<size_t>(ctx->src_slot_size());
    if (src_slot_size >= kMaxPretchNum) {
      REPORT_FFTS_ERROR("[DataTaskBuilder][UpdateSrcSlotAndPfBm] Already reach the maximum size of"
                        "prefetch bitmap of aic/aiv context.");
      return FAILED;
    }
    ctx->add_src_slot(prefetch_ctx_id);

    uint32_t enable_bm = ctx->prefetch_enable_bitmap();
    uint32_t once_bm = ctx->prefetch_once_bitmap();
    SetBitOne(src_slot_size, enable_bm);
    SetBitOne(src_slot_size, once_bm);
    ctx->set_prefetch_enable_bitmap(enable_bm);
    ctx->set_prefetch_once_bitmap(once_bm);
    ctx->set_prefetch_config(enable_bm | (once_bm << 4));
    return SUCCESS;
  }
  void UpdateRedundantNodes(const ge::NodePtr &node, vector<ge::NodePtr> &redundant_nodes);

  Status UpdateSuccListWithMemReuse(const ge::NodePtr &node, vector<ge::MemReuseInfo> &mem_reuse_infos,
                                    domi::FftsPlusTaskDef *ffts_plus_task_def,
                                    int &data_ctx_id, const size_t &window_id);
  Status GenInvalidSuccListWithMemReuse(const ge::NodePtr &node, size_t out_anchor_index,
                                        domi::FftsPlusTaskDef *ffts_plus_task_def,
                                        int &data_ctx_id, const size_t &window_id);
  /*
   * Invalid and write_back will use the following method.
   * Manual mode need override this method.
   *
   * Prefetch data context does not need to know successors.
   * succ_list is an output parameter. It contains all context ids and labeled
   * context ids if the peer node has more than 26 successors.
   * cons_cnt is the total of all successors.
   */
  virtual Status GetSuccessorContextId(uint32_t out_anchor_index, const ge::NodePtr &node,
                                       std::vector<uint32_t> &succ_list, uint32_t &cons_cnt);

  CACHE_OPERATION operation_;

  int64_t burst_len_ = 0;
};
}  // namespace ffts
#endif // FFTS_ENGINE_TASK_BUILDER_MODE_DATA_TASK_BUILDER_H_
