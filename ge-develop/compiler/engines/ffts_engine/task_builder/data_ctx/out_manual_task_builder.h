/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FFTS_ENGINE_TASK_BUILDER_DATA_CTX_OUT_MANUAL_TASK_BUILDER_H_
#define FFTS_ENGINE_TASK_BUILDER_DATA_CTX_OUT_MANUAL_TASK_BUILDER_H_
#include "task_builder/mode/data_task_builder.h"

namespace ffts {
class OutTaskBuilder : public DataTaskBuilder {
 public:
  OutTaskBuilder();
  explicit OutTaskBuilder(CACHE_OPERATION operation);
  ~OutTaskBuilder() override;

  Status FillManualDataCtx(size_t out_anchor_index, const ge::NodePtr &node, const DataContextParam &param,
                           domi::FftsPlusTaskDef *ffts_plus_task_def, domi::FftsPlusDataCtxDef *data_ctx_def) override;
  Status UpdateInvalidCtxWithMemReuse(const ge::NodePtr &node, int &data_ctx_id, const size_t &window_id,
                                      domi::FftsPlusTaskDef *ffts_plus_task_def) override;
  Status GetSuccessorContextId(uint32_t out_anchor_index, const ge::NodePtr &node,
                               std::vector<uint32_t> &succ_list, uint32_t &cons_cnt) override;

  Status GetSuccessorContextIdSpecial(uint32_t out_anchor_index,
                                      const ge::NodePtr &node, std::vector<uint32_t> &succ_list,
                                      uint32_t &cons_cnt);

  Status UptSuccListOfRelatedNode(const ge::NodePtr &node, const std::vector<uint32_t> &succ_list,
                                  int32_t &data_ctx_id,
                                  domi::FftsPlusTaskDef *ffts_plus_task_def) const;

  OutTaskBuilder(const OutTaskBuilder &builder) = delete;
  OutTaskBuilder &operator=(const OutTaskBuilder &builder) = delete;

 private:
  Status GetPeerInputContextId(const ge::NodePtr &node, const ge::NodePtr &peer_in_node,
                               vector<uint32_t> &peer_in_context_id);
  bool CheckManualDataAttr(const ge::OpDescPtr &op_desc, vector<int64_t> &output_addrs);
};

}  // namespace ffts
#endif // FFTS_ENGINE_TASK_BUILDER_DATA_CTX_OUT_MANUAL_TASK_BUILDER_H_
