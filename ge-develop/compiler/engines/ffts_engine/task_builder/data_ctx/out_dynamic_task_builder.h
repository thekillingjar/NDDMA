/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FFTS_ENGINE_TASK_BUILDER_DATA_CTX_OUT_DYNAMIC_TASK_BUILDER_H_
#define FFTS_ENGINE_TASK_BUILDER_DATA_CTX_OUT_DYNAMIC_TASK_BUILDER_H_
#include "task_builder/mode/data_task_builder.h"

namespace ffts {
class OutDynamicTaskBuilder : public DataTaskBuilder {
 public:
  OutDynamicTaskBuilder();
  explicit OutDynamicTaskBuilder(CACHE_OPERATION operation);
  ~OutDynamicTaskBuilder() override;

  Status FillDynamicDataCtx(const size_t &out_anchor_index, const ge::NodePtr &node,
                            domi::FftsPlusTaskDef *ffts_plus_task_def, const rtFftsPlusContextType_t &context_type,
                            const vector<uint32_t> &context_id_list) override;

  void RecordNewCtxIdList(vector<uint32_t> &data_ctx_id_list, vector<vector<int64_t>> &write_back_invalid_ctx_id_list,
                          domi::FftsPlusTaskDef *ffts_plus_task_def, const rtFftsPlusContextType_t &context_type,
                          uint32_t const_cnt) const;

  OutDynamicTaskBuilder(const OutDynamicTaskBuilder &builder) = delete;
  OutDynamicTaskBuilder &operator=(const OutDynamicTaskBuilder &builder) = delete;

};

}  // namespace ffts
#endif  // FFTS_ENGINE_TASK_BUILDER_DATA_CTX_OUT_DYNAMIC_TASK_BUILDER_H_
