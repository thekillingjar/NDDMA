/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FFTS_ENGINE_TASK_BUILDER_DATA_CTX_PREFETCH_MANUAL_TASK_BUILDER_H_
#define FFTS_ENGINE_TASK_BUILDER_DATA_CTX_PREFETCH_MANUAL_TASK_BUILDER_H_
#include "task_builder/mode/data_task_builder.h"

namespace ffts {
class PrefetchTaskBuilder : public DataTaskBuilder {
 public:
  PrefetchTaskBuilder();
  ~PrefetchTaskBuilder() override;

  Status FillManualDataCtx(size_t in_anchor_index, const ge::NodePtr &node, const DataContextParam &param,
                           domi::FftsPlusTaskDef *ffts_plus_task_def, domi::FftsPlusDataCtxDef *data_ctx_def) override;

  PrefetchTaskBuilder(const PrefetchTaskBuilder &builder) = delete;
  PrefetchTaskBuilder &operator=(const PrefetchTaskBuilder &builder) = delete;
};

}  // namespace ffts
#endif // FFTS_ENGINE_TASK_BUILDER_DATA_CTX_PREFETCH_MANUAL_TASK_BUILDER_H_
