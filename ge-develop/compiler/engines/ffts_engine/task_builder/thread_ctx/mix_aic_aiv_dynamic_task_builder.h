/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINE_FFTSENG_TASK_BUILDER_THREAD_CTX_MIX_AIC_AIV_DYNAMIC_TASK_BUILDER_H_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINE_FFTSENG_TASK_BUILDER_THREAD_CTX_MIX_AIC_AIV_DYNAMIC_TASK_BUILDER_H_
#include "task_builder/fftsplus_task_builder.h"

namespace ffts {
class MixAICAIVDynamicTaskBuilder : public FFTSPlusTaskBuilder {
 public:
  MixAICAIVDynamicTaskBuilder();
  ~MixAICAIVDynamicTaskBuilder() override;

  Status GenContextDef(const ge::NodePtr &node, domi::FftsPlusTaskDef *ffts_plus_task_def) override;
 private:
  MixAICAIVDynamicTaskBuilder(const MixAICAIVDynamicTaskBuilder &builder) = delete;
  MixAICAIVDynamicTaskBuilder &operator=(const MixAICAIVDynamicTaskBuilder &builder) = delete;

  Status AddAdditionalArgs(ge::OpDescPtr &op_desc, domi::FftsPlusTaskDef *ffts_plus_task_def,
                           vector<uint32_t> &auto_ctx_id_list) const;
};
}  // namespace ffts
#endif
