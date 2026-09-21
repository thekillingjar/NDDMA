/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_FFTSENG_TASK_BUILDER_DATA_CTX_CACHE_PERSISTENT_MANUAL_TASK_BUILDER_H_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_FFTSENG_TASK_BUILDER_DATA_CTX_CACHE_PERSISTENT_MANUAL_TASK_BUILDER_H_
#include "task_builder/fftsplus_task_builder.h"
namespace ffts {
class CachePersistTaskBuilder : public FFTSPlusTaskBuilder {
 public:
  CachePersistTaskBuilder() = default;

  ~CachePersistTaskBuilder() override = default;

  /*
   * @ingroup ffts
   * @brief   Generate tasks
   * @param   [in] node Node of compute graph
   * @param   [in] context Context for generate tasks
   * @param   [out] task_defs Save the generated tasks.
   * @return  SUCCESS or FAILED
   */
  Status GenContextDef(const ge::NodePtr &node, domi::FftsPlusTaskDef *ffts_plus_task_def) override;

  Status GenContextDef(const ge::Node &node, domi::FftsPlusTaskDef *ffts_plus_task_def) const;

  CachePersistTaskBuilder(const CachePersistTaskBuilder &builder) = delete;

  CachePersistTaskBuilder &operator=(const CachePersistTaskBuilder &builder) = delete;
};
}
#endif // AIR_COMPILER_GRAPHCOMPILER_ENGINES_FFTSENG_TASK_BUILDER_DATA_CTX_CACHE_PERSISTENT_MANUAL_TASK_BUILDER_H_
