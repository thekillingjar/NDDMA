/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_UTILS_FFTS_PLUS_TASK_BUILDER_MIX_l2_TASK_BUILDER_H_
#define FUSION_ENGINE_UTILS_FFTS_PLUS_TASK_BUILDER_MIX_l2_TASK_BUILDER_H_
#include <map>
#include <memory>
#include <vector>
#include "proto/task.pb.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "task_builder/fftsplus_task_builder.h"
namespace ffts {
using FftsPlusCtxDefPtr = std::shared_ptr<domi::FftsPlusCtxDef>;
class MixL2TaskBuilder : public FFTSPlusTaskBuilder {
 public:
  MixL2TaskBuilder();
  ~MixL2TaskBuilder() override;

  /*
   * @ingroup fe
   * @brief   Generate tasks
   * @param   [in] node Node of compute graph
   * @param   [in] context Context for generate tasks
   * @param   [out] task_defs Save the generated tasks.
   * @return  SUCCESS or FAILED
   */
  Status GenContextDef(const ge::NodePtr &node, domi::FftsPlusTaskDef *ffts_plus_task_def) override;
  Status FillContextData(const ge::NodePtr &node, const domi::FftsPlusMixAicAivCtxDef *src_ctx_def,
                         domi::FftsPlusMixAicAivCtxDef *dst_ctx_def) const;
 private:
  MixL2TaskBuilder(const MixL2TaskBuilder &builder) = delete;
  MixL2TaskBuilder &operator=(const MixL2TaskBuilder &builder) = delete;

  Status AddAdditionalArgs(ge::OpDescPtr &op_desc, domi::FftsPlusTaskDef *ffts_plus_task_def,
                           const size_t &ctx_num) const;
};
}
#endif
