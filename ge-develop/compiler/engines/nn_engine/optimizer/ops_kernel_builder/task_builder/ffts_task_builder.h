/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_OPS_KERNEL_BUILDER_TASK_BUILDER_FFTS_TASK_BUILDER_H_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_OPS_KERNEL_BUILDER_TASK_BUILDER_FFTS_TASK_BUILDER_H_

#include "inc/ffts_type.h"
#include "common/fe_log.h"
#include "adapter/ffts_adapter/ffts_task_builder_adapter.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "graph/debug/ge_attr_define.h"

namespace fe {
using FftsPlusCtxDefPtr = std::shared_ptr<domi::FftsPlusCtxDef>;
using FftsTaskBuilderAdapterPtr = std::shared_ptr<FftsTaskBuilderAdapter>;
class FftsTaskBuilder {
 public:
  Status GenerateFFTSPlusCtx(const ge::Node &node, const ge::RunContext &context, const std::string &engine_name);

 private:
  Status GenCtxParamAndCtxType(const ge::Node &node, ffts::TaskBuilderType &ctx_type);
  void SetAutoThreadIOAddrForDataCtx(const ge::OpDescPtr &op_desc);
  void SetManualThreadIOAddrForDataCtx(const ge::OpDescPtr &op_desc);

  Status GenManualAICAIVCtxDef(const ge::OpDescPtr &op_desc, FftsPlusCtxDefPtr ctx);
  Status GenAutoAICAIVCtxDef(const ge::OpDescPtr &op_desc, FftsPlusCtxDefPtr ctx);
  Status GenManualMixAICAIVCtxDef(const ge::OpDescPtr &op_desc, FftsPlusCtxDefPtr ctx);
  Status GenDynamicAICAIVCtxDef(const ge::OpDescPtr &op_desc, FftsPlusCtxDefPtr ctx);
  Status GenAutoMixAICAIVCtxDef(const ge::OpDescPtr &op_desc, FftsPlusCtxDefPtr ctx);
  Status GenDynamicMixAICAIVCtxDef(const ge::OpDescPtr &op_desc, FftsPlusCtxDefPtr ctx);
  Status GenMixL2CtxDef(const ge::OpDescPtr &op_desc, FftsPlusCtxDefPtr ctx);

  void GenMemSetCtxDef(const ge::OpDescPtr &op_desc) const;
  Status GenAutoAtomicCtxDef(const ge::OpDescPtr &op_desc);
  void FillMixAICAIVKernelName(const ge::OpDescPtr &op_desc, domi::FftsPlusMixAicAivCtxDef *mix_aic_aiv_ctx_def) const;

  template <typename T>
  Status GenCommonAutoAICAIVCtxDef(const ge::OpDescPtr &op_desc, T *ctx);

 private:
  TaskBuilderContext context_;
  ThreadParamOffset auto_thread_param_offset_;
  TaskArgs manual_thread_param_;
  using GenCtxFunc = Status (FftsTaskBuilder::*) (const ge::OpDescPtr &, FftsPlusCtxDefPtr);
  std::map<ffts::TaskBuilderType, GenCtxFunc> gen_ctx_func_map_ = {
          {ffts::TaskBuilderType::EN_TASK_TYPE_AIC_AIV, &FftsTaskBuilder::GenManualAICAIVCtxDef},
          {ffts::TaskBuilderType::EN_TASK_TYPE_AIC_AIV_AUTO, &FftsTaskBuilder::GenAutoAICAIVCtxDef},
          {ffts::TaskBuilderType::EN_TASK_TYPE_MIX_AIC_AIV, &FftsTaskBuilder::GenManualMixAICAIVCtxDef},
          {ffts::TaskBuilderType::EN_TASK_TYPE_AIC_AIV_DYNAMIC, &FftsTaskBuilder::GenDynamicAICAIVCtxDef},
          {ffts::TaskBuilderType::EN_TASK_TYPE_MIX_AIC_AIV_AUTO, &FftsTaskBuilder::GenAutoMixAICAIVCtxDef},
          {ffts::TaskBuilderType::EN_TASK_TYPE_MIX_AIC_AIV_DYNAMIC, &FftsTaskBuilder::GenDynamicMixAICAIVCtxDef},
          {ffts::TaskBuilderType::EN_TASK_TYPE_MIX_L2_AIC_AIV, &FftsTaskBuilder::GenMixL2CtxDef}
  };
};
}  // namespace fe
#endif  // AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_OPS_KERNEL_BUILDER_TASK_BUILDER_FFTS_TASK_BUILDER_H_

