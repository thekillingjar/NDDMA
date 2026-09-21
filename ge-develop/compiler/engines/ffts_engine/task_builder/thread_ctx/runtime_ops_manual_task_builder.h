/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINE_FFTSENG_TASK_BUILDER_THREAD_CTX_RUNTIME_OPS_MANUAL_TASK_BUILDER_H_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINE_FFTSENG_TASK_BUILDER_THREAD_CTX_RUNTIME_OPS_MANUAL_TASK_BUILDER_H_
#include <map>
#include <memory>
#include <vector>
#include "task_builder/fftsplus_task_builder.h"
#include "proto/task.pb.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "inc/ffts_type.h"
namespace ffts {
using FftsPlusCtxDefPtr = std::shared_ptr<domi::FftsPlusCtxDef>;
// used for label switch
class RuntimeOpsTaskBuilder : public FFTSPlusTaskBuilder {
 public:
  RuntimeOpsTaskBuilder();
  ~RuntimeOpsTaskBuilder() override;

  /*
   * @ingroup ffts
   * @brief   Generate tasks
   * @param   [in] node Node of compute graph
   * @param   [in] context Context for generate tasks
   * @param   [out] task_defs Save the generated tasks.
   * @return  SUCCESS or FAILED
   */
  Status GenContextDef(const ge::NodePtr &node, domi::FftsPlusTaskDef *ffts_plus_task_def) override;
 private:
  RuntimeOpsTaskBuilder(const RuntimeOpsTaskBuilder &builder) = delete;
  RuntimeOpsTaskBuilder &operator=(const RuntimeOpsTaskBuilder &builder) = delete;

 protected:
  Status JudgeContextType(const ge::NodePtr &node, rtFftsPlusContextType_t &context_type);
  Status FillCaseSwitchContextData(const domi::FftsPlusCaseSwitchCtxDef *case_switch_ctx_def_ptr,
                                   domi::FftsPlusCaseSwitchCtxDef *case_switch_ctx_def) const;
  Status FillCondSwitchContextData(const domi::FftsPlusCondSwitchCtxDef *cond_switch_ctx_def_ptr,
                                   domi::FftsPlusCondSwitchCtxDef *cond_switch_ctx_def) const;
  Status GenCondSwitchContextTaskDef(FftsPlusComCtx_t &sub_ffts_plus_context,
                                     domi::FftsPlusCondSwitchCtxDef *task_def_ptr);
  Status FillSdmaContextData(const domi::FftsPlusSdmaCtxDef *sdma_ctx_def_ptr,
                             domi::FftsPlusSdmaCtxDef *sdma_ctx_def) const;
  Status FillLabelContext(const ge::OpDescPtr &op_desc, domi::FftsPlusCtxDef *ffts_plus_ctx_def,
                          vector<FftsPlusComCtx_t> &sub_ffts_plus_context, size_t thread_id) const;
  Status FillSdmaContext(const ge::OpDescPtr &op_desc, domi::FftsPlusCtxDef *ffts_plus_ctx_def,
      FftsPlusCtxDefPtr &ctx_def_ptr, vector<FftsPlusComCtx_t> &sub_ffts_plus_context, size_t thread_id) const;
  Status FillCaseSwitchContext(const ge::NodePtr &node, domi::FftsPlusCtxDef *ffts_plus_ctx_def,
      FftsPlusCtxDefPtr &ctx_def_ptr, vector<FftsPlusComCtx_t> &sub_ffts_plus_context, size_t thread_id) const;
  Status FillCondSwitchContext(const ge::NodePtr &node, domi::FftsPlusCtxDef *ffts_plus_ctx_def,
      FftsPlusCtxDefPtr &ctx_def_ptr, vector<FftsPlusComCtx_t> &sub_ffts_plus_context, size_t thread_id) const;
  Status GetSwitchInputDataAddr(const ge::NodePtr &node, uint64_t &data_base) const;

  const std::map<rtFftsPlusContextType_t, std::unordered_set<std::string>> RTS_CONTEXT_TYPE_MAP = {
      {RT_CTX_TYPE_CASE_SWITCH, CASE_SWITCH_NODE_TYPE}, {RT_CTX_TYPE_LABEL, LABEL_NODE_TYPE},
      {RT_CTX_TYPE_SDMA, SDMA_NODE_TYPE}, {RT_CTX_TYPE_COND_SWITCH, COND_SWITCH_NODE_TYPE}};

  const std::map<std::string, rtFftsPlusCondType_t> RTS_COND_TYPE_VALUE_MAP = {
      {"Less", RT_COND_TYPE_LESS},
      {"LessEqual", RT_COND_TYPE_LESS_OR_EQUAL},
      {"Greater", RT_COND_TYPE_GREATER},
      {"GreaterEqual", RT_COND_TYPE_GREATER_OR_EQUAL},
      {"Equal", RT_COND_TYPE_EQUAL},
      {"NotEqual", RT_COND_TYPE_NOTEQUAL}};
};

}  // namespace ffts
#endif  // AIR_COMPILER_GRAPHCOMPILER_ENGINE_FFTSENG_TASK_BUILDER_THREAD_CTX_RUNTIME_OPS_MANUAL_TASK_BUILDER_H_
 
