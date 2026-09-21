/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_INC_COMMON_FE_GENTASK_UTILS_H_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_INC_COMMON_FE_GENTASK_UTILS_H_

#include "graph/utils/graph_utils.h"
#include "common/fe_log.h"
#include "proto/task.pb.h"
#include "exe_graph/lowering/exe_res_generation_ctx_builder.h"

namespace fe {

struct ParamDef {
  uint64_t tiling_context;
  uint64_t op_type;
  bool is_custom = false;
  bool is_prefix_ops_path = false;
  char soName[4096];
  char kernelName[4096];
  char opName[32] = "TilingSink";
  char hostInputInfo[16];
};

Status GenerateOpExtTask(const ge::Node &node, const bool is_tiling_sink,
                         std::vector<domi::TaskDef> &task_defs, bool &reg_flag);
bool IsCustomiseOp(const ge::OpDesc &op_desc);
bool IsPrefixOpsPath(const ge::OpDesc &op_desc, std::string &ops_path_name_prefix);
bool CheckTilingSink(const ge::Node &node);
Status GenerateTaskForSinkOp(const gert::ExeResGenerationContext* context, const ParamDef &param,
                             std::vector<domi::TaskDef> &tasks);
ge::Status CreateTilingTaskSuperKernel(const gert::ExeResGenerationContext* context,
    domi::TaskDef &aicpu_task, const ParamDef &param);
ge::Status GenerateTaskSuperKernel(const gert::ExeResGenerationContext* context,
                                   std::vector<domi::TaskDef> &tasks, const ParamDef &param);

} // namespace fe
#endif  // AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_INC_COMMON_FE_GENTASK_UTILS_H_
