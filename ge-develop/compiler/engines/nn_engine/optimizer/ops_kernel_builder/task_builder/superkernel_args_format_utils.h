/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_OPS_KERNEL_BUILDER_TASK_BUILDER_SUPERKERNEL_ARGS_FORMAT_UTILS_H_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_OPS_KERNEL_BUILDER_TASK_BUILDER_SUPERKERNEL_ARGS_FORMAT_UTILS_H_

#include <vector>
#include <map>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <securec.h>

#include "ops_kernel_builder/task_builder/task_builder.h"
#include "graph/utils/args_format_desc_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/args_format_desc.h"
#include "framework/common/taskdown_common.h"

namespace fe {
    ge::Status GenTaskForSuperKernel(const ge::Node &node, std::vector<std::vector<domi::TaskDef>> &subTasks,
                                     const std::vector<ge::Node *> &sub_nodes, std::vector<domi::TaskDef> &tasks);
    ge::Status GetArgFormatV2(domi::TaskDef &task_temp, std::string &args_format);
    ge::Status GetWorkspacePattern(const ge::Node &node, std::string &super_kernel_args_format, int64_t ws_size);
    bool IsAICpuKernelType(ge::ccKernelType kernel_type);
    bool IsAICpuTaskDef(domi::TaskDef &task_temp, domi::KernelContext *&kernel_context);
    uint64_t GetAtomicStubFuncId();
    std::string GetUniqueGraphIdForNode(const ge::OpDescPtr &super_kernel_op_desc);
    bool KernelLaunch(const std::string &stub_func, const uint32_t block_dim, const void *args,
                      uint32_t args_size, const rtSmDesc_t *sm_desc, domi::TaskDef &task_def);
    ge::Status FillTaskDefAfterGenTask(const ge::OpDescPtr &op_desc, domi::TaskDef &task_def, std::string &args_format);
    ge::Status SetArgFormatValue(uint32_t args_size_workspace, std::vector<std::vector<domi::TaskDef>> &subTasks,
                         const std::vector<ge::Node *> &sub_nodes, void *all_args_buff_total, uint32_t args_size_total);
    ge::Status GetAICoreArgFormatinLoop(std::string &sub_arg_format, const std::vector<ge::Node *> &sub_nodes, uint32_t &args_size,
                                        std::string &super_kernel_args_format, const ge::NodePtr &shared_node, const ge::NodePtr &sub_node, uint32_t cnt);
    ge::Status GetArgFormat(const std::vector<ge::Node *> &sub_nodes, uint32_t &args_size_total, std::vector<std::vector<domi::TaskDef>> &subTasks,
                            const ge::OpDescPtr &super_kernel_op_desc, std::vector<domi::TaskDef> &tasks, const ge::NodePtr &shared_node,
                            std::string &super_kernel_args_format);
    void CheckDFXOpen(uint32_t &args_size_workspace, const ge::Node &node, std::string &super_kernel_args_format, int64_t ws_size);
    int64_t GetSuperKernelWorkspace(const ge::Node &node);
}  // namespace fe
#endif  // AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_OPS_KERNEL_BUILDER_TASK_BUILDER_SUPERKERNEL_ARGS_FORMAT_UTILS_H_
