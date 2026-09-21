/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __INC_TESTS_FFTS_PLUS_PROTO_TOOLS_H
#define __INC_TESTS_FFTS_PLUS_PROTO_TOOLS_H

#include "proto/task.pb.h"
#include "graph/compute_graph.h"
#include "register/ffts_plus_task_update.h"

#define ADD_FFTS_PLUS_CTX(type, func_proto, func_init, ctx_node, args...)    \
  do {                                                                       \
    domi::FftsPlusCtxDef *ctx_def = ffts_plus_task_def->add_ffts_plus_ctx(); \
    ctx_def->set_op_index(ctx_node->GetOpDesc()->GetId());                   \
    ctx_def->set_context_type(static_cast<uint32_t>(type));                  \
    auto inner_ctx_def = ctx_def->func_proto();                              \
    func_init(inner_ctx_def, ##args);                                        \
} while (false)

#define ADD_FFTS_PLUS_CTX_NODE(type, func_proto, func_init, ctx_node, args...)  \
  do {                                                                          \
    domi::FftsPlusCtxDef *ctx_def = ffts_plus_task_def->add_ffts_plus_ctx();    \
    ctx_def->set_op_index(ctx_node->GetOpDesc()->GetId());                      \
    ctx_def->set_context_type(static_cast<uint32_t>(type));                     \
    auto inner_ctx_def = ctx_def->func_proto();                                 \
    func_init(ctx_node->GetOpDesc(), inner_ctx_def, ##args);                    \
} while (false)

#define ADD_FFTS_PLUS_CTX_MANUAL(type, func_proto, func_init, ctx_node, args...) \
  do {                                                                           \
    domi::FftsPlusCtxDef *ctx_def = ffts_plus_task_def->add_ffts_plus_ctx();     \
    ctx_def->set_op_index(ctx_node->GetOpDesc()->GetId());                       \
    ctx_def->set_context_type(static_cast<uint32_t>(type));                      \
    auto inner_ctx_def = ctx_def->func_proto();                                  \
    func_init(inner_ctx_def, ##args);                                            \
    inner_ctx_def->set_atm(0);                                                   \
  } while (false)

#define ADD_FFTS_PLUS_MIX_CTX(type, func_proto, func_init, ctx_node, args...) \
  {                                                                           \
    domi::FftsPlusCtxDef *ctx_def = ffts_plus_task_def->add_ffts_plus_ctx();  \
    ctx_def->set_op_index(ctx_node->GetOpDesc()->GetId());                    \
    ctx_def->set_context_type(static_cast<uint32_t>(type));                   \
    auto inner_ctx_def = ctx_def->func_proto();                               \
    func_init(inner_ctx_def, ##args);                                         \
  }

namespace ge {
void SetKnownOpKernel(const ComputeGraphPtr &graph, uint32_t &mem_offset);

void ResetNodeIndex();
OpDescPtr CreateOpDesc(std::string name = "", std::string type = "", int in_num = 0, int out_num = 0, bool is_dynamic = true);

NodePtr CreateNode(ComputeGraph &graph, const std::string &name, const std::string &type, int in_num, int out_num);

void SetNodeAnchorStatus(const NodePtr &node);

void InitFftsThreadSliceMap(const OpDescPtr &op_desc);

void InitTaskSQEInfo(domi::FftsPlusTaskDef *task_def);

void InitTaskAdditionalDataInfo(domi::FftsPlusTaskDef *task_def);

void InitCachePersistentCtx(domi::FftsPlusCachePersistCtxDef *ctx_def);

void InitAicAivCtx(domi::FftsPlusAicAivCtxDef *ctx_def, bool is_known = true);

void InitMixAicAivCtx(domi::FftsPlusMixAicAivCtxDef *ctx_def, bool is_auto = false, bool is_known = true);

void InitMixAicAivCtxForSingleKernel(domi::FftsPlusMixAicAivCtxDef *ctx_def, bool is_auto = false,
                                     bool is_known = true);

void InitSdmaCtx(domi::FftsPlusSdmaCtxDef *ctx_def);

void InitNotifyCtx(domi::FftsPlusNotifyCtxDef *ctx_def);

void InitWriteValueCtx(domi::FftsPlusWriteValueCtxDef *ctx_def);

void InitAicpuCtxCtx(const OpDescPtr &op_desc, domi::FftsPlusAicpuCtxDef *ctx_def, bool is_known = true);

void InitAicpuFwkCtxCtx(domi::FftsPlusAicpuCtxDef *ctx_def);

void InitDataCtx(domi::FftsPlusDataCtxDef *ctx_def);

void InitAicpuFwkCtxAndExtInfo(domi::FftsPlusAicpuCtxDef *ctx_def);

void InitAicpuCtxAndExtInfo(domi::FftsPlusAicpuCtxDef *ctx_def);

void InitCustomAicpuCtxAndExtInfo(domi::FftsPlusAicpuCtxDef *ctx_def);

void InitAtStartCtx(domi::FftsPlusAtStartCtxDef *ctx_def);

void InitAtEndCtx(domi::FftsPlusAtEndCtxDef *ctx_def);

void InitLabelCtx(domi::FftsPlusLabelCtxDef *ctx_def);

void InitCaseSwitchCtx(domi::FftsPlusCaseSwitchCtxDef *ctx_def);

void InitCaseDefaultCtx(domi::FftsPlusCaseDefaultCtxDef *ctx_def);

void InitCondSwitchCtx(domi::FftsPlusCondSwitchCtxDef *ctx_def);

void InitDsaCtx(domi::FftsPlusDsaCtxDef *ctx_def, const bool is_set_value);
}  // namespace ge
#endif // __INC_TESTS_FFTS_PLUS_PROTO_TOOLS_H
