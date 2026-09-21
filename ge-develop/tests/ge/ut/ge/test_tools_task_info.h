/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __INC_TEST_TOOLS_TASK_INFO_H
#define __INC_TEST_TOOLS_TASK_INFO_H

#include "proto/task.pb.h"
#include "graph/compute_graph.h"
#include "common/tbe_handle_store/tbe_kernel_store.h"
#include "common/tbe_handle_store/cust_aicpu_kernel_store.h"
#include "runtime/rt.h"
#include <set>

namespace ge {
extern std::set<std::string> actual_info_type;

void AddPartitionedCall(const ComputeGraphPtr &graph, const std::string &func_name, const ComputeGraphPtr &subgraph);

void AddFftsPartitionedCall(const ComputeGraphPtr &graph, const std::string &func_name, const ComputeGraphPtr &subgraph);

void AddCaseBranch(const ComputeGraphPtr &graph, const std::string &func_name, const ComputeGraphPtr &subgraph);

void AddIfBranchs(const ComputeGraphPtr &graph, const std::string &func_name,
                  const ComputeGraphPtr &then_graph, const ComputeGraphPtr &else_graph);

void SetUnknownOpKernel(const ComputeGraphPtr &graph, uint32_t &mem_offset, bool reset_index = false);
void DelStaticForOffline(const ComputeGraphPtr &graph, uint32_t &mem_offset);

void CleanAippNodeInfo(const ComputeGraphPtr &graph, const std::string &op_name);
void InitAippNodeDynamic(const ComputeGraphPtr &graph, const std::string &op_name);
void InitAippNodeStatic(const ComputeGraphPtr &graph, const std::string &op_name);
void InitAippNodeRelated(const ComputeGraphPtr &graph, const std::string &op_name, const std::string &related_name);

void InitConstantNode(const ComputeGraphPtr &graph, const std::string &op_name, int32_t const_value);
void InitConstantNode(const ComputeGraphPtr &graph, const std::string &op_name, const GeTensorDesc &tensor_desc, const std::string &const_value);

void InitKernelTaskDef(const ComputeGraphPtr &graph, domi::ModelTaskDef &model_def, const std::string &op_name,
                       const int64_t stream_id = 0);

void InitKernelTaskDef_TE(const ComputeGraphPtr &graph, domi::ModelTaskDef &model_def, const std::string &op_name, TBEKernelStore &kernel_store);
void InitKernelWithHandleTaskDef_TE(const ComputeGraphPtr &graph, domi::ModelTaskDef &model_def,
                                    const std::string &op_name, TBEKernelStore &kernel_store);
void InitKernelWithHandleTaskDef_Attached(const ComputeGraphPtr &graph, domi::ModelTaskDef &model_def,
                                          const std::string &op_name, TBEKernelStore &kernel_store);
void InitKernelTaskDef_Atomic(const ComputeGraphPtr &graph, domi::ModelTaskDef &model_def, const std::string &op_name,
                              TBEKernelStore &kernel_store);
void InitKernelTaskDef_TE_SM(const ComputeGraphPtr &graph, domi::ModelTaskDef &model_def, const std::string &op_name);

void InitKernelTaskDef_AI_CPU(const ComputeGraphPtr &graph, domi::ModelTaskDef &model_def, const std::string &op_name,
                              const std::string &extra_info = "");
void InitKernelTaskDef_CPU_AllShape(const ComputeGraphPtr &graph, domi::ModelTaskDef &model_def,
                                    const std::string &op_name);
void InitKernelTaskDef_CPU_Blocking(const ComputeGraphPtr &graph, domi::ModelTaskDef &model_def,
                                    const std::string &op_name);
void InitKernelTaskDef_CUST_CPU(const ComputeGraphPtr &graph, domi::ModelTaskDef &model_def, const std::string &op_name,
                                CustAICPUKernelStore &kernel_store, const std::string &extra_info="");

void InitKernelTaskDef_CUSTOM(const ComputeGraphPtr &graph, domi::ModelTaskDef &model_def, const std::string &op_name);

void InitKernelExTaskDef(const ComputeGraphPtr &graph, domi::ModelTaskDef &model_def, const std::string &op_name,
                         const std::string &extra_info = "");
void InitKernelExTaskDef_AllShape(const ComputeGraphPtr &graph, domi::ModelTaskDef &model_def,
                                  const std::string &op_name);
void InitKernelExTaskDef_Blocking(const ComputeGraphPtr &graph, domi::ModelTaskDef &model_def, const std::string &op_name);

void InitStreamActiveDef(const ComputeGraphPtr &graph, domi::ModelTaskDef &model_def, const std::string &op_name);

void InitStreamSwitchDef(const ComputeGraphPtr &graph, domi::ModelTaskDef &model_def, const std::string &op_name,
                         const int64_t stream_id = 0);

void InitStreamSwitchNDef(const ComputeGraphPtr &graph, domi::ModelTaskDef &model_def, const std::string &op_name);

void InitStreamMergeDef(const ComputeGraphPtr &graph, domi::ModelTaskDef &model_def, const std::string &op_name);

void InitLabelSwitchDef(const ComputeGraphPtr &graph, domi::ModelTaskDef &model_def, const std::string &op_name);

void InitLabelSetDef(const ComputeGraphPtr &graph, domi::ModelTaskDef &model_def, const std::string &op_name);

void InitLabelGotoDef(const ComputeGraphPtr &graph, domi::ModelTaskDef &model_def, const std::string &op_name);

void InitMemcpyAsyncDef(const ComputeGraphPtr &graph, domi::ModelTaskDef &model_def, const std::string &op_name);

void InitMemcpyAddrAsyncDef(const ComputeGraphPtr &graph, domi::ModelTaskDef &model_def, const std::string &op_name,
                            const int64_t stream_id = 0);

void InitEndGraphDef(const ComputeGraphPtr &graph, domi::ModelTaskDef &model_def, const std::string &op_name);

void InitHcclTaskDef(const ComputeGraphPtr &graph, domi::ModelTaskDef &model_def, const std::string &op_name,
                     const std::string &hccl_type);

void InitProfilerTaskDef(const ComputeGraphPtr &graph, domi::ModelTaskDef &model_def);

void InitEventTaskDef(const ComputeGraphPtr &graph, domi::ModelTaskDef &model_def);

void InitNotifyTaskDef(const ComputeGraphPtr &graph, domi::ModelTaskDef &model_def, uint32_t notify_id, const std::string &group_name);

void InitNotifyWaitTaskDef(const ComputeGraphPtr &graph, domi::ModelTaskDef &model_def, uint32_t notify_id, const std::string &group_name);

void InitFftsPlusCaseDefaultDef(const ComputeGraphPtr &graph, domi::FftsPlusCtxDef &ctx_def,
                                const std::string &op_name);

void InitFftsPlusNotifyDef(const ComputeGraphPtr &graph, domi::FftsPlusCtxDef &ctx_def, const std::string &op_name);

void InitWriteValueDef(const ComputeGraphPtr &graph, domi::FftsPlusCtxDef &ctx_def, const std::string &op_name);

void InitMixAicAivDef(const ComputeGraphPtr &graph, domi::FftsPlusCtxDef &ctx_def, const std::string &op_name,
                      bool is_auto = false);
void InitMixL2Def(const ComputeGraphPtr &graph, domi::FftsPlusCtxDef &ctx_def, const std::string &op_name);

void InitMixL2DefForIFA(const ComputeGraphPtr &graph, domi::FftsPlusCtxDef &ctx_def, const std::string &op_name);

void InitMixL2DefForIFATilingSink(const ComputeGraphPtr &graph, domi::FftsPlusCtxDef &ctx_def,
                                  const std::string &op_name);

void InitTaskDefForMC2(const ComputeGraphPtr &graph, domi::FftsPlusCtxDef &ctx_def, const std::string &op_name);

void InitSdmaDef(const ComputeGraphPtr &graph, domi::FftsPlusCtxDef &ctx_def, const std::string &op_name);

void InitDsaDef(const ComputeGraphPtr &graph, domi::FftsPlusCtxDef &ctx_def, const size_t workspcace_num,
                const std::string &op_name, const bool is_set_value);

void InitDataCtx(const ComputeGraphPtr &graph, domi::FftsPlusCtxDef &ctx_def, const std::string &op_name);

void InitCondSwitchCtx(const ComputeGraphPtr &graph, domi::FftsPlusCtxDef &ctx_def, const std::string &op_name);

void InitFusionTaskDef(const ComputeGraphPtr &graph, domi::ModelTaskDef &model_def);

void InitFftsplusTaskDef(const ComputeGraphPtr &graph, domi::TaskDef &model_def);

void InitFftsPlusCachePersistDef(const ComputeGraphPtr &graph, domi::FftsPlusCtxDef &ctx_def,
                                 const std::string &op_name);

void InitFftsPlusAicCtxDef(const ComputeGraphPtr &graph, domi::FftsPlusCtxDef &ctx_def, const std::string &op_name,
                           bool is_manual = false);

void InitFftsPlusAicCtxDefWithTilingData(const ComputeGraphPtr &graph, domi::FftsPlusCtxDef &ctx_def,
                                         const std::string &op_name);

void InitFftsPlusAicpuCtxDef(const ComputeGraphPtr &graph, domi::FftsPlusCtxDef &ctx_def, const std::string &op_name);

void InitCustomFftsPlusAicpuCtxDef(const ComputeGraphPtr &graph, domi::FftsPlusCtxDef &ctx_def,
                                   const std::string &op_name);

void InitFftsPlusAicpuFwkCtxDef(const ComputeGraphPtr &graph, domi::FftsPlusCtxDef &ctx_def, const std::string &op_name);

void InitCmoTaskDef(const ComputeGraphPtr &graph, domi::ModelTaskDef &model_def);

void InitCmoAddrTaskDef(const ComputeGraphPtr &graph, domi::ModelTaskDef &model_def, const std::string &op_name,
                        int max_size = 0);

void InitCmoBarrierTaskDef(const ComputeGraphPtr &graph, domi::ModelTaskDef &model_def);

void InitDSATaskDef(const ComputeGraphPtr &graph, domi::ModelTaskDef &model_def, const std::string &op_name,
                    const bool set_ptr_or_value);

void InitDvppTaskDef(const ComputeGraphPtr &graph, domi::ModelTaskDef &model_def, const std::string &op_name);

void InitNpuGetFloatStatusTaskDef(const ComputeGraphPtr &graph, domi::ModelTaskDef &model_def,
                                  const std::string &op_name);

void InitNpuClearFloatStatusTaskDef(const ComputeGraphPtr &graph, domi::ModelTaskDef &model_def,
                                    const std::string &op_name);

// init fusion op info for profiling
void InitFusionOpInfo(const ComputeGraphPtr &graph, const std::string &op_name);

// mock function to report profiling data
int32_t ReporterCallback(uint32_t moduleId, uint32_t type, void *data, uint32_t len);

void InitAiCpuAllShape(const OpDescPtr &op_desc, std::vector<uint8_t> &aicpu_ext_info);
} // namespace ge
#endif  // __INC_TEST_TOOLS_TASK_INFO_H
