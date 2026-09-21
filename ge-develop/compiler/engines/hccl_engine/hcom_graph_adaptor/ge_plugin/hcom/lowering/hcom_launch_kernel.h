/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef HCOM_LAUNCH_KERNEL_H
#define HCOM_LAUNCH_KERNEL_H

#include "hccl/base.h"
#include "hcom_build_graph.h"
#include "register/kernel_registry.h"
#include "register/kernel_registry_impl.h"
#include "exe_graph/lowering/shape_utils.h"
#include "exe_graph/runtime/tensor_data_utils.h"

namespace hccl {
constexpr int INPUT_INDEX_2 = 2;
constexpr int INPUT_INDEX_3 = 3;
constexpr int INPUT_INDEX_4 = 4;
constexpr int OUTPUT_INDEX_0 = 0;
constexpr int OUTPUT_INDEX_1 = 1;
constexpr int INDEX_PARA_2 = 2;
constexpr int INDEX_PARA_3 = 3;

constexpr u64 SHAPE_INFO_COUNT = 100; /* 动态send recv协商的shape info数据长度 */

struct HcomOpLaunchArgs {
  struct HcomOpAttr opAttr;
  void *stream;
  uint32_t inputNum;
  uint32_t outputNum;
  std::vector<void *> inputAddrs;
  std::vector<void *> outputAddrs;
  std::vector<gert::Shape> inputShapes;
  std::vector<gert::Shape> outputShapes;
};

struct HcomOpInputStruct {
  HcomOpLaunchArgs launchArgs;
  HcclComm hcclComm;
#ifndef OPEN_BUILD_PROJECT
  void *hcclCommPtr = nullptr;
#endif
  void *commInputPtr = nullptr;
  void *commOutputPtr = nullptr;
  std::vector<uint64_t> inputsCount = {};
  std::vector<uint32_t> sliceIdxs = {};
  u64 commInputSize = 0;
  u64 commOutputSize = 0;
  uint64_t count = 0;
  uint64_t sendCount = 0;
  uint64_t recvCount = 0;
  HcclWorkflowMode lastWorkflowMode = HcclWorkflowMode::HCCL_WORKFLOW_MODE_RESERVED;
};

std::vector<std::string> PrintLaunchArgs(const gert::KernelContext *context);
ge::graphStatus PrepareHcomKernel(gert::KernelContext *context);
ge::graphStatus LaunchHcomKernel(gert::KernelContext *context);
ge::graphStatus LaunchRecvKernel(gert::KernelContext *context);
ge::graphStatus LaunchHcomKernelInitComm(gert::KernelContext *context);
ge::graphStatus BuildHcclOutputShapeOutputs(const ge::FastNode *node, gert::KernelContext *context);
ge::graphStatus BuildPrepareHcomKernelOutput(const ge::FastNode *node, gert::KernelContext *context);
ge::graphStatus HcomGetRecvBeforeKernel(HcomOpLaunchArgs &args, std::vector<int64_t> &recvShape);

HcclResult GetHcomOpLaunchArgs(gert::KernelContext *context, HcomOpLaunchArgs &args);
HcclResult GetRecvOpLaunchArgs(gert::KernelContext *context, HcomOpLaunchArgs &args);
HcclResult HcomGetBatchAllReduceInfo(const HcomOpLaunchArgs &launchArgs, uint64_t maxCommCount,
                                     std::vector<uint32_t> &sliceIdxs, std::vector<uint64_t> &inputsCount);
HcclResult HcomCopyInputsToCCLbuff(const HcomOpLaunchArgs &launchArgs, uint32_t inputsNum, uint32_t inputsOffset,
                                   std::vector<uint64_t> &inputsCount, void *cclBuff, uint64_t cclBuffSize,
                                   uint64_t &commCount, std::vector<void *> &inputAddrs);
HcclResult HcomCopyCCLbuffToOutnputs(const HcomOpLaunchArgs &launchArgs, uint32_t outputsNum, uint32_t outputsOffset,
                                     std::vector<uint64_t> &outputsCount, void *cclBuff,
                                     std::vector<void *> &outputAddrs);
HcclResult GetCountByShape(const gert::Shape &shape, HcclDataType dataType, uint64_t &count);
HcclResult GetSendCountByShapeAlign(const gert::Shape &shape, HcclDataType dataType, uint64_t &count);
uint64_t RoundUp(const uint64_t originValue, const uint64_t multiple);

HcclResult HcomAllGatherKernel(HcomOpLaunchArgs &launchArgs, HcomOpInputStruct *inputStruct);
HcclResult HcomAllGatherVKernel(HcomOpLaunchArgs &launchArgs, HcomOpInputStruct *inputStruct);
HcclResult HcomAllReduceKernel(HcomOpLaunchArgs &launchArgs, HcomOpInputStruct *inputStruct);
HcclResult HcomBroadcastKernel(HcomOpLaunchArgs &launchArgs, HcomOpInputStruct *inputStruct);
HcclResult HcomReduceScatterKernel(HcomOpLaunchArgs &launchArgs, HcomOpInputStruct *inputStruct);
HcclResult HcomReduceScatterVKernel(HcomOpLaunchArgs &launchArgs, HcomOpInputStruct *inputStruct);
HcclResult HcomAllToAllVKernel(HcomOpLaunchArgs &launchArgs, HcomOpInputStruct *inputStruct);
HcclResult HcomAllToAllVCKernel(HcomOpLaunchArgs &launchArgs, HcomOpInputStruct *inputStruct);
HcclResult HcomAllToAllKernel(HcomOpLaunchArgs &launchArgs, HcomOpInputStruct *inputStruct);
HcclResult HcomReduceKernel(HcomOpLaunchArgs &launchArgs, HcomOpInputStruct *inputStruct);
HcclResult HcomSendKernel(HcomOpLaunchArgs &launchArgs, HcomOpInputStruct *inputStruct);

HcclResult HcomLaunchAllGatherKernel(const HcomOpInputStruct *inputStruct, std::vector<void *> &inputAddrs,
                                     std::vector<void *> &outputAddrs);
HcclResult HcomLaunchAllGatherVKernel(const HcomOpInputStruct *inputStruct, std::vector<void *> &inputAddrs,
                                      std::vector<void *> &outputAddrs);
HcclResult HcomLaunchAllReduceKernel(const HcomOpInputStruct *inputStruct, std::vector<void *> &inputAddrs,
                                     std::vector<void *> &outputAddrs);
HcclResult HcomLaunchBroadcastKernel(const HcomOpInputStruct *inputStruct, std::vector<void *> &inputAddrs,
                                     std::vector<void *> &outputAddrs);
HcclResult HcomLaunchReduceScatterKernel(const HcomOpInputStruct *inputStruct, std::vector<void *> &inputAddrs,
                                         std::vector<void *> &outputAddrs);
HcclResult HcomLaunchReduceScatterVKernel(const HcomOpInputStruct *inputStruct, std::vector<void *> &inputAddrs,
                                          std::vector<void *> &outputAddrs);
HcclResult HcomLaunchAllToAllVKernel(const HcomOpInputStruct *inputStruct, std::vector<void *> &inputAddrs,
                                     std::vector<void *> &outputAddrs);
HcclResult HcomLaunchAllToAllVCKernel(const HcomOpInputStruct *inputStruct, std::vector<void *> &inputAddrs,
                                      std::vector<void *> &outputAddrs);
HcclResult HcomLaunchAllToAllKernel(const HcomOpInputStruct *inputStruct, std::vector<void *> &inputAddrs,
                                    std::vector<void *> &outputAddrs);
HcclResult HcomLaunchReduceKernel(const HcomOpInputStruct *inputStruct, std::vector<void *> &inputAddrs,
                                  std::vector<void *> &outputAddrs);
HcclResult HcomLaunchSendKernel(const HcomOpInputStruct *inputStruct, std::vector<void *> &inputAddrs,
                                std::vector<void *> &outputAddrs);

#ifndef OPEN_BUILD_PROJECT
HcclResult HcomAllGatherKernelV2(HcomOpLaunchArgs &launchArgs, HcomOpInputStruct *inputStruct);
HcclResult HcomAllReduceKernelV2(HcomOpLaunchArgs &launchArgs, HcomOpInputStruct *inputStruct);
HcclResult HcomBroadcastKernelV2(HcomOpLaunchArgs &launchArgs, HcomOpInputStruct *inputStruct);
HcclResult HcomReduceScatterKernelV2(HcomOpLaunchArgs &launchArgs, HcomOpInputStruct *inputStruct);
HcclResult HcomAllToAllVKernelV2(HcomOpLaunchArgs &launchArgs, HcomOpInputStruct *inputStruct);
HcclResult HcomAllToAllVCKernelV2(HcomOpLaunchArgs &launchArgs, HcomOpInputStruct *inputStruct);
HcclResult HcomReduceKernelV2(HcomOpLaunchArgs &launchArgs, HcomOpInputStruct *inputStruct);

HcclResult HcomLaunchAllGatherKernelV2(const HcomOpInputStruct *inputStruct, std::vector<void *> &inputAddrs,
                                       std::vector<void *> &outputAddrs);
HcclResult HcomLaunchAllReduceKernelV2(const HcomOpInputStruct *inputStruct, std::vector<void *> &inputAddrs,
                                       std::vector<void *> &outputAddrs);
HcclResult HcomLaunchBroadcastKernelV2(const HcomOpInputStruct *inputStruct, std::vector<void *> &inputAddrs,
                                       std::vector<void *> &outputAddrs);
HcclResult HcomLaunchReduceScatterKernelV2(const HcomOpInputStruct *inputStruct, std::vector<void *> &inputAddrs,
                                           std::vector<void *> &outputAddrs);
HcclResult HcomLaunchAllToAllVKernelV2(const HcomOpInputStruct *inputStruct, std::vector<void *> &inputAddrs,
                                       std::vector<void *> &outputAddrs);
HcclResult HcomLaunchAllToAllVCKernelV2(const HcomOpInputStruct *inputStruct, std::vector<void *> &inputAddrs,
                                        std::vector<void *> &outputAddrs);
HcclResult HcomLaunchReduceKernelV2(const HcomOpInputStruct *inputStruct, std::vector<void *> &inputAddrs,
                                    std::vector<void *> &outputAddrs);
#endif

}  // namespace hccl
#endif  // HCOM_LAUNCH_KERNEL_H
