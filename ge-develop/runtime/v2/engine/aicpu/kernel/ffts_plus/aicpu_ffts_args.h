/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_AICPU_FFTS_ARGS_H
#define AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_AICPU_FFTS_ARGS_H

#include "graph/ge_error_codes.h"
#include "graph/def_types.h"
#include "exe_graph/runtime/kernel_context.h"

namespace gert {
enum class FFTSAicpuArgss {
  kInMemType,
  kOutMemType,
  kSessionId,
  kExtInfo,
  kExtLen,
  kArgInfo,
  kArgLen,
  kNodeName,
  kUnknownShapeType,
  kThreadParam,
  kNotLastInSlice,
  kNotLastOutSlice,
  kLastInSlice,
  kLastOutSlice,
  kCtxIds,
  kIoStart
};

enum class FFTSAicpuArgsOutKey {
  kFlushData,
  kArgAddr,
  kNum
};

enum class FFTSAicpuSoAndKernel {
  kCtxIds,
  kIndex,
  kSoNameDev,
  kSoNameSize,
  kSoNameData,
  kKernelNameDev,
  kKernelNameSize,
  kKernelNameData,
  kStream
};

namespace kernel {
  std::string PrintAIcpuArgsDetail(const KernelContext *context);
  std::vector<std::string> PrintAICpuArgs(const KernelContext *context);
}
} // gert
#endif // AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_AICPU_FFTS_ARGS_H
