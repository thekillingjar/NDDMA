/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_KERNEL_INFER_SHAPE_RANGE_H_
#define AIR_CXX_RUNTIME_V2_KERNEL_INFER_SHAPE_RANGE_H_

#include "exe_graph/runtime/infer_shape_context.h"
#include "framework/common/debug/ge_log.h"
#include "exe_graph/runtime/range.h"
#include "common/checker.h"
#include "graph/utils/type_utils.h"
#include "graph/fast_graph/fast_node.h"

namespace gert {
namespace kernel {
/*
 * 3n个输出，0~n-1 Range<Shape>, n~2n-1 min shape，2n~3n-1 max shape,n是node输出个数
 */
constexpr size_t kShapeRangeOutputOfNode = 3U;
using TensorRange = Range<Tensor>;
ge::graphStatus TransformAllOutputsMaxShape(const ComputeNodeInfo *compute_node_info, KernelContext *context);
ge::graphStatus BuildInferShapeRangeOutputs(const ge::FastNode *node, KernelContext *context);

/**
   * 用于在Kernel Trace中打印输入Range信息
   * @param context 执行时context数据
   * @param input_range_start_index 标识输入range在context中的起始index
   */
std::string PrintInputRangeInfo(const KernelContext *const context, const size_t &input_range_start_index);

/**
   * 用于在Kernel Trace中打印输出Range信息
   * @param context 执行时context数据
   */
std::string PrintOutputRangeInfo(const KernelContext *context);
} // namespace kernel
} // namespace gert
#endif // AIR_CXX_RUNTIME_V2_KERNEL_INFER_SHAPE_RANGE_H_
