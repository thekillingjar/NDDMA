/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_KERNEL_INFER_SHAPE_H_
#define AIR_CXX_RUNTIME_V2_KERNEL_INFER_SHAPE_H_
#include "exe_graph/runtime/infer_shape_context.h"
#include "framework/common/debug/ge_log.h"
#include "transfer_shape_according_to_format.h"
#include "graph/utils/type_utils.h"
#include "common/checker.h"
#include "graph/fast_graph/fast_node.h"

namespace gert {
namespace kernel {
enum class CreateOpFromBufferInputIndex {
  kOpBuffer,
  kOpBufferSize,
  // add new output definitions here
  kInputNum
};

enum class CompatibleInferShapeOrRangeInputIndex {
  kOperator,
  kInferFunc,
  // add new output definitions here
  kInputNum
};

inline ge::graphStatus TransformOutputShape(const ComputeNodeInfo *compute_node_info,
                                            const CompileTimeTensorDesc *output_td,
                                            StorageShape *storage_shape) {
  auto expand_dims = output_td->GetExpandDimsType();
  GE_CHK_STATUS_RET(expand_dims.Expand(storage_shape->GetOriginShape(), storage_shape->MutableStorageShape()),
                    "Node %s fail to expand dims.", compute_node_info->GetNodeName());
  if (output_td->GetOriginFormat() == output_td->GetStorageFormat()) {
    GELOGD("origin_format is the same as storage format[%s], skip transfer shape.",
           ge::TypeUtils::FormatToSerialString(output_td->GetOriginFormat()).c_str());
    return ge::GRAPH_SUCCESS;
  }
  bool is_success = transformer::ShapeTransferAccordingToFormat::TransferShape(
      output_td->GetOriginFormat(), output_td->GetStorageFormat(), output_td->GetDataType(),
      storage_shape->MutableStorageShape());
  if (!is_success) {
    GELOGE(ge::GRAPH_FAILED, "Fail to transfer node %s output shape according format.",
           compute_node_info->GetNodeName());
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus TransformAllOutputsShape(const ComputeNodeInfo *compute_node_info, KernelContext *context) {
  for (size_t index = 0U; index < compute_node_info->GetOutputsNum(); ++index) {
    auto output_td = compute_node_info->GetOutputTdInfo(index);
    auto storage_shape = context->GetOutputPointer<StorageShape>(index);
    GE_ASSERT_NOTNULL(output_td);
    GE_ASSERT_NOTNULL(storage_shape);
    GE_CHK_STATUS_RET(TransformOutputShape(compute_node_info, output_td, storage_shape),
                      "Fail to transfer node %s %zu output shape according format.",
                      compute_node_info->GetNodeName(), index);
  }
  return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus BuildInferShapeOutputs(const ge::FastNode *node, KernelContext *context) {
  (void) node;
  auto extend_context = reinterpret_cast<ExtendedKernelContext *>(context);
  GE_ASSERT_NOTNULL(extend_context);
  for (size_t index = 0; index < context->GetOutputNum(); index++) {
    auto chain = context->GetOutput(index);
    GE_ASSERT_NOTNULL(chain);
    auto output_desc = extend_context->GetOutputDesc(index);
    GE_ASSERT_NOTNULL(output_desc);
    chain->SetWithDefaultDeleter(new (std::nothrow) Tensor(StorageShape(),
        output_desc->GetFormat(), output_desc->GetDataType()));
  }
  return ge::GRAPH_SUCCESS;
}

/**
   * 用于在Kernel Trace中打印输入Shape信息
   * @param context 执行时context数据
   * @param input_range_start_index 标识输入shape在context中的起始index
   * @return 返回输入shape拼接后的字符
   */
std::string PrintInputShapeInfo(const KernelContext *const context, const size_t &input_shape_start_index);

/**
   * 用于在Kernel Trace中打印输出Shape信息
   * @param context 执行时context数据
   * @return 返回输出shape拼接后的字符
   */
std::string PrintOutputShapeInfo(const KernelContext *const context);

/**
   * 用于在Kernel Trace中打印结点名字
   * @param context 执行时context数据
   * @return 返回节点名
   */
std::string PrintNodeType(const KernelContext *context);
} // namespace kernel
} // namespace gert
#endif // AIR_CXX_RUNTIME_V2_KERNEL_INFER_SHAPE_H_
