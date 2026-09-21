/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "set_output_shape_kernel.h"
#include "engine/node_converter_utils.h"
#include "register/ffts_node_calculater_registry.h"
#include "engine/aicore/kernel/rt_ffts_plus_launch_args.h"
#include "common/dump/kernel_tracing_utils.h"
#include "exe_graph/runtime/tiling_data.h"
#include "register/kernel_registry_impl.h"
#include "register/op_tiling.h"
#include "common/math/math_util.h"
#include "utils/extern_math_util.h"
#include "exe_graph/runtime/tensor.h"
#include "exe_graph/lowering/shape_utils.h"
#include "engine/ffts_plus/converter/ffts_plus_proto_transfer.h"
namespace {
const size_t kMaxDimNum = 8U;
const size_t kShapeBufferNum = kMaxDimNum + 1U;
const uint32_t kDataTypeFlagBit = 31U;
enum class OutputShapeDtype {
  ISUINT32 = 0,
  ISUINT64 = 1,
};
}  // namespace
namespace gert {
namespace kernel {
// 判断一个int变量的每个bit位的值（1或者0）
//  num =4  --转成2进制--  100
//  1 << pos，
// 将1左移0位   1     100 & 1     结果得到是000  --> 说明该变量的右起第1位是0
// 将1左移1位   10    100 & 10    结果得到是000  --> 说明该变量的右起第2位是0
// 将1左移2位   100   100 & 100   结果得到是100  --> 说明该变量的右起第3位是1
OutputShapeDtype GetBitStatu(const uint32_t num, const uint32_t pos) {
  if (num & (1 << pos)) {  // 按位与之后的结果非0
    return OutputShapeDtype::ISUINT64;
  } else {
    return OutputShapeDtype::ISUINT32;
  }
}

// 将一个uint64_t数据类型数的第31个bit位置为0
uint64_t SetBit31ToZero(uint64_t &num) {
  return num &= (~(1U << kDataTypeFlagBit));
}

std::vector<std::string> SetOutputShapeKernelTrace(const KernelContext *context) {
  return {PrintNodeType(context), PrintOutputShapeInfo(context)};
}

ge::graphStatus SetOutputShapeProforUint32(KernelContext *context, const size_t output_size,
                                           const std::unique_ptr<uint8_t[]> host_shapebuffer) {
  const auto output_shapes_data = reinterpret_cast<uint32_t(*)[kShapeBufferNum]>(host_shapebuffer.get());
  for (size_t i = 0U; i < output_size; ++i) {
    auto output_shape = context->GetOutputPointer<StorageShape>(i);
    if (output_shape == nullptr) {
      continue;
    }
    const uint32_t dim_num = output_shapes_data[i][0];
    if (dim_num > kMaxDimNum) {
      KLOGE("output[%zu] dim_num=[%u] is greater than MaxDimNum[%zu]", i, dim_num, kMaxDimNum);
      return ge::GRAPH_FAILED;
    }
    Shape &origin_shape = output_shape->MutableOriginShape();
    Shape &storage_shape = output_shape->MutableStorageShape();
    origin_shape.SetDimNum(0);
    storage_shape.SetDimNum(0);
    for (size_t j = 1U; j <= dim_num; ++j) {
      origin_shape.AppendDim(static_cast<int64_t>(output_shapes_data[i][j]));
      storage_shape.AppendDim(static_cast<int64_t>(output_shapes_data[i][j]));
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SetOutputShapeProforUint64(KernelContext *context, const size_t output_size,
                                           const std::unique_ptr<uint8_t[]> host_shapebuffer) {
  const auto output_shapes_data = reinterpret_cast<uint64_t(*)[kShapeBufferNum]>(host_shapebuffer.get());
  for (size_t i = 0U; i < output_size; ++i) {
    auto output_shape = context->GetOutputPointer<StorageShape>(i);
    if (output_shape == nullptr) {
      continue;
    }
    const uint64_t dim_num = SetBit31ToZero(output_shapes_data[i][0]);
    if (dim_num > kMaxDimNum) {
      KLOGE("output[%zu] dim_num=[%u] is greater than MaxDimNum[%zu]", i, dim_num, kMaxDimNum);
      return ge::GRAPH_FAILED;
    }
    Shape &origin_shape = output_shape->MutableOriginShape();
    Shape &storage_shape = output_shape->MutableStorageShape();
    origin_shape.SetDimNum(0);
    storage_shape.SetDimNum(0);
    for (size_t j = 1U; j <= dim_num; ++j) {
      origin_shape.AppendDim(static_cast<int64_t>(output_shapes_data[i][j]));
      storage_shape.AppendDim(static_cast<int64_t>(output_shapes_data[i][j]));
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SetOutputShape(KernelContext *context) {
  auto shapebuffer_addr = context->GetInputValue<gert::GertTensorData *>(0);
  auto extend_context = reinterpret_cast<ExtendedKernelContext *>(context);
  auto compute_node_info = extend_context->GetComputeNodeInfo();
  FE_ASSERT_NOTNULL(shapebuffer_addr);
  FE_ASSERT_NOTNULL(compute_node_info);
  const size_t output_size = compute_node_info->GetOutputsNum();
  const size_t shapebuffer_size = output_size * sizeof(uint64_t) * kShapeBufferNum;
  std::unique_ptr<uint8_t[]> host_shapebuffer(new (std::nothrow) uint8_t[shapebuffer_size]);
  if (host_shapebuffer == nullptr) {
    KLOGE("Failed to alloc host memory for op[%s, %s].", compute_node_info->GetNodeName(),
          compute_node_info->GetNodeType());
    return ge::GRAPH_FAILED;
  }
  KLOGD("The value of shapebuffer_size is %zu, shapebuffer_addr->GetAddr() is %zu", shapebuffer_size,
        shapebuffer_addr->GetSize());
  rtError_t ret = rtMemcpy(host_shapebuffer.get(), static_cast<uint64_t>(shapebuffer_size), shapebuffer_addr->GetAddr(),
                           static_cast<uint64_t>(shapebuffer_size), RT_MEMCPY_DEVICE_TO_HOST);
  if (ret != RT_ERROR_NONE) {
    KLOGE("Failed to copy shape buffer data from device to host for op [%s, %s].", compute_node_info->GetNodeName(),
          compute_node_info->GetNodeType());
    return ge::GRAPH_FAILED;
  }
  const auto output_shapes_data_check = reinterpret_cast<uint32_t(*)[kShapeBufferNum]>(host_shapebuffer.get());
  if (GetBitStatu(output_shapes_data_check[0][0], kDataTypeFlagBit) == OutputShapeDtype::ISUINT64) {
    return SetOutputShapeProforUint64(context, output_size, std::move(host_shapebuffer));
  } else {
    return SetOutputShapeProforUint32(context, output_size, std::move(host_shapebuffer));
  }
}
REGISTER_KERNEL(SetOutputShape).RunFunc(SetOutputShape).TracePrinter(SetOutputShapeKernelTrace);
}  // namespace kernel
}  // namespace gert
