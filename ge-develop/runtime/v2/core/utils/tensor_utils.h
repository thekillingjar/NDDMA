/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_CORE_UTILS_TENSOR_UTILS_H_
#define AIR_CXX_RUNTIME_V2_CORE_UTILS_TENSOR_UTILS_H_
#include "exe_graph/runtime/shape.h"
#include "runtime/model_desc.h"
#include "kernel/memory/mem_block.h"
#include "exe_graph/runtime/gert_tensor_data.h"
#include "kernel/memory/multi_stream_mem_block.h"
#include "ge_common/debug/ge_log.h"

namespace gert {
ge::graphStatus HbmManager(void *block, TensorOperateType operate_type, void **out);
ge::graphStatus HostAddrManager(void *block, TensorOperateType operate_type, void **out);
class TensorUtils {
 public:
  static ge::Status CheckShapeByShapeRange(const Shape &shape, const ShapeRange &shape_range);

  // (ms)block to (g)td ...
  static TensorData ToTensorData(ge::MemBlock *block, size_t tensor_size, TensorPlacement placement);
  static GertTensorData ToGertTensorData(memory::MultiStreamMemBlock *block, TensorPlacement placement,
                                         int64_t stream_id);

  // td to gtd ...
  static ge::graphStatus RefTdToGtd(const TensorData &tensor_data, int64_t stream_id, GertTensorData &gtd) {
    auto placement = tensor_data.GetPlacement();
    if (placement == kFollowing) {
      // 当placement为following时，地址是following在Tensor的后面的，作为其中的成员TensorData无法获取到具体地址。
      // todo 下一步可以考虑在创建following的Tensor时，将地址同步设置到TensorData中
      GELOGE(ge::GRAPH_FAILED, "TensorData placement is following, so it can't get address, addr: %p",
             tensor_data.GetAddr());
      return ge::GRAPH_FAILED;
    }
    gtd = {tensor_data.GetAddr(), tensor_data.GetSize(), placement, stream_id};
    return ge::GRAPH_SUCCESS;
  }

  static ge::graphStatus ShareTensorToGtd(const Tensor &tensor, GertAllocator &l2_allocator, GertTensorData &gtd) {
    if (tensor.GetTensorData().manager_ == nullptr) {
      return RefTensorToGtd(tensor, l2_allocator.GetStreamId(), gtd);
    }
    return ShareTdToGtd(tensor.GetTensorData(), l2_allocator, gtd);
  }

  static ge::graphStatus RefTensorToGtd(const Tensor &tensor, int64_t stream_id, GertTensorData &gtd) {
    auto placement = tensor.GetPlacement();
    if (placement == kFollowing) {
      placement = kOnHost;
    }
    gtd = GertTensorData{const_cast<void *>(tensor.GetAddr()), tensor.GetSize(), placement, stream_id};
    return ge::GRAPH_SUCCESS;
  }

  static ge::graphStatus ShareTdToGtd(const TensorData &td, GertAllocator &l2_allocator, GertTensorData &gtd);

  // gtd to td ...
  static void RefGtdToTd(const GertTensorData &gtd, TensorData &td) {
    td = TensorData{gtd.GetAddr(), nullptr, gtd.GetSize(), gtd.GetPlacement()};
  }

  static void ShareGtdToTd(GertTensorData &gtd, TensorData &td);
};
}  // namespace gert
#endif  // AIR_CXX_RUNTIME_V2_CORE_UTILS_TENSOR_UTILS_H_
