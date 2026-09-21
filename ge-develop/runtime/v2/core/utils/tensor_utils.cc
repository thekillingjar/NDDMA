/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "core/utils/tensor_utils.h"
#include "common/checker.h"
#include "graph/types.h"
#include "kernel/memory/block_alloc_type.h"
#include "kernel/memory/ti_block_allocator.h"
#include "base/err_msg.h"
namespace {
ge::Status CheckDimInRange(int64_t cur_dim, int64_t lower_bound, int64_t upper_bound) {
  if (lower_bound < 0) {
    GELOGE(ge::FAILED, "lower bound[%ld] is less than 0", lower_bound);
    REPORT_PREDEFINED_ERR_MSG("E13025", std::vector<const ge::char_t *>({"reason"}),
                              std::vector<const ge::char_t *>({("Please check input shape range. Its lower bound " +
                                                                std::to_string(lower_bound) + " is less than 0")
                                                                   .c_str()}));
    return ge::PARAM_INVALID;
  }

  if (lower_bound > upper_bound) {
    GELOGE(ge::FAILED, "lower bound[%ld] is larger than upper bound[%ld]", lower_bound, upper_bound);
    REPORT_PREDEFINED_ERR_MSG("E13025", std::vector<const ge::char_t *>({"reason"}),
                              std::vector<const ge::char_t *>({("Please check input shape range. Its lower bound " +
                                                                std::to_string(lower_bound) + " is greater than upper bound " +
                                                                std::to_string(upper_bound) )
                                                                   .c_str()}));
    return ge::PARAM_INVALID;
  }

  if ((cur_dim > upper_bound) || (cur_dim < lower_bound)) {
    GELOGE(ge::FAILED, "cur dim[%ld] is not in shape range[%ld, %ld]", cur_dim, lower_bound, upper_bound);
    REPORT_PREDEFINED_ERR_MSG("E13025", std::vector<const ge::char_t *>({"reason"}),
                              std::vector<const ge::char_t *>({("Please check input shape. Its current dim " +
                                                                std::to_string(cur_dim) + " is not in shape range[" +
                                                                std::to_string(lower_bound) + ", " +
                                                                std::to_string(upper_bound) + "]")
                                                                   .c_str()}));
    return ge::PARAM_INVALID;
  }
  return ge::SUCCESS;
}
}
namespace gert {
ge::Status TensorUtils::CheckShapeByShapeRange(const Shape &shape, const ShapeRange &shape_range) {
  if (shape.IsScalar() || shape_range.GetMax().IsScalar() || shape_range.GetMin().IsScalar()) {
    GELOGD("Shape or shape range is empty, no need to check.");
    return ge::SUCCESS;
  }

  const auto shape_dim_num = shape.GetDimNum();
  const auto shape_range_max_dim_num = shape_range.GetMax().GetDimNum();
  const auto shape_range_min_dum_num = shape_range.GetMin().GetDimNum();
  if ((shape_dim_num != shape_range_max_dim_num) || (shape_dim_num != shape_range_min_dum_num)) {
    GELOGE(ge::FAILED, "Shape size %zu is different from min shape size %zu or max shape size %zu", shape_dim_num,
           shape_range_min_dum_num, shape_range_max_dim_num);
    const std::string reason = "The shape size " + std::to_string(shape_dim_num) + " is different from min shape size " +
                                std::to_string(shape_range_min_dum_num) + " or max shape size " + std::to_string(shape_range_max_dim_num);
    REPORT_PREDEFINED_ERR_MSG("E13025", std::vector<const ge::char_t *>({"reason"}),
                              std::vector<const ge::char_t *>({reason.c_str()}));
    return ge::PARAM_INVALID;
  }
  for (size_t idx = 0UL; idx < shape_dim_num; idx++) {
    const auto cur_dim = shape.GetDim(idx);
    if (cur_dim == ge::UNKNOWN_DIM) {
      GELOGD("[Check][InputShape]cur shape dim idx [%zu] is -1, no need to check.", idx);
      continue;
    }
    const auto lower_bound = shape_range.GetMin()[idx];
    const auto upper_bound =
        shape_range.GetMax()[idx] == ge::UNKNOWN_DIM ? std::numeric_limits<int64_t>::max() : shape_range.GetMax()[idx];
    if ((lower_bound == 1) && (upper_bound == std::numeric_limits<int64_t>::max())) {
      GELOGD("[Check][InputShape]All range does not check");
      continue;
    }
    GE_ASSERT_SUCCESS(CheckDimInRange(cur_dim, lower_bound, upper_bound));
  }
  return ge::SUCCESS;
}

GertTensorData TensorUtils::ToGertTensorData(memory::MultiStreamMemBlock *block, TensorPlacement placement,
                                             int64_t stream_id) {
  size_t size = 0UL;
  if (block != nullptr) {
    size = block->GetSize();
  }
  return GertTensorData{size, placement, stream_id, block};
}

ge::graphStatus HbmManager(void *block, TensorOperateType operate_type, void **out) {
  GE_ASSERT_NOTNULL(block);
  auto mem_block = reinterpret_cast<ge::MemBlock *>(block);
  if (operate_type == kGetTensorAddress) {
    GE_ASSERT_NOTNULL(out);
    *out = mem_block->GetAddr();
    return ge::GRAPH_SUCCESS;
  }
  if (operate_type == kFreeTensor) {
    mem_block->Free();
    return ge::GRAPH_SUCCESS;
  }
  if (operate_type == kPlusShareCount) {
    mem_block->AddCount();
    return ge::GRAPH_SUCCESS;
  }
  GELOGE(ge::PARAM_INVALID, "Unexpected operate type %d", static_cast<int32_t>(operate_type));
  return ge::PARAM_INVALID;
}

ge::graphStatus HostAddrManager(void *block, TensorOperateType operate_type, void **out) {
  if (block == nullptr) {
    return ge::GRAPH_FAILED;
  }
  auto host_mem = reinterpret_cast<memory::HostMem *>(block);
  if (operate_type == kGetTensorAddress) {
    GE_ASSERT_NOTNULL(out);
    *out = host_mem->GetAddr();
    return ge::GRAPH_SUCCESS;
  }
  if (operate_type == kFreeTensor) {
    if (host_mem->GetCount() > 0U) {
      if (host_mem->SubCount() == 0U) {
        delete host_mem;
      }
    }
    return ge::GRAPH_SUCCESS;
  }
  if (operate_type == kPlusShareCount) {
    host_mem->AddCount();
    return ge::GRAPH_SUCCESS;
  }

  return ge::GRAPH_FAILED;
}

TensorData TensorUtils::ToTensorData(ge::MemBlock *block, size_t tensor_size, TensorPlacement placement) {
  if (TensorPlacementUtils::IsOnDevice(placement)) {
    return TensorData{block, HbmManager, tensor_size, placement};
  }
  return TensorData{block, HostAddrManager, tensor_size, kOnHost};
}

void TensorUtils::ShareGtdToTd(GertTensorData &gtd, TensorData &td) {
  auto msb = reinterpret_cast<memory::MultiStreamMemBlock *>(gtd.GetGertMemBlock());
  if (msb == nullptr) {
    RefGtdToTd(gtd, td);
  } else {
    if (msb->GetAllocType().GetType() == memory::BlockAllocType::kTensorDataWrapped) {
      auto mb = reinterpret_cast<memory::TiBlock *>(msb->GetMemBlock());
      td.ShareFrom(mb->GetTd());
    } else {
      msb->GetMemBlock()->AddCount();
      td = TensorUtils::ToTensorData(msb->GetMemBlock(), gtd.GetSize(), gtd.GetPlacement());
    }
  }
}
ge::graphStatus TensorUtils::ShareTdToGtd(const TensorData &td, GertAllocator &l2_allocator, GertTensorData &gtd) {
  if (td.manager_ == nullptr) {
    GE_ASSERT_SUCCESS(RefTdToGtd(td, l2_allocator.GetStreamId(), gtd));
    return ge::GRAPH_SUCCESS;
  }
  return l2_allocator.ShareFromTensorData(td, gtd);
}
}  // namespace gert
