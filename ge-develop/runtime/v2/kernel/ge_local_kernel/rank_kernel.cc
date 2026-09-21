/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rank_kernel.h"
#include "securec.h"
#include "graph/ge_error_codes.h"
#include "graph/def_types.h"
#include "graph/utils/math_util.h"
#include "register/kernel_registry.h"
#include "common/checker.h"
#include "exe_graph/runtime/extended_kernel_context.h"
#include "graph/types.h"
#include "common/plugin/ge_make_unique_util.h"
#include "core/utils/tensor_utils.h"
#include "exe_graph/runtime/gert_mem_allocator.h"
#include "exe_graph/runtime/gert_tensor_data.h"

namespace gert {
namespace kernel {
ge::graphStatus RankKernel(KernelContext *context) {
  auto input_shape = context->GetInputPointer<gert::Shape>(static_cast<size_t>(RankKernelInputs::kShapeStart));
  GE_ASSERT_NOTNULL(input_shape);
  auto out_tensor_data =
      context->GetOutputPointer<gert::GertTensorData>(static_cast<size_t>(RankKernelOutputs::kTensorData));
  GE_ASSERT_NOTNULL(out_tensor_data);

  size_t shape_rank = input_shape->GetDimNum();
  auto output_rank = ge::PtrToPtr<void, int32_t>(out_tensor_data->GetAddr());
  GE_ASSERT_NOTNULL(output_rank);
  output_rank[0] = static_cast<int32_t>(shape_rank);
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus BuildRankOutput(const ge::FastNode *node, KernelContext *context) {
  (void) node;
  int32_t data_type_size = ge::GetSizeByDataType(ge::DT_INT32);
  GE_ASSERT_TRUE(!ge::AddOverflow(data_type_size, sizeof(GertTensorData), data_type_size));

  auto chain = context->GetOutput(0U);
  GE_ASSERT_NOTNULL(chain);
  auto out_data = ge::MakeUnique<uint8_t[]>(data_type_size);
  GE_ASSERT_NOTNULL(out_data);
  new (out_data.get())
      GertTensorData(out_data.get() + sizeof(GertTensorData), data_type_size - sizeof(GertTensorData), kOnHost, -1);
  chain->SetWithDefaultDeleter<uint8_t[]>(out_data.release());
  return ge::GRAPH_SUCCESS;
}

REGISTER_KERNEL(RankKernel).RunFunc(RankKernel).OutputsCreator(BuildRankOutput);
}  // namespace kernel
}  // namespace gert
