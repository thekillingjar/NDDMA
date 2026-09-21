/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "concat_offset_kernel.h"

#include <memory>

#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/op/ge_op_utils.h"
#include "framework/common/types.h"
#include "framework/common/debug/ge_log.h"
#include "graph/utils/type_utils.h"
#include "host_kernels/kernel_factory.h"

namespace ge {
namespace {
const size_t kConcatOffsetInputIndexZero = 0;
const size_t kConcatOffsetInputIndexOne = 1;
const int32_t kNumOne = 1;
}  // namespace
Status ConcatOffsetKernel::Compute(const OpDescPtr op_desc_ptr, const std::vector<ConstGeTensorPtr> &input,
                                   std::vector<GeTensorPtr> &v_output) {
  GELOGD("ConcatOffsetKernel in.");
  if (op_desc_ptr == nullptr) {
    GELOGE(PARAM_INVALID, "input opdesc is nullptr.");
    return PARAM_INVALID;
  }
  // validate attrs
  int32_t N = 0;
  if (!(AttrUtils::GetInt(op_desc_ptr, "N", N))) {
    GELOGW("Attr %s does not exist", "N");
    return NOT_CHANGED;
  }
  // follow IR def, the first input is concat_dim
  ConstGeTensorPtr input_0 = input[kConcatOffsetInputIndexZero];
  GE_CHECK_NOTNULL(input_0);
  int32_t concat_dim = *(const_cast<int32_t *>(reinterpret_cast<const int32_t *>(input_0->GetData().data())));
  // validate inputs
  if ((static_cast<int32_t>(input.size()) != (N + kNumOne)) || (input.size() <= kConcatOffsetInputIndexOne)) {
    GELOGW("The number of input for concat offset must be equal to %d, and must be more than one", (N + kNumOne));
    return NOT_CHANGED;
  }

  // calculate ouput dim
  GeShape output_shape = input[kConcatOffsetInputIndexOne]->GetTensorDesc().GetShape();
  int64_t output_size = output_shape.GetShapeSize();
  if (concat_dim >= output_size) {
    GELOGW("Concat dim is bigger than the size of output_shape.");
    return NOT_CHANGED;
  }
  GELOGI("Output shape size is %ld. ", output_size);
  int32_t offset = 0;
  if (output_size < 0) {
    GELOGE(FAILED, "Index is negative.");
    return FAILED;
  }
  unique_ptr<int32_t[]> buf(new (std::nothrow) int32_t[output_size]());
  if (buf == nullptr) {
    GELOGE(MEMALLOC_FAILED, "new buf failed");
    return INTERNAL_ERROR;
  }
  for (size_t i = 0; i < static_cast<size_t>(N); i++) {
    buf[concat_dim] = offset;
    // generate output, index 0 can always gets a GeTensorDesc object from any OpDescPtr.
    auto output_tensor_desc = op_desc_ptr->GetOutputDesc(0);
    GeTensorPtr output_ptr = MakeShared<GeTensor>(output_tensor_desc);
    if (output_ptr == nullptr) {
      GELOGW("Failed to fold node %s, out of memeory", op_desc_ptr->GetName().c_str());
      return NOT_CHANGED;
    }

    output_ptr->MutableTensorDesc().SetDataType(DT_INT32);
    output_ptr->MutableTensorDesc().SetShape(output_shape);
    GE_IF_BOOL_EXEC(output_ptr->SetData(reinterpret_cast<uint8_t *>(buf.get()),
                                        static_cast<size_t>(sizeof(DT_INT32) * output_size)) != GRAPH_SUCCESS,
                    GELOGW("set data failed.");
                    return NOT_CHANGED);
    v_output.push_back(output_ptr);
    // caculate offset
    const int32_t *input_shape =
        reinterpret_cast<const int32_t *>(input[i + kConcatOffsetInputIndexOne]->GetData().data());
    int64_t input_dim = input_shape[concat_dim];  // this index is valid, checked before
    if (input_dim > (INT64_MAX - offset)) {
      GELOGE(PARAM_INVALID, " %d and %ld addition can result in overflow!.", offset, input_dim);
      return INTERNAL_ERROR;
    }
    offset += input_dim;
  }
  GELOGD("ConcatOffsetKernel success");
  return SUCCESS;
}
REGISTER_COMPUTE_NODE_KERNEL(CONCATOFFSET, ConcatOffsetKernel);
}  // namespace ge
