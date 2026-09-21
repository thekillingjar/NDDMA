/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_KERNEL_COMPATIBLE_UTILS_H_
#define AIR_CXX_RUNTIME_V2_KERNEL_COMPATIBLE_UTILS_H_
#include "graph/op_desc.h"
#include "graph/detail/model_serialize_imp.h"
#include "proto/ge_ir.pb.h"
#include "common/util/mem_utils.h"
#include "graph_metadef/common/ge_common/util.h"

namespace gert {
namespace kernel {
class KernelCompatibleUtils {
 public:
  static ge::graphStatus UnSerializeOpDesc(const void *buffer_head, const size_t buffer_size, ge::OpDescPtr &op_desc) {
    ge::proto::OpDef op_def;
    op_def.ParseFromArray(buffer_head, buffer_size);
    ge::ModelSerializeImp model_serialize_imp;
    if (!model_serialize_imp.UnserializeOpDesc(op_desc, op_def)) {
      GELOGE(ge::GRAPH_FAILED, "Fail to UnserializeOpDesc.");
      return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
  }

  static ge::graphStatus ConvertRTTensorToGeTensor(const Tensor *src_tensor, ge::GeTensorPtr &dst_tensor,
                                                   std::vector<ge::GeTensorPtr> &tensor_holder) {
    GE_CHECK_NOTNULL(src_tensor);
    // convert to ge_tensor
    // build tensor desc
    auto dim_num = src_tensor->GetStorageShape().GetDimNum();
    vector<int64_t> dst_shape;
    for (size_t i = 0U; i < dim_num; ++i) {
      dst_shape.emplace_back(src_tensor->GetStorageShape().GetDim(i));
    }
    ge::GeTensorDesc tensor_desc(ge::GeShape(dst_shape), src_tensor->GetStorageFormat(), src_tensor->GetDataType());
    // build ge tensor
    auto host_addr = static_cast<const uint8_t *>(src_tensor->GetAddr());
    GE_CHECK_NOTNULL(host_addr);
    if (src_tensor->GetDataType() == ge::DT_STRING) {
      dst_tensor = ge::MakeShared<ge::GeTensor>(tensor_desc, host_addr, src_tensor->GetSize());
    } else {
      dst_tensor = ge::MakeShared<ge::GeTensor>(
        tensor_desc, host_addr, ge::GetSizeInBytes(src_tensor->GetShapeSize(), src_tensor->GetDataType()));
    }

    GE_CHECK_NOTNULL(dst_tensor);
    tensor_holder.emplace_back(dst_tensor);
    return ge::GRAPH_SUCCESS;
  }
};
}
}
#endif // AIR_CXX_RUNTIME_V2_KERNEL_COMPATIBLE_UTILS_H_
