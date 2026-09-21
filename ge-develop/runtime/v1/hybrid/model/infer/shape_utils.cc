/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hybrid/model/infer/shape_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/ge_tensor.h"
#include "common/debug/log.h"
#include "base/err_msg.h"

namespace ge {
namespace hybrid {

Status ShapeUtils::CopyShapeAndTensorSize(const GeTensorDesc &from, GeTensorDesc &to) {
  const auto &shape = from.GetShape();
  const auto &origin_shape = from.GetOriginShape();
  to.SetShape(shape);
  to.SetOriginShape(origin_shape);
  int64_t tensor_size = -1;
  (void)TensorUtils::GetSize(from, tensor_size);
  if (tensor_size <= 0) {
    const auto format = to.GetFormat();
    const DataType data_type = to.GetDataType();
    if (TensorUtils::CalcTensorMemSize(shape, format, data_type, tensor_size) != GRAPH_SUCCESS) {
      GELOGE(FAILED, "CalcTensorMemSize failed for data type: [%d], format [%d], shape [%s]", data_type, format,
             shape.ToString().c_str());
      REPORT_INNER_ERR_MSG("E19999", "CalcTensorMemSize failed for data type: [%d], format [%d], shape [%s]", data_type,
                        format, shape.ToString().c_str());
      return FAILED;
    }
  }
  GELOGD("Update input Shape: [%s] and OriginalShape: [%s], size = %ld",
         shape.ToString().c_str(),
         origin_shape.ToString().c_str(),
         tensor_size);

  TensorUtils::SetSize(to, tensor_size);
  return SUCCESS;
}
} // namespace hybrid
} // namespace ge
