/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "framework/common/ge_format_util.h"
#include "formats/formats.h"

namespace ge {
Status GeFormatUtil::TransShape(const TensorDesc &src_desc, const Format dst_format, std::vector<int64_t> &dst_shape) {
  return formats::TransTensorShape(src_desc.GetFormat(), src_desc.GetShape().GetDims(), src_desc.GetDataType(),
                                   dst_format, dst_shape);
}
}  // namespace ge
