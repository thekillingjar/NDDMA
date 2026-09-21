/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_COMMON_FORMATS_FORMATS_H_
#define GE_COMMON_FORMATS_FORMATS_H_

#include <vector>
#include "formats/register_format_transfer.h"
#include "graph/types.h"
#include "framework/common/ge_inner_error_codes.h"
#include "graph/ge_tensor.h"

namespace ge {
namespace formats {
/**
 * Convert the data format, and put the converted format and length in the result
 * @param args
 * @param result
 * @return
 */
Status TransDataFormat(const TransArgs &args, TransResult &result);

Status TransTensorShape(const Format src_format, const std::vector<int64_t> &src_shape, const DataType data_type,
                        const Format dst_format, std::vector<int64_t> &dst_shape);

bool IsTransFormatSupport(const TransArgs &args);
}  // namespace formats
}  // namespace ge
#endif  // GE_COMMON_FORMATS_FORMATS_H_
