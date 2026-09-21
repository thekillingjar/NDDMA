/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/utils/type_utils.h"
#include "base/utils/type_utils_impl.h"

namespace ge {

ge::AscendString TypeUtils::DataTypeToAscendString(const DataType &data_type) {
  return TypeUtilsImpl::DataTypeToAscendString(data_type);
}

DataType TypeUtils::AscendStringToDataType(const ge::AscendString &str) {
  return TypeUtilsImpl::AscendStringToDataType(str);
}

AscendString TypeUtils::FormatToAscendString(const Format &format) {
  return TypeUtilsImpl::FormatToAscendString(format);
}

Format TypeUtils::AscendStringToFormat(const AscendString &str) {
  return TypeUtilsImpl::AscendStringToFormat(str);
}

Format TypeUtils::DataFormatToFormat(const AscendString &str) {
  return TypeUtilsImpl::DataFormatToFormat(str);
}

}