/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_GE_RT_CONST_TYPES_H
#define AIR_CXX_GE_RT_CONST_TYPES_H
#include <string>
#include "graph/types.h"
namespace gert {
// ConstData节点所表示的类型
// ConstData不同于Data, 它将含有执行时信息语义，因此类型需要在lowering时确定
// 如果需要新增ConstData类型，请在下面的枚举中增加, 同时需要在kConstDataTypes对应增加key
enum class ConstDataType {
  kRtSession = 0,
  KWeight,
  kModelDescription,
  kTypeEnd,
};

// ConstData节点的unique key，用于在global data中索引对应的holder。
const ge::char_t* const kConstDataTypes[] = {"RtSession", "OuterWeightMem", "ModelDesc"};
inline std::string GetConstDataTypeStr(ConstDataType type) {
  auto len = sizeof(kConstDataTypes) / sizeof(ge::char_t *);
  if (static_cast<size_t>(type) >= len) {
    return "";
  }
  return kConstDataTypes[static_cast<int>(type)];
}
}
#endif  // AIR_CXX_GE_RT_CONST_TYPES_H
