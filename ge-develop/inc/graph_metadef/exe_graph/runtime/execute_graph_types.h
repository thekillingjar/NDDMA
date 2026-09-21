/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_INC_EXE_GRAPH_RUNTIME_EXECUTE_GRAPH_TYPES_H_
#define METADEF_CXX_INC_EXE_GRAPH_RUNTIME_EXECUTE_GRAPH_TYPES_H_
#include <cstdlib>
namespace gert {
/**
 * 执行图类型，在一个Model中，包含多张执行图，本枚举定义了所有执行图的类型
 */
enum class ExecuteGraphType {
  kInit,    //!< 初始化图，本张图在图加载阶段执行
  kMain,    //!< 主图，每次执行图时，均执行本张图
  kDeInit,  //!< 去初始化图，在图卸载时，执行本张图
  kNum
};

/**
 * 获取执行图的字符串描述
 * @param type 执行图类型枚举
 * @return
 */
inline const char *GetExecuteGraphTypeStr(const ExecuteGraphType type) {
  if (type >= ExecuteGraphType::kNum) {
    return nullptr;
  }
  constexpr const char *kStrs[static_cast<size_t>(ExecuteGraphType::kNum)] = {"Init", "Main", "DeInit"};
  return kStrs[static_cast<size_t>(type)];
}
}  // namespace gert
#endif  // METADEF_CXX_INC_EXE_GRAPH_RUNTIME_EXECUTE_GRAPH_TYPES_H_
