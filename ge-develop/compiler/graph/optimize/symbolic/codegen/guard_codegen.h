/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_COMPILER_GRAPH_OPTIMIZE_SYMBOLIC_CODEGEN_SYMBOL_CHECK_CODEGEN_H_
#define AIR_CXX_COMPILER_GRAPH_OPTIMIZE_SYMBOLIC_CODEGEN_SYMBOL_CHECK_CODEGEN_H_

#include "ge_common/ge_api_types.h"
#include "graph/compute_graph.h"
#include "exe_graph/runtime/tensor.h"

namespace ge {
// GuardCheckFunc函数原型, 返回值true表示检查通过，false表示检查失败
using GuardCheckFunc = bool (*)(
    gert::Tensor **tensors,  // 输入的tensor指针
    size_t num_tensors,           // 输入的tensor个数
    char_t *reason,  // 如果返回值为false，会往reason中写入失败的原因，函数内部不管理reason内存
    size_t reason_size  // reason的大小，单位为字节，不会写入超过reason_size的内容
);

class GuardCodegen {
 public:
  /**
   * 根据在编译过程中在ShapeEnv中产生的Guard关系表达式，产生GuardCheck代码，并编译成so,
   * 编译产生的so通过`_guard_check_so_data`属性打在graph上
   * @param graph：待编译guard check func的计算图
   * @return Status: SUCCESS表示成功
   */
  Status GuardFuncCodegenAndCompile(const ComputeGraphPtr &graph) const;
};
}  // namespace ge

#endif  // AIR_CXX_COMPILER_GRAPH_OPTIMIZE_SYMBOLIC_CODEGEN_SYMBOL_CHECK_CODEGEN_H_