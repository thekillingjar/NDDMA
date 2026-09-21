/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_REGISTER_GRAPH_OPTIMIZER_FUSION_COMMON_FUSION_PASS_DESC_H_
#define INC_REGISTER_GRAPH_OPTIMIZER_FUSION_COMMON_FUSION_PASS_DESC_H_
#include <stdint.h>
#include "graph/option/optimization_option_info.h"
namespace fe {
using PassAttr = uint64_t;
const PassAttr FORBIDDEN_CLOSE = 0x01UL;  // forbidden close, can not be closed by fusion switch
const PassAttr NEED_SORT = 0x02UL;  // need topological sorting before executing
const PassAttr SINGLE_SCENE_OPEN = 0x04UL;  // open for single op scene, can be close by fusion switch
const PassAttr FE_PASS = 0x08UL;  // graph passes and ub passes in air project
constexpr PassAttr ENABLE_AUTO_FUSION = 0x10UL;  // whether using auto match fusion frame
constexpr PassAttr ALWAYS_GENERALIZE = 0x20UL;
constexpr PassAttr PRUNING = 0x40UL; // whether the pass need to pre run before graph fusion
constexpr PassAttr ENABLE_FUSION_CHECK = 0x80UL; // enable do fusion check for matched fusion nodes during ub fusion
/*
 * Compile level reg, if reg multi level, the lowest level will take effect.
 * */
constexpr PassAttr COMPILE_LEVEL_O0 = 0x100UL; // pure dynamic
constexpr PassAttr COMPILE_LEVEL_O1 = 0x200UL; // static functional optimize
constexpr PassAttr COMPILE_LEVEL_O2 = 0x400UL; // no time and space balance optimize
constexpr PassAttr COMPILE_LEVEL_O3 = 0x800UL; // open all optimize
constexpr PassAttr PASS_BIT_MASK = 0x1UL;  // check if the loweset bit of pass is 1

enum class PassAttrType : int32_t {
  FRBDN_CLOSE = 0, // Mark those passes that cannot be turned off in graph mode
  NEED_TOPO_SORT = 1, // Mark those graph fusion passes that need topological sorting before executing
  SINGLE_OP_SCENE_MUST_ON = 2, // Mark those passes that must be turned on in single-op mode or jit_compile=false
  FE_PASS_FLAG = 3, // Mark those passes that belong to FE
  AUTO_FUSION_FLAG = 4, // Using auto match fusion frame
  /* The OpDescs in the patterns of this kind fusion pass are able to be generalized in all scenarios.
   * For example, they can ignore the value dependency restrict. */
  ALWAYS_GENERALIZE_FLAG = 5,
  PRUNING_FLAG = 6,
  FUSION_CHECK_FLAG = 7,  // Do fusion check for matched fusion nodes during ub fusion
  COMPILE_O0 = 8,
  COMPILE_O1 = 9,
  COMPILE_O2 = 10,
  COMPILE_O3 = 11,
};
bool IsPassAttrTypeOn(PassAttr pass_attr, PassAttrType attr_type);
void RegPassCompileLevel(const std::string &pass_name, PassAttr pass_attr);
}  // namespace fe
#endif  // INC_REGISTER_GRAPH_OPTIMIZER_FUSION_COMMON_FUSION_PASS_DESC_H_
