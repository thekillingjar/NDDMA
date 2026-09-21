/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "node_checker_register.h"
#include "ge/ge_api_error_codes.h"
#include <array>
namespace ge {
namespace {
std::vector<std::string> type_strs = {"ContinuousInput", "ContinuousOutput", "NoPaddingContinuousInput",
                                      "NoPaddingContinuousOutput"};
}

const char *SpecialNodeTypeStr(const SpecailNodeType &type) {
  if (static_cast<size_t>(type) >= type_strs.size()) {
    return "Unknow type";
  }
  return type_strs[static_cast<size_t>(type)].c_str();
}

std::array<SpecailNodeCheckerFunc, static_cast<size_t>(SpecailNodeType::kSpecialNodeTypeMax)>
    NodeCheckerRegister::funcs;
NodeCheckerRegister::NodeCheckerRegister(const SpecailNodeType &type, const SpecailNodeCheckerFunc &func) {
  if (type >= SpecailNodeType::kSpecialNodeTypeContinuousInput && type < SpecailNodeType::kSpecialNodeTypeMax) {
    funcs[static_cast<size_t>(type)] = func;
  }
}

SpecailNodeCheckerFunc GetSpecialNodeChecker(const SpecailNodeType &type) {
  if (type >= SpecailNodeType::kSpecialNodeTypeContinuousInput && type < SpecailNodeType::kSpecialNodeTypeMax) {
    return NodeCheckerRegister::funcs[static_cast<size_t>(type)];
  }
  return nullptr;
}
}  // namespace ge