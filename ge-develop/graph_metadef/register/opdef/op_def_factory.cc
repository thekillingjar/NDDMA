/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <algorithm>
#include <map>
#include <set>
#include "register/op_def.h"
#include "op_def_impl.h"
#include "register/op_def_factory.h"

namespace ops {
static std::map<ge::AscendString, OpDefCreator> g_opsdef_creator;
static std::map<ge::AscendString, OpDefFuncPtr> g_opsdef_creator_v2;
static std::vector<ge::AscendString> g_ops_list;
static std::set<ge::AscendString> g_ops_sink_list;

int OpDefFactory::OpDefRegister(const char *name, OpDefCreator creator) {
  g_opsdef_creator.emplace(name, creator);
  g_ops_list.emplace_back(name);
  return 0;
}

int __attribute__((weak)) OpDefFactory::OpDefRegisterV2(const char *name, OpDefFuncPtr ptr) {
  g_opsdef_creator_v2.emplace(name, ptr);
  g_ops_list.emplace_back(name);
  return 0;
}

OpDef OpDefFactory::OpDefCreate(const char *name) {
  auto it_v2 = g_opsdef_creator_v2.find(name);
  if (it_v2 != g_opsdef_creator_v2.cend()) {
    return it_v2->second(name);
  }
  auto it = g_opsdef_creator.find(name);
  if (it != g_opsdef_creator.cend()) {
    return it->second(name);
  }
  return OpDef("default");
}

std::vector<ge::AscendString> &OpDefFactory::GetAllOp(void) {
  return g_ops_list;
}

void OpDefFactory::OpTilingSinkRegister(const char *opType) {
  g_ops_sink_list.emplace(opType);
}

bool OpDefFactory::OpIsTilingSink(const char *opType) {
  return g_ops_sink_list.find(opType) != g_ops_sink_list.end();
}
}  // namespace ops
