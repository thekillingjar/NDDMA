/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ge_graph_dsl/assert/check_utils.h"
#include "graph/utils/dumper/ge_graph_dumper.h"
#include "ge_graph_default_checker.h"
#include "ge_graph_check_dumper.h"

GE_NS_BEGIN

bool CheckUtils::CheckGraph(const std::string &phase_id, const GraphCheckFun &fun) {
  auto &dumper = dynamic_cast<GeGraphCheckDumper &>(GraphDumperRegistry::GetDumper());
  return dumper.CheckFor(GeGraphDefaultChecker(phase_id, fun));
}

void CheckUtils::init() {
  static GeGraphCheckDumper checkDumper;
  GraphDumperRegistry::Register(checkDumper);
}

GE_NS_END
