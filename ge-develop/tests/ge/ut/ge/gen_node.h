/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UT_GE_Gen_Node_H_
#define UT_GE_Gen_Node_H_

#include <gtest/gtest.h>

#include "common/debug/log.h"
#include "common/debug/memory_dumper.h"
#include "common/op/ge_op_utils.h"
#include "common/types.h"
#include "graph/compute_graph.h"
#include "graph/op_desc.h"
#include "graph/types.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "register/op_registry.h"

static ge::NodePtr GenNodeFromOpDesc(ge::OpDescPtr op_desc);

static ge::NodePtr GenNodeFromOpDesc(ge::OpDescPtr op_desc) {
  if (!op_desc) {
    return nullptr;
  }
  static auto g = std::make_shared<ge::ComputeGraph>("g");
  return g->AddNode(std::move(op_desc));
}

#endif  // UT_GE_Gen_Node_H_
