/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ge_graph_dsl/op_desc/op_desc_cfg_repo.h"
#include "framework/common/types.h"
#include "ge_graph_dsl/op_desc/op_desc_cfg.h"

GE_NS_BEGIN

namespace {

#define OP_CFG(optype, ...)                   \
  {                                           \
    optype, OpDescCfg { optype, __VA_ARGS__ } \
  }

static std::map<OpType, OpDescCfg> cfg_repo{OP_CFG(DATA, 1, 1, FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224}),
                                            OP_CFG(ADD, 2, 1, FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224}),
                                            OP_CFG(ENTER, 1, 1, FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224}),
                                            OP_CFG(MERGE, 2, 1, FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224}),
                                            OP_CFG(CONSTANT, 0, 1, FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224}),
                                            OP_CFG(LESS, 2, 1, FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224}),
                                            OP_CFG(LOOPCOND, 1, 1, FORMAT_NCHW, DT_BOOL, {1, 1, 224, 224}),
                                            OP_CFG(SWITCH, 2, 2, FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224}),
                                            OP_CFG(EXIT, 1, 1, FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224}),
                                            OP_CFG(NEXTITERATION, 1, 1, FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224}),
                                            OP_CFG(CONSTANTOP, 0, 1, FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224}),
                                            OP_CFG(GETNEXT, 0, 1, FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224}),
                                            OP_CFG(VARIABLE, 1, 1)};
}  // namespace

const OpDescCfg *OpDescCfgRepo::FindBy(const OpType &id) {
  auto it = cfg_repo.find(id);
  if (it == cfg_repo.end()) {
    return nullptr;
  }
  return &(it->second);
}

GE_NS_END
