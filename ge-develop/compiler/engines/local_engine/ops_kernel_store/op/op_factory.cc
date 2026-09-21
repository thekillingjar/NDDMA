/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "engines/local_engine/ops_kernel_store/op/op_factory.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "graph/op_desc.h"
#include "base/err_msg.h"

namespace ge {
namespace ge_local {
OpFactory &OpFactory::Instance() {
  static OpFactory instance;
  return instance;
}

std::shared_ptr<Op> OpFactory::CreateOp(const Node &node, RunContext &run_context) {
  const std::map<std::string, OP_CREATOR_FUNC>::const_iterator iter = op_creator_map_.find(node.GetType());
  if (iter != op_creator_map_.end()) {
    return iter->second(node, run_context);
  }
  REPORT_INNER_ERR_MSG("E19999", "Not supported OP, type = %s, name = %s",
                     node.GetType().c_str(), node.GetName().c_str());
  GELOGE(FAILED, "[Check][Param] Not supported OP, type = %s, name = %s",
         node.GetType().c_str(), node.GetName().c_str());
  return nullptr;
}

void OpFactory::RegisterCreator(const std::string &type, const OP_CREATOR_FUNC &func) {
  if (func == nullptr) {
    GELOGW("Func is NULL.");
    return;
  }

  const std::map<std::string, OP_CREATOR_FUNC>::const_iterator iter = op_creator_map_.find(type);
  if (iter != op_creator_map_.end()) {
    GELOGW("%s creator already exist", type.c_str());
    return;
  }

  op_creator_map_[type] = func;
  all_ops_.emplace_back(type);
  GELOGI("Register creator for %s.", type.c_str());
}
}  // namespace ge_local
}  // namespace ge
