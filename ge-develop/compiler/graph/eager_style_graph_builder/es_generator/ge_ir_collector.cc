/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ge_ir_collector.h"
#include "graph_metadef/common/plugin/plugin_manager.h"
#include "graph/opsproto_manager.h"
#include "graph/operator_factory.h"
#include "graph/utils/op_desc_utils.h"

namespace ge {
namespace es {
namespace {
bool g_collected = false;
}
std::vector<OpDescPtr> GeIrCollector::CollectAndCreateAllOps() {
  if (!g_collected) {
    std::string ops_proto_path;
    PluginManager::GetOpsProtoPath(ops_proto_path);
    std::cout << "ops proto path: " << ops_proto_path << std::endl;
    OpsProtoManager::Instance()->Initialize(std::map<std::string, std::string>{{"ge.opsProtoLibPath", ops_proto_path}});
    g_collected = true;
  }

  std::vector<AscendString> all_op_types;
  OperatorFactory::GetOpsTypeList(all_op_types);

  std::vector<OpDescPtr> all_ops;
  for (const auto &op_type : all_op_types) {
    auto op = OperatorFactory::CreateOperator(op_type.GetString(), op_type.GetString());

    auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
    if (op_desc == nullptr) {
      std::cout << "Failed to get op desc for operator %s" << op_type.GetString() << std::endl;
      continue;
    }
    op.BreakConnect();
    all_ops.push_back(op_desc);
  }

  return all_ops;
}
}  // namespace es
}  // namespace ge