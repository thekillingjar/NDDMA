/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "npu_process_node_engine.h"
#include "dflow/compiler/pne/process_node_engine_manager.h"
#include "base/err_msg.h"
#include "common/util/trace_manager/trace_manager.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "framework/common/debug/ge_log.h"
#include "graph/partition/base_partitioner.h"

namespace ge {
Status NPUProcessNodeEngine::Initialize(const std::map<std::string, std::string> &options) {
  (void)options;
  engine_id_ = GetEngineName();
  return SUCCESS;
}

Status NPUProcessNodeEngine::Finalize() {
  return SUCCESS;
}

Status NPUProcessNodeEngine::BuildGraph(uint32_t graph_id, ComputeGraphPtr &compute_graph,
                                        const std::map<std::string, std::string> &options,
                                        const std::vector<GeTensor> &inputs, PneModelPtr &model) {
  GE_CHECK_NOTNULL(impl_);
  GE_CHK_STATUS_RET(impl_->BuildGraph(graph_id, compute_graph, options, inputs, model),
                    "Failed to build graph, graph_id=%u, graph_name=%s", graph_id, compute_graph->GetName().c_str());

  if (model != nullptr) {
    std::string logic_dev_id;
    if (AttrUtils::GetStr(compute_graph, ATTR_NAME_LOGIC_DEV_ID, logic_dev_id)) {
      GE_CHK_STATUS_RET(model->SetLogicDeviceId(logic_dev_id), "Failed to set logic device id");
      GELOGD("submodel = %s, logic device id = %s", model->GetModelName().c_str(), logic_dev_id.c_str());
    }
    std::string redundant_logic_dev_id;
    if (AttrUtils::GetStr(compute_graph, ATTR_NAME_REDUNDANT_LOGIC_DEV_ID, redundant_logic_dev_id)) {
      GE_CHK_STATUS_RET(model->SetRedundantLogicDeviceId(redundant_logic_dev_id),
                        "Failed to set redundant logic device id");
      GELOGD("submodel = %s, redundant logic device id = %s", model->GetModelName().c_str(),
             redundant_logic_dev_id.c_str());
    }
  }
  return SUCCESS;
}

const std::string &NPUProcessNodeEngine::GetEngineName() const {
  return PNE_ID_NPU;
}

void NPUProcessNodeEngine::SetImpl(ProcessNodeEngineImplPtr impl) {
  impl_ = impl;
}

REGISTER_PROCESS_NODE_ENGINE(PNE_ID_NPU, NPUProcessNodeEngine);
}  // namespace ge
