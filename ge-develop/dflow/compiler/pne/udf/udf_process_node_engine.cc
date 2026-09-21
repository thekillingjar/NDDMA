/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "udf_process_node_engine.h"

#include "base/err_msg.h"
#include "common/compile_profiling/ge_call_wrapper.h"
#include "common/util/mem_utils.h"
#include "common/util/trace_manager/trace_manager.h"
#include "framework/common/debug/ge_log.h"
#include "graph/utils/graph_utils.h"
#include "dflow/compiler/pne/process_node_engine_manager.h"
#include "udf_model.h"
#include "udf_model_builder.h"

namespace ge {
Status UdfProcessNodeEngine::Initialize(const std::map<std::string, std::string> &options) {
  (void)options;
  return SUCCESS;
}

Status UdfProcessNodeEngine::Finalize() {
  return SUCCESS;
}

Status UdfProcessNodeEngine::BuildGraph(uint32_t graph_id, ComputeGraphPtr &compute_graph,
                                        const std::map<std::string, std::string> &options,
                                        const std::vector<GeTensor> &inputs, PneModelPtr &model) {
  (void)graph_id;
  (void)options;
  (void)inputs;
  UdfModelPtr udf_model = MakeShared<UdfModel>(compute_graph);
  GE_CHECK_NOTNULL(udf_model);
  GE_CHK_STATUS_RET(UdfModelBuilder::GetInstance().Build(*udf_model), "Failed to build udf model.");
  model = udf_model;
  return SUCCESS;
}

REGISTER_PROCESS_NODE_ENGINE(PNE_ID_UDF, UdfProcessNodeEngine);
}  // namespace ge
