/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMPILE_PROCESS_NODE_ENGINE_NPU_PROCESS_NODE_ENGINE_H_
#define COMPILE_PROCESS_NODE_ENGINE_NPU_PROCESS_NODE_ENGINE_H_

#include "dflow/compiler/pne/process_node_engine.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/ge_types.h"

namespace ge {
class NPUProcessNodeEngine : public ProcessNodeEngine {
 public:
  Status Initialize(const std::map<std::string, std::string> &options) override;

  Status Finalize() override;

  Status BuildGraph(uint32_t graph_id, ComputeGraphPtr &compute_graph,
                    const std::map<std::string, std::string> &options, const std::vector<GeTensor> &inputs,
                    PneModelPtr &model) override;

  const std::string &GetEngineName() const override;

  void SetImpl(ProcessNodeEngineImplPtr impl) override;
};
}  // namespace ge

#endif  // COMPILE_PROCESS_NODE_ENGINE_NPU_PROCESS_NODE_ENGINE_H_
