/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UDF_UDF_PROCESS_NODE_ENGINE_H
#define UDF_UDF_PROCESS_NODE_ENGINE_H

#include "dflow/compiler/pne/process_node_engine.h"

namespace ge {
class UdfProcessNodeEngine : public ProcessNodeEngine {
 public:
  UdfProcessNodeEngine() {
    engine_id_ = PNE_ID_UDF;
  }
  ~UdfProcessNodeEngine() override = default;

  Status Initialize(const std::map<std::string, std::string> &options) override;

  Status Finalize() override;

  Status BuildGraph(uint32_t graph_id, ComputeGraphPtr &compute_graph,
                    const std::map<std::string, std::string> &options, const std::vector<GeTensor> &inputs,
                    PneModelPtr &model) override;

  const std::string &GetEngineName() const override {
    return engine_id_;
  }

  void SetImpl(ProcessNodeEngineImplPtr impl) override {
    impl_ = impl;
  }
};
}  // namespace ge

#endif  // UDF_UDF_PROCESS_NODE_ENGINE_H_
