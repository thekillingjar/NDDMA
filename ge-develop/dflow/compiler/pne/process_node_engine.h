/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_FRAMEWORK_PROCESS_NODE_ENGINE_H_
#define INC_FRAMEWORK_PROCESS_NODE_ENGINE_H_

#include <map>
#include <string>
#include <vector>

#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/ge_types.h"
#include "graph/ge_tensor.h"
#include "dflow/inc/data_flow/model/pne_model.h"

namespace ge {
class ProcessNodeEngineImpl {
 public:
  virtual ~ProcessNodeEngineImpl() = default;

  virtual Status BuildGraph(uint32_t graph_id, ComputeGraphPtr &compute_graph,
                            const std::map<std::string, std::string> &options, const std::vector<GeTensor> &inputs,
                            PneModelPtr &model) = 0;
};

using ProcessNodeEngineImplPtr = std::shared_ptr<ProcessNodeEngineImpl>;

class ProcessNodeEngine {
 public:
  ProcessNodeEngine() = default;
  virtual ~ProcessNodeEngine() = default;
  ProcessNodeEngine(const ProcessNodeEngine &other) = delete;
  ProcessNodeEngine &operator=(const ProcessNodeEngine &other) = delete;

 public:
  virtual Status Initialize(const std::map<std::string, std::string> &options) = 0;

  virtual Status Finalize() = 0;

  virtual Status BuildGraph(uint32_t graph_id, ComputeGraphPtr &compute_graph,
                            const std::map<std::string, std::string> &options, const std::vector<GeTensor> &inputs,
                            PneModelPtr &model) = 0;

  virtual const std::string &GetEngineName() const = 0;

  virtual void SetImpl(ProcessNodeEngineImplPtr impl) = 0;

 protected:
  std::string engine_id_;
  ProcessNodeEngineImplPtr impl_ = nullptr;
};

using ProcessNodeEnginePtr = std::shared_ptr<ProcessNodeEngine>;
}  // namespace ge

#endif  // INC_FRAMEWORK_PROCESS_NODE_ENGINE_H_
