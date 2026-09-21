/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GE_LOCAL_ENGINE_ENGINE_HOST_CPU_ENGINE_H_
#define GE_GE_LOCAL_ENGINE_ENGINE_HOST_CPU_ENGINE_H_

#include <mutex>
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/fmk_error_codes.h"
#include "graph/node.h"
#include "graph/gnode.h"
#include "graph/op_desc.h"
#include "graph/operator.h"
#include "graph_metadef/register/register.h"

namespace ge {
class HostCpuEngine {
 public:
  ~HostCpuEngine() = default;

  static HostCpuEngine &GetInstance();

  Status Initialize(const std::string &path_base);

  void Finalize() const;

  static Status Run(const NodePtr &node, HostCpuOp &kernel, const std::vector<ConstGeTensorPtr> &inputs,
                    std::vector<GeTensorPtr> &outputs);

  void *GetConstantFoldingHandle() const { return constant_folding_handle_; }

 private:
  HostCpuEngine() = default;

  Status LoadLib(const std::string &lib_path);

  ge::Status LoadOpsHostCpuNew();

  static Status GetEngineRealPath(std::string &path);

  static Status PrepareInputs(const ConstOpDescPtr &op_desc, const std::vector<ConstGeTensorPtr> &inputs,
                              std::map<std::string, const Tensor> &named_inputs);

  static Status PrepareOutputs(const ConstOpDescPtr &op_desc, std::vector<GeTensorPtr> &outputs,
                               std::map<std::string, Tensor> &named_outputs);

  static Status RunInternal(const OpDescPtr &op_desc, HostCpuOp &op_kernel,
                            const std::map<std::string, const Tensor> &named_inputs,
                            std::map<std::string, Tensor> &named_outputs);

  std::mutex mu_;
  std::vector<void *> lib_handles_;
  void *constant_folding_handle_ = nullptr;
  bool initialized_ = false;
};
}  // namespace ge
#endif  // GE_GE_LOCAL_ENGINE_ENGINE_HOST_CPU_ENGINE_H_
