/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_CORE_MODEL_V_2_EXECUTOR_BUILDER_H_
#define AIR_CXX_RUNTIME_V2_CORE_MODEL_V_2_EXECUTOR_BUILDER_H_
#include "runtime/model_v2_executor.h"
#include "exe_graph/runtime/continuous_buffer.h"
#include "common/model/ge_root_model.h"
#include "graph_executor_builder.h"

namespace gert {
class ModelV2ExecutorBuilder {
 public:
  ModelV2ExecutorBuilder() = default;
  explicit ModelV2ExecutorBuilder(RtSession *rt_session) :session_(rt_session) {}
  explicit ModelV2ExecutorBuilder(const ge::ExecuteGraphPtr &exe_graph);
  std::unique_ptr<ModelV2Executor> Build() const;
  std::unique_ptr<ModelV2Executor> Build(ExecutorType executor_type) const;
  std::unique_ptr<ModelV2Executor> Build(const ExecutorOption &executor_option) const;
  ModelV2ExecutorBuilder &ExecuteGraph(const ge::ExecuteGraphPtr &exe_graph);
  ModelV2ExecutorBuilder &ModelData(const ge::ModelData &model_data);
  ModelV2ExecutorBuilder &GeRootModel(const ge::GeRootModelPtr &root_model);

 private:
  const ContinuousBuffer *ReadInBuffer(ModelV2Executor &executor) const;
  const ContinuousBuffer *ReadInComputeNodeInfo(const ContinuousBuffer &buffer, ModelV2Executor &executor) const;
  const ContinuousBuffer *ReadInKernelExtendInfo(const ContinuousBuffer &buffer, ModelV2Executor &executor) const;
  ge::graphStatus ReadInModelDesc(ModelV2Executor &executor) const;
  std::unique_ptr<uint8_t[]> ReadBufferFromAttr(const char *attr_name) const;
  ge::graphStatus RestoreDeviceVarMem(ModelV2Executor &executor) const;
  void SetOutputReuseInputMemIndexes(ModelV2Executor &executor) const;

 private:
  ge::ExecuteGraphPtr exe_graph_;
  ge::ModelData model_data_;
  ge::GeRootModelPtr root_model_;
  RtSession *session_{nullptr};
};
}  // namespace gert

#endif  // AIR_CXX_RUNTIME_V2_CORE_MODEL_V_2_EXECUTOR_BUILDER_H_
