/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * Migration Guide: ge_api.h -> ge_api_v2.h
 *
 * ge_api_v2.h replaces ge_api.h, and libge_runner_v2.so replaces libge_runner.so.
 * The primary goals are to decouple from DataFlow interfaces and optimize compilation
 * and execution interfaces.
 *
 * Key Changes:
 *
 * 1. REMOVED INTERFACES:
 *    - All Feed/Fetch-prefixed interfaces (will be provided by DataFlow)
 *    - Interfaces with std::string parameters
 *    - SetGraphFixedFeatureMemoryBase/GetVariables/PaRemapped
 *    - AddGraphWithCopy renamed to AddGraphClone
 *
 * 2. UPGRADED INTERFACES:
 *    - GEInitializeV2, GEFinalizeV2 (replace GEInitialize, GEFinalize)
 *    - GEGetErrorMsgV3, GEGetWarningMsgV3 (replace GEGetErrorMsgV2, GEGetWarningMsgV2)
 *
 * 3. SESSION CLASS REPLACEMENT:
 *    - Session class replaced by new GeSession class
 *    - Implemented in libge_runner_v2.so
 *
 * 4. STREAMLINED COMPILATION & EXECUTION:
 *    - Merged two compilation interfaces (BuildGraph and CompileGraph)
 *    - Merged two WithStreamAsync execution interfaces
 *
 *    New interface signatures:
 *      Status CompileGraph(uint32_t graph_id, const std::vector<ge::Tensor> &inputs)
 *      Status LoadGraph(uint32_t graph_id, const std::map<AscendString, AscendString> &options, void *stream) const
 *      Status RunGraph(uint32_t graph_id, const std::vector<gert::Tensor> &inputs, std::vector<gert::Tensor> &outputs)
 *      Status RunGraphAsync(uint32_t graph_id, const std::vector<gert::Tensor> &inputs, RunAsyncCallbackV2 callback)
 *      Status RunGraphWithStreamAsync(uint32_t graph_id, void *stream,
 *                                     const std::vector<gert::Tensor> &inputs, std::vector<gert::Tensor> &outputs)
 *
 * 5. EXECUTION MODEL CHANGES:
 *    - Compile and Load steps are now optional for all three execution modes
 *    - The three execution modes are mutually exclusive and cannot be mixed
 *    - Added explicit constraints and validation checks
 *
 * Note: Link against libge_runner_v2.so instead of libge_runner.so
 * For detailed migration guide, see documentation
* ===================================================================================================================*/

#ifndef INC_EXTERNAL_GE_GE_API_V2_H_
#define INC_EXTERNAL_GE_GE_API_V2_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ge/ge_api_error_codes.h"
#include "ge/ge_api_types.h"
#include "ge/ge_graph_compile_summary.h"
#include "ge/ge_ir_build.h"
#include "graph/graph.h"
#include "graph/tensor.h"
#include "ge/ge_allocator.h"
#include "exe_graph/runtime/tensor.h"
namespace ge {
using RunCallback = std::function<Status(uint32_t graph_id, const std::map<AscendString, gert::Tensor>& params_list)>;
// Initialize GE
GE_FUNC_VISIBILITY Status GEInitializeV2(const std::map<AscendString, AscendString> &options);
// Finalize GE, release all resources
GE_FUNC_VISIBILITY Status GEFinalizeV2();
GE_FUNC_VISIBILITY ge::AscendString GEGetErrorMsgV3();
GE_FUNC_VISIBILITY ge::AscendString GEGetWarningMsgV3();

class GE_FUNC_VISIBILITY GeSession {
 public:
  explicit GeSession(const std::map<AscendString, AscendString> &options);
  ~GeSession();

  ///
  /// @ingroup client
  /// @brief add a graph with a specific graph id
  /// @param [in] graph_id graph id
  /// @return Status result of function
  ///
  Status AddGraph(uint32_t graph_id, const Graph &graph);

  ///
  /// @ingroup client
  /// @brief add a graph with a specific graphId and graphOptions
  /// @param [in] graphId graph id
  /// @param [in] graph the graph
  /// @param [in] options graph options
  /// @return Status result of function
  ///
  Status AddGraph(uint32_t graph_id, const Graph &graph, const std::map<AscendString, AscendString> &options);

  ///
  /// @ingroup client
  /// @brief add a copy graph with a specific graphId
  /// @param [in] graphId graph id
  /// @param [in] graph the graph
  /// @return Status result of function
  ///
  Status AddGraphClone(uint32_t graph_id, const Graph &graph);

  ///
  /// @ingroup client
  /// @brief add a copy graph with a specific graphId and graphOptions
  /// @param [in] graphId graph id
  /// @param [in] graph the graph
  /// @param [in] options graph options
  /// @return Status result of function
  ///
  Status AddGraphClone(uint32_t graph_id, const Graph &graph, const std::map<AscendString, AscendString> &options);

  ///
  /// @ingroup ge_graph
  /// @brief remove a graph
  /// @param [in] graph_d graph id
  /// @return Status result of function
  ///
  Status RemoveGraph(uint32_t graph_id);

  ///
  /// @ingroup ge_graph
  /// @brief check if the graph need to be recompiled
  /// @param [in] graph_d graph id
  /// @return true means need to be recompiled
  ///
  bool IsGraphNeedRebuild(uint32_t graph_id);

  ///
  /// @ingroup ge_graph
  /// @brief get session id
  /// @return session id
  ///
  uint64_t GetSessionId() const;

  /// @ingroup ge_graph
  /// @brief compile graph
  /// @param [in] graphId: graph id
  /// @return Status result of function
  Status CompileGraph(uint32_t graph_id);

  /// @ingroup ge_graph
  /// @brief compile graph
  /// @param [in] graphId: graph id
  /// @param [in] inputs: input data, It can be empty. If it is not empty, it is required
  /// to be the same as the number of model inputs, which is used to update the shape,
  /// data type, and format of the model input, i.e. the Data node
  /// @return Status result of function
  Status CompileGraph(uint32_t graph_id, const std::vector<ge::Tensor> &inputs);

  ///
  /// @ingroup ge_graph
  /// @brief load a graph with specific stream
  /// @param [in] graph_id graph id
  /// @param [in] stream specific stream
  /// @param [in] options graph options
  /// @return Status result of function
  ///
  Status LoadGraph(const uint32_t graph_id, const std::map<AscendString, AscendString> &options, void *stream) const;

  ///
  /// @ingroup ge_graph
  /// @brief run a graph synchronously
  /// @param [in] graphId graph id
  /// @param [in] inputs input data
  /// @param [out] outputs output data
  /// @return Status result of function
  ///
  Status RunGraph(uint32_t graph_id, const std::vector<gert::Tensor> &inputs, std::vector<gert::Tensor> &outputs);

  ///
  /// @ingroup ge_graph
  /// @brief run a graph with specific stream asynchronously, will launch a model execute task on stream
  /// @param [in] graph_id graph id
  /// @param [in] stream specific stream
  /// @param [in] inputs input data
  /// @param [out] outputs output data
  /// @return Status result of function
  ///
  Status RunGraphWithStreamAsync(uint32_t graph_id, void *stream, const std::vector<gert::Tensor> &inputs,
                                 std::vector<gert::Tensor> &outputs);

  ///
  /// @ingroup ge_graph
  /// @brief run graph asynchronously
  /// @param [in] graphId: graph id
  /// @param [in] inputs: input data
  /// @param [out] callback: callback while runing graph has been finished.
  ///                        The callback function will not be checked.
  ///                        Please ensure that the implementation of the function is trusted.
  /// @return Status result of function
  ///
  Status RunGraphAsync(uint32_t graph_id, const std::vector<gert::Tensor> &inputs, RunAsyncCallbackV2 callback);

  ///
  /// @ingroup ge_graph
  /// @brief register callback func with specific summary or checkpoint by users
  /// @param [in] key: func key
  /// @param [in] callback: callback  specific summary or checkpoint.
  ///                       The callback function will not be checked.
  ///                       Please ensure that the implementation of the function is trusted.
  /// @return Status result of function
  ///
  Status RegisterCallBackFunc(const char *key, const RunCallback &callback);

  ///
  /// @ingroup ge_graph
  /// @brief get graph resource summary after compiled
  /// @param [in] graphId: graph id
  /// @return share_ptr of CompiledGraphSummary
  ///
  CompiledGraphSummaryPtr GetCompiledGraphSummary(uint32_t graph_id);

  ///
  /// @ingroup ge_graph
  /// @brief set const memory base after compiled and before loaded, only allows setting once
  /// @param [in] graphId graph id
  /// @param [in] memory const memory base
  /// @param [out] size const memory size
  /// @return Status result of function
  ///
  Status SetGraphConstMemoryBase(uint32_t graph_id, const void *const memory, size_t size);

  ///
  /// @ingroup ge_graph
  /// @brief set or update fearture memory base after compiled
  /// @param [in] graphId graph id
  /// @param [in] memory feature map memory base, without input and output mem
  /// @param [out] size feature map memory size
  /// @return Status result of function
  ///
  Status UpdateGraphFeatureMemoryBase(uint32_t graph_id, const void *const memory, size_t size);

  ///
  /// @ingroup ge_graph
  /// @brief set fix feature memory base with type after compiled and before loaded, one type only allows setting once
  /// @param [in] graph_id graph id
  /// @param [in] type memory type
  /// @param [in] memory const memory base
  /// @param [out] size const memory size
  /// @return Status result of function
  ///
  Status SetGraphFixedFeatureMemoryBaseWithType(uint32_t graph_id, MemoryType type, const void *const memory,
                                                size_t size);

  ///
  /// @ingroup ge_graph
  /// @brief set or update tefreshable fearture memory base after compiled, not include fix memory
  /// @param [in] graphId graph id
  /// @param [in] memory feature map memory base, without input and output mem
  /// @param [out] size feature map memory size
  /// @return Status result of function
  ///
  Status UpdateGraphRefreshableFeatureMemoryBase(uint32_t graph_id, const void *const memory, size_t size);

  /// @ingroup ge_graph
  /// @brief register external allocator to GE.
  /// @param [in] stream stream handle
  /// @param [in] allocator_obj allocator object handle
  /// @return Status result of function
  Status RegisterExternalAllocator(const void *const stream, AllocatorPtr allocator) const;

  /// @ingroup ge_graph
  /// @brief unregister external allocator to GE.
  /// @param [in] stream stream handle
  /// @return Status result of function
  Status UnregisterExternalAllocator(const void *const stream) const;

  /// @ingroup ge_graph
  /// @brief obtain the compiled model
  /// @param [in] graph_id the graph id
  /// @param [in] model_buffer the modelbufferdata
  /// @return Status result of function
  ///
  Status GetCompiledModel(uint32_t graph_id, ModelBufferData &model_buffer);

 private:
  class Impl;
  std::shared_ptr<Impl> impl_;
};
}  // namespace ge

/// Support annotating GE nodes with the inference_rule attribute.
/// Inference logic for a node’s output shape and dtype
#define INFERENCE_RULE "_inference_rule"

extern "C" {
/// GE enhances its compilation capability by extending the IR representation
/// or by introducing new compiler options. 
/// Add feature capability query support between APP and GE. 
bool IsIrRepSupport(const char *rep);

ge::Status GetRegisteredIrDef(const char *op_type, std::vector<std::pair<ge::AscendString, ge::AscendString>> &inputs,
                              std::vector<std::pair<ge::AscendString, ge::AscendString>> &outputs,
                              std::vector<std::pair<ge::AscendString, ge::AscendString>> &attrs);
} // extern "C"

#endif  // INC_EXTERNAL_GE_GE_API_V2_H_
