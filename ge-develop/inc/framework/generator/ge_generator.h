/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_FRAMEWORK_GENERATOR_GE_GENERATOR_H_
#define INC_FRAMEWORK_GENERATOR_GE_GENERATOR_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "ge/ge_ir_build.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/ge_types.h"
#include "graph/ge_tensor.h"
#include "graph/graph.h"
#include "graph/op_desc.h"
#include "framework/omg/omg_inner_types.h"
#include "graph/detail/attributes_holder.h"

namespace ge {
const std::string kAttrSupportDynamicShape = "support_dynamicshape";

class GeRootModel;
class GE_FUNC_VISIBILITY GeGenerator {
 public:
  using InOutTensorRef = std::pair<const std::vector<ge::GeTensor> &, const std::vector<ge::GeTensor> &>;
  static GeGenerator &GetInstance() {
    static GeGenerator Instance;
    return Instance;
  }
  GeGenerator() = default;

  ~GeGenerator() { (void)Finalize(); }

  GeGenerator(const GeGenerator &) = delete;

  GeGenerator &operator=(const GeGenerator &) = delete;

  Status Initialize(const std::map<std::string, std::string> &options);
  Status Initialize(const std::map<std::string, std::string> &options, OmgContext &context);

  Status Finalize();
  Status GenerateOfflineModel(const Graph &graph, const std::string &file_name_prefix,
                              const std::vector<GeTensor> &inputs = std::vector<GeTensor>(),
                              OfflineModelFormat om_format = OfflineModelFormat::OM_FORMAT_DEFAULT);

  Status GenerateOnlineModel(const Graph &graph, const std::vector<GeTensor> &inputs, ge::ModelBufferData &model);

  Status GenerateInfershapeGraph(const Graph &graph);

  ///
  /// @ingroup ge
  /// @brief: Build single OP in Model.
  /// @param [in] op_desc: the OP description.
  /// @param [in] inputs: input tensors.
  /// @param [in] outputs: output tensors.
  /// @param [in] model_file_name: name of model file.
  /// @param [in] compile_flag: op build flag, accurate build is 0, fuzz build is 1
  /// @return SUCCESS or FAILED
  ///
  Status BuildSingleOpModel(OpDescPtr &op_desc, const std::vector<GeTensor> &inputs,
                            const std::vector<GeTensor> &outputs, const std::string &model_file_name,
                            int32_t compile_flag = 0);
  ///
  /// @ingroup ge
  /// @brief: Build single Op into model buff.
  /// @param [in] op_desc: the OP description.
  /// @param [in] inputs: input tensors.
  /// @param [in] outputs: output tensors.
  /// @param [in] engine_type: engine type.
  /// @param [in] compile_flag: op build flag, accurate build is 0, fuzz build is 1
  /// @param [out] model_buff: model buff of op.
  /// @return SUCCESS or FAILED
  Status BuildSingleOpModel(OpDescPtr &op_desc, const std::vector<GeTensor> &inputs,
                            const std::vector<GeTensor> &outputs,
                            OpEngineType engine_type, ModelBufferData &model_buff);
  Status BuildSingleOpModel(OpDescPtr &op_desc, const std::vector<GeTensor> &inputs,
                            const std::vector<GeTensor> &outputs,
                            OpEngineType engine_type, int32_t compile_flag, ModelBufferData &model_buff);
  Status BuildSingleOpModel(OpDescPtr &op_desc, const std::vector<GeTensor> &inputs,
                            const std::vector<GeTensor> &outputs, OpEngineType engine_type,
                            int32_t compile_flag, ModelBufferData &model_buff,
                            GraphStage graph_stage, ComputeGraphPtr &compute_graph);

  ///
  /// @ingroup ge
  /// @brief: Build single Op into model buff.
  /// @param [in] op_desc: the OP description.
  /// @param [in] inputs: input tensors.
  /// @param [in] outputs: output tensors.
  /// @param [in] graph_name: graph name.
  /// @param [out] graph: graph of single op.
  /// @return SUCCESS or FAILED
  Status BuildSingleOpGraph(const OpDescPtr &op_desc, const InOutTensorRef &inputs_outputs,
                            std::string graph_name, Graph &graph,
                            std::vector<std::pair<std::string, std::string>> &inputs_name_type) const;
  Status BuildOriginalGraphInfo(OpDescPtr &op_desc, const std::vector<GeTensor> &inputs,
                                const std::vector<GeTensor> &outputs, const std::string &model_file_name,
                                bool is_offline, GraphStage graph_stage, Graph &graph, ComputeGraphPtr &compute_graph,
                                std::vector<std::pair<std::string, std::string>> &inputs_name_type);
  // temp solution, ge_ir should share one session between some generators, delete after ge_ir use session manager
  Status SetCurrentSessionId(const uint64_t session_id) const;
  // temp solution, ge_ir should share one rebuild_ctrl between some generators, delete after ge_ir use session manager
  Status SetExternalGraphRebuildStateCtrl(void *rebuild_ctrl) const;
  using GeRootModelPtr = std::shared_ptr<ge::GeRootModel>;
  static Status SetModelNameForDump(const GeRootModelPtr &ge_root_model);
 private:
  Status GenerateModel(const Graph &graph, const std::string &file_name_prefix,
                       const std::vector<GeTensor> &inputs,
                       ge::ModelBufferData &model, bool is_offline = true,
                       OfflineModelFormat om_format = OfflineModelFormat::OM_FORMAT_DEFAULT);
  Status BuildSingleOp(OpDescPtr &op_desc, const std::vector<GeTensor> &inputs, const std::vector<GeTensor> &outputs,
                       const std::string &model_file_name, OpEngineType engine_type, ModelBufferData &model_buff,
                       ComputeGraphPtr &comp_graph, bool is_offline = true, int32_t compile_flag = 0,
                       GraphStage graph_stage = GraphStage::GRAPH_STAGE_RESERVED);
  static Status CheckEngineTypeSupport(const NodePtr &node, OpEngineType engine_type);
  static void RemoveConst(const std::vector<GeTensor> &inputs, std::vector<GeTensor> &outputs);
  static Status CheckForSingleOp(const OpDescPtr &op_desc, const std::vector<GeTensor> &inputs,
                                 const std::vector<GeTensor> &outputs);
  static Status InferFormatForSingleOp(const OpDescPtr &op_desc, const Graph &graph);
  static Status ResetAiCpuToDynamicShape(const ComputeGraphPtr &graph);

  static Status CreateGeneralizedBuildAttrs(const GeRootModelPtr &ge_root_model,
                                            const std::vector<GeTensor> &inputs,
                                            const std::vector<GeTensor> &outputs,
                                            const std::vector<std::pair<std::string, std::string>> &inputs_name_type,
                                            std::vector<ge::NamedAttrs> &generalized_build_attrs);
  static void AddExcludeEnginesOption(const OpDescPtr &op_desc, std::map<std::string, std::string> &graph_options);
  void AddShapeGeneralizedOption(std::map<std::string, std::string> &graph_options) const;
  void SetFuzzCompile(const std::vector<GeTensor> &inputs, int32_t compile_flag) const;
  bool IsFuzzCompileEnable() const;
  void ConvertOpInfosToOptions(const OpDescPtr &op_desc) const;
  static Status ResetInputOutputShape(const ComputeGraphPtr &graph,
                                      const std::vector<std::pair<std::string, std::string>> &inputs_name_type,
                                      std::vector<GeTensor> &inputs_dynamic,
                                      std::vector<GeTensor> &outputs_dynamic);
  static Status ResetOutputShapeRange(const OpDescPtr &op_desc, const size_t index,
                                      std::vector<std::pair<int64_t, int64_t>> &shape_range);
  static Status ResetTensorDesc(const size_t index, const GeShape &data_shape, std::vector<GeTensor> &vector_dynamic,
                                std::vector<std::pair<int64_t, int64_t>> &dynamic_shape_range);

  class Impl;
  std::shared_ptr<Impl> impl_;
};
}  // namespace ge

#endif  // INC_FRAMEWORK_GENERATOR_GE_GENERATOR_H_
