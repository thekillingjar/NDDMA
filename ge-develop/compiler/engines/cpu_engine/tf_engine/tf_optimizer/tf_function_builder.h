/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_TF_FUNCTION_BUILDER_H_
#define AICPU_TF_FUNCTION_BUILDER_H_

#include <string>
#include <vector>
#include <unordered_map>
#include "proto/tensorflow/graph.pb.h"
#include "graph/op_desc.h"
#include "graph/compute_graph.h"
#include "framework/common/ge_inner_error_codes.h"

namespace aicpu {
using NameRangeMap = std::unordered_map<string, std::pair<int, int>>;
class TfFunctionBuilder {
public:
  /**
   * Get tf function def builder instance
   * @return TfFunctionBuilder reference
   */
  static TfFunctionBuilder& GetInstance();

  /**
   * Build tf function def from ge compute graph
   * @param graph Ge Compute graph
   * @param func_name Function name
   * @param func_lib Function library
   * @param fused_node_def Fused ge node def
   * @param in_anchors Input data anchors
   * @param out_anchors Output data anchors
   * @return status whether this operation success
   */
  ge::Status BuildFunctionDef(ge::ComputeGraphPtr &graph,
                              const string &func_name,
                              domi::tensorflow::FunctionDefLibrary *func_lib,
                              domi::tensorflow::NodeDef *fused_node_def,
                              std::vector<ge::InDataAnchorPtr> &in_anchors,
                              std::vector<ge::OutDataAnchorPtr> &out_anchors) const;

private:
  /**
   * Create input nodes for graph
   * @param graph Ge Compute graph
   * @param in_anchors Input data anchors
   * @param arg_data_types Input arg data types
   * @return status whether this operation success
   */
  ge::Status CreateInputNodes(ge::ComputeGraphPtr &graph,
                              const std::vector<ge::InDataAnchorPtr> &in_anchors,
                              std::vector<domi::tensorflow::DataType> &arg_data_types) const;

  /**
   * Create output nodes for graph
   * @param graph Ge Compute graph
   * @param out_anchors Output data anchors
   * @param res_data_types Result data types
   * @return status whether this operation success
   */
  ge::Status CreateOutputNodes(ge::ComputeGraphPtr &graph,
                               const std::vector<ge::OutDataAnchorPtr> &out_anchors,
                               std::vector<domi::tensorflow::DataType> &res_data_types) const;

  /**
   * Transform ge graph to tf function def
   * @param graph Ge Compute graph
   * @param func_name Function name
   * @param func_def Function def
   * @return status whether this operation success
  */
  ge::Status TransGraphToFunctionDef(ge::ComputeGraphPtr &graph,
                                     const string &func_name,
                                     domi::tensorflow::FunctionDef *func_def) const;

  /**
   * Set input for node def
   * @param node_def Node def
   * @param in_anchors Input data anchors
  */
  void SetInputForNodeDef(domi::tensorflow::NodeDef *node_def, std::vector<ge::InDataAnchorPtr> &in_anchors) const;

  /**
   * Re-range output name for tf node
   * @param node_def Node def
   * @param op_def Op def
   * @param outputs Output arg name re-range map
   * @return status whether this operation success
  */
  __attribute__((visibility("hidden")))
  ge::Status NameRangesForNode(const domi::tensorflow::NodeDef &node_def,
                               const domi::tensorflow::OpDef &op_def,
                               NameRangeMap &outputs) const;

  /**
   * Re-range output name for tf function
   * @param node Ge node
   * @param tf_node_def tf node def
   * @param outputs Output arg name re-range map
   * @return status whether this operation success
  */
  __attribute__((visibility("hidden")))
  ge::Status NameRangesForFunc(const ge::NodePtr &node,
                               domi::tensorflow::NodeDef &tf_node_def,
                               NameRangeMap &outputs) const;

  /**
   * Compute arg range for arg
   * @param node_def Node def
   * @param arg_def Arg def
   * @param num Attr value number
   * @return status whether this operation success
  */
  ge::Status ComputeArgRange(const domi::tensorflow::NodeDef &node_def,
                             const domi::tensorflow::OpDef::ArgDef &arg_def,
                             int *num) const;

  /**
   * Remap function def
   * @param func_def Function def
   * @param func_name Function name
   * @param tensor_renaming_map Tensor renaming map
   * @param return_value_map Return value map
   * @return status whether this operation success
  */
  __attribute__((visibility("hidden")))
  ge::Status RemapFunctionDef(domi::tensorflow::FunctionDef *func_def,
                              const std::string &func_name,
                              std::unordered_map<std::string, std::string> &tensor_renaming_map,
                              std::unordered_map<std::string, std::string> &return_value_map) const;
  /**
   * Get tf output name range map
   * @param node Ge node
   * @param tf_node_def Tensorflow node def
   * @param outputs Output name range map
   * @param node_def_from_ir_mapping Whether nodedef from ir mapping
   * @return status whether this operation success
  */
  __attribute__((visibility("hidden")))
  ge::Status GetTfOutputNameRangeMap(const ge::NodePtr &node,
                                      domi::tensorflow::NodeDef &tf_node_def,
                                      NameRangeMap &outputs,
                                      bool &node_def_from_ir_mapping) const;

  /**
   * Get tf node def
   * @param node Ge node
   * @param tf_node_def Tf node def
   * @param from_ir_mapping Whether nodedef tranform from ir mapping
   * @return status whether this operation success
  */
  ge::Status GetTfNodeDef(const ge::NodePtr &node, domi::tensorflow::NodeDef &tf_node_def, bool &from_ir_mapping) const;

  /**
   * Contructor
  */
  TfFunctionBuilder() = default;

  /**
   * Destructor
  */
  ~TfFunctionBuilder() = default;

  // Copy prohibit
  TfFunctionBuilder(const TfFunctionBuilder &tf_function_builder) = delete;
  // Move prohibit
  TfFunctionBuilder(TfFunctionBuilder &&tf_function_builder) = delete;
  // Copy prohibit
  TfFunctionBuilder& operator=(const TfFunctionBuilder &tf_function_builder) = delete;
  // Move prohibit
  TfFunctionBuilder& operator=(TfFunctionBuilder &&tf_function_builder) = delete;
};
} // namespace aicpu
#endif // AICPU_TF_FUNCTION_BUILDER_H_

