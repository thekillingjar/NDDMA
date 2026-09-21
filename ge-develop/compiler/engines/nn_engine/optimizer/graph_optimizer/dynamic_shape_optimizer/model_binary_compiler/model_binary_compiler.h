/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_GRAPH_OPTIMIZER_DYNAMIC_SHAPE_OPTIMIZER_MODEL_BINARY_COMPILER_MODEL_BINARY_COMPILER_H_
#define COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_GRAPH_OPTIMIZER_DYNAMIC_SHAPE_OPTIMIZER_MODEL_BINARY_COMPILER_MODEL_BINARY_COMPILER_H_

#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/fe_utils.h"
#include "common/fe_error_code.h"
#include "common/graph/fe_graph_utils.h"
#include "common/sgt_slice_type.h"
#include "graph/compute_graph.h"
#include "graph/op_kernel_bin.h"

namespace fe {
using KernelLookup = std::function<ge::OpKernelBinPtr(const std::string &kernel_name)>;
class ModelBinaryCompiler {
 public:
  ModelBinaryCompiler();
  ~ModelBinaryCompiler();

  ModelBinaryCompiler(const ModelBinaryCompiler &) = delete;
  ModelBinaryCompiler &operator=(const ModelBinaryCompiler &) = delete;

  Status UpdateNodeInfoInOmSubGraph(ge::ComputeGraph &om_sub_graph, const KernelLookup &lookup) const;

 private:
  void UpdateVariableAttrFromOriNodeToCurNode(const ge::NodePtr &ori_node, const ge::NodePtr &cur_node) const;

  Status UpdateFuncSubGraphNetOutputInfo(const ge::NodePtr &ori_node, const ge::NodePtr &net_output) const;

  Status GetSubGraphsByCurNode(const ge::NodePtr &node, std::vector<ge::ComputeGraphPtr> &sub_graphs) const;

  Status UpdateConstValueByAttrBuffer(const ge::NodePtr &ori_node, const ge::NodePtr &node) const;

  Status UpdateSubGraphNodeInfoByOriginNode(const ge::NodePtr &ori_node,
      const KernelLookup &lookup, const ge::NodePtr &node) const;

  Status RecoverAttrsForSubGraphNode(const KernelLookup &lookup, const ge::NodePtr &node) const;

  Status RecoverAttrsWithKernelPrefix(const ge::OpDescPtr &op_desc, const std::string &kernel_prefix,
      const KernelLookup &lookup) const;

  Status UpdateTensorSliceInfo(const ge::NodePtr &ori_node, const ge::ComputeGraph &om_sub_graph) const;

  Status UpdateTensorSliceInfoToNode(bool is_input, const ffts::ThreadSliceMapPtr &slice_info_ptr,
    const std::unordered_map<ge::NodePtr, std::set<uint32_t>> &nodes_info) const;

  Status SetTensorSliceInfo(const ge::NodePtr &node, const std::set<uint32_t> &index,
      const bool &is_input, const ffts::ThreadSliceMapPtr &slice_info_ptr) const;

  template<typename T>
  Status UpdateSingleTensorSliceInfo(const std::set<uint32_t> &index, std::vector<std::vector<T>> &slice_infos,
      std::vector<std::vector<T>> &new_slice_infos) const;

  Status UpdateAxisAndTensorIndex(const std::set<uint32_t> &index,
      const std::vector<uint32_t> &tensor_indexes, const std::vector<uint32_t> &axis,
      std::vector<uint32_t> &new_tensor_indexes, std::vector<uint32_t> &new_axis) const;

  Status GetSubGraphTensorSliceNodesInfo(const ge::ComputeGraph &om_sub_graph,
      std::unordered_map<ge::NodePtr, std::set<uint32_t>> &first_nodes_info,
      std::unordered_map<ge::NodePtr, std::set<uint32_t>> &last_nodes_info) const;

  Status GetOutputTensorSliceNodesInfo(const ge::NodePtr &output_node,
      std::unordered_map<ge::NodePtr, std::set<uint32_t>> &last_nodes_info) const;

  bool NeedCopyTensorSliceInfo(const ge::NodePtr &ori_node, ffts::ThreadSliceMapPtr &slice_info_ptr) const;
};
} // namespace fe
#endif // COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_GRAPH_OPTIMIZER_DYNAMIC_SHAPE_OPTIMIZER_MODEL_BINARY_COMPILER_MODEL_BINARY_COMPILER_H_
