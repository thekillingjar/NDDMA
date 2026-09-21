/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PREPROCESS_INSERT_OP_UTIL_INSERT_AIPP_OP_H_
#define GE_GRAPH_PREPROCESS_INSERT_OP_UTIL_INSERT_AIPP_OP_H_

#include <memory>
#include <string>
#include <vector>
#include "graph/compute_graph.h"
#include "graph/preprocess/insert_op/base_insert_op.h"
#include "proto/insert_op.pb.h"
#include "graph/manager/graph_manager_utils.h"

namespace ge {
enum AippType { OLD_TYPE, NEW_TYPE };

class InsertAippOpUtil {
 public:
  static InsertAippOpUtil &Instance() {
    thread_local InsertAippOpUtil instance;
    return instance;
  }

  Status Init();

  Status Parse(const GraphManagerOptions& options);

  Status InsertAippOps(ge::ComputeGraphPtr &graph, std::string &aippConfigPath);

  void ClearNewOps();

  Status UpdateDataNodeByAipp(const ComputeGraphPtr &graph) const;

  Status RecordAIPPInfoToData(const ComputeGraphPtr &graph) const;

 private:
  Status CheckPositionNotRepeat();

  Status GetAippParams(const std::unique_ptr<domi::AippOpParams> &aippParams, const ge::NodePtr &aipp_node) const;

  Status CheckInputNamePositionNotRepeat() const;

  Status CheckInputRankPositionNoRepeat() const;

  Status CheckGraph(const ge::ComputeGraphPtr &graph) const;

  Status CheckAndCopyAippOpParams(const GraphManagerOptions& options);

  InsertAippOpUtil() = default;

  ~InsertAippOpUtil() = default;

  std::vector<std::unique_ptr<BaseInsertOp>> insert_ops_;

  std::unique_ptr<domi::InsertNewOps> insert_op_conf_;

  void UpdateMultiBatchInputDims(const OpDescPtr &data_opdesc, Format &old_format) const;
  Status UpdatePrevNodeByAipp(const NodePtr &node) const;
  Status GetDataRelatedNode(const NodePtr &node, std::map<NodePtr, std::set<NodePtr>> &data_next_node_map) const;
  Status GetAllAipps(const NodePtr &data_node, const NodePtr &node, std::vector<NodePtr> &aipps) const;
  Status GetInputOutputInfo(NodePtr &data_node, NodePtr &aipp_node, std::string &input, std::string &output) const;
  Status SetModelInputDims(NodePtr &data_node, NodePtr &aipp_node) const;
  Status FindMaxSizeNode(const ComputeGraphPtr &graph, const NodePtr &case_node, std::map<uint32_t, int64_t> &max_sizes,
                         std::map<uint32_t, GeTensorDescPtr> &aipp_inputs) const;
  Status UpdateCaseNode(const ComputeGraphPtr &graph, const NodePtr &case_node) const;
};
}  // namespace ge

#endif  // GE_GRAPH_PREPROCESS_INSERT_OP_UTIL_INSERT_AIPP_OP_H_
