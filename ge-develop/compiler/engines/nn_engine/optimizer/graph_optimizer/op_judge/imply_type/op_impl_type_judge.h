/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_OP_JUDGE_IMPLY_TYPE_OP_IMPL_TYPE_JUDGE_H_
#define FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_OP_JUDGE_IMPLY_TYPE_OP_IMPL_TYPE_JUDGE_H_

#include "graph_optimizer/op_judge/op_judge_base.h"
#include "graph/ge_local_context.h"
#include "ops_kernel_store/fe_ops_kernel_info_store.h"

namespace fe {
extern const std::string LX_CORE_TYPE_WITH_DYNAMIC_UPPER;
extern const std::string LX_CORE_TYPE_WITH_DYNAMIC_LOWER;
class OpImplTypeJudge : public OpJudgeBase {
 public:
  OpImplTypeJudge(const std::string& engine_name, FEOpsKernelInfoStorePtr fe_ops_kernel_info_store_ptr);
  ~OpImplTypeJudge() override;

  Status GetLXCoreType(ge::NodePtr &node_ptr);

  Status Judge(ge::ComputeGraph& graph) override;
  Status SetFormat(ge::ComputeGraph &graph) override;
  /**
   * judge imply type for the node
   * @param node_ptr current node
   * @return SUCCESS or FAIL
   */
  Status JudgeByNode(ge::NodePtr node_ptr);

  /**
   * judge the imply type for the node in sub graph
   * delete it after ischecksupported ok
   * supported ok
   * @param graph sub graph
   * @return SUCCESS or FAIL
   */
  Status JudgeInSubGraph(ge::ComputeGraph& graph);

  /**
   * judge the imply type for the node in sub graph, delete it after check
   * supported ok
   * @param node_ptr current node
   * @return SUCCESS or FAIL
   */
  Status JudgeInSubGraphByNode(ge::NodePtr node_ptr);

  static Status MultiThreadJudgeByNode(ge::NodePtr node_ptr, OpImplTypeJudge *op_format_dtype_judge_ptr,
                                       const ge::GEThreadLocalContext &ge_context);

  Status MultiThreadJudge(ge::ComputeGraph &graph);
 private:
  FEOpsKernelInfoStorePtr ops_kernel_info_store_ptr_;
  /**
   * set the op imply type
   * @param op_desc_ptr current op desc
   * @param impl_type the imply type of the op
   * @return SUCCESS or FAIL
   */
  Status SetOpImplType(const ge::NodePtr &node, OpImplType& imply_type);

  /**
   * set the op core type
   * @param op_desc_ptr current op desc
   * @return SUCCESS or FAIL
   */
  Status SetEngineType(ge::OpDescPtr op_desc_ptr);
};
}  // namespace fe

#endif  // FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_OP_JUDGE_IMPLY_TYPE_OP_IMPL_TYPE_JUDGE_H_
