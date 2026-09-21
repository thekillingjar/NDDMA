/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_OP_COMPILER_OP_COMPILER_H_
#define FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_OP_COMPILER_OP_COMPILER_H_

#include <map>
#include <functional>
#include <memory>
#include <string>
#include "common/fe_inner_error_codes.h"
#include "common/fe_log.h"
#include "common/fe_utils.h"
#include "common/graph_comm.h"
#include "graph/compute_graph.h"
#include "graph_optimizer/json_parser/tbe_json_parse.h"
#include "graph_optimizer/lx_fusion_optimizer/lx_fusion_optimizer.h"
#include "common/graph_tuner_errorcode.h"
#include "adapter/common/op_store_adapter_manager.h"
#include "ops_kernel_store/fe_ops_kernel_info_store.h"

namespace fe {
using ScopeJsonMap = std::map<int64_t, std::string>;
using BufferFusionFunc = std::function<Status(ge::ComputeGraph&, LxFusionOptimizeResult&)>;

class OpCompiler;
using OpCompilerPtr = std::shared_ptr<OpCompiler>;

// used in pass
void ProcCompressOpMemType(const ge::NodePtr &node);

class OpCompiler {
 public:
  using ConstComputeGraph = const ge::ComputeGraph;
  template <class T>
  using Vistor = RangeVistor<T, std::shared_ptr<ConstComputeGraph>>;

  OpCompiler(const std::string& compiler_name, const std::string& engine_name,
             const LxFusionOptimizerPtr &lx_fusion_optimizer);
  OpCompiler(const std::string& compiler_name, const std::string& engine_name, const bool need_post_process,
             const LxFusionOptimizerPtr &lx_fusion_optimizer);
  virtual ~OpCompiler();

  /**
   * @ingroup fe
   * @brief prohibit copy and assign construct
   */
  OpCompiler(const OpCompiler&) = delete;
  OpCompiler& operator=(const OpCompiler&) = delete;

  /*
   *  @ingroup fe
   *  @brief   initialize op compiler
   *  @return  SUCCESS or FAILED
   */
  Status Initialize();

  /*
   *  @ingroup fe
   *  @brief   finalize op compiler
   *  @return  SUCCESS or FAILED
   */
  Status Finalize();

  /*
   *  @ingroup fe
   *  @brief   precompile tbe op
   *  @param   [in|out] graph  compute graph
   *  @return  SUCCESS or FAILED
   */
  Status PreCompileOp(ge::ComputeGraph& graph);

  /*
   *  @ingroup fe
   *  @brief   update compile params
   *  @param   [in|out] graph  compute graph
   *  @return  SUCCESS or FAILED
   */
  Status UpdateCompileParams(const ge::ComputeGraph &graph) const;

  /*
   *  @ingroup fe
   *  @brief   precompile tbe thread op
   *  @param   [in|out] graph  compute graph
   *  @return  SUCCESS or FAILED
   */
  Status PreCompileThreadOpHelper(const ge::NodePtr& node, const ge::OpDescPtr& op_desc_ptr,
                                  const ge::OpDescPtr &old_op_desc, const string& session_graph_id,
                                  const ge::ComputeGraph& graph);

  Status PreCompileThreadOp(ge::ComputeGraph& graph, bool& sgt_flag);

  /* For op-tune scenario, the first compiling we need to ignore the compile strategy.
   * Because we need the compile result and roll back those which is failed in compiling as
   * fusion nodes to single nodes. And the second compiling is standard without any special
   * operations.
   * So we add input parameter ignore_compile_strategy to differentiate those two kinds of op-tune compile. */
  Status CompileOpOnly(CompileInfoParam &compile_info) const;

  Status CompileOp(ge::ComputeGraph& graph);
  /*
   *  @ingroup fe
   *  @brief   compile tbe op
   *  @param   [in|out] graph  compute graph
   *  @return  SUCCESS or FAILED
   */
  Status CompileOp(ge::ComputeGraph& graph,
                   std::vector<ge::NodePtr>& buff_fus_compile_failed_nodes,
                   const std::vector<ge::NodePtr>& buff_fus_rollback_nodes,
                   const std::vector<ge::NodePtr>& buff_fus_to_del_nodes);

  virtual Status GetFusionScope(ge::ComputeGraph& graph, ScopeNodeIdMap &fusion_nodes_map,
                                std::vector<ge::NodePtr> &nodes_be_compiled,
                                std::vector<ge::NodePtr> &all_nodes_after_lx_fusion);

  Status ReCompileOpAfterLxFusion(ge::ComputeGraph& graph, CompileInfoParam &compile_info,
                                  const LxFusionOptimizeResult &opt_ret);

  void GetNodesNeedRePrcmpl(const Vistor<ge::NodePtr> &all_nodes,
                            std::unordered_set<int64_t> &need_re_compile_scope_id,
                            std::vector<ge::NodePtr> &nodes_be_compiled,
                            std::vector<ge::NodePtr>& all_nodes_after_lx_fusion);

  virtual Status RunCompileProcess(ge::ComputeGraph& graph) {
    (void)graph;
    return SUCCESS;
  };

  virtual Status GetFusionScope(const ge::ComputeGraph& graph, const std::vector<ge::NodePtr>& buff_fus_rollback_nodes,
                                ScopeNodeIdMap &fusion_nodes_map, std::vector<ge::NodePtr> &nodes_be_compiled);

  const string& GetCompilerName() const;
  bool IsNeedPostProcess() const;
 protected:
  /*
   *  @ingroup fe
   *  @brief   update compile params
   *  @param   [in|out] graph  compute graph
   *  @return  SUCCESS or FAILED
   */
  virtual Status UpdateNodeCompileParams(const ge::NodePtr &node, const OpKernelInfoPtr &op_kernel_info_ptr) const;

  Status PostCompileOp(ge::ComputeGraph& graph, std::vector<ge::NodePtr>& buff_fus_compile_failed_nodes);

  /*
   *  @ingroup fe
   *  @brief   add node to fusion_node_map according to scope id
   *  @param   [in]  node           node pointer
   *  @param   [in] scope_id        scope id
   *  @param   [out] fusion_node_map  scope id and node map
   *  @return  SUCCESS or FAILED
   */
  Status AddNodeToFusionMap(ge::Node& node, const int64_t scope_id, ScopeNodeIdMap& fusion_node_map);

  /*
   *  @ingroup fe
   *  @brief   get scope node map
   *  @param   [in]  graph        node pointer
   *  @param   [out] fusion_map    scope id and node map
   *  @return  SUCCESS or FAILED
   */
  Status GetScopeNodeMap(const ge::ComputeGraph& graph, ScopeNodeIdMap& fusion_node_map);

  /* We add the parameter nodemal_node_id is because, in the second round compilation of
   * lx-fusion, the scope id will start from the minimum negative one. */
  Status GetScopeNodeMap(const ge::ComputeGraph& graph, const std::vector<ge::NodePtr>& scope_nodes,
                         ScopeNodeIdMap& fusion_node_map);

  /*
   *  insert Compress op between conv and weight
   *  @ingroup fe
   *  @brief   insert Compress op between conv and weight
   *  @param   [in]  node        node pointer
   *  @return  SUCCESS or FAILED
   */
  Status InsertCompressOp(const ge::NodePtr &node) const;

  bool HasInsertCompressOp(const ge::NodePtr &conv_node) const;

  Status UpdateCompressIndexShape(const ge::OpDescPtr &conv_desc, const std::vector<int64_t> &compress_param_vec) const;

  ge::OpDescPtr CreateCompressOp(const ge::OpDescPtr &conv_desc, const std::vector<int64_t> &compress_param_vec) const;

  Status AddCompressNodeAndCopyWeight(const ge::NodePtr &conv_node, const ge::OpDescPtr &compress_opdesc) const;

  /**
   * Check whether do compressing weight on conv or FC
   * @param node node pointer
   * @return SUCCESS or FAILED
   */
  Status SetCompressWeightAttr(ge::NodePtr node) const;

  Status ParseJsonAndUpdateOp(const ge::NodePtr &node, const ge::OpDescPtr op_desc_ptr,
                              const std::vector<CompileResultInfo> &compile_results) const;
  /*
   *  @ingroup fe
   *  @brief   after compile tbe op, parse json file and update compress op
   *  @param   [in] graph                 compute graph
   *  @param   [in] scope_json_map          scope_id and json file
   *  @param   [in] buff_fus_rollback_nodes  nodes need to roll back
   *  @param   [in] node_size              node_size in graph
   *  @return  SUCCESS or FAILED
   */
  Status ParseJsonAndCompressOp(const ge::ComputeGraph& graph, const CompileResultMap& compile_ret_map,
                                const std::vector<ge::NodePtr>& nodes_be_compiled) const;

  Status VerifyScopeIdAttr(const int64_t &scope_id, const bool &is_l1_fusion,
                           std::map<int64_t, bool> &fusion_scope_type_map);

  Status AddNormalTbeNodeIntoMap(ge::NodePtr node, ge::OpDescPtr op_desc_ptr,
                                 ScopeNodeIdMap& fusion_node_map,
                                 std::map<int64_t, bool> &fusion_scope_type_map);

  bool CheckCompileCondition(const ge::NodePtr &node) const;

  bool IsNeedPreCompile(ge::NodePtr &node, ge::OpDescPtr &op_desc_ptr, bool need_precompile_graph) const;

  void GetRePreCompileSwitch(ge::ComputeGraph &graph, string &session_graph_id, bool &need_precompile_graph) const;

  Status SetPreCompParameter(const ge::NodePtr &node, const FEOpsStoreInfo &op_store_info,
                             const string &session_graph_id, OpImplType imply_type,
                             std::unordered_map<OpStoreAdapterPtr, vector<PreCompileNodePara>> &pre_comp_map);

  bool StopCompileOpInTuningAndAfterBuilderMode() const;

  Status SetMemoryTypeForOutput(const ge::NodePtr &node, const OpKernelInfoPtr &op_kernel_info_ptr) const;

  Status ParseTvmJsonToSetAttr(const ge::NodePtr& node, const ge::OpDescPtr op_desc_ptr,
                               const CompileResultInfo &compile_result) const;

  bool CheckCompiledFusionGraph(const ge::ComputeGraph& graph) const;

  Status GetMixComFusionScope(const ge::ComputeGraph& graph, const std::vector<ge::NodePtr>& buff_fus_rollback_nodes,
                              ScopeNodeIdMap &fusion_nodes_map, std::vector<ge::NodePtr> &nodes_be_compiled);

  Status ReCompileL1FusionFailedNodes(const ge::ComputeGraph &graph, CompileInfoParam &compile_info);

  bool UpdateOpAttrByCannKbResult(const std::string &kb_result_str, const ge::OpDescPtr &op_desc) const;

  void MarkDuplicatedNode(const ge::ComputeGraph &graph) const;

  Status GenerateFormatTuneResult(ge::ComputeGraph& graph, LxFusionOptimizeResult& buffer_ret,
                                  const bool &need_re_compile) const;

  bool init_flag_;
  std::string compiler_name_;
  std::string engine_name_;
  bool need_post_process_;
  LxFusionOptimizerPtr lx_fusion_optimizer_ptr_;
};
}  // namespace fe
#endif  // FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_OP_COMPILER_OP_COMPILER_H_
