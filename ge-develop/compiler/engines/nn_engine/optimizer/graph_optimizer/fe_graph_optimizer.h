/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_FE_GRAPH_OPTIMIZER_H_
#define FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_FE_GRAPH_OPTIMIZER_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "adapter/adapter_itf/op_store_adapter.h"
#include "common/fe_inner_error_codes.h"
#include "common/fe_utils.h"
#include "common/unknown_shape_util.h"
#include "common/fusion_statistic/buffer_fusion_info_collecter.h"
#include "common/fusion_statistic/fusion_statistic_writer.h"
#include "common/graph/fe_graph_utils.h"
#include "common/optimizer/graph_optimizer.h"
#include "common/optimizer/graph_optimizer_types.h"
#include "common/optimizer/optimize_utility.h"
#include "format_selector/manager/format_dtype_setter.h"
#include "fusion_rule_manager/fusion_rule_manager.h"
#include "graph/compute_graph.h"
#include "graph/op_kernel_bin.h"
#include "graph_optimizer/dynamic_shape_optimizer/fuzzy_compiler/fuzzy_generalize.h"
#include "graph_optimizer/dynamic_shape_optimizer/model_binary_compiler/model_binary_compiler.h"
#include "graph_optimizer/fusion_common/fusion_pass_manager.h"
#include "graph_optimizer/graph_fusion/graph_fusion.h"
#include "graph_optimizer/heavy_format_propagation/heavy_format_propagation.h"
#include "graph_optimizer/node_optimizer/split_c_to_n_optimizer.h"
#include "graph_optimizer/node_optimizer/split_n_optimizer.h"
#include "graph_optimizer/node_optimizer/stridedwrite_optimizer.h"
#include "graph_optimizer/node_optimizer/stridedread_optimizer.h"
#include "graph_optimizer/op_axis_update/op_axis_update_desc.h"
#include "graph_optimizer/op_compiler/op_compiler.h"
#include "graph_optimizer/op_compiler/op_compiler_baseline.h"
#include "graph_optimizer/op_compiler/op_compiler_normal.h"
#include "graph_optimizer/op_compiler/op_compiler_optune.h"
#include "graph_optimizer/op_compiler/op_compiler_mstune_before_ub_match.h"
#include "graph_optimizer/op_judge/format_and_dtype/op_format_dtype_judge.h"
#include "graph_optimizer/op_judge/imply_type/op_impl_type_judge.h"
#include "graph_optimizer/op_setter/op_setter.h"
#include "graph_optimizer/shape_format_transfer/trans_node_manager/trans_node_manager.h"
#include "graph_optimizer/spacesize_calculator/spacesize_calculator.h"
#include "graph_optimizer/stream_graph_optimizer/l2_optimizer/l2_optimizer.h"
#include "graph_optimizer/ub_fusion/automatic_buffer_fusion.h"
#include "graph_optimizer/ub_fusion/buffer_fusion.h"
#include "graph_optimizer/lx_fusion_optimizer/lx_fusion_optimizer.h"
#include "graph_optimizer/weight_compress_flag/weight_compress_judge.h"
#include "ops_kernel_store/fe_ops_kernel_info_store.h"
#include "common/fe_error_code.h"
#include "base/err_msg.h"
#include "cmo/generate_cmo_type_manager.h"

namespace fe {
extern const string kStagePrepare;
extern const string kStageBeforeQuant;
extern const string kStageOrigin;
extern const string kStageAftFmtSlct;
extern const string kStageJudgeInsert;
extern const string kStageSetOpSlc;
extern const string kStagePreCompile;
extern const string kStageParseCompRst;
extern const string kStageLx;
extern const string kStageCompile;
extern const string kStageAfterMultiDims;

using OpImplyTypeJudgePtr = std::shared_ptr<OpImplTypeJudge>;
using OpSetterPtr = std::shared_ptr<OpSetter>;
using FormatDtypeSetterPtr = std::shared_ptr<FormatDtypeSetter>;
using OpFormatDtypeJudgePtr = std::shared_ptr<OpFormatDtypeJudge>;
using ReflectionBuilderPtr = std::shared_ptr<ge::RefRelations>;
using SpaceSizeCalculatorPtr = std::shared_ptr<SpaceSizeCalculator>;
using TransNodeManagerPtr = std::shared_ptr<TransNodeManager>;
using BufferFusionPtr = std::shared_ptr<BufferFusion>;
using AutomaticBufferFusionPtr = std::shared_ptr<AutomaticBufferFusion>;
using FusionPassManagerPtr = std::shared_ptr<FusionPassManager>;
using HeavyFormatPropagationPtr = std::shared_ptr<HeavyFormatPropagation>;
using OpAxisUpdateDescPtr = std::shared_ptr<OpAxisUpdateDesc>;
using SplitCToNOptimizerPtr = std::shared_ptr<SplitCToNOptimizer>;
using SplitNOptimizerPtr = std::shared_ptr<SplitNOptimizer>;
using BufferFusionInfoCollecterPtr = std::shared_ptr<BufferFusionInfoCollecter>;
using FuzzyGeneralizePtr = std::shared_ptr<fe::FuzzyGeneralize>;
using ModelBinaryCompilerPtr = std::shared_ptr<fe::ModelBinaryCompiler>;
using GenerateCMOTypeManagerPtr = std::shared_ptr<GenerateCMOTypeManager>;
using KernelLookup = std::function<ge::OpKernelBinPtr(const std::string &kernel_name)>;

enum class OpCompilerIndex{
  /* Base class. All common operations are in this class. */
  COMMON = 0,
  /* Standard l2buffer atc inference process without op-tune.
   * 1. Compile and do fusion recovery.
   * 2. Parse Json and set attributes.
   * 3. Do post process. */
  BASELINE = 1,

  /* Standard lx-fusion atc inference process without op-tune.
   * 1. Compile and do fusion recovery.
   * 2. Call LxFusion.
   * 3. Do pre-compilation and compilation for all
   * nodes with attributes NEED_RE_PRECOMPILE.
   * 4. Parse Json and set attributes.
   * 5. Do post process. */
  NORMAL = 2,

  /* Standard lx-fusion atc inference process with op-tune.
   * 1. Compile and do fusion recovery.
   * 2. Call LxFusion.
   * 3. Do pre-compilation for all nodes with attribute NEED_RE_PRECOMPILE and compilation for all nodes.
   * 4. Do not collect compilation result and return SUCCESS. */
  OPTUNE = 3,

  /* First step of Ms-tune process.
   * 1. Compile and do fusion recovery.
   * 2. return the subgraph to tuning tool. */
  MSTUNE_BEFORE_UB_MATCH = 4,
  OP_COMPILER_BOTTOM = 5,
};
class FEGraphOptimizer : public ge::GraphOptimizer {
 public:
  explicit FEGraphOptimizer(FEOpsKernelInfoStorePtr fe_ops_kernel_info_store_ptr,
                            std::string engine_name = fe::AI_CORE_NAME);
  ~FEGraphOptimizer() override;

  /**
   * @ingroup fe
   * @brief prohibit copy and assign construct
   */
  FEGraphOptimizer(const FEGraphOptimizer&) = delete;
  FEGraphOptimizer& operator=(const FEGraphOptimizer&) = delete;

  /*
   *  @ingroup fe
   *  @brief   initialize graph optimizer
   *  @param   [in] options
   *  @return  SUCCESS or FAILED
   */
  Status Initialize(const std::map<string, string>& options, ge::OptimizeUtility *const optimize_utility) override;

  /*
   *  @ingroup fe
   *  @brief   close graph optimizer
   *  @return  SUCCESS or FAILED
   */
  Status Finalize() override;

  Status FinalizeSessionInfo(ge::ComputeGraph& graph) override;

  /*
   *  @ingroup fe
   *  @brief   optimize original graph
   *  @param   [in|out] graph  compute graph
   *  @return  SUCCESS or FAILED
   */
  Status OptimizeOriginalGraph(ge::ComputeGraph& graph) override;

  /*
   *  @ingroup fe
   *  @brief   optimize fused graph
   *  @param   [in|out] graph   compute graph
   *  @return  SUCCESS or FAILED
   */
  Status OptimizeFusedGraph(ge::ComputeGraph& graph) override;

  /*
 *  @ingroup fe
 *  @brief   optimize fused graph for LXfusion
 *  @param   [in|out] graph   compute graph
 *  @return  SUCCESS or FAILED
 */
  Status OptimizeFusedGraphAfterGraphSlice(ge::ComputeGraph& graph) override;

  /*
   *  @ingroup fe
   *  @brief   optimize stream graph
   *  @param   [in|out] graph   compute graph
   *  @param   [in] context run_context
   *  @return  SUCCESS or FAILED
  */
  Status OptimizeStreamGraph(ge::ComputeGraph& stream_graph, const ge::RunContext& context) override;


  Status OptimizeStreamedWholeGraph(ge::ComputeGraph &graph) override;

  /*
   *  @ingroup fe
   *  @brief   get attribute of graph optimizer
   *  @param   [in|out] attrs
   *  @return  SUCCESS or FAILED
   */
  Status GetAttributes(ge::GraphOptimizerAttribute& attrs) const override;

  Status OptimizeWholeGraph(ge::ComputeGraph& graph) override;

  Status OptimizeGraphBeforeBuild(ge::ComputeGraph& graph) override;

  Status OptimizeGraphInit(ge::ComputeGraph& graph) override;

  Status OptimizeGraphPrepare(ge::ComputeGraph& graph) override;

  Status OptimizeOriginalGraphJudgeInsert(ge::ComputeGraph& graph) override;

  Status OptimizeOriginalGraphJudgeFormatInsert(ge::ComputeGraph& graph) override;

  Status OptimizeAfterGraphNormalization(const ge::ComputeGraphPtr &graph) override;

  Status GraphFusionAfterJudge(ge::ComputeGraph& graph) const;

  Status OptimizeAfterStage1(ge::ComputeGraph &graph) override;

  Status ShapeAndValueGeneralize(ge::ComputeGraph &graph) const;

  Status OptimizeSubgraphOfPrecompiledOp(ge::ComputeGraph &graph, const KernelLookup &lookup) override;

 private:
  void FeedStreamCtrlMap(ge::NodePtr &node, int64_t &event_id, StreamCtrlMap &stream_ctrls) const;

  void GenerateStreamCtrlMap(ge::ComputeGraph &graph, StreamCtrlMap &stream_ctrls) const;

  Status ConvertPartitionCalledOp(ge::ComputeGraph& graph);

  void StridedOptimize(ge::ComputeGraph& graph) const;

  /*
   *  link the
   *  @ingroup fe
   *  @brief   insert Compress op between conv and weight
   *  @param   [in]  graph        compute graph
   *  @return  SUCCESS or FAILED
   */
  Status HandleCompressOp(const ge::ComputeGraph &graph) const;

  Status HandleAclnnOp(ge::ComputeGraph &graph) const;

  Status OptimizeOriginalGraphOpJudgeAndFormatDtypeSetter(ge::ComputeGraph& graph) const;

  Status SplitOptimizer(ge::ComputeGraph& graph, const bool& need_set_virtual_op) const;

  Status UnfoldFuncOp(ge::ComputeGraph& graph) const;

  Status InsertClipByValue(ge::ComputeGraph& graph) const;

  Status CreateClipByValue(ge::ComputeGraph& graph, const ge::NodePtr& node, const bool& const_input) const;

  ge::NodePtr CreateSalcarConst(ge::ComputeGraph& graph, const std::string &clip_name,
                                const ge::GeTensorDesc &tensor_desc, const bool min_value) const;

  Status SubGraphCompile(ge::ComputeGraph& graph, const OpCompilerPtr &op_compiler,
                         const BufferFusionPtr &buffer_fusion_ptr);

  /* Do the following stuffs:
   * 1. Do statistics of all ub fusion pass.
   * 2. Merge all fused nodes.
   * 3. Calculate tensor size.
   * 4. Set atomic add info. */
  Status PostProcessAfterCompilingOp(ge::ComputeGraph& graph, const BufferFusionPtr& buffer_fusion_ptr);

  Status OptimizeFusedCompileOpAndCalcTensorSize(const OpCompilerPtr &op_compiler,
                                                 const BufferFusionPtr &buffer_fusion_ptr,
                                                 ge::ComputeGraph &graph);

  Status InsertTransNodesForAllGraph(ge::ComputeGraph& graph, TransNodeManagerPtr &trans_node_mgr_ptr) const;

  Status GraphFusionBeforeTransnodesInsertion(ge::ComputeGraph& graph) const;

  template <typename T>
  Status InitialzeOneCompiler(const string &compiler_name);

  template <typename T>
  Status InitialzeOpTuneCompiler(const string &compiler_name);

  Status InitializeAllOpCompiler();

  Status GetOpCompiler(OpCompilerPtr &op_compiler) const;

  Status BufferFusionMatch(ge::ComputeGraph& graph,
                           const std::shared_ptr<FusionCycleDetector> &cycle_detector,
                           const BufferFusionPtr &buffer_fusion_ptr) const;

  void ClearUnknowShapeAttr(const ge::ComputeGraph& graph) const;

  void AddAssignMemAttr(ge::ComputeGraph &graph) const;

  void ClearSameMemSet(ge::ComputeGraph &graph) const;

  void ConvertExtAttr2Json(const ge::ComputeGraph& graph, bool need_delete_ext_attr) const;

  void ConvertJson2ExtAttr(const ge::ComputeGraph& graph) const;

  void AllocMixResource(ge::ComputeGraph& graph) const;

  bool IsProcessBlockedByFusionSwitchCfg(const std::string &process_name);

  Status CompileMemSetOp(ge::ComputeGraph& graph) const;

  Status GetAndPreProcessForAtomicNodes(ge::ComputeGraph& graph,
      std::vector<ge::NodePtr> &atomic_node_vec, const std::string &memset_policy) const;

  Status CheckAndSetAtomicAttr(const ge::OpDescPtr &op_desc, bool &atomic_node_flag) const;

  Status PrecompileAndUbMatch(ge::ComputeGraph& graph, GraphCommPtr &graph_comm_ptr,
                              OpCompilerPtr &op_compiler, BufferFusionPtr &buffer_fusion_ptr) const;

  bool CheckNeedSetSliceInfo(ge::ComputeGraph& graph) const;

  Status OptimizeGraphForTiling(ge::ComputeGraph& graph) const;

  void MatchSuperkernelPlusNodes(const ge::ComputeGraph& graph) const;

  static void SetSkpScopeAttr(const std::vector<ge::NodePtr> &match_nodes);

  Status HeavyFormatPropagate(ge::ComputeGraph &graph, HeavyFormatPropagationPtr heavy_format_prop_ptr);

  bool CheckExportFusionResCond(ge::ComputeGraph &graph) const;

 private:
  // op info mgr
  FEOpsKernelInfoStorePtr ops_kernel_info_store_ptr_;

  // op compiler shared pointer
  std::vector<OpCompilerPtr> op_compiler_ptr_;

  OpSetterPtr op_setter_ptr_;

  // op judge pointer
  OpImplyTypeJudgePtr op_impl_type_judge_ptr_;

  // format_dtype_setter
  FormatDtypeSetterPtr format_dtype_setter_ptr_;

  // Space Size calculator pointer
  SpaceSizeCalculatorPtr space_size_calculator_ptr_;

  // l2 buffer optimizer pointer
  L2OptimizerPtr l2_optimize_ptr_;

  // lxfusion optimizer point
  LxFusionOptimizerPtr lx_fusion_optimizer_ptr_;

  // graph fusion ptr
  GraphFusionPtr graph_fusion_ptr_;

  // passes mngr
  FusionPassManagerPtr fusion_pass_mgr_ptr_;

  // rules mngr
  FusionRuleManagerPtr fusion_rule_mgr_ptr_;

  // priority mngr
  FusionPriorityMgrPtr fusion_priority_mgr_ptr_;

  // attribute of graph optimizer
  ge::GraphOptimizerAttribute graph_optimizer_attr_;

  // init flag
  bool init_flag_;
  std::mutex sort_lock_;
  std::mutex buffer_fusion_info_collecter_lock_;
  std::shared_ptr<std::mutex> fe_lock_ptr_;

  // update op axis.
  OpAxisUpdateDescPtr op_axis_update_desc_ptr_;

  SplitCToNOptimizerPtr split_c_to_n_optimizer_ptr_;

  SplitNOptimizerPtr split_n_optimizer_ptr_;

  BufferFusionInfoCollecterPtr buffer_fusion_info_collecter_ptr_;

  GenerateCMOTypeManagerPtr generate_cmo_type_manager_ptr_;

  ge::OptimizeUtility *optimize_utility_;
};
}  // namespace fe
#endif  // FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_FE_GRAPH_OPTIMIZER_H_
