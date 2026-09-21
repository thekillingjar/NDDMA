/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_OPTIMIZER_ADAPTER_TBE_ADAPTER_TBE_OP_STORE_ADAPTER_H_
#define FUSION_ENGINE_OPTIMIZER_ADAPTER_TBE_ADAPTER_TBE_OP_STORE_ADAPTER_H_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <nlohmann/json.hpp>
#include "adapter/adapter_itf/op_store_adapter.h"
#include "adapter/tbe_adapter/tbe_info/tbe_info_assembler.h"
#include "adapter/tbe_adapter/tbe_info/tbe_single_op_info_assembler.h"
#include "common/plugin_manager.h"
#include "graph_optimizer/graph_optimize_register_error_codes.h"
#include "tensor_engine/fusion_api.h"
#include "ops_kernel_store/fe_ops_kernel_info_store.h"

namespace fe {
using TbeInfoAssemblerPtr = std::shared_ptr<fe::TbeInfoAssembler>;
using TbeSingleOpInfoAssemblerPtr = std::shared_ptr<fe::TbeSingleOpInfoAssembler>;

class TbeOpStoreAdapter : public OpStoreAdapter {
 public:
  explicit TbeOpStoreAdapter(const std::string &engine_name);
  /*
   *  @ingroup fe
   *  @brief   initial resources needed by TbeCompilerAdapter, such as dlopen so
   * files
   *           and load function symbols etc.
   *  @return  SUCCESS or FAILED
   */
  Status Initialize(const std::map<std::string, std::string> &options) override;

  using OpStoreAdapter::CompileOp;
  /*
   *  @ingroup fe
   *  @brief   compile fused op and single op, and generate .o and json files
   *  @param   [in]  fusion_nodes_map  op id and fused sub-graph
   *  @param   [out] json_file_map_    keep path of .o and json of each op
   *  @return  SUCCESS or FAILED
   */
  Status CompileOp(CompileInfoParam &compile_info) override;

  /*
   *  @ingroup fe
   *  @brief   pre-compile and return pattern of op
   *  @return  SUCCESS or FAILED
   */
  Status PreCompileOp(vector<PreCompileNodePara> &compile_para_vec) override;

  Status TaskFusion(const ScopeNodeIdMap &fusion_nodes_map, CompileResultMap &compile_ret_map) override;

  Status InitializeInner(const std::map<std::string, std::string> &options);

  Status InitializeTeFusion(const std::map<std::string, std::string> &options);

  Status InitializeInnerHelp();

  Status SuperKernelCompile(const ScopeNodeIdMap &fusion_nodes_map, CompileResultMap &compile_ret_map) override;

  /*
   *  @ingroup fe
   *  @brief   finalize resources initialized in Initialize function,
   *           such as dclose so files etc.
   *  @return  SUCCESS or FAILED
   */
  Status Finalize() override;

  Status FinalizeSessionInfo(const std::string& session_graph_id) override;

  bool CheckSupport(const ge::NodePtr &node, CheckSupportParam &check_param,
                    const bool &is_dynamic_impl, std::string &reason) override;

  Status GetLXOpCoreType(const ge::NodePtr &node, const OpKernelInfoPtr &op_kernel_info_ptr,
                         const bool &is_dynamic_impl, std::string &lx_op_core_type_str) override;

  Status SelectOpFormat(const ge::NodePtr &node, const OpKernelInfoPtr &op_kernel_info_ptr,
                        const bool &is_dynamic_impl, const HeavyFormatInfo &heavy_format_info,
                        std::string &op_format_dtype_str) override;

  Status SetTbeOpSliceInfo(const ge::NodePtr &node_ptr, OpKernelInfoPtr &op_kernel_info_ptr) override;

  Status GeneralizeNode(const ge::NodePtr &node, const te::TbeOpInfo &op_info,
    te::TE_GENERALIZE_TYPE generalize_type) override;

  bool GetSpecificInfo(const te::TbeOpInfo &tbe_op_info, std::string &opSpecificInfo);

  Status GetRangeLimitType(const ge::NodePtr &node_ptr,
                           const te::TbeOpInfo &tbe_op_info, bool &is_limited) const override;

  Status LimitedNodesCheck(bool &is_support, const te::TbeOpInfo &tbe_op_info,
      std::vector<size_t> &upper_limited_input_indexs, std::vector<size_t> &lower_limited_input_indexs) override;

  Status IsGeneralizeFuncRegistered(bool &is_registered, const te::TbeOpInfo &op_info) override;

  Status GetOpUniqueKeys(const ge::NodePtr &node, const OpKernelInfoPtr &op_kernel_info_ptr,
                         std::vector<std::string> &op_unique_keys) override;

  Status FeedNodeGeneralInfo(const ge::NodePtr &node_ptr, NodeGeneralInfoPtr &node_info_ptr) const override;

  Status FeedNodeGeneralInfoFromOpStore(const ge::NodePtr &node_ptr,
                                        NodeGeneralInfoPtr &node_info_ptr) const override;

  Status GetRangeLimit(const NodeGeneralInfoPtr &node_info_ptr,
                       const ge::NodePtr &node_ptr) const override;

  Status GetDynamicPromoteType(const ge::NodePtr &node, const OpKernelInfoPtr &op_kernel_info_ptr,
                               std::string &promote_str) const override;

  Status UpdatePrebuildPattern();

  void GetAllCompileStatisticsMsg(std::vector<std::string> &statistics_msg) const override;

  bool JudgeBuiltInOppKernelInstalled() const override;

  std::string GetCurKernelMetaDir() const override;

  bool IsNeedSkipOpJudge(const ge::NodePtr &node, const OpKernelInfoPtr &op_kernel_info_ptr) const override;

  void SetOpsKernelInfoStore(const std::shared_ptr<ge::OpsKernelInfoStore> ops_kernel_info_store_ptr) override;

 private:
  std::string engine_name_;
  bool init_flag{false};
  bool support_parallel_compile{false};
  PluginManagerPtr plugin_manager_ptr{nullptr};
  OpsKernelInfoStorePtr fe_ops_kernel_info_store_ptr_;

  // function wrt TBE API
  function<bool(const te::TbeOpInfo &, std::string &)> SelectTbeOpFormat{nullptr};
  function<bool(te::TbeOpInfo &, te::CheckSupportedInfo &)> CheckTbeSupported{nullptr};
  function<bool(te::TbeOpInfo &, uint64_t, uint64_t)> PreBuildTbeOp{nullptr};
  function<te::OpBuildResCode(std::vector<ge::Node *>, ge::OpDescPtr, const std::vector<ge::NodePtr> &, uint64_t,
                              uint64_t, const std::string &)> TeFusion{nullptr};
  function<te::OpBuildResCode(std::vector<ge::Node *>, ge::OpDescPtr, const std::vector<ge::NodePtr> &, uint64_t,
                              uint64_t, uint64_t, const std::string &)> TeFusionV{nullptr};
  function<te::OpBuildResCode(uint64_t, uint64_t, ge::Node &)> FuzzBuildTbeOp{nullptr};
  function<te::OpBuildResCode(const std::vector<ge::Node *> &, ge::OpDescPtr,
                              uint64_t, uint64_t)> TaskFusionFunc{nullptr};
  function<te::OpBuildResCode(const std::vector<ge::Node *> &, ge::OpDescPtr,
                              uint64_t, uint64_t)> BuildSuperKernel{nullptr};
  function<te::LX_QUERY_STATUS(const te::TbeOpInfo &, std::string &)> GetOpInfo{nullptr};
  function<bool(const std::map<std::string, std::string> &, bool *)> TbeInitialize{nullptr};
  function<bool()> TbeFinalize{nullptr};
  function<bool(const std::string &)> TbeFinalizeSessionInfo{nullptr};
  function<bool(uint64_t, vector<te::FinComTask> &)> WaitAllFinished{nullptr};
  function<bool(const te::TbeOpInfo &, bool &)> CheckIsTbeGeneralizeFuncRegistered{nullptr};
  function<bool(const te::TbeOpInfo &, const te::TE_GENERALIZE_TYPE &, const ge::NodePtr &)> TeGeneralize{nullptr};
  function<bool(const te::TbeOpInfo &, std::string &)> GetOpSpecificInfo{nullptr};
  function<bool(const te::TbeOpInfo &, std::vector<std::string> &)> GetOpUniqueKeyFunc{nullptr};
  function<bool(const te::TbeOpInfo &, bool &, std::vector<size_t> &, std::vector<size_t> &)>
    DynamicShapeRangeCheck{nullptr};
  function<bool(std::vector<std::pair<std::string, std::string>> &)> QueryOpPattern{nullptr};
  function<void(std::vector<std::string> &)> GetAllCompileStatistics{nullptr};
  function<bool(bool, int64_t)> IsOppKernelInstalled{nullptr};
  function<std::string()> GetKernelMetaDir{nullptr};

  struct CompileTaskPara {
    uint64_t task_num;
    bool is_fusion_check;
    CompileResultMap *compile_ret_map;
    ScopeNodeIdMap *fusion_nodes_map;
    std::map<uint64_t, int64_t> task_scope_id_map;
    std::unordered_map<int64_t, vector<uint64_t>> scope_task_ids_map;
    std::unordered_map<uint64_t, te::FinComTask> failed_tasks;
    std::map<uint64_t, te::FinComTask> succ_tasks;

    std::unordered_map<uint64_t, ge::Node *> task_node_map;
    std::unordered_map<uint64_t, TbeOpInfoPtr> task_tbe_info_map;
    std::unordered_map<int64_t, bool> failed_task_able_to_delete;
    std::unordered_map<uint64_t, OpKernelInfoPtr> task_kernel_info_map;

    CompileTaskPara() : task_num(0), is_fusion_check(false), compile_ret_map(nullptr), fusion_nodes_map(nullptr) {}
    explicit CompileTaskPara(uint64_t num)
        : task_num(num), is_fusion_check(false), compile_ret_map(nullptr), fusion_nodes_map(nullptr) {}
    CompileTaskPara(uint64_t num, bool fusion_check, CompileResultMap *ret_map, ScopeNodeIdMap *nodes_map)
        : task_num(num), is_fusion_check(fusion_check), compile_ret_map(ret_map), fusion_nodes_map(nodes_map) {}
  };

  bool ConvertCheckSupportResult(const ge::NodePtr &node, const std::string &reason,
                                 te::CheckSupportedResult &is_supported) const;
  bool CheckUnsupportReason(const ge::NodePtr &node, const std::string &reason,
                            te::CheckSupportedResult &is_supported) const;

  Status SetOpCompileResult(const int64_t scope_id, const ge::OpDescPtr &compile_op_desc,
                            const bool is_replace, CompileResultMap &compile_ret_map) const;

  Status ParallelCompileOp(CompileInfoParam &compile_info);

  static Status DelScopeIdOfFailedNodes(CompileTaskPara &task_para);

  Status WaitTaskFinish(CompileTaskPara &task_para) const;

  Status ProcessSuccCompileTask(CompileTaskPara &task_para) const;

  Status ProcessFailCompileTask(CompileTaskPara &task_para, const CompileStrategy &compile_strategy);

  Status ProcessAllFailedCompileTasks(CompileTaskPara &task_para,
                                      std::vector<ge::NodePtr> &buff_fus_compile_failed_nodes,
                                      std::vector<ge::NodePtr> &l1_fusion_failed_nodes,
                                      const CompileStrategy &compile_strategy);

  Status SetFailedOpCompileTask(ge::Node* node, CompileTaskPara &task_para, const CompileStrategy &compile_strategy);

  Status RetryCompileFailOp(CompileTaskPara &task_para);

  Status ProcessSuccPreCompTask(CompileTaskPara &task_para) const;

  Status ProcessFailPreCompTask(CompileTaskPara &task_para) const;

  void SetOpDescCustomOp(ge::OpDescPtr op_desc) const;

  Status DoFuzzBuildTbeOp(std::vector<ge::Node *> &node_vec, uint64_t taskId, uint64_t thread_id);

  Status SetTeTask(vector<ge::Node *> &node_vec, uint64_t taskId,
                   const std::vector<ge::NodePtr> &buff_fus_to_del_nodes, const CompileStrategy &compile_strategy,
                   const bool &is_fusion_check);

  Status SgtSetTeTask(vector<ge::Node *> &node_vec, uint64_t taskId,
                      const std::vector<ge::NodePtr> &buff_fus_to_del_nodes,
                      const CompileStrategy &compile_strategy, uint64_t slice_shape_index = 0xFFFFFFFF);

  Status SetTaskFusionTask(const uint64_t thread_id, const int64_t scope_id, const std::vector<ge::Node *> &nodes,
                           CompileTaskPara &task_para) const;

  Status SetSuperKernelTask(const uint64_t thread_id, const int64_t scope_id,
                            const std::vector<ge::Node *> &nodes, CompileTaskPara &task_para) const;
  void SgtGetCompileStrategy(std::vector<ge::Node *> &node_vec, std::string &op_compile_strategy,
                             const CompileStrategy &compile_strategy) const;

  bool GetOpDslFilePath(const ge::OpDescPtr &op_desc, const OpKernelInfoPtr &op_kernel_info_ptr,
                        std::string &op_dsl_file_path) const;

  void GetAndSetOpsPathNamePrefix(const OpKernelInfoPtr &op_kernel_info_ptr, te::TbeOpInfo &tbe_op_info) const;

  TbeInfoAssemblerPtr tbe_info_assembler_ptr_;

  TbeSingleOpInfoAssemblerPtr tbe_single_op_info_assembler_ptr_;

  Status SetPreCompilePattern(ge::OpDescPtr op_desc, te::TbeOpInfo &op_info,
                              const OpKernelInfoPtr &op_kernel_info_ptr) const;

  TbeOpInfoPtr PreCompSetTbeOpInfo(PreCompileNodePara &comp_para);

  Status SetTensorDescShape(const ge::OpDescPtr &op_desc, const ge::GeTensorDescPtr &cur_tensor_desc) const;
  Status SetTensorDescRange(const ge::OpDescPtr &op_desc, const ge::GeTensorDescPtr &cur_tensor_desc) const;

  Status GenerateSingleOpRange(const PreCompileNodePara &comp_para) const;

  Status ParallelPreCompileOp(vector<PreCompileNodePara> &compile_para_vec);

  Status SerialPreCompileOp(vector<PreCompileNodePara> &compile_para_vec);

  static void ChangeBufferOptimize(const std::map<std::string, std::string> &options,
                                   std::map<std::string, std::string> &new_options);

  static void SetOpCompileInfo(const ge::OpDescPtr &op_desc_ptr, const std::vector<ge::Node *> &nodes);

  void SetOpTilingKey(std::vector<ge::Node *> &nodes, const ge::OpDescPtr &op_desc_ptr) const;

  void SetSPKAttr(std::vector<ge::Node *> &nodes, const ge::OpDescPtr &op_desc_ptr) const;

  bool StopCompileOpInTuningAndAfterUBMatchMode() const;

  bool StopWaitTaskFinishInTuningAndAfterBuilderMode(const bool is_fusion_check,
                                                     const CompileStrategy &compile_strategy) const;

  void SetFusionFailedId(const vector<ge::Node *> &fusion_nodes, const int64_t &fusion_failed_id) const;

  void SetCustomFlag(const ScopeNodeIdMap &fusion_nodes_map) const;

  bool CheckOpsPathPrefix(const ScopeNodeIdMap &fusion_nodes_map) const;

  void GetAutoMode(const ScopeNodeIdMap &fusion_nodes_map, bool &auto_mode) const;

  // initialize required tbe api for tbe adapter
  Status InitTbeFunctions(const PluginManagerPtr &plugin_manager_ptr_param);

  void RollBackAttributes(std::vector<ge::Node *> &failed_nodes) const;
  Status SetTaskToTeFusion(CompileTaskPara &task_para, const std::vector<ge::NodePtr> &buff_fus_to_del_nodes,
                           const CompileStrategy &compile_strategy, const bool &is_fusion_check);

  Status CompileMultiKernelSliceOp(ScopeNodeIdMap &fusion_nodes_map, CompileResultMap &compile_ret_map,
                                   std::vector<ge::NodePtr> &compile_failed_nodes,
                                   const std::vector<ge::NodePtr> &to_del_nodes);

  Status ProcessLxFusionFailCompileTasks(CompileTaskPara &task_para,
                                         std::vector<ge::NodePtr> &l1_fusion_failed_nodes,
                                         std::vector<ge::NodePtr> &buff_fus_failed_nodes) const;

  bool IsBuffFusOptimizedNodes(const std::vector<ge::Node *> &scope_op) const;

  bool IsL1FusionOptimizedNodes(const std::vector<ge::Node *> &nodes) const;

  void SaveMsTuneErrorMsg(CompileTaskPara &task_para) const;
  Status GetSgtSliceTaskRollbackNode(CompileTaskPara &task_para,
                                     std::vector<ge::NodePtr> &need_rollback_nodes) const;

  Status SetSgtTensorSliceInfoToNodes(std::vector<ge::Node*> &compile_nodes, int32_t thread_idx) const;

  Status SetTaskForOneScope(std::vector<ge::Node *> &nodes,
                            const int64_t scope_id,
                            const std::vector<ge::NodePtr> &to_del_nodes,
                            CompileTaskPara &task_para,
                            const CompileStrategy &compile_strategy);
  Status SetSgtSliceTaskToTeFusion(CompileTaskPara &task_para,
                                   const std::vector<ge::NodePtr> &to_del_nodes);
  Status ProcessSuccSgtSliceTask(CompileTaskPara &task_para) const;

  void ClearTaskPara(CompileTaskPara &task_para) const;

  Status UpdateTensorByMixPrecisionMode(const ge::NodePtr &node, const OpKernelInfoPtr &op_kernel_info_ptr,
      std::pair<std::vector<size_t>, std::vector<size_t>> &in_out_changed_idx_vec) const;

  void UpdateDtypeByAllowFp32ToFp16(const ge::OpDescPtr &op_desc, size_t input_or_output_index,
                                    std::pair<std::vector<size_t>, std::vector<size_t>> &in_out_changed_idx_vec,
                                    const bool &isinput) const;

  bool UpdateInputOrOutputDtype(const ge::OpDescPtr &op_desc, const ge::GeTensorDescPtr &tensor_desc,
                                const size_t input_or_output_index) const;
  bool AssembleTbeByMixPrecisionMode(const ge::NodePtr &node, const OpKernelInfoPtr &op_kernel_info_ptr,
                                     const bool &is_dynamic_impl, te::TbeOpInfo &op_info) const;

  inline bool IsFuzzCompileStrategy(const CompileStrategy &compile_strategy) const;

  static void ParseHardwareInfos(std::map<std::string, std::string> &new_options);

  static void DealOpDebugDir(std::map<std::string, std::string> &new_options);

  template <typename T>
  Status ParseJsonByKey(const std::string &json_str, const std::string &key, T &value) const;

  template <typename T>
  Status GetOpSpecificInfoByKey(const te::TbeOpInfo &op_info, const std::string &key, T &value) const;

  bool GetSelectOpFormat(const ge::NodePtr &node, std::string &op_select_format_str) const;

  void SetPrebuildPattern(const OpKernelInfoPtr &op_kernel_info_ptr, te::TbeOpInfo &op_info) const;

  void SetOpSpecificInfoToTbeOpInfo(const OpKernelInfoPtr &op_kernel_info_ptr, te::TbeOpInfo &op_info) const;
  void SubmitCompileProcessTrace(const uint64_t total_task_num, const uint64_t wait_task_num,
                                 const uint64_t time_interval_threshold,
                                 const std::chrono::time_point<std::chrono::high_resolution_clock> &now_time,
                                 std::chrono::time_point<std::chrono::high_resolution_clock> &last_trace_time) const;
  void SubmitLongTimeConstTrace(const vector<te::FinComTask> &fin_com_task, const CompileTaskPara &task_para,
                                const uint64_t time_interval_threshold,
                                const std::chrono::time_point<std::chrono::high_resolution_clock> &now_time,
                                std::chrono::time_point<std::chrono::high_resolution_clock> &last_trace_time) const;
  void SubmitCompileFinishTrace(const uint64_t total_task_num) const;
};
using TbeOpStoreAdapterPtr = std::shared_ptr<TbeOpStoreAdapter>;
}  // namespace fe
#endif  // FUSION_ENGINE_OPTIMIZER_ADAPTER_TBE_ADAPTER_TBE_OP_STORE_ADAPTER_H_
