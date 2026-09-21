/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_OPTIMIZER_OPS_KERNEL_STORE_FE_OPS_KERNEL_INFO_STORE_H_
#define FUSION_ENGINE_OPTIMIZER_OPS_KERNEL_STORE_FE_OPS_KERNEL_INFO_STORE_H_
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <vector>
#include "adapter/tbe_adapter/tbe_info/tbe_info_assembler.h"
#include "common/math_util.h"
#include "common/fe_op_info_common.h"
#include "common/opskernel/ops_kernel_info_store.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "common/util/json_util.h"
#include "common/util/op_info_util.h"
#include "common/optimizer/optimize_utility.h"
#include "graph/node.h"
#include "graph/op_desc.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/node_utils.h"
#include "ops_store/op_kernel_info.h"
#include "ops_kernel_store/sub_ops_store.h"
#include "ops_store/sub_op_info_store.h"

namespace fe {
class FusionPriorityManager;
using FusionPriorityMgrPtr = std::shared_ptr<FusionPriorityManager>;
using SubOpsStorePtr = std::shared_ptr<SubOpsStore>;
using OpsKernelInfoStorePtr = std::shared_ptr<ge::OpsKernelInfoStore>;
using TbeInfoAssemblerPtr = std::shared_ptr<fe::TbeInfoAssembler>;
struct CheckSuportResult {
  bool result;
  int64_t fe_impl_type;
  int64_t ge_impl_type;
};

struct MemsetWorkspaceInfo {
  std::vector<int64_t> work_space;
  std::vector<int64_t> work_space_bytes;
  std::vector<int32_t> dtype_list;
  std::vector<int64_t> init_value_int64_list;
  std::vector<float> init_value_float_list;
};

using FusionParenNodeAndIndex = struct {
  ge::NodePtr parent_node_ptr;
  std::unordered_map<ge::NodePtr, int64_t> data_attr_parent_index_map;
};

class FEOpsKernelInfoStore : public ge::OpsKernelInfoStore, public std::enable_shared_from_this<FEOpsKernelInfoStore> {
 public:
  explicit FEOpsKernelInfoStore(std::string engine_name = fe::AI_CORE_NAME);

  ~FEOpsKernelInfoStore() override;

  FEOpsKernelInfoStore(const FEOpsKernelInfoStore &) = delete;

  FEOpsKernelInfoStore &operator=(const FEOpsKernelInfoStore &) = delete;

  /*
   * @ingroup fe
   * @brief : Initialize the FEOpsKernelInfoStore, load the op info from the
   * specified .json file
   * @param[in] options: the reserved param
   * @return : SUCCESS/FAILED
   */
  Status Initialize(const map<string, string> &options) override;

  /*
   * @ingroup fe
   * @brief : finalize the FEOpsKernelInfoStore, clear the op info;
   * @param[in] None
   * @return : SUCCESS/FAILED
   */
  Status Finalize() override;

  Status CreateSession(const std::map<std::string, std::string> &session_options) override;

  Status DestroySession(const std::map<std::string, std::string> &session_options) override;

  /*
   * @ingroup fe
   * @brief : get the specified op info from the FEOpsKernelInfoStore;
   * @param[in] infos ： a struct containing the wanted fields of op info;
   * @return: SUCCESS/FAILED
   */
  void GetAllOpsKernelInfo(std::map<std::string, ge::OpInfo> &infos) const override;

  /*
   * @brief: Query all the FEOpsKernelInfoStores, get the highest priority
   * implement type of the Op;
   * @param[in] OpDesc: op
   * @param[in|out] OpImplType: the implement type of this Op;
   * @return: SUCCESS/FAILED
   */
  Status QueryHighPrioOpImplType(const ge::NodePtr &node, OpImplType &impl_type,
                                 OpKernelInfoPtr &op_kernel_ptr) const;

  Status GetAllSubOpsStore(std::map<std::string, SubOpsStorePtr> &all_sub_store_ptr) const;

  bool CheckSupported(const ge::OpDescPtr &op_desc_ptr, std::string &un_supported_reason) const override;

  bool CheckAccuracySupported(const ge::OpDescPtr &op_desc_ptr, std::string &un_supported_reason,
                              bool real_query = false) const override;

  bool CheckSupported(const ge::NodePtr &node, std::string &un_supported_reason) const override;

  bool CheckSupported(const ge::NodePtr &node, std::string &un_supported_reason,
                      ge::CheckSupportFlag &flag) const override;

  bool CheckAccuracySupported(const ge::NodePtr &node, std::string &un_supported_reason,
                              bool real_query = false) const override;

  bool CheckSupportedBase(const ge::NodePtr &node, std::string &un_supported_reason, CheckSupportMode mode,
                          bool real_query = false) const;

  const std::string& GetFEOpsKernelInfoStoreName() const;

  Status MultipleOpMergeFusionGraph(vector<ge::NodePtr> &node_vec, vector<ge::NodePtr> &atomic_node_vec,
                                    const bool is_fuzz_build);

  Status CompileOp(vector<ge::NodePtr> &node_vec) override;

  Status CompileOpRun(vector<ge::NodePtr> &node_vec) override;

  Status FuzzCompileOp(vector<ge::NodePtr> &node_vec) override;

  bool GetNodeSupportInfo(const ge::OperatorPtr &op, std::string &support_info) override;

  bool CheckSupportedByOpsStore(const ge::NodePtr &node, const CheckSupportMode &check_mode,
                                CheckSupportParam &check_param, OpImplType &impl_type) const;

  Status GetNotSupportedReasonByAttr(const uint64_t &reason, std::ostringstream &reason_oss) const;

  Status SetDynamicCustomOpStoreInfo(ge::ComputeGraph &graph);

  /* check support for trans-nodes by caching result. */
  bool CheckAccuracySupportByCache(const ge::OpDescPtr &op_desc_ptr);

  /* Store check support result for trans-nodes only. */
  Status StoreCheckSuportResultForTransNodes(const ge::OpDescPtr &op_desc_ptr, bool result);

  void SetCheckSupportedStaticFlag(bool stat_flag);

  Status PrePareCompileParameter(const ge::NodePtr &node, const string &op_type, OpImplType &impl_type,
                                 std::unordered_map<OpStoreAdapterPtr, vector<PreCompileNodePara>> &node_map);

  void SetGeneralizeRelatedParam(ge::OptimizeUtility *const utility,
                                 const FusionPriorityMgrPtr &fusion_priority_mgr_ptr);

  Status GetOpStoreAdapter(const OpImplType &impl_type, OpStoreAdapterPtr &op_store_adapter);

  Status CompileMemSet(vector<ge::NodePtr> &node_vec);

  void GetOpKernelInfoPtr(const ge::NodePtr &node_ptr, OpKernelInfoPtr &op_kernel_ptr) const;

  SubOpsStorePtr GetSubOpsStore(const OpImplType &impe_type) const;

  Status SetMemSetOpWorkspaceInfo(ge::OpDescPtr &op_desc, bool &atomic_node_flag) const;
 private:
  bool init_flag_;

  TbeInfoAssemblerPtr tbe_info_assembler_ptr_;

  std::map<std::string, SubOpsStorePtr> map_all_sub_store_info_;

  std::unordered_map<uint64_t, CheckSuportResult> checksupport_cache_;

  std::string op_kernel_store_type_;

  std::string engine_name_;

  bool check_support_static_flag_;

  ge::OptimizeUtility *optimize_utility_;

  FusionPriorityMgrPtr fusion_priority_mgr_ptr_;

  Status CompileOpGetTvmJsonInfo(ScopeNodeIdMap &fusion_nodes_map, CompileResultMap &compile_ret_map) const;

  Status CompileOpGetImplType(const ge::NodePtr &node, OpImplType &impl_type) const;

  bool IsExistInTBECustom(const ge::NodePtr &node_ptr);

  void GetAndSetCustomOpFilePath(const ge::NodePtr &node, const FEOpsStoreInfo &ops_store) const;

  void SetOpsPathNamePrefix(const ge::NodePtr &node, const OpKernelInfoPtr &op_kernel_ptr) const;

  Status GetDefFeOpsStoreInfo(FEOpsStoreInfo &fe_ops_store);

  Status GetOpImplyRealPath(std::string op_imply_relative_path, const OpImplType &op_impl_type,
                            std::string &op_store_real_path, std::string &op_imply_real_path,
                            const ge::NodePtr &node_ptr) const;

  Status UpdateOpImplyPath(const ge::NodePtr &node_ptr, std::string &op_store_real_path, const OpImplType &op_impl_type,
                           SubOpInfoStorePtr &sub_custom_ops_kernel_ptr) const;

  Status GetDynamicCustomOpStoreInfoByNode(const ge::NodePtr &node_ptr, vector<std::string> &json_files,
                                           SubOpInfoStorePtr &sub_dyna_custom_ops_store_ptr);

  Status SetDynaCustomOpStoreToAllStore(FEOpsStoreInfo &fe_ops_store,
                                        SubOpInfoStorePtr &sub_dyna_custom_ops_kernel_ptr);

  Status GetAllAtomicCleanNode(ge::NodePtr &node_ptr, vector<ge::NodePtr> &atomic_node_vec) const;

  Status CompileAndSetKernelNameForMemSet(const vector<ge::NodePtr> &node_vec,
                                               vector<ge::NodePtr> &memset_nodes);

  std::vector<uint32_t> CompileGetAtomicOutput(const ge::OpDescPtr &op_desc_ptr) const;

  Status SetAtomicOpAttr(ge::OpDescPtr &op_desc, bool &atomic_node_flag) const;
  
  Status CompileSetAtomicMemSetWorkSpace(ge::NodePtr &node_ptr,
                                         MemsetWorkspaceInfo &memset_workspace_info) const;

  bool CheckCustomOp(const ge::NodePtr &node, const FEOpsStoreInfo &ops_store) const;

  Status PreCompileAndCompile(std::unordered_map<OpStoreAdapterPtr, vector<PreCompileNodePara>> &node_map,
                              const ge::NodePtr &node, ScopeNodeIdMap &fusion_node_map, const bool is_fuzz_build);

  Status FuzzCompileAndGetResult(const ge::NodePtr &node, const OpStoreAdapterPtr &op_store_adapter,
                                 ScopeNodeIdMap &fusion_node_map) const;

  Status CompileSingleOp(const ge::NodePtr &node_ptr, const bool is_fuzz_build = false);

  Status CompileMultipleOp(vector<ge::NodePtr> &node_vec, const bool is_fuzz_build = false);

  bool IsNeededCompile(ge::OpDescPtr &op_desc_ptr) const;

  Status GetSubOpsStore(const std::string &op_store_name, SubOpsStorePtr &sub_ops_store,
                        std::ostringstream &reason_oss) const;

  Status GetOpKernel(const std::string &op_type, const FEOpsStoreInfo &ops_store, OpKernelInfoPtr &op_kernel_ptr,
      std::ostringstream &reason_oss, uint64_t &not_support_reason_id) const;

  Status SetCutSupportedInfo(const ge::NodePtr &node) override;

  Status FuzzGeneralAndCompileSingleOp(const ge::NodePtr &node_ptr);
  Status FuzzGeneralAndCompileFusionOp(const ge::NodePtr &node_ptr);
  Status GeneralizeSingleOp(const ge::NodePtr &node_ptr) const;

  bool CheckIsDynamicShape(const SubOpsStorePtr &sub_ops_store_ptr, const ge::NodePtr &node,
                           const CheckSupportMode &check_mode, CheckSupportParam &check_param) const;
  bool CheckIsStaticShape(const SubOpsStorePtr &sub_ops_store_ptr, const ge::NodePtr &node,
                          const CheckSupportMode &check_mode, CheckSupportParam &check_param) const;
  void JoinNotSupportDynamicReason(const OpKernelInfoPtr &op_kernel_ptr, const ge::NodePtr &node,
                                   UnSupportedReason &reason) const;
  void UpdateNodeShapeAndRange(const ge::NodePtr &node_ptr) const;
  void UpdateTensorShapeAndRange(const ge::OpDescPtr &op_desc, const ge::GeTensorDescPtr &tensor_desc) const;
  void BackupGraphParentNodeAndIndex(const ge::ComputeGraphPtr &graph, FusionParenNodeAndIndex &node_map) const;
  void RollbackGraphParentNodeAndIndex(const ge::ComputeGraphPtr &graph,
                                       const FusionParenNodeAndIndex &node_map) const;
  Status CompareTensorDescAndSubgraphData(const ge::ComputeGraphPtr &graph, const ge::NodePtr &node_ptr,
                                          const bool is_to_subgraph) const;
  Status CompileSubGraph(const ge::ComputeGraphPtr &graph);
  void UpdateSubGraphShapeAndRange(const ge::ComputeGraphPtr &graph) const;
  void CopyTensorDesc(const ge::GeTensorDescPtr &src, const ge::GeTensorDescPtr &dst) const;
  void ClearOpAttr(const ge::NodePtr &node_ptr) const;
  void SetConstValueAttr(ge::ConstNodePtr &const_node_ptr, const ge::OpDescPtr &op_desc_ptr) const;
  Status CompileOpTiling(const ge::NodePtr &node) const;
  Status SetWorkSpaceForAtomicMemSet(const MemsetWorkspaceInfo &memset_workspace_info, const ge::NodePtr &node,
      const ge::ComputeGraphPtr &tmp_graph, vector<ge::NodePtr> &memset_nodes) const;
};
using FEOpsKernelInfoStorePtr = std::shared_ptr<FEOpsKernelInfoStore>;
}  // namespace fe
#endif  // FUSION_ENGINE_OPTIMIZER_OPS_KERNEL_STORE_FE_OPS_KERNEL_INFO_STORE_H_
