/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_OPTIMIZER_ADAPTER_ADAPTER_ITF_OP_STORE_ADAPTER_H_
#define FUSION_ENGINE_OPTIMIZER_ADAPTER_ADAPTER_ITF_OP_STORE_ADAPTER_H_

#include "common/op_store_adapter_base.h"
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "nlohmann/json.hpp"
#include "graph/node.h"
#include "graph/op_desc.h"
#include "graph/op_kernel_bin.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/node_utils.h"
#include "common/fe_op_info_common.h"
#include "common/util/op_info_util.h"
#include "common/aicore_util_types.h"
#include "ops_store/op_kernel_info.h"
#include "tensor_engine/tbe_op_info.h"
#include "common/opskernel/ops_kernel_info_store.h"

namespace fe {
using te::TbeOpInfoPtr;
struct PreCompileNodePara {
  ge::Node *node;
  OpKernelInfoPtr op_kernel_info_ptr;
  std::string imply_type_str;
  std::string op_dsl_file_path;
  std::string session_graph_id;
  TbeOpInfoPtr tbe_op_info_ptr;
};

struct CheckSupportParam {
  OpKernelInfoPtr op_kernel_ptr;
  UnSupportedReason unsupport_reason;
  bool check_static = true;
  bool all_impl_checked = false;
  bool dynamic_compile_static = false;
};

using RootSet = std::unordered_set<ge::NodePtr>;

struct NodeGeneralInfo {
  bool is_found_in_opstore = false;
  bool is_sub_graph_node = false;
  bool is_support_dynamic_shape = false;
  bool is_limited_range = false;
  TbeOpInfoPtr op_info;
  OpKernelInfoPtr op_kernel;
  RootSet disjoint_root_set;
  std::map<ge::GeTensorDescPtr, RootSet> inputs_root_map;
};
using NodeGeneralInfoPtr = std::shared_ptr<NodeGeneralInfo>;
class OpStoreAdapter : public OpStoreAdapterBase {
 public:
  virtual ~OpStoreAdapter() override {}

  /*
   *  @ingroup fe
   *  @brief   initialize op adapter
   *  @return  SUCCESS or FAILED
   */
  virtual Status Initialize(const std::map<std::string, std::string> &options) override = 0;

  /*
   *  @ingroup fe
   *  @brief   finalize op adapter
   *  @return  SUCCESS or FAILED
   */
  virtual Status Finalize() override = 0;

  virtual Status FinalizeSessionInfo(const std::string& session_graph_id) {
    (void)session_graph_id;
    return SUCCESS;
  }
  /*
   *  @ingroup fe
   *  @brief   check op wether supported in ops store
   *  @param   [in] op_desc   infomation of op in ge
   *  @return  true or false
   */
  virtual bool CheckSupport(const ge::NodePtr &node, CheckSupportParam &check_param,
                            const bool &is_dynamic_impl, std::string &reason) {
    (void)node;
    (void)check_param;
    (void)is_dynamic_impl;
    (void)reason;
    return true;
  }

  virtual Status CompileOp(ScopeNodeIdMap &fusion_nodes_map, map<int64_t, std::string> &json_path_map,
                           std::vector<ge::NodePtr> &buff_fus_compile_failed_nodes,
                           const std::vector<ge::NodePtr> &buff_fus_to_del_nodes) override {
    (void)fusion_nodes_map;
    (void)json_path_map;
    (void)buff_fus_compile_failed_nodes;
    (void)buff_fus_to_del_nodes;
    return SUCCESS;
  }

  virtual Status CompileOp(CompileInfoParam &compile_info) override {
    (void)compile_info;
    return SUCCESS;
  }

  virtual Status PreCompileOp(vector<PreCompileNodePara> &compile_para_vec) {
    (void)compile_para_vec;
    return SUCCESS;
  }

  virtual Status TaskFusion(const ScopeNodeIdMap &fusion_nodes_map, CompileResultMap &compile_ret_map) {
    (void)fusion_nodes_map;
    (void)compile_ret_map;
    return SUCCESS;
  }

  virtual Status GetLXOpCoreType(const ge::NodePtr &node, const OpKernelInfoPtr &op_kernel_info_ptr,
                                 const bool &is_dynamic_impl, std::string &lx_op_core_type_str) {
    (void)node;
    (void)op_kernel_info_ptr;
    (void)is_dynamic_impl;
    (void)lx_op_core_type_str;
    return SUCCESS;
  }

  virtual Status SelectOpFormat(const ge::NodePtr &node, const OpKernelInfoPtr &op_kernel_info_ptr,
                                const bool &is_dynamic_impl, const HeavyFormatInfo &heavy_format_info,
                                std::string &op_format_dtype_str) {
    (void)node;
    (void)op_kernel_info_ptr;
    (void)is_dynamic_impl;
    (void)heavy_format_info;
    (void)op_format_dtype_str;
    return SUCCESS;
  }

  virtual Status SetTbeOpSliceInfo(const ge::NodePtr &node_ptr, OpKernelInfoPtr &op_kernel_info_ptr) {
    (void)node_ptr;
    (void)op_kernel_info_ptr;
    return SUCCESS;
  }

  virtual Status GeneralizeNode(const ge::NodePtr &node, const te::TbeOpInfo &op_info,
                                te::TE_GENERALIZE_TYPE generalize_type) {
    (void)node;
    (void)op_info;
    (void)generalize_type;
    return SUCCESS;
  }

  virtual Status GetRangeLimitType(const ge::NodePtr &node_ptr,
                                   const te::TbeOpInfo &tbe_op_info, bool &is_limited) const {
    (void)node_ptr;
    (void)tbe_op_info;
    (void)is_limited;
    return SUCCESS;
  }

  virtual Status LimitedNodesCheck(bool &is_support, const te::TbeOpInfo &tbe_op_info,
                                   std::vector<size_t> &upper_limited_input_indexs,
                                   std::vector<size_t> &lower_limited_input_indexs) {
    (void)is_support;
    (void)tbe_op_info;
    (void)upper_limited_input_indexs;
    (void)lower_limited_input_indexs;
    return SUCCESS;
  }

  virtual Status IsGeneralizeFuncRegistered(bool &is_registered, const te::TbeOpInfo &op_info) {
    (void)is_registered;
    (void)op_info;
    return SUCCESS;
  }

  virtual Status GetOpUniqueKeys(const ge::NodePtr &node, const OpKernelInfoPtr &op_kernel_info_ptr,
                                 std::vector<std::string> &op_unique_keys) {
    (void)node;
    (void)op_kernel_info_ptr;
    (void)op_unique_keys;
    return SUCCESS;
  }

  virtual Status FeedNodeGeneralInfo(const ge::NodePtr &node_ptr, NodeGeneralInfoPtr &node_info_ptr) const {
    (void)node_ptr;
    (void)node_info_ptr;
    return SUCCESS;
  }

  virtual Status FeedNodeGeneralInfoFromOpStore(const ge::NodePtr &node_ptr,
                                                NodeGeneralInfoPtr &node_info_ptr) const {
    (void)node_ptr;
    (void)node_info_ptr;
    return SUCCESS;
  }

  virtual Status GetRangeLimit(const NodeGeneralInfoPtr &node_info_ptr, const ge::NodePtr &node_ptr) const {
    (void)node_info_ptr;
    (void)node_ptr;
    return SUCCESS;
  }

  virtual Status UpdatePrebuildPattern() {
    return SUCCESS;
  }

  virtual Status GetDynamicPromoteType(const ge::NodePtr &node, const OpKernelInfoPtr &op_kernel_info_ptr,
                                       std::string &promote_str) const {
    (void)node;
    (void)op_kernel_info_ptr;
    (void)promote_str;
    return SUCCESS;
  }

  virtual void GetAllCompileStatisticsMsg(std::vector<std::string> &statistics_msg) const {
    (void)statistics_msg;
  }

  virtual bool JudgeBuiltInOppKernelInstalled() const {
    return true;
  }

  virtual Status SuperKernelCompile(const ScopeNodeIdMap &fusion_nodes_map, CompileResultMap &compile_ret_map) {
    (void)fusion_nodes_map;
    (void)compile_ret_map;
    return SUCCESS;
  }

  virtual std::string GetCurKernelMetaDir() const {
    return "";
  }

  virtual bool IsNeedSkipOpJudge(const ge::NodePtr &node, const OpKernelInfoPtr &op_kernel_info_ptr) const {
    (void)node;
    (void)op_kernel_info_ptr;
    return false;
  }
  
  virtual void SetOpsKernelInfoStore(const std::shared_ptr<ge::OpsKernelInfoStore> ops_kernel_info_store_ptr) {
    (void)ops_kernel_info_store_ptr;
  }
};
using OpStoreAdapterPtr = std::shared_ptr<OpStoreAdapter>;
}  // namespace fe
#endif  // FUSION_ENGINE_OPTIMIZER_ADAPTER_ADAPTER_ITF_OP_STORE_ADAPTER_H_
