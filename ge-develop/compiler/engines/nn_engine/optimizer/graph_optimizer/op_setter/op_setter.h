/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_OP_SETTER_OP_SETTER_H_
#define FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_OP_SETTER_OP_SETTER_H_

#include "ops_store/op_kernel_info.h"
#include "common/fe_context_utils.h"
#include "graph_optimizer/op_judge/format_and_dtype/update_desc/op_format_dtype_update_desc_base.h"

namespace fe {
class OpSetter {
 public:
  explicit OpSetter(const std::string &engine_name);
  virtual ~OpSetter();
  Status InitializeQuerier();
  Status SetOpInfo(const ge::ComputeGraph& graph) const;
  Status OnlySetOpDescAttr(const ge::ComputeGraph& graph) const;
  static Status SetTbeOpSliceInfo(const ge::NodePtr &node_ptr, const std::string &engine_name,
                                  OpKernelInfoPtr &op_kernel_info_ptr);
  static Status SetOpInfoByNode(const ge::NodePtr &node_ptr, const std::string &engine_name,
                                const bool only_set_attr = false);
  Status MultiThreadSetOpInfo(ge::ComputeGraph &graph, bool only_set_attr = false) const;

  void SetOpImplMode(const ge::ComputeGraph& graph) const;
  static void SetOpDebugAttr(const ge::ComputeGraph& graph);

  static void SetQuantDumpableAttr(const ge::ComputeGraph& graph);

  void SetFallbackAttr(const ge::ComputeGraph &graph, const fe::PrecisionMode &precision_mode, bool &need_update_stream_core_limit) const;

  void DeleteFusionScope(ge::ComputeGraph& graph) const;

 private:
  static Status SetOpDescAttr(const OpKernelInfoPtr& op_kernel_info_ptr, const std::string& attr_name,
                              const std::string& ge_attr_name, ge::OpDescPtr op_desc_ptr);
  static Status SetOpDescAttrByNode(const ge::OpDescPtr &op_desc_ptr, const OpKernelInfoPtr &op_kernel_info_ptr);
  static Status SetOpSliceInfoByNode(const ge::NodePtr &node_ptr, const std::string &engine_name,
                                     OpKernelInfoPtr &op_kernel_info_ptr);
  static void GetOpKernelInfoByImplType(const ge::OpDescPtr &op_desc_ptr, const std::string &engine_name,
                                        OpKernelInfoPtr &op_kernel_info_ptr);
  void SetOpImplModeByNode(const ge::NodePtr &node_ptr) const;
  void SetOpCustomImplModeByNode(const ge::NodePtr &node_ptr) const;
  bool FallbackConfigured(const ge::NodePtr &node_ptr, const OpKernelInfoPtr &op_kernel_info_ptr,
                          std::vector<bool> &fallback_flags) const;
  bool SupportFallback(const ge::NodePtr &node_ptr, const uint32_t &matched_index,
                       const OpKernelInfoPtr &op_kernel_info_ptr) const;
  void SetAttrForAclnnLowering(const ge::NodePtr &node_ptr, const fe::PrecisionMode &precision_mode) const;
  bool SetAclnnAttr(const ge::NodePtr &node_ptr, const uint32_t &matched_index,
                    const OpKernelInfoPtr &op_kernel_info_ptr, const fe::PrecisionMode &precision_mode) const;
  bool IsDefaultEnableAclnn(const ge::NodePtr &node_ptr) const;
  std::string engine_name_;
  FormatDtypeQuerierPtr format_dtype_querier_ptr_;
  static const std::map<string, string> attr_map_;
};
}  // namespace fe
#endif  // FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_OP_SETTER_OP_SETTER_H_
