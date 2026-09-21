/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_OP_COMPILER_OP_FORMAT_TUNE_H_
#define FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_OP_COMPILER_OP_FORMAT_TUNE_H_

#include "graph_optimizer/op_compiler/op_compiler.h"
#include "common/graph/fe_graph_utils.h"
#include "format_selector/manager/format_dtype_querier.h"
#include "graph_optimizer/op_judge/format_and_dtype/update_desc/op_format_dtype_update_desc_base.h"
#include "graph_optimizer/shape_format_transfer/trans_node_manager/trans_node_manager.h"

namespace fe {
using FormatDtypeQuerierPtr = std::shared_ptr<FormatDtypeQuerier>;
using OpFormatDtypeUpdateTensorDescBasePtr = std::shared_ptr<OpFormatDtypeUpdateDescBase>;
using TransNodeManagerPtr = std::shared_ptr<TransNodeManager>;

struct FormatTuneInfo {
ge::NodePtr &node;
const size_t &anchor_index;
ge::Format &new_format;
const bool &is_input;
};

class OpCompilerFormatTune {
 public:
  explicit OpCompilerFormatTune(const std::string& engine_name);
  ~OpCompilerFormatTune();

  Status SetTuneFormatReq(ge::ComputeGraph& graph, const FEOpsKernelInfoStorePtr &ops_kernel_info_store_ptr);

  bool IsNeedTuneFormat(const ge::NodePtr &node,
                        const OpKernelInfoPtr &op_kernel_info_ptr,
                        std::vector<int64_t> &input_tuneformat_index_vec,
                        std::vector<int64_t> &output_tuneformat_index_vec) const;

  void GetMatchedIndexVec(std::map<std::string, std::vector<ge::DataType>> &datatype_map,
                          const OpKernelInfoPtr &op_kernel_info_ptr,
                          const ge::OpDescPtr &op_desc_ptr);

  Status GetFormatSolutionSpace(const ge::NodePtr &node, const OpKernelInfoPtr &op_kernel_info_ptr,
                                const FEOpsKernelInfoStorePtr &ops_kernel_info_store_ptr,
                                std::vector<int64_t> &input_tuneformat_index_vec,
                                std::vector<int64_t> &output_tuneformat_index_vec);

  void FilterFormatIndex(const std::vector<InputOrOutputInfoPtr> &input_or_output_infos,
                         const std::vector<int64_t> &tuneformat_index_vec,
                         const ge::OpDesc::Vistor<ge::GeTensorDescPtr> &tensor_descs,
                         std::map<std::string, std::vector<ge::Format>> &format_map);

  bool IsLegalFormat(const ge::GeTensorDescPtr &tensor, const ge::Format &origin_format,
                     const ge::Format &cur_format) const;

  bool IsLegalOriginFormat(const ge::Format &origin_format, const ge::Format &cur_format) const;

  bool IsLegalHeavyFormat(const ge::GeTensorDescPtr &tensor, const ge::Format &origin_format,
                          const ge::Format &cur_format) const;

  Status SetTuneFormatReqAttr(const std::vector<InputOrOutputInfoPtr> &input_or_output_infos,
                              const ge::OpDesc::Vistor<ge::GeTensorDescPtr> &tensor_descs,
                              std::map<string, std::vector<ge::Format>> &format_map) const;

  Status GetFormatTuneKnowledge(ge::NodePtr &node, nlohmann::json &params_json);

  Status UpdataFormatAndShapeByFormatTune(FormatTuneInfo &fomat_tune_info);

  Status UpdataGraphByFormatTune(ge::ComputeGraph &graph, ge::NodePtr &node, const bool &update_graph_forward_flag,
                                 const bool &update_graph_backward_flag);

  Status UpdateTensorByCannKbResult(ge::NodePtr &node, const bool &is_input, nlohmann::json &json,
                                    bool &update_graph_flag);

  Status UpdateTensorByNodeAttr(ge::NodePtr &node, const bool &is_input, bool &update_graph_flag);

  Status UpdateTuneFormatByCannKbResult(ge::ComputeGraph &graph, bool &need_re_precompile);

  Status UpdateTuneFormatByNodeAttrInner(ge::ComputeGraph &graph);

  Status ReplacePldAndEnd(std::map<std::string, ge::NodePtr> &ori_node_map,
                          ge::ComputeGraphPtr &graph_ptr);

  Status UpdateTuneFormatByNodeAttr(ge::ComputeGraph &graph);
  
  static bool GetTuneKnowledgeResult(const ge::NodePtr &node, const std::vector<std::string> &op_unique_keys,
                                     const std::map<std::string, std::string> &search_config,
                                     std::string &kb_result_str);
 private:
  bool IsFftsPlusThreadReuseOp(const ge::NodePtr &node) const;
  bool HasTuneFormatSwitch(const ge::NodePtr &node, const OpKernelInfoPtr &op_kernel_info_ptr,
    std::vector<int64_t> &input_tuneformat_index_vec, std::vector<int64_t> &output_tuneformat_index_vec) const;

  std::string engine_name_;
  std::vector<int32_t> matched_format_index_vec;
};
using OpCompilerFormatTunePtr = std::shared_ptr<OpCompilerFormatTune>;
}

#endif // FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_OP_COMPILER_OP_FORMAT_TUNE_H_
