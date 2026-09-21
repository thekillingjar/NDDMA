/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_HYBRID_MODEL_HYBRID_MODEL_H_
#define GE_HYBRID_MODEL_HYBRID_MODEL_H_

#include <vector>
#include <queue>
#include "graph/load/model_manager/data_inputer.h"
#include "graph/load/model_manager/task_info/task_info.h"
#include "graph/node.h"
#include "hybrid/common/tensor_value.h"
#include "hybrid/model/node_item.h"
#include "hybrid/model/graph_item.h"
#include "common/model/ge_root_model.h"
#include "platform/platform_info.h"
#include "graph/load/model_manager/aipp_utils.h"
#include "graph/runtime_inference_context.h"
#include "graph/bin_cache/bin_cache_def.h"

namespace ge {
namespace hybrid {
class HybridModel {
 public:
  explicit HybridModel(GeRootModelPtr ge_model);

  ~HybridModel();

  Status Init(const bool is_single_op = false);

  const NodeItem *GetNodeItem(const NodePtr &node) const;

  uint64_t GetSessionId() const {
    return root_runtime_param_.session_id;
  }

  void *GetGlobalStep() const;

  GeModelPtr GetGeModel(const NodePtr &node) const;

  NodeItem *MutableNodeItem(const NodePtr &node) const;

  fe::PlatFormInfos GetPlatformInfo() const {
    return platform_infos_;
  }

  size_t TotalVarMemSize() const {
    return root_runtime_param_.var_size;
  }

  void SetDeviceId(const uint32_t device_id)  {
    device_id_ = device_id;
  }

  uint32_t GetDeviceId() const {
    return device_id_;
  }

  void SetModelId(const uint32_t model_id) {
    model_id_ = model_id;
  }

  void SetOmName(const std::string &om_name) {
    om_name_ = om_name;
  }

  const std::string &GetOmName() const {
    return om_name_;
  }

  void SetFileConstantWeightDir(const std::string &file_constant_weight_dir) {
    file_constant_weight_dir_ = file_constant_weight_dir;
  }

  const std::string &GetFileConstantWeightDir() const {
    return file_constant_weight_dir_;
  }

  uint32_t GetModelId() const {
    return model_id_;
  }

  bool IsSingleOp() const {
    return is_single_op_;
  }

  bool IsExecuteByRtV2() const {
    return execute_by_rt_v2_;
  }

  bool HasObserver() const {
    return has_observer_;
  }

  fuzz_compile::NodeBinMode GetNodeBinMode() const {
    return node_bin_mode_;
  }
  void SetNodeBinMode(const fuzz_compile::NodeBinMode node_bin_mode) {
    node_bin_mode_ = node_bin_mode;
  }

  TensorValue* GetVariable(const std::string &name) const;

  NodePtr GetVariableNode(const std::string &name) const;

  TensorValue* GetTensor(const NodePtr &node) const;

  TensorBuffer* GetModelWeight(const std::string &subgraph_name) const;

  const std::map<int64_t, std::vector<std::pair<int32_t, GeTensorPtr>>> &GetHostTensors() const;

  const std::vector<domi::TaskDef>* GetTaskDefs(const NodePtr &node) const;

  const GraphItem *GetRootGraphItem() const;

  const ComputeGraphPtr &GetRootGraph() const;

  const GraphItem *GetSubgraphItem(const std::string &graph_name) const;

  const GraphItem *GetSubgraphItem(const ComputeGraphPtr &subgraph) const;

  const std::string &GetModelName() const;

  Status GetDynamicBatchInfo(std::vector<std::vector<int64_t>> &batch_info, int32_t &dynamic_type) const;

  void GetUserDesignateShapeOrder(std::vector<std::string> &user_input_shape_order) const;

  void GetModelAttr(std::vector<std::string> &dynamic_output_shape_info) const;

  Status GetInputOutputDescInfo(std::vector<InputOutputDescInfo> &input_desc,
                                std::vector<InputOutputDescInfo> &output_desc,
                                std::vector<uint32_t> &input_formats,
                                std::vector<uint32_t> &output_formats);

  Status GetInputDescInfo(std::vector<InputOutputDescInfo> &input_desc, std::vector<uint32_t> &formats_result);

  void CreateOutput(const ConstGeTensorDescPtr &output_desc, InputOutputDescInfo &output_desc_info,
                    uint32_t &format_result) const;

  Status GetOutputDescInfo(std::vector<InputOutputDescInfo> &output_desc, std::vector<uint32_t> &formats_result) const;

  void CreateInputDimsInfo(const OpDescPtr &op_desc, InputOutputDescInfo &input) const;

  void SetModelDescVersion(const bool is_new_model_desc) { is_new_model_desc_ = is_new_model_desc; }

  void SetInputDimsAndShapeRangesInfo(const std::vector<int64_t> &model_input_dims,
                                      const std::vector<std::pair<int64_t, int64_t>> &shape_ranges,
                                      InputOutputDescInfo &input) const;
  void SaveSpecifyAttrValues();

  Status GetOpAttr(const std::string &op_name, const std::string &attr_name, std::string &attr_value) const;

  Status GetAippInfo(const uint32_t index, AippConfigInfo &aipp_info) const;
  Status GetAippType(const uint32_t index, InputAippType &aipp_type, size_t &aipp_data_index) const;
  bool CheckHostMemInputOptimization(const std::vector<NodePtr> &node_with_hostmem) const;
  void SetNeedHostMemOpt(const std::vector<NodePtr> &node_with_hostmem, const bool need_host_mem_opt) const;
  Status ReportProfilingData() const;
  void *GetOverflowAddr() const;
  Status SetOverflowAddr(void *const buffer, const size_t size);

 private:
  friend class HybridModelBuilder;
  friend class HybridModelAsyncExecutor;
  friend class HybridModelRtV2Executor;

  TensorValue* GetConstant(const NodePtr &node) const;
  Status InitAippInfoAndType();
  void PrintDynamicType() const;
  std::string model_name_;
  GeRootModelPtr ge_root_model_;
  std::map<uint32_t, NodeItem *> input_nodes_;
  ComputeGraphPtr root_graph_;
  ComputeGraphPtr orig_root_graph_;
  std::map<std::string, NodePtr> device_variable_nodes_; //lint !e148
  std::map<std::string, NodePtr> host_variable_nodes_; //lint !e148
  std::map<std::string, std::unique_ptr<TensorValue>> variable_tensors_;
  std::map<NodePtr, std::unique_ptr<TensorValue>> constant_tensors_;
  std::map<NodePtr, std::vector<domi::TaskDef>> task_defs_;
  std::map<NodePtr, GeModelPtr> known_shape_sub_models_;

  std::unique_ptr<GraphItem> root_graph_item_;
  std::map<std::string, std::unique_ptr<GraphItem>> subgraph_items_;
  std::map<NodePtr, std::unique_ptr<NodeItem>> node_items_;
  std::map<int64_t, std::vector<std::pair<int32_t, GeTensorPtr>>> host_tensors_;

  bool is_new_model_desc_ = false;    // support aipp
  bool is_single_op_ = false;
  bool has_observer_ = false;
  fuzz_compile::NodeBinMode node_bin_mode_;

  // runtime fields
  uint32_t device_id_ = 0U;
  uint32_t model_id_ = 0U;
  uint8_t *var_mem_base_ = nullptr;
  std::unique_ptr<TensorBuffer> globalworkspace_overflow_addr_;
  std::map<std::string, std::unique_ptr<TensorBuffer>> weight_buffer_map_;
  RuntimeParam root_runtime_param_;
  fe::PlatFormInfos platform_infos_;
  std::string om_name_;
  std::string file_constant_weight_dir_;
  std::unique_ptr<TensorBuffer> global_step_;
  // op name to attrs mapping
  std::map<std::string, std::map<std::string, std::vector<std::string>>> op_name_to_attrs_;
  std::map<uint32_t, AippConfigInfo> aipp_infos_;
  std::map<uint32_t, std::pair<InputAippType, size_t>> aipp_types_;

  bool execute_by_rt_v2_ = false;
};
}  // namespace hybrid
}  // namespace ge
#endif // GE_HYBRID_HYBRID_GRAPH_H_
