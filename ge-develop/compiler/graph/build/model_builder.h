/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_BUILD_MODEL_BUILDER_H_
#define GE_GRAPH_BUILD_MODEL_BUILDER_H_

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "framework/common/op/ge_op_utils.h"
#include "common/tbe_handle_store/tbe_kernel_store.h"
#include "common/tbe_handle_store/cust_aicpu_kernel_store.h"
#include "framework/common/types.h"
#include "framework/common/util.h"
#include "graph/compute_graph.h"
#include "graph/manager/graph_manager_utils.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/build/stream/stream_allocator.h"
#include "graph/model.h"
#include "graph/node.h"
#include "common/model/ge_model.h"
#include "framework/omg/omg_inner_types.h"
#include "graph/build/model_data_info.h"

namespace ge {
class ModelBuilder {
 public:
  ModelBuilder(uint64_t session_id, ge::ComputeGraphPtr compute_graph, const Graph2SubGraphInfoList &subgraphs,
               const std::map<std::string, int32_t> &stream_max_parallel_num, bool hcom_parallel,
               int32_t mode = static_cast<int32_t>(domi::BuildMode::GEN_TASK_WITHOUT_FUSION));

  ModelBuilder(const ModelBuilder &) = delete;

  ModelBuilder &operator=(const ModelBuilder &op) = delete;

  ~ModelBuilder();

  Status SaveDataToModel(ge::Model &model, ge::GeModel &ge_model);
  Status PreBuildModel();
  Status BuildModelForGetTask(ge::Model &model);
  Status BuildModelDefForStream(ge::Model &model);
  Status RefreshRealStream(std::unordered_map<int64_t, std::vector<domi::TaskDef>> &node_id_2_node_tasks);
  ge::Status BuildModelForGetDynShapeTask(ge::Model &model_def);
  Status AssignStreamForDynamicShapeGraph(ComputeGraphPtr &compute_graph);

  ge::Buffer GetWeightBuffer() const;

  Status MergeWeights();

  Status BuildModelForEvaluate(ModelDataInfo &model) const;

  void SetHasAssignedVarMemFlag(const bool flag) {
    has_assigned_var_mem = flag;
  }
  bool GetHasAssignedVarMemFlag() const {
    return has_assigned_var_mem;
  }

 protected:
  void AddNodeInputProperty() const;

  void ClearOriginalFormat() const;

 private:
  bool SetInputConst(const OpDescPtr &op_desc, const NodePtr &src_node,
                     size_t index, std::vector<bool> &is_input_const) const;

  void SetInputIsConst(const ge::NodePtr &n) const;

  void SetModelVersion(ge::Model &model) const;

  Status CalcOutputSize(const ge::NodePtr &n) const;

  Status AdjustConstWeightSize(const ge::NodePtr &node, size_t &mem_offset);

  Status SetInputOutputDesc();

  Status AdjustInputTensorFlag() const;

  Status BuildModelDef(ge::Model &model);

  Status BuildModelDefForMem(ge::Model &model);

  Status InitL1FusionOption();

  Status CompileSingleOp() const;

  void CollectCheckAicpuAttr(const OpDescPtr &op_desc, std::set<std::string> &aicpu_op_types,
                               std::set<std::string> &aicpu_tf_op_types) const;

  void SetModelCheckAicpuAttr(ge::Model &model, std::set<std::string> &aicpu_op_types,
                                std::set<std::string> &aicpu_tf_op_types) const;

  Status SaveSoftSyncOpWeight() const;
  Status SaveAtomicTBEKernel(const OpDescPtr &op_desc);

  Status SavaAtomicWorkspace(const OpDescPtr &op_desc) const;
  Status SaveNormalTBEKernel(const OpDescPtr &op_desc);
  Status SaveCustAiCpuKernel(const OpDescPtr &op_desc, std::set<std::string> &aicpu_name_set);
  Status SaveFftsPlusTBEKernel(const OpDescPtr &op_desc);
  TBEKernelPtr CreateOpTBEKernel(const OpDescPtr &op_desc, const std::string &prefix_kernel_name) const;
  // In order to optimize the size of om,
  // delete the attributes saved in tbekernelstore and nodes on the graph at the same time.
  void DelNodeRepeatSaveAttr();
  void ReuseWeightMem(const size_t output_size, GeTensorPtr &weight, bool &find_same_const, size_t &current_mem_offset);

  uint64_t session_id_;

  std::map<uint64_t, size_t> mem_type_to_mem_offset_;

  std::vector<std::vector<int64_t>> sub_mem_offsets_;

  std::unordered_map<size_t, std::vector<std::pair<void *, size_t>>> reuse_weight_map_;

  std::set<size_t> weight_offset_need_feeded_;

  size_t weight_offset_;

  ge::ComputeGraphPtr compute_graph_;

  const Graph2SubGraphInfoList &subgraphs_;

  StreamAllocator stream_allocator_;

  int64_t stream_num_;
  int64_t notify_num_;
  std::vector<uint32_t> notify_types_;
  int64_t event_num_;
  std::vector<int64_t> huge_streams_;

  uint32_t label_num_;

  ge::Buffer weight_buffer_;

  std::map<std::string, int32_t> stream_max_parallel_num_;
  bool hcom_parallel_;

  int32_t build_mode_;
  size_t max_mem_offset_;
  size_t host_max_mem_offset_;
  size_t host_svm_max_mem_offset_;
  size_t p2p_mem_offset_;
  size_t zero_copy_mem_size_;

  TBEKernelStore tbe_kernel_store_;
  CustAICPUKernelStore cust_aicpu_kernel_store_;

  uint8_t platform_type_;
  bool is_loop_graph_;
  bool is_l1_fusion_enable_;
  bool has_assigned_var_mem;
};
}  // namespace ge
#endif  // GE_GRAPH_BUILD_MODEL_BUILDER_H_
