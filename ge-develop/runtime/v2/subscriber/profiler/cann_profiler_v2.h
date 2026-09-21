/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_SUBSCRIBER_PROFILER_CANN_PROFILER_V2_H_
#define AIR_CXX_RUNTIME_V2_SUBSCRIBER_PROFILER_CANN_PROFILER_V2_H_
#include "base_executor_profiler.h"
#include "exe_graph/lowering/graph_frame.h"
#include "exe_graph/runtime/dfx_info_filler.h"
#include "register/kernel_registry.h"
#include "core/execution_data.h"
#include "engine/aicore/fe_rt2_common.h"
#include "aprof_pub.h"
#include "subscriber/profiler/dfx_extend_info.h"

namespace gert {
struct TensorInfoWrapper {
  MsprofAdditionalInfo tensor_info;
  uint64_t tensor_num;
  std::vector<Chain *> shapes;
  void FillShapeInfo();
  bool filled_by_kernel;
};

struct ProfNodeAdditionInfo {
  MsprofCompactInfo node_basic_info;
  std::vector<TensorInfoWrapper> tensor_info_wrappers;
  MsprofApi api;
  MsprofAdditionalInfo context_info;
  uint64_t input_tensor_num;
  uint64_t output_tensor_num;
  std::string json_info;
  bool mix_launch_enable{false};
};

class CannProfilingInfoWrapper : public ProfilingInfoWrapper {
 public:
  CannProfilingInfoWrapper() = default;
  explicit CannProfilingInfoWrapper(ProfNodeAdditionInfo *add_info) {
    add_infos_[static_cast<size_t>(NodeProfInfoType::kOriginalNode)] = add_info;
  }

  void SetProfNodeAdditionInfo(const NodeProfInfoType prof_info_type, ProfNodeAdditionInfo *const add_info) {
    add_infos_[static_cast<size_t>(prof_info_type)] = add_info;
  }

  void SetBlockDim(uint32_t block_dim) override {
    add_infos_[static_cast<size_t>(NodeProfInfoType::kOriginalNode)]->
        node_basic_info.data.nodeBasicInfo.blockDim = block_dim;
  }

  void SetBlockDim(const uint32_t block_dim, const NodeProfInfoType prof_info_type) override {
    auto prof_info = add_infos_[static_cast<size_t>(prof_info_type)];
    if (prof_info != nullptr) {
      prof_info->node_basic_info.data.nodeBasicInfo.blockDim = block_dim;
    }
  }

  void SetMixLaunchEnable(const bool mix_launch_enable) override {
    add_infos_[static_cast<size_t>(NodeProfInfoType::kOriginalNode)]->mix_launch_enable = mix_launch_enable;
  }

  void SetLaunchTimeStamp(const uint64_t begin_time, const uint64_t end_time,
                          const NodeProfInfoType prof_info_type) override {
    auto prof_info = add_infos_[static_cast<size_t>(prof_info_type)];
    if (prof_info != nullptr) {
      prof_info->api.beginTime = begin_time;
      prof_info->api.endTime = end_time;
    }
  }

  ge::graphStatus FillShapeInfo(const std::vector<std::vector<int64_t>> &input_shapes,
                                const std::vector<std::vector<int64_t>> &output_shapes) override {
    auto &add_info = add_infos_[static_cast<size_t>(NodeProfInfoType::kOriginalNode)];
    GE_ASSERT_EQ(input_shapes.size(), add_info->input_tensor_num);
    GE_ASSERT_EQ(output_shapes.size(), add_info->output_tensor_num);
    uint64_t shape_idx = 0;
    for (auto &tensor_info_wrapper : add_info->tensor_info_wrappers) {
      auto tensor_info = reinterpret_cast<MsprofTensorInfo *>(tensor_info_wrapper.tensor_info.data);
      for (uint64_t i = 0UL; i < tensor_info_wrapper.tensor_num; ++i) {
        const auto &shape = shape_idx < input_shapes.size() ? input_shapes[shape_idx] :
                                                            output_shapes[shape_idx - input_shapes.size()];
        ++shape_idx;
        for (size_t j = 0U; j < MSPROF_GE_TENSOR_DATA_SHAPE_LEN; ++j) {
          if (j < shape.size()) {
            tensor_info->tensorData[i].shape[j] = shape[j];
          } else {
            tensor_info->tensorData[i].shape[j] = 0UL;
          }
        }
      }
      tensor_info_wrapper.filled_by_kernel = true;
    }
    return ge::SUCCESS;
  }

 private:
  std::array<ProfNodeAdditionInfo *, static_cast<size_t>(NodeProfInfoType::kNodeTypeMax)> add_infos_{};
};

class CannProfilerV2 : public BaseExecutorProfiler {
 public:
  static void OnExecuteEvent(int32_t sub_exe_graph_type, CannProfilerV2 *profiler, ExecutorEvent event,
                             const void *node, KernelStatus result);

  explicit CannProfilerV2(const std::shared_ptr<const SubscriberExtendInfo> &extend_info);

  ge::Status DoProf(const Node *node);

  ge::Status RecordLaunchBeginTime(const Node &node);

  Chain *InitOutputChainFromEquivalentDataEdges(
      const bg::ValueHolderPtr &out_shape,
      std::unordered_map<std::string, ge::FastNode *> &kernel_names_2_exe_nodes) const;

  void Init();

  void IncreaseIterationNum() {
      ++iter_num_;
  }

  uint32_t GetIterationNum() const {
    return iter_num_;
  }

  const std::shared_ptr<const SubscriberExtendInfo> &GetExtendInfo() const {
    return extend_info_;
  }

  const std::unordered_map<uint64_t, ProfNodeAdditionInfo> &GetNamesToAddInfos() const {
    return name_hashes_to_node_addition_infos_;
  }

  void SetEngineType(NodeIdentity node_id, const std::string &kernel_type, const DfxExtendInfo *dfx_extend_info);

 private:
  ge::Status InitForCannDevice(const ExecutionData *execution_data);
  void InitBasicInfo(const uint64_t node_id, const DfxExtendInfo *dfx_extend_info, MsprofCompactInfo &basic_info) const;
  void InitLaunchApi(const uint64_t name_hash, const char *kernel_type, MsprofApi &api) const;
  ge::Status RecordNodeBasicInfo(NodeIdentity node_id, uint64_t prof_time, int32_t tid) const;
  ge::Status RecordTensorInfo(const uint64_t prof_time, const int32_t tid, ProfNodeAdditionInfo &addition_info) const;
  ge::Status InitBasicInfoAndTensorInfo(const ExecutionData &execution_data, const ge::ComputeGraphPtr &compute_graph,
      const std::unordered_map<uint64_t, DfxExtendInfo *> &node_name_to_dfx_extend_info);
  ge::Status InitShapeInfo(const ge::NodePtr &node,
                           std::unordered_map<std::string, ge::FastNode *> &kernel_names_2_exe_nodes);
  ge::Status InitOutputShapeInfo(const ge::NodePtr &node, std::vector<Chain *> &output_shapes,
                                 std::unordered_map<std::string, ge::FastNode *> &kernel_names_2_exe_nodes);
  ge::Status InitInputShapeInfo(const ge::NodePtr &node, const uint64_t name_hash, std::vector<Chain *> &input_shapes);
  void InitShapes();
  ge::Status InitProfInfoByGraphNode(const ge::NodePtr &node);
  void InitNodeCMOContextInfo(const ge::OpDescPtr &op_desc);
  void InitNodeContextInfo(const ge::OpDescPtr &op_desc);
  ge::Status InitNodeAtomicContextInfo(const ge::OpDescPtr &op_desc);
  void InitSingleCMOContextInfo(const ge::OpDescPtr &op_desc, const std::vector<uint32_t> &cmo_context_ids,
                                std::string &cmo_context_name, uint32_t task_type, size_t cmo_idx);
  void InitVectorCoreProfInfo(const ge::OpDescPtr &op_desc);
  ge::Status InitShapeAndContextInfo(const ge::ComputeGraphPtr &graph);
  ge::Status InitInfoByExecutionNode(uint64_t name_hash, NodeIdentity node_id, const char *kernel_type,
                                     const ComputeNodeInfo *compute_node_info, const DfxExtendInfo *dfx_extend_info);
  void FormatNodeIdMap(const std::unordered_map<uint64_t, std::vector<NodeIdentity>> &name_hash_to_node_ids);
  void ReadDfxExtendInfo(const ge::ExecuteGraph *const execute_graph,
                         std::unordered_map<uint64_t, DfxExtendInfo *> &node_name_to_dfx_extend_info);
  ge::Status DoProfByNodeId(NodeIdentity report_node, uint64_t prof_time, uint32_t tid);

  ge::Status ReportMixLaunchKernel(const NodeIdentity node_id);

 private:
  std::shared_ptr<const SubscriberExtendInfo> extend_info_{nullptr};
  bool is_device_prof_inited_{false};
  uint32_t iter_num_{1UL};
  std::unordered_map<uint64_t, ProfNodeAdditionInfo> name_hashes_to_node_addition_infos_{};
  std::vector<std::array<ProfNodeAdditionInfo *, static_cast<size_t>(NodeProfInfoType::kNodeTypeMax)>>
      node_addition_infos_{};
  std::unordered_map<uint64_t, std::vector<Chain *>> name_hashes_to_output_shapes_{};
  std::unordered_map<uint64_t, std::vector<Chain *>> name_hashes_to_input_shapes_{};
  std::unordered_map<uint64_t, NodeIdentity> name_hash_to_node_id_{};
  std::vector<KernelRegistry::ProfilingInfoFiller> exe_node_id_to_profiling_filler_{};
  std::vector<NodeIdentity> node_id_to_profiler_node_id_{};
  std::vector<CannProfilingInfoWrapper> exe_node_id_to_profiling_wrappers_{};
  std::vector<NodeIdentity> cur_launch_node_ids_{};
  std::set<NodeIdentity> davinci_model_node_ids_{};
  std::unordered_map<uint64_t, uint64_t> node_hash_to_atomic_node_hash_{};
};
}  // namespace gert
#endif  // AIR_CXX_RUNTIME_V2_SUBSCRIBER_PROFILER_CANN_PROFILER_V2_H_
