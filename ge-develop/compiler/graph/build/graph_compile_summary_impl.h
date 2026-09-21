/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_BUILD_GRAPH_COMPILE_SUMMARY_IMPL_H
#define GE_GRAPH_BUILD_GRAPH_COMPILE_SUMMARY_IMPL_H

#include <vector>
#include <memory>
#include "common/model/ge_model.h"
#include "ge/ge_api_types.h"
#include "common/model/ge_root_model.h"
#include "ge/ge_graph_compile_summary.h"
#include "common/memory/feature_memory_impl.h"

namespace ge {
enum class ModelIORole : int32_t {
  KInput = 0,
  KOutput = 1,
  KEnd = 2
};

class CompiledGraphSummary::Builder {
 public:
  Builder() = default;
  ~Builder() = default;
  static CompiledGraphSummaryPtr Build(const GeModelPtr &ge_model, const GeRootModelPtr &ge_root_model);
};

class CompiledGraphSummary::SummaryData {
 public:
  SummaryData() = default;
  ~SummaryData() = default;
  bool IsStatic() const { return is_static_; }
  bool IsFeatureMemoryBaseRefreshable() const { return is_feature_mem_refreshable_; }
  size_t GetConstMemorySize() const { return const_mem_size_; }
  size_t GetFeatureMemorySize() const { return feature_mem_size_; }
  size_t GetFixedFeatureMemorySize() const { return fixed_feature_mem_size_; }
  std::vector<FeatureMemoryPtr> GetAllFeatureMemoryTypeSize() const;
  size_t GetRefreshableFeatureMemorySize() const { return refreshable_feature_mem_size_; }
  size_t GetStreamNum() const { return stream_num_; }
  size_t GetEventNum() const { return event_num_; }
  std::shared_ptr<StreamAllocationSummary> GetStreamAllocationSummary() {
    return stream_allocation_summary_;
  }
  std::vector<ge::Shape> GetOutputShapes() { return netoutput_shapes_; }
  std::vector<ge::DataType> GetOutputDtypes() { return netoutput_dtypes_;}
  std::vector<std::pair<uint32_t, uint32_t>> GetIOIndexesWithSameAddr() const { return io_indexes_with_same_addr_; }
  const std::vector<ExternalWeightDescPtr> &GetExternalWeightPaths() const { return external_weight_paths_; }

 private:
  friend class CompiledGraphSummary::Builder;

  Status SetConstMemorySize(const GeModelPtr &ge_model);
  Status SetFeatureMemorySize(const GeModelPtr &ge_model);
  Status SetFixedFeatureMemorySize(const GeModelPtr &ge_model, const GeRootModelPtr &ge_root_model);
  Status SetRefreshablFeatureMemorySize(const GeModelPtr &ge_model);
  Status SetFeatureMemoryBaseRefreshable(const GeModelPtr &ge_model);
  Status SetStreamNum(const GeModelPtr &ge_model);
  Status SetEventNum(const GeModelPtr &ge_model);
  Status SetOutputTensorInfo(const GeModelPtr &ge_model);
  Status ConstructIoOffsetToRoleToIndex(const ComputeGraphPtr &compute_graph,
    std::map<int64_t, map<int32_t, std::vector<uint32_t>>> &io_offset_to_role_to_index) const;
  Status SetIOIndexesWithSameAddr(const GeModelPtr &ge_model);
  Status SetExternalWeightPaths(const GeModelPtr &ge_model);

  bool is_static_{false};
  bool is_feature_mem_refreshable_{false};
  size_t const_mem_size_{0UL};
  size_t feature_mem_size_{0UL};
  size_t fixed_feature_mem_size_{0UL};
  size_t refreshable_feature_mem_size_{0UL};
  /*
   * fixed_feature_mem_size_也会保存在feature_memory_,
   * 暂时不包含feature_mem_size_/refreshable_feature_mem_size_, 根据需要可扩展
   */
  std::vector<FeatureMemoryPtr> feature_memory_;
  size_t stream_num_{0UL};
  size_t event_num_{0UL};
  std::vector<ge::Shape> netoutput_shapes_;
  std::vector<ge::DataType> netoutput_dtypes_;
  std::vector<std::pair<uint32_t, uint32_t>> io_indexes_with_same_addr_;
  std::vector<ExternalWeightDescPtr> external_weight_paths_;
  std::shared_ptr<StreamAllocationSummary> stream_allocation_summary_{nullptr};
};

class StreamAllocationSummary::StreamAllocationSummaryImpl {
 public:
  const std::map<AscendString, AscendString> &ToStreamGraph() const;
  const std::map<AscendString, std::vector<LogicalStreamAllocationInfo>> &GetAllLogicalStreamInfos() const;
  Status CollectStreamInfos(const GeRootModelPtr &ge_root_model);

 private:
  Status CollectStreamInfosFromKnownGraph(const ComputeGraphPtr graph,
                                          std::vector<LogicalStreamAllocationInfo> &logical_stream_infos);
  Status CollectStreamInfosFromUnKnownGraph(const ComputeGraphPtr graph,
                                            std::vector<LogicalStreamAllocationInfo> &logical_stream_infos);
  Status CollectCustomStreamInfo(const ComputeGraphPtr graph,
                                 std::map<int64_t, LogicalStreamAllocationInfo> &logical_stream_id_to_stream_info) const;

 private:
  std::map<AscendString, AscendString> graph_to_stream_graph_;
  std::map<AscendString, std::vector<LogicalStreamAllocationInfo>> graph_to_logical_stream_infos_;
};

class LogicalStreamAllocationInfo::LogicalStreamAllocationInfoImpl {
 public:
  LogicalStreamAllocationInfoImpl();
  ~LogicalStreamAllocationInfoImpl() = default;
  AscendString ToStringInfo() const;
  int64_t GetLogicalStreamId() const;
  AscendString GetUsrStreamLabel() const;
  bool IsAssignedByStreamPass() const;
  std::vector<int64_t> GetAttachedStreamIds() const;
  size_t GetPhysicalStreamNum() const;
  size_t GetHcclFollowedStreamNum() const;
  const std::vector<GNode> &GetAllNodes() const;
  void SetLogicalStreamId(int64_t logical_stream_id);
  void SetUsrStreamLabel(const std::string &usr_stream_label);
  void SetIsAssignedByUsrStreamPass(bool is_assigned_by_usr_stream_pass);
  void SetAttachedStreamIds(const std::vector<int64_t> &attached_stream_ids);
  void SetPhysicalStreamNum(size_t physical_stream_num);
  void SetHcclFollowedStreamNum(size_t hccl_followed_stream_num);
  void AppendNode(const NodePtr &node);

 private:
  int64_t logical_stream_id_;
  std::string usr_stream_label_;
  bool is_assigned_by_usr_stream_pass_;
  std::vector<int64_t> attached_stream_ids_;
  size_t physical_stream_num_;
  size_t hccl_followed_stream_num_;
  std::vector<GNode> nodes_;
};

}  // namespace ge

#endif  // GE_GRAPH_BUILD_GRAPH_COMPILE_SUMMARY_IMPL_H
