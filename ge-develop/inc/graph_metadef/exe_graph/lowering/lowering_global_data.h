/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_LOWERING_LOWERING_GLOBAL_DATA_H_
#define AIR_CXX_RUNTIME_V2_LOWERING_LOWERING_GLOBAL_DATA_H_
#include <map>
#include "proto/task.pb.h"
#include "value_holder.h"
#include "exe_graph/runtime/tensor.h"
#include "exe_graph/runtime/allocator.h"
#include "exe_graph/runtime/execute_graph_types.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "exe_graph/lowering/lowering_opt.h"
#include "common/ge_common/ge_types.h"

namespace gert {
constexpr int64_t kRtMemoryTypeHbm = 0x2;
constexpr int64_t kDefaultMainStreamId = 0;
// todo change to get stream num from model_desc const data
constexpr const ge::char_t *kGlobalDataModelStreamNum = "ModelStreamNum";
class LoweringGlobalData {
 public:
  struct NodeCompileResult {
    const std::vector<domi::TaskDef> &GetTaskDefs() const {
      return task_defs;
    }
    std::vector<domi::TaskDef> task_defs;
  };

  std::vector<bg::ValueHolderPtr> LoweringAndSplitRtStreams(int64_t stream_num);
  bg::ValueHolderPtr GetStreamById(int64_t logic_stream_id) const;
  inline bg::ValueHolderPtr GetStream() const {
    int64_t current_stream_id = kDefaultMainStreamId;
    if ((bg::ValueHolder::GetCurrentFrame() != nullptr) &&
        (bg::ValueHolder::GetCurrentFrame()->GetCurrentComputeNode() != nullptr)) {
      current_stream_id = bg::ValueHolder::GetCurrentFrame()->GetCurrentComputeNode()->GetOpDesc()->GetStreamId();
    }
    return GetStreamById(current_stream_id);
  }

  void SetRtNotifies(const std::vector<bg::ValueHolderPtr> &notify_holders);
  bg::ValueHolderPtr GetNotifyById(int64_t logic_notify_id) const;

  const NodeCompileResult *FindCompiledResult(const ge::NodePtr &node) const;
  LoweringGlobalData &AddCompiledResult(const ge::NodePtr &node, NodeCompileResult compile_result);

  void *GetGraphStaticCompiledModel(const std::string &graph_name) const;
  LoweringGlobalData &AddStaticCompiledGraphModel(const std::string &graph_name, void *const model);

  bg::ValueHolderPtr GetL1Allocator(const AllocatorDesc &desc) const;
  LoweringGlobalData &SetExternalAllocator(bg::ValueHolderPtr &&allocator);
  LoweringGlobalData &SetExternalAllocator(bg::ValueHolderPtr &&allocator, const ExecuteGraphType graph_type);

  bg::ValueHolderPtr GetOrCreateL1Allocator(const AllocatorDesc desc);
  bg::ValueHolderPtr GetOrCreateL2Allocator(int64_t logic_stream_id, const AllocatorDesc desc);
  bg::ValueHolderPtr GetInitL2Allocator(const AllocatorDesc desc) const;
  bg::ValueHolderPtr GetMainL2Allocator(int64_t logic_stream_id, const AllocatorDesc desc) const;
  inline bg::ValueHolderPtr GetOrCreateAllocator(const AllocatorDesc desc) {
    int64_t current_stream_id = kDefaultMainStreamId;
    if ((bg::ValueHolder::GetCurrentFrame() != nullptr) &&
        (bg::ValueHolder::GetCurrentFrame()->GetCurrentComputeNode() != nullptr)) {
      current_stream_id = bg::ValueHolder::GetCurrentFrame()->GetCurrentComputeNode()->GetOpDesc()->GetStreamId();
    }
    return GetOrCreateL2Allocator(current_stream_id, desc);
  }
  bg::ValueHolderPtr GetOrCreateAllL2Allocators();

  bg::ValueHolderPtr GetOrCreateUniqueValueHolder(const std::string &name,
                                                  const std::function<bg::ValueHolderPtr()> &builder);
  std::vector<bg::ValueHolderPtr> GetOrCreateUniqueValueHolder(const std::string &name,
      const std::function<std::vector<bg::ValueHolderPtr>()> &builder);
  bg::ValueHolderPtr GetUniqueValueHolder(const std::string &name) const;
  void SetUniqueValueHolder(const std::string &name, const bg::ValueHolderPtr &holder);
  void SetValueHolders(const string &name, const bg::ValueHolderPtr &holder);
  size_t GetValueHoldersSize(const string &name);

  void SetModelWeightSize(const size_t require_weight_size);
  size_t GetModelWeightSize() const;

  // TODOO: 先合入metadef，air适配结束，删除metadef中未调用的实现
  const OpImplSpaceRegistryV2Ptr GetSpaceRegistryV2(
      OppImplVersionTag opp_impl_version = OppImplVersionTag::kOpp) const {
    if (opp_impl_version >= OppImplVersionTag::kVersionEnd) {
      return nullptr;
    }
    return space_registries_v2_[static_cast<size_t>(opp_impl_version)];
  };
  const OpImplSpaceRegistryV2Array &GetSpaceRegistriesV2() const {
    return space_registries_v2_;
  };
  void SetSpaceRegistriesV2(const OpImplSpaceRegistryV2Array &space_registries) {
    space_registries_v2_ = space_registries;
  }

  const LoweringOption &GetLoweringOption() const;
  void SetLoweringOption(const LoweringOption &lowering_option);

  void SetStaicModelWsSize(const int64_t require_ws_size) {
    static_model_ws_size = require_ws_size;
  }

  int64_t GetStaticModelWsSize() const {
    return static_model_ws_size;
  }

  void SetFixedFeatureMemoryBase(const void * const memory, const size_t size) {
    fixed_feature_mem_[kRtMemoryTypeHbm] = std::make_pair(memory, size);
  }

  const std::pair<const void *, size_t> &GetFixedFeatureMemoryBase() const {
    const auto iter = fixed_feature_mem_.find(kRtMemoryTypeHbm);
    if (iter != fixed_feature_mem_.end()) {
      return iter->second;
    }
    static std::pair<const void *, size_t> dummy_result;
    return dummy_result;
  }

  void SetFixedFeatureMemoryBase(const int64_t type, const void * const memory, const size_t size) {
    fixed_feature_mem_[type] = std::make_pair(memory, size);
  }

  /*
   * 获取图所需fixed feature memory地址和长度
   * 1 地址为nullptr，长度为0：用户设置的结果，比较特殊，表示不需要GE默认申请fixed内存
   * 2 地址为nullptr，长度不为0，表示需要fixed内存，但是用户没有设置。GE要默认申请fixed内存
   * 3 地址不为nullptr，长度不为0，用户设置的结果。
   */
  const std::map<int64_t, std::pair<const void *, size_t>> &GetAllTypeFixedFeatureMemoryBase() const {
    return fixed_feature_mem_;
  }

  void SetFileConstantMem(const std::vector<ge::FileConstantMem> &file_constant_mems) {
    for (const auto &item : file_constant_mems) {
      file_constant_mems_[item.file_name] = item;
    }
  }

  const ge::FileConstantMem *GetFileConstantMem(const std::string &file_name) const {
    const auto iter = file_constant_mems_.find(file_name);
    if (iter != file_constant_mems_.end()) {
      return &iter->second;
    }
    return nullptr;
  }
  const std::map<std::string, ge::FileConstantMem> &GetAllFileConstantMems() const {
    return file_constant_mems_;
  }

  // if user has call aclmdlSetExternalWeightAddress
  bool IsUserSetFileConstantMem() const {
    return !file_constant_mems_.empty();
  }

  bool IsSingleStreamScene() const {
    return is_single_stream_scene_;
  }

  void SetHostResourceCenter(void *host_resource_center_ptr) {
    host_resource_center_ = host_resource_center_ptr;
  }
  void *GetHostResourceCenter() {
    return host_resource_center_;
  }

 private:
  struct HolderByGraphs {
    bg::ValueHolderPtr holders[static_cast<size_t>(ExecuteGraphType::kNum)];
  };
  struct HoldersByGraphs {
    std::vector<bg::ValueHolderPtr> holders[static_cast<size_t>(ExecuteGraphType::kNum)];
  };

  bg::ValueHolderPtr GetOrCreateInitL2Allocator(const AllocatorDesc desc);
  bg::ValueHolderPtr GetExternalAllocator(const bool from_init, const string &key, const AllocatorDesc &desc);
  bool CanUseExternalAllocator(const ExecuteGraphType &graph_type, const TensorPlacement placement) const;
 private:
  std::unordered_map<std::string, NodeCompileResult> node_name_to_compile_result_holders_;
  std::map<std::string, void *> graph_to_static_models_;
  std::unordered_map<std::string, std::vector<bg::ValueHolderPtr>> unique_name_to_value_holders_;
  HoldersByGraphs streams_;
  HoldersByGraphs notifies_;
  HolderByGraphs external_allocators_;
  // todo need delete and change to const_data after const_data is ready
  int64_t model_weight_size_;
  int64_t static_model_ws_size;
  OpImplSpaceRegistryV2Array space_registries_v2_;
  LoweringOption lowering_option_;
  // addr为nullptr，但size不为0，表示用户没有设置fixed内存，需要GE默认申请fixed内存
  std::map<int64_t, std::pair<const void *, size_t>> fixed_feature_mem_;
  bool is_single_stream_scene_{true};
  void *host_resource_center_{nullptr};
  // user set file constant device memory, key is file name
  std::map<std::string, ge::FileConstantMem> file_constant_mems_;
};
}  // namespace gert
#endif  // AIR_CXX_RUNTIME_V2_LOWERING_LOWERING_GLOBAL_DATA_H_
