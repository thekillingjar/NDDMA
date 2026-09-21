/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_EXECUTOR_HYBRID_COMMON_BIN_CACHE_NODE_COMPILE_CACHE_MODULE_H_
#define AIR_CXX_EXECUTOR_HYBRID_COMMON_BIN_CACHE_NODE_COMPILE_CACHE_MODULE_H_
#include <cstdint>
#include "graph/op_desc.h"
#include "ge/ge_api_types.h"
#include "graph/node.h"
#include "proto/task.pb.h"
#include "graph/cache_policy/cache_policy.h"
#include "register/op_tiling_info.h"

namespace ge {
namespace {
  const std::string COMPILE_INFO_JSON = "compile_info_json";
  const std::string COMPILE_INFO_KEY = "compile_info_key";
  const std::string ATOMIC_COMPILE_INFO_JSON = "_atomic_compile_info_json";
  const std::string ATOMIC_COMPILE_INFO_KEY = "_atomic_compile_info_key";
  constexpr char_t const *kAttrOpParamSize = "op_para_size";
  constexpr char_t const *kAttrAtomicOpParamSize = "atomic_op_para_size";
}
enum class KernelLaunchBinType : std::uint32_t
{
  kStubFunc = 0, // after register with stub name
  kWithHandle, // register with handle
  kBinTypeEnd
};

class NodeCompileCacheItem {
 public:
  NodeCompileCacheItem() = default;
  ~NodeCompileCacheItem() = default;

  static Status Build(const KernelLaunchBinType bin_type, const NodePtr &node, void *handle,
                      NodeCompileCacheItem &item);

  uint64_t GetCacheItemId() const;
  void SetCacheItemId(const uint64_t cache_item_id);
  /**
   * @brief Get the Bin Handle object
   *        if bin type is kWithHandle, return handle
   *        if bin type is kStubFunc, return stub_func
   * @return void*  stub_func or handle
   */
  void *GetBinHandle() const;
  KernelLaunchBinType GetBinType() const;
  const optiling::OpCompileInfo *GetCompileInfo() const;
  const optiling::OpCompileInfo *GetAtomicCompileInfo() const;
  bool IsSupportDynamic() const;
  int64_t GetMaxTilingSize() const;
  int64_t GetAtomicMaxTilingSize() const;

 private:
  uint64_t cache_item_id_ = UINT64_MAX;
  KernelLaunchBinType bin_type_ = KernelLaunchBinType::kBinTypeEnd;
  void *handle_ = nullptr; // content follow bin_type. Its stubfunc when bin_type is kStubFunc
  optiling::OpCompileInfo op_compile_info_;
  optiling::OpCompileInfo atomic_op_compile_info_;
  int64_t max_tiling_size_ = -1;
  int64_t atomic_max_tiling_size_ = -1;
  bool is_dynamic_ = false;
};

class NodeCompileCacheModule {
 public:
  NodeCompileCacheModule();
  void Initialize();
  void Finalize();
  NodeCompileCacheItem *FindCompileCache(const NodePtr &node);
  NodeCompileCacheItem *AddCompileCache(const NodePtr &node, NodeCompileCacheItem &item);

 private:
  Status GetCompileCacheDescFromOp(const NodePtr &node, std::shared_ptr<CompileCacheDesc> &cache_desc,
                                   const bool need_range) const;
  Status GetOpAttrMem(OpDesc &op_desc, CompileCacheDesc &cache_desc) const;
  Status CopyAttrToMem(const std::map<std::string, AnyValue> &all_attributes, std::unique_ptr<uint8_t[]> &attr_mem,
                       const std::set<string> &ordered_origin_attr_name, const size_t attr_size) const;
  Status GetAttrTotalSize(const std::map<std::string, AnyValue> &all_attributes,
    const std::set<string> &ordered_origin_attr_name, size_t &attr_size) const;

  Status CopyAttrValues(const AnyValue &attr_value, uint8_t *base, const size_t max_size, size_t &offset) const;
  size_t GetAttrSize(const AnyValue &attr_value) const;
  Status GetFusionOpCacheDesc(const NodePtr &node, CompileCacheDesc &cache_desc) const;
  Status GetInputConstTensor(const NodePtr &node, CompileCacheDesc &cache_desc) const;
  void InsertCompileCacheDesc(const NodePtr &node, std::shared_ptr<CompileCacheDesc> &cache_desc);
  std::shared_ptr<CompileCacheDesc> GetCompileCacheDesc(const NodePtr &node);
  void UpdateTensorInfos(const NodePtr &node, CompileCacheDesc &cache_desc) const;

 private:
  std::unique_ptr<CachePolicy> ccp_;
  std::mutex ids_to_cci_mu_;
  std::unordered_map<CacheItemId, NodeCompileCacheItem> ids_to_cci_;
  std::mutex node_to_cache_desc_map_mu_;
  std::unordered_map<uintptr_t, std::shared_ptr<CompileCacheDesc>> node_to_cache_desc_map_;
};
}  // namespace ge

#endif // AIR_CXX_EXECUTOR_HYBRID_COMMON_BIN_CACHE_NODE_COMPILE_CACHE_MODULE_H_
