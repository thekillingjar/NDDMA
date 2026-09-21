/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_TILING_RESOURCE_MANAGER_H
#define AIR_CXX_TILING_RESOURCE_MANAGER_H

#include <unordered_map>
#include "host_resource_manager.h"
#include "graph/host_resource/host_resource.h"
#include "register/op_tiling_info.h"
#include "ge/ge_api_error_codes.h"
#include "graph/op_desc.h"
namespace ge {
using TilingId = int64_t;
enum class TilingResourceType : uint8_t { kAiCore = 0x0U, kAtomic = 0xFFU };

class TilingResource : public HostResource {
 public:
  explicit TilingResource(const std::shared_ptr<optiling::utils::OpRunInfo> &op_run_info) : op_run_info_(op_run_info) {}
  ~TilingResource() override = default;
  const std::shared_ptr<optiling::utils::OpRunInfo> GetOpRunInfo() const {
    return op_run_info_;
  }

 private:
  const std::shared_ptr<optiling::utils::OpRunInfo> op_run_info_;
};

class TilingResourceManager : public HostResourceManager {
 public:
  const HostResource *GetResource(const std::shared_ptr<AttrHolder> &attr_holder, int64_t type) const override;
  Status AddResource(const std::shared_ptr<AttrHolder> &attr_holder, int64_t type,
                     std::shared_ptr<HostResource> &host_resource) override;
  Status TakeoverResources(const std::shared_ptr<AttrHolder> &attr_holder) override;
  Status RecoverResource(const TilingId tiling_id, const std::shared_ptr<HostResource> &host_resource);

  const std::unordered_map<const optiling::utils::OpRunInfo *, std::vector<TilingId>> &GetResourceToKeys() const {
    return resource_to_keys_;
  }

  void UseOpIdAsTilingId() {
    use_op_id_as_key_ = true;
  }

  TilingResourceManager() = default;
  ~TilingResourceManager() override = default;

  TilingResourceManager(const TilingResourceManager &tiling_resource_manager) = delete;
  TilingResourceManager(const TilingResourceManager &&tiling_resource_manager) = delete;
  TilingResourceManager &operator=(const TilingResourceManager &tiling_resource_manager) = delete;
  TilingResourceManager &operator=(TilingResourceManager &&tiling_resource_manager) = delete;

 private:
  TilingId GenTilingId(const TilingResourceType tiling_type) const;
  static TilingId GenTilingId(const OpDesc &op_desc, const TilingResourceType tiling_type);

  Status TakeoverResources(const OpDescPtr &op_desc);

  std::unordered_map<TilingId, std::shared_ptr<HostResource>> key_to_resources_;
  std::unordered_map<const optiling::utils::OpRunInfo *, std::vector<TilingId>> resource_to_keys_;
  bool use_op_id_as_key_{false};
};
}  // namespace ge

#endif  // AIR_CXX_TILING_RESOURCE_MANAGER_H
