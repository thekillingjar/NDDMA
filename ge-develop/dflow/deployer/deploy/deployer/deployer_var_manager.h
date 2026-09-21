/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_HETEROGENEOUS_DEPLOY_DEPLOYER_DEPLOYER_VAR_MANAGER_H_
#define AIR_RUNTIME_HETEROGENEOUS_DEPLOY_DEPLOYER_DEPLOYER_VAR_MANAGER_H_

#include <map>
#include <future>
#include "ge/ge_api_types.h"
#include "proto/deployer.pb.h"
#include "common/blocking_queue.h"
#include "deploy/execfwk/builtin_executor_client.h"

namespace ge {
class DeployerVarManager {
 public:
  DeployerVarManager() = default;
  ~DeployerVarManager();

  Status Initialize(deployer::VarManagerInfo var_manager_info);

  Status ProcessSharedContent(const deployer::SharedContentDescription &shared_content_desc,
                              const size_t size,
                              const size_t offset,
                              const uint32_t queue_id);

  void SetVarManagerInfo(deployer::VarManagerInfo var_manager_info);

  const deployer::VarManagerInfo &GetVarManagerInfo() const;

  deployer::VarManagerInfo &MutableVarManagerInfo();

  uint8_t *GetVarMemBase();

  Status GetVarMemAddr(const uint64_t &offset,
                       const uint64_t &total_length,
                       void **dev_addr,
                       bool need_malloc = true);

  const std::map<std::string, deployer::SharedContentDescription> &GetSharedContentDescs() const;

  uint64_t GetVarMemSize() const;

  void SetBasePath(const std::string &base_path);

  void SetShareVarMem(bool share_var_mem);

  bool IsShareVarMem() const;

  Status MallocVarMem(const uint64_t var_size, const uint32_t device_id, void **dev_addr);

  void Finalize();

 private:
  rtMbufPtr_t var_mbuf_ = nullptr;
  uint8_t *var_mem_base_ = nullptr;
  uint64_t var_mem_size_ = 0;
  bool share_var_mem_ = true;
  std::map<std::string, deployer::SharedContentDescription> shared_content_descs_;
  deployer::VarManagerInfo var_manager_info_;
  std::vector<rtMbufPtr_t> var_mbuf_vec_;
  std::map<uint64_t, void *> offset_and_var_map_;
  std::string base_path_;
  std::string receiving_node_name_;
  std::unique_ptr<std::ostream> output_stream_;
  std::unique_ptr<BuiltinExecutorClient> executor_client_;
};
}  // namespace ge

#endif  // AIR_RUNTIME_HETEROGENEOUS_DEPLOY_DEPLOYER_DEPLOYER_VAR_MANAGER_H_
