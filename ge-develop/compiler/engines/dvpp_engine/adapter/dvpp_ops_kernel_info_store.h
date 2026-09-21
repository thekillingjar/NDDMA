/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DVPP_ENGINE_ADAPTER_DVPP_OPS_KERNEL_INFO_STORE_H_
#define DVPP_ENGINE_ADAPTER_DVPP_OPS_KERNEL_INFO_STORE_H_

#include "common/dvpp_info_store.h"

namespace dvpp {
class DvppOpsKernelInfoStore : public ge::OpsKernelInfoStore {
 public:
  /**
   * @brief Construct
   */
  DvppOpsKernelInfoStore() = default;

  /**
   * @brief Destructor
   */
  virtual ~DvppOpsKernelInfoStore() = default;

  // Copy constructor prohibited
  DvppOpsKernelInfoStore(
      const DvppOpsKernelInfoStore& dvpp_ops_kernel_info_store) = delete;

  // Move constructor prohibited
  DvppOpsKernelInfoStore(
      const DvppOpsKernelInfoStore&& dvpp_ops_kernel_info_store) = delete;

  // Copy assignment prohibited
  DvppOpsKernelInfoStore& operator=(
      const DvppOpsKernelInfoStore& dvpp_ops_kernel_info_store) = delete;

  // Move assignment prohibited
  DvppOpsKernelInfoStore& operator=(
      DvppOpsKernelInfoStore&& dvpp_ops_kernel_info_store) = delete;

  /**
   * @brief initialize dvpp ops kernel info store
   * @param options initial options
   * @return status whether success
   */
  ge::Status Initialize(
      const std::map<std::string, std::string>& options) override;

  /**
   * @brief close dvpp ops kernel info store
   * @return status whether success
   */
  ge::Status Finalize() override;

  /**
   * @brief Return all ops info of dvpp
   * @param infos reference of a map, return ops name and detail
   */
  void GetAllOpsKernelInfo(
      std::map<std::string, ge::OpInfo>& infos) const override;

  /**
   * @brief check if dvpp support the op
   * @param op_desc_ptr op desc pointer
   * @param unsupported_reason if not support, need to write reason
   * @return whether support the op
   */
  bool CheckSupported(const ge::OpDescPtr& op_desc_ptr,
                      std::string& unsupported_reason) const override;

  /**
   * @brief check if dvpp support the op
   * @param node node pointer
   * @param unsupported_reason if not support, need to write reason
   * @param flag dvpp do not use it
   * @return whether support the op
   */
  bool CheckSupported(const ge::NodePtr& node,
                      std::string& unsupported_reason,
                      ge::CheckSupportFlag& flag) const override;

  /**
   * @brief current is used in stars, convert op_desc to vpu_config or cmdlist
   * @param task task info
   * @return status whether success
   */
  ge::Status LoadTask(ge::GETaskInfo& task) override;

  /**
   * @brief current is used in stars, free vpu_config or cmdlist memory
   * @param task task info
   * @return status whether success
   */
  ge::Status UnloadTask(ge::GETaskInfo& task) override;

 private:
  std::shared_ptr<DvppInfoStore> dvpp_info_store_{nullptr};
}; // class DvppOpsKernelInfoStore
} // namespace dvpp

#endif // DVPP_ENGINE_ADAPTER_DVPP_OPS_KERNEL_INFO_STORE_H_