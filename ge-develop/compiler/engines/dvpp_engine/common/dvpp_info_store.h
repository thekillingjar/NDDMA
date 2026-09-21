/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DVPP_ENGINE_COMMON_DVPP_INFO_STORE_H_
#define DVPP_ENGINE_COMMON_DVPP_INFO_STORE_H_

#include "util/dvpp_error_code.h"
#include "common/opskernel/ops_kernel_info_store.h"

namespace dvpp {
class DvppInfoStore {
 public:
  /**
   * @brief Construct
   */
  DvppInfoStore() = default;

  /**
   * @brief Destructor
   */
  virtual ~DvppInfoStore() = default;

  // Copy constructor prohibited
  DvppInfoStore(const DvppInfoStore& dvpp_info_store) = delete;

  // Move constructor prohibited
  DvppInfoStore(const DvppInfoStore&& dvpp_info_store) = delete;

  // Copy assignment prohibited
  DvppInfoStore& operator=(const DvppInfoStore& dvpp_info_store) = delete;

  // Move assignment prohibited
  DvppInfoStore& operator=(DvppInfoStore&& dvpp_info_store) = delete;

  /**
   * @brief Initialize related resources of dvpp kernel info store
   * @param file_name dvpp ops info json file
   * @return status whether success
   */
  DvppErrorCode InitInfoStore(std::string& file_name);

  /**
   * @brief Return all ops info of dvpp
   * @param infos reference of a map, return ops name and detail
   */
  void GetAllOpsKernelInfo(std::map<std::string, ge::OpInfo>& infos) const;

  /**
   * @brief check if dvpp support the op
   * @param node node pointer
   * @param unsupported_reason if not support, need to write reason
   * @return whether support the op
   */
  bool CheckSupported(const ge::NodePtr& node,
                      std::string& unsupported_reason) const;

  /**
   * @brief current is used in stars, convert op_desc to vpu_config or cmdlist
   * @param task task info
   * @return status whether success
   */
  virtual DvppErrorCode LoadTask(ge::GETaskInfo& task) = 0;

  /**
   * @brief current is used in stars, free vpu_config or cmdlist memory
   * @param task task info
   * @return status whether success
   */
  virtual DvppErrorCode UnloadTask(ge::GETaskInfo& task) = 0;

 private:
  void LoadAllOpsKernelInfo();

 private:
  std::map<std::string, ge::OpInfo> all_ops_kernel_info_;
}; // class DvppInfoStore
} // namespace dvpp

#endif // DVPP_ENGINE_COMMON_DVPP_INFO_STORE_H_
