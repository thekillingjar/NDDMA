/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_AICPU_CUST_KERNEL_INFO_H_
#define AICPU_AICPU_CUST_KERNEL_INFO_H_

#include "common/aicpu_ops_kernel_info_store/kernel_info.h"

namespace aicpu {
using KernelInfoPtr = std::shared_ptr<KernelInfo>;

class AicpuCustKernelInfo : public KernelInfo {
 public:
  /**
   * Singleton access method, Thread safe
   * @return return unique instance in global scope
   */
  static KernelInfoPtr Instance();

  virtual ~AicpuCustKernelInfo() = default;

  /**
   * Constructor
   */
  AicpuCustKernelInfo() = default;

  bool GetCustJsonFile(const std::string &path);

  bool GetBuiltInCustJsonFile(const std::string &path);
  
  bool ReadCustOpInfoFromJsonFile(const std::string &path);

 protected:
  /**
   * Read json file in specified path(based on source file's current path) e.g.
   * ops_data/aicpu_kernel.json
   * @return whether read file successfully
   */
  bool ReadOpInfoFromJsonFile() override;

 private:

  // Copy prohibited
  AicpuCustKernelInfo(const AicpuCustKernelInfo& aicpu_cust_kernel_info) =
      delete;

  // Move prohibited
  AicpuCustKernelInfo(const AicpuCustKernelInfo&& aicpu_cust_kernel_info) =
      delete;

  // Copy prohibited
  AicpuCustKernelInfo& operator=(
      const AicpuCustKernelInfo& aicpu_cust_kernel_info) = delete;

  // Move prohibited
  AicpuCustKernelInfo& operator=(AicpuCustKernelInfo&& aicpu_cust_kernel_info) =
      delete;

 private:
  // singleton instance
  static KernelInfoPtr instance_;
  bool ReadCustJsonFile(const std::string &user_name, const std::string &cust_ops_file_path);
  bool ReadBuiltInCustJsonFile(const std::string &user_name, const std::string &cust_ops_file_path);
  bool RecordBuiltInCustConfigFile(const std::string &builtin_custom_config_dir) const;
  bool IsAicpuJsonFile(const std::string &filename) const;
  bool LoadJsonFile(const std::string &file_path, nlohmann::json &out_json) const;
  bool SaveJsonFile(const std::string &file_path, const nlohmann::json &content) const;
};
}  // namespace aicpu
#endif  // AICPU_AICPU_CUST_KERNEL_INFO_H_
