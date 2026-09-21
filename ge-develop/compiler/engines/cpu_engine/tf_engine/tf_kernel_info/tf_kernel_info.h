/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_TF_KERNEL_INFO_H_
#define AICPU_TF_KERNEL_INFO_H_

#include "common/aicpu_ops_kernel_info_store/kernel_info.h"
#include "proto/fwk_adapter.pb.h"

namespace aicpu {
using KernelInfoPtr = std::shared_ptr<KernelInfo>;

class TfKernelInfo : public KernelInfo {
 public:
  /**
   * Destructor
   */
  virtual ~TfKernelInfo() = default;

  /**
   * @return kernel info object
   */
  static KernelInfoPtr Instance();

  /**
   * init tf kernel info, get ir to tf parser
   * @param options passed by GE, used to config parameter
   * @return status whether this operation successful
   */
  ge::Status Initialize(const std::map<std::string, std::string> &options) override;

  /**
   * For ops in AIcore, Call CompileOp before Generate task in AICPU
   * @param node Node information
   * @return status whether operation successful
   */
  ge::Status CompileOp(ge::NodePtr &node) override;

  // Copy prohibited
  TfKernelInfo(const TfKernelInfo &tf_kernel_info) = delete;

  // Move prohibited
  TfKernelInfo(const TfKernelInfo &&tf_kernel_info) = delete;

  // Copy prohibited
  TfKernelInfo &operator=(const TfKernelInfo &tf_kernel_info) = delete;

  // Move prohibited
  TfKernelInfo &operator=(TfKernelInfo &&tf_kernel_info) = delete;

  void LoadSupportedOps();
  bool IsSupportedOps(const std::string &op) const override;

 protected:
  /**
   * Read json file in specified path(based on source file's current path) e.g. ops_data/tf_kernel.json
   * @return whether read file successfully
   */
  bool ReadOpInfoFromJsonFile() override;

 private:
  /**
   * Constructor
   */
  TfKernelInfo() = default;
private:
  // singleton instance
  static KernelInfoPtr instance_;
  std::unordered_set<std::string> supported_ops_;
};
} // namespace aicpu
#endif // AICPU_TF_KERNEL_INFO_H_
