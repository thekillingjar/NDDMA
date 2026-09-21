/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_KERNEL_LAUNCH_KERNEL_AICPU_BIN_HANDLER_H_
#define AIR_CXX_RUNTIME_V2_KERNEL_LAUNCH_KERNEL_AICPU_BIN_HANDLER_H_

#include <memory>
#include <string>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include "ge/ge_api_error_codes.h"
#include "runtime/rts/rts_kernel.h"
#include "graph/op_kernel_bin.h"

namespace gert {
class OpJsonBinHandler {
public:
  OpJsonBinHandler() = default;
  virtual ~OpJsonBinHandler() {
    std::unique_lock lock(bin_mutex_);
    bin_handle_ = nullptr;
  }

  virtual ge::graphStatus LoadBinary(const std::string &json_path);

  inline rtBinHandle GetBinHandle() const {
    std::shared_lock lock(bin_mutex_);
    return bin_handle_;
  }

  static bool IsSupportBinHandle();
private:
  mutable std::shared_mutex bin_mutex_{};
  rtBinHandle bin_handle_ = nullptr;

  static bool is_support_;
  static std::once_flag init_flag_;
};

class OpDataBinHandler {
public:
  OpDataBinHandler() = default;
  virtual ~OpDataBinHandler() {
    std::unique_lock lock(bin_mutex_);
    bin_handle_ = nullptr;
  }

  ge::graphStatus LoadBinary(const std::string &so_name, const ge::OpKernelBinPtr &kernel_bin);
  inline rtBinHandle GetBinHandle() const {
    std::shared_lock lock(bin_mutex_);
    return bin_handle_;
  }

private:
  mutable std::shared_mutex bin_mutex_{};
  rtBinHandle bin_handle_ = nullptr;
};

class TfJsonBinHandler : public OpJsonBinHandler {
public:
  static TfJsonBinHandler &Instance();
};

class AicpuJsonBinHandler : public OpJsonBinHandler {
public:
  static AicpuJsonBinHandler &Instance();
};

using OpDataBinHandlerPtr = std::shared_ptr<OpDataBinHandler>;

class CustBinHandlerManager {
public:
  static CustBinHandlerManager &Instance();
  ge::graphStatus LoadAndGetBinHandle(const std::string &so_name, const ge::OpKernelBinPtr &kernel_bin,
                                      rtBinHandle &handle);
  ge::graphStatus GetBinHandle(const std::string &so_name, rtBinHandle &handle);

private:
  CustBinHandlerManager() = default;
  ~CustBinHandlerManager() = default;

  CustBinHandlerManager(const CustBinHandlerManager &) = delete;
  CustBinHandlerManager &operator=(const CustBinHandlerManager &) = delete;
  CustBinHandlerManager &operator=(CustBinHandlerManager &&) = delete;
  CustBinHandlerManager(CustBinHandlerManager &&) = delete;

  ge::OpKernelBinPtr GetKernelBin(const std::string &so_name);
  bool GetCustAicpuBinFromFile(const std::string &so_name, ge::OpKernelBinPtr &kernel_bin);
  std::string GetCustSoFolderPath() const;

  mutable std::recursive_mutex mutex_{};
  // {ctx, {so_name, bin_handler}}
  std::unordered_map<uintptr_t, std::unordered_map<std::string, OpDataBinHandlerPtr>> bin_manager_ = {};
  std::unordered_map<std::string, ge::OpKernelBinPtr> kernel_manager_ = {};
};
} // namespace gert

#endif // AIR_CXX_RUNTIME_V2_KERNEL_LAUNCH_KERNEL_AICPU_BIN_HANDLER_H_