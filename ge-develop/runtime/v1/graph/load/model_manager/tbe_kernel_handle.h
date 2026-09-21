/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_LOAD_MODEL_MANAGER_TBE_KERNEL_HANDLE_H
#define GE_GRAPH_LOAD_MODEL_MANAGER_TBE_KERNEL_HANDLE_H

#include <set>
#include <string>
#include <thread>

#include "graph/op_desc.h"
#include "common/tbe_handle_store/tbe_kernel_store.h"
#include "runtime/rt.h"

namespace ge {
bool IsTbeTask(const OpDescPtr &op_desc);
bool IsAllKernelTask(const OpDescPtr &op_desc);
bool IsNeedAtomicCleanTask(const OpDescPtr &op_desc);

class TBEKernelHandle {
 public:
  TBEKernelHandle() = default;
  ~TBEKernelHandle() = default;

  std::string GetBinHandleKey(const OpDesc &op_desc, const std::string &prefix = "",
                              const bool is_atomic_kernel = false) const;

  /// @ingroup ge
  /// @brief Register TBE kernel to RTS for static or dynamic shape
  /// @param [in] op_desc : current op.
  /// @param [in] prefix: kernel name attr name prefix.
  /// @return SUCCESS / other error code.
  Status Register(const OpDescPtr &op_desc, const std::string &prefix = "");

  /// @ingroup ge
  /// @brief Register TBE kernel to RTS for FFTS Task, Ref: InitTbeHandleInAutoMode.
  /// @param [in] op_desc : current op.
  /// @return SUCCESS / other error code.
  Status RegisterAutoThreadHandle(const OpDescPtr &op_desc, const TBEKernelStore &tbe_kernel_store = TBEKernelStore());

  /// @ingroup ge
  /// @brief Register TBE kernel to RTS for statically compiled kernel.
  /// @param [in] op_desc : current op.
  /// @param [in] prefix: kernel name attr name prefix.
  /// @param [in] tbe_kernel_store: store to find kernel.
  /// @param [in] is_atomic_kernel: atomic kernel flag.
  /// @return SUCCESS / other error code.
  Status RegisterStaticHandle(const OpDescPtr &op_desc, const std::string &prefix,
                              const TBEKernelStore &tbe_kernel_store = TBEKernelStore(),
                              const bool is_atomic_kernel = false);

/// @ingroup ge
/// @brief Register TBE kernel to RTS for common Task for statically compiled kernel.
/// @param [in] op_desc : current op.
/// @param [in] prefix: kernel name attr name prefix.
/// @param [in] tbe_kernel_store: store to find kernel.
/// @return SUCCESS / other error code.
  Status RegisterDynamicKernel(const OpDescPtr &op_desc, const std::string &prefix = "",
                               const TBEKernelStore &tbe_kernel_store = TBEKernelStore());

  void SetModelId(const uint32_t model_id) { model_id_ = model_id; }

  void SetSessionId(const uint64_t session_id) { session_id_ = session_id; }

  /// @ingroup ge
  /// @brief Clean all registered kernel.
  /// @return None
  void CleanTbeHandle() noexcept;

  Status GetAddrAndPrefCnt(const OpDescPtr &op_desc, const std::string &kernel_name, const std::string &prefix,
                           std::vector<std::pair<void *, uint32_t>> &addr_and_pref_cnt) const;

 private:
  Status FunctionRegister(const OpDescPtr &op_desc, const std::string &bin_handle_key, const OpKernelBinPtr &tbe_kernel,
                          const std::string &prefix, const uint32_t thread_index = UINT32_MAX);
  Status InitBinaryMagic(const OpDescPtr &op_desc, const uint32_t thread_index, rtDevBinary_t &binary,
                         const std::string &prefix = "") const;
  Status InitMetaData(const OpDescPtr &op_desc, const uint32_t thread_index, void *bin_handle,
                      const std::string &prefix = "") const;
  Status InitKernelName(const OpDescPtr &op_desc, const uint32_t thread_index, std::string &kernel_name,
                        const std::string &prefix = "") const;

  /// @ingroup ge
  /// @brief Init kernel name for thread slice.
  /// @return SUCCESS / other error code.
  Status ThreadKernelName(const OpDescPtr &op_desc, const uint32_t thread_index,
                          std::string &kernel_name) const;

  void StoreTbeHandle(const std::string &handle_key);

  uint32_t model_id_{0U};
  uint64_t session_id_{0UL};
  std::map<std::string, uint32_t> used_tbe_handle_map_;
  std::map<std::string, std::vector<std::pair<void *, uint32_t>>> addr_and_pref_cnt_;
};
} // namespace ge
#endif  // GE_GRAPH_LOAD_MODEL_MANAGER_TBE_KERNEL_HANDLE_H
