/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef RTS_ENGINE_OPS_KERNEL_STORE_RTS_FFTS_PLUS_OPS_KERNEL_INFO_H
#define RTS_ENGINE_OPS_KERNEL_STORE_RTS_FFTS_PLUS_OPS_KERNEL_INFO_H

#include <map>
#include <string>

#include "common/opskernel/ops_kernel_info_store.h"

namespace cce {
namespace runtime {
using std::map;
using std::string;

class RtsFftsPlusOpsKernelInfoStore : public ge::OpsKernelInfoStore {
 public:
  RtsFftsPlusOpsKernelInfoStore() = default;

  ~RtsFftsPlusOpsKernelInfoStore() override = default;

  /**
   * Initialize related resources of the rts kernel info store
   * @return status whether this operation success
   */
  ge::Status Initialize(const map<string, string> &options) override;

  /**
   * Release related resources of the rts kernel info store
   * @return status whether this operation success
   */
  ge::Status Finalize() override;

  /**
   * Check to see if an operator is fully supported or partially supported.
   * @param opDesc OpDesc information
   * @param reason unsupported reason
   * @return bool value indicate whether the operator is fully supported
   */
  bool CheckSupported(const ge::OpDescPtr &opDesc, std::string &reason) const override;

  /**
   * Returns the full operator information.
   * @param infos reference of a map,
   *        contain operator's name and detailed information
   */
  void GetAllOpsKernelInfo(map<string, ge::OpInfo> &infos) const override;

  /**
   * Create session
   * @param session_options Session Options
   * @return status whether this operation success
   */
  ge::Status CreateSession(const map<string, string> &sessionOptions) override;

  /**
   * Destroy session
   * @param session_options Session Options
   * @return status whether this operation success
   */
  ge::Status DestroySession(const map<string, string> &sessionOptions) override;

  // Copy prohibited
  RtsFftsPlusOpsKernelInfoStore(const RtsFftsPlusOpsKernelInfoStore &opsKernelStore) = delete;

  // Move prohibited
  RtsFftsPlusOpsKernelInfoStore(const RtsFftsPlusOpsKernelInfoStore &&opsKernelStore) = delete;

  // Copy prohibited
  RtsFftsPlusOpsKernelInfoStore &operator=(const RtsFftsPlusOpsKernelInfoStore &opsKernelStore) = delete;

  // Move prohibited
  RtsFftsPlusOpsKernelInfoStore &operator=(RtsFftsPlusOpsKernelInfoStore &&opsKernelStore) = delete;

 private:
  // store op name and OpInfo key-value pair
  map<string, ge::OpInfo> opInfoMap_;
};
}  // namespace runtime
}  // namespace cce

#endif  // RTS_FFTS_PLUS_OPS_KERNEL_INFO_H
