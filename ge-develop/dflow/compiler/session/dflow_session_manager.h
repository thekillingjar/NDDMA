/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DFLOW_SESSION_SESSION_MANAGER_H_
#define DFLOW_SESSION_SESSION_MANAGER_H_

#include <map>
#include <memory>
#include <string>
#include "ge/ge_api_types.h"
#include "dflow_session_impl.h"
namespace ge {
namespace dflow {
using SessionPtr = std::shared_ptr<ge::DFlowSessionImpl>;

class DFlowSessionManager {
 public:
  DFlowSessionManager() = default;

  ~DFlowSessionManager() = default;

  /// @ingroup ge_session
  /// @brief initialize session manager
  /// @return Status result of function
  void Initialize();

  /// @ingroup ge_session
  /// @brief finalize session manager
  /// @return Status result of function
  void Finalize();

  /// @ingroup ge_session
  /// @brief create session
  /// @param [in] options session config options
  /// @param [out] session_id session id
  /// @return Status result of function
  SessionPtr CreateSession(const std::map<std::string, std::string> &options, uint64_t &session_id);

  /// @ingroup ge_session
  /// @brief destroy the session with specific session id
  /// @param [in] session_id session id
  /// @return Status result of function
  Status DestroySession(uint64_t session_id);

  /// @ingroup ge_session
  /// @brief get session with specific session id
  /// @param [in] session_id session id
  /// @return SessionPtr session
  SessionPtr GetSession(uint64_t session_id);

 private:
  std::map<uint64_t, SessionPtr> session_manager_map_;
  std::mutex mutex_;
  bool init_flag_ = false;
};
} // namespace dflow
}  // namespace ge

#endif  // DFLOW_SESSION_SESSION_MANAGER_H_
