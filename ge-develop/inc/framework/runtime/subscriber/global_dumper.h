/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_INC_FRAMEWORK_RUNTIME_EXECUTOR_GLOBAL_DUMPER_H_
#define AIR_CXX_INC_FRAMEWORK_RUNTIME_EXECUTOR_GLOBAL_DUMPER_H_
#include <mutex>

#include "built_in_subscriber_definitions.h"
#include "framework/common/ge_visibility.h"
#include "graph/compute_graph.h"
#include "common/debug/ge_log.h"

namespace ge {
class ExceptionDumper;
}
namespace gert {
struct GlobalDumperSwitchHandler {
  void *arg;
  ge::Status (*on_event)(void *arg, uint64_t event);
};
// for global info for exception_dump and global switch
class VISIBILITY_EXPORT GlobalDumper {
 public:
  GlobalDumper(const GlobalDumper &) = delete;
  GlobalDumper(GlobalDumper &&) = delete;
  GlobalDumper &operator=(const GlobalDumper &) = delete;
  GlobalDumper &operator==(GlobalDumper &&) = delete;
  static GlobalDumper *GetInstance();

  static void OnGlobalDumperSwitch(void *ins, uint64_t enable_flags);

  ge::ExceptionDumper *MutableExceptionDumper();

  void SetEnableFlags(const uint64_t enable_flags) {
    enable_flags_ = enable_flags;
    PublishEvent();
  }

  void RegisterHandler(const void *key, GlobalDumperSwitchHandler handler) {
    if (handler.on_event == nullptr) {
      GELOGW("Register global dumper handler use a nullptr, ignore it");
      return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (!keys_to_handler_.emplace(key, handler).second) {
      return;
    }
    (void)handler.on_event(handler.arg, enable_flags_);
  }

  void UnregisterHandler(const void * const key) {
    const std::lock_guard<std::mutex> lock(mutex_);
    (void)keys_to_handler_.erase(key);
  }

  size_t GetHandleSize() {
    std::lock_guard<std::mutex> lock(mutex_);
    return keys_to_handler_.size();
  }

  // 判断是否有需要通过OnEvent接口做dump的Dump工具被打开
  bool IsEnableSubscribeDump() const {
    return static_cast<bool>(enable_flags_ &
        (BuiltInSubscriberUtil::EnableBit<DumpType>(DumpType::kNeedSubscribe) - 1UL));
  };

  bool IsEnable(DumpType dump_type) const {
    return static_cast<bool>(enable_flags_ & BuiltInSubscriberUtil::EnableBit<DumpType>(dump_type));
  }

  std::set<ge::ExceptionDumper *> &GetInnerExceptionDumpers() {
    std::lock_guard<std::mutex> lk(mutex_);
    return exception_dumpers_;
  }

  // this api is for test
  void ClearInnerExceptionDumpers() {
    std::lock_guard<std::mutex> lk(mutex_);
    exception_dumpers_.clear();
  }

  void RegisterExceptionDumpers(ge::ExceptionDumper *exception_dumper) {
    std::lock_guard<std::mutex> lk(mutex_);
    const auto ret = exception_dumpers_.insert(exception_dumper);
    if (!ret.second) {
      GELOGW("[Dumper] Save exception dumper of davinci model failed.");
    }
  }

  void UnregisterExceptionDumpers(ge::ExceptionDumper *exception_dumper) {
    std::lock_guard<std::mutex> lk(mutex_);
    (void)exception_dumpers_.erase(exception_dumper);
  }

 private:
  GlobalDumper();
  uint64_t enable_flags_{0UL};
  std::mutex mutex_;
  // each davinci model has own exception dumper
  std::set<ge::ExceptionDumper *> exception_dumpers_{};
  std::map<const void *, GlobalDumperSwitchHandler> keys_to_handler_;
  void PublishEvent() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto &key_to_handler : keys_to_handler_) {
      auto &handler = key_to_handler.second;
      (void)handler.on_event(handler.arg, enable_flags_);
    }
  }
};
}  // namespace gert
#endif
