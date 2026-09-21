/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_TESTS_ST_STUBS_UTILS_SYNCHRONIZER_H_
#define AIR_CXX_TESTS_ST_STUBS_UTILS_SYNCHRONIZER_H_
#include <mutex>
namespace ge {
class Synchronizer {
 public:
  Synchronizer() {
    mu_.lock();
  }
  void OnDone() {
    mu_.unlock();
  }
  bool WaitFor(int64_t seconds) {
    return mu_.try_lock_for(std::chrono::seconds(seconds));
  }
 private:
  std::timed_mutex mu_;
};
}
#endif //AIR_CXX_TESTS_ST_STUBS_UTILS_SYNCHRONIZER_H_
