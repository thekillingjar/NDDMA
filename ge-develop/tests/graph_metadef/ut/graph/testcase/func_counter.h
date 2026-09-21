/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_FUNC_COUNTER_H
#define METADEF_CXX_FUNC_COUNTER_H
#include <iostream>
#include <vector>
namespace ge {
struct FuncCounter {
  static size_t construct_times;
  static size_t copy_construct_times;
  static size_t move_construct_times;
  static size_t copy_assign_times;
  static size_t move_assign_times;
  static size_t destruct_times;
  FuncCounter() {
    ++construct_times;
  }
  ~FuncCounter() {
    ++destruct_times;
  }
  FuncCounter(const FuncCounter &) {
    ++copy_construct_times;
  }
  FuncCounter(FuncCounter &&) noexcept {
    ++move_construct_times;
  }
  FuncCounter &operator=(const FuncCounter &) {
    ++copy_assign_times;
    return *this;
  }
  FuncCounter &operator=(FuncCounter &&) {
    ++move_assign_times;
    return *this;
  }
  static void Clear() {
    construct_times = 0;
    copy_construct_times = 0;
    copy_assign_times = 0;
    move_construct_times = 0;
    move_assign_times = 0;
    destruct_times = 0;
  }
  static std::vector<size_t> GetTimes() {
    return {construct_times,      destruct_times,    copy_construct_times,
            move_construct_times, copy_assign_times, move_assign_times};
    ;
  }
  static bool AllTimesZero() {
    if (construct_times != 0) {
      std::cout << "construct_times not 0" << std::endl;
      return false;
    }
    if (destruct_times != 0) {
      std::cout << "destruct_times not 0" << std::endl;
      return false;
    }
    if (copy_construct_times != 0) {
      std::cout << "copy_construct_times not 0" << std::endl;
      return false;
    }
    if (move_construct_times != 0) {
      std::cout << "move_construct_times not 0" << std::endl;
      return false;
    }
    if (copy_assign_times != 0) {
      std::cout << "copy_assign_times not 0" << std::endl;
      return false;
    }
    if (move_assign_times != 0) {
      std::cout << "move_assign_times not 0" << std::endl;
      return false;
    }
    return true;
  }

  static size_t GetClearConstructTimes() {
    auto tmp = construct_times;
    construct_times = 0;
    return tmp;
  }
  static size_t GetClearCopyConstructTimes() {
    auto tmp = copy_construct_times;
    copy_construct_times = 0;
    return tmp;
  }
  static size_t GetClearMoveConstructTimes() {
    auto tmp = move_construct_times;
    move_construct_times = 0;
    return tmp;
  }
  static size_t GetClearCopyAssignTimes() {
    auto tmp = copy_assign_times;
    copy_assign_times = 0;
    return tmp;
  }
  static size_t GetClearMoveAssignTimes() {
    auto tmp = move_assign_times;
    move_assign_times = 0;
    return tmp;
  }
  static size_t GetClearDestructTimes() {
    auto tmp = destruct_times;
    destruct_times = 0;
    return tmp;
  }
};

}

#endif  //METADEF_CXX_FUNC_COUNTER_H
