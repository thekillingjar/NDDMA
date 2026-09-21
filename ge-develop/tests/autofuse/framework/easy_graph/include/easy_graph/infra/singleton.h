/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HD18A3E42_2801_4BEB_B365_03633D1D81C4
#define HD18A3E42_2801_4BEB_B365_03633D1D81C4

#include "easy_graph/eg.h"

EG_NS_BEGIN

template<typename T>
struct Singleton {
  static T &GetInstance() {
    static T instance;
    return instance;
  }

  Singleton(const Singleton &) = delete;
  Singleton &operator=(const Singleton &) = delete;

 protected:
  Singleton() {}
};

#define SINGLETON(object) struct object : ::EG_NS::Singleton<object>

EG_NS_END

#endif
