/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef H05B2224D_B926_4FC0_A936_97B52B8A98DB
#define H05B2224D_B926_4FC0_A936_97B52B8A98DB

#include "easy_graph/infra/default.h"

EG_NS_BEGIN

namespace details {
template<typename T>
struct Interface {
  virtual ~Interface() {}
};
}  // namespace details

#define INTERFACE(Intf) struct Intf : ::EG_NS::details::Interface<Intf>

#define ABSTRACT(...) virtual __VA_ARGS__ = 0

#define OVERRIDE(...) virtual __VA_ARGS__ override

#define EXTENDS(...) , ##__VA_ARGS__
#define IMPLEMENTS(...) EXTENDS(__VA_ARGS__)

EG_NS_END

#endif
