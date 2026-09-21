/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HE7D53295_59F0_46B0_A881_D6A33B1F9C14
#define HE7D53295_59F0_46B0_A881_D6A33B1F9C14

#include <type_traits>
#include "easy_graph/graph/box.h"

EG_NS_BEGIN

namespace detail {
template<typename Anything>
struct BoxWrapper : Anything, Box {
  using Anything::Anything;
};

template<typename Anything>
using BoxedAnything = std::conditional_t<std::is_base_of_v<Box, Anything>, Anything, BoxWrapper<Anything>>;
}  // namespace detail

#define BOX_WRAPPER(Anything) ::EG_NS::detail::BoxedAnything<Anything>
#define BOX_OF(Anything, ...) ::EG_NS::BoxPacking<Anything>(__VA_ARGS__)

EG_NS_END

#endif
