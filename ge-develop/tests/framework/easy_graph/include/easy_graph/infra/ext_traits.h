/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HF25A2E8D_A775_44BF_88D1_166DFF56186A
#define HF25A2E8D_A775_44BF_88D1_166DFF56186A

#include <type_traits>
#include "easy_graph/eg.h"

EG_NS_BEGIN

template<typename T, typename... TS>
using all_same_traits = typename std::enable_if<std::conjunction<std::is_same<T, TS>...>::value>::type;

template<typename T, typename... TS>
using all_same_but_none_traits = typename std::enable_if<
    std::disjunction<std::bool_constant<not(sizeof...(TS))>, std::conjunction<std::is_same<T, TS>...>>::value>::type;

#define ALL_SAME_CONCEPT(TS, T) all_same_traits<T, TS...> * = nullptr
#define ALL_SAME_BUT_NONE_CONCEPT(TS, T) ::EG_NS::all_same_but_none_traits<T, TS...> * = nullptr

#define SUBGRAPH_CONCEPT(GS, G) ALL_SAME_BUT_NONE_CONCEPT(GS, G)

EG_NS_END

#endif
