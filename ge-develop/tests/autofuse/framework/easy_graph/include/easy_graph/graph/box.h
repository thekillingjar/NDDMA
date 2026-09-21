/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef H4AA49861_3311_4114_8687_1C7D04FA43B9
#define H4AA49861_3311_4114_8687_1C7D04FA43B9

#include <memory>
#include "easy_graph/infra/keywords.h"

EG_NS_BEGIN

INTERFACE(Box){};

using BoxPtr = std::shared_ptr<Box>;

template<typename Anything, typename... Args>
BoxPtr BoxPacking(Args &&... args) {
  return std::make_shared<Anything>(std::forward<Args>(args)...);
}

template<typename Anything>
Anything *BoxUnpacking(const BoxPtr &box) {
  return dynamic_cast<Anything *>(box.get());
}

EG_NS_END

#endif
