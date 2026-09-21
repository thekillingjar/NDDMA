/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef D52AA06185E34BBFB714FFBCDAB0D53A
#define D52AA06185E34BBFB714FFBCDAB0D53A

#include "ge_graph_dsl/ge.h"
#include <exception>
#include <string>

GE_NS_BEGIN

struct AssertError : std::exception {
  AssertError(const char *file, int line, const std::string &info);

 private:
  const char *what() const noexcept override;

 private:
  std::string info;
};

GE_NS_END

#endif
