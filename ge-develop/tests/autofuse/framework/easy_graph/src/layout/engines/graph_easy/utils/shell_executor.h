/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HB141A993_B14A_4A1D_A1DD_353D33AE77A2
#define HB141A993_B14A_4A1D_A1DD_353D33AE77A2

#include <string>
#include "easy_graph/eg.h"
#include "easy_graph/infra/status.h"

EG_NS_BEGIN

struct ShellExecutor {
  static Status execute(const std::string &script);
};

EG_NS_END

#endif
