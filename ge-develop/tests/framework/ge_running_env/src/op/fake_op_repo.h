/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DBF6CE7CD4AC4A83BA4ED4B372FC66E4
#define DBF6CE7CD4AC4A83BA4ED4B372FC66E4

#include "ge_running_env/fake_ns.h"
#include "graph/operator_factory.h"

FAKE_NS_BEGIN

struct FakeOpRepo {
  static void Reset();
  static void Regist(const std::string &operator_type, const OpCreator);
  static void Regist(const std::string &operator_type, const InferShapeFunc);
};

FAKE_NS_END
#endif
