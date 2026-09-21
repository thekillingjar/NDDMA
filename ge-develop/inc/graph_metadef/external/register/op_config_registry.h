/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_EXTERNAL_REGISTER_OP_CONFIG_REGISTRY_H_
#define INC_EXTERNAL_REGISTER_OP_CONFIG_REGISTRY_H_

#include "register/op_def.h"

namespace ops {
using OpAICoreConfigFunc = OpAICoreConfig (*)();

class OpConfigRegistry {
public:
  OpConfigRegistry();
  void RegisterOpAICoreConfig(const char* name, const char* socVersion, OpAICoreConfigFunc func);
};

std::map<ge::AscendString, OpAICoreConfigFunc> GetOpAllAICoreConfig(const char* name);
}

#define REGISTER_OP_AICORE_CONFIG(opType, socVersion, opFunc) REGISTER_OP_AICORE_CONFIG_UNIQ_HELPER(opType, socVersion, (opFunc), __COUNTER__)

#define REGISTER_OP_AICORE_CONFIG_UNIQ_HELPER(opType, socVersion, opFunc, counter) REGISTER_OP_AICORE_CONFIG_UNIQ(opType, socVersion, (opFunc), counter)

#define REGISTER_OP_AICORE_CONFIG_UNIQ(opType, socVersion, opFunc, counter)                 \
  static uint32_t g_##opType##Op##socVersion##ConfigRegistryInterfV1##counter = [](void) {  \
    ops::OpConfigRegistry configRegistry;                                                   \
    configRegistry.RegisterOpAICoreConfig(#opType, #socVersion, opFunc);                    \
    return 0;                                                                               \
  }()

#endif  // INC_EXTERNAL_REGISTER_OP_CONFIG_REGISTRY_H_