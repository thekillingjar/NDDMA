/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_REGISTER_OP_CONFIG_REGISTRY_IMPL_H_
#define INC_REGISTER_OP_CONFIG_REGISTRY_IMPL_H_

#include <map>
#include "register/op_config_registry.h"

namespace ops {
class OpConfigRegistryImpl {
public:
  static OpConfigRegistryImpl &GetInstance();
  void AddAICoreConfig(const char* name, const char* socVersion, OpAICoreConfigFunc func);
  std::map<ge::AscendString, OpAICoreConfigFunc> GetOpAllAICoreConfig(const char* name);

private:
  std::map<ge::AscendString, std::map<ge::AscendString, OpAICoreConfigFunc>> funcData_;
};
}

#endif  // INC_REGISTER_OP_CONFIG_REGISTRY_IMPL_H_