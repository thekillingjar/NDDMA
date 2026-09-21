/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_REGISTER_OP_EXT_CALC_PARAM_REGISTRY_H_
#define INC_REGISTER_OP_EXT_CALC_PARAM_REGISTRY_H_

#include <string>
#include <functional>
#include <vector>
#include "graph/node.h"
#include "external/ge_common/ge_api_types.h"

namespace fe {
using OpExtCalcParamFunc = ge::Status (*)(const ge::Node &node);
class OpExtCalcParamRegistry {
  public:
    OpExtCalcParamRegistry() {};
    ~OpExtCalcParamRegistry() {};
    static OpExtCalcParamRegistry &GetInstance();
    OpExtCalcParamFunc FindRegisterFunc(const std::string &op_type) const;
    void Register(const std::string &op_type, OpExtCalcParamFunc const func);

  private:
    std::unordered_map<std::string, OpExtCalcParamFunc> names_to_register_func_;
  };
class OpExtGenCalcParamRegister {
  public:
    OpExtGenCalcParamRegister(const char *op_type, OpExtCalcParamFunc func) noexcept;
};
} // namespace fe

#ifdef __GNUC__
#define ATTRIBUTE_USED __attribute__((used))
#else
#define ATTRIBUTE_USED
#endif

#define REGISTER_NODE_EXT_CALC_PARAM_COUNTER2(type, func, counter)                  \
  static const fe::OpExtGenCalcParamRegister g_reg_op_ext_gentask_##counter ATTRIBUTE_USED =  \
      fe::OpExtGenCalcParamRegister(type, func)
#define REGISTER_NODE_EXT_CALC_PARAM_COUNTER(type, func, counter)                    \
  REGISTER_NODE_EXT_CALC_PARAM_COUNTER2(type, func, counter)
#define REGISTER_NODE_EXT_CALC_PARAM(type, func)                                \
  REGISTER_NODE_EXT_CALC_PARAM_COUNTER(type, func, __COUNTER__)
#endif // INC_REGISTER_OP_EXT_CALC_PARAM_REGISTRY_H_
