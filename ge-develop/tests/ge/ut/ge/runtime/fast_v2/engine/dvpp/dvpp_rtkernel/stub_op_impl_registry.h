/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef STUB_OP_IMPL_REGISTRY_H_
#define STUB_OP_IMPL_REGISTRY_H_
#include <initializer_list>

namespace gert {
class OpImplRegister {
public:
    OpImplRegister(const char *op_type);
    ~OpImplRegister();

    OpImplRegister& InputsDataDependency(std::initializer_list<int32_t> inputs) {return *this;}
};
OpImplRegister::OpImplRegister(const char *op_type) {}
OpImplRegister::~OpImplRegister() {}
}
#define IMPL_OP(op_type) static gert::OpImplRegister op_impl_register_##op_type = gert::OpImplRegister(#op_type)
#endif  // ifndef STUB_OP_IMPL_REGISTRY_H_
