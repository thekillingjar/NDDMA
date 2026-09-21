/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/device_op_impl_registry.h"
#include "register/op_def_factory.h"
namespace optiling {
class DeviceOpImplRegisterImpl {};
DeviceOpImplRegister::DeviceOpImplRegister(const char *opType) {
  ops::OpDefFactory::OpTilingSinkRegister(opType);
}
DeviceOpImplRegister::~DeviceOpImplRegister() = default;
DeviceOpImplRegister::DeviceOpImplRegister(DeviceOpImplRegister &&other) noexcept = default;
DeviceOpImplRegister::DeviceOpImplRegister(const DeviceOpImplRegister &other) {
  (void)other;
}
DeviceOpImplRegister &DeviceOpImplRegister::Tiling(SinkTilingFunc func) {
  (void)func;
  return *this;
}
}