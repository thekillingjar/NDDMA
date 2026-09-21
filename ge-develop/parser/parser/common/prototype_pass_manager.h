/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PARSER_PROTOTYPE_PASS_MANAGER_H
#define PARSER_PROTOTYPE_PASS_MANAGER_H

#include "register/prototype_pass_registry.h"

namespace ge {
class ProtoTypePassManager {
 public:
  static ProtoTypePassManager &Instance();

  Status Run(google::protobuf::Message *message, const domi::FrameworkType &fmk_type) const;

  ~ProtoTypePassManager() = default;

 private:
  ProtoTypePassManager() = default;
};
}  // namespace ge

#endif  // PARSER_PROTOTYPE_PASS_MANAGER_H
