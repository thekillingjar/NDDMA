/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PARSER_COMMON_REGISTER_TBE_H_
#define PARSER_COMMON_REGISTER_TBE_H_

#include "register/op_registry.h"

namespace ge {
class OpRegistrationTbe {
 public:
  static OpRegistrationTbe *Instance();

    bool Finalize(const OpRegistrationData &reg_data, bool is_train = false, bool is_custom_op = false);

 private:
    bool RegisterParser(const OpRegistrationData &reg_data, bool is_custom_op = false) const;
};
}  // namespace ge

#endif  // PARSER_COMMON_REGISTER_TBE_H_
