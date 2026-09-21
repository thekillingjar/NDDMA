/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PARSER_MESSAGE2OPERATOR_H
#define PARSER_MESSAGE2OPERATOR_H

#include "ge/ge_api_error_codes.h"
#include "graph/operator.h"
#include "google/protobuf/message.h"

namespace ge {
class Message2Operator {
 public:
  static Status ParseOperatorAttrs(const google::protobuf::Message *message, int depth, ge::Operator &ops);

 private:
  static Status ParseField(const google::protobuf::Reflection *reflection, const google::protobuf::Message *message,
                           const google::protobuf::FieldDescriptor *field, int depth, ge::Operator &ops);

  static Status ParseRepeatedField(const google::protobuf::Reflection *reflection,
                                   const google::protobuf::Message *message,
                                   const google::protobuf::FieldDescriptor *field, ge::Operator &ops);
};
}  // namespace ge
#endif  // PARSER_MESSAGE2OPERATOR_H
