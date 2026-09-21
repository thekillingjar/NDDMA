/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/op_def/ir_pb_converter.h"
#include "parser/common/op_parser_factory.h"
#include "framework/omg/parser/parser_types.h"

#include "parser/tensorflow/tensorflow_identity_parser.h"

using domi::TENSORFLOW;
using ge::parser::IDENTITY;
using ge::parser::READVARIABLEOP;

namespace ge {
REGISTER_OP_PARSER_CREATOR(TENSORFLOW, READVARIABLEOP, TensorFlowIdentityParser);
}  // namespace ge
