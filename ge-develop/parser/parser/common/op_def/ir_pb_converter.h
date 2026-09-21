/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DOMI_COMMON_OP_DEF_IR_PB_CONVERTER_H
#define DOMI_COMMON_OP_DEF_IR_PB_CONVERTER_H

#include "framework/common/fmk_error_codes.h"
#include "parser/common/op_def/operator.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/op_desc_utils_ex.h"
#include "proto/om.pb.h"

namespace ge {
domi::Status ConvertToOpDesc(const ParserOperator &op, ge::OpDescPtr &op_def);

domi::Status ConvertFromOpDesc(const ge::OpDescPtr op_def, ParserOperator &op);
}  // namespace ge

#endif  // DOMI_COMMON_OP_DEF_IR_PB_CONVERTER_H
