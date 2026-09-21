/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef REGISTER_REGISTER_UTILS_H
#define REGISTER_REGISTER_UTILS_H

#include <google/protobuf/message.h>
#include "register/register_types.h"
#include "register/register_error_codes.h"
#include "graph_metadef/register/register.h"
#include "graph/operator.h"

namespace domi {
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY Status OperatorAutoMapping(const Message *op_src, ge::Operator &op);
}  // namespace domi
#endif  // REGISTER_REGISTER_UTILS_H
