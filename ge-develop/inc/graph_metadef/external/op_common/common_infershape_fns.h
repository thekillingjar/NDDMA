/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file common_infershape_fns.h
 * \brief
 */

#ifndef EXTERNAL_OP_COMMON_INFERSHAPE_FNS_H_
#define EXTERNAL_OP_COMMON_INFERSHAPE_FNS_H_

#include "exe_graph/runtime/shape.h"
#include "exe_graph/runtime/infer_shape_context.h"

namespace opcommon {
ge::graphStatus InferShape4BroadcastOp(gert::InferShapeContext* context);
ge::graphStatus InferShape4ReduceOp(gert::InferShapeContext* context);
ge::graphStatus InferShape4ElewiseOp(gert::InferShapeContext* context);
} // namespace opcommon

#endif // EXTERNAL_OP_COMMON_INFERSHAPE_FNS_H_
