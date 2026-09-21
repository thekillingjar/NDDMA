/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_INC_EXE_GRAPH_RUNTIME_SHAPE_INFERENCE_H_
#define METADEF_CXX_INC_EXE_GRAPH_RUNTIME_SHAPE_INFERENCE_H_

#include "graph/op_desc.h"
namespace gert {
extern ge::graphStatus InferDataTypeOnCompile(const ge::OpDescPtr &op_desc);
extern ge::graphStatus InferShapeRangeOnCompile(const ge::Operator &op, const ge::OpDescPtr &op_desc);
extern ge::graphStatus InferShapeOnCompile(const ge::Operator &op, const ge::OpDescPtr &op_desc);
extern ge::graphStatus InferFormatOnCompile(const ge::Operator &op, const ge::OpDescPtr &op_desc);
extern bool IsInferFormatV2Registered(const ge::OpDescPtr &op_desc);
}  // namespace gert
#endif  // METADEF_CXX_INC_EXE_GRAPH_RUNTIME_SHAPE_INFERENCE_H_
