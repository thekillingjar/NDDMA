/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_TEST_STD_STRUCTS_H
#define METADEF_CXX_TEST_STD_STRUCTS_H
#include "proto/ge_ir.pb.h"
#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "graph/utils/attr_utils.h"

namespace ge {
GeTensorDesc StandardTd_5d_1_1_224_224();
void ExpectStandardTdProto_5d_1_1_224_224(const proto::TensorDescriptor &input_td);
}
#endif  //METADEF_CXX_TEST_STD_STRUCTS_H
