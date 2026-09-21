/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_MY_OP_IdentityN_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_MY_OP_IdentityN_H_
#include "esb_funcs.h"
#include <stdint.h>
#include "graph/types.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  EsCTensorHolder **y;
  int64_t y_num;
} EsIdentityNOutput;

EsIdentityNOutput MyEsIdentityN(EsCTensorHolder **x, int64_t x_num);
#ifdef __cplusplus
}
#endif
#endif
