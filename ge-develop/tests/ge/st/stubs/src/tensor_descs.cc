/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "utils/tensor_descs.h"
#include "graph/utils/graph_utils.h"
#include "graph/debug/ge_attr_define.h"

USING_FAKE_NS;

TensorDescs::TensorDescs(int size) {
  for (int i = 0; i < size; i++) {
    GeTensorDesc tensor_desc_0(GeShape(), FORMAT_NCHW, DT_INT32);
    AttrUtils::SetInt(tensor_desc_0, ATTR_NAME_PLACEMENT, 1);
    tensor_descs_.emplace_back(tensor_desc_0);
  }
}

