/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_1249A0BF64B44060BAF658B265D6D356
#define INC_1249A0BF64B44060BAF658B265D6D356

#include "stdint.h"
#include "vector"
#include "ge_running_env/fake_ns.h"
#include "graph/ge_tensor.h"

FAKE_NS_BEGIN

class GeTensorDesc;

struct TensorDescs {
  TensorDescs(int size);
  std::vector<GeTensorDesc>& Value(){
    return tensor_descs_;
  }

 private:
  std::vector<GeTensorDesc> tensor_descs_;
};

FAKE_NS_END

#endif
