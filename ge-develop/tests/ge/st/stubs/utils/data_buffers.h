/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_A1978929F76A40E88093DE26127A4E25
#define INC_A1978929F76A40E88093DE26127A4E25

#include "stdint.h"
#include "vector"
#include "ge_running_env/fake_ns.h"

FAKE_NS_BEGIN

struct DataBuffer;

struct DataBuffers {
  DataBuffers(int size, bool is_huge_buffer = false);

  inline std::vector<DataBuffer>& Value(){
    return buffers_;
  }

 private:
  std::vector<DataBuffer> buffers_;
  uint8_t buffer_addr[10240];
  // just for st
  uint8_t huge_buffer_addr[202400*3];
};

FAKE_NS_END

#endif
