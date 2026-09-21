/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_EXECUTOR_HYBRID_COMMON_OP_BINARY_CACHE_H_
#define AIR_CXX_EXECUTOR_HYBRID_COMMON_OP_BINARY_CACHE_H_
#include "graph/bin_cache/node_compile_cache_module.h"

namespace ge {
class OpBinaryCache {
 public:
  static OpBinaryCache &GetInstance();

  NodeCompileCacheModule *GetNodeCcm() {return &nccm_;}

 private:
  OpBinaryCache();

 private:
  NodeCompileCacheModule nccm_;
};
} // namespace ge
#endif // AIR_CXX_EXECUTOR_HYBRID_COMMON_OP_BINARY_CACHE_H_
