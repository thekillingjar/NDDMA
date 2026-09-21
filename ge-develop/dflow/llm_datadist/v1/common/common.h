/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_LLM_ENGINE_COMMON_COMMON_H_
#define AIR_RUNTIME_LLM_ENGINE_COMMON_COMMON_H_

#include <vector>
#include <map>
#include "ge/ge_api_types.h"
#include "common/llm_inner_types.h"
namespace llm {
struct CacheEntry {
  uint64_t num_blocks = 0U; // > 0 means is blocks when cache_mem_type is not MIX
  uint32_t batch_size;
  int32_t seq_len_dim_index = -1;
  uint64_t tensor_size;
  uint64_t stride; // batch stride or block size
  int32_t ext_ref_count = 0;
  CachePlacement placement;
  std::vector<std::shared_ptr<void>> cache_addrs;
  std::map<uint64_t, std::pair<uint32_t, uint64_t>> id_to_batch_index_and_size;
  bool is_owned = false;
  bool remote_accessible = true;
  CacheMemType cache_mem_type = CacheMemType::CACHE;
};
}  // namespace llm

#endif  // AIR_RUNTIME_LLM_ENGINE_COMMON_COMMON_H_
