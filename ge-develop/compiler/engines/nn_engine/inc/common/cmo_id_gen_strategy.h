/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_GRAPH_COMPILER_ENGINES_NNENG_INC_COMMON_CMO_ID_GEN_STRATEGY_H_
#define AIR_COMPILER_GRAPH_COMPILER_ENGINES_NNENG_INC_COMMON_CMO_ID_GEN_STRATEGY_H_

#include <mutex>
#include "graph/node.h"
#include "graph_optimizer/graph_optimize_register_error_codes.h"

namespace fe {
class CMOIdGenStrategy {
 public:
  CMOIdGenStrategy();
  ~CMOIdGenStrategy();

  CMOIdGenStrategy(const CMOIdGenStrategy &) = delete;
  CMOIdGenStrategy &operator=(const CMOIdGenStrategy &) = delete;

  static CMOIdGenStrategy& Instance();
  Status Finalize();
  uint16_t GenerateCMOId(const ge::Node &node);
  void UpdateReuseMap(const int64_t &topo_id, const uint16_t &cmo_id);

 private:

  uint32_t GetAtomicId() const;

  std::mutex cmo_id_mutex_;
  std::unordered_map<uint16_t, int64_t> reuse_cmo_id_map_;
};
}  // namespace fe
#endif  // AIR_COMPILER_GRAPH_COMPILER_ENGINES_NNENG_INC_COMMON_CMO_ID_GEN_STRATEGY_H_
