/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_CMO_GENERATE_CMO_TYPE_MANAGER_H_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_CMO_GENERATE_CMO_TYPE_MANAGER_H_

#include "generate_cmo_type_base.h"
#include "common/fe_inner_error_codes.h"

namespace fe {
using GenerateCMOTypeBasePtr = std::shared_ptr<GenerateCMOTypeBase>;

class GenerateCMOTypeManager {
public:
  GenerateCMOTypeManager() = default;
  ~GenerateCMOTypeManager() = default;

  Status Initialize();

  Status Finalize();

  void GenerateType(const ge::NodePtr &node, const StreamCtrlMap &stream_ctrls,
                    std::unordered_map<ge::NodePtr, ge::NodePtr> &prefetch_cache_map,
                    std::map<uint32_t, std::map<int64_t, ge::NodePtr>> &stream_node_map) const;

private:
  Status Register(CmoType type);

  bool init_flag_;
  std::map<CmoType, GenerateCMOTypeBasePtr> cmo_generate_map_{};
};
} // namespace fe
#endif
