/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_LOWERING_STATIC_COMPILED_GRAPH_CONVERTER_H_
#define AIR_CXX_RUNTIME_V2_LOWERING_STATIC_COMPILED_GRAPH_CONVERTER_H_
#include "common/model/ge_root_model.h"
#include "common/ge_visibility.h"
#include "exe_graph/lowering/lowering_global_data.h"
#include "register/node_converter_registry.h"

namespace gert {
VISIBILITY_EXPORT
const LowerResult *LoweringStaticCompiledComputeGraph(const ge::ComputeGraphPtr &graph,
                                                      LoweringGlobalData &global_data);

LowerResult LoweringStaticModel(const ge::ComputeGraphPtr &graph, ge::GeModel *static_model,
                                const std::vector<bg::DevMemValueHolderPtr> &input_addrs,
                                LoweringGlobalData &global_data);

bool IsGraphStaticCompiled(const ge::ComputeGraphPtr &graph);
bool IsNeedLoweringAsStaticCompiledGraph(const ge::ComputeGraphPtr &graph, LoweringGlobalData &global_data);
bool IsStaticCompiledGraphHasTaskToLaunch(ge::GeModel *static_model);
}  // namespace gert
#endif  // AIR_CXX_RUNTIME_V2_LOWERING_STATIC_COMPILED_GRAPH_CONVERTER_H_
