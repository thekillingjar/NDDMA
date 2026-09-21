/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_BG_VARIABLE_H_
#define AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_BG_VARIABLE_H_
#include "common/checker.h"
#include "exe_graph/lowering/dev_mem_value_holder.h"
#include "exe_graph/lowering/lowering_global_data.h"

namespace gert {
namespace bg {
std::vector<DevMemValueHolderPtr> Variable(const std::string &var_id, LoweringGlobalData &global_data,
                                           const int64_t stream_id);

DevMemValueHolderPtr GetVariableAddr(const std::string &var_id, LoweringGlobalData &global_data,
                                     const int64_t stream_id);

ValueHolderPtr GetVariableShape(const std::string &var_id, LoweringGlobalData &global_data,
                                const int64_t stream_id);
}  // namespace bg
}  // namespace gert
#endif  // AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_BG_VARIABLE_H_
