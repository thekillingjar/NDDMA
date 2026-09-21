/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DVPP_CONVERT_DVPP_GRAPH_BUILDER_BG_GENERATE_SQE_AND_LAUNCH_TASK_H_
#define DVPP_CONVERT_DVPP_GRAPH_BUILDER_BG_GENERATE_SQE_AND_LAUNCH_TASK_H_
#include "exe_graph/lowering/value_holder.h"

namespace gert {
namespace bg {
ValueHolderPtr GenerateSqeAndLaunchTask(
    const std::vector<ValueHolderPtr> &input_shapes,
    const std::vector<ValueHolderPtr> &output_shapes,
    const std::vector<ValueHolderPtr> &addrs_info,
    const ValueHolderPtr stream);
} // namespace bg
} // namespace gert
#endif // DVPP_CONVERT_DVPP_GRAPH_BUILDER_BG_GENERATE_SQE_AND_LAUNCH_TASK_H_
