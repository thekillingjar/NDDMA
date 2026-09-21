/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_BUILD_MEMORY_MEM_INPLACE_H_
#define GE_GRAPH_BUILD_MEMORY_MEM_INPLACE_H_

#include <memory>
#include "graph/build/memory/block_mem_assigner.h"
#include "graph/compute_graph.h"

namespace ge {
/**
 * 处理图中所有的配置了inplace功能的算子
 * @param mem_assist_info 内存辅助信息，包含了图中所有算子的内存信息
 * @return 成功返回SUCCESS，否则返回FAILED
 */
Status ProcessInplace(MemAssistInfo &mem_assist_info);
}  // namespace ge
#endif  // GE_GRAPH_BUILD_MEMORY_MEM_INPLACE_H_
