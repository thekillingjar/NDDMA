/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_BG_EXT_INFO_H_
#define AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_BG_EXT_INFO_H_
#include "exe_graph/lowering/value_holder.h"
#include "exe_graph/lowering/lowering_global_data.h"

namespace gert {
namespace bg {
struct BlockInfo {
  ValueHolderPtr is_block_op;
  ValueHolderPtr rt_event;
  ValueHolderPtr event_id;
  ValueHolderPtr async_timeout;
};

struct ExtShapeInfo {
  std::vector<ValueHolderPtr> input_shapes;
  std::vector<ValueHolderPtr> output_shapes;
};

ValueHolderPtr BuildExtInfo(const ge::NodePtr node, const std::string &ext_info,
                            const std::vector<ValueHolderPtr> &addrs, const ValueHolderPtr &session_id,
                            const BlockInfo &block_info);
ValueHolderPtr UpdateExtInfo(const ge::OpDescPtr &op_desc, const ExtShapeInfo &ext_shape_info,
                             const ValueHolderPtr &ext_handle, const ValueHolderPtr &stream);
} // bg
} // gert
#endif // AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_BG_EXT_INFO_H_
