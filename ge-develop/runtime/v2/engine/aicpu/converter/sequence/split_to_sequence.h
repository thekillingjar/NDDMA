/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_SPLIT_TO_SEQUENCE_H
#define AIR_CXX_RUNTIME_V2_SPLIT_TO_SEQUENCE_H

#include "graph/node.h"
#include "exe_graph/lowering/value_holder.h"
#include "register/node_converter_registry.h"
#include "exe_graph/lowering/lowering_global_data.h"

namespace gert {
namespace bg {
struct StoreSequenceArg {
  ValueHolderPtr session_id;
  ValueHolderPtr contanier_id;
  ValueHolderPtr inner_tensor_addrs;
  ValueHolderPtr inner_tensor_shapes;
  ValueHolderPtr input_num;
  ValueHolderPtr keep_dims_holder;
  ValueHolderPtr axis_holder;
  ValueHolderPtr input_x_shape;
  ValueHolderPtr output_tensor;
};
} // bg
LowerResult LoweringSplitToSeuqnece(const ge::NodePtr &node, const LowerInput &lower_input);
} // gert
#endif // AIR_CXX_RUNTIME_V2_SPLIT_TO_SEQUENCE_H
