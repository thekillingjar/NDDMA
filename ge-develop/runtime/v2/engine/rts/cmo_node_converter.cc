/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include "framework/common/ge_types.h"
#include "common/hyper_status.h"
#include "graph/utils/node_utils.h"
#include "graph_builder/converter_checker.h"
#include "engine/node_converter_utils.h"
#include "runtime/rt_stars_define.h"

namespace gert {
constexpr uint32_t kMaxPrefetchLen = 120U * 1024U * 1024U;
constexpr char_t const *kAttrMaxSize = "max_size";
constexpr char_t const *kAttrAddrOffset = "offset";

LowerResult LoweringCmoNode(const ge::NodePtr &node, const LowerInput &lower_input) {
  // op_code
  uint32_t op_code{0U};
  ge::AttrUtils::GetInt(node->GetOpDesc(), "type", op_code);
  auto op_code_holder = bg::ValueHolder::CreateConst(&op_code, sizeof(uint32_t));
  auto cmo_task_info = bg::ValueHolder::CreateSingleDataOutput("InitCmoArgs", {op_code_holder});

  // update inner_len && src
  LOWER_REQUIRE(!lower_input.input_addrs.empty(), "Input addrs is empty for node [%s].", node->GetNamePtr());
  LOWER_REQUIRE(!lower_input.input_shapes.empty(), "Input shapes is empty for node [%s].", node->GetNamePtr());
  ge::DataType dt = node->GetOpDescBarePtr()->GetInputDesc(0U).GetDataType();
  auto dt_holder = bg::ValueHolder::CreateConst(&dt, sizeof(ge::DataType));

  uint32_t max_len{0U};
  (void)ge::AttrUtils::GetInt(node->GetOpDescBarePtr(), kAttrMaxSize, max_len);
  if (max_len == 0U) {
    max_len = kMaxPrefetchLen;
  }
  int64_t offset{0};
  (void)ge::AttrUtils::GetInt(node->GetOpDescBarePtr(), kAttrAddrOffset, offset);
  auto max_len_holder = bg::ValueHolder::CreateConst(&max_len, sizeof(uint32_t));
  auto offset_holder = bg::ValueHolder::CreateConst(&offset, sizeof(int64_t));
  std::vector<bg::ValueHolderPtr> cmo_calc_inputs{
      cmo_task_info, lower_input.input_addrs[0UL], lower_input.input_shapes[0UL], dt_holder, max_len_holder,
      offset_holder};

  auto update_cmo_args = bg::ValueHolder::CreateSingleDataOutput("UpdatePrefetchTaskInfo", cmo_calc_inputs);

  // launch
  std::vector<bg::ValueHolderPtr> launch_inputs{cmo_task_info, lower_input.global_data->GetStream()};
  auto launch_holder = bg::ValueHolder::CreateSingleDataOutput("LaunchCmoTask", launch_inputs);
  (void)bg::ValueHolder::AddDependency(update_cmo_args, launch_holder);
  GELOGD("Lowering cmo node[%s] end, op_code:[%u], max_len:[%u].", node->GetNamePtr(), op_code, max_len);
  return {HyperStatus::Success(), {launch_holder}, {}, {}};
}
REGISTER_NODE_CONVERTER_PLACEMENT("Cmo", kOnDeviceHbm, LoweringCmoNode);
}  // namespace gert
