/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_builder/bg_infer_shape.h"
#include "graph_builder/bg_memory.h"
#include "framework/common/ge_types.h"
#include "common/hyper_status.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/node_utils.h"
#include "graph_builder/converter_checker.h"
#include "graph_builder/bg_tensor.h"
#include "exe_graph/lowering/frame_selector.h"
#include "exe_graph/lowering/lowering_global_data.h"
#include "graph_builder/bg_rt_session.h"

namespace gert {
namespace bg {
namespace {
static std::atomic<std::uint64_t> g_container_id(0U);
} // namespace

bg::ValueHolderPtr GetContainerIdHolder(const LowerInput &lower_input) {
  auto builder = [&lower_input] () -> bg::ValueHolderPtr {
    uint64_t container_id = ++g_container_id;
    auto container_id_holder = bg::ValueHolder::CreateConst(&container_id, sizeof(uint64_t));
    auto session_id = GetSessionId(*lower_input.global_data);
    auto create_session = bg::FrameSelector::OnInitRoot([&session_id]() -> std::vector<bg::ValueHolderPtr> {
      auto init_graph_session_id = bg::HolderOnInit(session_id);
      auto create_session_local = bg::ValueHolder::CreateSingleDataOutput("CreateSession", {init_graph_session_id});
      bg::ValueHolder::CreateVoidGuarder("DestroySession", create_session_local, {});
      return {create_session_local};
    });

    auto clear_container_builder = [&session_id, &container_id_holder] () -> bg::ValueHolderPtr {
      return bg::ValueHolder::CreateVoid<bg::ValueHolder>("ClearContainer", {session_id, container_id_holder});
    };
    bg::FrameSelector::OnMainRootLast(clear_container_builder);
    return container_id_holder;
  };

  return lower_input.global_data->GetOrCreateUniqueValueHolder("container_id", builder);
}
} // namespace bg
} // namespace gert
