/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bg_rt_session.h"
#include "common/checker.h"
#include "exe_graph/lowering/frame_selector.h"
#include "framework/runtime/gert_const_types.h"

namespace gert {
namespace bg {
ValueHolderPtr GetRtSession(LoweringGlobalData &global_data) {
  return global_data.GetUniqueValueHolder(GetConstDataTypeStr(ConstDataType::kRtSession));
}

ValueHolderPtr GetSessionId(LoweringGlobalData &global_data) {
  auto outputs = FrameSelector::OnInitRoot([&global_data]() -> std::vector<ValueHolderPtr> {
    auto rt_session = bg::HolderOnInit(bg::GetRtSession(global_data));
    return {ValueHolder::CreateSingleDataOutput("GetSessionId", {rt_session})};
  });
  return outputs[0];
}
}  // namespace bg
}  // namespace gert
