/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bg_variable.h"
#include "common/checker.h"
#include "exe_graph/lowering/lowering_global_data.h"
#include "exe_graph/lowering/frame_selector.h"
#include "bg_rt_session.h"
#include "kernel/common_kernel_impl/variable.h"

namespace gert {
namespace bg {
std::vector<DevMemValueHolderPtr> Variable(const std::string &var_id, LoweringGlobalData &global_data,
                                           const int64_t stream_id) {
  auto builder = [&global_data, &var_id, &stream_id]() -> std::vector<bg::ValueHolderPtr> {
    return bg::FrameSelector::OnInitRoot([&global_data, &var_id, &stream_id]() -> std::vector<bg::ValueHolderPtr> {
      auto rt_session = bg::HolderOnInit(bg::GetRtSession(global_data));
      auto id = bg::ValueHolder::CreateConst(var_id.data(), var_id.size() + 1U, true);
      auto gert_allocator = global_data.GetOrCreateAllocator({kOnDeviceHbm, AllocatorUsage::kAllocNodeWorkspace});
      auto split_var_outputs =
          bg::DevMemValueHolder::CreateDataOutput("SplitVariable", {rt_session, id, gert_allocator},
                                                  static_cast<size_t>(kernel::SplitVariableOutputs::kNum), stream_id);
      std::vector<bg::ValueHolderPtr> out_holder(split_var_outputs.cbegin(), split_var_outputs.cend());
      return out_holder;
    });
  };
  std::vector<DevMemValueHolderPtr> split_var_holder;
  for (const auto &holder : global_data.GetOrCreateUniqueValueHolder(var_id, builder)) {
    split_var_holder.emplace_back(std::dynamic_pointer_cast<DevMemValueHolder>(holder));
  }
  return split_var_holder;
}

DevMemValueHolderPtr GetVariableAddr(const std::string &var_id, LoweringGlobalData &global_data,
                                     const int64_t stream_id) {
  auto var_shape_and_addr = Variable(var_id, global_data, stream_id);
  return var_shape_and_addr.size() == 2U ? var_shape_and_addr[1] : nullptr;
}

ValueHolderPtr GetVariableShape(const std::string &var_id, LoweringGlobalData &global_data, const int64_t stream_id) {
  auto var_shape_and_addr = Variable(var_id, global_data, stream_id);
  return var_shape_and_addr.size() == 2U ? var_shape_and_addr[0] : nullptr;
}
}  // namespace bg
}  // namespace gert
