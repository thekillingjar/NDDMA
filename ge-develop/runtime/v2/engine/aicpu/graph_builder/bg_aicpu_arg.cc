/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bg_aicpu_arg.h"
#include "graph/debug/ge_attr_define.h"
#include "proto/task.pb.h"
#include "graph_builder/bg_memory.h"
#include "aicpu_engine_struct.h"
#include "graph_builder/bg_infer_shape.h"
#include "common/checker.h"
#include "engine/aicpu/kernel/aicpu_resource_manager.h"
#include "bg_ext_info.h"
#include "graph_builder/bg_infer_shape_range.h"
#include "exe_graph/lowering/frame_selector.h"
#include "engine/aicpu/converter/aicpu_callback.h"
#include "graph_builder/bg_rt_session.h"
#include "engine/aicpu/kernel/block_op_utils.h"
#include "framework/common/ge_types.h"
#include "engine/node_converter_utils.h"

namespace gert {
namespace bg {
namespace {
constexpr size_t kArgSize = 2U;
constexpr size_t kArgHandleSize = 1U;

bool NeedDeviceExt(const ge::NodePtr node) {
  int32_t unknown_shape_type_val = 0;
  (void) ge::AttrUtils::GetInt(node->GetOpDescBarePtr(), ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, unknown_shape_type_val);
  return (IsAicpuOutputUnknownShape(node)) &&
    (unknown_shape_type_val == static_cast<int32_t>(ge::DEPEND_SHAPE_RANGE)) &&
    (node->GetOpDescBarePtr()->GetOpKernelLibName() != ge::kEngineNameHostCpu);
}

BlockInfo InitBlockInfo(const ge::OpDescPtr &op_desc, const bool &is_memcpy_task) {
  rtEvent_t rt_event = nullptr;
  uint32_t event_id = 0;
  bool is_blocking_aicpu_op = false;
  uint32_t async_timeout = 0xFFFFFFFF;
  if (!is_memcpy_task) {
    (void) ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_IS_BLOCKING_OP, is_blocking_aicpu_op);
    if (op_desc->HasAttr(ge::ATTR_NAME_BLOCKING_OP_TIMEOUT)) {
      (void)ge::AttrUtils::GetInt(op_desc, ge::ATTR_NAME_BLOCKING_OP_TIMEOUT, async_timeout);
    }
  }

  BlockInfo block_info;
  block_info.rt_event = ValueHolder::CreateConst(&rt_event, sizeof(rt_event));
  block_info.event_id = ValueHolder::CreateConst(&event_id, sizeof(event_id));
  block_info.is_block_op = ValueHolder::CreateConst(&is_blocking_aicpu_op, sizeof(is_blocking_aicpu_op));
  block_info.async_timeout = ValueHolder::CreateConst(&async_timeout, sizeof(async_timeout));

  if (is_blocking_aicpu_op) {
    bool is_device_support = false;
    (void)CheckDeviceSupportBlockingAicpuOpProcess(is_device_support);
    if (is_device_support) {
      auto event_holder = bg::ValueHolder::CreateDataOutput("CreateEvent", {}, 2U); // rt_event & event_id
      bg::ValueHolder::CreateVoidGuarder("DestroyEvent", event_holder[0], {});
      block_info.rt_event = event_holder[0];
      block_info.event_id = event_holder[1];
    }
  }
  return block_info;
}

std::vector<ValueHolderPtr> GetBaseCCArgs(const ge::NodePtr node, const domi::KernelDef &kernel_def, size_t io_num) {
  std::vector<ValueHolderPtr> inputs;
  // io_num, node_name
  const auto &node_name = node->GetName();
  const bool need_device_ext = NeedDeviceExt(node);
  inputs.emplace_back(ValueHolder::CreateConst(&io_num, sizeof(io_num)));
  inputs.emplace_back(ValueHolder::CreateConst(node_name.c_str(), node_name.size() + 1, true));
  inputs.emplace_back(ValueHolder::CreateConst(&need_device_ext, sizeof(need_device_ext)));

  // args
  const auto &kernel_arg = kernel_def.args();
  const size_t arg_size = kernel_arg.size();
  if (arg_size < kernel_def.args_size()) {
    GELOGE(ge::PARAM_INVALID, "[Check][KernelDef]Op[%s]arg_size %zu is smaller then size %u in kernel_def",
           node_name.c_str(), arg_size, kernel_def.args_size());
    return {nullptr}; // check valueholder nullptr will return
  }
  inputs.emplace_back(ValueHolder::CreateConst(kernel_arg.c_str(), arg_size, true));
  inputs.emplace_back(ValueHolder::CreateConst(&arg_size, sizeof(arg_size)));

  // ext_info size
  const auto &ext_info = kernel_def.kernel_ext_info();
  const size_t ext_size = ext_info.size();
  if (ext_size < kernel_def.kernel_ext_info_size()) {
    GELOGE(ge::PARAM_INVALID, "[Check][KernelDef]Op[%s]ext_size %zu is smaller then size %u in kernel_def",
           node_name.c_str(), ext_size, kernel_def.kernel_ext_info_size());
    return {nullptr}; // check valueholder nullptr will return
  }
  inputs.emplace_back(ValueHolder::CreateConst(&ext_size, sizeof(ext_size)));
  return inputs;
}

std::vector<ValueHolderPtr> BuildTfAicpuArgImpl(const ge::NodePtr node, const TfArgsInfo &tf_args_info,
                                                const bool &is_memcpy_task) {
  // get anchor on init graph, not output for init node on root graph
  const auto &session_id = bg::HolderOnInit(tf_args_info.session_id);
  const auto &step_id = bg::HolderOnInit(tf_args_info.step_id);
  const auto &kernel_ex_def = tf_args_info.kernel_ex_def;
  ValueHolder::CreateSingleDataOutput("EnsureCreateTfSession", {session_id});

  std::vector<ValueHolderPtr> inputs;
  // io_num, node_name
  const auto &node_name = node->GetName();
  const bool need_device_ext = NeedDeviceExt(node);
  inputs.emplace_back(ValueHolder::CreateConst(&tf_args_info.io_num, sizeof(tf_args_info.io_num)));
  inputs.emplace_back(ValueHolder::CreateConst(node_name.c_str(), node_name.size() + 1, true));
  inputs.emplace_back(ValueHolder::CreateConst(&need_device_ext, sizeof(need_device_ext)));

  // STR_FWK_OP_KERNEL
  const auto kernel_arg = kernel_ex_def.args();
  const size_t arg_size = kernel_arg.size();
  if (arg_size < kernel_ex_def.args_size()) {
    GELOGE(ge::PARAM_INVALID, "[Check][KernelDef]Op[%s]arg_size %zu is smaller then size %u in kernel_def",
           node_name.c_str(), arg_size, kernel_ex_def.args_size());
    return {nullptr, nullptr}; // check valueholder nullptr will return
  }
  inputs.emplace_back(ValueHolder::CreateConst(kernel_arg.data(), arg_size, true));
  inputs.emplace_back(ValueHolder::CreateConst(&arg_size, sizeof(arg_size)));

  // ext_info
  const auto ext_info = kernel_ex_def.kernel_ext_info();
  const size_t ext_info_size = ext_info.size();
  if (ext_info_size < kernel_ex_def.kernel_ext_info_size()) {
    GELOGE(ge::PARAM_INVALID, "[Check][KernelDef]Op[%s]ext_info_size %zu is smaller then size %u in kernel_def",
           node_name.c_str(), ext_info_size, kernel_ex_def.kernel_ext_info_size());
    return {nullptr, nullptr}; // check valueholder nullptr will return
  }
  inputs.emplace_back(ValueHolder::CreateConst(&ext_info_size, sizeof(ext_info_size)));

  BlockInfo block_info = InitBlockInfo(node->GetOpDesc(), is_memcpy_task);
  inputs.emplace_back(block_info.is_block_op);
  inputs.emplace_back(block_info.rt_event);
  inputs.emplace_back(block_info.async_timeout);
  // workspace/task_info
  const auto task_info = kernel_ex_def.task_info();
  const size_t task_size = task_info.size();
  if (task_size < kernel_ex_def.task_info_size()) {
    GELOGE(ge::PARAM_INVALID, "[Check][KernelDef]Op[%s]task_size %zu is smaller then %u in kernel_def",
           node_name.c_str(), task_size, kernel_ex_def.task_info_size());
    return {nullptr, nullptr}; // check valueholder nullptr will return
  }
  inputs.emplace_back(ValueHolder::CreateConst(task_info.data(), task_size, true));
  inputs.emplace_back(ValueHolder::CreateConst(&task_size, sizeof(task_size)));

  // build args handle
  inputs.emplace_back(session_id);
  inputs.emplace_back(step_id);
  auto tf_args = ValueHolder::CreateSingleDataOutput("BuildTfArgs", inputs);

  // build ext_info handle
  auto ext_addrs = ValueHolder::CreateDataOutput("GetExtInfoAddrFromArgs", {tf_args}, 2U); // host addr & device addr
  auto ext_handle = BuildExtInfo(node, ext_info, ext_addrs, session_id, block_info);

  return std::vector<ValueHolderPtr>({tf_args, ext_handle, block_info.rt_event});
}

std::vector<ValueHolderPtr> BuildHostCCAicpuArgImpl(const ge::NodePtr node, const domi::KernelDef &kernel_def,
                                                    const size_t io_num, ValueHolderPtr& session_id) {
  session_id = bg::HolderOnInit(session_id);

  // build args handle & load constant folding lib
  auto inputs = GetBaseCCArgs(node, kernel_def, io_num);
  auto host_args = ValueHolder::CreateSingleDataOutput("BuildHostCCArgs", inputs);
  AicpuResourceManager::GetInstance().LoadConstantFoldingLib();

  // build ext_info handle
  auto ext_info = kernel_def.kernel_ext_info();
  auto ext_addrs = ValueHolder::CreateDataOutput("GetExtInfoAddrFromArgs", {host_args}, 2U); // host addr & device addr
  BlockInfo block_info = InitBlockInfo(node->GetOpDesc(), false);
  auto ext_handle = BuildExtInfo(node, ext_info, ext_addrs, session_id, block_info);

  return std::vector<ValueHolderPtr>({host_args, ext_handle, block_info.rt_event});
}

std::vector<ValueHolderPtr> BuildCCAicpuArgImpl(const ge::NodePtr node, const domi::KernelDef &kernel_def,
                                                const size_t io_num, ValueHolderPtr& session_id,
                                                const bool &is_memcpy_task) {
  session_id = bg::HolderOnInit(session_id);

  // base args
  auto inputs = GetBaseCCArgs(node, kernel_def, io_num);
  BlockInfo block_info = InitBlockInfo(node->GetOpDesc(), is_memcpy_task);
  inputs.emplace_back(block_info.is_block_op);
  inputs.emplace_back(block_info.rt_event);
  inputs.emplace_back(block_info.async_timeout);

  // kernel_name
  const auto &kernel_name = kernel_def.kernel_name();
  const auto kernel_name_size = kernel_name.size() + 1U;
  inputs.emplace_back(ValueHolder::CreateConst(kernel_name.c_str(), kernel_name_size, true));
  inputs.emplace_back(ValueHolder::CreateConst(&kernel_name_size, sizeof(kernel_name_size)));

  // so_name
  const auto &so_name = kernel_def.so_name();
  const auto so_name_size = so_name.size() + 1U;
  inputs.emplace_back(ValueHolder::CreateConst(so_name.c_str(), so_name_size, true));
  inputs.emplace_back(ValueHolder::CreateConst(&so_name_size, sizeof(so_name_size)));

  // build args handle
  auto cc_args = ValueHolder::CreateSingleDataOutput("BuildCCArgs", inputs);

  // build ext_info handle
  auto ext_info = kernel_def.kernel_ext_info();
  auto ext_addrs = ValueHolder::CreateDataOutput("GetExtInfoAddrFromArgs", {cc_args}, 2U); // host addr & device addr
  auto ext_handle = BuildExtInfo(node, ext_info, ext_addrs, session_id, block_info);

  return std::vector<ValueHolderPtr>({cc_args, ext_handle, block_info.rt_event});
}
} // namespace
AicpuArgs BuildTfAicpuArg(const ge::NodePtr node, const TfArgsInfo &tf_args_info, const bool &is_memcpy_task) {
  auto res = FrameSelector::OnInitRoot([&node, &tf_args_info, &is_memcpy_task]() -> std::vector<ValueHolderPtr> {
    return BuildTfAicpuArgImpl(node, tf_args_info, is_memcpy_task);
  });
  if (res.size() >= kArgSize) {
    return {res[0], res[1]};
  } else {
    return {nullptr, nullptr};
  }
}

AicpuArgs BuildHostCCAicpuArg(const ge::NodePtr node, const domi::KernelDef &kernel_def, const size_t io_num,
                              ValueHolderPtr& session_id) {
  auto res = FrameSelector::OnInitRoot([&node, &kernel_def, &io_num, &session_id]() -> std::vector<ValueHolderPtr> {
    return BuildHostCCAicpuArgImpl(node, kernel_def, io_num, session_id);
  });
  if (res.size() >= kArgSize) {
    return {res[0], res[1]};
  } else {
    return {nullptr, nullptr};
  }
}

AicpuArgs BuildCCAicpuArg(const ge::NodePtr node, const domi::KernelDef &kernel_def, const size_t io_num,
                          ValueHolderPtr& session_id, const bool &is_memcpy_task) {
  auto res = FrameSelector::OnInitRoot(
      [&node, &kernel_def, &io_num, &session_id, &is_memcpy_task]() -> std::vector<ValueHolderPtr> {
        return BuildCCAicpuArgImpl(node, kernel_def, io_num, session_id, is_memcpy_task);
      });
  if (res.size() >= kArgSize) {
    return {res[0], res[1]};
  } else {
    return {nullptr, nullptr};
  }
}

ValueHolderPtr CalcBlockDim(const ge::OpDescPtr &op_desc, const std::vector<ValueHolderPtr> &input_shapes) {
  bool is_support_block = false;
  (void) ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_SUPPORT_BLOCKDIM_FLAG, is_support_block);
  if (!IsUnkownShape(op_desc) || !is_support_block || input_shapes.empty()) {
    uint32_t block_dim = 1U;
    auto block_dim_holder = ValueHolder::CreateConst(&block_dim, sizeof(block_dim));
    return block_dim_holder;
  }

  int32_t axis_index = -1;
  (void) ge::AttrUtils::GetInt(op_desc, ge::ATTR_NAME_BLOCKDIM_INDEX, axis_index);
  auto idx_holder = ValueHolder::CreateConst(&axis_index, sizeof(axis_index));
  return ValueHolder::CreateSingleDataOutput("CalcBlockDim", {idx_holder, input_shapes[0]});
}

std::vector<ValueHolderPtr> GetMemAllocShape(const ge::NodePtr &node,
                                             const std::vector<ValueHolderPtr> &input_shapes,
                                             LoweringGlobalData &global_data) {
  if (!IsOutputUnkownShape(node->GetOpDesc())) {
    return NodeConverterUtils::CreateOutputShapes(node->GetOpDesc());
  }
  GELOGI("Op[%s] is unknown output shape, start to lowering infershape.", node->GetName().c_str());
  int32_t unknown_shape_type_val = 0;
  (void) ge::AttrUtils::GetInt(node->GetOpDescBarePtr(), ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, unknown_shape_type_val);
  if (unknown_shape_type_val == static_cast<int32_t>(ge::DEPEND_SHAPE_RANGE)) {
    return InferMaxShape(node, input_shapes, global_data);
  }
  return InferStorageShape(node, input_shapes, global_data);
}

std::vector<ValueHolderPtr> BuildCCArgsFunctionHandleImpl(const ge::NodePtr node) {
  std::vector<ValueHolderPtr> inputs;

  const ge::OpDescPtr op_desc = node->GetOpDesc();
  inputs.emplace_back(ValueHolder::CreateConst(op_desc.get(), sizeof(ge::OpDesc), false));

  const auto &node_type = node->GetType();
  inputs.emplace_back(ValueHolder::CreateConst(node_type.c_str(), node_type.size()+1U, true));

  auto bin_handle = ValueHolder::CreateSingleDataOutput("BuildCCArgsBinHandle", inputs);
  return std::vector<ValueHolderPtr>({bin_handle});
}

RtsArgs BuildCCArgsBinHandle(const ge::NodePtr node) {
  auto res = FrameSelector::OnInitRoot(
    [&node]() -> std::vector<ValueHolderPtr> {
    return BuildCCArgsFunctionHandleImpl(node);
  });
  if (res.size() >= kArgHandleSize) {
    return {res[0]};
  } else {
    return {nullptr};
  }
}

std::vector<ValueHolderPtr> BuildTfArgsFunctionHandleImpl(const ge::NodePtr node) {
  std::vector<ValueHolderPtr> inputs;

  const ge::OpDescPtr op_desc = node->GetOpDesc();
  inputs.emplace_back(ValueHolder::CreateConst(op_desc.get(), sizeof(ge::OpDesc), false));

  const auto &node_type = node->GetType();
  inputs.emplace_back(ValueHolder::CreateConst(node_type.c_str(), node_type.size()+1U, true));

  auto bin_handle = ValueHolder::CreateSingleDataOutput("BuildTfArgsBinHandle", inputs);
  return std::vector<ValueHolderPtr>({bin_handle});
}

RtsArgs BuildTfArgsBinHandle(const ge::NodePtr node) {
  auto res = FrameSelector::OnInitRoot(
    [&node]() -> std::vector<ValueHolderPtr> {
    return BuildTfArgsFunctionHandleImpl(node);
  });
  if (res.size() >= kArgHandleSize) {
    return {res[0]};
  } else {
    return {nullptr};
  }
}

} // namespace bg
} // namespace gert
