/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rt_ffts_plus_launch_args.h"
#include "register/ffts_node_converter_registry.h"
#include "register/ffts_node_calculater_registry.h"
#include "engine/aicore/fe_rt2_common.h"
#include "runtime/mem.h"
#include "common/math/math_util.h"
#include "exe_graph/runtime/kernel_context.h"
#include "register/kernel_registry_impl.h"
#include "exe_graph/runtime/tiling_data.h"
#include "common/plugin/ge_make_unique_util.h"

namespace gert {
namespace {
ge::graphStatus SetDynamicArgsSize(RtFFTSKernelLaunchArgs::ArgsDesc &args_desc, bool is_input, size_t desc_size,
                                   size_t all_size) {
  desc_size = desc_size * sizeof(DynDesc);
  desc_size = ge::MemSizeAlign(desc_size, static_cast<uint32_t>(sizeof(uint64_t)));
  GELOGD("Calc dynamic %s io with desc_size: %zu and all_size: %zu.", is_input ? "input" : "output", desc_size, all_size);
  if (is_input) {
    FE_ASSERT_TRUE(args_desc.SetSize(RtFFTSKernelLaunchArgs::kDyInputsDescData, desc_size));
    FE_ASSERT_TRUE(args_desc.SetSize(RtFFTSKernelLaunchArgs::kDyInputsHostAddr, all_size));
  } else {
    FE_ASSERT_TRUE(args_desc.SetSize(RtFFTSKernelLaunchArgs::kDyOutputsDescData, desc_size));
    FE_ASSERT_TRUE(args_desc.SetSize(RtFFTSKernelLaunchArgs::kDyOutputsHostAddr, all_size));
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus RedirectLaunchArgs(KernelContext *context) {
  auto args_para = context->GetInputValue<NodeMemPara *>(static_cast<size_t>(0));
  FE_ASSERT_NOTNULL(args_para);
  auto tiling_data = context->GetOutput(static_cast<size_t>(AllocFFTSArgOutputs::kTilingDataBase));
  FE_ASSERT_NOTNULL(tiling_data);
  auto tail_tiling_data = context->GetOutput(static_cast<size_t>(AllocFFTSArgOutputs::kTailTilingDataBase));
  FE_ASSERT_NOTNULL(tail_tiling_data);
  auto atomic_tiling_data = context->GetOutput(static_cast<size_t>(AllocFFTSArgOutputs::kAtomTilingDataBase));
  FE_ASSERT_NOTNULL(atomic_tiling_data);
  auto atomic_tail_tiling_data = context->GetOutput(static_cast<size_t>(AllocFFTSArgOutputs::kAtomTailTilingDataBase));
  FE_ASSERT_NOTNULL(atomic_tail_tiling_data);
  GELOGD("Redirecting launch arguments with host addr: %lx.", args_para->host_addr);
  auto arg = reinterpret_cast<RtFFTSKernelLaunchArgs *>(args_para->host_addr);
  if (arg->RedirectTilingAddr() != ge::GRAPH_SUCCESS) {
    GELOGE(ge::FAILED, "Args redirect tiling addr failed.");
    return ge::GRAPH_FAILED;
  }
  tiling_data->Set(&arg->GetTilingData(), nullptr);
  tail_tiling_data->Set(&arg->GetTailTilingData(), nullptr);
  atomic_tiling_data->Set(&arg->GetAtomTilingData(), nullptr);
  atomic_tail_tiling_data->Set(&arg->GetAtomTailTilingData(), nullptr);
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(RedirectLaunchArgs).RunFunc(RedirectLaunchArgs);
}  // namespace

std::unique_ptr<uint8_t[]> RtFFTSKernelLaunchArgs::ComputeNodeDesc::Create(size_t &total_size) {
  total_size = sizeof(ComputeNodeDesc);
  auto holder = ge::MakeUnique<uint8_t[]>(total_size);
  if (holder == nullptr) {
    return nullptr;
  }
  return holder;
}

ge::graphStatus RtFFTSKernelLaunchArgs::CalcAtomicArgsSize(const RtFFTSKernelLaunchArgs::ComputeNodeDesc &node_desc,
                                                           RtFFTSKernelLaunchArgs::ArgsDesc &args_desc) {
  if (!node_desc.need_atomic) {
    if (!args_desc.SetSize(kAtomTilingData, 0)) {
      return ge::GRAPH_FAILED;
    }
    if (!args_desc.SetSize(kAtomTailTilingData, 0)) {
      return ge::GRAPH_FAILED;
    }
    if (!args_desc.SetSize(kAtomArgsHostAddr, 0)) {
      return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
  }
  size_t args_size = node_desc.workspace_cap + node_desc.output_num;
  size_t tiling_data_size = 0U;
  if (node_desc.max_atom_tiling_data > 0) {
    args_size++;
    if (TilingData::CalcTotalSize(node_desc.max_atom_tiling_data, tiling_data_size) != ge::GRAPH_SUCCESS) {
      return ge::GRAPH_FAILED;
    }
    tiling_data_size = ge::MemSizeAlign(tiling_data_size);
  }

  if (!args_desc.SetSize(kAtomTilingData, tiling_data_size)) {
    return ge::GRAPH_FAILED;
  }
  tiling_data_size = 0;
  if (node_desc.max_atom_tail_tiling_data > 0) {
    args_size++;
    if (TilingData::CalcTotalSize(node_desc.max_atom_tail_tiling_data, tiling_data_size) != ge::GRAPH_SUCCESS) {
      return ge::GRAPH_FAILED;
    }
    tiling_data_size = ge::MemSizeAlign(tiling_data_size);
  }
  if (!args_desc.SetSize(kAtomTailTilingData, tiling_data_size)) {
    return ge::GRAPH_FAILED;
  }
  args_size *= node_desc.thread_num_max;
  args_size *= sizeof(TensorAddress);
  if (!args_desc.SetSize(kAtomArgsHostAddr, args_size)) {
    return ge::GRAPH_FAILED;
  }
  GELOGD("Added atomic tiling size: %zu, args size: %zu.", tiling_data_size, args_size);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus RtFFTSKernelLaunchArgs::CalcDynamicArgsSize(const ge::NodePtr &node,
                                                            RtFFTSKernelLaunchArgs::ArgsDesc &args_desc,
                                                            std::vector<std::vector<int64_t>> &dyn_io_vv, bool is_input,
                                                            DynDescInfo &dyn_desc_v) {
  auto dyn_num = dyn_io_vv.size();
  GELOGD("Calc dynamic %s group num: %zu.", is_input ? "input" : "output", dyn_num);
  size_t all_size = 0;
  size_t desc_size = 0;
  for (size_t i = 0; i < dyn_num; ++i) {
    auto io_num = dyn_io_vv[i].size();
    size_t addr_offset = 1;  // first is ptr_offset
    for (size_t j = 0; j < io_num; ++j) {
      auto index = dyn_io_vv[i][j];
      auto tensor_desc =
          is_input ? node->GetOpDesc()->MutableInputDesc(index) : node->GetOpDesc()->MutableOutputDesc(index);
      FE_ASSERT_NOTNULL(tensor_desc);
      auto dim_num = tensor_desc->GetShape().IsScalar() ? 1 : tensor_desc->GetShape().GetDimNum();
      if (dim_num > MAX_DIM_NUM || dim_num == 0U) {
        dim_num = MAX_DIM_NUM;
      }
      addr_offset++;           // dim && cnt
      addr_offset += dim_num;  // dim value
    }
    addr_offset = addr_offset * sizeof(TensorAddress);

    size_t group_size = addr_offset + io_num * sizeof(TensorAddress);  // ptrs
    desc_size += io_num;
    for (size_t j = 0; j < io_num; ++j) {
      auto index = dyn_io_vv[i][j];
      GELOGD("%s io: %ld is dynamic.", is_input ? "Input" : "Output", index);
      DynDesc dyn_desc;
      dyn_desc.is_group_first = (j == 0);
      dyn_desc.io_index = index;
      dyn_desc.io_group_id = j;
      dyn_desc.ptr_offset = addr_offset;
      dyn_desc.dyn_num = io_num;
      dyn_desc.group_size = group_size;
      dyn_desc.group_offset = all_size;
      if (is_input) {
        dyn_desc_v.dyn_in_desc.emplace_back(dyn_desc);
      } else {
        dyn_desc_v.dyn_out_desc.emplace_back(dyn_desc);
      }
    }
    FE_ASSERT_TRUE(!ge::AddOverflow(all_size, group_size, all_size));
  }
  return SetDynamicArgsSize(args_desc, is_input, desc_size, all_size);
}

ge::graphStatus RtFFTSKernelLaunchArgs::CalcTotalSize(const ge::NodePtr &node,
                                                      RtFFTSKernelLaunchArgs::ComputeNodeDesc &node_desc,
                                                      RtFFTSKernelLaunchArgs::ArgsDesc &args_desc, size_t &total_size,
                                                      DynDescInfo &dyn_desc_v) {
  size_t tiling_data_size = 0U;
  if (node_desc.max_tiling_data > 0) {
    if (TilingData::CalcTotalSize(node_desc.max_tiling_data, tiling_data_size) != ge::GRAPH_SUCCESS) {
      return ge::GRAPH_FAILED;
    }
    tiling_data_size = ge::MemSizeAlign(tiling_data_size);
  }
  if (!args_desc.SetSize(kTilingData, tiling_data_size)) {
    return ge::GRAPH_FAILED;
  }
  size_t tail_tiling_data_size = 0;
  if (node_desc.max_tail_tiling_data > 0) {
    if (TilingData::CalcTotalSize(node_desc.max_tail_tiling_data, tail_tiling_data_size) != ge::GRAPH_SUCCESS) {
      return ge::GRAPH_FAILED;
    }
    tail_tiling_data_size = ge::MemSizeAlign(tail_tiling_data_size);
  }
  if (!args_desc.SetSize(kTailTilingData, tail_tiling_data_size)) {
    return ge::GRAPH_FAILED;
  }

  std::vector<std::vector<int64_t>> dyn_in_vv;
  (void)ge::AttrUtils::GetListListInt(node->GetOpDesc(), kDynamicInputsIndexes, dyn_in_vv);
  std::vector<std::vector<int64_t>> dyn_out_vv;
  (void)ge::AttrUtils::GetListListInt(node->GetOpDesc(), kDynamicOutputsIndexes, dyn_out_vv);
  size_t sub_size = 0;
  for (const auto &dyn_v : dyn_in_vv) {
    if (dyn_v.size() > 1) {
      sub_size += (dyn_v.size() - 1);
    }
  }
  for (const auto &dyn_v : dyn_out_vv) {
    if (dyn_v.size() > 1) {
      sub_size += (dyn_v.size() - 1);
    }
  }
  GELOGD("Dynamic in group size[%zu] out group size[%zu] sub size[%zu].", dyn_in_vv.size(), dyn_out_vv.size(), sub_size);
  size_t args_size = node_desc.input_num + node_desc.output_num + node_desc.addr_num + node_desc.workspace_cap + 1;
  if (node_desc.dynamic_folded) {
    // sub size will be folded
    args_size -= sub_size;
  }

  args_size *= node_desc.thread_num_max;
  args_size *= sizeof(TensorAddress);
  if (!args_desc.SetSize(kArgsHostAddr, args_size)) {
    return ge::GRAPH_FAILED;
  }

  if (CalcAtomicArgsSize(node_desc, args_desc) != ge::GRAPH_SUCCESS) {
    GELOGE(ge::FAILED, "Calculate atomic args size failed.");
    return ge::GRAPH_FAILED;
  }

  if (CalcDynamicArgsSize(node, args_desc, dyn_in_vv, true, dyn_desc_v) != ge::GRAPH_SUCCESS) {
    GELOGE(ge::FAILED, "Calculate dynamic input args size failed.");
    return ge::GRAPH_FAILED;
  }
  if (CalcDynamicArgsSize(node, args_desc, dyn_out_vv, false, dyn_desc_v) != ge::GRAPH_SUCCESS) {
    GELOGE(ge::FAILED, "Calculate dynamic output args size failed.");
    return ge::GRAPH_FAILED;
  }

  if (ge::AddOverflow(sizeof(RtFFTSKernelLaunchArgs), args_desc.GetArgsTotalSize(), total_size)) {
    return ge::GRAPH_FAILED;
  }
  GELOGD("FFTS tiling size:%zu, base size:%zu, total size:%zu.", tiling_data_size, args_desc.GetArgsTotalSize(),
         total_size);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus RtFFTSKernelLaunchArgs::Init(const RtFFTSKernelLaunchArgs::ComputeNodeDesc &node_desc,
                                             const RtFFTSKernelLaunchArgs::ArgsDesc &args_desc,
                                             DynDescInfo &dyn_desc_v) {
  GELOGD("Init ffts args.");
  if (memcpy_s(&node_desc_, sizeof(node_desc_), &node_desc, sizeof(node_desc)) != EOK) {
    return ge::GRAPH_FAILED;
  }
  if (memcpy_s(&args_desc_, sizeof(args_desc_), &args_desc, sizeof(args_desc)) != EOK) {
    return ge::GRAPH_FAILED;
  }

  auto dy_input_desc_addr = GetArgsPointer<TensorAddress>(FFTSArgsType::kDyInputsDescData);
  FE_ASSERT_NOTNULL(dy_input_desc_addr);
  size_t dy_input_desc_cap = args_desc_.GetCap(FFTSArgsType::kDyInputsDescData);
  if (memset_s(dy_input_desc_addr, dy_input_desc_cap, 0, dy_input_desc_cap) != EOK) {
    return ge::GRAPH_FAILED;
  }

  auto dy_output_desc_addr = GetArgsPointer<TensorAddress>(FFTSArgsType::kDyOutputsDescData);
  FE_ASSERT_NOTNULL(dy_output_desc_addr);
  size_t dy_output_desc_cap = args_desc_.GetCap(FFTSArgsType::kDyOutputsDescData);
  if (memset_s(dy_output_desc_addr, dy_output_desc_cap, 0, dy_output_desc_cap) != EOK) {
    return ge::GRAPH_FAILED;
  }
  args_size_ = args_desc.GetArgsTotalSize();
  tiling_offset_ = args_desc.GetCap(kTilingData);
  tiling_abs_pos_ = GetArgsPointer<uint8_t>(kTilingData) - reinterpret_cast<uint8_t *>(this);
  size_t args_pos_bt = args_desc.GetOffset(kArgsHostAddr);
  args_pos_ = args_pos_bt / sizeof(uintptr_t);
  args_abs_pos_ = GetArgsPointer<uint8_t>(kArgsHostAddr) - reinterpret_cast<uint8_t *>(this);
  args_pos_bt = args_desc.GetOffset(kAtomArgsHostAddr);
  atom_args_pos_ = args_pos_bt / sizeof(uintptr_t);
  atom_args_abs_pos_ = GetArgsPointer<uint8_t>(kAtomArgsHostAddr) - reinterpret_cast<uint8_t *>(this);
  atom_tiling_offset_ = args_desc.GetCap(kAtomTilingData);
  atom_tiling_abs_pos_ = GetArgsPointer<uint8_t>(kAtomTilingData) - reinterpret_cast<uint8_t *>(this);
  GELOGD(
      "Args total size[%zu], tiling:offset[%zu]abs_pos[%zu], args:pos[%zu]abs_pos[%zu],"
      " atom:args[%zu]tiling[%zu]offset[%zu].",
      args_size_, tiling_offset_, tiling_abs_pos_, args_pos_, args_abs_pos_, atom_args_abs_pos_, atom_tiling_abs_pos_,
      atom_tiling_offset_);

  auto dyn_in_desc_v = GetArgsPointer<DynDesc>(kDyInputsDescData);
  dyn_in_num_ = dyn_desc_v.dyn_in_desc.size();
  for (size_t i = 0; i < dyn_in_num_; ++i) {
    dyn_in_desc_v[i] = dyn_desc_v.dyn_in_desc[i];
    GELOGD("Dynamic in io index:%ld.", dyn_in_desc_v[i].io_index);
  }

  auto dyn_out_desc_v = GetArgsPointer<DynDesc>(kDyOutputsDescData);
  dyn_out_num_ = dyn_desc_v.dyn_out_desc.size();
  for (size_t i = 0; i < dyn_out_num_; ++i) {
    dyn_out_desc_v[i] = dyn_desc_v.dyn_out_desc[i];
    GELOGD("Dynamic out io index:%ld.", dyn_out_desc_v[i].io_index);
  }
  GELOGD("Node dynamic input/output size: [%zu]/[%zu].", dyn_in_num_, dyn_out_num_);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus RtFFTSKernelLaunchArgs::RedirectTilingAddr() {
  if (!ge::IntegerChecker<uint16_t>::Compat(args_desc_.offsets[kTilingData])) {
    GELOGE(ge::FAILED, "TilingData offset overflow %zu", args_desc_.offsets[kTilingData]);
    return ge::GRAPH_FAILED;
  }
  if (!ge::IntegerChecker<uint16_t>::Compat(args_desc_.offsets[kTailTilingData])) {
    GELOGE(ge::FAILED, "Tail tilingData offset overflow %zu", args_desc_.offsets[kTailTilingData]);
    return ge::GRAPH_FAILED;
  }
  tiling_data_.Init(node_desc_.max_tiling_data, GetArgsPointer<void>(kTilingData));
  tail_tiling_data_.Init(node_desc_.max_tail_tiling_data, GetArgsPointer<void>(kTailTilingData));
  GELOGD("Tiling offset: [%zu][%zu], args offset: [%zu].", args_desc_.offsets[kTilingData],
         args_desc_.offsets[kTailTilingData], args_desc_.offsets[kArgsHostAddr]);

  if (node_desc_.need_atomic) {
    atom_tiling_data_.Init(node_desc_.max_atom_tiling_data, GetArgsPointer<void>(kAtomTilingData));
    atom_tail_tiling_data_.Init(node_desc_.max_atom_tail_tiling_data, GetArgsPointer<void>(kAtomTailTilingData));
  }
  return ge::GRAPH_SUCCESS;
}

void RtFFTSKernelLaunchArgs::SetDynInAddr(size_t in_index, size_t &arg_index, uintptr_t data_base, const Shape &shape,
                                          void *args_dev_base) {
  auto dyn_in_desc_v = GetArgsPointer<DynDesc>(kDyInputsDescData);
  size_t i = 0;
  for (; i < dyn_in_num_; ++i) {
    if (dyn_in_desc_v[i].io_index == in_index) {
      GELOGD("Found dynamic in io index: %ld.", in_index);
      break;
    }
  }
  if (i == dyn_in_num_) {
    GetArgsPointer<uintptr_t>(kArgsHostAddr)[arg_index++] = data_base;
    return;
  }
  auto &dyn_desc = dyn_in_desc_v[i];
  size_t dev_pos = GetArgsPointer<uint8_t>(kDyInputsHostAddr) - reinterpret_cast<uint8_t *>(this);
  if (dyn_desc.is_group_first) {
    GELOGD("Input io index: %ld is the first in its group.", dyn_desc.io_index);
    void *dyn_dev_ptr = static_cast<void *>(&(static_cast<uint8_t *>(args_dev_base)[dev_pos + dyn_desc.group_offset]));
    GetArgsPointer<uintptr_t>(kArgsHostAddr)[arg_index++] = ge::PtrToValue(dyn_dev_ptr);
  }
  auto dyn_host_ptr = GetArgsPointer<void>(kDyInputsHostAddr);
  SetDynShape(shape, static_cast<uint8_t *>(dyn_host_ptr), dyn_desc, shape_offset_);
  auto tmp_addr = static_cast<void *>(GetArgsPointer<uint8_t>(kDyInputsHostAddr) + dyn_desc.group_offset);
  auto io_addr_ptr = &(static_cast<uintptr_t *>(tmp_addr)[dyn_desc.ptr_offset / sizeof(TensorAddress)]);
  io_addr_ptr[dyn_desc.io_group_id] = data_base;
  return;
}

void RtFFTSKernelLaunchArgs::SetDynOutAddr(size_t out_index, size_t &arg_index, uintptr_t data_base, const Shape &shape,
                                           void *args_dev_base) {
  auto dyn_out_desc_v = GetArgsPointer<DynDesc>(kDyOutputsDescData);
  size_t i = 0;
  for (; i < dyn_out_num_; ++i) {
    if (dyn_out_desc_v[i].io_index == out_index) {
      GELOGD("Find dynamic out io index: %ld.", out_index);
      break;
    }
  }
  if (i == dyn_out_num_) {
    GetArgsPointer<uintptr_t>(kArgsHostAddr)[arg_index++] = data_base;
    return;
  }
  auto &dyn_desc = dyn_out_desc_v[i];
  size_t dev_pos = GetArgsPointer<uint8_t>(kDyOutputsHostAddr) - reinterpret_cast<uint8_t *>(this);
  if (dyn_desc.is_group_first) {
    GELOGD("Output io index: %ld is the first in the group.", dyn_desc.io_index);
    void *dyn_dev_ptr = static_cast<void *>(&(static_cast<uint8_t *>(args_dev_base)[dev_pos + dyn_desc.group_offset]));
    GetArgsPointer<uintptr_t>(kArgsHostAddr)[arg_index++] = ge::PtrToValue(dyn_dev_ptr);
  }
  auto dyn_host_ptr = GetArgsPointer<void>(kDyOutputsHostAddr);
  SetDynShape(shape, static_cast<uint8_t *>(dyn_host_ptr), dyn_desc, shape_offset_);
  auto tmp_addr = static_cast<void *>(GetArgsPointer<uint8_t>(kDyOutputsHostAddr) + dyn_desc.group_offset);
  auto io_addr_ptr = &(static_cast<uintptr_t *>(tmp_addr)[dyn_desc.ptr_offset / sizeof(TensorAddress)]);
  io_addr_ptr[dyn_desc.io_group_id] = data_base;
  return;
}

ge::graphStatus RtFFTSKernelLaunchArgs::SetIoAddr(size_t io_index, size_t &arg_index, uintptr_t data_base,
                                                  const Shape &shape, void *args_dev_base) {
  auto base_in_64 = GetArgsPointer<uintptr_t>(kArgsHostAddr);
  if (!node_desc_.dynamic_folded) {
    base_in_64[arg_index++] = data_base;
    return ge::GRAPH_SUCCESS;
  }
  if (io_index < node_desc_.input_num) {
    SetDynInAddr(io_index, arg_index, data_base, shape, args_dev_base);
  } else {
    size_t out_index = io_index - node_desc_.input_num;
    SetDynOutAddr(out_index, arg_index, data_base, shape, args_dev_base);
  }
  return ge::GRAPH_SUCCESS;
}

const DynDesc *RtFFTSKernelLaunchArgs::GetDyDescByIoIndex(size_t io_index) {
  if (io_index < node_desc_.input_num) {
    auto dyn_in_desc_v = GetArgsPointer<DynDesc>(kDyInputsDescData);
    for (size_t i = 0; i < dyn_in_num_; ++i) {
      if (dyn_in_desc_v[i].io_index == io_index) {
        return &dyn_in_desc_v[i];
      }
    }
  } else {
    size_t out_index = io_index - node_desc_.input_num;
    auto dyn_out_desc_v = GetArgsPointer<DynDesc>(kDyOutputsDescData);
    for (size_t i = 0; i < dyn_out_num_; ++i) {
      if (dyn_out_desc_v[i].io_index == out_index) {
        return &dyn_out_desc_v[i];
      }
    }
  }
  return nullptr;
}

std::unique_ptr<uint8_t[]> RtFFTSKernelLaunchArgs::Create(const ge::NodePtr &node,
                                                          RtFFTSKernelLaunchArgs::ComputeNodeDesc &node_desc,
                                                          size_t &size) {
  size_t total_size = 0;
  ArgsDesc args_desc{0};
  DynDescInfo dyn_desc_v;
  if (CalcTotalSize(node, node_desc, args_desc, total_size, dyn_desc_v) != ge::GRAPH_SUCCESS) {
    return nullptr;
  }
  auto holder = ge::MakeUnique<uint8_t[]>(total_size);
  if (holder == nullptr) {
    return nullptr;
  }
  if (reinterpret_cast<RtFFTSKernelLaunchArgs *>(holder.get())->Init(node_desc, args_desc, dyn_desc_v) !=
      ge::GRAPH_SUCCESS) {
    return nullptr;
  }
  size = total_size;
  return holder;
}
}  // namespace gert
