/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rt_kernel_launch_args_ex.h"
#include "exe_graph/runtime/kernel_context.h"
#include "register/kernel_registry.h"
#include "exe_graph/runtime/tiling_data.h"
#include "common/plugin/ge_make_unique_util.h"
#include "common/checker.h"
#include "runtime/subscriber/global_tracer.h"
#include "graph/def_types.h"

namespace gert {
namespace {
constexpr size_t kAlignBytes4 = 4U;

ge::Status ParseMergedCopyInfoFromArgsInfo(const ArgsInfosDesc &args_info_desc, size_t &back_fill_secondary_addr_num,
                                           size_t &merged_copy_size) {
  back_fill_secondary_addr_num = 0U;
  merged_copy_size = 0U;
  const size_t args_info_num = args_info_desc.GetArgsInfoNum();
  for (size_t idx = 0U; idx < args_info_num; ++idx) {
    auto args_info = args_info_desc.GetArgsInfoByIndex(idx);
    FE_ASSERT_NOTNULL(args_info);
    if (args_info->arg_format == ArgsInfosDesc::ArgsInfo::ArgsInfoFormat::FOLDED_DESC_ADDR && args_info->folded_first) {
      FE_ASSERT_TRUE(!ge::AddOverflow(merged_copy_size, args_info->dyn_desc.group_size, merged_copy_size));
      ++back_fill_secondary_addr_num;
    }
  }
  GELOGD("Node back-fill L2 addr count is %zu.", back_fill_secondary_addr_num);
  return ge::SUCCESS;
}
std::unique_ptr<uint8_t[]> CreateFlexibleArrayHolderImpl(size_t struct_body_size, size_t flexible_mem_size,
                                                         size_t &total_size) {
  if (ge::AddOverflow(flexible_mem_size, struct_body_size, total_size)) {
    return nullptr;
  }
  auto holder = ge::MakeUnique<uint8_t[]>(total_size);
  return holder;
}
ge::graphStatus UpdateIoArgsInfo(RtKernelLaunchArgsEx &rt_args, int16_t relative_offset) {
  auto relative_arg_offset = static_cast<int64_t>(relative_offset);
  auto &io_args_info = rt_args.GetIoArgsInfo();
  for (size_t idx = 0U; idx < io_args_info.GetIoArgNum(); ++idx) {
    auto io_arg = io_args_info.GetIoArgByIndex(idx);
    FE_ASSERT_NOTNULL(io_arg);
    if (io_arg->is_need_merged_copy) {
      int64_t rectified_arg_offset = 0;
      FE_ASSERT_TRUE(
          !ge::AddOverflow(static_cast<int64_t>(io_arg->arg_offset), relative_arg_offset, rectified_arg_offset));
      FE_ASSERT_TRUE(rectified_arg_offset >= 0);
      IoArgsInfo::IoArg rectified_io_arg = {static_cast<size_t>(rectified_arg_offset), io_arg->start_index,
                                            io_arg->arg_num, io_arg->is_input, true, io_arg->folded_first,
                                            io_arg->dyn_desc, io_arg->data_size};
      io_args_info.SetIoArgByIndex(idx, rectified_io_arg);
    }
  }
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus UpdateHostInputInfo(RtKernelLaunchArgsEx &rt_args, int16_t relative_offset) {
  auto relative_data_offset = static_cast<int64_t>(relative_offset);
  auto base = rt_args.GetBase();
  FE_ASSERT_NOTNULL(base);
  int64_t rectified_data_offset = 0;
  for (size_t idx = 0U; idx < rt_args.GetBackFillSecondaryAddrNum(); ++idx) {
    FE_ASSERT_TRUE(!ge::AddOverflow(static_cast<int64_t>(base->hostInputInfoPtr[idx].dataOffset), relative_data_offset,
                                    rectified_data_offset));
    FE_ASSERT_TRUE(rectified_data_offset >= 0);
    base->hostInputInfoPtr[idx].dataOffset =
        static_cast<decltype(rtHostInputInfo_t::dataOffset)>(rectified_data_offset);
  }
  return ge::GRAPH_SUCCESS;
}
void DebugForIoArgsInfo(const IoArgsInfo &io_args_info) {
  if (GlobalTracer::GetInstance()->GetEnableFlags() == 0U) {
    return;
  }
  std::stringstream ss;
  ss << "Print io args info, total num[" << io_args_info.GetIoArgNum() << "].";
  GELOGD("%s", ss.str().c_str());
  ss.str("");
  for (size_t idx = 0U; idx < io_args_info.GetIoArgNum(); ++idx) {
    auto io_arg = io_args_info.GetIoArgByIndex(idx);
    if (io_arg) {
      ss << "Io args info[" << idx << "]: arg offset[" << io_arg->arg_offset << "], start index[" << io_arg->start_index
         << "], arg num[" << io_arg->arg_num << "], is input[" << io_arg->is_input << "], is need merged copy["
         << io_arg->is_need_merged_copy << "], folded_first[." << io_arg->folded_first << "]";
      GELOGD("%s", ss.str().c_str());
      ss.str("");
    }
  }
}
}  // namespace
bool RtKernelLaunchArgsEx::ComputeNodeDesc::ArgsCountValid() const {
  if (!ge::IntegerChecker<uint32_t>::Compat(input_num) || !ge::IntegerChecker<uint32_t>::Compat(output_num) ||
      !ge::IntegerChecker<uint32_t>::Compat(workspace_cap) || !ge::IntegerChecker<uint32_t>::Compat(max_tiling_data) ||
      !ge::IntegerChecker<uint32_t>::Compat(compiled_args_size)) {
    return false;
  }
  return true;
}
std::unique_ptr<uint8_t[]> RtKernelLaunchArgsEx::ComputeNodeDesc::Create(size_t compiled_arg_size, size_t &total_size) {
  auto holder = CreateFlexibleArrayHolderImpl(sizeof(ComputeNodeDesc), compiled_arg_size, total_size);
  FE_ASSERT_NOTNULL(holder);

  auto node_desc = reinterpret_cast<ComputeNodeDesc *>(holder.get());
  node_desc->compiled_args_size = compiled_arg_size;

  return holder;
}
std::unique_ptr<uint8_t[]> ArgsInfosDesc::Create(size_t args_infos_size, size_t &total_size) {
  auto holder = CreateFlexibleArrayHolderImpl(sizeof(ArgsInfosDesc), args_infos_size, total_size);
  FE_ASSERT_NOTNULL(holder);

  auto args_info_desc = reinterpret_cast<ArgsInfosDesc *>(holder.get());
  args_info_desc->args_infos_size_ = args_infos_size;

  return holder;
}
bool RtKernelLaunchArgsEx::CalcTotalSize(const RtKernelLaunchArgsEx::ComputeNodeDesc &node_desc,
                                         const ArgsInfosDesc &args_info_desc, RtKernelLaunchArgsEx::ArgsDesc &args_desc,
                                         size_t &total_size) {
  FE_ASSERT_TRUE(node_desc.ArgsCountValid());
  // reserve IoArgsInfo at the begining of args
  total_size = 0U;
  const auto io_arg_num = args_info_desc.GetArgsInfoNum();
  FE_ASSERT_TRUE(!ge::AddOverflow(total_size, io_arg_num * sizeof(IoArgsInfo::IoArg), total_size));
  // caution: we must call `SetSize` in the order of enum ArgsType
  FE_ASSERT_TRUE(args_desc.SetSize(ArgsType::kArgsCompiledArgs, node_desc.compiled_args_size));
  FE_ASSERT_TRUE(
      args_desc.SetSize(ArgsType::kArgsInputsAddr, args_info_desc.GetValidInputArgsNum() * sizeof(TensorAddress)));
  FE_ASSERT_TRUE(
      args_desc.SetSize(ArgsType::kArgsOutputsAddr, args_info_desc.GetValidOutputArgsNum() * sizeof(TensorAddress)));
  FE_ASSERT_TRUE(
      args_desc.SetSize(ArgsType::kShapeBufferAddr, node_desc.need_shape_buffer ? sizeof(TensorAddress) : 0U));
  FE_ASSERT_TRUE(args_desc.SetSize(ArgsType::kWorkspacesAddr, node_desc.workspace_cap * sizeof(TensorAddress)));
  FE_ASSERT_TRUE(args_desc.SetSize(ArgsType::kTilingDataAddr, sizeof(TensorAddress)));
  FE_ASSERT_TRUE(args_desc.SetSize(ArgsType::kOverflowAddr, node_desc.need_overflow ? sizeof(TensorAddress) : 0U));
  size_t tiling_data_size;
  FE_ASSERT_TRUE(TilingData::CalcTotalSize(node_desc.max_tiling_data, tiling_data_size) == ge::GRAPH_SUCCESS);
  tiling_data_size = ge::RoundUp(tiling_data_size, sizeof(uintptr_t));
  FE_ASSERT_TRUE(args_desc.SetSize(ArgsType::kTilingData, tiling_data_size));

  size_t back_fill_secondary_addr_num = 0U;
  size_t merged_copy_size = 0U;
  FE_ASSERT_SUCCESS(ParseMergedCopyInfoFromArgsInfo(args_info_desc, back_fill_secondary_addr_num, merged_copy_size));
  // dynamic io direct addr use hostInputData mem for merged copy
  size_t host_input_data_size = 0U;
  FE_ASSERT_TRUE(
      !ge::AddOverflow(kMaxHostInputDataLen + kHostInputDataAlginSize, merged_copy_size, host_input_data_size));
  FE_ASSERT_TRUE(args_desc.SetSize(ArgsType::kHostInputData, host_input_data_size));
  size_t host_input_info_num = 0U;
  FE_ASSERT_TRUE(!ge::AddOverflow(node_desc.input_num, back_fill_secondary_addr_num, host_input_info_num));
  FE_ASSERT_TRUE(args_desc.SetSize(ArgsType::kHostInputInfo, host_input_info_num * sizeof(rtHostInputInfo_t)));
  FE_ASSERT_TRUE(!ge::AddOverflow(sizeof(RtKernelLaunchArgsEx), total_size, total_size));
  FE_ASSERT_TRUE(!ge::AddOverflow(total_size, args_desc.GetArgsTotalSize(), total_size));
  return true;
}

bool RtKernelLaunchArgsEx::Init(const RtKernelLaunchArgsEx::ComputeNodeDesc &node_desc,
                                const ArgsInfosDesc &args_info_desc, const RtKernelLaunchArgsEx::ArgsDesc &args_desc,
                                const size_t total_size) {
  total_size_ = total_size;
  FE_ASSERT_SUCCESS(ParseMergedCopyInfoFromArgsInfo(args_info_desc, back_fill_secondary_addr_num_, merged_copy_size_));
  host_input_data_size_ = merged_copy_size_;
  FE_ASSERT_EOK(memcpy_s(&node_desc_, sizeof(node_desc_), &node_desc, sizeof(node_desc)));
  FE_ASSERT_EOK(memcpy_s(&args_desc_, sizeof(args_desc_), &args_desc, sizeof(args_desc)));
  FE_ASSERT_EOK(memcpy_s(&args_info_desc_, sizeof(args_info_desc_), &args_info_desc, sizeof(args_info_desc)));
  FE_ASSERT_SUCCESS(InitIoArgsInfo(args_info_desc, node_desc));

  // init base
  base.args = data + io_args_info_.GetIoArgSize();
  auto host_data_base_addr = GetArgsPointer<TensorAddress>(ArgsType::kHostInputData);
  FE_ASSERT_NOTNULL(host_data_base_addr);
  size_t host_data_mem_cap = args_desc_.GetCap(ArgsType::kHostInputData);
  if (memset_s(host_data_base_addr, host_data_mem_cap, 0, host_data_mem_cap) != EOK) {
    return false;
  }
  base.isNoNeedH2DCopy = false;
  base.hostInputInfoPtr = GetArgsPointer<rtHostInputInfo_t>(ArgsType::kHostInputInfo);
  base.hostInputInfoNum = back_fill_secondary_addr_num_;
  size_t tiling_addr_offset = args_desc_.offsets[static_cast<uint32_t>(ArgsType::kTilingDataAddr)];
  FE_ASSERT_TRUE((tiling_addr_offset <= std::numeric_limits<decltype(base.tilingAddrOffset)>::max()),
                 "tilingAddrOffset overflow %zu", tiling_addr_offset);
  size_t tiling_data_offset = args_desc_.offsets[static_cast<uint32_t>(ArgsType::kTilingData)];
  FE_ASSERT_TRUE((tiling_data_offset <= std::numeric_limits<decltype(base.tilingDataOffset)>::max()),
                 "tilingDataOffset overflow %zu", tiling_data_offset);
  // base.argsSize, base.hasTiling will be updated when execute
  base.tilingAddrOffset = args_desc_.offsets[static_cast<uint32_t>(ArgsType::kTilingDataAddr)];
  base.tilingDataOffset = args_desc_.offsets[static_cast<uint32_t>(ArgsType::kTilingData)];

  // init base.args
  if (node_desc_.compiled_args_size > 0) {
    if (memcpy_s(GetArgsPointer<void>(ArgsType::kArgsCompiledArgs), args_desc_.GetCap(ArgsType::kArgsCompiledArgs),
                 node_desc.compiled_args, node_desc.compiled_args_size) != EOK) {
      return false;
    }
  }
  *GetArgsPointer<TensorAddress>(ArgsType::kTilingDataAddr) = nullptr;
  FE_ASSERT_SUCCESS(InitHostInputInfoByDynamicIo());

  tilingData.Init(node_desc.max_tiling_data, GetArgsPointer<void>(ArgsType::kTilingData));

  // init ArgsCacheInfo
  args_cache_info_.cache_status = TilingCacheStatus::kDisabled;
  args_cache_info_.tiling_cache = nullptr;
  return true;
}
std::unique_ptr<uint8_t[]> RtKernelLaunchArgsEx::Create(const RtKernelLaunchArgsEx::ComputeNodeDesc &node_desc,
                                                        const ArgsInfosDesc &args_info_desc) {
  size_t total_size = 0U;
  ArgsDesc args_desc{0};
  if (!CalcTotalSize(node_desc, args_info_desc, args_desc, total_size)) {
    return nullptr;
  }

  auto holder = ge::MakeUnique<uint8_t[]>(total_size);
  if (holder == nullptr) {
    return nullptr;
  }
  auto launch_arg = reinterpret_cast<RtKernelLaunchArgsEx *>(holder.get());
  if (!launch_arg->Init(node_desc, args_info_desc, args_desc, total_size)) {
    return nullptr;
  }
  return holder;
}

std::unique_ptr<uint8_t[]> RtKernelLaunchArgsEx::MakeCopy() {
  auto new_arg_holder = ge::MakeUnique<uint8_t[]>(total_size_);
  FE_ASSERT_NOTNULL(new_arg_holder);
  FE_ASSERT_EOK(memcpy_s(new_arg_holder.get(), total_size_, GetBase(), total_size_));
  auto& new_arg = *reinterpret_cast<RtKernelLaunchArgsEx *>(new_arg_holder.get());
  const auto args_info_num = new_arg.args_info_desc_.GetArgsInfoNum();
  new_arg.io_args_info_.Init(args_info_num, new_arg.data);
  // 1. update base, base.argsSize, base.hasTiling will be updated before launch
  // The offset of kHostInputData will be updated in CopyFlowLaunch, the current offset may be incorrect
  new_arg.base.args = new_arg.data + new_arg.io_args_info_.GetIoArgSize();
  new_arg.base.hostInputInfoPtr = new_arg.GetArgsPointer<rtHostInputInfo_t>(ArgsType::kHostInputInfo);
  new_arg.base.hostInputInfoNum = new_arg.back_fill_secondary_addr_num_;
  // offsets are guaranteed not to overflow in Init
  new_arg.base.tilingAddrOffset = new_arg.args_desc_.offsets[static_cast<uint32_t>(ArgsType::kTilingDataAddr)];
  new_arg.base.tilingDataOffset = new_arg.args_desc_.offsets[static_cast<uint32_t>(ArgsType::kTilingData)];
  *new_arg.GetArgsPointer<TensorAddress>(ArgsType::kTilingDataAddr) = nullptr;
  FE_ASSERT_SUCCESS(new_arg.InitHostInputInfoByDynamicIo());
  // 2. update tilingData
  new_arg.tilingData.Init(new_arg.node_desc_.max_tiling_data, new_arg.GetArgsPointer<void>(ArgsType::kTilingData));
  new_arg.tilingData.SetDataSize(tilingData.GetDataSize());

  return new_arg_holder;
}

ge::graphStatus RtKernelLaunchArgsEx::UpdateMergedCopyInfo() {
  base.hostInputInfoNum = GetBackFillSecondaryAddrNum();
  host_input_data_size_ = GetMergedCopySize();

  auto &tiling_data = GetTilingData();
  auto tiling_data_size = tiling_data.GetDataSize();  // real tiling size
  tiling_data_size = ge::RoundUp(tiling_data_size, sizeof(uintptr_t));
  GELOGD("tiling data offset %zu, tiling data size %zu", args_desc_.GetOffset(ArgsType::kTilingData), tiling_data_size);
  if (tiling_data_size >
      (args_desc_.GetOffset(ArgsType::kHostInputInfo) - args_desc_.GetOffset(ArgsType::kTilingData))) {
    GELOGE(ge::FAILED, "tiling data size %zu, arg end offset %zu, tilling data offset %zu", tiling_data_size,
           args_desc_.GetOffset(ArgsType::kArgsTypeEnd), args_desc_.GetOffset(ArgsType::kTilingData));
    return ge::GRAPH_FAILED;
  }
  auto pre_host_input_data_offset = static_cast<int16_t>(GetArgsOffset(RtKernelLaunchArgsEx::ArgsType::kHostInputData));
  if (!args_desc_.UpdateOffset(ArgsType::kHostInputData,
                               args_desc_.GetOffset(ArgsType::kTilingData) + tiling_data_size)) {
    GELOGE(ge::FAILED, "update host input data offset failed, tiling offset %zu, tiling data size %zu",
           args_desc_.GetOffset(ArgsType::kTilingData), tiling_data_size);
    return ge::GRAPH_FAILED;
  }
  const auto max_input_host_data_len = kMaxHostInputDataLen + merged_copy_size_;
  if ((args_desc_.GetOffset(ArgsType::kHostInputData) + max_input_host_data_len) >
      args_desc_.GetOffset(ArgsType::kHostInputInfo)) {
    GELOGE(ge::FAILED, "host input data offset %zu, host input data len %zu, host input info offset %zu",
           args_desc_.GetOffset(ArgsType::kHostInputData), max_input_host_data_len,
           args_desc_.GetOffset(ArgsType::kHostInputInfo));
    return ge::GRAPH_FAILED;
  }
  auto cur_host_input_data_offset = static_cast<int16_t>(GetArgsOffset(RtKernelLaunchArgsEx::ArgsType::kHostInputData));
  int16_t relative_offset = 0;
  FE_ASSERT_TRUE(!ge::AddOverflow(cur_host_input_data_offset, (0 - pre_host_input_data_offset), relative_offset));
  GELOGD("Host input offset from [%d] to [%d] with shift [%d].", pre_host_input_data_offset, cur_host_input_data_offset,
         relative_offset);
  FE_ASSERT_SUCCESS(UpdateIoArgsInfo(*this, relative_offset));
  FE_ASSERT_SUCCESS(UpdateHostInputInfo(*this, relative_offset));
  GELOGD("After update, tiling data offset: %zu, tiling data size: %zu, host input data offset: %zu.",
         args_desc_.GetOffset(ArgsType::kTilingData), tiling_data_size, args_desc_.GetOffset(ArgsType::kHostInputData));
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus RtKernelLaunchArgsEx::UpdateHostInputArgs(HostInputInfo &host_mem_input) {
  const size_t max_host_input_data_len = kMaxHostInputDataLen + merged_copy_size_;
  FE_ASSERT_TRUE(host_input_data_size_ < max_host_input_data_len);
  auto dst_len_left = max_host_input_data_len - host_input_data_size_;
  size_t tensor_size = host_mem_input.host_tensor_size;
  FE_ASSERT_TRUE(dst_len_left >= tensor_size);

  auto host_data_base_addr = GetArgsPointer<TensorAddress>(ArgsType::kHostInputData);
  size_t host_mem_offset = args_desc_.GetOffset(ArgsType::kHostInputData);
  auto host_inputs_info = GetArgsPointer<rtHostInputInfo_t>(ArgsType::kHostInputInfo);
  void *const data_addr = ge::ValueToPtr(ge::PtrToValue(host_data_base_addr) + host_input_data_size_);
  GELOGD("data_addr is %p, host_input_addr is %p", data_addr, host_mem_input.host_input_addr);
  if ((tensor_size > 0U) && (memcpy_s(data_addr, dst_len_left, host_mem_input.host_input_addr, tensor_size) != EOK)) {
    GELOGE(ge::FAILED, "dst length is %zu, src length is %zu", dst_len_left, tensor_size);
    return ge::GRAPH_FAILED;
  }
  FE_ASSERT_TRUE((host_mem_offset + host_input_data_size_) <=
                 static_cast<uint32_t>(std::numeric_limits<decltype(rtHostInputInfo_t::dataOffset)>::max()));
  auto inputs_index_data = static_cast<const int32_t *>(host_mem_input.inputs_index->GetData());
  for (size_t i = 0U; i < host_mem_input.inputs_index->GetSize(); ++i) {
    auto input_index = inputs_index_data[i];
    for (size_t idx = 0U; idx < io_args_info_.GetIoArgNum(); ++idx) {
      auto io_arg = io_args_info_.GetIoArgByIndex(idx);
      FE_ASSERT_NOTNULL(io_arg);
      if (!io_arg->is_input) {
        continue;
      }
      if (input_index != io_arg->start_index) {
        continue;
      }
      host_inputs_info[base.hostInputInfoNum].addrOffset =
          static_cast<decltype(rtHostInputInfo_t::addrOffset)>(io_arg->arg_offset);
      host_inputs_info[base.hostInputInfoNum].dataOffset =
          static_cast<decltype(rtHostInputInfo_t::dataOffset)>(host_mem_offset + host_input_data_size_);
      GELOGD("input index %d, addr offset %u, data offset %u", input_index,
             host_inputs_info[base.hostInputInfoNum].addrOffset, host_inputs_info[base.hostInputInfoNum].dataOffset);
      ++base.hostInputInfoNum;
    }
  }
  size_t align_size = ge::RoundUp(static_cast<uint64_t>(tensor_size), kAlignBytes4);
  host_input_data_size_ += align_size;
  return ge::GRAPH_SUCCESS;
}
/*
 * |------------------------------------------------args----------------------------------------------------|
 * |--Compile-args--|--input_addr--|--output_addr--|--back_fill_secondary_addr--|--Tiling data--|--Merge Copy
 * Data--|--Copy Flow Data--| 对于随路拷贝的host tensor,需要在最后做一次32字节对齐再加32的动作， Copy Flow Data
 * 是紧跟在Merge Copy Data之后，计算好对齐size后，重新计算host_input_data_size_（减去原来长度，加上对齐长度）
 * 1. 加载时已保证copy flow data的size已经是kMaxHostInputDataLen(128) + kHostInputDataAlginSize(32), 此时对齐不会越界
 * 2. 对齐时需要保证UpdateMergedCopyInfo跟UpdateHostInputArgs已经执行完成（逻辑已保证）
 * */
ge::graphStatus RtKernelLaunchArgsEx::AlignHostInputSize() {
  GELOGD("Before aligning the host input data size, the host input data size is %zu, and the merged copy size is %zu.",
         host_input_data_size_, merged_copy_size_);
  FE_ASSERT(host_input_data_size_ >= GetMergedCopySize());
  auto copy_flow_size = host_input_data_size_ - GetMergedCopySize();
  size_t align_copy_flow_size = 0U;
  FE_ASSERT_TRUE(!ge::AddOverflow(ge::RoundUp(copy_flow_size, kHostInputDataAlginSize), kHostInputDataAlginSize,
                                  align_copy_flow_size));
  FE_ASSERT_TRUE(align_copy_flow_size >= copy_flow_size);
  FE_ASSERT_TRUE(!ge::AddOverflow(host_input_data_size_, align_copy_flow_size - copy_flow_size, host_input_data_size_));
  GELOGD("After aligning the host input data size is %zu, the original host input data size was %zu.", align_copy_flow_size,
         host_input_data_size_);
  return ge::GRAPH_SUCCESS;
}

ge::Status RtKernelLaunchArgsEx::InitIoArgsInfo(const ArgsInfosDesc &args_info_desc, const ComputeNodeDesc &node_desc) {
  const auto args_info_num = args_info_desc.GetArgsInfoNum();
  io_args_info_.Init(args_info_num, data);

  auto args_input_addr_offset = args_desc_.GetOffset(ArgsType::kArgsInputsAddr);
  auto host_input_data_base = args_desc_.GetOffset(ArgsType::kHostInputData);
  size_t host_in_data_size = 0U;
  size_t real_arg_idx = 0U;
  for (size_t arg_idx = 0U; arg_idx < args_info_num; ++arg_idx) {
    auto args_info = args_info_desc.GetArgsInfoByIndex(arg_idx);
    FE_ASSERT_NOTNULL(args_info);
    bool is_input = args_info->arg_type == ArgsInfosDesc::ArgsInfo::ArgsInfoType::INPUT;
    auto arg_num = static_cast<size_t>(args_info->arg_size);
    GELOGD("Args_info init, idx[%zu], start_index:%d.", arg_idx, args_info->start_index);
    if (args_info->start_index == -1 || args_info->start_index == 0xFFFF) {
      auto arg_offset = args_input_addr_offset + real_arg_idx * sizeof(TensorAddress);
      IoArgsInfo::IoArg io_arg = {static_cast<size_t>(arg_offset), args_info->start_index, arg_num, is_input, false,
                                  args_info->folded_first, args_info->dyn_desc, args_info->data_size};
      io_args_info_.SetIoArgByIndex(arg_idx, io_arg);
      ++real_arg_idx;
      continue;
    }
    auto start_index =
        is_input ? args_info->start_index : args_info->start_index + static_cast<int32_t>(node_desc.input_num);
    if (args_info->arg_format == ArgsInfosDesc::ArgsInfo::ArgsInfoFormat::DIRECT_ADDR) {
      auto arg_offset = args_input_addr_offset + real_arg_idx * sizeof(TensorAddress);
      IoArgsInfo::IoArg io_arg = {static_cast<size_t>(arg_offset),
                                  start_index,
                                  arg_num,
                                  is_input,
                                  false,
                                  args_info->folded_first,
                                  args_info->dyn_desc,
                                  args_info->data_size};
      io_args_info_.SetIoArgByIndex(arg_idx, io_arg);
      ++real_arg_idx;
      continue;
    }
    if (args_info->arg_format == ArgsInfosDesc::ArgsInfo::ArgsInfoFormat::FOLDED_DESC_ADDR) {
      auto arg_offset = host_input_data_base + args_info->dyn_desc.group_offset + args_info->dyn_desc.ptr_offset;
      arg_offset = arg_offset + args_info->dyn_desc.io_group_id * sizeof(TensorAddress);
      IoArgsInfo::IoArg io_arg = {static_cast<size_t>(arg_offset),
                                  start_index,
                                  arg_num,
                                  is_input,
                                  true,
                                  args_info->folded_first,
                                  args_info->dyn_desc,
                                  args_info->data_size};
      io_args_info_.SetIoArgByIndex(arg_idx, io_arg);
      GELOGD("Dynamic arg idx[%zu], io idx[%ld], first[%d], group_size[%zu].", arg_idx, args_info->dyn_desc.io_index,
             args_info->folded_first, args_info->dyn_desc.group_size);
      if (args_info->folded_first) {
        ++real_arg_idx;
        host_in_data_size += args_info->dyn_desc.group_size;
      }
      continue;
    }
  }
  FE_ASSERT_TRUE(host_input_data_size_ == host_in_data_size, "[PARAM][INVALID]Host input data size[%zu], expect[%zu].",
                 host_in_data_size, host_input_data_size_);
  DebugForIoArgsInfo(io_args_info_);
  return ge::SUCCESS;
}

/*
 *  |-----------kHostInputData----------|
 *  |i0|i_p1|i3|i_p2|i6|o0|o_p1|o3|-----|info|i1|i2|info|i4|i5|info|o1|o2|
 *  |--|->addrOffset1-------------------|->dataOffset
 *  |----------|->addrOffset2----------------------|->dataOffset
 *  |---------------------|->addrOffset3----------------------|->dataOffset
 * */
ge::Status RtKernelLaunchArgsEx::InitHostInputInfoByDynamicIo() {
  size_t host_input_info_idx = 0U;
  size_t args_input_addr_offset = args_desc_.GetOffset(ArgsType::kArgsInputsAddr);
  base.hostInputInfoPtr = GetArgsPointer<rtHostInputInfo_t>(ArgsType::kHostInputInfo);
  auto input_data_base = GetArgsOffset(ArgsType::kHostInputData);
  const size_t io_arg_num = io_args_info_.GetIoArgNum();
  size_t real_idx = 0;
  for (size_t idx = 0U; idx < io_arg_num; ++idx) {
    auto io_arg = io_args_info_.GetIoArgByIndex(idx);
    FE_ASSERT_NOTNULL(io_arg);
    if (io_arg->is_need_merged_copy && io_arg->folded_first) {
      FE_ASSERT_TRUE(
          host_input_info_idx < back_fill_secondary_addr_num_,
          "[PARAM][INVALID]Current host input info idx[%zu], expect less than back-fill secondary addr num[%zu]",
          host_input_info_idx, back_fill_secondary_addr_num_);
      base.hostInputInfoPtr[host_input_info_idx].addrOffset = static_cast<decltype(rtHostInputInfo_t::addrOffset)>(
          args_input_addr_offset + real_idx * sizeof(TensorAddress));
      base.hostInputInfoPtr[host_input_info_idx].dataOffset =
          static_cast<decltype(rtHostInputInfo_t::dataOffset)>(input_data_base + io_arg->dyn_desc.group_offset);
      ++host_input_info_idx;
      ++real_idx;
    }
    if (!io_arg->is_need_merged_copy) {
      ++real_idx;
    }
  }
  FE_ASSERT_TRUE(base.hostInputInfoNum == host_input_info_idx);
  return ge::SUCCESS;
}

ge::graphStatus AllocLaunchArg(KernelContext *context) {
  // 该做的事情在outputs creator中已经做完了
  (void)context;
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus CreateOutputsForAllocLaunchArg(const ge::FastNode *node, KernelContext *context) {
  (void)node;
  auto node_desc = context->GetInputPointer<RtKernelLaunchArgsEx::ComputeNodeDesc>(
      static_cast<size_t>(AllocLaunchArgInputs::kNodeDesc));
  FE_ASSERT_NOTNULL(node_desc);
  auto args_info_desc =
      context->GetInputPointer<ArgsInfosDesc>(static_cast<size_t>(AllocLaunchArgInputs::kArgsInfoDesc));
  FE_ASSERT_NOTNULL(args_info_desc);
  auto av_holder = context->GetOutput(static_cast<size_t>(AllocLaunchArgOutputs::kHolder));
  FE_ASSERT_NOTNULL(av_holder);
  auto av_arg = context->GetOutput(static_cast<size_t>(AllocLaunchArgOutputs::kRtArg));
  FE_ASSERT_NOTNULL(av_arg);
  auto av_arg_arg = context->GetOutput(static_cast<size_t>(AllocLaunchArgOutputs::kRtArgArgsBase));
  FE_ASSERT_NOTNULL(av_arg_arg);
  auto av_tiling_data = context->GetOutput(static_cast<size_t>(AllocLaunchArgOutputs::kTilingDataBase));
  FE_ASSERT_NOTNULL(av_tiling_data);
  auto rt_arg = RtKernelLaunchArgsEx::Create(*node_desc, *args_info_desc);
  FE_ASSERT_NOTNULL(rt_arg);

  auto arg = reinterpret_cast<RtKernelLaunchArgsEx *>(rt_arg.get());
  av_arg->Set(arg->GetBase(), nullptr);
  av_arg_arg->Set(arg->GetArgBase(), nullptr);
  av_tiling_data->Set(&arg->GetTilingData(), nullptr);
  av_holder->SetWithDefaultDeleter<uint8_t[]>(rt_arg.release());

  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(AllocLaunchArg).RunFunc(AllocLaunchArg).OutputsCreator(CreateOutputsForAllocLaunchArg);

void AssignDynamicDesc(DynDesc &dyn_desc, size_t j, int64_t index, size_t dyn_num) {
  dyn_desc.is_group_first = (j == 0);
  dyn_desc.io_index = static_cast<uint32_t>(index);
  dyn_desc.io_group_id = j;
  dyn_desc.dyn_num = dyn_num;
}

ge::graphStatus CalcDynamicArg(const ge::NodePtr &node, std::vector<std::vector<int64_t>> &dyn_in_vv, bool is_input,
                               std::vector<ArgsInfosDesc::ArgsInfo> &args_infos, size_t &pre_size) {
  int64_t offset = (args_infos[0].start_index == 0xFFFF) ? 1 : 0;
  size_t args_size = args_infos.size();
  size_t in_num = args_size - node->GetAllOutDataAnchorsSize() - offset;
  GELOGD("in args size[%zu] of total size[%zu].", in_num, args_size);
  for (const auto &dyn_v : dyn_in_vv) {
    size_t dyn_num = dyn_v.size();
    GELOGD("Node[%s]'s %s dynamic group size is %zu.", node->GetNamePtr(), is_input ? "Input" : "Output", dyn_num);
    size_t addr_offset = 1;  // first is ptr_offset
    for (size_t i = 0; i < dyn_num; ++i) {
      int64_t index = dyn_v[i];
      const auto &tensor_desc = is_input ? node->GetOpDescBarePtr()->MutableInputDesc(index)
                                         : node->GetOpDescBarePtr()->MutableOutputDesc(index);
      FE_ASSERT_NOTNULL(tensor_desc);
      index = is_input ? index : (index + in_num);
      index += offset;
      if (static_cast<size_t>(index) >= args_size) {
        GELOGE(ge::FAILED, "Index %ld over args_info size %zu.", index, args_infos.size());
        return ge::GRAPH_FAILED;
      }
      auto &arg_info = args_infos[index];
      arg_info.be_folded = (i != 0);
      arg_info.arg_format = ArgsInfosDesc::ArgsInfo::ArgsInfoFormat::FOLDED_DESC_ADDR;
      arg_info.folded_first = (i == 0);
      auto dim_num = tensor_desc->GetShape().IsScalar() ? 1 : tensor_desc->GetShape().GetDimNum();
      if (dim_num > MAX_DIM_NUM || dim_num == 0U) {
        dim_num = MAX_DIM_NUM;
      }
      addr_offset++;           // dim && cnt
      addr_offset += dim_num;  // dim value
    }
    addr_offset = addr_offset * sizeof(TensorAddress);
    size_t group_size = addr_offset + dyn_num * sizeof(TensorAddress);  // ptrs
    for (size_t j = 0; j < dyn_num; ++j) {
      auto index = dyn_v[j];
      auto arg_index = is_input ? index : (index + in_num);
      arg_index += offset;
      auto &dyn_desc = args_infos[arg_index].dyn_desc;
      AssignDynamicDesc(dyn_desc, j, index, dyn_num);
      dyn_desc.ptr_offset = addr_offset;
      dyn_desc.group_size = group_size;
      dyn_desc.group_offset = pre_size;
      GELOGD("%s io: %ld is dynamic with ptr_offset[%zu], group_size[%zu], and group_offset[%zu].",
             is_input ? "Input" : "Output", index, addr_offset, group_size, pre_size);
    }
    FE_ASSERT_TRUE(!ge::AddOverflow(pre_size, group_size, pre_size));
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ProcFoldedDescArgs(const ge::NodePtr &compute_node, std::vector<ArgsInfosDesc::ArgsInfo> &args_infos) {
  GELOGD("Node [%s] process folded args info.", compute_node->GetNamePtr());
  const auto op_desc = compute_node->GetOpDescBarePtr();
  FE_ASSERT_NOTNULL(op_desc);
  std::vector<std::vector<int64_t>> dyn_in_vv;
  (void)ge::AttrUtils::GetListListInt(op_desc, kDynamicInputsIndexes, dyn_in_vv);
  std::vector<std::vector<int64_t>> dyn_out_vv;
  (void)ge::AttrUtils::GetListListInt(op_desc, kDynamicOutputsIndexes, dyn_out_vv);
  size_t pre_size = 0U;
  FE_ASSERT_GRAPH_SUCCESS(CalcDynamicArg(compute_node, dyn_in_vv, true, args_infos, pre_size));
  FE_ASSERT_GRAPH_SUCCESS(CalcDynamicArg(compute_node, dyn_out_vv, false, args_infos, pre_size));
  return ge::GRAPH_SUCCESS;
}
/*
 * |-- uint64_t --|
 * |  ptr_offset  |  -- addr offset to ptr1
 * | dim_n | cnt  |  \
 * |     dim_0    |   \
 * |     dim_1    |     dynamic in 1 shape
 * |     .....    |   /
 * |     dim_n    |  /
 * | dim_n | cnt  |  \
 * |     dim_0    |   \
 * |     dim_1    |     dynamic in 2 shape
 * |     .....    |   /
 * |     dim_n    |  /
 * |     ptr1     |  -- dynamic in 1 addr
 * |     ptr2     |  -- dynamic in 2 addr
 * */
void SetDynShape(const Shape &shape, uint8_t *host_addr, const DynDesc &dyn_desc, size_t &shape_offset) {
  void *grp_host_base = &(host_addr[dyn_desc.group_offset]);
  if (dyn_desc.is_group_first) {
    static_cast<uintptr_t *>(grp_host_base)[0] = dyn_desc.ptr_offset;
    shape_offset = 1U;
  }
  uintptr_t *base_in_64 = &(static_cast<uintptr_t *>(grp_host_base)[shape_offset]);
  auto base_addr = static_cast<void *>(base_in_64);
  auto dim_num = shape.IsScalar() ? 1 : shape.GetDimNum();
  static_cast<uint32_t *>(base_addr)[0] = dim_num;  // dim num
  static_cast<uint32_t *>(base_addr)[1] = 1;        // cnt
  for (size_t i = 0; i < dim_num; ++i) {
    GELOGD("Io: %ld dim[%zu] with val[%ld].", dyn_desc.io_index, i, shape.GetDim(i));
    base_in_64[1 + i] = shape.IsScalar() ? 1 : shape.GetDim(i);
  }
  shape_offset += (dim_num + 1);
  return;
}
}  // namespace gert
