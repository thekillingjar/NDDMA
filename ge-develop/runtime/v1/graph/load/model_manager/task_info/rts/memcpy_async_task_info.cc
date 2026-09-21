/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/load/model_manager/task_info/rts/memcpy_async_task_info.h"

#include "graph/load/model_manager/davinci_model.h"
#include "graph/load/model_manager/model_utils.h"
#include "common/checker.h"
#include "common/runtime_api_wrapper.h"

namespace ge {
Status MemcpyAsyncTaskInfo::Init(const domi::TaskDef &task_def, DavinciModel *const davinci_model,
                                 const PisToArgs &args, const PisToPersistentWorkspace &persistent_workspace,
                                 const IowAddrs &iow_addrs) {
  GELOGI("MemcpyAsyncTaskInfo Init Start");
  (void)persistent_workspace;
  GE_CHECK_NOTNULL(davinci_model);
  davinci_model_ = davinci_model;
  GE_CHK_STATUS_RET_NOLOG(SetStream(task_def.stream_id(), davinci_model_->GetStreamList()));

  const domi::MemcpyAsyncDef &memcpy_async = task_def.memcpy_async();
  count_ = memcpy_async.count();
  kind_ = memcpy_async.kind();
  dst_max_ = memcpy_async.dst_max();
  op_index_ = memcpy_async.op_index();
  const OpDescPtr op_desc = davinci_model_->GetOpByIndex(op_index_);
  if (op_desc == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Can't get op_desc from davinci_model by index:%u", memcpy_async.op_index());
    GELOGE(INTERNAL_ERROR, "[Get][Op] Task op index:%u out of range", memcpy_async.op_index());
    return INTERNAL_ERROR;
  }
  op_desc_ = op_desc;
  GE_ASSERT_TRUE((!(iow_addrs.input_logic_addrs.empty())),
                 "[Check][Param] Op:%s, input logic addr list is empty.", op_desc->GetName().c_str());

  GE_ASSERT_TRUE((!(iow_addrs.output_logic_addrs.empty())),
                 "[Check][Param] Op:%s, output logic addr list is empty.", op_desc->GetName().c_str());

  const auto args_placement = ((args_mem_type_ & RT_MEMORY_TS) == 0U)
      ? ArgsPlacement::kArgsPlacementHbm : ArgsPlacement::kArgsPlacementTs;
  // todo: 对于model with queue的场景, memcpy链接到了model io后，不能支持零拷贝，不能把自身转换成二级指针拷贝
  // 原因：model with queue的场景，memcpy链接到了model io后，如果支持零拷贝，他的零拷贝动作是device aicpu来完成的
  // memcpy把自己转换成二级指针拷贝后的args 是ts内存，device aicpu没有访问ts内存的权限，无法完成两拷贝刷新args的动作
  if (((iow_addrs.input_logic_addrs[0].support_refresh) || (iow_addrs.output_logic_addrs[0].support_refresh)
      || davinci_model->GetPhysicalMemoryRefreshable()) &&
      (!((davinci_model_->IsArgsUpdateByDeviceAicpu()) && (args_placement == ArgsPlacement::kArgsPlacementTs)))) {
    GE_ASSERT_TRUE((args[static_cast<size_t>(args_placement)].dev_addr != 0U),
                   "[Check][Param] Op:%s, args_placement:%d, dev addr is nullptr.",
                   op_desc->GetName().c_str(), args_placement);
    src_ = PtrToPtr<void, uint8_t>(ValueToPtr(args[static_cast<size_t>(args_placement)].dev_addr));
    dst_ = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(src_) + sizeof(void *)));
    kind_ = RT_MEMCPY_ADDR_DEVICE_TO_DEVICE;
    GE_CHK_STATUS_RET(SetIoAddrs(iow_addrs), "[Set][Addrs] failed, op:%s", op_desc->GetName().c_str());
    const auto addrs = VPtrToValue(io_addrs_);
    GE_ASSERT_SUCCESS(args_io_addrs_updater_.Init(davinci_model_->GetLogicalMemAllocation(), addrs,
        io_addr_mem_types_, {op_desc->GetName(), op_desc->GetType()}));
    davinci_model_->SetZeroCopyAddr(op_desc, addrs, io_addrs_.data(), static_cast<uintptr_t>(PtrToValue(src_)),
                                    (sizeof(void *) * 2U), 0U, {});

    GELOGI("MemcpyAsyncTaskInfo Init Success, op name %s, src:%p, dst:%p, args_placement:%d, max:%" PRIu64
           ", count:%" PRIu64 ".logic stream id: %u, stream: %p.", op_desc->GetName().c_str(), src_, dst_,
           static_cast<int32_t>(args_placement), dst_max_, count_, task_def.stream_id(), stream_);
    return SUCCESS;
  }

  logical_src_addr_ = iow_addrs.input_logic_addrs[0U].logic_addr;
  src_ = PtrToPtr<void, uint8_t>(ValueToPtr(logical_src_addr_));

  logical_dst_addr_ = iow_addrs.output_logic_addrs[0U].logic_addr;
  dst_ = PtrToPtr<void, uint8_t>(ValueToPtr(logical_dst_addr_));

  GE_ASSERT_SUCCESS(davinci_model_->DisableZeroCopy(src_));
  GE_ASSERT_SUCCESS(davinci_model_->DisableZeroCopy(dst_));

  GELOGI("MemcpyAsyncTaskInfo Init Success, op name %s, logic[0x%" PRIx64 ", 0x%" PRIx64 "], src:%p, dst:%p, max:%" PRIu64
         ", count:%" PRIu64 ", logic stream id: %u, stream: %p.", op_desc->GetName().c_str(), memcpy_async.src(),
         memcpy_async.dst(), src_, dst_, dst_max_, count_, task_def.stream_id(), stream_);
  return SUCCESS;
}

Status MemcpyAsyncTaskInfo::Distribute() {
  GE_ASSERT_NOTNULL(op_desc_);
  GELOGI("MemcpyAsyncTaskInfo Distribute Start. op %s, src:%p, dst:%p, dst_max:%" PRIu64 ", count:%" PRIu64 ", kind:%u",
         op_desc_->GetName().c_str(), src_, dst_, dst_max_, count_, kind_);
  SetTaskTag(op_desc_->GetName().c_str());

  MemcpyAsyncMemInfo memcpy_info;
  memcpy_info.dst = dst_;
  memcpy_info.destMax = dst_max_;
  memcpy_info.src = src_;
  memcpy_info.cnt = count_;
  const rtError_t rt_ret = ge::rtMemcpyAsyncWithCfg(memcpy_info, static_cast<rtMemcpyKind_t>(kind_),
                                                    stream_, qosCfg_);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_INNER_ERR_MSG("E19999", "Call rtMemcpyAsyncWithCfg failed, size:%" PRIu64 ", ret:%d", dst_max_, rt_ret);
    GELOGE(RT_FAILED, "[Call][rtMemcpyAsyncWithCfg] failed, size:%" PRIu64 ", ret:%d", dst_max_, rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  if (!domi::GetContext().is_online_model) {
    op_desc_.reset(); // Release OpDesc after Distribute.
  }
  is_support_redistribute_ = true;
  GELOGI("MemcpyAsyncTaskInfo Distribute Success, stream: %p.", stream_);
  return SUCCESS;
}

Status MemcpyAsyncTaskInfo::ParseTaskRunParam(const domi::TaskDef &task_def, DavinciModel *const davinci_model,
                                              TaskRunParam &task_run_param) {
  GE_CHECK_NOTNULL(davinci_model);
  const domi::MemcpyAsyncDef &memcpy_async = task_def.memcpy_async();
  const OpDescPtr op_desc = davinci_model->GetOpByIndex(memcpy_async.op_index());
  GE_ASSERT_NOTNULL(op_desc);
  uint8_t *src = nullptr;
  uint8_t *dst = nullptr;
  const Status ret1 = ModelUtils::GetRtAddress(davinci_model->GetRuntimeParam(),
                                               static_cast<uintptr_t>(memcpy_async.src()),
                                               src, logical_src_mem_type_);
  const Status ret2 = ModelUtils::GetRtAddress(davinci_model->GetRuntimeParam(),
                                               static_cast<uintptr_t>(memcpy_async.dst()),
                                               dst, logical_dst_mem_type_);
  if ((ret1 != SUCCESS) || (ret2 != SUCCESS)) {
    return PARAM_INVALID;
  }
  task_run_param.parsed_input_addrs.push_back({PtrToValue(src), logical_src_mem_type_, true, {0}});

  // dst_ needs different address for different chips
  std::vector<int64_t> memory_type_list;
  (void)AttrUtils::GetListInt(op_desc, ATTR_NAME_OUTPUT_MEM_TYPE_LIST, memory_type_list);
  const RuntimeParam &rts_param = davinci_model->GetRuntimeParam();
  if ((!memory_type_list.empty()) && (memory_type_list[0U] == static_cast<int64_t>(RT_MEMORY_TS))) {
    GE_CHECK_GE(memcpy_async.dst(), rts_param.logic_mem_base);
    const uint64_t mem_offset = memcpy_async.dst() - rts_param.logic_mem_base;
    dst = PtrToPtr<void, uint8_t>(
        rts_param.ts_mem_mall->Acquire(static_cast<int64_t>(mem_offset), memcpy_async.dst_max()));
    GE_CHECK_NOTNULL(dst);
    logical_dst_mem_type_ = RT_MEMORY_TS;
  }
  task_run_param.parsed_output_addrs.push_back({PtrToValue(dst), logical_dst_mem_type_, true, {0}});

  constexpr uint32_t args_size = static_cast<uint32_t>(sizeof(void *) * 2U);
  args_mem_type_ = rtGetTsMemType(MEM_REQUEST_FEATURE_DEFAULT, args_size);
  pls_ = ((args_mem_type_ & RT_MEMORY_TS) == 0U)
      ? ArgsPlacement::kArgsPlacementHbm : ArgsPlacement::kArgsPlacementTs;
  task_run_param.args_descs.push_back({static_cast<int64_t>(args_size), pls_});

  GELOGI("op_desc:%s, src:%p, src_mem_type:0x%" PRIx64 ", dst:%p, dst_mem_type:0x%" PRIx64 ", "
         "args_mem_type:0x%x, args_placement:%d",
         op_desc->GetName().c_str(), src, logical_src_mem_type_, dst, logical_dst_mem_type_, args_mem_type_,
         static_cast<int32_t>(pls_));

  return SUCCESS;
}

Status MemcpyAsyncTaskInfo::SetIoAddrs(const IowAddrs &iow_addrs) {
  logical_src_addr_ = iow_addrs.input_logic_addrs[0U].logic_addr;
  logical_dst_addr_ = iow_addrs.output_logic_addrs[0U].logic_addr;
  io_addrs_.emplace_back(ValueToPtr(iow_addrs.input_logic_addrs[0U].logic_addr));
  io_addrs_.emplace_back(ValueToPtr(iow_addrs.output_logic_addrs[0U].logic_addr));
  io_addr_mem_types_.emplace_back(iow_addrs.input_logic_addrs[0U].memory_type);
  io_addr_mem_types_.emplace_back(iow_addrs.output_logic_addrs[0U].memory_type);
  return SUCCESS;
}

Status MemcpyAsyncTaskInfo::UpdateHostArgs(const std::vector<uint64_t> &active_mem_base_addr,
                                           void *const host_args,
                                           const size_t host_args_max_len) {
  GELOGI("MemcpyAsyncTaskInfo::UpdateArgs in.");
  GE_ASSERT_SUCCESS(args_io_addrs_updater_.SetArgIoAddrs(active_mem_base_addr, host_args, host_args_max_len));
  GELOGI("MemcpyAsyncTaskInfo::UpdateArgs success.");
  return SUCCESS;
}

Status MemcpyAsyncTaskInfo::GetTaskArgsRefreshInfos(std::vector<TaskArgsRefreshInfo> &infos) {
  args_io_addrs_updater_.GenArgsRefreshInfos(infos, 0UL, pls_);
  return SUCCESS;
}

// todo: model args manager 功能适配完毕后，GetMemType, args_mem_type_接口也可以删除
uint32_t MemcpyAsyncTaskInfo::GetMemType() {
  return args_mem_type_;
}

int64_t MemcpyAsyncTaskInfo::ParseOpIndex(const domi::TaskDef &task_def) const {
  const domi::MemcpyAsyncDef &memcpy_async = task_def.memcpy_async();
  return static_cast<int64_t>(memcpy_async.op_index());
}

REGISTER_TASK_INFO(MODEL_TASK_MEMCPY_ASYNC, MemcpyAsyncTaskInfo);
}  // namespace ge
