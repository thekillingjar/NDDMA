/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/load/model_manager/task_info/args_io_addrs_updater.h"
#include "common/checker.h"
#include "graph/load/model_manager/model_utils.h"
namespace ge {

// todo: model args manager适配完成后, 这个Init接口要废弃
Status ArgsIoAddrsUpdater::Init(std::vector<MemAllocation> &logical_mem_allocations,
                                const std::vector<uint64_t> &logical_addrs,
                                const std::vector<uint64_t> &mem_types,
                                const ArgsIoAddrsUpdater::OpInfo &op_info) {
  GE_ASSERT_TRUE(logical_addrs.size() == mem_types.size());
  std::vector<uint8_t> refreshable_flags;
  ModelUtils::GetAddrRefreshableFlagsByMemTypes(mem_types, refreshable_flags);
  return Init(logical_mem_allocations, logical_addrs, refreshable_flags, op_info);
}

// todo: model args manager整改完成之后, task info 调用该接口
Status ArgsIoAddrsUpdater::Init(std::vector<MemAllocation> &logical_mem_allocations,
                                const std::vector<uint64_t> &logical_addrs,
                                const std::vector<uint8_t> &refreshable_flags,
                                const ArgsIoAddrsUpdater::OpInfo &op_info) {
  GE_ASSERT_TRUE(logical_addrs.size() == refreshable_flags.size());
  GE_ASSERT_TRUE(logical_mem_allocations.size() >= 1U); // absolute allocation
  const auto match_id = [&logical_mem_allocations] (const uint64_t addr) -> size_t {
    for (auto &item : logical_mem_allocations) {
      if ((addr >= item.logical_addr) && (addr < (item.logical_addr + item.data_size))) {
        item.hit_count++;
        return static_cast<size_t>(item.id);
      }
    }
    return (logical_mem_allocations.size() - 1U); // 匹配不上allocation表，按照absolute处理
  };

  op_info_ = op_info;
  GELOGI("[Args][Init] op_name:%s, op_type:%s, logical_addrs size:%zu.", op_info_.op_name.c_str(),
      op_info_.op_type.c_str(), logical_addrs.size());
  for (size_t io_index = 0U; io_index < logical_addrs.size(); io_index++) {
    const auto &addr = logical_addrs[io_index];
    const auto refreshable_flag = refreshable_flags[io_index];
    const size_t id_addr_matched = match_id(addr);
    const size_t id = (refreshable_flag == 1U) ? id_addr_matched : (logical_mem_allocations.size() - 1U);
    if (id_addr_matched >= logical_mem_allocations.size()) {
      REPORT_INNER_ERR_MSG("E19999", "Can not match any logic mem item");
      GELOGE(INTERNAL_ERROR, "addr:0x%" PRIx64 " can not match any logic mem item", addr);
      return INTERNAL_ERROR;
    }

    MemAllocationAndOffset id_and_offset =
      {id, (addr - logical_mem_allocations[id].logical_addr), logical_mem_allocations[id_addr_matched].type};
    v_mem_allocation_id_and_offset_.emplace_back(id_and_offset);
    GELOGI("[Args][Init] op_name:%s, op_type:%s, logical_addr[%zu]:0x%" PRIx64 ", id:%zu, "
           "offset:0x%" PRIx64 ", refreshable:%hhu.",
           op_info_.op_name.c_str(), op_info_.op_type.c_str(), io_index, addr, id_and_offset.id, id_and_offset.offset,
           refreshable_flag);
  }

  return SUCCESS;
}

Status ArgsIoAddrsUpdater::SetArgIoAddrs(const std::vector<uint64_t> &active_mem_base_addr, void *const host_args,
                                         const size_t host_args_len) const {
  GE_ASSERT_TRUE(v_mem_allocation_id_and_offset_.size() * sizeof(uint64_t) <= host_args_len);
  GELOGI("[Args][Updater] op_name:%s, op_type:%s, host_args:%p.", op_info_.op_name.c_str(),
         op_info_.op_type.c_str(), host_args);
  uint64_t *const host_args_tmp = PtrToPtr<void, uint64_t>(host_args);
  size_t io_index = 0U;
  for (const auto &item : v_mem_allocation_id_and_offset_) {
    GE_ASSERT_TRUE(item.id < active_mem_base_addr.size());
    host_args_tmp[io_index] = active_mem_base_addr[item.id] + item.offset;

    GELOGI("[Args][Updater] op_name:%s, op_type:%s, active_addr[%zu]:0x%" PRIx64 ", id:%zu, "
           "offset:0x%" PRIx64 ", active_base:0x%" PRIx64,
           op_info_.op_name.c_str(), op_info_.op_type.c_str(), io_index, host_args_tmp[io_index], item.id,
           item.offset, active_mem_base_addr[item.id]);
    io_index++;
  }
  return SUCCESS;
}

void ArgsIoAddrsUpdater::GenArgsRefreshInfos(std::vector<TaskArgsRefreshInfo> &infos,
                                             const uint64_t host_args_bas_offset,
                                             const ArgsPlacement pls) {
  for (size_t i = 0U; i < v_mem_allocation_id_and_offset_.size(); i++) {
    TaskArgsRefreshInfo info;
    info.id = v_mem_allocation_id_and_offset_[i].id;
    info.offset = v_mem_allocation_id_and_offset_[i].offset;
    info.io_index = static_cast<uint64_t>(i);
    info.args_offset = host_args_bas_offset + static_cast<uint64_t>(i * sizeof(uint64_t));
    info.placement = pls;
    info.args_format_policy = ArgsFormatPolicy::kAddrAll;
    infos.emplace_back(info);
  }
}

}  // namespace ge
