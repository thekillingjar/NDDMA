/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "memory_app_type_classifier.h"
#include "model_utils.h"

namespace ge {
static constexpr std::array<const char_t *, static_cast<size_t>(MemoryAppType::kEnd) + 1U> g_memory_app_types_to_str{
    "weight", "feature map", "model io", "unknown"};
const char_t *GetMemoryAppTypeStr(MemoryAppType t) {
  if (t > MemoryAppType::kEnd) {
    t = MemoryAppType::kEnd;
  }
  return g_memory_app_types_to_str[static_cast<size_t>(t)];
}
MemoryAppTypeClassifier::MemoryAppTypeClassifier(const std::vector<MemAllocation> &allocations,
    const size_t fm_start_id) {
  for (const auto &allocation : allocations) {
    // fusion段在fm段之前，逻辑地址和fm段会有重叠
    if (allocation.id < fm_start_id) {
      fusion_mem_allocations_.push_back(std::make_pair(allocation.logical_addr, allocation.data_size));
      continue;
    }
    if (allocation.type == MemAllocation::Type::FEATURE_MAP ||
        allocation.type == MemAllocation::Type::FIXED_FEATURE_MAP) {
      // FEATURE_MAP可能有多个
      (void) sort_fm_allocations_.insert(allocation);
    }
  }
}
MemoryAppType MemoryAppTypeClassifier::ClassifyByLogicalAddr(
    const std::pair<uint64_t, uint64_t> &mem_type_and_logical_addr) const {
  // todo: 简单修改：如果地址类型为TS类型，这里按照kMemoryTypeFeatureMap返回
  // 原因：TS的场景不可能被分成Weight，TS当前也不支持设置成模型的边界，也不可能是model io
  if (mem_type_and_logical_addr.first == static_cast<uint64_t>(RT_MEMORY_TS)) {
    return MemoryAppType::kMemoryTypeFeatureMap;
  }

  // ModelIo的物理内存可能是未申请的，而model io的逻辑地址仍然按照fm_base + offset的方法计算得到，
  // 这就导致了model io地址可能是野指针，与其他的真实申请的、物理地址出现冲突。
  // 在动态shape的静态shape子图场景，fm_base为0，整个fm地址空间都是野指针，也有同样的问题。
  // 因此要判断一块地址是fm内存，需要在首先判断到这块内存的type是fm内存的基础上
  if (ModelUtils::IsFeatureMapOrModelIoType(mem_type_and_logical_addr.first)) {
    const uint64_t logical_addr = mem_type_and_logical_addr.second;
    // 如果是fusion段，识别为io类型
    for (const auto &fusion_mem_allocations : fusion_mem_allocations_) {
      if ((logical_addr >= fusion_mem_allocations.first) &&
          (logical_addr < (fusion_mem_allocations.first + fusion_mem_allocations.second))) {
        return MemoryAppType::kMemoryTypeModelIo;
      }
    }

    MemAllocation allocation_info{};
    allocation_info.logical_addr = logical_addr;
    auto it = sort_fm_allocations_.upper_bound(allocation_info);
    if ((it != sort_fm_allocations_.end()) && (logical_addr >= it->logical_addr)
        && (logical_addr < (it->logical_addr + it->data_size))) {
      if (it->type == MemAllocation::Type::FIXED_FEATURE_MAP) {
        return MemoryAppType::kMemoryTypeFix; // 如果是fix fm段，识别为fix类型
      }
      return MemoryAppType::kMemoryTypeFeatureMap;
    } else {
      return MemoryAppType::kMemoryTypeModelIo;
    }
  } else {
    return MemoryAppType::kMemoryTypeFix;
  }
}
std::map<std::pair<uint64_t, uint64_t>, MemoryAppType> MemoryAppTypeClassifier::ClassifyByTaskRunParams(
    const std::vector<TaskRunParam> &params) const {
  std::map<std::pair<uint64_t, uint64_t>, MemoryAppType> logical_addrs_to_memory_type;
  for (const auto &param : params) {
    ClassifyAddrs(param.parsed_input_addrs, logical_addrs_to_memory_type);
    ClassifyAddrs(param.parsed_output_addrs, logical_addrs_to_memory_type);
    ClassifyAddrs(param.parsed_workspace_addrs, logical_addrs_to_memory_type);
  }
  return logical_addrs_to_memory_type;
}
void MemoryAppTypeClassifier::ClassifyAddrs(
    const std::vector<AddrDesc> &addrs,
    std::map<std::pair<uint64_t, uint64_t>, MemoryAppType> &logical_addrs_to_memory_type) const {
  const bool is_debug_enable = IsLogEnable(GE_MODULE_NAME, DLOG_DEBUG);
  /*
   * 逻辑地址冲突现状、处理和未来
   * 当前的逻辑地址实际上是加载时申请到的物理地址，但是存在如下例外情况，导致当前逻辑地址不是一个完备的编址方式，
   * 会出现逻辑地址冲突的情况：
   * 1. 零拷贝场景，modelio部分的地址段可能没有申请，而modelio的逻辑地址仍然正常编址，
   *    导致实际申请的地址可能与modelio部分冲突
   * 2. 动态shape的静态shape子图场景，fm的基地址为0，导致fm地址与实际申请的其他地址冲突
   *
   * 因此：
   * 1. 使用逻辑地址做key是不完整的，需要使用memory_type+logical_addr共同做key才能保证无冲突
   * 2. 由于同样是fm内存，其memory_type可以有多种表达，这导致使用memory_type+logical_addr做key，
   *    又会导致本来是同一块地址，由于memory type不同，索引成两块了。不过这个问题现在看我们不太在乎，
   *    因为当前ModelArgsManager暂时没有需要合并内存的需求
   * 2. TaskInfo返回的AddrDesc中，仍然需要返回memory type，来协助外部判断一块地址是否是fm内存
   *
   * todo: 下一步重构，应该做到
   * 1. 逻辑地址规范化编址，保证逻辑地址绝对唯一
   * 2. 逻辑地址由框架统一解析，只有少数封装到特定task def中的，由TaskInfo解析
   */
  for (const auto &addr_desc : addrs) {
    const std::pair<uint64_t, uint64_t> t_and_a{addr_desc.memory_type, addr_desc.logic_addr};
    if (logical_addrs_to_memory_type.count(t_and_a) == 0UL) {
      const auto memory_app_type = ClassifyByLogicalAddr(t_and_a);
      logical_addrs_to_memory_type[t_and_a] = memory_app_type;
      if (is_debug_enable) {
        GELOGD("Classify memory type 0x%llx addr 0x%llx memory app type %s(%d)", addr_desc.memory_type,
               addr_desc.logic_addr, GetMemoryAppTypeStr(memory_app_type), static_cast<int32_t>(memory_app_type));
      }
    }
  }
}
}  // namespace ge
