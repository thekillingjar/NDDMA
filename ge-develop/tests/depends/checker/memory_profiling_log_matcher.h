/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_MEMORY_PROFILING_LOG_MATCHER_H
#define AIR_CXX_MEMORY_PROFILING_LOG_MATCHER_H
#include <string>
#include <sstream>
namespace gert {
/**
 * 重要：本头文件中定义的正则表达式与工具`memory profiling viewer`中的一致，本质上用于看护该工具的有效性。
 * 若需要修改本头文件中的正则表达式，为了避免工具失效，则需要同步修改和发布工具，工具对应正则表达式地址：
 * https://gitee.com/sheng-nan/memory-profiling-viewer/blob/master/src/components/record.js
 */
constexpr const char *kAllocRe = R"(\[KernelTrace]\[(.*)]\[MEM]Alloc memory at stream (\d+), block (((\(nil\))|(0x\w+))), address (0x\w+), size (\d+))";
constexpr const char *kFreeRe = R"(\[KernelTrace]\[(.*)]\[MEM]Free memory at stream (-?\d+), address (0x\w+))";
constexpr const char *kMirrorStream = R"(\[KernelTrace]\[(.*)]Get rts stream (((\(nil\))|(0x\w+))) from logical stream (\d+))";
constexpr const char *kLocalRecycleRe = R"(\[KernelTrace]\[(.*)]\[MEM]Local-recycle memory at stream (\d+), address (\w+))";
constexpr const char *kBorrowRecycleRe = R"(\[KernelTrace]\[(.*)]\[MEM]Borrow-recycle memory at stream (\d+), address (0x\w+))";
constexpr const char *kBirthRecycleRe = R"(\[KernelTrace]\[(.*)]\[MEM]Recycle memory at stream (\d+), address (\w+))";
constexpr const char *kWander = R"(\[KernelTrace]\[(.*)]\[MEM]Memory wandering from stream (\d+) to (\d+), address (0x\w+))";
constexpr const char *kRecycleSendEvent = R"(\[KernelTrace]\[(.*)]\[MEM]Send memory recycling event from stream (\d+), event id (\d+), address (0x\w+))";
constexpr const char *kRecycleRecvEvent = R"(\[KernelTrace]\[(.*)]\[MEM]Wait memory recycling event at stream (\d+), event id (\d+), address (0x\w+))";
constexpr const char *kMigrationSendEvent = R"(\[KernelTrace]\[(.*)]\[MEM]Send memory migration event from stream (\d+), event id (\d+), address (0x\w+))";
constexpr const char *kMigrationRecvEvent = R"(\[KernelTrace]\[(.*)]\[MEM]Wait memory migration event at stream (\d+), event id (\d+), address (0x\w+))";
constexpr const char *kPoolExpand = R"(\[MEM]Expand memory pool at stream (((\(nil\))|(0x\w+))), address (0x\w+), size (\d+))";
constexpr const char *kPoolShrink = R"(\[MEM]Shrink memory pool at stream (((\(nil\))|(0x\w+))), address (0x\w+))";
constexpr const char *kSendEvent = R"(\[KernelTrace]\[(.*)]Sent event (\d+) RT event (((\(nil\))|(0x\w+))) from stream (\d+))";
constexpr const char *kWaitEvent = R"(\[KernelTrace]\[(.*)]Waited event (\d+) RT event (((\(nil\))|(0x\w+))) at stream (\d+))";
constexpr const char *kSendEventWithMem = R"(\[KernelTrace]\[(.*)]\[MEM]Send memory (.*) event from stream (\d+), event id (\d+), address (0x\w+))";
constexpr const char *kWaitEventWithMem = R"(\[KernelTrace]\[(.*)]\[MEM]Wait memory (.*) event at stream (\d+), event id (\d+), address (0x\w+))";
template<typename T>
inline std::string ToHex(T p) {
  std::stringstream ss;
  ss << std::showbase << std::hex << (uint64_t)p;
  return ss.str();
}
}
#endif  // AIR_CXX_MEMORY_PROFILING_LOG_MATCHER_H
