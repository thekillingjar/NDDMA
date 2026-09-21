/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_COMMON_DEBUG_MEMORY_DUMPER_H_
#define GE_COMMON_DEBUG_MEMORY_DUMPER_H_

#include <cstdint>

#include "framework/common/types.h"
#include "mmpa/mmpa_api.h"
#include "framework/common/ge_inner_error_codes.h"

namespace ge {
constexpr int32_t kInvalidFd = (-1);
// MemoryDumper：dump memory data for internal test
// Output in one time: using DumpToFile
// Open file at one time and output multiple times: create  MemoryDumper object first, and using Open/Dump/Close
class MemoryDumper {
 public:
  MemoryDumper() = default;
  ~MemoryDumper();

  // Assignment/copy is not allowed to avoid repeated release
  MemoryDumper &operator=(const MemoryDumper &dumper)& = delete;
  MemoryDumper(const MemoryDumper &dumper) = delete;

  /** @ingroup domi_common
   *  @brief write memory data to file, if the filename does not exist, create it first
   *  @param [in] filename  the output file path, specific to filename
   *  @param [in] data the memory data
   *  @param [in] len length of data
   *  @return SUCCESS  output success
   *  @return FAILED   output failed
   *  @author
   */
  static Status DumpToFile(const char_t *const filename, const void * const data, const uint64_t len);

  /** @ingroup domi_common
   *  @brief close the Dump file
   *  @return SUCCESS  success
   *  @return FAILED   failed
   *  @author
   */
  void Close() noexcept;

 private:
  /** @ingroup domi_common
   *  @brief open the dump file
   *  @param [in] filename the output file path, specific to filename
   *  @return int32_t the file handle after file open, -1 means open file failed
   *  @author
   */
  static int32_t OpenFile(const std::string &filename);
  static void PrintErrorMsg(const std::string &error_msg);
  int32_t fd_ = kInvalidFd;
};
}  // namespace ge

#endif  // GE_COMMON_DEBUG_MEMORY_DUMPER_H_
