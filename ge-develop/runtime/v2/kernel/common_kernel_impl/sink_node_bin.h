/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_KERNEL_COMMON_KERNEL_IMPL_SINK_NODE_BIN_H_
#define AIR_CXX_RUNTIME_V2_KERNEL_COMMON_KERNEL_IMPL_SINK_NODE_BIN_H_
#include <cstdint>
#include <type_traits>
#include "exe_graph/runtime/base_type.h"
#include "engine/aicore/kernel/aicore_update_kernel.h"
#include "runtime/kernel.h"

namespace gert {
namespace kernel {
// This structure is equivalent to `rtDevBinary_t` defined in RTS and can be cast to it.
// In difference, `BinData` is followed by the bin-data **contiguous in memory**
struct BinData {
  uint32_t magic;    // magic number
  uint32_t version;  // version of binary
  const void *data;  // binary data
  uint64_t length;   // binary length
  uint64_t placeholder;
};
enum class MIX_KERNEL_REQ_TYPE {
  MIX_SINGLE_KERNEL,
  MIX_MULTI_KERNEL,
  ENHANCED_MIX_SINGLE_KERNEL,
  ENHANCED_MIX_DYNAMIC,
  INVALID_KERNEL_TYPE,
};

enum class MIX_KERNEL_TYPE {
  MIX_AIC_ONLY = 1,
  MIX_AIV_ONLY = 2,
  MIX_AIC_AIV  = 3,
  MIX_INVALID_TYPE,
};

enum class MIX_KERNEL_FUNC_NUM {
  MIX_WITH_ONE_KERNEL = 1,
  MIX_WITH_TWO_KERNEL = 2,
  MIX_INVALID_KERNEL,
};

struct KernelFunctionCtx {
  uint32_t kernel_type;
  uint32_t flag;
  uint64_t tiling_key;
  uint64_t tail_tiling_key;
  void *bin_handler;
  void *kernel_unique_ids_addr;
  KernelFunctionCtx()
      : kernel_type(0xFFFF), flag(0xFFFF), tiling_key(0xFFFF), tail_tiling_key(0xFFFF), bin_handler(nullptr),
        kernel_unique_ids_addr(nullptr) {}
  KernelFunctionCtx(uint32_t k_type, uint32_t flag_, uint64_t tiling_key_, uint64_t tail_tiling_key_,
                    void *bin_handler_, void *kernel_unique_ids_addr_)
      : kernel_type(k_type),
        flag(flag_),
        tiling_key(tiling_key_),
        tail_tiling_key(tail_tiling_key_),
        bin_handler(bin_handler_),
        kernel_unique_ids_addr(kernel_unique_ids_addr_) {}
  ~KernelFunctionCtx() = default;
};
static_assert(std::is_standard_layout<BinData>::value, "The class BinData must be a POD");

UINT32 ParseKernelInfo(const rtKernelDetailInfo_t &kernel_info, const rtKernelDetailInfo_t &tail_kernel_info,
                       AICoreSinkRet *sink_ret);
bool CheckKernelInfoValid(const rtKernelDetailInfo_t &kernel_info);
UINT32 GetKernelFuncInfoByKernelType(const KernelFunctionCtx &ctx, const std::string &prefix, AICoreSinkRet *sink_ret);

enum class MIXDyInKey {
  KERNEL_NUM = 0,
  KERNEL_TYPE,
  TILING_KEY,
  TAIL_TILING_KEY,
  PREFIX_START,
  RESERVED
};
}  // namespace kernel
}  // namespace gert
#endif  // AIR_CXX_RUNTIME_V2_KERNEL_COMMON_KERNEL_IMPL_SINK_NODE_BIN_H_
