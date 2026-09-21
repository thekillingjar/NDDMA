/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_ADAPTER_TBE_ADAPTER_KERNEL_LAUNCH_L2_CACHE_KERNEL_LAUNCH_H_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_ADAPTER_TBE_ADAPTER_KERNEL_LAUNCH_L2_CACHE_KERNEL_LAUNCH_H_

#include <memory>
#include <vector>
#include "common/aicore_util_types.h"
#include "common/util/op_info_util.h"
#include "adapter/tbe_adapter/kernel_launch/tbe_kernel_launch.h"
#include "graph/debug/ge_attr_define.h"
#include "securec.h"

namespace fe {
class L2CacheKernelLaunch : public TbeKernelLaunch {
 public:
  explicit L2CacheKernelLaunch(int32_t input_num) : TbeKernelLaunch(input_num) {};
  ~L2CacheKernelLaunch() override{};
  size_t GetAppendArgsSizeOf() override;
  size_t GetAppendArgsNum() override;
  Status AddAppendArgs(const ge::Node &node, void *all_args_buff, const uint32_t &args_size) override;

 private:
  Status GenerateReadModes(const ge::Node &node, vector<uint64_t> &read_modes) const;
  bool IsLifeCycleEnd(const ge::Node &node, const ge::GeTensorDescPtr &input_desc, int input_idx) const;
  L2CacheReadMode GenerateReadMode(const ge::Node &node, const ge::GeTensorDescPtr &input_desc, int input_idx,
                                   bool is_life_cycle_end) const;
  L2CacheReadMode GenRmForSpecialInputOps(const ge::NodePtr &src_node, bool is_enable_reuse_mem) const;
};
}  // namespace fe
#endif  // AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_ADAPTER_TBE_ADAPTER_KERNEL_LAUNCH_L2_CACHE_KERNEL_LAUNCH_H_
