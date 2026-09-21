/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMPILER_GRAPHCOMPILER_ENGINES_NNENG_INC_OPS_STORE_BINARY_KERNEL_INFO_H_
#define COMPILER_GRAPHCOMPILER_ENGINES_NNENG_INC_OPS_STORE_BINARY_KERNEL_INFO_H_
#include <string>
#include <vector>
#include <unordered_set>
#include "ops_store/sub_op_info_store.h"
#include "ops_store/op_kernel_info_constructor.h"


namespace fe {
class BinaryKernelInfo;

class BinaryKernelInfo {
  public:
    BinaryKernelInfo() = default;
    ~BinaryKernelInfo() = default;

    BinaryKernelInfo(const BinaryKernelInfo &) = delete;
    BinaryKernelInfo &operator=(const BinaryKernelInfo &) = delete;

    static BinaryKernelInfo &Instance();

    Status Initialize(const std::string &config_file_path);

    void Finalize();

    bool IsBinSupportDynamicRank(const std::string &op_type) const;

  private:
    bool is_init_ = false;
    std::unordered_set<std::string> support_dynamic_rank_ops_;
};
}
#endif  // FUSION_ENGINE_INC_BINARY_KERNEL_INFO_H_
