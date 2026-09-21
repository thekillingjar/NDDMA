/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_INC_EXE_GRAPH_TILING_CONTEXT_H_
#define METADEF_CXX_INC_EXE_GRAPH_TILING_CONTEXT_H_
#include <cstdint>
namespace ge {
enum class graphStatus {
  SUCCESS = 0,
  FAILED = 1
};
}

namespace fe {
class PlatFormInfos {};
}  // namespace fe

namespace gert {
/**
 * tiling kernel的context
 */
class TilingContext {
 public:
  ge::graphStatus SetTilingKey(const uint64_t tiling_key) {
    tiling_key_ = tiling_key;
    return ge::graphStatus::SUCCESS;
  }
  /**
   * 获取tiling key
   * @return tiling key，获取失败时
   */
  uint64_t GetTilingKey() const {
    return tiling_key_;
  }
  /**
   * 设置block dim
   * @param block_dim block dim
   * @return 成功时返回ge::GRAPH_SUCCESS
   */
  ge::graphStatus SetBlockDim(const uint32_t block_dim) {
    block_dim_ = block_dim;
    return ge::graphStatus::SUCCESS;
  }
  /**
   * 获取block dim
   * @return block dim
   */
  uint32_t GetBlockDim() const {
    return block_dim_;
  }
  /**
   * 获取 fe::PlatFormInfos 指针
   * @return fe::PlatFormInfos 指针
   */
  fe::PlatFormInfos *GetPlatformInfo() const {
    fe::PlatFormInfos *platform_info = new fe::PlatFormInfos();
    return platform_info;
  }
private:
  uint32_t block_dim_{1u};
  uint64_t tiling_key_{0u};
};

}  // namespace gert
#endif  // METADEF_CXX_INC_EXE_GRAPH_TILING_CONTEXT_H_
