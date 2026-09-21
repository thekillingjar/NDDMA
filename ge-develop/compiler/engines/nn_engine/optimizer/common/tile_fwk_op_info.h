/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_GRAPH_OPTIMIZER_JSON_PARSER_FATBIN_INFO_H_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_GRAPH_OPTIMIZER_JSON_PARSER_FATBIN_INFO_H_

#include <map>
#include <vector>
#include <string>
#include <mutex>
#include "register/graph_optimizer/graph_optimize_register_error_codes.h"
#include "graph/op_kernel_bin.h"

namespace fe {
struct FatbinHeaderInfo {
  uint64_t tilingKeyNum = 0;
  std::vector<uint64_t> tilingKeyList;
  std::vector<size_t> binOffsets;
  FatbinHeaderInfo(uint64_t tiling_key_num) : tilingKeyNum(tiling_key_num) {
    tilingKeyList.resize(tiling_key_num);
    binOffsets.resize(tiling_key_num);
  }
};

struct SubkernelInfo {
  ge::OpKernelBinPtr subkernel_ptr;
  int32_t block_dim = 0;
  int64_t workspace_size = 0;
  std::string kernel_name = "";
};

using FatbinKernelInfoMap = std::map<uint64_t, SubkernelInfo>;

enum class KernelContextType : uint64_t {
  OpBinary = 0,
  Kernel,
  Bottom
};

#pragma pack(8)
struct KernelHeader
{
  size_t dataOffset[static_cast<size_t>(KernelContextType::Bottom)];
  size_t dataSize[static_cast<size_t>(KernelContextType::Bottom)];
};
#pragma pack()

class TileFwkOpInfo {
 public:
  TileFwkOpInfo(const TileFwkOpInfo &) = delete;
  TileFwkOpInfo &operator=(const TileFwkOpInfo &) = delete;
  static TileFwkOpInfo& Instance();

  bool CheckFatbinInfo(const std::string &op_type) const;

  Status GetFatbinInfo(const std::string &op_type, const uint64_t &config_key,
                       SubkernelInfo &subkernel_info) const;

  void SetFatbinInfo(const std::string &op_type, const FatbinKernelInfoMap &fatbin_kernel_info_map);

  bool GetTileFwkOpFlag(const std::string &op_type, bool &tile_fwk_op_flag) const;
  
  void SetTileFwkOpFlag(const std::string &op_type, const bool &tile_fwk_op_flag);

 private:
  TileFwkOpInfo();
  ~TileFwkOpInfo();
  mutable std::mutex fatbin_info_mutex_;
  mutable std::mutex tile_fwk_op_flag_mutex_;
  std::map<std::string, FatbinKernelInfoMap> fatbin_info_map_;
  std::map<std::string, bool> tile_fwk_op_flag_map_;
};
}  // namespace fe

#endif  // AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_GRAPH_OPTIMIZER_JSON_PARSER_FATBIN_INFO_H_
