/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_NODE_CONVERTER_AUTOFUSE_NODE_CONVERTER_H_
#define AIR_CXX_RUNTIME_V2_NODE_CONVERTER_AUTOFUSE_NODE_CONVERTER_H_

#include "register/node_converter_registry.h"
#include "kernel/common_kernel_impl/tiling.h"
#include "mmpa/mmpa_api.h"

namespace gert {
enum class SymbolTilingInputIndex {
  kTilingParse = 1U,
  kFwkData,
  kEnd
};

enum class CacheableSymbolTilingInput {
  kTilingParse = 1U,
  kUsedSymbolSource,
  kCacheableFwkData,
  kEnd
};

enum class DlsymKernelInputIndex {
  kFuncName,
  kSoHandles,
  kEnd
};

enum class GetSymbolTilingCacheKeyInput {
  kGetSymbolTilingCacheKeyFunc = 1U,
  kAllSymbolNum,
  kNum
};

enum class PrepareSymbolTilingCacheOutput {
  kUsedSymbolSource,
  kCacheableFwkData,
  kNum
};

enum class GetAutofuseFuncsOutput {
  kAutofuseSo,
  kTilingFunc,
  kInferShape,
  kGetTilingCacheKey,
  kTilingParse,
  kDfxInputSymbolInfo,
  kNum
};

enum class SymbolInferShapeInput {
  kAllSymbolNum = 1U,
  kDfxInputSymbolInfoFunc,
  kInferFunc,
  kNum
};

class AutofuseNodeConveter {
 public:
  static bg::ValueHolderPtr GetAutofuseHandle(
      LoweringGlobalData &global_data, const ge::NodePtr &node, GetAutofuseFuncsOutput index);
  static std::vector<bg::ValueHolderPtr> GetSymbolInputsWithSize(
      LoweringGlobalData &global_data, const std::vector<bg::ValueHolderPtr> &input_shapes,
      const std::string &graph_name);
  static bg::ValueHolderPtr GetAllSymbolNumHolder(LoweringGlobalData &global_data, const ge::NodePtr &node);
};

LowerResult LoweringAutofuseNode(const ge::NodePtr &node, const LowerInput &lower_input);
}
#endif  // AIR_CXX_RUNTIME_V2_NODE_CONVERTER_AUTOFUSE_NODE_CONVERTER_H_
