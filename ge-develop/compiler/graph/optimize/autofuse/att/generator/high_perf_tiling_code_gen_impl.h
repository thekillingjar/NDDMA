/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATT_HIGH_PERF_TILING_CODE_GEN_IMPL_H_
#define ATT_HIGH_PERF_TILING_CODE_GEN_IMPL_H_

#include <string>
#include "tiling_code_gen_impl.h"

namespace att {
class HighPerfTilingCodeGenImpl : public TilingCodeGenImpl {
 public:
  explicit HighPerfTilingCodeGenImpl(const std::string &op_name, const TilingCodeGenConfig &config,
                                     const TilingModelInfo &model_infos,
                                     const ScoreFuncs &score_funcs,
                                     const bool is_uniq_group)
      : TilingCodeGenImpl(op_name, config, model_infos, score_funcs, is_uniq_group) {}
  ~HighPerfTilingCodeGenImpl() override = default;

 protected:
  ge::Status GenExternFuncDef() override;
  ge::Status GenTilingImplPublicFunc() override;
  ge::Status GenToolFuncs() override;
  ge::Status GenSolverBaseClass() override;
  ge::Status GenSolverTiling(const ModelInfo &model_info) override;
  ge::Status GenDoTiling(const ModelInfo &model_info) override;
};
}  // namespace att
#endif