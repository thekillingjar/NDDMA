/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_OPTIMIZER_FORMAT_SELECTOR_BUILTIN_BROADCAST_FORMAT_PROCESS_BROADCAST_PROCESS_FRACTAL_Z_H_
#define FUSION_ENGINE_OPTIMIZER_FORMAT_SELECTOR_BUILTIN_BROADCAST_FORMAT_PROCESS_BROADCAST_PROCESS_FRACTAL_Z_H_

#include "format_selector/builtin/broadcast/format_process/broadcast_format_process.h"

namespace fe {
class BroadcastProcessFractalZ : public BroadcastFormatProcess {
 public:
  BroadcastProcessFractalZ(){};
  ~BroadcastProcessFractalZ() override {};

  bool CheckPartNonScalarInputs(const FormatProccessInputArg &input_arg) override;

  bool CheckAllNonScalarInputs(const FormatProccessArgs &args) override;
};
}  // namespace fe

#endif  // FUSION_ENGINE_OPTIMIZER_FORMAT_SELECTOR_BUILTIN_BROADCAST_FORMAT_PROCESS_BROADCAST_PROCESS_FRACTAL_Z_H_
