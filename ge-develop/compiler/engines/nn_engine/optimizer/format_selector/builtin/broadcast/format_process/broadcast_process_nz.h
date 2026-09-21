/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_OPTIMIZER_FORMAT_SELECTOR_BUILTIN_BROADCAST_FORMAT_PROCESS_BROADCAST_PROCESS_NZ_H_
#define FUSION_ENGINE_OPTIMIZER_FORMAT_SELECTOR_BUILTIN_BROADCAST_FORMAT_PROCESS_BROADCAST_PROCESS_NZ_H_

#include "format_selector/builtin/broadcast/format_process/broadcast_format_process.h"

namespace fe {
class BroadcastProcessNz : public BroadcastFormatProcess {
 public:
  BroadcastProcessNz(){};
  ~BroadcastProcessNz() override {};
  bool CheckOriginFormat(const std::vector<ge::Format> &input_formats,
                         const vector<ge::GeShape> &input_shapes) override;
  bool CheckPartNonScalarInputs(const FormatProccessInputArg &input_arg) override;
  bool CheckAllNonScalarInputs(const FormatProccessArgs &args) override;

 private:
  bool CheckAxisNeedBroadcast(const size_t &index, const std::vector<ge::GeShape> &shapes) const;
  const size_t LAST_DIM = 1;
  const size_t PENULTIMATE_DIM = 2;
};
}  // namespace fe

#endif  // FUSION_ENGINE_OPTIMIZER_FORMAT_SELECTOR_BUILTIN_BROADCAST_FORMAT_PROCESS_BROADCAST_PROCESS_NZ_H_
