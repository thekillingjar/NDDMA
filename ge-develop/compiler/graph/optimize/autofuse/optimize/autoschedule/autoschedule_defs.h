/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPTIMIZE_AUTOSCHEDULE_AUTOSCHEDULE_DEFS_H_
#define OPTIMIZE_AUTOSCHEDULE_AUTOSCHEDULE_DEFS_H_
#include "ascendc_ir/ascendc_ir_core/ascendc_ir.h"

namespace optimize::autoschedule {
struct AutoScheduleOutput {
  ge::AscGraph scheduled_graph;
  // <dst_var_name, src_var>>
  std::map<std::string, ge::Expression> var_relations_;
  std::string score_func;

  explicit AutoScheduleOutput(const char *name = "default") : scheduled_graph(name) {}
};
}  // namespace optimize::autoschedule

#endif  // OPTIMIZE_AUTOSCHEDULE_AUTOSCHEDULE_DEFS_H_
