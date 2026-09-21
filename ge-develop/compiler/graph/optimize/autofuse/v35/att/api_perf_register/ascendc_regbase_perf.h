/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_ASCENDC_REGBASE_PERF_H
#define AUTOFUSE_ASCENDC_REGBASE_PERF_H
#include "api_perf_register/utils/vf_perf_utils.h"
#include "api_perf_register/utils/api_perf_utils.h"
namespace att {
namespace ascendcperf_v2 {
// 工具函数，提取重复代码
struct RepeatParams {
  Expr repeat_elm;
  Expr repeat_time;
};
RepeatParams CalculateRepeatParams(const ge::DataType& input_dtype, const Expr& cal_count);
// 注册V2性能
ge::Status CompareGEPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status CompareEQPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status CompareNEPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status CompareGTPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status CompareLEPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status CompareLTPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status AbsPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status ExpPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status LnPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status SqrtPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status RsqrtPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status DivPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status ReciprocalPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status ReluPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status MaxPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status MinPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status NegPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status MeanPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status AddPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status SubPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status MulPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status LeakyReluPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status CastPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status SumPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status RemovePadPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status WherePerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status PowPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status ErfPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status TanhPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status SigmoidPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status GeluPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status SignPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status LogicalNotPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status LogicalOrPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status LogicalAndPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status ClipByValuePerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status BitwiseAndPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status FloorDivPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
}
}  // namespace att

#endif  // AUTOFUSE_ASCENDC_REGBASE_PERF_H
