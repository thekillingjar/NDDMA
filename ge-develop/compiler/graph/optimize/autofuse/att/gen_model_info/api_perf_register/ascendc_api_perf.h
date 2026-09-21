/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef API_PERF_REGISTER_ASCENDC_API_PERF_H_
#define API_PERF_REGISTER_ASCENDC_API_PERF_H_

#include <map>
#include <unordered_map>
#include <sstream>
#include "base/model_info.h"
#include "utils/api_perf_utils.h"

namespace att {
// 负责注册所有的评估函数（包括ASCIR API, AscendC Api）
class EvalCosts {
public:
  static EvalCosts &Instance();

  void RegisterFunc(const std::string &op_type, const Perf &perf_func) {
    func_container_[op_type] = perf_func;
  }
  void RegisterAscendCFunc(const std::string &op_type, const AscendCPerf &ascendc_perf_func) {
    ascendc_func_container_[op_type] = ascendc_perf_func;
  }

  Perf GetFunc(const std::string &op_type) {
    if (func_container_.find(op_type) == func_container_.end()) {
      return nullptr;
    }
    return func_container_[op_type];
  }
  AscendCPerf GetAscendCFunc(const std::string &op_type) {
    if (ascendc_func_container_.find(op_type) == ascendc_func_container_.end()) {
      return nullptr;
    }
    return ascendc_func_container_[op_type];
  }

 private:
  EvalCosts() = default;
  ~EvalCosts() = default;

private:
  std::unordered_map<std::string, Perf> func_container_;
  std::unordered_map<std::string, AscendCPerf> ascendc_func_container_;
};

class FuncRegister {
public:
  FuncRegister(const std::string &op_type, const Perf &func) {
    EvalCosts::Instance().RegisterFunc(op_type, func);
  }
  ~FuncRegister() = default;
};
class AscendcFuncRegister {
public:
  AscendcFuncRegister(const std::string &op_type, const AscendCPerf &func) {
    EvalCosts::Instance().RegisterAscendCFunc(op_type, func);
  }
  ~AscendcFuncRegister() = default;
};

// 1.用于获取AscendC API的性能函数
// 2.默认ASCIR的性能函数也通过该接口获取
Perf GetPerfFunc(const std::string &op_type);
AscendCPerf GetAscendCPerfFunc(const std::string &op_type);

namespace ascendcperf {
ge::Status LoadPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status GetDatasize(const TensorShapeInfo &shape, Expr &dim_product);
ge::Status AbsPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status AddsPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status AddPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status AndPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status BlockReduceMaxPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status BlockReduceMinPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status BrcbPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status CastPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status CopyUbtoUbPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status CopyPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status CompareEQPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status CompareScalarEQPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status CompareGEPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status CompareScalarGEPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status CompareGTPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status CompareScalarGTPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status CompareLEPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status CompareScalarLEPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status CompareLTPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status CompareScalarLTPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status CompareNEPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status CompareScalarNEPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status DivPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status DuplicatePerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status ErfPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status ExpPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status GatherPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status GatherMaskPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status MaxsPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status MaxPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status MinsPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status MinPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status MulsPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status MulPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status OrPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status PairReduceSumPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status ReciprocalPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status ReluPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status RsqrtPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status SelectPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status SigmoidPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status SignPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status SqrtPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status SubPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status TanhPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status WholeReduceSumPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status WholeReduceMaxPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
ge::Status WholeReduceMinPerf(const NodeDetail &node_info, PerfOutputInfo &perf);
// load store的API在不同形态的拟合形式不一样，暂不放在本文件定义
}
// AscendC和ASCIR均可以使用下面的宏注册
#define REGISTER_EVAL_FUNC(op_type, func_name) \
  FuncRegister eval_##op_type(op_type, func_name)
#define REGISTER_ASCENDC_EVAL_FUNC(op_type, func_name) \
  AscendcFuncRegister ascendc_eval_##op_type(op_type, func_name)

#define REGISTER_ASCENDC_EVAL_FUNC_TAG(op_type, tag, func_name) \
  AscendcFuncRegister JOIN_A_B_C(eval_, op_type, tag)((op_type) + (#tag), (func_name))

#define REGISTER_EVAL_FUNC_TAG(op_type, tag, func_name) \
  FuncRegister JOIN_A_B_C(eval_, op_type, tag)((op_type) + (#tag), (func_name))
} // namespace att
#endif // API_PERF_REGISTER_ASCENDC_API_PERF_H_
