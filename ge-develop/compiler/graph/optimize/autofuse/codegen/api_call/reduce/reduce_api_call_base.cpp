/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <sstream>
#include <string>
#include "attr_utils.h"
#include "ascir_ops.h"
#include "common_utils.h"
#include "graph/ascendc_ir/utils/asc_tensor_utils.h"
#include "api_call/utils/api_call_factory.h"
#include "api_call/utils/api_call_utils.h"

namespace reduce_base {
using namespace codegen;
using namespace ge::ops;
using namespace ge::ascir_op;
using namespace ascgen_utils;

// 用于将代码中的"first"和"last"相互替换
static void ReplaceSS(std::string& str, const std::string& oldSubStr, const std::string& newSubStr)
{
  size_t pos = 0;
  while ((pos = str.find(oldSubStr, pos)) != std::string::npos) {
    str.replace(pos, oldSubStr.length(), newSubStr);
    pos += newSubStr.length();
  }
  return;
}

static void ReplaceSSWithSwappingFirstAndLast(std::string firstAndFirst_actual, std::string lastAndLast_actual, const bool &isAllAxisReduce, std::stringstream &ss)
{
  if (isAllAxisReduce) {
    ReplaceSS(firstAndFirst_actual, "first", "last");
    ReplaceSS(lastAndLast_actual, "last", "first");
  }
  ss << firstAndFirst_actual << ";\n" << lastAndLast_actual << ";\n";
  return;
}

/*  
  返回AR或者RA的fist_size和last_size；
  为了避免由于存在src[i].axis_size=1(此时strides=0)导致误判为R轴，所以在遍历过程中过滤掉了src[i].axis_size为1的情况；
  用last_not_1_axis_size_index来记录上一次的axis_size != 1的位置。
*/
void ReduceMergedSizeCodeGen(const TPipe &tpipe, std::stringstream &ss, const Tensor &src, const Tensor &dst,
                             bool is_tail = false) {
  std::stringstream first;
  std::stringstream first_actual;
  std::stringstream last;
  std::stringstream last_actual;
  first << "{\nuint32_t " << (is_tail ? "first_tail" : "first") << " = 1";
  first_actual << "uint32_t " << (is_tail ? "first_tail_actual" : "first_actual") << " = 1";
  last << "uint32_t " << (is_tail ? "last_tail" : "last") << " = 1";
  last_actual << "uint32_t " << (is_tail ? "last_tail_actual" : "last_actual") << " = 1";
  std::string dtype_name;
  Tensor::DtypeName(dst.dtype, dtype_name);
  bool is_first = true;
  const size_t num_axes = src.vectorized_axis.size();
  ascir::SizeExpr lastNonZeroStride = Zero;
  size_t last_not_1_axis_size_index = 0xFFFFFFFF;
  bool isAllAxisReduce = true;
  for (size_t i = 0; i < num_axes; ++i) {
    isAllAxisReduce = isAllAxisReduce && (dst.vectorized_strides[i] == 0);
    const auto axis      = tpipe.tiler.GetAxis(src.vectorized_axis[i]);
    const auto axis_size = tpipe.tiler.AxisSize(src.vectorized_axis[i]);
    if (i == num_axes - 1U) {
      if (is_first && !isAllAxisReduce) {
        last << " * " << KernelUtils::SizeAlign() << "(" << axis_size << ", 32/sizeof(" << dtype_name << "))";
        last_actual << " * " << KernelUtils::SizeAlign() << "(" << axis.actual_size << ", 32/sizeof(" << dtype_name << "))";
      } else if (is_first && isAllAxisReduce) { // 这种情况最后会统一替换为last
        first << " * " << KernelUtils::SizeAlign() << "(" << axis_size << ", 32/sizeof(" << dtype_name << "))";
        first_actual << " * " << KernelUtils::SizeAlign() << "(" << axis.actual_size << ", 32/sizeof(" << dtype_name << "))";
      } else {
        last << " * " << tpipe.tiler.Size(lastNonZeroStride);
        last_actual << " * " << tpipe.tiler.Size(lastNonZeroStride);
      }
      break;
    }
    if (axis_size == "1") {
      continue;
    }
    if (is_first && last_not_1_axis_size_index != 0xFFFFFFFF) {
      is_first = !((dst.vectorized_strides[i] == 0 && dst.vectorized_strides[last_not_1_axis_size_index] != 0) ||
                (dst.vectorized_strides[i] != 0 && dst.vectorized_strides[last_not_1_axis_size_index] == 0));
    }
    if (!is_first) {
      if (src.vectorized_strides[i] != Zero) {
        lastNonZeroStride = src.vectorized_strides[i];
      }
      last << " * " << axis_size;
      last_actual << " * " << axis.actual_size;
    } else {
      first << " * " << axis_size;
      first_actual << " * " << axis.actual_size;
      last_not_1_axis_size_index = i;
    }
  }
  ReplaceSSWithSwappingFirstAndLast(first.str() + ";\n" + first_actual.str(), last.str() + ";\n" + last_actual.str(), isAllAxisReduce, ss);
}

bool IsNeedMultiReduce(const Tiler &tiler, const Tensor &input, const Tensor &output, ascir::AxisId axis_id) {
  int64_t total_count = 0;
  int64_t valid_count = 0;
  std::function<void(ascir::AxisId)> recursive_functor = [&tiler, &input, &output, &total_count,
                                                          &valid_count, &recursive_functor](ascir::AxisId id) {
    Axis axis = tiler.GetAxis(id);
    auto pos = std::find(output.axis.begin(), output.axis.end(), id);
    if (pos != output.axis.end()) {
      size_t diff = pos - output.axis.begin();
      total_count++;
      valid_count = output.axis_strides[diff] == Zero && input.axis_strides[diff] != Zero ? valid_count + 1 : valid_count;
      return;
    }
    for (size_t i = 0; i < axis.from.size(); i++) {
      auto from_axis = tiler.GetAxis(axis.from[i]);
      bool need_recursive = from_axis.type != Axis::Type::kAxisTypeOriginal;
      auto pos = std::find(output.axis.begin(), output.axis.end(), axis.from[i]);
      if (pos != output.axis.end()) {
        size_t diff = pos - output.axis.begin();
        total_count++;
        valid_count = output.axis_strides[diff] == Zero && input.axis_strides[diff] != Zero ? valid_count + 1 : valid_count;
        return;
      }
      if (need_recursive) {
        for (size_t j = 0; j < from_axis.from.size(); j++) {
          recursive_functor(from_axis.from[j]);
        }
      }
    }
  };
  recursive_functor(axis_id);
  return total_count == valid_count;
}

void ReduceMeanCodeGen(std::string &dtype_name, const TPipe &tpipe, const Tensor &dst,
                       std::stringstream &ss) {
  std::set<ascir::AxisId> r_from_axis;
  for (size_t i = 0; i < dst.axis_strides.size(); i++) {
    if (dst.axis_strides[i] == 0) {  // 如果目标张量的轴步长为0
      auto axis_id = dst.axis[i];  // 获取当前轴ID
      // 定义递归函数用于收集原始轴
      std::function<void(int)> collect_original_axes = [&tpipe, &r_from_axis, &collect_original_axes](int current_axis_id) {
        auto axis = tpipe.tiler.GetAxis(current_axis_id);  // 获取当前轴对象
        if (axis.type == ascir::Axis::Type::kAxisTypeOriginal) {
          r_from_axis.insert(current_axis_id);  // 如果是原始轴则加入集合
          return;
        }
        // 否则递归处理所有来源轴
        for (auto from_axis_id : axis.from) {
          collect_original_axes(from_axis_id);
        }
      };
      collect_original_axes(axis_id);  // 从当前轴开始递归收集
    }
  }
  ss << "const float dimr_recip = 1.0f / (";
  uint32_t count = 0;
  for (auto axis_id : r_from_axis) {
    if (count++ == 0) {
      ss << tpipe.tiler.AxisSize(axis_id);
    } else {
      ss << " * " << tpipe.tiler.AxisSize(axis_id);
    }
  }
  ss << ");" << std::endl;
  ss << "Muls(" << dst << ", " << dst << ", " << "dimr_recip, " << KernelUtils::SizeAlign() << "(" << "reduce_dim_a" << ", 32 / sizeof(" << dtype_name << ")));" << std::endl;
  return;
}

void GetIsArAndPattern(const Tensor &y, bool &isAr, std::string &reduce_pattern)
{
  isAr = (y.vectorized_strides.back() == 0);
  std::unordered_map<bool, std::string> reduce_pattern_map = {{true, "AscendC::Pattern::Reduce::AR"},
                                                              {false, "AscendC::Pattern::Reduce::RA"}};
  reduce_pattern = reduce_pattern_map[isAr];
  return;
}

bool IsTilerLastReduceAxis(const Tensor &tensor) {
  int count = 0;
  for (auto stride : tensor.vectorized_strides) {
    if (stride == 0) {
      count++;
    }
  }
  return count == 1;
}

void ReduceInitCodeGen(const Tensor &x, const Tensor &y, const int &type_value, std::stringstream &ss, const TPipe &tpipe)
{
  if (x.isAr) {
    std::string dtype_name;
    Tensor::DtypeName(y.dtype, dtype_name); // 调用处已经做过校验
    std::string is_last_axis_str = IsTilerLastReduceAxis(y) ? "true" : "false";
    ss << "ReduceInit<" << dtype_name << ", " << type_value << ", " << is_last_axis_str << ">("
      << x << ", " << "first_actual" << ", last" << ", last_actual, " << tpipe.tiler.GetAxis(x.vectorized_axis.back()).actual_size
      << ");" << std::endl;

    ss << "AscendC::PipeBarrier<PIPE_V>();" << std::endl;
  }
  return;
}

void ReduceDimACodeGen(const Tensor &x, const std::string &apiName, std::stringstream &ss)
{
  if (apiName == "Mean") {
    if (x.isAr) {
      ss << "reduce_dim_a = first_actual;" << std::endl;
    } else {
      ss << "reduce_dim_a = last_actual;" << std::endl;
    }
  }
  return;
}

}  // namespace codegen