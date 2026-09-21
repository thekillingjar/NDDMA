/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_COMPILER_GRAPH_OPTIMIZE_AUTOFUSE_LOWERING_LOOP_COMMON_H_
#define AIR_CXX_COMPILER_GRAPH_OPTIMIZE_AUTOFUSE_LOWERING_LOOP_COMMON_H_
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "common/checker.h"
#include "graph/node.h"
#include "graph/ge_tensor.h"
#include "graph/symbolizer/symbolic.h"
#include "graph/attribute_group/attr_group_symbolic_desc.h"
#include "graph/utils/type_utils.h"

namespace ge {
namespace loop {
/**
 * LoopAxis用于表达一个Loop lowering后的Ascend IR输出值的坐标轴
 */
struct LoopAxis {
  std::vector<std::string> names;   // 轴的名字
  std::vector<Expression> axis;     // 轴的循环变量，类似for (int i = 0; i < N; i++) 中的i
  std::vector<Expression> repeats;  // 轴的遍历范围，类似for (int i = 0; i < N; i++) 中的N

  static LoopAxis FromDims(const std::vector<Expression> &dims, const std::string &prefix = "z") {
    LoopAxis loop_axis;
    for (size_t i = 0U; i < dims.size(); i++) {
      std::string axis_name = prefix + std::to_string(i);
      auto axis = ge::Symbol(axis_name.c_str());
      loop_axis.names.push_back(axis_name);
      loop_axis.axis.push_back(axis);
      loop_axis.repeats.push_back(dims[i]);
    }
    return loop_axis;
  }

  friend std::ostream &operator<<(std::ostream &os, const LoopAxis &obj) {
    os << "[";
    for (size_t i = 0U; i < obj.axis.size(); i++) {
      os << obj.axis[i];
      if (i + 1U < obj.axis.size()) {
        os << ", ";
      }
    }
    os << "]";
    return os;
  }

 private:
  LoopAxis() = default;
};

/**
 * Index用于表达一个Loop lowering后的Ascend IR输出值的坐标，其长度与该Adscend IR的rank一致。
 * 并且Index的每个元素是一个Expression，用于表达一个坐标的某个维度的值。值的范围a[i] ∈ [0, shape[i]).
 */
using Index = std::vector<ge::Expression>;

/**
 * TensorLoopDesc用于表达一个Loop lowering后的Ascend IR输出值的循环描述，其包含了一个Loop的所有信息。
 * 包括循环的遍历范围、循环的步长。
 */
struct TensorLoopDesc {
  TensorLoopDesc(std::vector<Expression> repeats, std::vector<Expression> strides)
      : repeats(std::move(repeats)), strides(std::move(strides)) {}
  std::vector<Expression> repeats;
  std::vector<Expression> strides;

  friend std::ostream &operator<<(std::ostream &os, const TensorLoopDesc &obj) {
    os << "Loop{";
    os << "repeats=[";
    for (size_t i = 0U; i < obj.repeats.size(); i++) {
      os << obj.repeats[i];
      if (i + 1U < obj.repeats.size()) {
        os << ", ";
      }
    }
    os << "], strides=[";
    for (size_t i = 0U; i < obj.strides.size(); i++) {
      os << obj.strides[i];
      if (i + 1U < obj.strides.size()) {
        os << ", ";
      }
    }
    os << "]}";
    return os;
  }
};

/**
 * Lowering表达的执行过程中流转的变量，它是一个基类，不同的Lowering目标会继承并实现自己的Var子类类型，
 * 例如AscendC IR的Var类型，用于表示AscendC IR的变量，其会持有一个AscendC IR的输出类型作为成员。
 */
class Var {
 public:
  virtual ~Var() = default;
  virtual std::string Name() = 0;
  virtual bool IsValid() const = 0;
};
using VarPtr = std::shared_ptr<Var>;

/**
 * CseVar是一个Var的包装类，用于为Loop过程流转的Var变量提供公共的CSE能力，其持有一个Var的智能指针来支持不同类型的Var。
 */
class CseVar {
 public:
  CseVar() = default;
  explicit CseVar(VarPtr var) : var_(std::move(var)) {}

  template <typename T>
  auto Get() const -> std::shared_ptr<T> {
    return std::dynamic_pointer_cast<T>(var_);
  }

  bool operator==(const CseVar &other) const {
    return var_ == other.var_;
  }

  bool operator!=(const CseVar &other) const {
    return var_ != other.var_;
  }

  [[nodiscard]] bool IsValid() const {
    return var_ != nullptr && var_->IsValid();
  }

  friend std::ostream &operator<<(std::ostream &os, const CseVar &obj) {
    os << obj.Name();
    return os;
  }

  [[nodiscard]] std::string Name() const {
    std::stringstream ss;
    ss << "Cse(" << (var_ ? var_->Name() : "NULL") << ")";
    return ss.str();
  }

 private:
  VarPtr var_ = nullptr;
};

/**
 * LoopOp是一个IR表达，用于表达一个Ascend IR表达为Loop时的Scalar计算行为。其屏蔽了AscendC IR等低级IR的调用细节，
 * 使得实现Ascend IR时只需要关注高层次的Scalar计算逻辑。
 */
class LoopOp;
using LoopOpPtr = std::shared_ptr<LoopOp>;

/**
 * LoopKernel是一个函数类型签名，用于表达一个LoopOp的计算逻辑。
 */
using LoopKernel = std::function<CseVar(const std::vector<CseVar> &)>;
using PrintKernel = std::function<std::string(const std::vector<std::string> &)>;
using InferDtypeKernel = std::function<bool(const std::vector<DataType> &, std::vector<DataType> &)>;

/**
 * LowerFunc用于表达一个Ascend IR的Lowering执行逻辑，其输入是一个NodePtr，输出是一个graphStatus。
 */
using LowerFunc = std::function<graphStatus(const ge::NodePtr &node)>;

enum class ReduceType : int32_t {
  SUM,
  MEAN,
  MAX,
  MIN,
  PROD,
  ANY,
  ALL,
  END
};

inline std::string ReduceTypeToString(ReduceType type) {
  switch (type) {
    case ReduceType::SUM:
      return "Sum";
    case ReduceType::MEAN:
      return "Mean";
    case ReduceType::MAX:
      return "Max";
    case ReduceType::MIN:
      return "Min";
    case ReduceType::PROD:
      return "Prod";
    case ReduceType::ANY:
      return "Any";
    case ReduceType::ALL:
      return "All";
    default:
      return "Invalid";
  }
}

inline std::ostream& operator<<(std::ostream& os, ReduceType type) {
  os << ReduceTypeToString(type);
  return os;
}

enum class FuseType : int32_t {
  kDefault = 0,
  kPointwise,
  kReduction,
  kConcat,
  kSplit,
  kSliceSplit,
  kGather,
  kTranspose,
  kCube,
  kExtern
};

inline std::string FuseTypeToString(FuseType type) {
  switch (type) {
    case FuseType::kDefault:
      return "default";
    case FuseType::kPointwise:
      return "pointwise";
    case FuseType::kReduction:
      return "reduce";
    case FuseType::kConcat:
      return "concat";
    case FuseType::kSplit:
      return "split";
    case FuseType::kSliceSplit:
      return "slice";
    case FuseType::kGather:
      return "gather";
    case FuseType::kTranspose:
      return "transpose";
    case FuseType::kCube:
      return "cube";
    default:
      return "extern";
  }
}

// some common functions
template <typename T, typename F>
std::string StrJoin(const std::vector<T> &vec, F f, const std::string &sep = ", ") {
  if (vec.empty()) {
    return "[]";
  }
  std::string res = "[" + f(vec[0]);
  for (size_t i = 1U; i < vec.size(); ++i) {
    res += sep + f(vec[i]);
  }
  return res + "]";
}

inline std::string StrJoin(const std::vector<std::string> &vec, const std::string &sep = ", ") {
  return StrJoin(vec, [](const string &s) { return s; }, sep);
}

inline std::string StrJoin(const std::vector<Expression> &vec, const std::string &sep = ", ") {
  return StrJoin(vec, [](const Expression &s) { return std::string(s.Str().get()); }, sep);
}

inline std::string StrJoin(const std::vector<ge::DataType> &vec, const std::string &sep = ", ") {
  return StrJoin(vec, [](const ge::DataType &s) { return TypeUtils::DataTypeToSerialString(s); }, sep);
}

std::string BufferName(const AnchorPtr &buffer);
std::string BufferName(const Anchor *anchor);
graphStatus GetBufferShape(const OutDataAnchorPtr &buffer, std::vector<Expression> &dims);
graphStatus GetBufferShape(const InDataAnchorPtr &buffer, std::vector<Expression> &dims);
graphStatus GetBufferShape(const OutDataAnchor *buffer, std::vector<Expression> &dims);
graphStatus GetBufferDataType(const InDataAnchor *buffer, DataType &data_type);
ConstGeTensorDescPtr GetBufferDesc(const OutDataAnchor *buffer);
void GetStrideAndOffset(const Expression &expr, const std::vector<Expression> &symbols,
                        std::vector<Expression> &strides, Expression &offset);
std::vector<Expression> ContiguousStrides(const std::vector<Expression> &dims);
}  // namespace loop
}  // namespace ge

#endif  // AIR_CXX_COMPILER_GRAPH_OPTIMIZE_AUTOFUSE_LOWERING_LOOP_COMMON_H_
