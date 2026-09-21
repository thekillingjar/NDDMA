/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_ESB_FUNCS_CPP_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_ESB_FUNCS_CPP_H_

#include "all_ops.h"
#include <memory>
#include <stdexcept>
#include "graph/graph.h"

namespace es {
class Tensor {
 public:
  Tensor(EsbTensor *tensor) : tensor_(tensor) {}

  Tensor &SetShape(const std::vector<int64_t> &dims) {
    EsSetShape(tensor_, dims.data(), static_cast<int64_t>(dims.size()));
    return *this;
  }
  Tensor &SetSymbolShape(const std::vector<const char *> &dims) {
    EsSetSymbolShape(tensor_, dims.data(), static_cast<int64_t>(dims.size()));
    return *this;
  }

  Tensor &SetInputSymbolShape(const std::vector<const char *> &dims) {
    EsSetInputSymbolShape(tensor_, dims.data(), static_cast<int64_t>(dims.size()));
    return *this;
  }

  Tensor operator+(const Tensor &other) const {
    return EsAdd(tensor_, other.tensor_);
  }

  Tensor operator-(const Tensor &other) const {
    return EsSub(tensor_, other.tensor_);
  }

  Tensor operator*(const Tensor &other) const {
    return EsMul(tensor_, other.tensor_);
  }

  Tensor operator/(const Tensor &other) const {
    return EsDiv(tensor_, other.tensor_);
  }

  EsbTensor *GetEsbTensor() const {
    return tensor_;
  }

 private:
  friend class Graph;
  EsbTensor *tensor_;
};

class Graph {
 public:
  explicit Graph(const char *name) : graph_(EsCreateGraph(name), EsDestroyGraph) {}
  Tensor CreateInput(int index, const char *name, const char *type) {
    return EsCreateGraphInputWithDetails(graph_.get(), index, name, type);
  }
  Tensor CreateInput(int index) {
    return EsCreateGraphInput(graph_.get(), index);
  }
  Tensor CreateInput(int index, const char *name) {
    return EsCreateGraphInputWithDetails(graph_.get(), index, name, nullptr);
  }
  Tensor CreateVector(const std::vector<int64_t> &value) {
    return EsCreateVectorInt64(graph_.get(), value.data(), static_cast<int64_t>(value.size()));
  }
  Tensor CreateScalar(int64_t value) {
    return EsCreateScalarInt64(graph_.get(), value);
  }
  Tensor CreateScalar(int32_t value) {
    return EsCreateScalarInt32(graph_.get(), value);
  }
  // 直接写字面值常量的时候，C++会将其默认为推导为`double`，但是一般构图这么做是不合理的，因此需要显式指定类型
  // 这也有额外的问题，如果构图真的希望使用double，那么需要手动创建`double`类型的Const
  Tensor CreateScalar(double value) {
    throw std::invalid_argument("Double is not support");
    return EsCreateScalarDouble(graph_.get(), static_cast<float>(value));
  }
  Tensor CreateScalar(float value) {
    return EsCreateScalarFloat(graph_.get(), value);
  }
  Tensor CreateVariable(int32_t index, const int64_t *value, const std::vector<int64_t> &dims,
                        const std::string &container = "", const std::string &shared_name = "") {
    return EsCreateVariableInt64(graph_.get(), index, value, dims.data(), dims.size(), container.c_str(),
                                 shared_name.c_str());
  }
  Tensor CreateVariable(int32_t index, const int32_t *value, const std::vector<int64_t> &dims,
                        const std::string &container = "", const std::string &shared_name = "") {
    return EsCreateVariableInt32(graph_.get(), index, value, dims.data(), dims.size(), container.c_str(),
                                 shared_name.c_str());
  }
  Tensor CreateVariable(int32_t index, const float *value, const std::vector<int64_t> &dims,
                        const std::string &container = "", const std::string &shared_name = "") {
    return EsCreateVariableFloat(graph_.get(), index, value, dims.data(), dims.size(), container.c_str(),
                                 shared_name.c_str());
  }
  int SetOutput(const Tensor &tensor, int index) {
    return EsSetGraphOutput(tensor.tensor_, index);
  }
  std::unique_ptr<ge::Graph> Build() const {
    return std::unique_ptr<ge::Graph>(static_cast<ge::Graph *>(EsBuildGraph(graph_.get())));
  }
  std::unique_ptr<ge::Graph> Build(const std::vector<Tensor> &outputs) {
    for (size_t i = 0U; i < outputs.size(); ++i) {
      SetOutput(outputs[i], static_cast<int>(i));
    }
    return Build();
  }
  EsbGraph *GetEsbGraph() const {
    return graph_.get();
  }
  static Graph WrapFromEsb(EsbGraph *esb_graph) {
    return Graph{esb_graph};
  }

 private:
  static void DoNothing(EsbGraph *) {}
  explicit Graph(EsbGraph *esb_graph) : graph_(esb_graph, Graph::DoNothing) {}
  std::unique_ptr<EsbGraph, decltype(&EsDestroyGraph)> graph_;
};

// todo 方法使用了if constexpr，因此需要C++17支持，如果不希望引入C++17，可以使用穷举特化的方式，会罗嗦一些
template<typename T>
Tensor EnsureTensor(T &&value, const Tensor &same_graph_tensor) {
  const auto graph = EsGetOwnerGraph(same_graph_tensor.GetEsbTensor());
  if (graph == nullptr) {
    return nullptr;
  }
  using DecayedT = typename std::decay<T>::type;
  if constexpr (std::is_same<DecayedT, double>::value) {
    return Graph::WrapFromEsb(graph).CreateScalar(static_cast<float>(value));
  } else {
    return Graph::WrapFromEsb(graph).CreateScalar(value);
  }
}
inline std::vector<EsbTensor *> TensorsToEsbTensors(const std::vector<Tensor> &tensors) {
  std::vector<EsbTensor *> esb_tensors;
  for (const auto &tensor : tensors) {
    esb_tensors.push_back(tensor.GetEsbTensor());
  }
  return esb_tensors;
}
}  // namespace es

#endif  // AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_ESB_FUNCS_CPP_H_
