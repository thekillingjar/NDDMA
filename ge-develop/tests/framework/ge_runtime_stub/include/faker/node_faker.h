/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_FAKER_NODE_FAKER_H_
#define AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_FAKER_NODE_FAKER_H_
#include "graph/node.h"
#include "graph/compute_graph.h"
#include "graph/fast_graph/execute_graph.h"
namespace gert {
class ComputeNodeFaker {
 public:
  ComputeNodeFaker() : graph_(std::make_shared<ge::ComputeGraph>("TempGraph")) {}
  explicit ComputeNodeFaker(ge::ComputeGraphPtr graph) : graph_(std::move(graph)) {}
  ComputeNodeFaker &IoNum(size_t input_num, size_t output_num, ge::DataType data_type = ge::DT_FLOAT);
  ComputeNodeFaker &InputNames(std::vector<std::string> names);
  ComputeNodeFaker &OutputNames(std::vector<std::string> names);
  ComputeNodeFaker &NameAndType(std::string name, std::string type);
  template <typename T>
  ComputeNodeFaker &Attr(const char *key, const T &value) {
    attr_keys_to_value_[key] = ge::AnyValue::CreateFrom<T>(value);
    return *this;
  }
  ge::NodePtr Build();

 private:
  ge::ComputeGraphPtr graph_;
  std::string name_;
  std::string type_;
  std::vector<ge::GeTensorDesc> inputs_desc_;
  std::vector<ge::GeTensorDesc> outputs_desc_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::map<std::string, ge::AnyValue> attr_keys_to_value_;
};

class FastNodeFaker {
 public:
  FastNodeFaker() : graph_(std::make_shared<ge::ExecuteGraph>("TempGraph")) {}
  explicit FastNodeFaker(ge::ExecuteGraphPtr graph) : graph_(std::move(graph)) {}
  FastNodeFaker &IoNum(size_t input_num, size_t output_num, ge::DataType data_type = ge::DT_FLOAT);
  FastNodeFaker &InputNames(std::vector<std::string> names);
  FastNodeFaker &OutputNames(std::vector<std::string> names);
  FastNodeFaker &NameAndType(std::string name, std::string type);
  template <typename T>
  FastNodeFaker &Attr(const char *key, const T &value) {
    attr_keys_to_value_[key] = ge::AnyValue::CreateFrom<T>(value);
    return *this;
  }
  ge::FastNode *Build();

 private:
  ge::ExecuteGraphPtr graph_;
  std::string name_;
  std::string type_;
  std::vector<ge::GeTensorDesc> inputs_desc_;
  std::vector<ge::GeTensorDesc> outputs_desc_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::map<std::string, ge::AnyValue> attr_keys_to_value_;
};
}  // namespace gert
#endif  // AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_FAKER_NODE_FAKER_H_
