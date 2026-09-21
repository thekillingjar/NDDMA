/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_COMMON_SUMMARY_CHECKER_H_
#define AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_COMMON_SUMMARY_CHECKER_H_
#include <string>
#include <set>
#include <sstream>
#include <iostream>
#include <cstdlib>
#include "graph/compute_graph.h"
#include "graph/fast_graph/execute_graph.h"
#include "pretty_table.h"
#include "graph/ascendc_ir/ascendc_ir_core/ascendc_ir.h"

namespace gert {
class SummaryChecker {
 public:
  SummaryChecker() = default;
  explicit SummaryChecker(ge::ComputeGraphPtr graph) : graph_(std::move(graph)) {}

  virtual std::string StrictAllNodeTypes(const std::map<std::string, size_t> &node_types_to_count) const {
    return StrictNodeTypes(graph_->GetAllNodes(), node_types_to_count);
  }

  virtual std::string StrictDirectNodeTypes(const std::map<std::string, size_t> &node_types_to_count) const {
    return StrictNodeTypes(graph_->GetDirectNode(), node_types_to_count);
  }

  virtual std::string DirectNodeTypesOnlyCareAbout(const std::map<std::string, size_t> &node_types_to_count) const {
    return LooseNodeTypes(graph_->GetDirectNode(), node_types_to_count);
  }

 protected:
  template<typename T>
  std::string StrictNodeTypes(const T &nodes, const std::map<std::string, size_t> &node_types_to_count) const {
    std::map<std::string, size_t> actual_node_types_to_count;
    for (const auto &node : nodes) {
      actual_node_types_to_count[node->GetType()]++;
    }
    if (actual_node_types_to_count != node_types_to_count) {
      return PrintDiff(actual_node_types_to_count, node_types_to_count);
    }
    return "success";
  }

  template<typename T>
  std::string LooseNodeTypes(const T &nodes, const std::map<std::string, size_t> &node_types_to_count) const {
    std::map<std::string, size_t> actual_node_types_to_count;
    for (const auto &node : nodes) {
      if (node_types_to_count.count(node->GetType()) > 0) {
        actual_node_types_to_count[node->GetType()]++;
      }
    }
    if (actual_node_types_to_count != node_types_to_count) {
      return PrintDiff(actual_node_types_to_count, node_types_to_count);
    }
    return "success";
  }
  static std::string PrintDiff(const std::map<std::string, size_t> &actual_types_to_num,
                               const std::map<std::string, size_t> &expect_types_to_num) {
    PrettyTable pt;
    pt.SetHeader({"Actual Type", "Actual Num", "Expect Type", "Expect Num"});
    auto actual_iter = actual_types_to_num.begin();
    auto expect_iter = expect_types_to_num.begin();
    while (actual_iter != actual_types_to_num.end() || expect_iter != expect_types_to_num.end()) {
      std::string actual_type = "-";
      std::string actual_num = "-";
      std::string expect_type = "-";
      std::string expect_num = "-";
      bool same_row = false;
      if (actual_iter != actual_types_to_num.end() && expect_iter != expect_types_to_num.end()) {
        if (actual_iter->first == expect_iter->first) {
          actual_type = actual_iter->first;
          actual_num = std::to_string(actual_iter->second);
          expect_type = expect_iter->first;
          expect_num = std::to_string(expect_iter->second);
          same_row = (*actual_iter == *expect_iter);
          ++actual_iter, ++expect_iter;
        } else {
          if (actual_iter->first < expect_iter->first) {
            actual_type = actual_iter->first;
            actual_num = std::to_string(actual_iter->second);
            ++actual_iter;
          } else {
            expect_type = expect_iter->first;
            expect_num = std::to_string(expect_iter->second);
            ++expect_iter;
          }
        }
      } else if (actual_iter == actual_types_to_num.end()) {
        expect_type = expect_iter->first;
        expect_num = std::to_string(expect_iter->second);
        ++expect_iter;
      } else if (expect_iter == expect_types_to_num.end()) {
        actual_type = actual_iter->first;
        actual_num = std::to_string(actual_iter->second);
        ++actual_iter;
      } else {
        // 两个都到end了
        throw std::exception();
      }
      if (same_row) {
        pt.AddRow({actual_type, actual_num, expect_type, expect_num});
      } else {
        pt.AddColorRow({actual_type, actual_num, expect_type, expect_num});
      }
    }

    std::stringstream ss;
    pt.Print(ss);
    return ss.str();
  }

 private:
  ge::ComputeGraphPtr graph_;
};

class ExeGraphSummaryChecker : public SummaryChecker{
 public:
  explicit ExeGraphSummaryChecker(ge::ExecuteGraph *graph) : graph_(graph) {}

  std::string StrictAllNodeTypes(const std::map<std::string, size_t> &node_types_to_count) const override {
    return StrictNodeTypes(graph_->GetAllNodes(), node_types_to_count);
  }

  std::string StrictDirectNodeTypes(const std::map<std::string, size_t> &node_types_to_count) const override {
    return StrictNodeTypes(graph_->GetDirectNode(), node_types_to_count);
  }

  std::string DirectNodeTypesOnlyCareAbout(const std::map<std::string, size_t> &node_types_to_count) const override {
    return LooseNodeTypes(graph_->GetDirectNode(), node_types_to_count);
  }

 private:
  ge::ExecuteGraph *graph_;
};

class AscGraphSummaryChecker : public SummaryChecker {
 public:
  explicit AscGraphSummaryChecker(const std::shared_ptr<ge::AscGraph> &graph) : graph_(graph) {
    for (const auto &node : graph_->GetAllNodes()) {
      nodes_.push_back(node);
    }
  }

  std::string StrictAllNodeTypes(const std::map<std::string, size_t> &node_types_to_count) const override {
    // AscGraph提供的GetAllNodes()方法返回的是AscNodeVisitor，AscNodeVisitor未提供const迭代器
    return StrictNodeTypes(nodes_, node_types_to_count);
  }

  std::string StrictDirectNodeTypes(const std::map<std::string, size_t> &node_types_to_count) const override {
    return StrictNodeTypes(nodes_, node_types_to_count);
  }

  std::string DirectNodeTypesOnlyCareAbout(const std::map<std::string, size_t> &node_types_to_count) const override {
    return LooseNodeTypes(nodes_, node_types_to_count);
  }

 private:
  const std::shared_ptr<ge::AscGraph> &graph_;
  std::vector<ge::NodePtr> nodes_;
};
}  // namespace gert

#define STRICT_DIRECT_NODE_TYPES(graph, expect_types)                         \
  auto ret = gert::SummaryChecker(graph).StrictDirectNodeTypes(expect_types); \
  EXPECT_TRUE(ret == "success") << ret
#define STRICT_ALL_NODE_TYPES(graph, ...)                                   \
  auto ret = gert::SummaryChecker(graph).StrictAllNodeTypes(##__VA_ARGS__); \
  EXPECT_TRUE(ret == "success") << ret

#define EXE_GRAPH_STRICT_DIRECT_NODE_TYPES(graph, expect_types)                           \
  auto ret = gert::ExeGraphSummaryChecker(graph).StrictDirectNodeTypes(expect_types); \
  EXPECT_TRUE(ret == "success") << ret
#define EXE_GRAPH_STRICT_ALL_NODE_TYPES(graph, ...)                                     \
  auto ret = gert::ExeGraphSummaryChecker(graph).StrictAllNodeTypes(##__VA_ARGS__); \
  EXPECT_TRUE(ret == "success") << ret

#endif  // AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_COMMON_SUMMARY_CHECKER_H_
