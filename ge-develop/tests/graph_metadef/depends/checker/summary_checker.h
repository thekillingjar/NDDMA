/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_TESTS_DEPENDS_CHECKER_SUMMARY_CHECKER_H_
#define METADEF_CXX_TESTS_DEPENDS_CHECKER_SUMMARY_CHECKER_H_
#include <string>
#include <set>
#include <sstream>
#include <iostream>
#include <cstdlib>
#include "graph/compute_graph.h"
#include "graph/fast_graph/execute_graph.h"
#include "pretty_table.h"
namespace gert {
class SummaryChecker {
 public:
  explicit SummaryChecker(ge::ComputeGraphPtr graph) : graph_(std::move(graph)) {}

  std::string StrictAllNodeTypes(const std::map<std::string, size_t> &node_types_to_count) {
    return StrictNodeTypes(graph_->GetAllNodes(), node_types_to_count);
  }

  std::string StrictDirectNodeTypes(const std::map<std::string, size_t> &node_types_to_count) {
    return StrictNodeTypes(graph_->GetDirectNode(), node_types_to_count);
  }

 private:
  template<typename T>
  std::string StrictNodeTypes(const T &nodes, const std::map<std::string, size_t> &node_types_to_count) {
    std::map<std::string, size_t> actual_node_types_to_count;
    for (const auto &node : nodes) {
      actual_node_types_to_count[node->GetType()]++;
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

class ExeGraphSummaryChecker {
 public:
  explicit ExeGraphSummaryChecker(ge::ExecuteGraph *graph) : graph_(graph) {}

  std::string StrictAllNodeTypes(const std::map<std::string, size_t> &node_types_to_count) {
    return StrictNodeTypes(graph_->GetAllNodes(), node_types_to_count);
  }

  std::string StrictDirectNodeTypes(const std::map<std::string, size_t> &node_types_to_count) {
    return StrictNodeTypes(graph_->GetDirectNode(), node_types_to_count);
  }

 private:
  template<typename T>
  std::string StrictNodeTypes(const T &nodes, const std::map<std::string, size_t> &node_types_to_count) {
    std::map<std::string, size_t> actual_node_types_to_count;
    for (const auto &node : nodes) {
      actual_node_types_to_count[node->GetType()]++;
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
  ge::ExecuteGraph *graph_;
};
}  // namespace gert

#define STRICT_DIRECT_NODE_TYPES(graph, expect_types) auto ret = gert::SummaryChecker(graph).StrictDirectNodeTypes(expect_types); EXPECT_TRUE(ret == "success") << ret
#define STRICT_ALL_NODE_TYPES(graph, ...) auto ret = gert::SummaryChecker(graph).StrictAllNodeTypes(##__VA_ARGS__); EXPECT_TRUE(ret == "success") << ret

#define EXE_GRAPH_STRICT_DIRECT_NODE_TYPES(graph, expect_types) auto ret = gert::ExeGraphSummaryChecker(graph).StrictDirectNodeTypes(expect_types); EXPECT_TRUE(ret == "success") << ret
#define EXE_GRAPH_STRICT_ALL_NODE_TYPES(graph, ...) auto ret = gert::ExeGraphSummaryChecker(graph).StrictAllNodeTypes(##__VA_ARGS__); EXPECT_TRUE(ret == "success") << ret

#endif  //METADEF_CXX_TESTS_DEPENDS_CHECKER_SUMMARY_CHECKER_H_
