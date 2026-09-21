/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <string>
#include "graph/utils/graph_dump_utils.h"
#include "graph/ge_context.h"
#include "graph_metadef/graph/utils/file_utils.h"
#include "graph_builder_utils.h"
#include "common/share_graph.h"
#include "mmpa/mmpa_api.h"

namespace ge {
std::stringstream GetFilePathWhenDumpPathSet(const string &ascend_work_path) {
  std::stringstream dump_file_path;
  dump_file_path << ascend_work_path << "/pid_" << mmGetPid() << "_deviceid_" << GetContext().DeviceId() << "/";
  return dump_file_path;
}
namespace {
std::string GetSpecificFilePath(const std::string &file_path, const string &suffix) {
  DIR *dir;
  struct dirent *ent;
  dir = opendir(file_path.c_str());
  if (dir == nullptr) {
    return "";
  }
  while ((ent = readdir(dir)) != nullptr) {
    if (strstr(ent->d_name, suffix.c_str()) != nullptr) {
      std::string d_name(ent->d_name);
      closedir(dir);
      return file_path + "/" + d_name;
    }
  }
  closedir(dir);
  return "";
}
} // namespace
using ExecuteSharedGraph = SharedGraph<ExecuteGraphPtr, ut::ExecuteGraphBuilder>;
class UtestGraphDumpUtils : public testing::Test {
 public:
  static void SetUpTestCase() {
    dump_graph_path_ = "./test_ge_graph_path";
    mmSetEnv("DUMP_GE_GRAPH", "1", 1);
    mmSetEnv("DUMP_GRAPH_LEVEL", "1", 1);
    mmSetEnv("DUMP_GRAPH_PATH", dump_graph_path_.c_str(), 1);
  }
  static void TearDownTestCase() {
    unsetenv("DUMP_GE_GRAPH");
    unsetenv("DUMP_GRAPH_LEVEL");
    unsetenv("DUMP_GRAPH_PATH");
    system(("rm -rf " + dump_graph_path_).c_str());
  }

  static std::string dump_graph_path_;
};
std::string UtestGraphDumpUtils::dump_graph_path_;

TEST_F(UtestGraphDumpUtils, DumpGraph_Ok_with_execute_graph) {
  auto exe_graph = ExecuteSharedGraph::BuildGraphWithSubGraph();
  DumpGraph(exe_graph.get(), "exe_graph_test");

  std::stringstream dump_file_path = GetFilePathWhenDumpPathSet(dump_graph_path_);
  std::string file_path = ge::RealPath(dump_file_path.str().c_str());
  // root graph
  ComputeGraphPtr compute_graph = std::make_shared<ComputeGraph>("GeTestGraph");
  ASSERT_EQ(GraphUtils::LoadGEGraph(GetSpecificFilePath(file_path, "_exe_graph_test.txt").c_str(), *compute_graph), true);
  ASSERT_EQ(compute_graph->GetDirectNodesSize(), 5);
  ASSERT_EQ(compute_graph->GetAllSubgraphs().size(), 2);
  // sub graph 0 should not dump
  ComputeGraphPtr subgraph_0 = std::make_shared<ComputeGraph>("subgraph_0");
  ASSERT_EQ(GraphUtils::LoadGEGraph(GetSpecificFilePath(file_path, "_exe_graph_test_sub_graph_0.txt").c_str(), *subgraph_0), false);
  // sub graph 1 should not dump
  ComputeGraphPtr subgraph_1 = std::make_shared<ComputeGraph>("subgraph_1");
  ASSERT_EQ(GraphUtils::LoadGEGraph(GetSpecificFilePath(file_path, "_exe_graph_test_sub_graph_1.txt").c_str(), *subgraph_1), false);

  system(("rm -rf " + dump_graph_path_).c_str());
}
}  // namespace ge
