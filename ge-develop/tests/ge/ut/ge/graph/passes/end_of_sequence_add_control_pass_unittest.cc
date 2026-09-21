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
#include "macro_utils/dt_public_scope.h"
#include "api/gelib/gelib.h"
#include "macro_utils/dt_public_unscope.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/passes/standard_optimize/end_of_sequence_add_control_pass.h"

namespace ge {
class UtestEndOfSequenceAddControlPass : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

namespace {
const char *const kEngineName = "DNN_VM_GE_LOCAL";
void InitGeLib() {
  map<string, string> options;
  Status ret = ge::GELib::Initialize(options);
  EXPECT_EQ(ret, SUCCESS);
  auto instance_ptr = ge::GELib::GetInstance();
  EXPECT_NE(instance_ptr, nullptr);

  //  SchedulerConf conf;
  SchedulerConf scheduler_conf;
  scheduler_conf.name = kEngineName;
  scheduler_conf.cal_engines[kEngineName] = std::make_shared<EngineConf>();
  scheduler_conf.cal_engines[kEngineName]->name = kEngineName;
  scheduler_conf.cal_engines[kEngineName]->scheduler_id = kEngineName;
  scheduler_conf.cal_engines[kEngineName]->skip_assign_stream = true;
  map<string, SchedulerConf> scheduler_confs;
  scheduler_confs["scheduler"] = scheduler_conf;
  instance_ptr->DNNEngineManagerObj().schedulers_[kEngineName] = scheduler_conf;
}

/**
 *      netoutput
 *          |
 *        addn
 *          |
 *     identity  end_of_seq
 *          |      |
 *         IteratorV2
 */
static ComputeGraphPtr BuildGraph() {
  const auto IteratorV2 = OP_CFG(FRAMEWORKOP).Attr(ATTR_NAME_STREAM_LABEL, "label");
  DEF_GRAPH(g1) {
    CHAIN(NODE("IteratorV2", IteratorV2)->EDGE(0, 0)->NODE("identity", IDENTITY));
    CHAIN(NODE("IteratorV2", IteratorV2)->EDGE(0, 0)->NODE("end_of_seq", ENDOFSEQUENCE));
    CHAIN(NODE("identity", IDENTITY)->EDGE(0, 0)->NODE("addn", ADDN));
    CHAIN(NODE("addn", ADDN)->EDGE(0, 0)->NODE("net_output", NETOUTPUT));
  };
  const auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetDirectNode()) {
    node->GetOpDesc()->SetOpEngineName("stub_engine");
    if (node->GetType() == IDENTITY) {
      node->GetOpDesc()->SetOpEngineName(kEngineName);
    }
  }
  return graph;
}
} // namespace

TEST_F(UtestEndOfSequenceAddControlPass, test_normal_succ) {
  const auto graph = BuildGraph();
  InitGeLib();
  EndOfSequenceAddControlPass pass;
  auto ret = pass.Run(graph);
  EXPECT_EQ(ret, SUCCESS);
  const auto end_of_seq = graph->FindFirstNodeMatchType(ENDOFSEQUENCE);
  ASSERT_NE(end_of_seq, nullptr);
  EXPECT_EQ(end_of_seq->GetOutControlNodes().size(), 1);
  ge::GELib::GetInstance()->Finalize();
}

TEST_F(UtestEndOfSequenceAddControlPass, test_no_need_add_ctrl) {
  DEF_GRAPH(g2) {
    CHAIN(NODE("var1", VARIABLE)->EDGE(0, 0)->NODE("add1", ADD));
    CHAIN(NODE("var2", VARIABLE)->EDGE(0, 1)->NODE("add1", ADD));
    CHAIN(NODE("add1", ADD)->EDGE(0, 0)->NODE("net_output", NETOUTPUT));
  };
  const auto graph = ToComputeGraph(g2);
  EndOfSequenceAddControlPass pass;
  auto ret = pass.Run(graph);
  EXPECT_EQ(ret, SUCCESS);

  graph->SetParentGraph(BuildGraph());
  ret = pass.Run(graph);
  EXPECT_EQ(ret, SUCCESS);
}
}  // namespace ge
