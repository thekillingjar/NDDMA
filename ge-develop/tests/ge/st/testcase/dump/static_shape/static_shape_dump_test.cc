/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dump/utils/dump_test_fixture.h"
#include "dump/utils/dump_session_wrapper.h"
#include "dump/utils/dump_config_builder.h"
#include "ge/ut/ge/test_tools_task_info.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "engine/aicore/fe_rt2_common.h"

using namespace ge;
using StaticShapeDumpST = DumpST<false>;

namespace ge {
extern void SetAicAivOpKernel(const ComputeGraphPtr &, const std::string, TBEKernelStore *);
}

namespace {
Status GenerateTaskForHccl(const Node &node, RunContext &, std::vector<domi::TaskDef> &tasks) {
  domi::TaskDef task_def;
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_HCCL));
  task_def.set_stream_id(0);
  auto &kernel_hccl_def = *task_def.mutable_kernel_hccl();
  auto op_desc = node.GetOpDesc();
  kernel_hccl_def.set_hccl_type(op_desc->GetTypePtr());
  kernel_hccl_def.set_op_index(op_desc->GetId());
  tasks.emplace_back(task_def);
  return SUCCESS;
}

//             (0,0)
//   +-----------------------------------------------+
//   |                                               v
// +---------+  (0,0)   +---------------+  (0,1)   +--------+  (0,0)   +-----------+
// | _data_0 | -------> | _all_reduce_0 | -------> | _add_0 | -------> | _output_0 |
// +---------+          +---------------+          +--------+          +-----------+
//                        ^
//                        | (0,1)
//                        |
//                      +---------------+
//                      |    _data_1    |
//                      +---------------+
GraphAndIoNum BuildGraph_HcclAdd() {
  MockForGenerateTask("ops_kernel_info_hccl", GenerateTaskForHccl);

  auto data_0 = OP_CFG(DATA).InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0)
                            .TensorDesc(FORMAT_ND, DT_FLOAT, {kTensorDim}).Build("_data_0");
  auto data_1 = OP_CFG(DATA).InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 1)
                            .TensorDesc(FORMAT_ND, DT_BOOL, {1}).Build("_data_1");
  auto hcom_0 = OP_CFG(HCOMALLREDUCE).InCnt(2).OutCnt(2)
                                     .Attr(HCOM_ATTR_REDUCE_TYPE, "sum")
                                     .Attr(HCOM_ATTR_ROOT_RANK, 0)
                                     .Build("_all_reduce_0");
  auto add_0 = OP_CFG(ADD).InCnt(2).OutCnt(1)
                          .Attr(ATTR_NAME_KERNEL_BIN_ID, "_add_0_fake_id")
                          .TensorDesc(FORMAT_NCHW, DT_FLOAT, {kTensorDim}).Build("_add_0");
  auto output_0 = OP_CFG(NETOUTPUT)
                      .Attr(ATTR_NAME_KERNEL_BIN_ID, "_output_0_fake_id")
                      .Build("_output_0");
  DEF_GRAPH(hccl_add) {
    CHAIN(NODE(data_0)->EDGE(0, 0)->NODE(hcom_0));
    CHAIN(NODE(data_1)->EDGE(0, 1)->NODE(hcom_0));
    CHAIN(NODE(data_0)->EDGE(0, 0)->NODE(add_0));
    CHAIN(NODE(hcom_0)->EDGE(0, 1)->NODE(add_0));
    CHAIN(NODE(add_0)->NODE(output_0));
  };

  auto graph = ToComputeGraph(hccl_add);
  graph->SetGraphUnknownFlag(false);
  return { graph, 2, 1 };
}
} // namespace anonymous

TEST_F(StaticShapeDumpST, Static_DataDump_Hccl_Graph) {
  auto config = DataDumpConfigBuilder();
  config.Commit();
  SessionWrapper wrapper(BuildGraph_HcclAdd());

  wrapper.Run();
  wrapper.Finalize();
  EXPECT_EQ(checker_->GetLoadOpMappingInfoSize(), 3UL);
  EXPECT_TRUE(checker_->CheckOpMappingInfoClear());
}

TEST_F(StaticShapeDumpST, Static_DataDump_Graph) {
  auto config = DataDumpConfigBuilder();
  config.Commit();
  SessionWrapper wrapper(BuildGraph_OneAdd(false));

  wrapper.Run();
  EXPECT_EQ(checker_->GetLoadOpMappingInfoSize(), 1UL);

  wrapper.Run();
  wrapper.Finalize();
  EXPECT_EQ(checker_->GetUnLoadOpMappingInfoSize(), 1UL);
  EXPECT_TRUE(checker_->CheckOpMappingInfoClear());
  EXPECT_EQ(checker_->GetStepId(), 1U);
}

TEST_F(StaticShapeDumpST, Static_OverflowDump_Hccl_Graph) {
  auto config = OverflowDumpConfigBuilder();
  config.Commit();
  SessionWrapper wrapper(BuildGraph_HcclAdd());

  wrapper.Run();
  wrapper.Finalize();
  EXPECT_EQ(checker_->GetLoadOpMappingInfoSize(), 1UL);
  EXPECT_TRUE(checker_->CheckOpMappingInfoClear());
}

TEST_F(StaticShapeDumpST, Static_OverflowDump_Graph) {
  auto config = OverflowDumpConfigBuilder();
  config.Commit();
  SessionWrapper wrapper(BuildGraph_OneAdd(false));

  wrapper.Run();
  wrapper.Finalize();
  EXPECT_EQ(checker_->GetLoadOpMappingInfoSize(), 1UL);
  EXPECT_TRUE(checker_->CheckOpMappingInfoClear());
}

TEST_F(StaticShapeDumpST, Static_DataDump_Hccl_Graph_In_Watcher_Mode) {
  auto config = DataDumpConfigBuilder().ModelConfig({"_add_0"}, {"_all_reduce_0"});
  config.Commit();
  SessionWrapper wrapper(BuildGraph_HcclAdd());

  wrapper.Run();
  wrapper.Finalize();
  EXPECT_EQ(checker_->GetLoadOpMappingInfoSize(), 1UL);
  EXPECT_TRUE(checker_->CheckOpMappingInfoClear());
}
