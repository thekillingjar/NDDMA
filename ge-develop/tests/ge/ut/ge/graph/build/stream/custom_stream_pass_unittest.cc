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
#include <ge_running_env/fake_op.h>
#include <ge_running_env/ge_running_env_faker.h>
#include "engines/manager/engine_manager/dnnengine_manager.h"
#include "api/gelib/gelib.h"

#include "graph/build/stream/stream_utils.h"
#include "common/multi_stream_share_graph.h"
#include "graph/build/stream/logical_stream_allocator.h"
#include "graph/partition/engine_partitioner.h"
#include <graph_utils_ex.h>
#include <common/share_graph.h>
#include <debug/ge_attr_define.h>
#include <ge_running_env/dir_env.h>
#include <register/register_custom_pass.h>

#include "graph/build/stream/dynamic_stream_allocator.h"

namespace ge {
 namespace {
  Status PureKnownShape_SimpleCase(const ConstGraphPtr &graph, StreamPassContext &context) {
   AscendString graph_name;
   graph->GetName(graph_name);
   if (graph_name != "PureKnownShape_SimpleCase") {
    return SUCCESS;
   }
   std::cout << "before current max stream id is "<< context.GetCurrMaxStreamId()<< std::endl;
   for (auto n : graph->GetDirectNode()) {
    AscendString name;
    n.GetName(name);
    if (name != "add2") {
     continue;
    }
    context.SetStreamId(n, context.AllocateNextStreamId());
   }
   std::cout << "after current max stream id is "<< context.GetCurrMaxStreamId()<< std::endl;
   return SUCCESS;
  }

  Status PureUnknownShape_SimpleCase(const ConstGraphPtr &graph, StreamPassContext &context) {
   AscendString graph_name;
   graph->GetName(graph_name);
   if (graph_name != "PureUnknownShape_SimpleCase") {
    return SUCCESS;
   }
   std::cout << "before current max stream id is "<< context.GetCurrMaxStreamId()<< std::endl;
   for (auto n : graph->GetDirectNode()) {
    AscendString name;
    n.GetName(name);
    if (name != "nonzero") {
     continue;
    }
    context.SetStreamId(n, context.AllocateNextStreamId());
   }
   std::cout << "after current max stream id is "<< context.GetCurrMaxStreamId()<< std::endl;
   return SUCCESS;
  }
 } // namespace
class CustomStreamPassUT : public testing::Test {
 protected:
  static void SetUpTestSuite() {
   // mock ge engine
   std::map<string, string> options;
   EXPECT_EQ(GELib::Initialize(options), SUCCESS);
   ge::GELib::GetInstance()->OpsKernelManagerObj().ops_kernel_store_.clear();
   GeRunningEnvFaker().Reset()
     .Install(FakeEngine("DNN_VM_GE_LOCAL").KernelInfoStore("DNN_VM_GE_LOCAL_OP_STORE"))
     .Install(FakeEngine("AIcoreEngine").KernelInfoStore("AIcoreEngine"))
     .Install(FakeEngine("DNN_VM_RTS").KernelInfoStore("DNN_VM_RTS_OP_STORE"))
     .Install(FakeEngine("DNN_HCCL").KernelInfoStore("DNN_HCCL"))
     .Install(FakeOp(IF).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
     .Install(FakeOp(DATA).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
     .Install(FakeOp(NETOUTPUT).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
     .Install(FakeOp(CONSTANT).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
     .Install(FakeOp(PARTITIONEDCALL).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
     .Install(FakeOp(RESHAPE).InfoStoreAndBuilder("DNN_VM_GE_LOCAL_OP_STORE"))
     .Install(FakeOp(ADD).InfoStoreAndBuilder("AIcoreEngine"))
     .Install(FakeOp(MUL).InfoStoreAndBuilder("AIcoreEngine"))
     .Install(FakeOp(RANK).InfoStoreAndBuilder("AIcoreEngine"))
     .Install(FakeOp("NonZero").InfoStoreAndBuilder("AIcoreEngine"))
     .Install(FakeOp(RELU).InfoStoreAndBuilder("AIcoreEngine"))
     .Install(FakeOp(HCOMALLREDUCE).InfoStoreAndBuilder("DNN_HCCL"));

   // prepare engin conf
   EngineConfPtr conf1 = std::make_shared<EngineConf>();
   conf1->id = "AIcoreEngine";
   EngineConfPtr conf2 = std::make_shared<EngineConf>();
   conf2->id = "DNN_VM_GE_LOCAL";
   conf2->skip_assign_stream = true;
   conf2->attach = true;
   EngineConfPtr conf3 = std::make_shared<EngineConf>();
   conf3->id = "DNN_HCCL";
   conf3->independent = true;
   conf3->skip_assign_stream = false;
   conf3->attach = false;
   SchedulerConf scheduler_conf;
   for (auto &conf: {conf1, conf2, conf3}) {
    conf->scheduler_id = "scheduler";
    scheduler_conf.cal_engines[conf->id] = conf;
   }
   auto instance_ptr = ge::GELib::GetInstance();
   EXPECT_NE(instance_ptr, nullptr);
   instance_ptr->DNNEngineManagerObj().schedulers_["scheduler"] = scheduler_conf;
   dlog_setlevel(GE_MODULE_NAME, 1 ,0);
  }
  static void TearDownTestSuite() {
   auto instance_ptr = ge::GELib::GetInstance();
   EXPECT_NE(instance_ptr, nullptr);
   instance_ptr->DNNEngineManagerObj().schedulers_.clear();
   GeRunningEnvFaker().Reset();
   instance_ptr->Finalize();
  }
};
/*
 纯静态模型，不带子图
 内置流分配add1和add2在同一个流上， stream id为1
 自定义pass将add2单独分流
 预期结果：add2的stream id为2

  *  NetOutput
  *      |
  *     add2
  *    |     \
  *    \     add1
  *     \   /   \
  *      data1   data2
 */
TEST_F(CustomStreamPassUT, PureKnownShape_SimpleCase) {
 REGISTER_CUSTOM_PASS("PureKnownShape_SimpleCase_Pass")
    .CustomAllocateStreamPassFn(PureKnownShape_SimpleCase)
    .Stage(CustomPassStage::kAfterAssignLogicStream);
 auto compute_graph = gert::ShareGraph::BuildTwoAddNodeKnownShapeGraph();
 compute_graph->SetName("PureKnownShape_SimpleCase");
 // 二拆
 EnginePartitioner partitioner;
 AttrUtils::SetStr(compute_graph, ATTR_NAME_SESSION_GRAPH_ID, "0"); // parttion required
 EXPECT_EQ(partitioner.Partition(compute_graph, EnginePartitioner::Mode::kSecondPartitioning), SUCCESS);
 const auto &graph_2_subgraphlist = partitioner.GetSubGraphMap();
 EXPECT_EQ(graph_2_subgraphlist.size(), 1);

 std::map<std::string, int32_t> max_parallel_num;
 LogicalStreamAllocator allocator(max_parallel_num);
 int64_t total_stream_num = 0;
 int64_t main_stream_num = 0;
 EXPECT_EQ(allocator.Assign(compute_graph, graph_2_subgraphlist, total_stream_num, main_stream_num), SUCCESS);

 EXPECT_EQ(total_stream_num, 2);
 EXPECT_EQ(main_stream_num, 2);
 auto add2 = compute_graph->FindNode("add2");
 EXPECT_EQ(add2->GetOpDesc()->GetStreamId(), 1);
}
 /*
 纯动态模型，不带子图
 *            NetOutput
 *                |
 *              nonzero
 *             /     \
 *           add1     \
 *          /   \      \
 *         /   data2    |
 *        /             |
 *      data1 ----------+
 */
 TEST_F(CustomStreamPassUT, PureUnknownShape_SimpleCase) {
 REGISTER_CUSTOM_PASS("PureUnknownShape_SimpleCase")
     .CustomAllocateStreamPassFn(PureUnknownShape_SimpleCase)
     .Stage(CustomPassStage::kAfterAssignLogicStream);
 auto compute_graph = gert::ShareGraph::BuildAiCoreThirdClassNodeGraph();
 compute_graph->SetName("PureUnknownShape_SimpleCase");
 // 二拆
 EnginePartitioner partitioner;
 AttrUtils::SetStr(compute_graph, ATTR_NAME_SESSION_GRAPH_ID, "0"); // parttion required
 EXPECT_EQ(partitioner.Partition(compute_graph, EnginePartitioner::Mode::kSecondPartitioning), SUCCESS);
 const auto &graph_2_subgraphlist = partitioner.GetSubGraphMap();
 EXPECT_EQ(graph_2_subgraphlist.size(), 1);

 const auto dynamic_stream_allocator = MakeShared<DynamicStreamAllocator>();
 EXPECT_EQ(dynamic_stream_allocator->AssignStreamsForDynamicShapeGraph(compute_graph, graph_2_subgraphlist), SUCCESS);

 EXPECT_EQ(dynamic_stream_allocator->GetStreamNum(), 2);
 EXPECT_EQ(dynamic_stream_allocator->GetEventNum(), 2);

 auto nozero = compute_graph->FindNode("nonzero");
 EXPECT_EQ(nozero->GetOpDesc()->GetStreamId(), 1);
 std::vector<int64_t> send_ids;
 std::vector<int64_t> recv_ids;
 AttrUtils::GetListInt(nozero->GetOpDesc(), ATTR_NAME_SEND_EVENT_IDS, send_ids);
 AttrUtils::GetListInt(nozero->GetOpDesc(), ATTR_NAME_RECV_EVENT_IDS, recv_ids);
 EXPECT_EQ(send_ids.size(), 1);
 EXPECT_EQ(send_ids[0], 1);
 EXPECT_EQ(recv_ids.size(), 1);
 EXPECT_EQ(recv_ids[0], 0);
}
/**
 *  在不设置stream label的情况下，开启通信并行，hcom在stram0, a和b在stream1
 *        data
 ///         \
 ///        hcom(0)
 ///         | \
 ///       a(1) b(1)
 ///         \  \
 ///      netoutput
   设置hcom,a,b的stream label为stream1，与通信并行配置冲突，预期结果为 hcom和a、b都在同一条stream
 */
 TEST_F(CustomStreamPassUT, PureKnownShape_UserStreamLableOnHcom_LabelConfilctWithOption) {
 auto graph = gert::ShareGraph::BuildHcomGraphWithTwoOutputs(HCOMALLREDUCE);
 auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
 auto hcom1 = compute_graph->FindNode("hcom_1");
 auto a = compute_graph->FindNode("a");
 auto b = compute_graph->FindNode("b");
 AttrUtils::SetStr(hcom1->GetOpDesc(), public_attr::USER_STREAM_LABEL, "stream1");
 AttrUtils::SetInt(hcom1->GetOpDesc(), HCOM_ATTR_FUSION, 1);

 AttrUtils::SetStr(a->GetOpDesc(), public_attr::USER_STREAM_LABEL, "stream1");
 AttrUtils::SetStr(b->GetOpDesc(), public_attr::USER_STREAM_LABEL, "stream1");
 // 二拆
 EnginePartitioner partitioner;
 AttrUtils::SetStr(compute_graph, ATTR_NAME_SESSION_GRAPH_ID, "0"); // parttion required
 EXPECT_EQ(partitioner.Partition(compute_graph, EnginePartitioner::Mode::kSecondPartitioning), SUCCESS);
 const auto &graph_2_subgraphlist = partitioner.GetSubGraphMap();

 std::map<std::string, int32_t> max_parallel_num;
 LogicalStreamAllocator allocator(max_parallel_num);
 allocator.EnableHcomParallel(true);
 int64_t total_stream_num = 0;
 int64_t main_stream_num = 0;
 EXPECT_EQ(allocator.Assign(compute_graph, graph_2_subgraphlist, total_stream_num, main_stream_num), SUCCESS);

 EXPECT_EQ(total_stream_num, 2);
 EXPECT_EQ(main_stream_num, 2);
 auto hcom_stream_id = hcom1->GetOpDesc()->GetStreamId();
 auto a_stream_id = a->GetOpDesc()->GetStreamId();
 auto b_stream_id = b->GetOpDesc()->GetStreamId();
 EXPECT_EQ(hcom_stream_id, a_stream_id);
 EXPECT_EQ(hcom_stream_id, b_stream_id);
}
} // namespace ge
