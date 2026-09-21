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

#include "exe_graph/runtime/kernel_context.h"           // class KernelContext
#include "graph_builder/bg_memory.h"                    // AllocOutputMemory
#include "register/node_converter_registry.h"           // REGISTER_NODE_CONVERTER
#include "register/kernel_registry.h"                   // REGISTER_KERNEL

using namespace ge;
using DynamicShapeDumpST = DumpST<true>;

namespace gert {
// What does this mean?
LowerResult FakeLoweringHcom(const ge::NodePtr &node, const LowerInput &lower_input) {
  size_t output_size = 8U;
  auto size_holder = bg::ValueHolder::CreateConst(&output_size, sizeof(output_size));
  auto output_addrs = bg::AllocOutputMemory(kOnDeviceHbm, node, {size_holder}, *(lower_input.global_data));
  auto compute_holder = bg::ValueHolder::CreateVoid<bg::ValueHolder>(
      "LaunchHcomKernel", {lower_input.input_addrs[1], output_addrs[0], lower_input.global_data->GetStream()});

  return {HyperStatus::Success(), {compute_holder}, {lower_input.input_shapes[0]}, output_addrs};
}
REGISTER_NODE_CONVERTER("_fake_lowering_hcom", FakeLoweringHcom);
REGISTER_KERNEL(LaunchHcomKernel).RunFunc([](KernelContext *) { return ge::GRAPH_SUCCESS; });
} //namespace gert

namespace {
void MarkSingleOp(const ComputeGraphPtr &graph) {
  AttrUtils::SetBool(graph, ATTR_SINGLE_OP_SCENE, true);
}

void MarkLoweringAsHccl(const ComputeGraphPtr &graph) {
  auto node = graph->FindFirstNodeMatchType(ADD);
  AttrUtils::SetStr(node->GetOpDesc(), kAttrLowingFunc, "_fake_lowering_hcom");
}

void MarkLoweringAsAclnn(const ComputeGraphPtr &graph) {
  auto node = graph->FindFirstNodeMatchType(ADD);
  AttrUtils::SetStr(node->GetOpDesc(), kAttrLowingFunc, "AclnnLoweringFunc");
}
ge::graphStatus OpExecutePrepareDoNothing(gert::OpExecutePrepareContext *op_exe_prepare_context) {
  op_exe_prepare_context->SetWorkspaceSizes({100});
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus OpExecuteLaunchDoNothing(gert::OpExecuteLaunchContext *) {
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferShapeForAdd(gert::InferShapeContext *context) {
  GELOGD("InferShapeForAdd");
  auto input_shape_0 = *context->GetInputShape(0);
  auto input_shape_1 = *context->GetInputShape(1);
  auto output_shape = context->GetOutputShape(0);
  if (input_shape_0.GetDimNum() != input_shape_1.GetDimNum()) {
    auto min_num = std::min(input_shape_0.GetDimNum(), input_shape_1.GetDimNum());
    if (min_num != 1) {
      GELOGE(ge::PARAM_INVALID, "Add param invalid, input_shape_0.GetDimNum() is %zu,  input_shape_1.GetDimNum() is %zu",
             input_shape_0.GetDimNum(), input_shape_1.GetDimNum());
    } else {
      if (input_shape_1.GetDimNum() > 1) {
        *output_shape = input_shape_1;
      } else {
        *output_shape = input_shape_0;
      }
      return ge::GRAPH_SUCCESS;
    }
  }
  if (input_shape_0.GetDimNum() == 0) {
    *output_shape = input_shape_1;
    return ge::GRAPH_SUCCESS;
  }
  if (input_shape_1.GetDimNum() == 0) {
    *output_shape = input_shape_0;
    return ge::GRAPH_SUCCESS;
  }
  output_shape->SetDimNum(input_shape_0.GetDimNum());
  for (size_t i = 0; i < input_shape_0.GetDimNum(); ++i) {
    output_shape->SetDim(i, std::max(input_shape_0.GetDim(i), input_shape_1.GetDim(i)));
  }

  return ge::GRAPH_SUCCESS;
}

struct AddCompileInfo {
  int64_t a;
  int64_t b;
};
ge::graphStatus TilingForAdd(gert::TilingContext *context) {
  GELOGD("TilingForAdd");
  auto ci = context->GetCompileInfo<AddCompileInfo>();
  GE_ASSERT_NOTNULL(ci);
  auto tiling_data = context->GetRawTilingData();
  GE_ASSERT_NOTNULL(tiling_data);
  tiling_data->Append(*ci);
  tiling_data->Append(ci->a);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingParseForAdd(gert::KernelContext *context) {
  return ge::GRAPH_SUCCESS;
}

} // anonymous namespace

TEST_F(DynamicShapeDumpST, Dynamic_OverflowDump_Graph) {
  auto config = OverflowDumpConfigBuilder();
  config.Commit();
  SessionWrapper wrapper(BuildGraph_OneAdd(true));

  // check dump task
  wrapper.Run();
  EXPECT_EQ(checker_->GetLoadOpMappingInfoSize(), 2U);
  EXPECT_EQ(checker_->GetUnLoadOpMappingInfoSize(), 1U);
  EXPECT_TRUE(checker_->CheckOpMappingInfoClear());
  EXPECT_EQ(checker_->GetModelName("_add_0"), "one_add");
  EXPECT_EQ(checker_->GetModelId("_add_0"), 1);
  checker_->ClearAllOpMappingInfos();

  wrapper.Run();
  EXPECT_EQ(checker_->GetLoadOpMappingInfoSize(), 2U);
  EXPECT_EQ(checker_->GetStepId(), 1U);
}

TEST_F(DynamicShapeDumpST, Dynamic_OverflowDump_SingleOp) {
  auto config = OverflowDumpConfigBuilder();
  config.Commit();
  SessionWrapper wrapper(BuildGraph_OneAdd(true) | MarkSingleOp);

  // check dump task
  wrapper.Run();
  EXPECT_EQ(checker_->GetLoadOpMappingInfoSize(), 2U);
  EXPECT_EQ(checker_->GetUnLoadOpMappingInfoSize(), 1U);
  // FIXME: BUG?
  // EXPECT_TRUE(checker_->CheckOpMappingInfoClear());
  EXPECT_EQ(checker_->GetModelName("_add_0"), "");
  EXPECT_EQ(checker_->GetModelId("_add_0"), UINT32_MAX);
  EXPECT_EQ(checker_->GetStepId(), UINT32_MAX);
}

TEST_F(DynamicShapeDumpST, Dynamic_OverflowDump_Hccl_Graph) {
  auto config = OverflowDumpConfigBuilder();
  config.Commit();
  SessionWrapper wrapper(BuildGraph_OneAdd(true) | MarkLoweringAsHccl);

  // check dump task
  wrapper.Run();
  EXPECT_EQ(checker_->GetLoadOpMappingInfoSize(), 1U);
}

TEST_F(DynamicShapeDumpST, Dynamic_OverflowDump_Hccl_SingleOp) {
  auto config = OverflowDumpConfigBuilder();
  config.Commit();
  SessionWrapper wrapper(BuildGraph_OneAdd(true) | MarkLoweringAsHccl | MarkSingleOp);

  // check dump task
  wrapper.Run();
  // 暂不支持Hccl算子的溢出检测，所以仅有1个OpDebug任务
  EXPECT_EQ(checker_->GetLoadOpMappingInfoSize(), 1U);
}

TEST_F(DynamicShapeDumpST, Dynamic_OverflowDump_Aclnn_Graph) {
  auto config = OverflowDumpConfigBuilder();
  config.Commit();
  SessionWrapper wrapper(BuildGraph_OneAdd(true) | MarkLoweringAsAclnn);

  // check dump task
  wrapper.Run();
  EXPECT_EQ(checker_->GetLoadOpMappingInfoSize(), 2U);
}

TEST_F(DynamicShapeDumpST, Dynamic_OverflowDump_Aclnn_SingleOP) {
  auto config = OverflowDumpConfigBuilder();
  config.Commit();
  SessionWrapper wrapper(BuildGraph_OneAdd(true) | MarkLoweringAsAclnn | MarkSingleOp);

  // check dump task
  wrapper.Run();
  EXPECT_EQ(checker_->GetLoadOpMappingInfoSize(), 2U);
}

TEST_F(DynamicShapeDumpST, Dynamic_OverflowDump_Aclnn_SingleOP_Twostages) {
  auto space_registry = std::make_shared<gert::OpImplSpaceRegistryV2>();
  auto op_impl_func = space_registry->CreateOrGetOpImpl("Add");
  op_impl_func->infer_shape = InferShapeForAdd;
  op_impl_func->tiling = TilingForAdd;
  op_impl_func->tiling_parse = TilingParseForAdd;
  op_impl_func->op_execute_prepare_func = OpExecutePrepareDoNothing;
  op_impl_func->op_execute_launch_func = OpExecuteLaunchDoNothing;
  op_impl_func->op_execute_func = nullptr;

  auto default_space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(space_registry);
  auto config = OverflowDumpConfigBuilder();
  config.Commit();
  SessionWrapper wrapper(BuildGraph_OneAdd(true) | MarkLoweringAsAclnn | MarkSingleOp);

  // check dump task
  wrapper.Run();
  EXPECT_EQ(checker_->GetLoadOpMappingInfoSize(), 2U);
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(default_space_registry);
}

TEST_F(DynamicShapeDumpST, Dynamic_DataDump_Graph) {
  auto config = DataDumpConfigBuilder();
  config.Commit();
  SessionWrapper wrapper(BuildGraph_OneAdd(true));

  // check dump task
  wrapper.Run();
  EXPECT_EQ(checker_->GetLoadOpMappingInfoSize(), 3U);
  EXPECT_EQ(checker_->GetUnLoadOpMappingInfoSize(), 0U);
  EXPECT_TRUE(checker_->CheckOpMappingInfoClear());
  EXPECT_EQ(checker_->GetModelName("_add_0"), "one_add");
  EXPECT_EQ(checker_->GetModelId("_add_0"), 1U);
  checker_->ClearAllOpMappingInfos();

  wrapper.Run();
  EXPECT_EQ(checker_->GetLoadOpMappingInfoSize(), 3U);
  EXPECT_EQ(checker_->GetStepId(), 1U);
}

TEST_F(DynamicShapeDumpST, Dynamic_DataDump_SingleOp) {
  auto config = DataDumpConfigBuilder();
  config.Commit();
  SessionWrapper wrapper(BuildGraph_OneAdd(true) | MarkSingleOp);

  // check dump task
  wrapper.Run();
  EXPECT_EQ(checker_->GetLoadOpMappingInfoSize(), 3U);
  EXPECT_EQ(checker_->GetUnLoadOpMappingInfoSize(), 0U);
  EXPECT_TRUE(checker_->CheckOpMappingInfoClear());
  EXPECT_EQ(checker_->GetModelName("_add_0"), "");
  EXPECT_EQ(checker_->GetModelId("_add_0"), UINT32_MAX);
  EXPECT_EQ(checker_->GetStepId(), UINT32_MAX);
}

TEST_F(DynamicShapeDumpST, Dynamic_DataDump_Hccl_Graph) {
  auto config = DataDumpConfigBuilder();
  config.Commit();
  SessionWrapper wrapper(BuildGraph_OneAdd(true) | MarkLoweringAsHccl);

  // check dump task
  wrapper.Run();
  EXPECT_EQ(checker_->GetLoadOpMappingInfoSize(), 4U);
}

TEST_F(DynamicShapeDumpST, Dynamic_DataDump_Hccl_SingleOp) {
  auto config = DataDumpConfigBuilder();
  config.Commit();
  SessionWrapper wrapper(BuildGraph_OneAdd(true) | MarkLoweringAsHccl | MarkSingleOp);

  // check dump task
  wrapper.Run();
  EXPECT_EQ(checker_->GetLoadOpMappingInfoSize(), 4U);
}

TEST_F(DynamicShapeDumpST, Dynamic_DataDump_Aclnn_Graph) {
  auto config = DataDumpConfigBuilder();
  config.Commit();
  SessionWrapper wrapper(BuildGraph_OneAdd(true) | MarkLoweringAsAclnn);

  // check dump task
  wrapper.Run();
  EXPECT_EQ(checker_->GetLoadOpMappingInfoSize(), 3U);
  EXPECT_EQ(checker_->GetTaskInputSize("_add_0"), 2U);
}

TEST_F(DynamicShapeDumpST, Dynamic_DataDump_Aclnn_SingleOp) {
  auto config = DataDumpConfigBuilder();
  config.Commit();
  SessionWrapper wrapper(BuildGraph_OneAdd(true) | MarkLoweringAsAclnn | MarkSingleOp);

  // check dump task
  wrapper.Run();
  EXPECT_EQ(checker_->GetLoadOpMappingInfoSize(), 3U);
}

TEST_F(DynamicShapeDumpST, Dynamic_WatchDump_Graph) {
  auto config = DataDumpConfigBuilder().ModelConfig({"_add_0"}, {"_add_1"});
  config.Commit();
  SessionWrapper wrapper(BuildGraph_TwoAdd(true));

  // check dump task
  wrapper.Run();
  // WatchDump not supported for dynamic shape.
  EXPECT_EQ(checker_->GetLoadOpMappingInfoSize(), 0U);
}
