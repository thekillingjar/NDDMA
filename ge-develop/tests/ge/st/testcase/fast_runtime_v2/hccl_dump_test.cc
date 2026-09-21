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
#include <mmpa/mmpa_api.h>
#include "core/execution_data.h"
#include "common/bg_test.h"
#include "core/builder/model_v2_executor_builder.h"
#include "common/dump/dump_manager.h"
#include "common/dump/dump_properties.h"
#include "framework/runtime/model_v2_executor.h"
#include "framework/common/types.h"
#include <kernel/memory/mem_block.h>
#include "kernel/tensor_attr.h"
#include "graph/utils/graph_utils.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "common/share_graph.h"
#include "lowering/model_converter.h"
#include "core/executor_error_code.h"
#include "stub/gert_runtime_stub.h"
#include "faker/fake_value.h"
#include "faker/kernel_run_context_facker.h"
#include "core/builder/node_types.h"
#include "core/utils/tensor_utils.h"
#include "exe_graph/runtime/compute_node_info.h"
#include "exe_graph/runtime/kernel_context.h"

#include "macro_utils/dt_public_scope.h"
#include "subscriber/dumper/executor_dumper.h"
#include "macro_utils/dt_public_unscope.h"

namespace gert {
class ExecutorDumperST : public bg::BgTest {
 protected:
  void TearDown() {
    GlobalDumper::GetInstance()->SetEnableFlags(0UL);
  }
};

/**
* 用例描述：hccl算子的datadump
*
* 预置条件：
* 1. hccl算子的dumpunit
*
* 测试步骤：
* 1. 构造hccl算子的dumpunit
* 2. 构造executorDump类
* 3. 使能datadump
* 4. 构造输入Tensor，shape为[1024]，输出Tensor的shape为[1024]，执行
*
* 预期结果：
* 1. 执行成功
*/
TEST_F(ExecutorDumperST, DoHcclDataDump_Ok) {
  StorageShape storage_shape{{4}, {4, 1}};
  kernel::BuildTensorAttr attr{kOnDeviceHbm, ge::DT_FLOAT16, {ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, ExpandDimsType()}};
  Tensor tensor_holder{storage_shape, attr.storage_format, attr.placement, attr.data_type, nullptr};
  tensor_holder.MutableTensorData() = TensorData{(void *)1024, nullptr, 0, kOnDeviceHbm};
  GertTensorData tensor_data;
  TensorUtils::RefTdToGtd(tensor_holder.GetTensorData(), -1, tensor_data);
  auto context_holder_1 =
      KernelRunContextFaker().KernelIONum(1, 2).Outputs({&tensor_holder, &tensor_data}).Build();
  const auto &sub_extend_info = ge::MakeShared<const SubscriberExtendInfo>();
  GlobalDumper::GetInstance()->SetEnableFlags(20);
  ExecutorDumper dumper(sub_extend_info);
  auto &node_add_dump_unit = dumper.node_names_to_dump_units_["hcom_reduce"];
  ASSERT_NE(context_holder_1.GetContext<KernelContext>(), nullptr);

  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("HcomReduce", "HcomReduce");
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  ge::NodePtr node_info = graph->AddNode(op_desc);
  node_add_dump_unit.node = node_info;
  node_add_dump_unit.output_addrs.push_back(nullptr);
  node_add_dump_unit.input_addrs.push_back(nullptr);
  node_add_dump_unit.output_shapes.push_back(nullptr);
  node_add_dump_unit.input_shapes.push_back(nullptr);
  node_add_dump_unit.output_addrs[0] = context_holder_1.GetContext<KernelContext>()->GetOutput(1);
  node_add_dump_unit.input_addrs[0] = context_holder_1.GetContext<KernelContext>()->GetOutput(1);
  gert::StorageShape x2_shape = {{4, 8, 16, 32, 64}, {4, 8, 16, 4, 2, 16, 16}};
  gert::StorageShape x1_shape = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};
  auto context_holder_2 = KernelRunContextFaker()
                              .Inputs({&x1_shape})
                              .Outputs({&x2_shape})
                              .KernelIONum(1, 1)
                              .NodeIoNum(1, 1)
                              .IrInstanceNum({1})
                              .Build();
  node_add_dump_unit.output_shapes[0] = context_holder_2.GetContext<KernelContext>()->GetOutput(0);
  node_add_dump_unit.input_shapes[0] = const_cast<Chain *>(context_holder_2.GetContext<KernelContext>()->GetInput(0));
  ge::DumpProperties dump_properties;
  dump_properties.AddPropertyValue("ALL_MODEL_NEED_DUMP_AND_IT_IS_NOT_A_MODEL_NAME", {"test"});
  dump_properties.SetDumpMode("all");
  ge::DumpManager::GetInstance().AddDumpProperties(ge::kInferSessionId, dump_properties);

  Node node;
  size_t total_size = 0UL;
  ComputeNodeInfo::CalcSize(1, 1, 1, total_size);
  auto compute_node_info_holder = ge::ComGraphMakeUnique<uint8_t[]>(total_size);
  ComputeNodeInfo* compute_node_info = ge::PtrToPtr<uint8_t, ComputeNodeInfo>(compute_node_info_holder.get());
  compute_node_info->Init(1, 1, 1, "hcom_reduce", "HcomReduce");
  uint8_t holder[sizeof(KernelExtendInfo)];
  KernelExtendInfo* extend_kernel_info = reinterpret_cast<KernelExtendInfo *>(holder);
  extend_kernel_info->SetKernelName("hcom_kernel_launch");
  extend_kernel_info->SetKernelType("LaunchHcomKernel");
  KernelRunContext kernel_run_context;
  kernel_run_context.input_size = 1;
  kernel_run_context.output_size = 1;
  kernel_run_context.compute_node_info = static_cast<void *>(compute_node_info);
  kernel_run_context.kernel_extend_info = static_cast<void *>(extend_kernel_info);
  node.context = kernel_run_context;

  const auto properties = ge::DumpManager::GetInstance().GetDumpProperties(ge::kInferSessionId);
  EXPECT_EQ(dumper.DataDump(&node, kExecuteStart), ge::SUCCESS);
  EXPECT_EQ(dumper.DataDump(&node, kExecuteEnd), ge::SUCCESS);
}
}
