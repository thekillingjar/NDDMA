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
#include "stub/gert_runtime_stub.h"
#include "core/builder/model_v2_executor_builder.h"
#include "core/utils/tensor_utils.h"
#include "common/dump/dump_manager.h"
#include "common/dump/dump_properties.h"
#include "framework/runtime/model_v2_executor.h"
#include "framework/common/types.h"
#include "kernel/tensor_attr.h"
#include "graph/utils/graph_utils.h"
#include "common/share_graph.h"
#include "faker/fake_value.h"
#include "faker/kernel_run_context_facker.h"
#include "faker/ge_model_builder.h"
#include "lowering/model_converter.h"
#include "core/executor_error_code.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "engine/ffts_plus/converter/ffts_plus_proto_transfer.h"
#include "register/ffts_node_calculater_registry.h"
#include "engine/aicore/kernel/aicore_update_kernel.h"

#include "macro_utils/dt_public_scope.h"
#include "common/dump/dump_op.h"
#include "graph/load/model_manager/davinci_model.h"
#include "subscriber/dumper/executor_dumper.h"
#include "common/global_variables/diagnose_switch.h"
#include "macro_utils/dt_public_unscope.h"

namespace gert {
namespace {
uint64_t FindFirstNonEmptyId(ExecutorDumper *dumper) {
  for (uint64_t i = 0U; i < dumper->kernel_idxes_to_dump_units_.size(); ++i) {
    if (!dumper->kernel_idxes_to_dump_units_[i].empty()) {
      return i;
    }
  }
  return std::numeric_limits<uint64_t>::max();
}
}  // namespace

class ExecutorDumperST : public bg::BgTest {
 protected:
  void TearDown() {
    GlobalDumper::GetInstance()->SetEnableFlags(0UL);
  }
};

/**
* 用例描述：ffts+算子的datadump
*
* 预置条件：
* 1. ffts+算子的dumpunit
*
* 测试步骤：
* 1. 构造ffts+算子的dumpunit
* 2. 构造executorDump类
* 3. 使能datadump
*
* 预期结果：
* 1. 执行成功
*/
TEST_F(ExecutorDumperST, DoFftsDataDump_Ok) {
  GlobalDumper::GetInstance()->SetEnableFlags(1UL);
  auto compute_graph = ShareGraph::BuildSingleNodeGraph();
  ge::AttrUtils::SetBool(compute_graph, ge::ATTR_SINGLE_OP_SCENE, true);
  compute_graph->TopologicalSorting();
  GeModelBuilder builder(compute_graph);
  auto ge_root_model = builder.BuildGeRootModel();
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);
  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);

  const auto &extend_info = ge::MakeShared<const SubscriberExtendInfo>(
      model_executor.get(), exe_graph, ge_root_model->GetRootGraph(), ge::ModelData(), ge_root_model, SymbolsToValue{},
      ge_root_model->GetCurModelId(), ge_root_model->GetModelName(), nullptr,
      std::unordered_map<std::string, TraceAttr>{});
  auto dumper = ge::MakeUnique<ExecutorDumper>(extend_info);
  ASSERT_NE(dumper, nullptr);
  ASSERT_EQ(dumper->Init(), ge::SUCCESS);
  uint64_t id = FindFirstNonEmptyId(dumper.get()); 

  ge::diagnoseSwitch::EnableDataDump();
  ge::DumpProperties dump_properties;
  dump_properties.AddPropertyValue("ALL_MODEL_NEED_DUMP_AND_IT_IS_NOT_A_MODEL_NAME", {"test"});
  dump_properties.SetDumpMode("all");
  ge::DumpManager::GetInstance().AddDumpProperties(ge::kInferSessionId, dump_properties);
  
  kernel::AICoreThreadParam thread_param;
  uint32_t thread_dim = 2U;
  auto out_type = ContinuousVector::Create<uint32_t>(1);
  auto out_type_vec = reinterpret_cast<ContinuousVector *>(out_type.get());
  out_type_vec->SetSize(1);
  auto out_type_ptr = reinterpret_cast<uint32_t*>(out_type_vec->MutableData());
  out_type_ptr[0] = 0;
  auto context_holder_1 = KernelRunContextFaker()
                            .NodeName(std::move("add1"))
                            .KernelIONum(static_cast<size_t>(kernel::ArgsInKey::kNUM), 0)
                            .KernelType("FFTSUpdateAICoreArgs")
                            .KernelName("FFTSUpdateAICoreArgs")
                            .Build();
  context_holder_1.value_holder[static_cast<size_t>(kernel::ArgsInKey::THREAD_PARAM)].Set(&thread_param, nullptr);
  context_holder_1.value_holder[static_cast<size_t>(kernel::ArgsInKey::IN_MEM_TYPE)].Set(out_type_vec, nullptr);
  context_holder_1.value_holder[static_cast<size_t>(kernel::ArgsInKey::OUT_MEM_TYPE)].Set(out_type_vec, nullptr);
  context_holder_1.value_holder[static_cast<size_t>(kernel::ArgsInKey::THREAD_DIM)]
      .Set(reinterpret_cast<void *>(thread_dim), nullptr);
  size_t size = sizeof(Node) + sizeof(AsyncAnyValue *) * 6;
  Node *launch_node_1 = (Node *)malloc(size);
  launch_node_1->node_id = id;
  memcpy(&launch_node_1->context, context_holder_1.context, sizeof(KernelRunContext) + 6 * sizeof(AsyncAnyValue *));
  launch_node_1->func = nullptr;

  uint64_t offset_vec[5] = {0};
  auto thread_offset = ContinuousVector::Create<uint64_t>(2);
  auto thread_offset_vec = reinterpret_cast<ContinuousVector *>(thread_offset.get());
  thread_offset_vec->SetSize(2);
  auto thread_offset_vec_ptr = reinterpret_cast<uint64_t *>(thread_offset_vec->MutableData());
  thread_offset_vec_ptr[0] = reinterpret_cast<uint64_t>(&offset_vec[0]);
  thread_offset_vec_ptr[1] = reinterpret_cast<uint64_t>(&offset_vec[1]);
  auto context_holder_auto_1 = KernelRunContextFaker()
                                   .NodeName(std::move("add1"))
                                   .KernelIONum(6, 0)
                                   .KernelType("FFTSUpdateAutoAICoreArgs")
                                   .KernelName("FFTSUpdateAutoAICoreArgs")
                                   .Build();
  context_holder_auto_1.value_holder[5].Set(thread_offset_vec, nullptr);
  context_holder_auto_1.value_holder[3].Set(reinterpret_cast<void *>(thread_dim), nullptr);
  Node *launch_auto_node_1 = (Node *)malloc(sizeof(Node) + sizeof(AsyncAnyValue *) * 6);
  launch_auto_node_1->node_id = id;
  memcpy(&launch_auto_node_1->context, context_holder_auto_1.context,
         sizeof(KernelRunContext) + 6 * sizeof(AsyncAnyValue *));
  launch_auto_node_1->func = nullptr;

  auto ctx_ids = ContinuousVector::Create<int32_t>(4);
  auto ctx_ids_vec = reinterpret_cast<ContinuousVector *>(ctx_ids.get());
  ctx_ids_vec->SetSize(4);
  auto ctx_ids_ptr = reinterpret_cast<int32_t *>(ctx_ids_vec->MutableData());
  for (size_t i = 0U; i < ctx_ids_vec->GetSize(); i++) {
    ctx_ids_ptr[i] = i;
  }

  size_t descBufLen = sizeof(rtFftsPlusComCtx_t) * static_cast<size_t>(10);
  size_t total_size = sizeof(TransTaskInfo) + descBufLen + sizeof(rtFftsPlusSqe_t);
  auto holder = ge::MakeUnique<uint8_t[]>(total_size);
  TransTaskInfo *task_info_ptr = reinterpret_cast<TransTaskInfo*>(holder.get());
  size_t buf_offset = sizeof(rtFftsPlusSqe_t);
  task_info_ptr->offsets[static_cast<size_t>(InfoStType::kDescBuf)] = buf_offset;
  task_info_ptr->rt_task_info.descBufLen = descBufLen;
  const size_t args_size = sizeof(rtFftsPlusComCtx_t);
  auto *buff_ptr = &task_info_ptr->args[args_size];
  for (int i = 0; i < 4; ++i) {
    auto context = reinterpret_cast<rtFftsPlusAicAivCtx_t*>(buff_ptr);
    context->contextType = RT_CTX_TYPE_AICORE;
    context->threadDim = 1U;
    buff_ptr += sizeof(rtFftsPlusComCtx_t);
  }
  auto *const ffts_plus_sqe = ge::PtrToPtr<uint8_t, rtFftsPlusSqe_t>(task_info_ptr->args);
  ffts_plus_sqe->totalContextNum = 16;

  auto &dump_unit = dumper->node_names_to_dump_units_["add1"];
  StorageShape storage_shape{{4}, {4, 1}};
  kernel::BuildTensorAttr attr{kOnDeviceHbm, ge::DT_FLOAT16, {ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, ExpandDimsType()}};
  Tensor tensor_holder{storage_shape, attr.storage_format, attr.placement, attr.data_type, nullptr};
  tensor_holder.MutableTensorData() = TensorData{(void *)1024, nullptr, 0, kOnDeviceHbm};
  GertTensorData tensor_data;
  TensorUtils::RefTdToGtd(tensor_holder.GetTensorData(), -1 ,tensor_data);
  auto context_holder_4 =
      KernelRunContextFaker().KernelIONum(1, 2).Outputs({&tensor_holder, &tensor_data}).Build();
  dump_unit.output_addrs[0] = context_holder_4.GetContext<KernelContext>()->GetOutput(1);
  dump_unit.input_addrs[0] = context_holder_4.GetContext<KernelContext>()->GetOutput(1);
  dump_unit.input_addrs[1] = context_holder_4.GetContext<KernelContext>()->GetOutput(1);
  gert::StorageShape x2_shape = {{4, 8, 16, 32, 64}, {4, 8, 16, 4, 2, 16, 16}};
  gert::StorageShape x1_shape = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};
  auto context_holder_5 = KernelRunContextFaker()
                              .Inputs({&x1_shape})
                              .Outputs({&x2_shape})
                              .KernelIONum(1, 1)
                              .NodeIoNum(1, 1)
                              .IrInstanceNum({1})
                              .Build();
  dump_unit.output_shapes[0] = context_holder_5.GetContext<KernelContext>()->GetOutput(0);
  dump_unit.input_shapes[0] = const_cast<Chain *>(context_holder_5.GetContext<KernelContext>()->GetInput(0));
  dump_unit.input_shapes[1] = const_cast<Chain *>(context_holder_5.GetContext<KernelContext>()->GetInput(0));
  ge::Context dump_context;
  dump_context.context_id = 0;
  dump_context.thread_id = 0;
  dump_unit.context_list.emplace_back(dump_context);
  
  const auto properties = ge::DumpManager::GetInstance().GetDumpProperties(ge::kInferSessionId);
  EXPECT_EQ(dumper->DoDataDump(dump_unit, properties), ge::SUCCESS);
  
  rtStream_t stream_ = reinterpret_cast<void *>(0x12);
  NodeMemPara node_para;
  TransTaskInfo *pre_data_ptr = reinterpret_cast<TransTaskInfo *>(holder.get());
  pre_data_ptr->offsets[static_cast<size_t>(InfoStType::kDescBuf)] = buf_offset;
  pre_data_ptr->rt_task_info.descBufLen = descBufLen;
  pre_data_ptr->rt_task_info.descBuf = holder.get() + sizeof(TransTaskInfo);
  node_para.host_addr = pre_data_ptr;
  node_para.dev_addr = pre_data_ptr;
  auto context_holder = KernelRunContextFaker()
                            .NodeName(std::move("add1"))
                            .KernelIONum(2, 0)
                            .Inputs({reinterpret_cast<void *>(stream_), &node_para})
                            .KernelType("LaunchFFTSPlusTask")
                            .KernelName("LaunchFFTSPlusTask")
                            .Build();
  size = sizeof(Node) + sizeof(AsyncAnyValue *) * 2;
  Node *launch_node = (Node *)malloc(size);
  launch_node->node_id = id;
  memcpy(&launch_node->context, context_holder.context, sizeof(KernelRunContext) + 2 * sizeof(AsyncAnyValue *));
  launch_node->func = nullptr;

  // dynamic dump ffts_plus
  EXPECT_EQ(dumper->UpdateFftsplusLaunchTask(launch_node), ge::SUCCESS);
  
  ge::DumpOp dump_op;
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("GatherV2", "GatherV2");
  ge::Context context;
  context.context_id = 0;
  context.thread_id = 0;
  ge::RealAddressAndSize real_address_and_size;
  real_address_and_size.address = 1U;
  real_address_and_size.size = 1U;
  context.input.emplace_back(real_address_and_size);
  context.output.emplace_back(real_address_and_size);
  std::vector<ge::Context> dump_context_1;
  dump_context_1.emplace_back(context);
  std::vector<uintptr_t> input_addrs;
  std::vector<uintptr_t> output_addrs;
  int64_t *addr_in = (int64_t *)malloc(1024);
  int64_t *addr_out = (int64_t *)malloc(1024);
  input_addrs.push_back(reinterpret_cast<uintptr_t>(addr_in));
  output_addrs.push_back(reinterpret_cast<uintptr_t>(addr_out));
  dump_op.SaveFftsSubOpInfo(op_desc, dump_context_1);
  void *load_dump_info = nullptr;
  uint32_t load_dump_len = 0U;
  void *unload_dump_info = nullptr;
  uint32_t unload_dump_len = 0U;
  EXPECT_EQ(dump_op.GenerateFftsDump(properties, load_dump_info, load_dump_len,
                                     unload_dump_info, unload_dump_len, dumper->IsSingleOpScene()), ge::SUCCESS);
  ge::DumpProperties dump_properties2;
  dump_properties2.AddPropertyValue("ALL_MODEL_NEED_DUMP_AND_IT_IS_NOT_A_MODEL_NAME", {"test"});
  dump_properties2.SetDumpMode("input");
  EXPECT_EQ(dump_op.GenerateFftsDump(dump_properties2, load_dump_info, load_dump_len,
                                     unload_dump_info, unload_dump_len, dumper->IsSingleOpScene()), ge::SUCCESS);
  dump_properties2.SetDumpMode("output");
  EXPECT_EQ(dump_op.GenerateFftsDump(dump_properties2, load_dump_info, load_dump_len,
                                     unload_dump_info, unload_dump_len, dumper->IsSingleOpScene()), ge::SUCCESS);
  dump_properties2.SetDumpMode("bad_mode");
  EXPECT_EQ(dump_op.GenerateFftsDump(dump_properties2, load_dump_info, load_dump_len,
                                     unload_dump_info, unload_dump_len, dumper->IsSingleOpScene()), ge::SUCCESS);

  auto context_holder_6 = KernelRunContextFaker().KernelIONum(static_cast<size_t>(kernel::UpdateKey::RESERVED), 1)
      .NodeName(std::move("add1"))
      .NodeIoNum(2,2).IrInputNum(2)
      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
      .KernelType("AICoreUpdateContext")
      .KernelName("AICoreUpdateContext")
      .Build();
  Shape shape0({600, 600, 800});
  Shape shape1({800, 800, 1000});
  auto in_slice = ContinuousVector::Create<Shape>(12);
  auto in_slice_vec = reinterpret_cast<ContinuousVector *>(in_slice.get());
  in_slice_vec->SetSize(2);
  auto in_slice_ptr = reinterpret_cast<Shape *>(in_slice_vec->MutableData());
  in_slice_ptr[0] = shape0;
  in_slice_ptr[1] = shape1;

  auto out_slice = ContinuousVector::Create<Shape>(12);
  auto out_slice_vec = reinterpret_cast<ContinuousVector *>(out_slice.get());
  out_slice_vec->SetSize(2);
  auto out_slice_ptr = reinterpret_cast<Shape *>(out_slice_vec->MutableData());
  out_slice_ptr[0] = shape0;
  out_slice_ptr[1] = shape1;
  context_holder_6.value_holder[static_cast<size_t>(kernel::UpdateKey::LAST_IN_SLICE)].Set(in_slice_vec, nullptr);
  context_holder_6.value_holder[static_cast<size_t>(kernel::UpdateKey::LAST_OUT_SLICE)].Set(out_slice_vec, nullptr);
  size = sizeof(Node) + sizeof(AsyncAnyValue *) * 6;
  Node *launch_node_4 = (Node *)malloc(size);
  launch_node_4->node_id = id;
  memcpy(&launch_node_4->context, context_holder_6.context, sizeof(KernelRunContext) + 6 * sizeof(AsyncAnyValue *));
  launch_node_4->func = nullptr;

  free(launch_node);
  free(launch_node_1);
  free(launch_node_4);
  free(addr_in);
  free(addr_out);
  free(launch_auto_node_1);
}
}
