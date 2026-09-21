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

#include "faker/kernel_run_context_facker.h"
#include "register/kernel_registry.h"
#include "common/ge_inner_attrs.h"
#include "common/share_graph.h"
#include "common/checker.h"
#include "core/execution_data.h"
#include "graph/compute_graph.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/utils/tensor_utils.h"
#include "lowering/graph_converter.h"
#include "framework/runtime/rt_session.h"
#include "framework/runtime/rt_var_manager.h"
#include "faker/global_data_faker.h"
#include "faker/model_desc_holder_faker.h"
#include "exe_graph/runtime/gert_tensor_data.h"
#include "kernel/memory/host_mem_allocator.h"
#include "kernel/memory/caching_mem_allocator.h"
#include "kernel/common_kernel_impl/variable.h"
#include "graph/utils/execute_graph_utils.h"

namespace gert {
class UtestRt2VarManager : public RtVarManager {
 public:
  ge::Status GetVarShapeAndMemory(const std::string &id, gert::StorageShape &shape,
                                  gert::TensorData &memory) const override {
    GELOGI("Var manager %p Get variable %s", this, id.c_str());
    auto iter = named_variables_.find(id);
    GE_ASSERT(iter != named_variables_.end());

    shape = iter->second.first;
    memory.ShareFrom(iter->second.second);
    return ge::SUCCESS;
  }

  void SetVarShapeAndMemory(const std::string &id, gert::StorageShape &shape, gert::TensorData &memory) {
    GELOGI("Var manager %p Set variable %s", this, id.c_str());
    auto &shape_and_memory = named_variables_[id];
    shape_and_memory.first = shape;
    shape_and_memory.second.ShareFrom(memory);
  }

 private:
  std::map<std::string, std::pair<gert::StorageShape, gert::TensorData>> named_variables_;
};
namespace {
Shape GeShapeToGertShape(const ge::GeShape &ge_shape) {
  Shape gert_shape;
  gert_shape.SetDimNum(ge_shape.GetDimNum());
  for (size_t i = 0; i < ge_shape.GetDimNum(); ++i) {
    gert_shape.SetDim(i, ge_shape.GetDim(i));
  }
  return gert_shape;
}
}  // namespace

TEST(VariableUtest, SplitVariable) {
  memory::HostMemAllocator host_mem_allocator;
  memory::HostGertMemAllocator gert_allocator(&host_mem_allocator);
  const KernelRegistry::KernelFuncs *kernel = KernelRegistry::GetInstance().FindKernelFuncs("SplitVariable");
  ASSERT_NE(kernel, nullptr);

  auto graph = SingleNodeGraphBuilder("main", "Variable").NumInputs(0).NumOutputs(1).Build();
  graph->SetGraphUnknownFlag(true);
  auto variable = graph->FindFirstNodeMatchType("Variable");
  ASSERT_NE(variable, nullptr);

auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();
  auto model_desc_holder = ModelDescHolderFaker().Build();
  auto exe_graph = GraphConverter()
      .SetModelDescHolder(&model_desc_holder)
      .ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);

  ASSERT_EQ(exe_graph->GetAllSubgraphs().size(), 3U);
  auto init_graph = exe_graph->GetAllSubgraphs()[0];

  auto split_variable = ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_graph, "SplitVariable");
  ASSERT_NE(split_variable, nullptr);

  constexpr size_t input_num = static_cast<size_t>(kernel::SplitVariableInputs::kNum);
  constexpr size_t output_num = static_cast<size_t>(kernel::SplitVariableOutputs::kNum);
  ASSERT_EQ(split_variable->GetDataInNum(), input_num);
  ASSERT_EQ(split_variable->GetDataOutNum(), output_num);

  auto run_context = KernelRunContextFaker().KernelIONum(input_num, output_num).NodeIoNum(0, 1).Build();

  auto kernel_context = run_context.GetContext<KernelContext>();
  auto var_id = (char *)"VAR_VariableUtest.SplitVariable";
  
  RtSession rt_session(1);
  rt_session.SetVarManager(nullptr);
  kernel_context->MutableInput(0)->Set(&rt_session, nullptr);
  kernel_context->MutableInput(1)->Set(var_id, nullptr);
  kernel_context->MutableInput(2)->Set(&gert_allocator, nullptr);

  ASSERT_EQ(kernel->outputs_creator(split_variable, run_context), ge::GRAPH_SUCCESS);

  ASSERT_NE(kernel->run_func(kernel_context), ge::GRAPH_SUCCESS);  // variable manager not init

  static UtestRt2VarManager manager;
  rt_session.SetVarManager(&manager);
  ASSERT_NE(kernel->run_func(kernel_context), ge::GRAPH_SUCCESS);  // variable not found

  ge::GeTensorDesc var_desc;
  var_desc.SetShape(ge::GeShape({1, 1, 2, 3}));
  var_desc.SetOriginShape(ge::GeShape({2, 3}));
  var_desc.SetPlacement(ge::Placement::kPlacementDevice);

  gert::StorageShape gert_shape;
  gert_shape.MutableOriginShape() = GeShapeToGertShape(var_desc.GetOriginShape());
  gert_shape.MutableStorageShape() = GeShapeToGertShape(var_desc.GetShape());
  void *magic_addr = reinterpret_cast<void*>(0x123abc);
  TensorData tensor_data(magic_addr);
  tensor_data.SetPlacement(TensorPlacement::kOnDeviceHbm);
  manager.SetVarShapeAndMemory(var_id, gert_shape, tensor_data);
  ASSERT_EQ(kernel->run_func(kernel_context), ge::GRAPH_SUCCESS);  // variable has been initialized

  auto storage_shape = kernel_context->GetOutputPointer<StorageShape>(0);
  ASSERT_EQ(storage_shape->GetOriginShape(), GeShapeToGertShape(var_desc.GetOriginShape()));
  ASSERT_EQ(storage_shape->GetStorageShape(), GeShapeToGertShape(var_desc.GetShape()));
  auto storage_shape_tensor = kernel_context->GetOutputPointer<Tensor>(0);
  ASSERT_EQ(storage_shape_tensor->GetOriginShape(), GeShapeToGertShape(var_desc.GetOriginShape()));
  ASSERT_EQ(storage_shape_tensor->GetStorageShape(), GeShapeToGertShape(var_desc.GetShape()));
  ASSERT_EQ(storage_shape_tensor->GetAddr(), nullptr);

  auto gert_td = kernel_context->GetOutputPointer<GertTensorData>(1);
  ASSERT_EQ(gert_td->GetAddr(), tensor_data.GetAddr());
  ASSERT_EQ(gert_td->GetPlacement(), TensorPlacement::kOnDeviceHbm);
  run_context.FreeAll();
}
}  // namespace gert