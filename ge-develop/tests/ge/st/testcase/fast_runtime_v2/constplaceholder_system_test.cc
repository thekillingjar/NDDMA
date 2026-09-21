/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "runtime/gert_api.h"
#include <gtest/gtest.h>

#include "common/share_graph.h"
#include "faker/ge_model_builder.h"
#include "faker/fake_value.h"
#include "stub/gert_runtime_stub.h"
#include "graph/types.h"
#include "lowering/model_converter.h"
#include "graph/any_value.h"
#include "graph/operator_reg.h"
namespace ge {
REG_OP(ConstPlaceHolder)
    .OUTPUT(y, TensorType::ALL())
    .REQUIRED_ATTR(origin_shape, ListInt)
    .REQUIRED_ATTR(origin_format, Int)
    .REQUIRED_ATTR(storage_shape, ListInt)
    .REQUIRED_ATTR(storage_format, Int)
    .REQUIRED_ATTR(expand_dim_rules, String)
    .REQUIRED_ATTR(dtype, Type)
    .REQUIRED_ATTR(addr, Int)
    .REQUIRED_ATTR(size, Int)
    .ATTR(placement, Int, 1)
    .OP_END_FACTORY_REG(ConstPlaceHolder)
}
namespace gert {
class ConstPlaceHolderST : public testing::Test {};

TEST_F(ConstPlaceHolderST, ConstPlaceHolderSTOK) {
  auto ConstPlaceHolderData = FakeTensors({5, 5}, 1);
  auto const_place_holder_addr = ConstPlaceHolderData.GetTensorList()[0]->GetAddr();
  auto graph = ShareGraph::BuildSingleConstPlaceHolderGraph(const_place_holder_addr, 100L);
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.BuildGeRootModel();

  LoweringOption option;
  ModelConverter::Args args(option, nullptr, nullptr, nullptr, nullptr);
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model, args);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);

  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto output = TensorFaker().StorageShape({5 ,5}).OriginShape({5, 5}).Build();
  output.GetTensor()->SetPlacement(kOnDeviceHbm);
  std::vector<Tensor *> outputs = {output.GetTensor()};

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  runtime_stub.Clear();
  ASSERT_EQ(model_executor->Execute({i3.value}, {}, 0, outputs.data(), outputs.size()),
            ge::GRAPH_SUCCESS);

  auto &launch_args = runtime_stub.GetRtsRuntimeStub().GetLaunchWithHandleArgs();
  ASSERT_EQ(launch_args.size(), 0U);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}
}  // namespace gert