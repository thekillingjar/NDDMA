/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_impl/dynamic_rnn_impl.h"
#include "register/op_impl_registry.h"
#include <gtest/gtest.h>
#include "faker/kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/tiling_data.h"
namespace gert {
ge::graphStatus InferShapeForDynamicRNNV3(InferShapeContext *context);
}
namespace gert_test {
class DynamicRNNV3UT : public testing::Test {};
TEST_F(DynamicRNNV3UT, InferShapeOk) {
  ASSERT_NE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("DynamicRNNV3"), nullptr);
  auto infer_shape_func = gert::InferShapeForDynamicRNNV3;
  ASSERT_NE(infer_shape_func, nullptr);

  gert::StorageShape x_shape = {{1, 16, 256}, {1, 16, 1, 16, 16}};
  gert::StorageShape w_shape = {{768, 2048}, {48, 128, 16, 16}};
  gert::StorageShape b_shape = {{2048}, {2048}};
  gert::StorageShape init_h_shape = {{1, 16, 512}, {1, 32, 1, 16, 16}};
  gert::StorageShape init_c_shape = {{1, 16, 512}, {1, 32, 1, 16, 16}};
  gert::StorageShape project_shape = {{512, 512}, {32, 32, 16, 16}};

  std::vector<gert::StorageShape> output_shapes(8);
  std::vector<void *> output_shapes_ref(8);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(6, 8)
                    .IrInstanceNum({1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1})
                    .InputShapes({&x_shape, &w_shape, &b_shape, &init_h_shape, &init_c_shape, &project_shape})
                    .OutputShapes(output_shapes_ref)
                    .Build();

  EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  // todo check more output shapes
}

TEST_F(DynamicRNNV3UT, TilingOk) {
  gert::StorageShape x_shape = {{1, 16, 256}, {1, 16, 1, 16, 16}};
  gert::StorageShape w_shape = {{768, 2048}, {48, 128, 16, 16}};
  gert::StorageShape b_shape = {{2048}, {2048}};
  gert::StorageShape init_h_shape = {{1, 16, 512}, {1, 32, 1, 16, 16}};
  gert::StorageShape init_c_shape = {{1, 16, 512}, {1, 32, 1, 16, 16}};
  gert::StorageShape project_shape = {{512, 512}, {32, 32, 16, 16}};

  std::vector<gert::StorageShape> output_shapes(8, {{1, 16, 512}, {1, 32, 1, 16, 16}});
  std::vector<gert::StorageShape *> output_shapes_ref(8);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }
  // compile info
  gert::DynamicRNNV3CompileInfo compile_info;
  compile_info.tune_shape_list = {{100, 2}, {200, 3}, {-1, -1}};

  // tiling data
  auto param = gert::TilingData::CreateCap(2048);
  auto holder = gert::TilingContextFaker()
                    .NodeIoNum(6, 8)
                    .IrInstanceNum({1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1})
                    .InputShapes({&x_shape, &w_shape, &b_shape, &init_h_shape, &init_c_shape, &project_shape})
                    .OutputShapes(output_shapes_ref)
                    .CompileInfo(&compile_info)
                    .TilingData(param.get())
                    .Build();

  ASSERT_NE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("DynamicRNNV3"), nullptr);
  auto tiling_func = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("DynamicRNNV3")->tiling;
  ASSERT_NE(tiling_func, nullptr);

  EXPECT_EQ(tiling_func(holder.GetContext<gert::TilingContext>()), ge::GRAPH_SUCCESS);
  // todo check tiling result
}

TEST_F(DynamicRNNV3UT, TilingParseOk) {
  char *json_str = (char *)"{\"vars\": {\"tune_shape_list\":[[-1,-1,0]]}}";
  gert::DynamicRNNV3CompileInfo compile_info;
  auto holder = gert::KernelRunContextFaker()
                    .KernelIONum(1, 1)
                    .Inputs({json_str})
                    .Outputs({&compile_info})
                    .IrInstanceNum({1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1})
                    .Build();

  ASSERT_NE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("DynamicRNNV3"), nullptr);
  auto tiling_parse_func = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("DynamicRNNV3")->tiling_parse;
  ASSERT_NE(tiling_parse_func, nullptr);

  EXPECT_EQ(tiling_parse_func(holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
  std::vector<std::vector<int64_t>> expect_value{{-1,-1,0}};
  EXPECT_EQ(compile_info.tune_shape_list, expect_value);
}

TEST_F(DynamicRNNV3UT, DetaDepencyOk) {
  ASSERT_NE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("DynamicRNNV3"), nullptr);
  EXPECT_FALSE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("DynamicRNNV3")->IsInputDataDependency(0));
  EXPECT_FALSE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("DynamicRNNV3")->IsInputDataDependency(1));
  EXPECT_FALSE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("DynamicRNNV3")->IsInputDataDependency(2));
  EXPECT_FALSE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("DynamicRNNV3")->IsInputDataDependency(3));
}
}  // namespace gert_test