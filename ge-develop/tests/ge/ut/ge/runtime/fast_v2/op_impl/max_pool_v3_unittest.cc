/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_impl/maxpoolv3/max_pool_v3_impl.h"
#include "register/op_impl_registry.h"
#include <gtest/gtest.h>
#include "faker/kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/kernel_context.h"
#include "exe_graph/runtime/tiling_data.h"
namespace gert {
ge::graphStatus InferShapeForMaxPoolV3(InferShapeContext *context);
}
namespace gert_test {
class MaxPoolV3UT : public testing::Test {};
TEST_F(MaxPoolV3UT, InferShapeOk) {
  ASSERT_NE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("MaxPoolV3"), nullptr);
  auto infer_shape_func = gert::InferShapeForMaxPoolV3;
  ASSERT_NE(infer_shape_func, nullptr);

  gert::StorageShape x_shape = {{1, 4, 56, 56, 16}, {1, 4, 56, 56, 16}};

  std::vector<gert::StorageShape> output_shapes(1);
  std::vector<void *> output_shapes_ref(1);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(1, 1)
                    .IrInstanceNum({1})
                    .InputShapes({&x_shape})
                    .OutputShapes(output_shapes_ref)
                    .NodeAttrs({{"ksize", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 3, 3})},
                                {"strides", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 2, 2})},
                                {"padding_mode", ge::AnyValue::CreateFrom<std::string>("CALCULATED")},
                                {"pads", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                                {"data_format", ge::AnyValue::CreateFrom<std::string>("NCHW")},
                                {"global_pooling", ge::AnyValue::CreateFrom<bool>(false)},
                                {"ceil_mode", ge::AnyValue::CreateFrom<bool>(false)}})
                    .Build();

  EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(MaxPoolV3UT, InferShapeAttrs) {
using namespace gert;
  ASSERT_NE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("MaxPoolV3"), nullptr);
  auto infer_shape_func = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("MaxPoolV3")->infer_shape;
  ASSERT_NE(infer_shape_func, nullptr);

  StorageShape x_shape = {{1, 4, 56, 56, 16}, {1, 4, 56, 56, 16}};
  StorageShape o1;
  auto holder = InferShapeContextFaker()
                    .IrInputNum(1)
                    .InputShapes({&x_shape})
                    .OutputShapes({&o1})
                    .NodeIoNum(1, 1)
                    .NodeAttrs({{"ksize", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 3, 3})},
                                {"strides", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 2, 2})},
                                {"padding_mode", ge::AnyValue::CreateFrom<std::string>("CALCULATED")},
                                {"pads", ge::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                                {"data_format", ge::AnyValue::CreateFrom<std::string>("NCHW")},
                                {"global_pooling", ge::AnyValue::CreateFrom<bool>(false)},
                                {"ceil_mode", ge::AnyValue::CreateFrom<bool>(false)}})
                    .Build();

  auto context = holder.GetContext<InferShapeContext>();

  auto attrs = context->GetAttrs();
  ASSERT_NE(attrs, nullptr);
  ASSERT_EQ(attrs->GetAttrNum(), 7);

  auto attr2 = attrs->GetAttrPointer<char>(2);
  ASSERT_NE(attr2, nullptr);
  EXPECT_STREQ(attr2, "CALCULATED");

  auto attr4 = attrs->GetAttrPointer<char>(4);
  ASSERT_NE(attr4, nullptr);
  EXPECT_STREQ(attr4, "NCHW");

  auto attr5 = attrs->GetAttrPointer<bool>(5);
  ASSERT_NE(attr5, nullptr);
  EXPECT_FALSE(*attr5);

  auto attr6 = attrs->GetAttrPointer<float>(6);
  ASSERT_NE(attr6, nullptr);
  EXPECT_EQ(*attr6, false);

  EXPECT_EQ(attrs->GetAttrPointer<bool>(7), nullptr);
  EXPECT_EQ(infer_shape_func(holder.GetContext<InferShapeContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(MaxPoolV3UT, TilingOk) {
  gert::StorageShape x_shape = {{1, 4, 56, 56, 16}, {1, 4, 56, 56, 16}};

  std::vector<gert::StorageShape> output_shapes(1);
  std::vector<gert::StorageShape *> output_shapes_ref(1);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }
  // compile info
  int32_t ub_ele = 126976;
  int32_t core_num = 32;
  int32_t ksize_h = 3;
  int32_t ksize_w = 3;
  int32_t strides_h = 2;
  int32_t strides_w = 2;
  int32_t padding = 2;    // SAME
  int32_t ceil_mode = 0;  // floor
  int32_t pad_top = 1;
  int32_t pad_bottom = 1;
  int32_t pad_left = 1;
  int32_t pad_right = 1;
  int32_t global = 0;
  gert::MaxPoolV3CompileInfo compile_info;
  compile_info.core_num = core_num;
  compile_info.ksize_h = ksize_h;
  compile_info.ksize_w = ksize_w;
  compile_info.strides_h = strides_h;
  compile_info.strides_w = strides_w;
  compile_info.padding = padding;
  compile_info.ceil_mode = ceil_mode;
  compile_info.pad_top = pad_top;
  compile_info.pad_bottom = pad_bottom;
  compile_info.pad_left = pad_left;
  compile_info.pad_right = pad_right;
  compile_info.global = global;
  compile_info.ub_ele = ub_ele;

  // tiling data
  auto param = gert::TilingData::CreateCap(2048);
  auto holder = gert::TilingContextFaker()
                    .NodeIoNum(1, 1)
                    .IrInstanceNum({1})
                    .InputShapes({&x_shape})
                    .OutputShapes(output_shapes_ref)
                    .CompileInfo(&compile_info)
                    .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0)
                    .TilingData(param.get())
                    .Build();

  ASSERT_NE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("MaxPoolV3"), nullptr);
  auto tiling_func = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("MaxPoolV3")->tiling;
  ASSERT_NE(tiling_func, nullptr);

  EXPECT_EQ(tiling_func(holder.GetContext<gert::TilingContext>()), ge::GRAPH_SUCCESS);
  // todo check tiling result
}

TEST_F(MaxPoolV3UT, TilingParseOk) {
  char *json_str = (char *)"{\"vars\": {\"ub_ele\": 126976, \"core_num\": 32, \"ksize_h\": 3, \"ksize_w\": 3, \"strides_h\": 2, "
      "\"strides_w\": 2, \"padding\": 2, \"ceil_mode\": 0, \"pad_top\": 1, \"pad_bottom\": 1, \"pad_left\": 1, "
      "\"pad_right\": 1, \"global\": 0}}";
  gert::MaxPoolV3CompileInfo compile_info;
  auto holder = gert::KernelRunContextFaker()
                    .KernelIONum(1, 1)
                    .Inputs({json_str})
                    .Outputs({&compile_info})
                    .IrInstanceNum({1})
                    .Build();

  ASSERT_NE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("MaxPoolV3"), nullptr);
  auto tiling_prepare_func = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("MaxPoolV3")->tiling_parse;
  ASSERT_NE(tiling_prepare_func, nullptr);

  EXPECT_EQ(tiling_prepare_func(holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
  int32_t ub_ele = 126976;
  int32_t core_num = 32;
  int32_t ksize_h = 3;
  int32_t ksize_w = 3;
  int32_t strides_h = 2;
  int32_t strides_w = 2;
  int32_t padding = 2;    // SAME
  int32_t ceil_mode = 0;  // floor
  int32_t pad_top = 1;
  int32_t pad_bottom = 1;
  int32_t pad_left = 1;
  int32_t pad_right = 1;
  int32_t global = 0;
  EXPECT_EQ(compile_info.ub_ele, ub_ele);
  EXPECT_EQ(compile_info.core_num, core_num);
  EXPECT_EQ(compile_info.ksize_h, ksize_h);
  EXPECT_EQ(compile_info.ksize_w, ksize_w);
  EXPECT_EQ(compile_info.strides_h, strides_h);
  EXPECT_EQ(compile_info.strides_w, strides_w);
  EXPECT_EQ(compile_info.padding, padding);
  EXPECT_EQ(compile_info.ceil_mode, ceil_mode);
  EXPECT_EQ(compile_info.pad_top, pad_top);
  EXPECT_EQ(compile_info.pad_bottom, pad_bottom);
  EXPECT_EQ(compile_info.pad_left, pad_left);
  EXPECT_EQ(compile_info.pad_right, pad_right);
  EXPECT_EQ(compile_info.global, global);
}

TEST_F(MaxPoolV3UT, DetaDepencyOk) {
  ASSERT_NE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("MaxPoolV3"), nullptr);
  EXPECT_FALSE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("MaxPoolV3")->IsInputDataDependency(0));
  EXPECT_FALSE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("MaxPoolV3")->IsInputDataDependency(1));
  EXPECT_FALSE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("MaxPoolV3")->IsInputDataDependency(2));
  EXPECT_FALSE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("MaxPoolV3")->IsInputDataDependency(3));
}
}  // namespace gert_test