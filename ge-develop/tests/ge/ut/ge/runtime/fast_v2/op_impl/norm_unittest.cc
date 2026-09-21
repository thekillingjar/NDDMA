/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_impl/autotiling/norm.h"
#include "register/op_impl_registry.h"
#include <gtest/gtest.h>
#include "faker/kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tiling_data.h"
namespace gert {
ge::graphStatus NormTiling(TilingContext *context);
}
namespace gert_test {
class NormUT : public testing::Test {};

enum TilingOutputIndex {
  kOutputTilingKey,
  kOutputBlockDim,
  kOutputAtomicCleanFlag,
  kOutputTilingData,
  kOutputWorkspace,

  // add new output definitions here
  kOutputNum
};

TEST_F(NormUT, TilinglistattrOk) {
  gert::StorageShape input_shape = {{11, 20, 512}, {11, 20, 512}};

  std::vector<gert::StorageShape> output_shapes(1, {{11, 20, 512}, {11, 20, 512}});
  std::vector<gert::StorageShape *> output_shapes_ref(1);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }
  // compile info
  gert::NormCompileInfo op_compile_info;
  op_compile_info.input_type = {0};
  op_compile_info.reduce_attr_index = 0;
  op_compile_info.is_reduce_attr_is_int = false;
  op_compile_info.reduce_axis_type = 9;
  op_compile_info.core_num = 32;
  op_compile_info.min_block_size = 16;
  op_compile_info.pad_max_entire_size = 128;
  op_compile_info.exist_output_after_reduce = false;
  op_compile_info.exist_workspace_after_reduce = false;
  op_compile_info.available_ub_size = {{4000, {15824, 16360, 15824}}};
  op_compile_info.workspace_info = {{300400000, {32}}};
  op_compile_info.norm_vars = {{300400000, {20000, 20001, 30000, 40000}}};
  op_compile_info.is_fuse_axis = true;

  // tiling data
  auto param = gert::TilingData::CreateCap(2048);
  auto workspace_holder = gert::ContinuousVector::Create<size_t>(8);
  auto workspace = reinterpret_cast<gert::ContinuousVector *>(workspace_holder.get());
  auto holder = gert::TilingContextFaker()
                    .NodeIoNum(1, 1)
                    .IrInstanceNum({1})
                    .InputShapes({&input_shape})
                    .OutputShapes(output_shapes_ref)
                    .CompileInfo(&op_compile_info)
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({{"list_int", ge::AnyValue::CreateFrom<std::vector<int64_t>>({2})}})
                    .TilingData(param.get())
                    .Workspace(workspace)
                    .Build();

  ASSERT_NE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("DefaultImpl"), nullptr);
  auto tiling_func = gert::NormTiling;
  ASSERT_NE(tiling_func, nullptr);

  gert::TilingContext *context = holder.GetContext<gert::TilingContext>();
  ge::Status status = tiling_func(context);

  EXPECT_EQ(workspace->GetSize(), 1); // 生成一块workspace
  EXPECT_EQ(reinterpret_cast<const size_t *>(workspace->GetData())[0], 32); // workspace的大小是32

  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto data = reinterpret_cast<const int32_t *>(context->GetRawTilingData()->GetData());
  size_t data_num = context->GetRawTilingData()->GetDataSize() / sizeof(int32_t);
  int expect_data[4] = {220, 512, 7, 7};
  for (size_t i = 0; i < data_num; i++) {
    EXPECT_EQ(data[i], expect_data[i]);
  }
}

TEST_F(NormUT, TilingOk) {
  gert::StorageShape input_shape = {{11, 20, 512}, {11, 20, 512}};

  std::vector<gert::StorageShape> output_shapes(1, {{11, 20, 512}, {11, 20, 512}});
  std::vector<gert::StorageShape *> output_shapes_ref(1);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }
  // compile info
  gert::NormCompileInfo op_compile_info;
  op_compile_info.input_type = {0};
  op_compile_info.reduce_attr_index = 0;
  op_compile_info.is_reduce_attr_is_int = true;
  op_compile_info.reduce_axis_type = 9;
  op_compile_info.core_num = 32;
  op_compile_info.min_block_size = 16;
  op_compile_info.pad_max_entire_size = 128;
  op_compile_info.exist_output_after_reduce = false;
  op_compile_info.exist_workspace_after_reduce = false;
  op_compile_info.available_ub_size = {{4000, {15824, 16360, 15824}}};
  op_compile_info.workspace_info = {{300400000, {32}}};
  op_compile_info.norm_vars = {{300400000, {20000, 20001, 30000, 40000}}};
  op_compile_info.is_fuse_axis = true;

  // tiling data
  auto param = gert::TilingData::CreateCap(2048);
  auto workspace_holder = gert::ContinuousVector::Create<size_t>(8);
  auto workspace = reinterpret_cast<gert::ContinuousVector *>(workspace_holder.get());
  auto holder = gert::TilingContextFaker()
                    .NodeIoNum(1, 1)
                    .IrInstanceNum({1})
                    .InputShapes({&input_shape})
                    .OutputShapes(output_shapes_ref)
                    .CompileInfo(&op_compile_info)
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({{"int", ge::AnyValue::CreateFrom<int64_t>(-1)}})
                    .TilingData(param.get())
                    .Workspace(workspace)
                    .Build();

  ASSERT_NE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("DefaultImpl"), nullptr);
  auto tiling_func = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("DefaultImpl")->tiling;
  ASSERT_NE(tiling_func, nullptr);

  gert::TilingContext *context = holder.GetContext<gert::TilingContext>();
  ge::Status status = tiling_func(context);

  EXPECT_EQ(workspace->GetSize(), 1); // 生成一块workspace
  EXPECT_EQ(reinterpret_cast<const size_t *>(workspace->GetData())[0], 32); // workspace的大小是32

  EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  auto data = reinterpret_cast<const int32_t *>(context->GetRawTilingData()->GetData());
  size_t data_num = context->GetRawTilingData()->GetDataSize() / sizeof(int32_t);
  int expect_data[4] = {220, 512, 7, 7};
  for (size_t i = 0; i < data_num; i++) {
    EXPECT_EQ(data[i], expect_data[i]);
  }
}

template<typename T1, typename T2>
static bool compare_map(const std::unordered_map<T1, T2>& map1, const std::unordered_map<T1, T2>& map2) {
  if (map1.size() != map2.size()) {
    return false;
  }
  for (const auto& it: map1) {
    if (map2.count(it.first) == 0) {
      return false;
    }
    if (map1.at(it.first) != map2.at(it.first)) {
      return false;
    }
  }
  return true;
}

static bool compare_norm_struct(const gert::NormCompileInfo& actual_struct,
                                const gert::NormCompileInfo& expect_struct) {

  if (actual_struct.ori_reduce_axis != expect_struct.ori_reduce_axis) {
    return false;
  }
  if (actual_struct.ori_broadcast_axis != expect_struct.ori_broadcast_axis) {
    return false;
  }
  if (actual_struct.is_broadcast_axis_known != expect_struct.is_broadcast_axis_known) {
    return false;
  }
  if (actual_struct.input_type != expect_struct.input_type) {
    return false;
  }
  if (actual_struct.exist_output_after_reduce != expect_struct.exist_output_after_reduce) {
    return false;
  }
  if (actual_struct.exist_workspace_after_reduce != expect_struct.exist_workspace_after_reduce) {
    return false;
  }
  if (!compare_map(actual_struct.available_ub_size, expect_struct.available_ub_size)) {
    return false;
  }
  if (actual_struct.core_num != expect_struct.core_num) {
    return false;
  }
  if (actual_struct.min_block_size != expect_struct.min_block_size) {
    return false;
  }
  if (actual_struct.pad_max_entire_size != expect_struct.pad_max_entire_size) {
    return false;
  }
  if (!compare_map(actual_struct.workspace_info, expect_struct.workspace_info)) {
    return false;
  }
  if (!compare_map(actual_struct.norm_vars, expect_struct.norm_vars)) {
    return false;
  }
  if (actual_struct.is_fuse_axis != expect_struct.is_fuse_axis) {
    return false;
  }
  if (actual_struct.is_const != expect_struct.is_const) {
    return false;
  }
  if (actual_struct.is_const_post != expect_struct.is_const_post) {
    return false;
  }
  if (!compare_map(actual_struct.const_block_dims, expect_struct.const_block_dims)) {
    return false;
  }
  if (actual_struct.reduce_attr_index != expect_struct.reduce_attr_index) {
    return false;
  }
  if (actual_struct.is_reduce_attr_is_int != expect_struct.is_reduce_attr_is_int) {
    return false;
  }
  if (actual_struct.reduce_axis_type != expect_struct.reduce_axis_type) {
    return false;
  }
  if (actual_struct.broadcast_axis_type != expect_struct.broadcast_axis_type) {
    return false;
  }
  return true;
}

TEST_F(NormUT, TilingParseOk) {
  std::string json_str = R"({"reduce_axis_attr_index": 0, "reduce_axis_attr_dtype": "ListInt", "_reduce_axis_type": 9, "_broadcast_axis_type_list": [1, 2], "_fuse_axis": false, "_input_type": [0], "_ori_reduce_axis": [1], "_ori_broadcast_axis": [1], "_pattern": "Norm", "_common_info": [32, 8, 128], "_available_ub_size": {"4000": [15792, 16120, 15792]}, "_is_const": true, "_const_shape_post": true, "_const_block_dims": {"4000": 25}, "_exist_workspace_after_reduce": false, "_exist_output_after_reduce": false})";

  char *json = const_cast<char *>(json_str.c_str());
  gert::NormCompileInfo compile_info;
  auto holder = gert::KernelRunContextFaker()
                    .KernelIONum(1, 1)
                    .Inputs({json})
                    .Outputs({&compile_info})
                    .IrInstanceNum({1})
                    .Build();

  ASSERT_NE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("DefaultImpl"), nullptr);
  auto tiling_parse_func = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("DefaultImpl")->tiling_parse;
  ASSERT_NE(tiling_parse_func, nullptr);
  EXPECT_EQ(tiling_parse_func(holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
  gert::NormCompileInfo expect_struct;
  expect_struct.reduce_attr_index = 0;
  expect_struct.is_reduce_attr_is_int = false;
  expect_struct.reduce_axis_type = 9;
  expect_struct.broadcast_axis_type = {1, 2};
  expect_struct.input_type = {0};
  expect_struct.ori_reduce_axis = {1};
  expect_struct.ori_broadcast_axis = {1};
  expect_struct.is_broadcast_axis_known = true;
  expect_struct.core_num = 32;
  expect_struct.min_block_size = 8;
  expect_struct.pad_max_entire_size = 128;
  expect_struct.exist_output_after_reduce = false;
  expect_struct.exist_workspace_after_reduce = false;
  expect_struct.available_ub_size = {{4000, {15792, 16120, 15792}}};
  expect_struct.is_fuse_axis = false;
  expect_struct.is_const = true;
  expect_struct.is_const_post = true;
  expect_struct.const_block_dims = {{4000, 25}};
  ASSERT_TRUE(compare_norm_struct(compile_info, expect_struct));
}

}  // namespace gert_test