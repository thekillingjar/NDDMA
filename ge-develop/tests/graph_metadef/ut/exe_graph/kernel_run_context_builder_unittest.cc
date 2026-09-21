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
#include "lowering/kernel_run_context_builder.h"
#include "register/op_impl_registry.h"
#include "faker/space_registry_faker.h"
namespace gert {
class KernelRunContextBuilderUT : public testing::Test {};

TEST_F(KernelRunContextBuilderUT, SetBufferPoolOk) {
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("test0", "test1");
  KernelRunContextBuilder builder;
  ge::graphStatus ret = ge::GRAPH_FAILED;
  auto holder = builder.Build(op_desc, ret);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto compute_node_info = reinterpret_cast<const ComputeNodeInfo *>(
      holder.context_->GetComputeNodeExtend());
  EXPECT_EQ(std::string(compute_node_info->GetNodeName()), "test0");
  EXPECT_EQ(std::string(compute_node_info->GetNodeType()), "test1");
}

TEST_F(KernelRunContextBuilderUT, SetInputsOutputsOk) {
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("test0", "test1");
  KernelRunContextBuilder builder;
  gert::StorageShape shape1({1,2,3,4}, {1,2,3,4});
  gert::StorageShape shape2({2,2,3,4}, {2,2,3,4});
  gert::StorageShape shape3({3,2,3,4}, {3,2,3,4});
  ge::graphStatus ret = ge::GRAPH_FAILED;
  auto holder = builder.Inputs({{&shape1, nullptr}, {&shape2, nullptr}}).Outputs({&shape3}).Build(op_desc, ret);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  auto context = holder.context_;
  EXPECT_EQ(context->GetInputNum(), 2);
  EXPECT_EQ(context->GetOutputNum(), 1);
  EXPECT_TRUE(context->GetInputPointer<StorageShape>(0) == &shape1);
  EXPECT_TRUE(context->GetInputPointer<StorageShape>(1) == &shape2);
  EXPECT_TRUE(context->GetOutputPointer<StorageShape>(0) == &shape3);
}

TEST_F(KernelRunContextBuilderUT, SetInputsOutputsDataTypeOk) {
ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("test0", "test1");
KernelRunContextBuilder builder;

ge::DataType in_datatype_1 = ge::DT_INT4;
ge::DataType in_datatype_2 = ge::DT_INT8;
ge::DataType out_datatype = ge::DT_INT8;
ge::graphStatus ret = ge::GRAPH_FAILED;
auto holder = builder
                  .Inputs({{reinterpret_cast<void *>(in_datatype_1), nullptr},
                           {reinterpret_cast<void *>(in_datatype_2), nullptr}})
                  .Outputs({reinterpret_cast<void *>(out_datatype)})
                  .Build(op_desc, ret);
EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
auto context = holder.context_;
EXPECT_EQ(context->GetInputNum(), 2);
EXPECT_EQ(context->GetOutputNum(), 1);
EXPECT_TRUE(*context->GetInputPointer<ge::DataType>(0) == in_datatype_1);
EXPECT_TRUE(*context->GetInputPointer<ge::DataType>(1) == in_datatype_2);
EXPECT_TRUE(*context->GetOutputPointer<ge::DataType>(0) == out_datatype);
}

TEST_F(KernelRunContextBuilderUT, BuildContextHolderSuccessWhenOpLossAttrs) {
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("test0", "test1");
  op_desc->AppendIrAttrName("attr1");
  KernelRunContextBuilder builder;
  ge::graphStatus ret = ge::GRAPH_FAILED;
  auto holder = builder.Build(op_desc, ret);
  ASSERT_EQ(ret, ge::GRAPH_SUCCESS);
  ASSERT_NE(holder.context_holder_, nullptr);
  EXPECT_NE(holder.compute_node_extend_holder_, nullptr);
}

TEST_F(KernelRunContextBuilderUT, Get2AttrsFromCtx_OpBothHas1IrAttrAnd1PrivateAttr) {
  SpaceRegistryFaker::SetefaultSpaceRegistryNull();
  IMPL_OP(TestOpWithPrivateAttr1).PrivateAttr("test_private_attr", static_cast<int64_t>(100));
  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2(true);
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("testop1", "TestOpWithPrivateAttr1");
  op_desc->AppendIrAttrName("test_ir_attr");
  ge::AttrUtils::SetInt(op_desc, "test_ir_attr", 10);
  ge::AttrUtils::SetInt(op_desc, "test_private_attr", 1000);
  KernelRunContextBuilder builder;
  ge::graphStatus ret = ge::GRAPH_FAILED;
  auto holder = builder.Build(op_desc, ret);
  ASSERT_EQ(ret, ge::GRAPH_SUCCESS);
  ASSERT_NE(holder.context_holder_, nullptr);
  ASSERT_NE(holder.compute_node_extend_holder_, nullptr);

  auto kernel_ctx = holder.GetKernelContext();
  ASSERT_NE(kernel_ctx, nullptr);
  auto tiling_ctx = reinterpret_cast<TilingContext *>(kernel_ctx);
  auto runtime_attrs = tiling_ctx->GetAttrs();
  ASSERT_NE(runtime_attrs, nullptr);
  ASSERT_EQ(runtime_attrs->GetAttrNum(), 2U);
  auto attr0 = runtime_attrs->GetAttrPointer<int64_t>(0);
  EXPECT_EQ(*attr0, 10);
  auto attr1 = runtime_attrs->GetAttrPointer<int64_t>(1);
  EXPECT_EQ(*attr1, 1000);
  SpaceRegistryFaker::SetefaultSpaceRegistryNull();
}

TEST_F(KernelRunContextBuilderUT, OnlyGetIrAttrsFromCtx_OpNotRegisterPrivateAttr) {
  SpaceRegistryFaker::UpdateOpImplToDefaultSpaceRegistry();
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("testop1", "TestOpWithPrivateAttr2");
  op_desc->AppendIrAttrName("test_ir_attr");
  ge::AttrUtils::SetInt(op_desc, "test_ir_attr", 10);
  ge::AttrUtils::SetInt(op_desc, "test_private_attr", 1000);
  KernelRunContextBuilder builder;
  ge::graphStatus ret = ge::GRAPH_FAILED;
  auto holder = builder.Build(op_desc, ret);
  ASSERT_EQ(ret, ge::GRAPH_SUCCESS);
  ASSERT_NE(holder.context_holder_, nullptr);
  ASSERT_NE(holder.compute_node_extend_holder_, nullptr);

  auto kernel_ctx = holder.GetKernelContext();
  ASSERT_NE(kernel_ctx, nullptr);
  auto tiling_ctx = reinterpret_cast<TilingContext *>(kernel_ctx);
  auto runtime_attrs = tiling_ctx->GetAttrs();
  ASSERT_NE(runtime_attrs, nullptr);
  ASSERT_EQ(runtime_attrs->GetAttrNum(), 1U);
  auto attr0 = runtime_attrs->GetAttrPointer<int64_t>(0);
  EXPECT_EQ(*attr0, 10);
  SpaceRegistryFaker::SetefaultSpaceRegistryNull();
}
}  // namespace gert
