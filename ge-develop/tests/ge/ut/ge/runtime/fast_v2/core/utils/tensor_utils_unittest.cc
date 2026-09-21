/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "core/utils/tensor_utils.h"
#include <gtest/gtest.h>
#include "kernel/memory/host_mem_allocator.h"
#include "kernel/memory/single_stream_l2_allocator.h"
#include "kernel/memory/caching_mem_allocator.h"
#include "faker/multi_stream_allocator_faker.h"

namespace gert {
class TensorUtilsUT : public testing::Test {};
TEST_F(TensorUtilsUT, CheckShape_Ok_WithShapeInPositiveRange) {
  Shape min_shape = {1, 2, 4};
  Shape max_shape = {4, 5, 6};
  Shape shape({3, 4, 5});
  EXPECT_EQ(TensorUtils::CheckShapeByShapeRange(shape, {min_shape, max_shape}), ge::SUCCESS);
}
TEST_F(TensorUtilsUT, CheckShape_Ok_WithShapeInNegativeRange) {
  Shape min_shape = {1, 2, 4};
  Shape max_shape = {-1, -1, 6};
  Shape shape({999, 999, 6});
  EXPECT_EQ(TensorUtils::CheckShapeByShapeRange(shape, {min_shape, max_shape}), ge::SUCCESS);
}
TEST_F(TensorUtilsUT, CheckShape_Ok_WithUnknownDimInShape) {
  Shape min_shape = {1, 2, 4};
  Shape max_shape = {4, 5, 6};
  Shape shape({-1, 2, 6});
  EXPECT_EQ(TensorUtils::CheckShapeByShapeRange(shape, {min_shape, max_shape}), ge::SUCCESS);
}
TEST_F(TensorUtilsUT, CheckShape_Fail_WithInvalidShapeRange) {
  Shape min_shape1 = {-1, 4, 5};
  Shape max_shape1 = {4, 4, 5};
  Shape shape1({3, 4, 5});
  EXPECT_NE(TensorUtils::CheckShapeByShapeRange(shape1, {min_shape1, max_shape1}), ge::SUCCESS);
  Shape min_shape = {2, 4, 5};
  Shape max_shape = {4, 2, 5};
  Shape shape({3, 3, 5});
  EXPECT_NE(TensorUtils::CheckShapeByShapeRange(shape, {min_shape, max_shape}), ge::SUCCESS);
}
TEST_F(TensorUtilsUT, CheckShape_Fail_WithDimBiggerThanMaxRange) {
  Shape min_shape = {1, 5, 3};
  Shape max_shape = {4, 5, 4};
  Shape shape({6, 6, 6});
  EXPECT_NE(TensorUtils::CheckShapeByShapeRange(shape, {min_shape, max_shape}), ge::SUCCESS);
}
TEST_F(TensorUtilsUT, CheckShape_Fail_WithDimSmallerThanMinRange) {
  Shape min_shape = {2, 5, 3};
  Shape max_shape = {4, 5, 4};
  Shape shape({2, 2, 2});
  EXPECT_NE(TensorUtils::CheckShapeByShapeRange(shape, {min_shape, max_shape}), ge::SUCCESS);
}
TEST_F(TensorUtilsUT, CheckShape_Fail_WithShapeSizeSmallerThanRangeSize) {
  Shape min_shape = {2, 5, 3};
  Shape max_shape = {4, 5, 4};
  Shape shape({2, 2});
  EXPECT_NE(TensorUtils::CheckShapeByShapeRange(shape, {min_shape, max_shape}), ge::SUCCESS);
}
TEST_F(TensorUtilsUT, HostAddrManager_ParamInvalid) {
  void *out = nullptr;
  auto ret = HostAddrManager(nullptr, kGetTensorAddress, &out);
  EXPECT_NE(ret, ge::SUCCESS);

  void *block = (void *)0x10;
  ret = HostAddrManager(block, kTensorOperateType, &out);
  EXPECT_NE(ret, ge::SUCCESS);
}
TEST_F(TensorUtilsUT, ToGertTensorData_Host_ShareFrom_Free_success) {
  auto l1_allocator = std::make_shared<memory::HostMemAllocator>();
  auto host_gert_mem_allocator1 = memory::HostGertMemAllocator(l1_allocator.get());

  auto gert_mem_block = reinterpret_cast<memory::MultiStreamMemBlock *>(host_gert_mem_allocator1.Malloc(100U));
  auto gert_tensor_data1 = TensorUtils::ToGertTensorData(gert_mem_block, kOnHost, 0);

  auto gert_tensor_data2 = GertTensorData();
  ASSERT_EQ(gert_tensor_data1.GetAddr(), gert_mem_block->GetAddr());

  gert_tensor_data2.ShareFrom(gert_tensor_data1);
  ASSERT_EQ(gert_tensor_data2.GetAddr(), gert_mem_block->GetAddr());
  gert_tensor_data1.Free();
  ASSERT_EQ(gert_tensor_data1.GetAddr(), nullptr);
  ASSERT_NE(gert_mem_block->GetAddr(), nullptr);

  gert_tensor_data2.Free();
  ASSERT_EQ(gert_tensor_data2.GetAddr(), nullptr);
}

TEST_F(TensorUtilsUT, ToGertTensorData_Caching_ShareFrom_Free_success) {
  auto l1_allocator = std::make_shared<memory::CachingMemAllocator>(0);
  auto single_stream_l2_allocator = memory::SingleStreamL2Allocator(l1_allocator.get());

  auto gert_mem_block = reinterpret_cast<memory::MultiStreamMemBlock *>(single_stream_l2_allocator.Malloc(100U));
  auto gert_tensor_data1 = TensorUtils::ToGertTensorData(gert_mem_block, kOnDeviceHbm, 0);

  auto gert_tensor_data2 = GertTensorData();
  ASSERT_EQ(gert_tensor_data1.GetAddr(), gert_mem_block->GetAddr());

  gert_tensor_data2.ShareFrom(gert_tensor_data1);
  ASSERT_EQ(gert_tensor_data2.GetAddr(), gert_mem_block->GetAddr());
  gert_tensor_data1.Free();
  ASSERT_EQ(gert_tensor_data1.GetAddr(), nullptr);
  ASSERT_NE(gert_mem_block->GetAddr(), nullptr);

  gert_tensor_data2.Free();
  ASSERT_EQ(gert_tensor_data2.GetAddr(), nullptr);
}

TEST_F(TensorUtilsUT, RefTdToGtd_Success) {
  auto addr = (void *)0x2000;
  size_t size = 1024U;
  TensorPlacement placement = kOnDeviceHbm;
  TensorData td{addr, nullptr, size, placement};
  GertTensorData gtd;
  ASSERT_EQ(gtd.GetAddr(), nullptr);
  ASSERT_EQ(TensorUtils::RefTdToGtd(td, 0, gtd), ge::GRAPH_SUCCESS);
  ASSERT_EQ(gtd.GetAddr(), addr);
  ASSERT_EQ(gtd.GetSize(), size);
  ASSERT_EQ(gtd.GetPlacement(), placement);
  ASSERT_EQ(gtd.GetStreamId(), 0);
}

TEST_F(TensorUtilsUT, RefTdToGtd_kFollowing_Failed) {
  auto addr = (void *)0x2000;
  size_t size = 1024U;
  TensorPlacement placement = kFollowing;
  TensorData td{addr, nullptr, size, placement};
  GertTensorData gtd;
  ASSERT_EQ(gtd.GetAddr(), nullptr);
  ASSERT_EQ(TensorUtils::RefTdToGtd(td, 0, gtd), ge::GRAPH_FAILED);
}

TEST_F(TensorUtilsUT, RefTensorToGtd_kFollowing_Success) {
  auto addr = (void *)0x2000;
  size_t size = 1024U;
  TensorPlacement placement = kFollowing;
  TensorData td{addr, nullptr, size, placement};
  Tensor tensor;
  tensor.SetData(std::move(td));
  GertTensorData gtd;
  ASSERT_EQ(gtd.GetAddr(), nullptr);
  ASSERT_EQ(TensorUtils::RefTensorToGtd(tensor, 0, gtd), ge::GRAPH_SUCCESS);
  ASSERT_EQ(gtd.GetAddr(), tensor.GetAddr());
  ASSERT_EQ(gtd.GetSize(), size);
  ASSERT_EQ(gtd.GetPlacement(), kOnHost);
  ASSERT_EQ(gtd.GetStreamId(), 0);
}

TEST_F(TensorUtilsUT, ShareTdToGtd_Success) {
  auto holder = memory::MultiStreamAllocatorFaker().StreamNum(1).Build();
  size_t size = 1024U;
  auto mem_block = holder.l1_allocator->Malloc(size);
  auto td = TensorUtils::ToTensorData(mem_block, size, kOnDeviceHbm);
  GertTensorData gtd;
  ASSERT_EQ(TensorUtils::ShareTdToGtd(td, *holder.at(0), gtd), ge::GRAPH_SUCCESS);
  ASSERT_EQ(gtd.GetAddr(), td.GetAddr());
  ASSERT_EQ(gtd.GetSize(), td.GetSize());
  ASSERT_EQ(gtd.GetPlacement(), td.GetPlacement());
  ASSERT_EQ(gtd.GetStreamId(), 0);
}

TEST_F(TensorUtilsUT, RefGtdToTd_Success) {
  auto addr = (void *)0x2000;
  size_t size = 1024U;
  TensorPlacement placement = kOnDeviceHbm;
  GertTensorData gtd{addr, size, placement, 0};
  TensorData td;
  TensorUtils::RefGtdToTd(gtd, td);
  ASSERT_EQ(td.GetAddr(), addr);
  ASSERT_EQ(td.GetSize(), size);
  ASSERT_EQ(td.GetPlacement(), placement);
}

TEST_F(TensorUtilsUT, ShareGtdToTd_Success) {
  auto holder = memory::MultiStreamAllocatorFaker().StreamNum(1).Build();
  auto gtd = holder.at(0)->MallocTensorData(1024U);
  TensorData td;
  TensorUtils::ShareGtdToTd(gtd, td);
  ASSERT_EQ(td.GetAddr(), gtd.GetAddr());
  ASSERT_EQ(td.GetSize(), gtd.GetSize());
  ASSERT_EQ(td.GetPlacement(), gtd.GetPlacement());
}
}  // namespace gert