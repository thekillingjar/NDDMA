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

#include "exe_graph/runtime/storage_shape.h"
#include "faker/kernel_run_context_facker.h"
#include "register/kernel_registry.h"

namespace gert {
namespace kernel {
ge::graphStatus CalcTensorSizeFromStorage(KernelContext *context);
ge::graphStatus CalcTensorSizeFromShape(KernelContext *context);
ge::graphStatus CalcUnalignedTensorSizeFromStorage(KernelContext *context);
}  // namespace kernel
struct CalcTensorSizeKernelTest : public testing::Test {
  CalcTensorSizeKernelTest() {
    calcStorageShapeSize = KernelRegistry::GetInstance().FindKernelFuncs("CalcTensorSizeFromStorage");
    calcShapeSize = KernelRegistry::GetInstance().FindKernelFuncs("CalcTensorSizeFromShape");
    calcUnalignedStorageShapeSize = KernelRegistry::GetInstance().FindKernelFuncs("CalcUnalignedTensorSizeFromStorage");
  }
  const KernelRegistry::KernelFuncs *calcStorageShapeSize;
  const KernelRegistry::KernelFuncs *calcShapeSize;
  const KernelRegistry::KernelFuncs *calcUnalignedStorageShapeSize;
};

TEST_F(CalcTensorSizeKernelTest, CalcTensorSizeFromShape_Success_AlignedTo32) {
  StorageShape input{{1, 2, 4}, {2, 2, 4}};

  auto run_context = BuildKernelRunContext(2, 1);
  run_context.value_holder[0].Set((void *)ge::DT_INT32, nullptr);
  run_context.value_holder[1].Set(&input, nullptr);

  ASSERT_EQ(kernel::CalcTensorSizeFromShape(run_context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(run_context.value_holder[2].GetValue<uint64_t>(), 64);
}

TEST_F(CalcTensorSizeKernelTest, CalcTensorSizeFromShape_Success_NotAlignedTo32) {
  StorageShape input{{1, 2, 3}, {1, 2, 3}};

  auto run_context = BuildKernelRunContext(2, 1);
  run_context.value_holder[0].Set((void *)ge::DT_INT32, nullptr);
  run_context.value_holder[1].Set(&input, nullptr);

  ASSERT_EQ(kernel::CalcTensorSizeFromShape(run_context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(run_context.value_holder[2].GetValue<uint64_t>(), 64);
}

TEST_F(CalcTensorSizeKernelTest, CalcUnalignedTensorSizeFromStorage_Success) {
  StorageShape input{{1, 2, 4}, {2, 2, 4}};

  auto run_context = BuildKernelRunContext(2, 1);
  run_context.value_holder[0].Set((void *)ge::DT_INT32, nullptr);
  run_context.value_holder[1].Set(&input, nullptr);

  ASSERT_EQ(calcUnalignedStorageShapeSize->run_func(run_context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(run_context.value_holder[2].GetValue<uint64_t>(), 64);
  auto msgs = calcUnalignedStorageShapeSize->trace_printer(run_context);
  ASSERT_EQ(msgs.size(), 1);
  ASSERT_TRUE(msgs[0].find("Shape") != std::string::npos);
}

TEST_F(CalcTensorSizeKernelTest, FromStorageTracePrinter_ShapeIsNull) {
  auto run_context = BuildKernelRunContext(2, 1);
  run_context.value_holder[0].Set((void *)ge::DT_INT32, nullptr);
  run_context.value_holder[1].Set(nullptr, nullptr);
  auto msgs = calcUnalignedStorageShapeSize->trace_printer(run_context);
  ASSERT_EQ(msgs.size(), 1);
  ASSERT_TRUE(msgs[0].find("storage shape or tensor size is null") != std::string::npos);
}

TEST_F(CalcTensorSizeKernelTest, CalcUnalignedTensorSizeFromStorage_Failed_ShapeSizeOverflow) {
  auto dim_value = INT64_MAX / 16;
  StorageShape input{{dim_value, 2, 4}, {dim_value, 2, 4}};

  auto run_context = BuildKernelRunContext(2, 1);
  run_context.value_holder[0].Set((void *)ge::DT_UINT64, nullptr);
  run_context.value_holder[1].Set(&input, nullptr);

  ASSERT_NE(calcUnalignedStorageShapeSize->run_func(run_context), ge::GRAPH_SUCCESS);
}

TEST_F(CalcTensorSizeKernelTest, CalcTensorSizeFromShape_Failed_ShapeSizeOverflow) {
  StorageShape input{{INT64_MAX, 2, 4}, {INT64_MAX, 2, 4}};

  auto run_context = BuildKernelRunContext(2, 1);
  run_context.value_holder[0].Set((void *)ge::DT_INT8, nullptr);
  run_context.value_holder[1].Set(&input, nullptr);

  ASSERT_NE(calcShapeSize->run_func(run_context), ge::GRAPH_SUCCESS);
}
TEST_F(CalcTensorSizeKernelTest, CalcTensorSizeFromShape_Failed_TensorSizeOverflow) {
  auto dim_value = INT64_MAX / 16;
  StorageShape input{{dim_value, 2, 4}, {dim_value, 2, 4}};

  auto run_context = BuildKernelRunContext(2, 1);
  run_context.value_holder[0].Set((void *)ge::DT_UINT64, nullptr);
  run_context.value_holder[1].Set(&input, nullptr);

  ASSERT_NE(calcShapeSize->run_func(run_context), ge::GRAPH_SUCCESS);
}

TEST_F(CalcTensorSizeKernelTest, CalcTensorSizeFromStorage_Success_AlignInt32) {
  StorageShape input{{1, 2, 4}, {2, 2, 4}};
  auto run_context = BuildKernelRunContext(2, 1);
  run_context.value_holder[0].Set((void *)ge::DT_INT32, nullptr);
  run_context.value_holder[1].Set(&input, nullptr);

  calcStorageShapeSize->run_func(run_context.GetContext<KernelContext>());
  ASSERT_EQ(run_context.value_holder[2].GetValue<uint64_t>(), 96);
  auto msgs = calcStorageShapeSize->trace_printer(run_context);
  ASSERT_EQ(msgs.size(), 1);
  ASSERT_TRUE(msgs[0].find("Shape") != std::string::npos);
}

TEST_F(CalcTensorSizeKernelTest, CalcTensorSizeFromStorage_Success_UnalignInt32) {
  StorageShape input{{1, 2, 4}, {2, 2, 5}};
  auto run_context = BuildKernelRunContext(2, 1);
  run_context.value_holder[0].Set((void *)ge::DT_INT32, nullptr);
  run_context.value_holder[1].Set(&input, nullptr);

  calcStorageShapeSize->run_func(run_context.GetContext<KernelContext>());
  ASSERT_EQ(run_context.value_holder[2].GetValue<uint64_t>(), 128);
}
TEST_F(CalcTensorSizeKernelTest, CalcTensorSizeFromStorage_Success_Int2) {
  StorageShape input{{1, 2, 4}, {2, 2, 4}};
  auto run_context = BuildKernelRunContext(2, 1);
  run_context.value_holder[0].Set((void *)ge::DT_INT2, nullptr);
  run_context.value_holder[1].Set(&input, nullptr);

  calcStorageShapeSize->run_func(run_context);
  ASSERT_EQ(run_context.value_holder[2].GetValue<uint64_t>(), 64);
}

TEST_F(CalcTensorSizeKernelTest, CalcTensorSizeFromStorage_Failed_UnknownShape) {
  StorageShape input{{1, -1, 4}, {2, -2, 4}};
  auto run_context = BuildKernelRunContext(2, 1);
  run_context.value_holder[0].Set((void *)ge::DT_INT2, nullptr);
  run_context.value_holder[1].Set(&input, nullptr);

  ASSERT_NE(calcStorageShapeSize->run_func(run_context), ge::GRAPH_SUCCESS);
}

TEST_F(CalcTensorSizeKernelTest, CalcTensorSizeFromStorage_Failed_UnknownShapeInt2) {
  StorageShape input{{1, -1, 4}, {2, -2, 4}};
  auto run_context = BuildKernelRunContext(2, 1);
  run_context.value_holder[0].Set((void *)ge::DT_INT2, nullptr);
  run_context.value_holder[1].Set(&input, nullptr);

  ASSERT_NE(calcStorageShapeSize->run_func(run_context), ge::GRAPH_SUCCESS);
}

TEST_F(CalcTensorSizeKernelTest, CalcTensorSizeFromStorage_Succ_SupportDT_STRING_REF) {
  StorageShape input{{1, 1, 4}, {2, 2, 4}};
  auto run_context = BuildKernelRunContext(2, 1);
  run_context.value_holder[0].Set((void *)ge::DT_STRING_REF, nullptr);
  run_context.value_holder[1].Set(&input, nullptr);

  // DT_STRING_REF is supported now
  ASSERT_EQ(calcStorageShapeSize->run_func(run_context), ge::GRAPH_SUCCESS);
}

TEST_F(CalcTensorSizeKernelTest, CalcTensorSizeFromStorage_Failed_Nullptr) {
  auto run_context = BuildKernelRunContext(2, 1);
  ASSERT_EQ(calcStorageShapeSize->run_func(run_context), ge::GRAPH_PARAM_INVALID);
}

TEST_F(CalcTensorSizeKernelTest, CalcTensorSizeFromShape_Failed_Nullptr) {
  auto run_context = BuildKernelRunContext(2, 1);
  ASSERT_EQ(calcShapeSize->run_func(run_context), ge::GRAPH_PARAM_INVALID);
}

TEST_F(CalcTensorSizeKernelTest, CalcTensorSizeFromShape_support_DTSTRING) {
  auto run_context = BuildKernelRunContext(2, 1);
  StorageShape input{{1, 1, 4}, {2, 2, 4}};
  run_context.value_holder[0].Set((void *)ge::DT_STRING, nullptr);
  run_context.value_holder[1].Set(&input, nullptr);
  ASSERT_EQ(calcShapeSize->run_func(run_context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(run_context.value_holder[2].GetValue<uint64_t>(), 96);
  auto msgs = calcShapeSize->trace_printer(run_context);
  ASSERT_EQ(msgs.size(), 1);
  ASSERT_TRUE(msgs[0].find("Shape") != std::string::npos);
}

TEST_F(CalcTensorSizeKernelTest, CalcTensorSizeFromShapeTracePrinter_ShapeIsNull) {
  auto run_context = BuildKernelRunContext(2, 1);
  run_context.value_holder[0].Set((void *)ge::DT_STRING, nullptr);
  run_context.value_holder[1].Set(nullptr, nullptr);
  auto msgs = calcShapeSize->trace_printer(run_context);
  ASSERT_EQ(msgs.size(), 1);
  ASSERT_TRUE(msgs[0].find("shape or tensor size is null") != std::string::npos);
}

TEST_F(CalcTensorSizeKernelTest, CalcTensorSizeFromShape_support_DTSTRING_OverFlow) {
  auto run_context = BuildKernelRunContext(2, 1);
  StorageShape input{{std::numeric_limits<int64_t>::max(), 1, 1}, {std::numeric_limits<int64_t>::max(), 1, 1}};
  run_context.value_holder[0].Set((void *)ge::DT_STRING, nullptr);
  run_context.value_holder[1].Set(&input, nullptr);
  ASSERT_EQ(calcShapeSize->run_func(run_context), ge::GRAPH_FAILED);
}

}  // namespace gert