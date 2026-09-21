/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/checker.h"
#include <gtest/gtest.h>
#include "runtime/base.h"
#include "hyper_status.h"
namespace {
template<typename T>
T JustReturn(T val) {
  return val;
}

ge::graphStatus StatusFuncUseStatusFunc(ge::graphStatus val) {
  GE_ASSERT_SUCCESS(JustReturn(val));
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus StatusFuncUseBoolFunc(bool val) {
  GE_ASSERT_TRUE(JustReturn(val));
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus StatusFuncUsePointerFunc(void *val) {
  GE_ASSERT_NOTNULL(JustReturn(val));
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus StatusFuncUseUniquePtrFunc(std::unique_ptr<uint8_t[]> val) {
  GE_ASSERT_NOTNULL(JustReturn(std::move(val)));
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus StatusFuncUseSharedPtrFunc(const std::shared_ptr<uint8_t[]> &val) {
  GE_ASSERT_NOTNULL(JustReturn(val));
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus StatusFuncUseEOKFunc(int val) {
  GE_ASSERT_EOK(JustReturn(val));
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus StatusFuncUseRtFunc(int32_t val) {
  GE_ASSERT_RT_OK(JustReturn(val));
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus StatusFuncUseHyperStatusFunc(gert::HyperStatus val) {
  GE_ASSERT_HYPER_SUCCESS(JustReturn(std::move(val)));
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus StatusFuncUseStatus(ge::graphStatus val) {
  GE_ASSERT_SUCCESS(val);
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus StatusFuncUseBool(bool val) {
  GE_ASSERT_TRUE(val);
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus StatusFuncUsePointer(void *val) {
  GE_ASSERT_NOTNULL(val);
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus StatusFuncUseUniquePtr(const std::unique_ptr<uint8_t[]> &val) {
  GE_ASSERT_NOTNULL(val);
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus StatusFuncUseSharedPtr(const std::shared_ptr<uint8_t[]> &val) {
  GE_ASSERT_NOTNULL(val);
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus StatusFuncUseEOK(int val) {
  GE_ASSERT_EOK(val);
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus StatusFuncUseRt(int32_t val) {
  GE_ASSERT_RT_OK(val);
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus StatusFuncUseHyperStatus(gert::HyperStatus val) {
  GE_ASSERT_HYPER_SUCCESS(val);
  return ge::GRAPH_SUCCESS;
}

bool BoolFuncUseStatus(ge::graphStatus val) {
  GE_ASSERT_SUCCESS(val);
  return true;
}
bool BoolFuncUseBool(bool val) {
  GE_ASSERT_TRUE(val);
  return true;
}
bool BoolFuncUsePointer(void *val) {
  GE_ASSERT_NOTNULL(val);
  return true;
}
bool BoolFuncUseUniquePtr(const std::unique_ptr<uint8_t[]> &val) {
  GE_ASSERT_NOTNULL(val);
  return true;
}
bool BoolFuncUseSharedPtr(const std::shared_ptr<uint8_t[]> &val) {
  GE_ASSERT_NOTNULL(val);
  return true;
}
bool BoolFuncUseEOK(int val) {
  GE_ASSERT_EOK(val);
  return true;
}
bool BoolFuncUseRt(int32_t val) {
  GE_ASSERT_RT_OK(val);
  return true;
}
bool BoolFuncUseHyperStatus(gert::HyperStatus val) {
  GE_ASSERT_HYPER_SUCCESS(val);
  return true;
}

int64_t g_a = 0xff;
void *PointerFuncUseStatus(ge::graphStatus val) {
  GE_ASSERT_SUCCESS(val);
  return (void*)&g_a;
}
void *PointerFuncUseBool(bool val) {
  GE_ASSERT_TRUE(val);
  return (void*)&g_a;
}
void *PointerFuncUsePointer(void *val) {
  GE_ASSERT_NOTNULL(val);
  return (void*)&g_a;
}
void *PointerFuncUseUniquePtr(const std::unique_ptr<uint8_t[]> &val) {
  GE_ASSERT_NOTNULL(val);
  return (void*)&g_a;
}
void *PointerFuncUseSharedPtr(const std::shared_ptr<uint8_t[]> &val) {
  GE_ASSERT_NOTNULL(val);
  return (void*)&g_a;
}
void *PointerFuncUseEOK(int val) {
  GE_ASSERT_EOK(val);
  return (void*)&g_a;
}
void *PointerFuncUseRt(int32_t val) {
  GE_ASSERT_RT_OK(val);
  return (void*)&g_a;
}
void *PointerFuncUseHyperStatus(gert::HyperStatus val) {
  GE_ASSERT_HYPER_SUCCESS(val);
  return (void*)&g_a;
}
}  // namespace
class CheckerUT : public testing::Test {};
TEST_F(CheckerUT, ReturnStatusOk) {
  ASSERT_NE(StatusFuncUseStatus(ge::FAILED), ge::GRAPH_SUCCESS);
  ASSERT_NE(StatusFuncUseStatus(ge::GRAPH_FAILED), ge::GRAPH_SUCCESS);
  ASSERT_EQ(StatusFuncUseStatus(ge::GRAPH_SUCCESS), ge::GRAPH_SUCCESS);

  ASSERT_NE(StatusFuncUseBool(false), ge::GRAPH_SUCCESS);
  ASSERT_EQ(StatusFuncUseBool(true), ge::GRAPH_SUCCESS);

  int64_t a;
  ASSERT_NE(StatusFuncUsePointer(nullptr), ge::GRAPH_SUCCESS);
  ASSERT_EQ(StatusFuncUsePointer(&a), ge::GRAPH_SUCCESS);

  std::unique_ptr<uint8_t[]> b = std::unique_ptr<uint8_t[]>(new uint8_t[100]);
  ASSERT_NE(StatusFuncUseUniquePtr(nullptr), ge::GRAPH_SUCCESS);
  ASSERT_EQ(StatusFuncUseUniquePtr(b), ge::GRAPH_SUCCESS);

  auto c = std::shared_ptr<uint8_t[]>(new uint8_t[100]);
  ASSERT_NE(StatusFuncUseSharedPtr(nullptr), ge::GRAPH_SUCCESS);
  ASSERT_EQ(StatusFuncUseSharedPtr(c), ge::GRAPH_SUCCESS);

  ASSERT_NE(StatusFuncUseEOK(EINVAL), ge::GRAPH_SUCCESS);
  ASSERT_EQ(StatusFuncUseEOK(EOK), ge::GRAPH_SUCCESS);

  ASSERT_NE(StatusFuncUseHyperStatus(gert::HyperStatus::ErrorStatus("hello")), ge::GRAPH_SUCCESS);
  ASSERT_EQ(StatusFuncUseHyperStatus(gert::HyperStatus::Success()), ge::GRAPH_SUCCESS);

  ASSERT_NE(StatusFuncUseRt(RT_EXCEPTION_DEV_RUNNING_DOWN), ge::GRAPH_SUCCESS);
  ASSERT_EQ(StatusFuncUseRt(RT_ERROR_NONE), ge::GRAPH_SUCCESS);
}

TEST_F(CheckerUT, ReturnBoolOk) {
  ASSERT_NE(BoolFuncUseStatus(ge::FAILED), true);
  ASSERT_NE(BoolFuncUseStatus(ge::GRAPH_FAILED), true);
  ASSERT_EQ(BoolFuncUseStatus(ge::GRAPH_SUCCESS), true);

  ASSERT_NE(BoolFuncUseBool(false), true);
  ASSERT_EQ(BoolFuncUseBool(true), true);

  int64_t a;
  ASSERT_NE(BoolFuncUsePointer(nullptr), true);
  ASSERT_EQ(BoolFuncUsePointer(&a), true);

  std::unique_ptr<uint8_t[]> b = std::unique_ptr<uint8_t[]>(new uint8_t[100]);
  ASSERT_NE(BoolFuncUseUniquePtr(nullptr), true);
  ASSERT_EQ(BoolFuncUseUniquePtr(b), true);

  auto c = std::shared_ptr<uint8_t[]>(new uint8_t[100]);
  ASSERT_NE(BoolFuncUseSharedPtr(nullptr), true);
  ASSERT_EQ(BoolFuncUseSharedPtr(c), true);

  ASSERT_NE(BoolFuncUseEOK(EINVAL), true);
  ASSERT_EQ(BoolFuncUseEOK(EOK), true);

  ASSERT_NE(BoolFuncUseHyperStatus(gert::HyperStatus::ErrorStatus("hello")), true);
  ASSERT_EQ(BoolFuncUseHyperStatus(gert::HyperStatus::Success()), true);

  ASSERT_NE(BoolFuncUseRt(RT_EXCEPTION_DEV_RUNNING_DOWN), true);
  ASSERT_EQ(BoolFuncUseRt(RT_ERROR_NONE), true);
}

TEST_F(CheckerUT, ReturnPointerOk) {
  ASSERT_EQ(PointerFuncUseStatus(ge::FAILED), nullptr);
  ASSERT_EQ(PointerFuncUseStatus(ge::GRAPH_FAILED), nullptr);
  ASSERT_NE(PointerFuncUseStatus(ge::GRAPH_SUCCESS), nullptr);

  ASSERT_EQ(PointerFuncUseBool(false), nullptr);
  ASSERT_NE(PointerFuncUseBool(true), nullptr);

  int64_t a;
  ASSERT_EQ(PointerFuncUsePointer(nullptr), nullptr);
  ASSERT_NE(PointerFuncUsePointer(&a), nullptr);

  std::unique_ptr<uint8_t[]> b = std::unique_ptr<uint8_t[]>(new uint8_t[100]);
  ASSERT_EQ(PointerFuncUseUniquePtr(nullptr), nullptr);
  ASSERT_NE(PointerFuncUseUniquePtr(b), nullptr);

  auto c = std::shared_ptr<uint8_t[]>(new uint8_t[100]);
  ASSERT_EQ(PointerFuncUseSharedPtr(nullptr), nullptr);
  ASSERT_NE(PointerFuncUseSharedPtr(c), nullptr);

  ASSERT_EQ(PointerFuncUseEOK(EINVAL), nullptr);
  ASSERT_NE(PointerFuncUseEOK(EOK), nullptr);

  ASSERT_EQ(PointerFuncUseHyperStatus(gert::HyperStatus::ErrorStatus("hello")), nullptr);
  ASSERT_NE(PointerFuncUseHyperStatus(gert::HyperStatus::Success()), nullptr);

  ASSERT_EQ(PointerFuncUseRt(RT_EXCEPTION_DEV_RUNNING_DOWN), nullptr);
  ASSERT_NE(PointerFuncUseRt(RT_ERROR_NONE), nullptr);
}

TEST_F(CheckerUT, ReturnInFunc) {
  ASSERT_NE(StatusFuncUseStatusFunc(ge::FAILED), ge::GRAPH_SUCCESS);
  ASSERT_NE(StatusFuncUseStatusFunc(ge::GRAPH_FAILED), ge::GRAPH_SUCCESS);
  ASSERT_EQ(StatusFuncUseStatusFunc(ge::GRAPH_SUCCESS), ge::GRAPH_SUCCESS);

  ASSERT_NE(StatusFuncUseBoolFunc(false), ge::GRAPH_SUCCESS);
  ASSERT_EQ(StatusFuncUseBoolFunc(true), ge::GRAPH_SUCCESS);

  int64_t a;
  ASSERT_NE(StatusFuncUsePointerFunc(nullptr), ge::GRAPH_SUCCESS);
  ASSERT_EQ(StatusFuncUsePointerFunc(&a), ge::GRAPH_SUCCESS);

  std::unique_ptr<uint8_t[]> b = std::unique_ptr<uint8_t[]>(new uint8_t[100]);
  ASSERT_NE(StatusFuncUseUniquePtrFunc(nullptr), ge::GRAPH_SUCCESS);
  ASSERT_EQ(StatusFuncUseUniquePtrFunc(std::move(b)), ge::GRAPH_SUCCESS);

  auto c = std::shared_ptr<uint8_t[]>(new uint8_t[100]);
  ASSERT_NE(StatusFuncUseSharedPtrFunc(nullptr), ge::GRAPH_SUCCESS);
  ASSERT_EQ(StatusFuncUseSharedPtrFunc(c), ge::GRAPH_SUCCESS);

  ASSERT_NE(StatusFuncUseEOKFunc(EINVAL), ge::GRAPH_SUCCESS);
  ASSERT_EQ(StatusFuncUseEOKFunc(EOK), ge::GRAPH_SUCCESS);

  ASSERT_NE(StatusFuncUseHyperStatusFunc(gert::HyperStatus::ErrorStatus("hello")), ge::GRAPH_SUCCESS);
  ASSERT_EQ(StatusFuncUseHyperStatusFunc(gert::HyperStatus::Success()), ge::GRAPH_SUCCESS);

  ASSERT_NE(StatusFuncUseRtFunc(RT_EXCEPTION_DEV_RUNNING_DOWN), ge::GRAPH_SUCCESS);
  ASSERT_EQ(StatusFuncUseRtFunc(RT_ERROR_NONE), ge::GRAPH_SUCCESS);
}

TEST_F(CheckerUT, MicroTest) { // Keep this the last case!!!
  std::string error_msg;
#ifdef GELOGE
#undef GELOGE
#endif
#define GELOGE(v, ...) error_msg = std::string(CreateErrorMsg(__VA_ARGS__).data())
  [&error_msg]() { GE_ASSERT(false); }();
  EXPECT_EQ(error_msg, "Assert false failed");
  [&error_msg]() { GE_ASSERT(false, "Something error"); }();
  EXPECT_EQ(error_msg, "Something error");
  [&error_msg]() { GE_ASSERT(false, "%s error", "Many things"); }();
  EXPECT_EQ(error_msg, "Many things error");

  [&error_msg]() { GE_ASSERT_NOTNULL(nullptr); }();
  EXPECT_EQ(error_msg, "Assert ((nullptr) != nullptr) failed");
  [&error_msg]() { GE_ASSERT_NOTNULL(nullptr, "%s error", "Nullptr"); }();
  EXPECT_EQ(error_msg, "Nullptr error");

  [&error_msg]()->bool { GE_ASSERT_EQ(0, 1); return true;}();
  EXPECT_EQ(error_msg, "Assert (0 == 1) failed, expect 1 actual 0");
#undef GELOGE
}
