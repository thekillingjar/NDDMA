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
#include "ge_executor.h"
#include "types.h"
#include "profiling.h"
#include "toolchain/prof_api.h"
#include "dbg_main.h"
#include "ge/ge_error_codes.h"
#include "gmock_stub/ascend_rt_stub.h"
#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <securec.h>
#include <unistd.h>
using namespace testing;
using ::testing::_;
using ::testing::Eq;
using ::testing::Invoke;
using ::testing::NotNull;
using ::testing::Return;

class UtestGEProfilingTest : public testing::Test {
 protected:
  void SetUp() {
    ON_CALL(MmpaStubMock::GetInstance(), mmMalloc(_)).WillByDefault(Invoke(mmMalloc_Normal_Invoke));
    ON_CALL(MmpaStubMock::GetInstance(), mmTellFile(_)).WillByDefault(Invoke(mmTellFile_Normal_Invoke));
    ON_CALL(MmpaStubMock::GetInstance(), mmReadFile(_,_,_,_)).WillByDefault(Invoke(mmReadFile_Normal_Invoke));
    ON_CALL(RtStubMock::GetInstance(), rtDumpInit()).WillByDefault(Return(0));
    ON_CALL(RtStubMock::GetInstance(), access(_,_)).WillByDefault(Return(0));
    ON_CALL(RtStubMock::GetInstance(), rtMalloc(_,_,_,_)).WillByDefault(Invoke(rtMalloc_Normal_Invoke));
    ON_CALL(RtStubMock::GetInstance(), rtMemcpy(_,_,_,_,_)).WillByDefault(Invoke(rtMemcpy_Normal_Invoke));
  }
  void TearDown() {
    Mock::VerifyAndClearExpectations(&MmpaStubMock::GetInstance());
    Mock::VerifyAndClearExpectations(&RtStubMock::GetInstance());
  }
};

uint32_t msprof_init(uint32_t dataType, void *data, uint32_t dataLen) {
  (void)dataType;
  (void)data;
  (void)dataLen;
  return 0;
}
uint32_t msprof_init_unnormal(uint32_t dataType, void *data, uint32_t dataLen) {
  (void)dataType;
  (void)data;
  (void)dataLen;
  return 1;
}
typedef int32_t (*ProfCommandHandle)(uint32_t type, void *data, uint32_t len);
uint32_t msprof_register_callback(uint32_t moduleId, ProfCommandHandle handle) {
  (void)moduleId;
  (void)handle;
  SetMsprofEnable(true);
  return 0;
}
uint32_t msprof_register_callback_unnormal(uint32_t moduleId, ProfCommandHandle handle) {
  (void)moduleId;
  (void)handle;
  SetMsprofEnable(false);
  return 1;
}
uint64_t msprof_gethashid(const char *hashInfo, size_t length) {
  (void)hashInfo;
  (void)length;
  return 0;
}
uint64_t msprof_gettime() {
  return 0;
}
uint32_t msprof_reportdata(uint32_t agingFlag, const void *data, uint32_t length) {
  (void)agingFlag;
  (void)data;
  (void)length;
  return 0;
}
uint32_t msprof_reportdata_unnormal(uint32_t agingFlag, const void *data, uint32_t length) {
  (void)agingFlag;
  (void)data;
  (void)length;
  return 1;
}
int32_t msprof_deinit() {
  return 0;
}
int32_t msprof_deinit_unnormal() {
  return 1;
}
int32_t msprof_notify(uint32_t chipId, uint32_t deviceId, bool isOpen) {
  (void)chipId;
  (void)deviceId;
  (void)isOpen;
  return 0;
}
int32_t msprof_notify_unnormal(uint32_t chipId, uint32_t deviceId, bool isOpen) {
  (void)chipId;
  (void)deviceId;
  (void)isOpen;
  return 1;
}

TEST_F(UtestGEProfilingTest, profiling_so_callback) {
  MsprofCommandHandle data;
  (void)memset_s(&data, sizeof(MsprofCommandHandle), 0, sizeof(MsprofCommandHandle));
  MsprofCtrlHandleFunc(0, &data, sizeof(MsprofCommandHandle));

  int32_t ret = MsprofCtrlHandleFunc(1, &data, sizeof(MsprofCommandHandle) - 1);
  EXPECT_NE(ret, SUCCESS);

  ret = MsprofCtrlHandleFunc(1, NULL, sizeof(MsprofCommandHandle));
  EXPECT_NE(ret, SUCCESS);

  data.profSwitch = MSPROF_TASK_TIME_L0;
  MsprofCtrlHandleFunc(1, &data, sizeof(MsprofCommandHandle));
  bool flag = DbgGetprofEnable();
  ASSERT_EQ(flag, true);

  data.profSwitch = 0x00000000ULL;
  MsprofCtrlHandleFunc(1, &data, sizeof(MsprofCommandHandle));
  flag = DbgGetprofEnable();
  ASSERT_EQ(flag, false);
}

TEST_F(UtestGEProfilingTest, profiling_so_unnormal) {
  EXPECT_CALL(RtStubMock::GetInstance(), dlopen(_, _)).Times(1).WillOnce(Return(nullptr));
  EXPECT_CALL(RtStubMock::GetInstance(), dlerror()).Times(1).WillOnce(Return(nullptr));
  const char *filePath = "../tests/test_c/ut/testcase/executor/data/acl_profiling.json";
  Status ret = DbgInit(filePath);
  ASSERT_NE(ret, SUCCESS);

  bool flag = DbgGetprofEnable();
  ASSERT_EQ(flag, false);

  uint32_t modelId = 0;
  char om[] = "concat_nano";
  ret = DbgLoadModelPostProcess(modelId, om, NULL, NULL);
  ASSERT_EQ(ret, SUCCESS);

  ret = DbgDeInit();
  ASSERT_EQ(ret, SUCCESS);

  ret = DbgNotifySetDevice(0, 0);
  ASSERT_EQ(ret, SUCCESS);

  ret = DbgDeInit();
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestGEProfilingTest, profiling_so_test_with_null_initFunc) {
  char path[4096];
  ssize_t len = readlink("/proc/self/exe", path, sizeof(path) - 1);
  printf("len = %ld, binary path:%s\n", len, path);
  int testNum = 1;
  void *ptr = &testNum;
  EXPECT_CALL(RtStubMock::GetInstance(), dlopen(_, _)).Times(1).WillOnce(Return(ptr));
  EXPECT_CALL(RtStubMock::GetInstance(), dlsym(_, _))
      .Times(1)
      .WillOnce(Return(nullptr));
  EXPECT_CALL(RtStubMock::GetInstance(), dlclose(_)).Times(1).WillOnce(Return(0));

  const char *filePath = "../tests/test_c/ut/testcase/executor/data/acl_profiling.json";
  Status ret = DbgInit(filePath);
  ASSERT_NE(ret, SUCCESS);

  ret = DbgDeInit();
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestGEProfilingTest, profiling_so_test_with_null_callbackFunc) {
  int testNum = 1;
  void *ptr = &testNum;
  EXPECT_CALL(RtStubMock::GetInstance(), dlopen(_, _)).Times(1).WillOnce(Return(ptr));
  EXPECT_CALL(RtStubMock::GetInstance(), dlsym(_, _))
      .Times(2)
      .WillOnce(Return((void *)msprof_init))
      .WillOnce(Return(nullptr));
  EXPECT_CALL(RtStubMock::GetInstance(), dlclose(_)).Times(1).WillOnce(Return(0));

  const char *filePath = "../tests/test_c/ut/testcase/executor/data/acl_profiling.json";
  Status ret = DbgInit(filePath);
  ASSERT_NE(ret, SUCCESS);

  ret = DbgDeInit();
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestGEProfilingTest, profiling_so_test_with_null_hashFunc) {
  int testNum = 1;
  void *ptr = &testNum;
  EXPECT_CALL(RtStubMock::GetInstance(), dlopen(_, _)).Times(1).WillOnce(Return(ptr));
  EXPECT_CALL(RtStubMock::GetInstance(), dlsym(_, _))
      .Times(3)
      .WillOnce(Return((void *)msprof_init))
      .WillOnce(Return((void *)msprof_register_callback))
      .WillOnce(Return(nullptr));
  EXPECT_CALL(RtStubMock::GetInstance(), dlclose(_)).Times(1).WillOnce(Return(0));

  const char *filePath = "../tests/test_c/ut/testcase/executor/data/acl_profiling.json";
  Status ret = DbgInit(filePath);
  ASSERT_NE(ret, SUCCESS);

  ret = DbgDeInit();
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestGEProfilingTest, profiling_so_test_with_null_timeFunc) {
  int testNum = 1;
  void *ptr = &testNum;
  EXPECT_CALL(RtStubMock::GetInstance(), dlopen(_, _)).Times(1).WillOnce(Return(ptr));
  EXPECT_CALL(RtStubMock::GetInstance(), dlsym(_, _))
      .Times(4)
      .WillOnce(Return((void *)msprof_init))
      .WillOnce(Return((void *)msprof_register_callback))
      .WillOnce(Return((void *)msprof_gethashid))
      .WillOnce(Return(nullptr));
  EXPECT_CALL(RtStubMock::GetInstance(), dlclose(_)).Times(1).WillOnce(Return(0));

  const char *filePath = "../tests/test_c/ut/testcase/executor/data/acl_profiling.json";
  Status ret = DbgInit(filePath);
  ASSERT_NE(ret, SUCCESS);

  ret = DbgDeInit();
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestGEProfilingTest, profiling_so_test_with_null_reportFunc) {
  int testNum = 1;
  void *ptr = &testNum;
  EXPECT_CALL(RtStubMock::GetInstance(), dlopen(_, _)).Times(1).WillOnce(Return(ptr));
  EXPECT_CALL(RtStubMock::GetInstance(), dlsym(_, _))
      .Times(5)
      .WillOnce(Return((void *)msprof_init))
      .WillOnce(Return((void *)msprof_register_callback))
      .WillOnce(Return((void *)msprof_gethashid))
      .WillOnce(Return((void *)msprof_gettime))
      .WillOnce(Return(nullptr));
  EXPECT_CALL(RtStubMock::GetInstance(), dlclose(_)).Times(1).WillOnce(Return(0));

  const char *filePath = "../tests/test_c/ut/testcase/executor/data/acl_profiling.json";
  Status ret = DbgInit(filePath);
  ASSERT_NE(ret, SUCCESS);

  ret = DbgDeInit();
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestGEProfilingTest, profiling_so_test_with_null_finalizeFunc) {
  int testNum = 1;
  void *ptr = &testNum;
  EXPECT_CALL(RtStubMock::GetInstance(), dlopen(_, _)).Times(1).WillOnce(Return(ptr));
  EXPECT_CALL(RtStubMock::GetInstance(), dlsym(_, _))
      .Times(6)
      .WillOnce(Return((void *)msprof_init))
      .WillOnce(Return((void *)msprof_register_callback))
      .WillOnce(Return((void *)msprof_gethashid))
      .WillOnce(Return((void *)msprof_gettime))
      .WillOnce(Return((void *)msprof_reportdata))
      .WillOnce(Return(nullptr));
  EXPECT_CALL(RtStubMock::GetInstance(), dlclose(_)).Times(1).WillOnce(Return(0));

  const char *filePath = "../tests/test_c/ut/testcase/executor/data/acl_profiling.json";
  Status ret = DbgInit(filePath);
  ASSERT_NE(ret, SUCCESS);

  ret = DbgDeInit();
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestGEProfilingTest, profiling_so_test_with_null_deviceFunc) {
  int testNum = 1;
  void *ptr = &testNum;
  EXPECT_CALL(RtStubMock::GetInstance(), dlopen(_, _)).Times(1).WillOnce(Return(ptr));
  EXPECT_CALL(RtStubMock::GetInstance(), dlsym(_, _))
      .Times(7)
      .WillOnce(Return((void *)msprof_init))
      .WillOnce(Return((void *)msprof_register_callback))
      .WillOnce(Return((void *)msprof_gethashid))
      .WillOnce(Return((void *)msprof_gettime))
      .WillOnce(Return((void *)msprof_reportdata))
      .WillOnce(Return((void *)msprof_deinit))
      .WillOnce(Return(nullptr));
  EXPECT_CALL(RtStubMock::GetInstance(), dlclose(_)).Times(1).WillOnce(Return(0));

  const char *filePath = "../tests/test_c/ut/testcase/executor/data/acl_profiling.json";
  Status ret = DbgInit(filePath);
  ASSERT_NE(ret, SUCCESS);

  ret = DbgDeInit();
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestGEProfilingTest, profiling_so_test_with_unnormal_callbackFunc) {
  int testNum = 1;
  void *ptr = &testNum;
  EXPECT_CALL(RtStubMock::GetInstance(), dlopen(_, _)).Times(1).WillOnce(Return(ptr));
  EXPECT_CALL(RtStubMock::GetInstance(), dlsym(_, _))
      .Times(7)
      .WillOnce(Return((void *)msprof_init))
      .WillOnce(Return((void *)msprof_register_callback_unnormal))
      .WillOnce(Return((void *)msprof_gethashid))
      .WillOnce(Return((void *)msprof_gettime))
      .WillOnce(Return((void *)msprof_reportdata))
      .WillOnce(Return((void *)msprof_deinit))
      .WillOnce(Return((void *)msprof_notify));

  const char *filePath = "../tests/test_c/ut/testcase/executor/data/acl_profiling.json";
  Status ret = DbgInit(filePath);
  ASSERT_NE(ret, SUCCESS);

  EXPECT_CALL(RtStubMock::GetInstance(), dlclose(_)).Times(1).WillOnce(Return(0));
  ret = DbgDeInit();
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestGEProfilingTest, profiling_so_test_with_unnormal_initFunc) {
  int testNum = 1;
  void *ptr = &testNum;
  EXPECT_CALL(RtStubMock::GetInstance(), dlopen(_, _)).Times(1).WillOnce(Return(ptr));
  EXPECT_CALL(RtStubMock::GetInstance(), dlsym(_, _))
      .Times(7)
      .WillOnce(Return((void *)msprof_init_unnormal))
      .WillOnce(Return((void *)msprof_register_callback))
      .WillOnce(Return((void *)msprof_gethashid))
      .WillOnce(Return((void *)msprof_gettime))
      .WillOnce(Return((void *)msprof_reportdata))
      .WillOnce(Return((void *)msprof_deinit))
      .WillOnce(Return((void *)msprof_notify));

  const char *filePath = "../tests/test_c/ut/testcase/executor/data/acl_profiling.json";
  Status ret = DbgInit(filePath);
  ASSERT_NE(ret, SUCCESS);

  EXPECT_CALL(RtStubMock::GetInstance(), dlclose(_)).Times(1).WillOnce(Return(0));
  ret = DbgDeInit();
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestGEProfilingTest, profiling_so_test_with_invalid_reportFunc) {
  int testNum = 1;
  void *ptr = &testNum;
  EXPECT_CALL(RtStubMock::GetInstance(), dlopen(_, _)).Times(1).WillOnce(Return(ptr));
  EXPECT_CALL(RtStubMock::GetInstance(), dlsym(_, _))
      .Times(5)
      .WillOnce(Return((void *)msprof_init))
      .WillOnce(Return((void *)msprof_register_callback))
      .WillOnce(Return((void *)msprof_gethashid))
      .WillOnce(Return((void *)msprof_gettime))
      .WillOnce(Return(nullptr)); // test reportFunc is null
  EXPECT_CALL(RtStubMock::GetInstance(), dlclose(_)).Times(1).WillOnce(Return(0));

  const char *filePath = "../tests/test_c/ut/testcase/executor/data/acl_profiling.json";
  Status ret = DbgInit(filePath);
  uint32_t modelId = 0;
  char om[] = "concat_nano";
  ret = DbgLoadModelPostProcess(modelId, om, NULL, NULL);
  ASSERT_EQ(ret, SUCCESS);

  ret = DbgDeInit();
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestGEProfilingTest, profiling_so_test_with_null_handle) {
  int testNum = 1;
  void *ptr = &testNum;
  EXPECT_CALL(RtStubMock::GetInstance(), dlopen(_, _)).Times(1).WillOnce(Return(ptr));
  EXPECT_CALL(RtStubMock::GetInstance(), dlsym(_, _))
      .Times(7)
      .WillOnce(Return((void *)msprof_init))
      .WillOnce(Return((void *)msprof_register_callback))
      .WillOnce(Return((void *)msprof_gethashid))
      .WillOnce(Return((void *)msprof_gettime))
      .WillOnce(Return((void *)msprof_reportdata_unnormal))
      .WillOnce(Return((void *)msprof_deinit))
      .WillOnce(Return((void *)msprof_notify));

  const char *filePath = "../tests/test_c/ut/testcase/executor/data/acl_profiling.json";
  Status ret = DbgInit(filePath);

  uint32_t modelId = 0;
  char om[] = "concat_nano";
  ret = DbgLoadModelPostProcess(modelId, om, NULL, NULL);
  ASSERT_NE(ret, SUCCESS);

  EXPECT_CALL(RtStubMock::GetInstance(), dlclose(_)).Times(1).WillOnce(Return(0));
  ret = DbgDeInit();
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestGEProfilingTest, profiling_so_test_with_unnormal_deinitFunc) {
  int testNum = 1;
  void *ptr = &testNum;
  EXPECT_CALL(RtStubMock::GetInstance(), dlopen(_, _)).Times(1).WillOnce(Return(ptr));
  EXPECT_CALL(RtStubMock::GetInstance(), dlsym(_, _))
      .Times(7)
      .WillOnce(Return((void *)msprof_init))
      .WillOnce(Return((void *)msprof_register_callback))
      .WillOnce(Return((void *)msprof_gethashid))
      .WillOnce(Return((void *)msprof_gettime))
      .WillOnce(Return((void *)msprof_reportdata))
      .WillOnce(Return((void *)msprof_deinit_unnormal))
      .WillOnce(Return((void *)msprof_notify));

  const char *filePath = "../tests/test_c/ut/testcase/executor/data/acl_profiling.json";
  Status ret = DbgInit(filePath);

  EXPECT_CALL(RtStubMock::GetInstance(), dlclose(_)).Times(1).WillOnce(Return(0));
  ret = DbgDeInit();
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestGEProfilingTest, profiling_so_test_with_unnormal_notifyFunc) {
  int testNum = 1;
  void *ptr = &testNum;
  EXPECT_CALL(RtStubMock::GetInstance(), dlopen(_, _)).Times(1).WillOnce(Return(ptr));
  EXPECT_CALL(RtStubMock::GetInstance(), dlsym(_, _))
      .Times(7)
      .WillOnce(Return((void *)msprof_init))
      .WillOnce(Return((void *)msprof_register_callback))
      .WillOnce(Return((void *)msprof_gethashid))
      .WillOnce(Return((void *)msprof_gettime))
      .WillOnce(Return((void *)msprof_reportdata))
      .WillOnce(Return((void *)msprof_deinit))
      .WillOnce(Return((void *)msprof_notify_unnormal));

  const char *filePath = "../tests/test_c/ut/testcase/executor/data/acl_profiling.json";
  Status ret = DbgInit(filePath);

  ret = DbgNotifySetDevice(0, 0);
  ASSERT_NE(ret, SUCCESS);

  EXPECT_CALL(RtStubMock::GetInstance(), dlclose(_)).Times(1).WillOnce(Return(0));
  ret = DbgDeInit();
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestGEProfilingTest, profiling_so_with_unnormal_jsonConfig) {
  const char *filePath = "../tests/test_c/ut/testcase/executor/data/acl_profiling_unnormal.json";
  Status ret = DbgInit(filePath);
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestGEProfilingTest, profiling_so_normal) {
  int testNum = 1;
  void *ptr = &testNum;
  EXPECT_CALL(RtStubMock::GetInstance(), dlopen(_, _)).Times(1).WillOnce(Return(ptr));
  EXPECT_CALL(RtStubMock::GetInstance(), dlsym(_, _))
      .Times(7)
      .WillOnce(Return((void *)msprof_init))
      .WillOnce(Return((void *)msprof_register_callback))
      .WillOnce(Return((void *)msprof_gethashid))
      .WillOnce(Return((void *)msprof_gettime))
      .WillOnce(Return((void *)msprof_reportdata))
      .WillOnce(Return((void *)msprof_deinit))
      .WillOnce(Return((void *)msprof_notify));

  const char *filePath = "../tests/test_c/ut/testcase/executor/data/acl_profiling.json";
  Status ret = DbgInit(filePath);
  ASSERT_EQ(ret, SUCCESS);

  bool profEnable = DbgGetprofEnable();
  ASSERT_EQ(profEnable, true);

  uint32_t modelId = 0;
  char om[] = "concat_nano";
  ret = DbgLoadModelPostProcess(modelId, om, NULL, NULL);
  ASSERT_EQ(ret, SUCCESS);

  ret = DbgNotifySetDevice(0, 0);
  ASSERT_EQ(ret, SUCCESS);

  EXPECT_CALL(RtStubMock::GetInstance(), dlclose(_)).Times(1).WillOnce(Return(0));
  ret = DbgDeInit();
  ASSERT_EQ(ret, SUCCESS);
}
