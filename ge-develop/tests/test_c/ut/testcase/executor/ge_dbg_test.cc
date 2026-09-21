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
#include <sstream>
#include <algorithm>
#include <memory>
#include <fstream>
#include <string>
#include <cstdio>
#include "json_parser.h"
#include "ge_executor.h"
#include "types.h"
#include "dump.h"
#include "dump_thread_manager.h"
#include "parse_dbg_file.h"
#include "parse_json_file.h"
#include "tlv_parse.h"
#include "dump_config.h"
#include "ge/ge_error_codes.h"
#include "gmock_stub/ascend_rt_stub.h"
#include "vector.h"
#include <securec.h>
using namespace testing;
using ::testing::Return;
using ::testing::Eq;
using ::testing::NotNull;
using ::testing::Invoke;
using ::testing::_;

class UtestGEDBGTest : public testing::Test {
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

Status InitDataDumpMock(void) {
  return SUCCESS;
}

Status StopDataDumpMock(void) {
  return SUCCESS;
}

TEST_F(UtestGEDBGTest, GeDbgDumpInit) {
  int testNum = 1;
  void *ptr = &testNum;
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmTellFile(_)).WillRepeatedly(Invoke(mmTellFile_Normal_Invoke));
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmMalloc(_)).WillRepeatedly(Invoke(mmMalloc_Normal_Invoke));
  EXPECT_CALL(RtStubMock::GetInstance(), dlopen(_,_)).Times(1).WillOnce(Return(ptr));
  EXPECT_CALL(RtStubMock::GetInstance(), dlclose(_)).Times(1).WillOnce(Return(0));
  EXPECT_CALL(RtStubMock::GetInstance(), dlsym(_,_)).Times(2).WillOnce(Return((void *)StopDataDumpMock))
    .WillOnce(Return((void *)StopDataDumpMock));
  const char *filePath = "../tests/test_c/ut/testcase/data/acl.json";
  EXPECT_CALL(RtStubMock::GetInstance(), rtDumpInit()).WillRepeatedly(Return(0));
  Status ret = DbgDumpInit(filePath);
  ASSERT_EQ(ret, SUCCESS);
  ret = DbgDumpDeInit();
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestGEDBGTest, GeDbgDumpInitWithFailAccess) {
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmMalloc(_)).WillRepeatedly(Invoke(mmMalloc_Normal_Invoke));
  const char *filePath = "../tests/test_c/ut/testcase/data/acl.json";
  EXPECT_CALL(RtStubMock::GetInstance(), access(_,_)).Times(1).WillOnce(Return(1));
  Status ret = DbgDumpInit(filePath); // covers access fail in CheckDumpPath
  ASSERT_EQ(ret, SUCCESS);
  EXPECT_CALL(RtStubMock::GetInstance(), access(_,_)).WillRepeatedly(Return(0));
  ret = DbgDumpDeInit();
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestGEDBGTest, GeDbgDumpInitWithGetModelNameFail) {
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmMalloc(_)).WillRepeatedly(Invoke(mmMalloc_Normal_Invoke));
  const char *filePath = "../tests/test_c/ut/testcase/data/acl.json";
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmMalloc(_)).Times(2).WillOnce(mmMalloc_Normal_Invoke).WillOnce(Invoke(mmMalloc_Abnormal_Invoke));
  Status ret = DbgDumpInit(filePath); // covers malloc fail in GetModelName
  ASSERT_EQ(ret, SUCCESS);
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmMalloc(_)).WillRepeatedly(Invoke(mmMalloc_Normal_Invoke));
  ret = DbgDumpDeInit();
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestGEDBGTest, GeDbgDumpInitWithGetModelLayerFail) {
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmMalloc(_)).WillRepeatedly(Invoke(mmMalloc_Normal_Invoke));
  const char *filePath = "../tests/test_c/ut/testcase/data/acl.json";
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmMalloc(_)).Times(3).WillOnce(mmMalloc_Normal_Invoke).WillOnce(mmMalloc_Normal_Invoke).WillOnce(Invoke(mmMalloc_Abnormal_Invoke));
  Status ret = DbgDumpInit(filePath); // covers malloc fail in GetModelLayer
  ASSERT_EQ(ret, SUCCESS);
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmMalloc(_)).WillRepeatedly(Invoke(mmMalloc_Normal_Invoke));
  ret = DbgDumpDeInit();
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestGEDBGTest, GeDbgDumpInitWithNullHandle) {
  const char *filePath = "";
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmMalloc(_)).WillRepeatedly(Invoke(mmMalloc_Normal_Invoke));
  Status ret = DbgDumpInit(filePath);
  ASSERT_EQ(ret, SUCCESS);

  void *handle = DbgCreateModelHandle();
  ASSERT_EQ(handle, nullptr);

  rtMdlLoad_t mdlLoad;
  mdlLoad.totalTaskNum = 1;
  std::string dbgFilePath = "concat_nano";
  size_t taskDescSize = 1000UL;
  ret = DbgDumpPreProcess(&mdlLoad, taskDescSize, (char *)dbgFilePath.c_str(), handle);
  ASSERT_EQ(ret, SUCCESS);

  DbgFreeLoadDumpInfo(handle);
}

TEST_F(UtestGEDBGTest, GeDbgDumpInitWithNullStopDumpFunc) {
  int testNum = 1;
  void *ptr = &testNum;
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmMalloc(_)).WillRepeatedly(Invoke(mmMalloc_Normal_Invoke));
  EXPECT_CALL(RtStubMock::GetInstance(), dlopen(_,_)).Times(1).WillOnce(Return(ptr));
  EXPECT_CALL(RtStubMock::GetInstance(), dlsym(_,_)).Times(2).WillOnce(Return((void *)InitDataDumpMock))
    .WillOnce(Return(nullptr));
  const char *filePath = "../tests/test_c/ut/testcase/data/acl.json";
  Status ret = DbgDumpInit(filePath);
  ASSERT_NE(ret, SUCCESS);
}

TEST_F(UtestGEDBGTest, GeDbgCreateModelHandle) {
  int testNum = 1;
  void *ptr = &testNum;
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmMalloc(_)).WillRepeatedly(Invoke(mmMalloc_Normal_Invoke));
  EXPECT_CALL(RtStubMock::GetInstance(), dlopen(_,_)).Times(1).WillOnce(Return(ptr));
  EXPECT_CALL(RtStubMock::GetInstance(), dlclose(_)).Times(1).WillOnce(Return(0));
  EXPECT_CALL(RtStubMock::GetInstance(), dlsym(_,_)).Times(2).WillOnce(Return((void *)StopDataDumpMock))
    .WillOnce(Return((void *)StopDataDumpMock));
  const char *filePath = "../tests/test_c/ut/testcase/data/acl.json";
  Status ret = DbgDumpInit(filePath);
  ASSERT_EQ(ret, SUCCESS);
  void *handle = DbgCreateModelHandle();
  ASSERT_NE(handle, nullptr);
  DbgFreeLoadDumpInfo(handle);
  ret = DbgDumpDeInit();
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestGEDBGTest, GeInitDumpThreadWithDumpInitFail) {
  EXPECT_CALL(RtStubMock::GetInstance(), rtDumpInit()).Times(1).WillOnce(Return(-1));
  Status ret = InitDumpThread();
  ASSERT_NE(ret, SUCCESS);
}

TEST_F(UtestGEDBGTest, GeInitDumpThreadWithDlfuncFail) {
  EXPECT_CALL(RtStubMock::GetInstance(), rtDumpInit()).WillRepeatedly(Return(0));
  EXPECT_CALL(RtStubMock::GetInstance(), dlopen(_,_)).Times(1).WillOnce(Return(nullptr));
  Status ret = InitDumpThread();
  ASSERT_NE(ret, SUCCESS);
  DeInitDumpThread();
  int testNum = 1;
  void *ptr = &testNum;
  EXPECT_CALL(RtStubMock::GetInstance(), dlopen(_,_)).Times(1).WillOnce(Return(ptr));
  EXPECT_CALL(RtStubMock::GetInstance(), dlclose(_)).Times(1).WillOnce(Return(0));
  EXPECT_CALL(RtStubMock::GetInstance(), dlsym(_,_)).Times(1).WillOnce(Return(nullptr));
  ret = InitDumpThread();
  ASSERT_NE(ret, SUCCESS);
  DeInitDumpThread();
}

TEST_F(UtestGEDBGTest, GeDumpInitAndDeinitWithNormalMode) {
  const char *filePath1 = "../tests/test_c/ut/testcase/data/aclCase1.json";
  Status ret = DbgDumpInit(filePath1);
  ASSERT_EQ(ret, SUCCESS);
  ret = DbgDumpDeInit();
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestGEDBGTest, HandleDumpConfigWithDebugOff) {
  int testNum = 1;
  void *ptr = &testNum;
  EXPECT_CALL(RtStubMock::GetInstance(), dlopen(_,_)).Times(1).WillOnce(Return(ptr));
  EXPECT_CALL(RtStubMock::GetInstance(), dlclose(_)).Times(1).WillOnce(Return(0));
  EXPECT_CALL(RtStubMock::GetInstance(), dlsym(_,_)).Times(2).WillOnce(Return((void *)StopDataDumpMock))
    .WillOnce(Return((void *)StopDataDumpMock));
  const char *filePath2 = "../tests/test_c/ut/testcase/data/aclCase2.json";
  Status ret = DbgDumpInit(filePath2);
  ASSERT_EQ(ret, SUCCESS);
  ret = DbgDumpDeInit();
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestGEDBGTest, HandleDumpConfigWithStatsData) {
  int testNum = 1;
  void *ptr = &testNum;
  EXPECT_CALL(RtStubMock::GetInstance(), dlopen(_,_)).Times(1).WillOnce(Return(ptr));
  EXPECT_CALL(RtStubMock::GetInstance(), dlclose(_)).Times(1).WillOnce(Return(0));
  EXPECT_CALL(RtStubMock::GetInstance(), dlsym(_,_)).Times(2).WillOnce(Return((void *)StopDataDumpMock))
    .WillOnce(Return((void *)StopDataDumpMock));
  const char *filePath3 = "../tests/test_c/ut/testcase/data/aclCase3.json";
  Status ret = DbgDumpInit(filePath3);
  ASSERT_EQ(ret, SUCCESS);
  ret = DbgDumpDeInit();
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestGEDBGTest, HandleDumpConfigWithInvalidFormat) {
  const char *filePath4 = "../tests/test_c/ut/testcase/data/aclCase4.json";
  Status ret = DbgDumpInit(filePath4);
  ASSERT_EQ(ret, SUCCESS);
  ret = DbgDumpDeInit();
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestGEDBGTest, HandleDumpConfigWithDebugOn) {
  const char *filePath5 = "../tests/test_c/ut/testcase/data/aclCase5.json";
  Status ret = DbgDumpInit(filePath5);
  ASSERT_EQ(ret, SUCCESS);
  ret = DbgDumpDeInit();
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestGEDBGTest, HandleDumpConfigWithUnnormalWithNoList) {
  const char *filePath1 = "../tests/test_c/ut/testcase/data/aclCaseInvalidList1.json";
  Status ret = DbgDumpInit(filePath1);
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestGEDBGTest, HandleDumpConfigUnnormalWithEmptyList) {
  const char *filePath2 = "../tests/test_c/ut/testcase/data/aclCaseInvalidList2.json";
  Status ret = DbgDumpInit(filePath2);
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestGEDBGTest, GeHandleDumpConfigUnnormalWithEmptyModelName) {
  const char *filePath3 = "../tests/test_c/ut/testcase/data/aclCaseInvalidList3.json";
  Status ret = DbgDumpInit(filePath3);
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestGEDBGTest, HandleDumpConfigUnnormalWithEmptyListOnExceptionMode) {
  const char *filePath4 = "../tests/test_c/ut/testcase/data/aclCaseInvalidList4.json";
  Status ret = DbgDumpInit(filePath4);
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestGEDBGTest, HandleDumpConfigUnnormalWithUnlegalList) {
  const char *filePath5 = "../tests/test_c/ut/testcase/data/aclCaseInvalidList5.json";
  Status ret = DbgDumpInit(filePath5);
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestGEDBGTest, HandleDumpConfigUnnormalWithEmptyLayer) {
  const char *filePath6 = "../tests/test_c/ut/testcase/data/aclCaseInvalidList6.json";
  Status ret = DbgDumpInit(filePath6);
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestGEDBGTest, HandleDumpConfigUnnormalWithNoPath) {
  const char *filePath11 = "../tests/test_c/ut/testcase/data/aclCaseInvalidPath.json";
  Status ret = DbgDumpInit(filePath11);
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestGEDBGTest, HandleDumpConfigUnnormalWithEmptyPath) {
  const char *filePath12 = "../tests/test_c/ut/testcase/data/aclCaseInvalidPath2.json";
  Status ret = DbgDumpInit(filePath12);
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestGEDBGTest, HandleDumpConfigUnnormalWithUnlegalPath) {
  const char *filePath13 = "../tests/test_c/ut/testcase/data/aclCaseInvalidPath3.json";
  Status ret = DbgDumpInit(filePath13);
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestGEDBGTest, HandleDumpConfigUnnormalWithInvalidScene) {
  const char *filePath21 = "../tests/test_c/ut/testcase/data/aclCaseInvalidScene.json";
  Status ret = DbgDumpInit(filePath21);
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestGEDBGTest, HandleDumpConfigUnnormalWithInvalidMode) {
  const char *filePath22 = "../tests/test_c/ut/testcase/data/aclCaseInvalidMode.json";
  Status ret = DbgDumpInit(filePath22);
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestGEDBGTest, HandleDumpConfigUnnormalWithInvalidData) {
  const char *filePath23 = "../tests/test_c/ut/testcase/data/aclCaseInvalidData.json";
  Status ret = DbgDumpInit(filePath23);
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestGEDBGTest, HandleDumpConfigUnnormalWithInvalidSwitch) {
  const char *filePath24 = "../tests/test_c/ut/testcase/data/aclCaseInvalidSwitch.json";
  Status ret = DbgDumpInit(filePath24);
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestGEDBGTest, GeDbgDumpUnnormal) {
  SetDumpEnableFlag(true);
  Status ret = DbgDumpPreProcess(NULL, 0, NULL, NULL);
  ASSERT_NE(ret, SUCCESS);

  uint32_t modelId = 1U;
  uint64_t stepId = 1;
  uint64_t *stepIdAddr = &stepId;
  ret = DbgDumpPostProcess(modelId, stepIdAddr, NULL);
  ASSERT_EQ(ret, ACL_ERROR_GE_MEMORY_ALLOCATION);
  SetDumpEnableFlag(false);
}

TEST_F(UtestGEDBGTest, GeParseDbgCaseWithNoDbgFile) {
  int testNum = 1;
  void *ptr = &testNum;
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmMalloc(_)).WillRepeatedly(Invoke(mmMalloc_Normal_Invoke));
  EXPECT_CALL(RtStubMock::GetInstance(), dlopen(_,_)).Times(1).WillOnce(Return(ptr));
  EXPECT_CALL(RtStubMock::GetInstance(), dlclose(_)).Times(1).WillOnce(Return(0));
  EXPECT_CALL(RtStubMock::GetInstance(), dlsym(_,_)).Times(2).WillOnce(Return((void *)StopDataDumpMock))
    .WillOnce(Return((void *)StopDataDumpMock));
  const char *filePath = "../tests/test_c/ut/testcase/data/acl.json";
  Status ret = DbgDumpInit(filePath);
  ASSERT_EQ(ret, SUCCESS);
  void *handle = DbgCreateModelHandle();
  ASSERT_NE(handle, nullptr);

  EXPECT_CALL(MmpaStubMock::GetInstance(), mmMalloc(_)).Times(1).WillOnce(Invoke(mmMalloc_Abnormal_Invoke));
  void *handle2 = DbgCreateModelHandle();
  ASSERT_EQ(handle2, nullptr);
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmMalloc(_)).WillRepeatedly(Invoke(mmMalloc_Normal_Invoke));

  rtMdlLoad_t mdlLoad;
  mdlLoad.totalTaskNum = 1;
  std::string dbgFilePath = "concat_nano";
  size_t taskDescSize = 1000UL;
  ret = DbgDumpPreProcess(&mdlLoad, taskDescSize, (char *)dbgFilePath.c_str(), handle);
  ASSERT_EQ(ret, ACL_ERROR_GE_INTERNAL_ERROR);
  ret = DbgDumpPreProcess(&mdlLoad, taskDescSize, (char *)dbgFilePath.c_str(), NULL);
  ASSERT_EQ(ret, ACL_ERROR_GE_MEMORY_ALLOCATION);
  DbgFreeLoadDumpInfo(handle);
  ret = DbgDumpDeInit();
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestGEDBGTest, GeParseDbgCaseWithDumpProcessMallocFail) {
    uint8_t file[] =
    {0,0,0,0,90,90,90,90,0,0,0,0,11,0,0,0,99,111,110,99,97,116,95,110,97,110,111,1,0,0,0,198,
     1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,169,1,0,0,
     0,0,0,0,8,0,0,0,99,111,110,99,97,116,118,50,1,0,0,0,9,0,0,0,67,111,110,99,97,116,86,50,
     68,2,0,0,0,12,0,0,0,8,0,0,0,99,111,110,99,97,116,118,50,4,0,0,0,180,0,0,0,2,0,0,
     0,1,0,0,0,1,0,0,0,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,120,114,0,
     0,0,0,0,0,48,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,0,0,0,0,60,57,0,0,0,0,0,
     0,1,0,0,0,16,0,0,0,1,0,0,0,0,0,0,0,60,57,0,0,0,0,0,0,1,0,0,0,1,0,0,
     0,3,0,0,0,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,60,0,0,0,0,0,0,0,48,0,0,
     0,0,0,0,0,16,0,0,0,1,0,0,0,0,0,0,0,30,0,0,0,0,0,0,0,1,0,0,0,16,0,0,
     0,1,0,0,0,0,0,0,0,30,0,0,0,0,0,0,0,5,0,0,0,120,0,0,0,1,0,0,0,1,0,0,
     0,1,0,0,0,3,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,16,0,0,0,0,0,0,0,0,0,0,
     0,0,0,0,0,180,114,0,0,0,0,0,0,64,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,0,0,0,
     0,90,57,0,0,0,0,0,0,1,0,0,0,16,0,0,0,1,0,0,0,0,0,0,0,90,57,0,0,0,0,0,
     0,2,0,0,0,8,0,0,0,99,111,110,99,97,116,118,50,8,0,0,0,48,0,0,0,1,0,0,0,0,115,0,
     0,0,0,0,0,224,114,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,224,229,0,0,0,0,0,0,0,0,0,0};
  int len = sizeof(file) / sizeof(uint8_t);
  void *outBuff = nullptr;
  outBuff = malloc(len);
  (void)memcpy_s(outBuff, len, file, len);
  std::string fileName;
  fileName.append("./concat_nano").append(".dbg");
  FILE *f = fopen(fileName.c_str(), "wb+");
  fwrite((uint8_t *)outBuff, len, sizeof(char), f);
  fclose(f);
  free(outBuff);
  int testNum = 1;
  void *ptr = &testNum;
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmMalloc(_)).WillRepeatedly(Invoke(mmMalloc_Normal_Invoke));
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmReadFile(_,_,_,_)).WillRepeatedly(Invoke(mmReadFile_Normal_Invoke));
  EXPECT_CALL(RtStubMock::GetInstance(), dlopen(_,_)).Times(1).WillOnce(Return(ptr));
  EXPECT_CALL(RtStubMock::GetInstance(), dlclose(_)).Times(1).WillOnce(Return(0));
  EXPECT_CALL(RtStubMock::GetInstance(), dlsym(_,_)).Times(2).WillOnce(Return((void *)StopDataDumpMock))
    .WillOnce(Return((void *)StopDataDumpMock));
  const char *filePath = "../tests/test_c/ut/testcase/data/acl.json";
  Status ret = DbgDumpInit(filePath);
  ASSERT_EQ(ret, SUCCESS);
  rtMdlLoad_t mdlLoad;
  mdlLoad.totalTaskNum = 1;
  std::string dbgFilePath = "concat_nano";
  size_t taskDescSize = 1000UL;

  void *handle = DbgCreateModelHandle();
  ASSERT_NE(handle, nullptr);

  EXPECT_CALL(MmpaStubMock::GetInstance(), mmMalloc(_)).Times(1).WillOnce(Invoke(mmMalloc_Abnormal_Invoke));
  ret = DbgDumpPreProcess(&mdlLoad, taskDescSize, (char *)dbgFilePath.c_str(), handle);
  ASSERT_NE(ret, SUCCESS); // cover malloc fail in DbgDumpPreProcess

  EXPECT_CALL(MmpaStubMock::GetInstance(), mmMalloc(_)).Times(3).WillOnce(Invoke(mmMalloc_Normal_Invoke))
    .WillOnce(Invoke(mmMalloc_Normal_Invoke)).WillOnce(Invoke(mmMalloc_Abnormal_Invoke));
  ret = DbgDumpPreProcess(&mdlLoad, taskDescSize, (char *)dbgFilePath.c_str(), handle);
  ASSERT_NE(ret, SUCCESS); // covers malloc fail in ProModelName

  EXPECT_CALL(MmpaStubMock::GetInstance(), mmMalloc(_)).WillRepeatedly(Invoke(mmMalloc_Normal_Invoke));
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmMalloc(_)).Times(1).WillOnce(Invoke(mmMalloc_Abnormal_Invoke));
  ret = SendLoadInfoToAicpu((ModelDbgHandle *)handle); // covers rtMalloc fail in InitLoadDumpInfo
  ASSERT_NE(ret, SUCCESS);

  DbgFreeLoadDumpInfo(handle);
  ret = DbgDumpDeInit();
  ASSERT_EQ(ret, SUCCESS);
  remove(fileName.c_str());
}

TEST_F(UtestGEDBGTest, GeParseDbgCaseWithSetDumpFlagError) {
    uint8_t file[] =
    {0,0,0,0,90,90,90,90,0,0,0,0,11,0,0,0,99,111,110,99,97,116,95,110,97,110,111,1,0,0,0,198,
     1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,169,1,0,0,
     0,0,0,0,8,0,0,0,99,111,110,99,97,116,118,50,1,0,0,0,9,0,0,0,67,111,110,99,97,116,86,50,
     68,2,0,0,0,12,0,0,0,8,0,0,0,99,111,110,99,97,116,118,50,4,0,0,0,180,0,0,0,2,0,0,
     0,1,0,0,0,1,0,0,0,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,120,114,0,
     0,0,0,0,0,48,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,0,0,0,0,60,57,0,0,0,0,0,
     0,1,0,0,0,16,0,0,0,1,0,0,0,0,0,0,0,60,57,0,0,0,0,0,0,1,0,0,0,1,0,0,
     0,3,0,0,0,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,60,0,0,0,0,0,0,0,48,0,0,
     0,0,0,0,0,16,0,0,0,1,0,0,0,0,0,0,0,30,0,0,0,0,0,0,0,1,0,0,0,16,0,0,
     0,1,0,0,0,0,0,0,0,30,0,0,0,0,0,0,0,5,0,0,0,120,0,0,0,1,0,0,0,1,0,0,
     0,1,0,0,0,3,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,16,0,0,0,0,0,0,0,0,0,0,
     0,0,0,0,0,180,114,0,0,0,0,0,0,64,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,0,0,0,
     0,90,57,0,0,0,0,0,0,1,0,0,0,16,0,0,0,1,0,0,0,0,0,0,0,90,57,0,0,0,0,0,
     0,2,0,0,0,8,0,0,0,99,111,110,99,97,116,118,50,8,0,0,0,48,0,0,0,1,0,0,0,0,115,0,
     0,0,0,0,0,224,114,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,224,229,0,0,0,0,0,0,0,0,0,0};
  int len = sizeof(file) / sizeof(uint8_t);
  void *outBuff = nullptr;
  outBuff = malloc(len);
  (void)memcpy_s(outBuff, len, file, len);
  std::string fileName;
  fileName.append("./concat_nano").append(".dbg");
  FILE *f = fopen(fileName.c_str(), "wb+");
  fwrite((uint8_t *)outBuff, len, sizeof(char), f);
  fclose(f);
  free(outBuff);
  int testNum = 1;
  void *ptr = &testNum;
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmMalloc(_)).WillRepeatedly(Invoke(mmMalloc_Normal_Invoke));
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmReadFile(_,_,_,_)).WillRepeatedly(Invoke(mmReadFile_Normal_Invoke));
  EXPECT_CALL(RtStubMock::GetInstance(), dlopen(_,_)).Times(1).WillOnce(Return(ptr));
  EXPECT_CALL(RtStubMock::GetInstance(), dlclose(_)).Times(1).WillOnce(Return(0));
  EXPECT_CALL(RtStubMock::GetInstance(), dlsym(_,_)).Times(2).WillOnce(Return((void *)StopDataDumpMock))
    .WillOnce(Return((void *)StopDataDumpMock));
  EXPECT_CALL(RtStubMock::GetInstance(), rtSetTaskDescDumpFlag(_,_,_)).Times(2).WillOnce(Return(-1)).WillOnce(Return(RT_ERROR_NONE));
  EXPECT_CALL(RtStubMock::GetInstance(), rtMsgSend(_,_,_,_,_)).Times(4).WillOnce(Return(-1)).WillOnce(Return(RT_ERROR_NONE)).WillOnce(Return(-1))
    .WillOnce(Return(RT_ERROR_NONE));
  const char *filePath = "../tests/test_c/ut/testcase/data/acl.json";
  Status ret = DbgDumpInit(filePath);
  ASSERT_EQ(ret, SUCCESS);
  void *handle0 = DbgCreateModelHandle();
  ASSERT_NE(handle0, nullptr);
  rtMdlLoad_t mdlLoad;
  mdlLoad.totalTaskNum = 1;
  std::string dbgFilePath = "concat_nano";
  size_t taskDescSize = 1000UL;
  ret = DbgDumpPreProcess(&mdlLoad, taskDescSize, (char *)dbgFilePath.c_str(), handle0);
  ASSERT_EQ(ret, SUCCESS);
  DbgFreeLoadDumpInfo(handle0);

  void *handle = DbgCreateModelHandle();
  ASSERT_NE(handle, nullptr);
  ret = DbgDumpPreProcess(&mdlLoad, taskDescSize, (char *)dbgFilePath.c_str(), handle);
  ASSERT_EQ(ret, SUCCESS);
  uint32_t modelId = 1U;
  uint64_t stepId = 1;
  uint64_t *stepIdAddr = &stepId;

  EXPECT_CALL(RtStubMock::GetInstance(), rtMalloc(_,_,_,_)).WillRepeatedly(Invoke(rtMalloc_Normal_Invoke));
  ret = DbgDumpPostProcess(modelId, stepIdAddr, handle);
  ASSERT_NE(ret, SUCCESS);
  DbgStepIdCounterPlus(handle);

  ret = DbgDumpPostProcess(modelId, stepIdAddr, handle);
  ASSERT_EQ(ret, SUCCESS);
  DbgStepIdCounterPlus(NULL);
  DbgStepIdCounterPlus(handle);
  DbgFreeLoadDumpInfo(NULL);
  ret = SendUnLoadInfoToAicpu((ModelDbgHandle *)handle); // covers rtMsgSend err
  ASSERT_NE(ret, SUCCESS);

  DbgFreeLoadDumpInfo(handle);
  ret = DbgDumpDeInit();
  ASSERT_EQ(ret, SUCCESS);
  remove(fileName.c_str());

  ret = SendUnLoadInfoToAicpu((ModelDbgHandle *)handle);
  ASSERT_NE(ret, SUCCESS);
}

TEST_F(UtestGEDBGTest, GeParseDbgCaseWithDumpProcessMemcpyFail) {
    uint8_t file[] =
    {0,0,0,0,90,90,90,90,0,0,0,0,11,0,0,0,99,111,110,99,97,116,95,110,97,110,111,1,0,0,0,198,
     1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,169,1,0,0,
     0,0,0,0,8,0,0,0,99,111,110,99,97,116,118,50,1,0,0,0,9,0,0,0,67,111,110,99,97,116,86,50,
     68,2,0,0,0,12,0,0,0,8,0,0,0,99,111,110,99,97,116,118,50,4,0,0,0,180,0,0,0,2,0,0,
     0,1,0,0,0,1,0,0,0,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,120,114,0,
     0,0,0,0,0,48,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,0,0,0,0,60,57,0,0,0,0,0,
     0,1,0,0,0,16,0,0,0,1,0,0,0,0,0,0,0,60,57,0,0,0,0,0,0,1,0,0,0,1,0,0,
     0,3,0,0,0,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,60,0,0,0,0,0,0,0,48,0,0,
     0,0,0,0,0,16,0,0,0,1,0,0,0,0,0,0,0,30,0,0,0,0,0,0,0,1,0,0,0,16,0,0,
     0,1,0,0,0,0,0,0,0,30,0,0,0,0,0,0,0,5,0,0,0,120,0,0,0,1,0,0,0,1,0,0,
     0,1,0,0,0,3,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,16,0,0,0,0,0,0,0,0,0,0,
     0,0,0,0,0,180,114,0,0,0,0,0,0,64,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,0,0,0,
     0,90,57,0,0,0,0,0,0,1,0,0,0,16,0,0,0,1,0,0,0,0,0,0,0,90,57,0,0,0,0,0,
     0,2,0,0,0,8,0,0,0,99,111,110,99,97,116,118,50,8,0,0,0,48,0,0,0,1,0,0,0,0,115,0,
     0,0,0,0,0,224,114,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,224,229,0,0,0,0,0,0,0,0,0,0};
  int len = sizeof(file) / sizeof(uint8_t);
  void *outBuff = nullptr;
  outBuff = malloc(len);
  (void)memcpy_s(outBuff, len, file, len);
  std::string fileName;
  fileName.append("../tests/test_c/ut/testcase/data/concat_nano").append(".dbg");
  FILE *f = fopen(fileName.c_str(), "wb+");
  fwrite((uint8_t *)outBuff, len, sizeof(char), f);
  fclose(f);
  free(outBuff);
  int testNum = 1;
  void *ptr = &testNum;
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmMalloc(_)).WillRepeatedly(Invoke(mmMalloc_Normal_Invoke));
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmReadFile(_,_,_,_)).WillRepeatedly(Invoke(mmReadFile_Normal_Invoke));
  EXPECT_CALL(RtStubMock::GetInstance(), dlopen(_,_)).Times(1).WillOnce(Return(ptr));
  EXPECT_CALL(RtStubMock::GetInstance(), dlclose(_)).Times(1).WillOnce(Return(0));
  EXPECT_CALL(RtStubMock::GetInstance(), dlsym(_,_)).Times(2).WillOnce(Return((void *)StopDataDumpMock))
    .WillOnce(Return((void *)StopDataDumpMock));
  EXPECT_CALL(RtStubMock::GetInstance(), rtSetTaskDescDumpFlag(_,_,_)).Times(1).WillOnce(Return(RT_ERROR_NONE));
  const char *filePath = "../tests/test_c/ut/testcase/data/aclCase8.json";
  Status ret = DbgDumpInit(filePath);
  ASSERT_EQ(ret, SUCCESS);
  void *handle = DbgCreateModelHandle();
  ASSERT_NE(handle, nullptr);
  rtMdlLoad_t mdlLoad;
  mdlLoad.totalTaskNum = 1;
  std::string dbgFilePath = "concat_nano";
  size_t taskDescSize = 1000UL;
  ret = DbgDumpPreProcess(&mdlLoad, taskDescSize, (char *)dbgFilePath.c_str(), handle);
  ASSERT_EQ(ret, SUCCESS);
  DbgFreeLoadDumpInfo(handle);
  ret = DbgDumpDeInit();
  ASSERT_EQ(ret, SUCCESS);
  remove(fileName.c_str());
}

TEST_F(UtestGEDBGTest, GeParseDbgCaseUnnormal) {
    uint8_t file[] =
    {0,0,0,0,91,91,91,91,0,0,0,0,11,0,0,0,99,111,110,99,97,116,95,110,97,110,111,1,0,0,0,198,
     1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,169,1,0,0,
     0,0,0,0,8,0,0,0,99,111,110,99,97,116,118,50,1,0,0,0,9,0,0,0,67,111,110,99,97,116,86,50,
     68,2,0,0,0,12,0,0,0,8,0,0,0,99,111,110,99,97,116,118,50,4,0,0,0,180,0,0,0,2,0,0,
     0,1,0,0,0,1,0,0,0,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,120,114,0,
     0,0,0,0,0,48,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,0,0,0,0,60,57,0,0,0,0,0,
     0,1,0,0,0,16,0,0,0,1,0,0,0,0,0,0,0,60,57,0,0,0,0,0,0,1,0,0,0,1,0,0,
     0,3,0,0,0,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,60,0,0,0,0,0,0,0,48,0,0,
     0,0,0,0,0,16,0,0,0,1,0,0,0,0,0,0,0,30,0,0,0,0,0,0,0,1,0,0,0,16,0,0,
     0,1,0,0,0,0,0,0,0,30,0,0,0,0,0,0,0,5,0,0,0,120,0,0,0,1,0,0,0,1,0,0,
     0,1,0,0,0,3,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,16,0,0,0,0,0,0,0,0,0,0,
     0,0,0,0,0,180,114,0,0,0,0,0,0,64,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,0,0,0,
     0,90,57,0,0,0,0,0,0,1,0,0,0,16,0,0,0,1,0,0,0,0,0,0,0,90,57,0,0,0,0,0,
     0,2,0,0,0,8,0,0,0,99,111,110,99,97,116,118,50,8,0,0,0,48,0,0,0,1,0,0,0,0,115,0,
     0,0,0,0,0,224,114,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,224,229,0,0,0,0,0,0,0,0,0,0};
  int len = sizeof(file) / sizeof(uint8_t);
  void *outBuff = nullptr;
  outBuff = malloc(len);
  (void)memcpy_s(outBuff, len, file, len);
  std::string fileName;
  fileName.append("./concat_nano").append(".dbg");
  FILE *f = fopen(fileName.c_str(), "wb+");
  fwrite((uint8_t *)outBuff, len, sizeof(char), f);
  fclose(f);
  free(outBuff);
  int testNum = 1;
  void *ptr = &testNum;
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmMalloc(_)).WillRepeatedly(Invoke(mmMalloc_Normal_Invoke));
  EXPECT_CALL(RtStubMock::GetInstance(), dlopen(_,_)).Times(1).WillOnce(Return(ptr));
  EXPECT_CALL(RtStubMock::GetInstance(), dlclose(_)).Times(1).WillOnce(Return(0));
  EXPECT_CALL(RtStubMock::GetInstance(), dlsym(_,_)).Times(2).WillOnce(Return((void *)StopDataDumpMock))
    .WillOnce(Return((void *)StopDataDumpMock));
  const char *filePath = "../tests/test_c/ut/testcase/data/aclCase8.json";
  Status ret = DbgDumpInit(filePath);
  ASSERT_EQ(ret, SUCCESS);
  void *handle = DbgCreateModelHandle();
  ASSERT_NE(handle, nullptr);
  rtMdlLoad_t mdlLoad;
  mdlLoad.totalTaskNum = 1;
  std::string dbgFilePath = "concat_nano";
  size_t taskDescSize = 1000UL;

  std::string dbgFilePathInvalid = "concat_nano1";
  ret = DbgDumpPreProcess(&mdlLoad, taskDescSize, (char *)dbgFilePathInvalid.c_str(), handle);
  ASSERT_NE(ret, SUCCESS);

  ret = DbgDumpPreProcess(&mdlLoad, taskDescSize, (char *)dbgFilePath.c_str(), handle);
  ASSERT_NE(ret, SUCCESS);
  uint32_t modelId = 1U;
  uint64_t stepId = 1;
  uint64_t *stepIdAddr = &stepId;
  ret = DbgDumpPostProcess(modelId, stepIdAddr, handle);
  ASSERT_EQ(ret, SUCCESS);
  DbgFreeLoadDumpInfo(handle);
  ret = DbgDumpDeInit();
  ASSERT_EQ(ret, SUCCESS);
  remove(fileName.c_str());
}

TEST_F(UtestGEDBGTest, GeOpNameMatchOpErr) {
  int testNum = 1;
  void *ptr = &testNum;
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmMalloc(_)).WillRepeatedly(Invoke(mmMalloc_Normal_Invoke));
  EXPECT_CALL(RtStubMock::GetInstance(), dlopen(_,_)).Times(1).WillOnce(Return(ptr));
  EXPECT_CALL(RtStubMock::GetInstance(), dlclose(_)).Times(1).WillOnce(Return(0));
  EXPECT_CALL(RtStubMock::GetInstance(), dlsym(_,_)).Times(2).WillOnce(Return((void *)StopDataDumpMock))
    .WillOnce(Return((void *)StopDataDumpMock));
  const char *filePath = "../tests/test_c/ut/testcase/data/acl.json";
  Status ret = DbgDumpInit(filePath);
  ASSERT_EQ(ret, SUCCESS);
  const char *opName = "concatv2";
  uint16_t opNameLen = strlen(opName);
  const char *mdlName = "concat_nano";
  bool result = IsOpNameMatch((uint8_t *)opName, opNameLen, mdlName);
  ASSERT_EQ(result, true);

  const char *opNameUnnormal = "concatv2_1";
  uint16_t opNameLenUnnormal = strlen(opNameUnnormal);
  result = IsOpNameMatch((uint8_t *)opNameUnnormal, opNameLenUnnormal, mdlName);
  ASSERT_EQ(result, false);

  ret = DbgDumpDeInit();
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestGEDBGTest, GeOpNameMatchOpMallocErr) {
  int testNum = 1;
  void *ptr = &testNum;
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmMalloc(_)).WillRepeatedly(Invoke(mmMalloc_Normal_Invoke));
  EXPECT_CALL(RtStubMock::GetInstance(), dlopen(_,_)).Times(1).WillOnce(Return(ptr));
  EXPECT_CALL(RtStubMock::GetInstance(), dlclose(_)).Times(1).WillOnce(Return(0));
  EXPECT_CALL(RtStubMock::GetInstance(), dlsym(_,_)).Times(2).WillOnce(Return((void *)StopDataDumpMock))
    .WillOnce(Return((void *)StopDataDumpMock));
  const char *filePath = "../tests/test_c/ut/testcase/data/acl.json";
  Status ret = DbgDumpInit(filePath);
  ASSERT_EQ(ret, SUCCESS);
  const char *opName = "concatv2";
  uint16_t opNameLen = strlen(opName);
  const char *mdlName = "concat_nano";
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmMalloc(_)).Times(2).WillOnce(Invoke(mmMalloc_Abnormal_Invoke)).
    WillOnce(Invoke(mmMalloc_Abnormal_Invoke));
  bool result = IsOpNameMatch((uint8_t *)opName, opNameLen, mdlName);
  ASSERT_EQ(result, false);

  bool result2 = IsOriOpNameMatch((uint8_t *)opName, opNameLen, mdlName);
  ASSERT_EQ(result2, false);

  ret = DbgDumpDeInit();
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestGEDBGTest, GeOpNameMatchNull) {
  int testNum = 1;
  void *ptr = &testNum;
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmMalloc(_)).WillRepeatedly(Invoke(mmMalloc_Normal_Invoke));
  EXPECT_CALL(RtStubMock::GetInstance(), dlopen(_,_)).Times(1).WillOnce(Return(ptr));
  EXPECT_CALL(RtStubMock::GetInstance(), dlclose(_)).Times(1).WillOnce(Return(0));
  EXPECT_CALL(RtStubMock::GetInstance(), dlsym(_,_)).Times(2).WillOnce(Return((void *)StopDataDumpMock))
    .WillOnce(Return((void *)StopDataDumpMock));
  const char *filePath = "../tests/test_c/ut/testcase/data/acl.json";
  Status ret = DbgDumpInit(filePath);
  ASSERT_EQ(ret, SUCCESS);
  const char *mdlName = "concat_nano";
  bool result1 = IsOpNameMatch(nullptr, 0, mdlName);
  ASSERT_EQ(result1, false);
  bool result2 = IsOriOpNameMatch(nullptr, 0, mdlName);
  ASSERT_EQ(result2, false);
  ret = DbgDumpDeInit();
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestGEDBGTest, GeOpNameMatchMdlErr) {
  int testNum = 1;
  void *ptr = &testNum;
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmMalloc(_)).WillRepeatedly(Invoke(mmMalloc_Normal_Invoke));
  EXPECT_CALL(RtStubMock::GetInstance(), dlopen(_,_)).Times(1).WillOnce(Return(ptr));
  EXPECT_CALL(RtStubMock::GetInstance(), dlclose(_)).Times(1).WillOnce(Return(0));
  EXPECT_CALL(RtStubMock::GetInstance(), dlsym(_,_)).Times(2).WillOnce(Return((void *)StopDataDumpMock))
    .WillOnce(Return((void *)StopDataDumpMock));
  const char *filePath = "../tests/test_c/ut/testcase/data/acl.json";
  Status ret = DbgDumpInit(filePath);
  ASSERT_EQ(ret, SUCCESS);
  const char *opName = "concatv2";
  uint16_t opNameLen = strlen(opName);
  const char *mdlName = "concat_nano11";
  bool result1 = IsOpNameMatch((uint8_t *)opName, opNameLen, mdlName);
  ASSERT_EQ(result1, false);
  bool result2 = IsOriOpNameMatch((uint8_t *)opName, opNameLen, mdlName);
  ASSERT_EQ(result2, false);
  ret = DbgDumpDeInit();
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestGEDBGTest, GeOpNameMatchSuccess) {
  int testNum = 1;
  void *ptr = &testNum;
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmMalloc(_)).WillRepeatedly(Invoke(mmMalloc_Normal_Invoke));
  EXPECT_CALL(RtStubMock::GetInstance(), dlopen(_,_)).Times(1).WillOnce(Return(ptr));
  EXPECT_CALL(RtStubMock::GetInstance(), dlclose(_)).Times(1).WillOnce(Return(0));
  EXPECT_CALL(RtStubMock::GetInstance(), dlsym(_,_)).Times(2).WillOnce(Return((void *)StopDataDumpMock))
    .WillOnce(Return((void *)StopDataDumpMock));
  const char *filePath = "../tests/test_c/ut/testcase/data/aclCase6.json";
  Status ret = DbgDumpInit(filePath);
  ASSERT_EQ(ret, SUCCESS);
  const char *opName = "concatv2";
  uint16_t opNameLen = strlen(opName);
  const char *mdlName = "concat_nano";
  bool result1 = IsOpNameMatch((uint8_t *)opName, opNameLen, mdlName);
  ASSERT_EQ(result1, true);
  bool result2 = IsOriOpNameMatch((uint8_t *)opName, opNameLen, mdlName);
  ASSERT_EQ(result2, true);
  ret = DbgDumpDeInit();
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestGEDBGTest, GeDumpPostProcessCaseWithFormatErr) {
  const char *filePath4 = "../tests/test_c/ut/testcase/data/aclCase4.json";
  Status ret = DbgDumpInit(filePath4);
  ASSERT_EQ(ret, SUCCESS);
  uint32_t modelId = 0U;
  uint64_t stepId = 1;
  uint64_t *stepIdAddr = &stepId;
  void *dbgHandle = nullptr;
  ret = DbgDumpPostProcess(modelId, stepIdAddr, dbgHandle);
  ASSERT_EQ(ret, SUCCESS);
  DbgStepIdCounterPlus(dbgHandle);
  ret = DbgDumpDeInit();
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestGEDBGTest, GeDumpPostProcessCaseWithDumpOn) {
    uint8_t file[] =
    {0,0,0,0,90,90,90,90,0,0,0,0,11,0,0,0,99,111,110,99,97,116,95,110,97,110,111,1,0,0,0,198,
     1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,169,1,0,0,
     0,0,0,0,8,0,0,0,99,111,110,99,97,116,118,50,1,0,0,0,9,0,0,0,67,111,110,99,97,116,86,50,
     68,2,0,0,0,12,0,0,0,8,0,0,0,99,111,110,99,97,116,118,50,4,0,0,0,180,0,0,0,2,0,0,
     0,1,0,0,0,1,0,0,0,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,120,114,0,
     0,0,0,0,0,48,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,0,0,0,0,60,57,0,0,0,0,0,
     0,1,0,0,0,16,0,0,0,1,0,0,0,0,0,0,0,60,57,0,0,0,0,0,0,1,0,0,0,1,0,0,
     0,3,0,0,0,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,60,0,0,0,0,0,0,0,48,0,0,
     0,0,0,0,0,16,0,0,0,1,0,0,0,0,0,0,0,30,0,0,0,0,0,0,0,1,0,0,0,16,0,0,
     0,1,0,0,0,0,0,0,0,30,0,0,0,0,0,0,0,5,0,0,0,120,0,0,0,1,0,0,0,1,0,0,
     0,1,0,0,0,3,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,16,0,0,0,0,0,0,0,0,0,0,
     0,0,0,0,0,180,114,0,0,0,0,0,0,64,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,0,0,0,
     0,90,57,0,0,0,0,0,0,1,0,0,0,16,0,0,0,1,0,0,0,0,0,0,0,90,57,0,0,0,0,0,
     0,2,0,0,0,8,0,0,0,99,111,110,99,97,116,118,50,8,0,0,0,48,0,0,0,1,0,0,0,0,115,0,
     0,0,0,0,0,224,114,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,224,229,0,0,0,0,0,0,0,0,0,0};
  int len = sizeof(file) / sizeof(uint8_t);
  void *outBuff = nullptr;
  outBuff = malloc(len);
  (void)memcpy_s(outBuff, len, file, len);
  std::string fileName;
  fileName.append("./concat_nano").append(".dbg");
  FILE *f = fopen(fileName.c_str(), "wb+");
  fwrite((uint8_t *)outBuff, len, sizeof(char), f);
  fclose(f);
  free(outBuff);
  int testNum = 1;
  void *ptr = &testNum;
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmMalloc(_)).WillRepeatedly(Invoke(mmMalloc_Normal_Invoke));
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmReadFile(_,_,_,_)).WillRepeatedly(Invoke(mmReadFile_Normal_Invoke));
  EXPECT_CALL(RtStubMock::GetInstance(), dlopen(_,_)).Times(1).WillOnce(Return(ptr));
  EXPECT_CALL(RtStubMock::GetInstance(), dlclose(_)).Times(1).WillOnce(Return(0));
  EXPECT_CALL(RtStubMock::GetInstance(), dlsym(_,_)).Times(2).WillOnce(Return((void *)StopDataDumpMock))
    .WillOnce(Return((void *)StopDataDumpMock));
  const char *filePath = "../tests/test_c/ut/testcase/data/aclCase7.json";
  Status ret = DbgDumpInit(filePath);
  ASSERT_EQ(ret, SUCCESS);
  void *handle = DbgCreateModelHandle();
  ASSERT_NE(handle, nullptr);
  rtMdlLoad_t mdlLoad;
  mdlLoad.totalTaskNum = 1;
  std::string dbgFilePath = "concat_nano";
  size_t taskDescSize = 1000UL;

  EXPECT_CALL(MmpaStubMock::GetInstance(), mmReadFile(_,_,_,_)).Times(1).WillOnce(Invoke(mmReadFile_Abnormal_Invoke));
  ret = DbgDumpPreProcess(&mdlLoad, taskDescSize, (char *)dbgFilePath.c_str(), handle);
  ASSERT_NE(ret, SUCCESS); // cover mmReadFile fail in ParseDbgFile
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmReadFile(_,_,_,_)).WillRepeatedly(Invoke(mmReadFile_Normal_Invoke));

  EXPECT_CALL(MmpaStubMock::GetInstance(), mmMalloc(_)).WillRepeatedly(Invoke(mmMalloc_Normal_Invoke));
  ret = DbgDumpPreProcess(&mdlLoad, taskDescSize, (char *)dbgFilePath.c_str(), handle);
  ASSERT_EQ(ret, SUCCESS);
  uint32_t modelId = 1U;
  uint64_t stepId = 1;
  uint64_t *stepIdAddr = &stepId;
  ret = DbgDumpPostProcess(modelId, stepIdAddr, handle);
  ASSERT_EQ(ret, SUCCESS);
  DbgFreeLoadDumpInfo(handle);
  ret = DbgDumpDeInit();
  ASSERT_EQ(ret, SUCCESS);
  remove(fileName.c_str());
}

TEST_F(UtestGEDBGTest, GeDumpOverFlowCaseWithEmptyList) {
  uint8_t file[] =
    {0,0,0,0,90,90,90,90,0,0,0,0,11,0,0,0,99,111,110,99,97,116,95,110,97,110,111,1,0,0,0,198,
     1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,169,1,0,0,
     0,0,0,0,8,0,0,0,99,111,110,99,97,116,118,50,1,0,0,0,9,0,0,0,67,111,110,99,97,116,86,50,
     68,2,0,0,0,12,0,0,0,8,0,0,0,99,111,110,99,97,116,118,50,4,0,0,0,180,0,0,0,2,0,0,
     0,1,0,0,0,1,0,0,0,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,120,114,0,
     0,0,0,0,0,48,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,0,0,0,0,60,57,0,0,0,0,0,
     0,1,0,0,0,16,0,0,0,1,0,0,0,0,0,0,0,60,57,0,0,0,0,0,0,1,0,0,0,1,0,0,
     0,3,0,0,0,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,60,0,0,0,0,0,0,0,48,0,0,
     0,0,0,0,0,16,0,0,0,1,0,0,0,0,0,0,0,30,0,0,0,0,0,0,0,1,0,0,0,16,0,0,
     0,1,0,0,0,0,0,0,0,30,0,0,0,0,0,0,0,5,0,0,0,120,0,0,0,1,0,0,0,1,0,0,
     0,1,0,0,0,3,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,16,0,0,0,0,0,0,0,0,0,0,
     0,0,0,0,0,180,114,0,0,0,0,0,0,64,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,0,0,0,
     0,90,57,0,0,0,0,0,0,1,0,0,0,16,0,0,0,1,0,0,0,0,0,0,0,90,57,0,0,0,0,0,
     0,2,0,0,0,8,0,0,0,99,111,110,99,97,116,118,50,8,0,0,0,48,0,0,0,1,0,0,0,0,115,0,
     0,0,0,0,0,224,114,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,224,229,0,0,0,0,0,0,0,0,0,0};
  int len = sizeof(file) / sizeof(uint8_t);
  void *outBuff = nullptr;
  outBuff = malloc(len);
  (void)memcpy_s(outBuff, len, file, len);
  std::string fileName;
  fileName.append("./concat_nano").append(".dbg");
  FILE *f = fopen(fileName.c_str(), "wb+");
  fwrite((uint8_t *)outBuff, len, sizeof(char), f);
  fclose(f);
  free(outBuff);
  int testNum = 1;
  void *ptr = &testNum;
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmMalloc(_)).WillRepeatedly(Invoke(mmMalloc_Normal_Invoke));
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmReadFile(_,_,_,_)).WillRepeatedly(Invoke(mmReadFile_Normal_Invoke));
  EXPECT_CALL(RtStubMock::GetInstance(), dlopen(_,_)).Times(1).WillOnce(Return(ptr));
  EXPECT_CALL(RtStubMock::GetInstance(), dlclose(_)).Times(1).WillOnce(Return(0));
  EXPECT_CALL(RtStubMock::GetInstance(), dlsym(_,_)).Times(2).WillOnce(Return((void *)StopDataDumpMock))
    .WillOnce(Return((void *)StopDataDumpMock));
  EXPECT_CALL(RtStubMock::GetInstance(), rtMsgSend(_,_,_,_,_)).Times(2).WillOnce(Return(RT_ERROR_NONE)).WillOnce(Return(RT_ERROR_NONE));
  const char *filePath = "../tests/test_c/ut/testcase/data/aclCase9.json";
  Status ret = DbgDumpInit(filePath);
  ASSERT_EQ(ret, SUCCESS);
  void *handle = DbgCreateModelHandle();
  ASSERT_NE(handle, nullptr);
  rtMdlLoad_t mdlLoad;
  mdlLoad.totalTaskNum = 1;
  std::string dbgFilePath = "concat_nano";
  size_t taskDescSize = 1000UL;

  EXPECT_CALL(MmpaStubMock::GetInstance(), mmMalloc(_)).Times(2).WillOnce(Invoke(mmMalloc_Normal_Invoke))
    .WillOnce(Invoke(mmMalloc_Abnormal_Invoke));
  ret = DbgDumpPreProcess(&mdlLoad, taskDescSize, (char *)dbgFilePath.c_str(), handle);
  ASSERT_NE(ret, SUCCESS); // covers malloc fail in ParseDbgFile
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmMalloc(_)).WillRepeatedly(Invoke(mmMalloc_Normal_Invoke));

  ret = DbgDumpPreProcess(&mdlLoad, taskDescSize, (char *)dbgFilePath.c_str(), handle);
  ASSERT_EQ(ret, SUCCESS); // covers debug_en == 1

  uint32_t modelId = 1U;
  uint64_t stepId = 1;
  uint64_t *stepIdAddr = &stepId;
  ret = DbgDumpPostProcess(modelId, stepIdAddr, handle);
  ASSERT_EQ(ret, SUCCESS);
  DbgFreeLoadDumpInfo(handle);
  ret = DbgDumpDeInit();
  ASSERT_EQ(ret, SUCCESS);
  remove(fileName.c_str());
}

TEST_F(UtestGEDBGTest, GeDumpOverFlowCaseWithNoList) {
  int testNum = 1;
  void *ptr = &testNum;
  EXPECT_CALL(RtStubMock::GetInstance(), dlopen(_,_)).Times(1).WillOnce(Return(ptr));
  EXPECT_CALL(RtStubMock::GetInstance(), dlclose(_)).Times(1).WillOnce(Return(0));
  EXPECT_CALL(RtStubMock::GetInstance(), dlsym(_,_)).Times(2).WillOnce(Return((void *)StopDataDumpMock))
    .WillOnce(Return((void *)StopDataDumpMock));
  const char *filePath = "../tests/test_c/ut/testcase/data/aclCase10.json";
  Status ret = DbgDumpInit(filePath);
  ASSERT_EQ(ret, SUCCESS);
  ret = DbgDumpDeInit();
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestGEDBGTest, GeDumpOverFlowCaseWithNormalList) {
  const char *filePath = "../tests/test_c/ut/testcase/data/aclCase13.json";
  Status ret = DbgDumpInit(filePath);
  ASSERT_EQ(ret, SUCCESS);
  ret = DbgDumpDeInit();
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestGEDBGTest, GeDumpExceptionCaseWithSwitchOff) {
  uint8_t file[] =
    {0,0,0,0,90,90,90,90,0,0,0,0,11,0,0,0,99,111,110,99,97,116,95,110,97,110,111,1,0,0,0,198,
     1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,169,1,0,0,
     0,0,0,0,8,0,0,0,99,111,110,99,97,116,118,50,1,0,0,0,9,0,0,0,67,111,110,99,97,116,86,50,
     68,2,0,0,0,12,0,0,0,8,0,0,0,99,111,110,99,97,116,118,50,4,0,0,0,180,0,0,0,2,0,0,
     0,1,0,0,0,1,0,0,0,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,120,114,0,
     0,0,0,0,0,48,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,0,0,0,0,60,57,0,0,0,0,0,
     0,1,0,0,0,16,0,0,0,1,0,0,0,0,0,0,0,60,57,0,0,0,0,0,0,1,0,0,0,1,0,0,
     0,3,0,0,0,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,60,0,0,0,0,0,0,0,48,0,0,
     0,0,0,0,0,16,0,0,0,1,0,0,0,0,0,0,0,30,0,0,0,0,0,0,0,1,0,0,0,16,0,0,
     0,1,0,0,0,0,0,0,0,30,0,0,0,0,0,0,0,5,0,0,0,120,0,0,0,1,0,0,0,1,0,0,
     0,1,0,0,0,3,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,16,0,0,0,0,0,0,0,0,0,0,
     0,0,0,0,0,180,114,0,0,0,0,0,0,64,0,0,0,0,0,0,0,16,0,0,0,1,0,0,0,0,0,0,
     0,90,57,0,0,0,0,0,0,1,0,0,0,16,0,0,0,1,0,0,0,0,0,0,0,90,57,0,0,0,0,0,
     0,2,0,0,0,8,0,0,0,99,111,110,99,97,116,118,50,8,0,0,0,48,0,0,0,1,0,0,0,0,115,0,
     0,0,0,0,0,224,114,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,224,229,0,0,0,0,0,0,0,0,0,0};
  int len = sizeof(file) / sizeof(uint8_t);
  void *outBuff = nullptr;
  outBuff = malloc(len);
  (void)memcpy_s(outBuff, len, file, len);
  std::string fileName;
  fileName.append("./concat_nano").append(".dbg");
  FILE *f = fopen(fileName.c_str(), "wb+");
  fwrite((uint8_t *)outBuff, len, sizeof(char), f);
  fclose(f);
  free(outBuff);
  int testNum = 1;
  void *ptr = &testNum;
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmMalloc(_)).WillRepeatedly(Invoke(mmMalloc_Normal_Invoke));
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmReadFile(_,_,_,_)).WillRepeatedly(Invoke(mmReadFile_Normal_Invoke));
  EXPECT_CALL(RtStubMock::GetInstance(), dlopen(_,_)).Times(1).WillOnce(Return(ptr));
  EXPECT_CALL(RtStubMock::GetInstance(), dlclose(_)).Times(1).WillOnce(Return(0));
  EXPECT_CALL(RtStubMock::GetInstance(), dlsym(_,_)).Times(2).WillOnce(Return((void *)StopDataDumpMock))
    .WillOnce(Return((void *)StopDataDumpMock));
  EXPECT_CALL(RtStubMock::GetInstance(), rtMsgSend(_,_,_,_,_)).Times(2).WillOnce(Return(RT_ERROR_NONE)).WillOnce(Return(RT_ERROR_NONE));
  const char *filePath = "../tests/test_c/ut/testcase/data/aclCase11.json";
  Status ret = DbgDumpInit(filePath);
  ASSERT_EQ(ret, SUCCESS);
  void *handle = DbgCreateModelHandle();
  ASSERT_NE(handle, nullptr);
  rtMdlLoad_t mdlLoad;
  mdlLoad.totalTaskNum = 1;
  std::string dbgFilePath = "concat_nano";
  size_t taskDescSize = 1000UL;
  ret = DbgDumpPreProcess(&mdlLoad, taskDescSize, (char *)dbgFilePath.c_str(), handle);
  ASSERT_EQ(ret, SUCCESS);
  uint32_t modelId = 1U;
  uint64_t stepId = 1;
  uint64_t *stepIdAddr = &stepId;
  ret = DbgDumpPostProcess(modelId, stepIdAddr, handle);
  ASSERT_EQ(ret, SUCCESS);
  DbgFreeLoadDumpInfo(handle);
  ret = DbgDumpDeInit();
  ASSERT_EQ(ret, SUCCESS);
  remove(fileName.c_str());
}

TEST_F(UtestGEDBGTest, GeDumpExceptionCaseWithSwitchOn) {
  const char *filePath = "../tests/test_c/ut/testcase/data/aclCase14.json";
  Status ret = DbgDumpInit(filePath);
  ASSERT_EQ(ret, SUCCESS);
  ret = DbgDumpDeInit();
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestGEDBGTest, GeDumpNormal) {
  int testNum = 1;
  void *ptr = &testNum;
  EXPECT_CALL(RtStubMock::GetInstance(), dlopen(_,_)).Times(1).WillOnce(Return(ptr));
  EXPECT_CALL(RtStubMock::GetInstance(), dlclose(_)).Times(1).WillOnce(Return(0));
  EXPECT_CALL(RtStubMock::GetInstance(), dlsym(_,_)).Times(2).WillOnce(Return((void *)StopDataDumpMock))
    .WillOnce(Return((void *)StopDataDumpMock));
  const char *filePath = "../tests/test_c/ut/testcase/data/aclCase12.json";
  Status ret = DbgDumpInit(filePath);
  ASSERT_EQ(ret, SUCCESS);
  ret = DbgDumpDeInit();
  ASSERT_EQ(ret, SUCCESS);
}