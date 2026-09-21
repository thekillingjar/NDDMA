/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <iostream>
#include <stdio.h>
#include <cstring>

#define protected public
#define private public

#undef private
#undef protected

#include "acl/acl.h"
#include "acl_ge_stub.h"
#include "acl/acl_base.h"
#include "model_desc_internal.h"
#include "vector.h"
#include "mmpa_api.h"
#include "ge_executor_stub.h"

using namespace std;
using namespace testing;
using ::testing::Return;
using ::testing::Eq;
using ::testing::NotNull;
using ::testing::Invoke;
using ::testing::_;
class AclMdlTest : public testing::Test {
protected:
 void SetUp(){
  ON_CALL(MmpaStubMock::GetInstance(), mmMalloc(_))
      .WillByDefault(Invoke(mmMalloc_Normal_Invoke));
  }

  void TearDown(){
    Mock::VerifyAndClearExpectations(&MmpaStubMock::GetInstance());
    Mock::VerifyAndClearExpectations(&MockFunctionTest::aclStubInstance());
    Mock::VerifyAndClearExpectations(&GeExecutorStubMock::GetInstance());
  }
};

TEST_F(AclMdlTest, aclmdlCreateDataset_mmMallocErr) {
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmMalloc(_)).WillOnce(Invoke(mmMalloc_Abnormal_Invoke));
  aclmdlDataset *dataSet = aclmdlCreateDataset();
  EXPECT_EQ(dataSet, nullptr);
}

TEST_F(AclMdlTest, DataSet) {
  aclmdlDataset *dataSet = aclmdlCreateDataset();
  EXPECT_NE(dataSet, nullptr);
  aclError ret = aclmdlAddDatasetBuffer(dataSet, nullptr);
  EXPECT_NE(ret, ACL_SUCCESS);
  ret = aclmdlAddDatasetBuffer(dataSet, (aclDataBuffer *)0x01);
  EXPECT_EQ(ret, ACL_SUCCESS);
  size_t size = aclmdlGetDatasetNumBuffers(nullptr);
  EXPECT_EQ(size, 0);
  size = aclmdlGetDatasetNumBuffers(dataSet);
  EXPECT_EQ(size, 1);
  aclDataBuffer *dataBuffer = aclmdlGetDatasetBuffer(dataSet, 1);
  EXPECT_EQ(dataBuffer, nullptr);
  dataBuffer = aclmdlGetDatasetBuffer(nullptr, 0);
  EXPECT_EQ(dataBuffer, nullptr);
  dataBuffer = aclmdlGetDatasetBuffer(dataSet, 0);
  EXPECT_NE(dataBuffer, nullptr);
  ret = aclmdlAddDatasetBuffer(nullptr, nullptr);
  EXPECT_NE(ret, ACL_SUCCESS);
  ret = aclmdlDestroyDataset(nullptr);
  EXPECT_NE(ret, ACL_SUCCESS);
  DataSet *dataSetTmp = (DataSet *)dataSet;
  dataSetTmp->io_addr = (uint64_t *)malloc(sizeof(uint64_t));
  ret = aclmdlDestroyDataset(dataSet);
  EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(AclMdlTest, aclmdlQuerySize_normal) {
  const char *fileName = "/home";
  size_t memSize;
  size_t weightSize;
  EXPECT_CALL(GeExecutorStubMock::GetInstance(), GetMemAndWeightSize(_,_,_)).Times(1).WillOnce(Return(SUCCESS));
  aclError ret = aclmdlQuerySize(fileName, &memSize, &weightSize);
  EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(AclMdlTest, aclmdlQuerySize_abnormal) {
  const char *fileName = "/home";
  size_t memSize;
  size_t weightSize;

  aclError ret = aclmdlQuerySize(nullptr, nullptr, nullptr);
  EXPECT_NE(ret, ACL_SUCCESS);

  EXPECT_CALL(GeExecutorStubMock::GetInstance(), GetMemAndWeightSize(_,_,_)).Times(1).WillOnce(Return(ACL_ERROR_INVALID_PARAM));
  ret = aclmdlQuerySize(fileName, &memSize, &weightSize);
  EXPECT_NE(ret, ACL_SUCCESS);
}

TEST_F(AclMdlTest, aclmdlQueryExeOMDesc_normal) {
  const char *fileName = "/home";
  aclmdlExeOMDesc mdlPartitionSize;
  EXPECT_CALL(GeExecutorStubMock::GetInstance(), GetPartitionSize(_,_)).Times(1).WillOnce(Return(SUCCESS));
  aclError ret = aclmdlQueryExeOMDesc(fileName, &mdlPartitionSize);
  EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(AclMdlTest, aclmdlQueryExeOMDesc_abnormal) {
  const char *fileName = "/home";
  aclmdlExeOMDesc mdlPartitionSize;
  aclError ret = aclmdlQueryExeOMDesc(nullptr, nullptr);
  EXPECT_NE(ret, ACL_SUCCESS);

  EXPECT_CALL(GeExecutorStubMock::GetInstance(), GetPartitionSize(_,_)).Times(1).WillOnce(Return(ACL_ERROR_INVALID_PARAM));
  ret = aclmdlQueryExeOMDesc(fileName, &mdlPartitionSize);
  EXPECT_NE(ret, ACL_SUCCESS);
}

TEST_F(AclMdlTest, aclmdlExecuteAsyncV2_paramInvalid) {
  aclError ret = aclmdlExecuteAsyncV2(1, nullptr, nullptr, nullptr, nullptr);
  EXPECT_NE(ret, ACL_SUCCESS);
}

TEST_F(AclMdlTest, aclmdlExecuteAsyncV2_workSpacePtrNotSet) {
  aclmdlDataset *dataset = aclmdlCreateDataset();
  EXPECT_NE(dataset, nullptr);
  aclDataBuffer *dataBuffer = (aclDataBuffer *)malloc(100);
  aclError ret = aclmdlAddDatasetBuffer(dataset, dataBuffer);
  EXPECT_EQ(ret, ACL_SUCCESS);
  aclmdlExecConfigHandle handle;

  handle.workPtr = NULL;
  ret = aclmdlExecuteAsyncV2(1, dataset, dataset, nullptr, &handle);
  EXPECT_NE(ret, ACL_SUCCESS);

  aclrtStream stream = (void *)0x0010;
  ret = aclmdlExecuteAsyncV2(1, dataset, dataset, stream, &handle);
  EXPECT_NE(ret, ACL_SUCCESS);

  free(dataBuffer);
  ret = aclmdlDestroyDataset(dataset);
  EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(AclMdlTest, aclmdlExecuteAsyncV2_ExecModelErr) {
  aclmdlDataset *dataset = aclmdlCreateDataset();
  EXPECT_NE(dataset, nullptr);
  aclDataBuffer *dataBuffer = (aclDataBuffer *)malloc(100);
  aclError ret = aclmdlAddDatasetBuffer(dataset, dataBuffer);
  EXPECT_EQ(ret, ACL_SUCCESS);
  aclmdlExecConfigHandle handle;
  size_t workSize = 100;
  handle.workPtr = malloc(workSize);
  handle.workSize = workSize;
  uint32_t modelId = 1;
  EXPECT_CALL(GeExecutorStubMock::GetInstance(), ExecModel(_, _, _, _, _))
      .Times(1)
      .WillOnce(Return(ACL_ERROR_INVALID_PARAM));
  ret = aclmdlExecuteAsyncV2(modelId, dataset, dataset, nullptr, &handle);
  EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
  free(dataBuffer);
  free(handle.workPtr);
  ret = aclmdlDestroyDataset(dataset);
  EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(AclMdlTest, aclmdlExecuteAsyncV2_normal) {
  aclmdlDataset *dataset = aclmdlCreateDataset();
  EXPECT_NE(dataset, nullptr);
  aclDataBuffer *dataBuffer = (aclDataBuffer *)malloc(100);
  aclError ret = aclmdlAddDatasetBuffer(dataset, dataBuffer);
  EXPECT_EQ(ret, ACL_SUCCESS);
  aclmdlExecConfigHandle handle;
  size_t workSize = 100;
  handle.workPtr = malloc(workSize);
  handle.workSize = workSize;
  uint32_t modelId = 1;
  EXPECT_CALL(GeExecutorStubMock::GetInstance(), ExecModel(_, _, _, _, _))
      .Times(1)
      .WillOnce(Return(SUCCESS));
  ret = aclmdlExecuteAsyncV2(modelId, dataset, dataset, nullptr, &handle);
  EXPECT_EQ(ret, ACL_SUCCESS);
  free(dataBuffer);
  free(handle.workPtr);
  ret = aclmdlDestroyDataset(dataset);
  EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(AclMdlTest, aclmdlCreateExecConfigHandle_mmMallocErr) {
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmMalloc(_)).WillOnce(Invoke(mmMalloc_Abnormal_Invoke));
  aclmdlExecConfigHandle *handle = aclmdlCreateExecConfigHandle();
  EXPECT_EQ(handle, nullptr);
}

TEST_F(AclMdlTest, aclmdlSetExecConfigOpt_attrInvalid) {
  aclmdlExecConfigHandle *handle = aclmdlCreateExecConfigHandle();
  void *p = (void *)0x0001;
  // 1. not support this attr
  aclError ret = aclmdlSetExecConfigOpt(handle, ACL_MDL_STREAM_SYNC_TIMEOUT, &p, sizeof(p));
  EXPECT_NE(ret, ACL_SUCCESS);
  // 2. enum out of bounds
  ret = aclmdlSetExecConfigOpt(handle, (aclmdlExecConfigAttr)(ACL_MDL_MEC_TIMETHR_SIZET + 1), &p, sizeof(p));
  EXPECT_NE(ret, ACL_SUCCESS);
  aclmdlDestroyExecConfigHandle(handle);
}

TEST_F(AclMdlTest, aclmdlSetExecConfigOpt_valueSizeInvalid) {
  aclmdlExecConfigHandle *handle = aclmdlCreateExecConfigHandle();
  void *p = (void *)0x0001;
  aclError ret = aclmdlSetExecConfigOpt(handle, ACL_MDL_WORK_ADDR_PTR, &p, 0);
  EXPECT_NE(ret, ACL_SUCCESS);

  size_t size = 100UL;
  ret = aclmdlSetExecConfigOpt(handle, ACL_MDL_WORK_SIZET, &size, 0);
  EXPECT_NE(ret, ACL_SUCCESS);
  aclmdlDestroyExecConfigHandle(handle);
}

TEST_F(AclMdlTest, aclmdlSetExecConfigOpt_normal_setWorkSpace) {
  aclmdlExecConfigHandle *handle = aclmdlCreateExecConfigHandle();
   void *p = (void *)0x0001;
  size_t size = 100UL;
  aclError ret = aclmdlSetExecConfigOpt(handle, ACL_MDL_WORK_ADDR_PTR, &p, sizeof(p));
  EXPECT_EQ(ret, ACL_SUCCESS);
  EXPECT_EQ(handle->workPtr, p);

  ret = aclmdlSetExecConfigOpt(handle, ACL_MDL_WORK_SIZET, &size, sizeof(size));
  EXPECT_EQ(ret, ACL_SUCCESS);
  aclmdlDestroyExecConfigHandle(handle);

}

TEST_F(AclMdlTest, aclmdlSetExecConfigOpt_normal_setAICQOS) {
  aclmdlExecConfigHandle *handle = aclmdlCreateExecConfigHandle();
  size_t size = 100UL;
  aclError ret = aclmdlSetExecConfigOpt(handle, ACL_MDL_AICQOS_SIZET, &size, sizeof(size));
  EXPECT_EQ(ret, ACL_SUCCESS);
  aclmdlDestroyExecConfigHandle(handle);
}

TEST_F(AclMdlTest, aclmdlSetExecConfigOpt_normal_setAICOST) {
  aclmdlExecConfigHandle *handle = aclmdlCreateExecConfigHandle();
  size_t size = 100UL;
  aclError ret = aclmdlSetExecConfigOpt(handle, ACL_MDL_AICOST_SIZET, &size, sizeof(size));
  EXPECT_EQ(ret, ACL_SUCCESS);
  aclmdlDestroyExecConfigHandle(handle);
}

TEST_F(AclMdlTest, aclmdlSetExecConfigOpt_normal_setMPAIMID) {
  aclmdlExecConfigHandle *handle = aclmdlCreateExecConfigHandle();
  size_t size = 100UL;
  aclError ret = aclmdlSetExecConfigOpt(handle, ACL_MDL_MPAIMID_SIZET, &size, sizeof(size));
  EXPECT_EQ(ret, ACL_SUCCESS);
  aclmdlDestroyExecConfigHandle(handle);
}

TEST_F(AclMdlTest, aclmdlSetExecConfigOpt_normal_setMEC_TIMETHR) {
  aclmdlExecConfigHandle *handle = aclmdlCreateExecConfigHandle();
  size_t size = 100UL;
  aclError ret = aclmdlSetExecConfigOpt(handle, ACL_MDL_MEC_TIMETHR_SIZET, &size, sizeof(size));
  EXPECT_EQ(ret, ACL_SUCCESS);
  aclmdlDestroyExecConfigHandle(handle);
}

TEST_F(AclMdlTest, aclmdlExecuteV2_paramInvalid) {
  aclError ret = aclmdlExecuteV2(1, nullptr, nullptr, nullptr, nullptr);
  EXPECT_NE(ret, ACL_SUCCESS);
}

TEST_F(AclMdlTest, aclmdlExecuteV2_normal) {
  aclmdlDataset *dataset = aclmdlCreateDataset();
  EXPECT_NE(dataset, nullptr);

  aclDataBuffer *dataBuffer = (aclDataBuffer *)malloc(100);
  aclError ret = aclmdlAddDatasetBuffer(dataset, dataBuffer);
  EXPECT_EQ(ret, ACL_SUCCESS);
  aclmdlExecConfigHandle *handle = aclmdlCreateExecConfigHandle();
  void *p = (void *)0x0001;
  size_t size = 100UL;

  ret = aclmdlSetExecConfigOpt(handle, ACL_MDL_WORK_ADDR_PTR, &p, sizeof(p));
  EXPECT_EQ(ret, ACL_SUCCESS);
  EXPECT_EQ(handle->workPtr, p);

  ret = aclmdlSetExecConfigOpt(handle, ACL_MDL_WORK_SIZET, &size, sizeof(size));
  EXPECT_EQ(ret, ACL_SUCCESS);

  EXPECT_CALL(GeExecutorStubMock::GetInstance(), ExecModel(_,_,_,_,_)).Times(1).WillOnce(Return(SUCCESS));
  ret = aclmdlExecuteV2(1, dataset, dataset, nullptr, handle);
  EXPECT_EQ(ret, ACL_SUCCESS);

  free(dataBuffer);
  ret = aclmdlDestroyDataset(dataset);
  EXPECT_EQ(ret, ACL_SUCCESS);
  aclmdlDestroyExecConfigHandle(handle);
}


TEST_F(AclMdlTest, aclmdlCreateConfigHandle_mmMallocErr) {
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmMalloc(_)).WillOnce(Invoke(mmMalloc_Abnormal_Invoke));
  aclmdlConfigHandle *handle = aclmdlCreateConfigHandle();
  EXPECT_EQ(handle, nullptr);
}

TEST_F(AclMdlTest, aclmdlSetConfigOpt_attrErr) {
  aclmdlConfigHandle *handle = aclmdlCreateConfigHandle();
  size_t type = ACL_MDL_LOAD_FROM_FILE;
  //1. not support these atts, 0，7-14
  aclError ret = aclmdlSetConfigOpt(handle, aclmdlConfigAttr(ACL_MDL_WORKSPACE_MEM_OPTIMIZE), &type, 0);
  EXPECT_NE(ret, ACL_SUCCESS);
  //2. attr out of bounds
  ret = aclmdlSetConfigOpt(handle, aclmdlConfigAttr(0x100), &type, 0);
  EXPECT_NE(ret, ACL_SUCCESS);
  aclmdlDestroyConfigHandle(handle);
}

TEST_F(AclMdlTest, aclmdlSetConfigOpt_loadTypeErr) {
  aclmdlConfigHandle *handle = aclmdlCreateConfigHandle();
  size_t type = ACL_MDL_LOAD_FROM_FILE;
  //1. load type size invalid
  aclError ret = aclmdlSetConfigOpt(handle, ACL_MDL_LOAD_TYPE_SIZET, &type, 0);
  EXPECT_NE(ret, ACL_SUCCESS);

  //2. load type should be between 1 and 4
  type = ACL_MDL_LOAD_FROM_FILE_WITH_Q;
  ret = aclmdlSetConfigOpt(handle, ACL_MDL_LOAD_TYPE_SIZET, &type, sizeof(type));
  EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
  aclmdlDestroyConfigHandle(handle);
}

TEST_F(AclMdlTest, aclmdlSetConfigOpt_loadTypeSucc) {
  aclmdlConfigHandle *handle = aclmdlCreateConfigHandle();
  size_t type = ACL_MDL_LOAD_FROM_FILE;
  aclError ret  = aclmdlSetConfigOpt(handle, ACL_MDL_LOAD_TYPE_SIZET, &type, sizeof(type));
  EXPECT_EQ(ret, ACL_SUCCESS);
  aclmdlDestroyConfigHandle(handle);
}

TEST_F(AclMdlTest, aclmdlSetConfigOpt_mdlPath) {
  aclmdlConfigHandle *handle = aclmdlCreateConfigHandle();
  const char *path = "/home";
  // valueSize invalid
  aclError ret = aclmdlSetConfigOpt(handle, ACL_MDL_PATH_PTR, &path, 0);
  EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
  ret = aclmdlSetConfigOpt(handle, ACL_MDL_PATH_PTR, &path, sizeof(path));
  EXPECT_EQ(ret, ACL_SUCCESS);
  EXPECT_EQ(strcmp(handle->loadPath, path), 0);
  aclmdlDestroyConfigHandle(handle);
}

TEST_F(AclMdlTest, aclmdlSetConfigOpt_memPtr) {
  aclmdlConfigHandle *handle = aclmdlCreateConfigHandle();
  void *p = (void *)0x0004;;
  // valueSize invalid
  aclError ret = aclmdlSetConfigOpt(handle, ACL_MDL_MEM_ADDR_PTR, &p, 0);
  EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
  ret = aclmdlSetConfigOpt(handle, ACL_MDL_MEM_ADDR_PTR, &p, sizeof(p));
  EXPECT_EQ(ret, ACL_SUCCESS);
  aclmdlDestroyConfigHandle(handle);
}

TEST_F(AclMdlTest, aclmdlSetConfigOpt_memSize) {
  aclmdlConfigHandle *handle = aclmdlCreateConfigHandle();
  size_t memSize = 1;
  // valueSize invalid
  aclError ret = aclmdlSetConfigOpt(handle, ACL_MDL_MEM_SIZET, &memSize, 0);
  EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
  ret = aclmdlSetConfigOpt(handle, ACL_MDL_MEM_SIZET, &memSize, sizeof(memSize));
  EXPECT_EQ(ret, ACL_SUCCESS);
  aclmdlDestroyConfigHandle(handle);
}

TEST_F(AclMdlTest, aclmdlSetConfigOpt_weightAddr) {
  aclmdlConfigHandle *handle = aclmdlCreateConfigHandle();
  void *p = (void *)0x0004;
  // valueSize invalid
  aclError ret = aclmdlSetConfigOpt(handle, ACL_MDL_WEIGHT_ADDR_PTR, &p, 0);
  EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
  ret = aclmdlSetConfigOpt(handle, ACL_MDL_WEIGHT_ADDR_PTR, &p, sizeof(p));
  EXPECT_EQ(ret, ACL_SUCCESS);
  EXPECT_EQ(handle->exeOMInfo.weightPtr, p);
  aclmdlDestroyConfigHandle(handle);
}

TEST_F(AclMdlTest, aclmdlSetConfigOpt_weightSize) {
  aclmdlConfigHandle *handle = aclmdlCreateConfigHandle();
  size_t weightSize = 1;
  // valueSize invalid
  aclError ret = aclmdlSetConfigOpt(handle, ACL_MDL_WEIGHT_SIZET, &weightSize, 0);
  EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
  ret = aclmdlSetConfigOpt(handle, ACL_MDL_WEIGHT_SIZET, &weightSize, sizeof(weightSize));
  EXPECT_EQ(ret, ACL_SUCCESS);
  aclmdlDestroyConfigHandle(handle);
}

TEST_F(AclMdlTest, aclmdlSetConfigOpt_mdlDesPtr) {
  aclmdlConfigHandle *handle = aclmdlCreateConfigHandle();
  void *p = (void *)0x0004;
  // valueSize invalid
  aclError ret = aclmdlSetConfigOpt(handle, ACL_MDL_MODEL_DESC_PTR, &p, 0);
  EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
  ret = aclmdlSetConfigOpt(handle, ACL_MDL_MODEL_DESC_PTR, &p, sizeof(p));
  EXPECT_EQ(ret, ACL_SUCCESS);
  EXPECT_EQ(handle->exeOMInfo.modelDescPtr, p);
  aclmdlDestroyConfigHandle(handle);
}

TEST_F(AclMdlTest, aclmdlSetConfigOpt_mdlDesSize) {
  aclmdlConfigHandle *handle = aclmdlCreateConfigHandle();
  size_t mdlDesSize = 1;
  // valueSize invalid
  aclError ret = aclmdlSetConfigOpt(handle, ACL_MDL_MODEL_DESC_SIZET, &mdlDesSize, 0);
  EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
  ret = aclmdlSetConfigOpt(handle, ACL_MDL_MODEL_DESC_SIZET, &mdlDesSize, sizeof(mdlDesSize));
  EXPECT_EQ(ret, ACL_SUCCESS);
  aclmdlDestroyConfigHandle(handle);
}

TEST_F(AclMdlTest, aclmdlSetConfigOpt_kernelPtr) {
  aclmdlConfigHandle *handle = aclmdlCreateConfigHandle();
  void *p = (void *)0x0004;
  // valueSize invalid
  aclError ret = aclmdlSetConfigOpt(handle, ACL_MDL_KERNEL_PTR, &p, 0);
  EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
  ret = aclmdlSetConfigOpt(handle, ACL_MDL_KERNEL_PTR, &p, sizeof(p));
  EXPECT_EQ(ret, ACL_SUCCESS);
  EXPECT_EQ(handle->exeOMInfo.kernelPtr, p);
  aclmdlDestroyConfigHandle(handle);
}

TEST_F(AclMdlTest, aclmdlSetConfigOpt_kernelSize) {
  aclmdlConfigHandle *handle = aclmdlCreateConfigHandle();
  size_t kernelSize = 1;
  // valueSize invalid
  aclError ret = aclmdlSetConfigOpt(handle, ACL_MDL_KERNEL_SIZET, &kernelSize, 0);
  EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
  ret = aclmdlSetConfigOpt(handle, ACL_MDL_KERNEL_SIZET, &kernelSize, sizeof(kernelSize));
  EXPECT_EQ(ret, ACL_SUCCESS);
  aclmdlDestroyConfigHandle(handle);
}

TEST_F(AclMdlTest, aclmdlSetConfigOpt_kernelArgsPtr) {
  aclmdlConfigHandle *handle = aclmdlCreateConfigHandle();
  void *p = (void *)0x0004;
  // valueSize invalid
  aclError ret = aclmdlSetConfigOpt(handle, ACL_MDL_KERNEL_ARGS_PTR, &p, 0);
  EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
  ret = aclmdlSetConfigOpt(handle, ACL_MDL_KERNEL_ARGS_PTR, &p, sizeof(p));
  EXPECT_EQ(ret, ACL_SUCCESS);
  EXPECT_EQ(handle->exeOMInfo.kernelArgsPtr, p);
  aclmdlDestroyConfigHandle(handle);
}

TEST_F(AclMdlTest, aclmdlSetConfigOpt_kernelArgsSize) {
  aclmdlConfigHandle *handle = aclmdlCreateConfigHandle();
  size_t kernelArgsSize = 1;
  // valueSize invalid
  aclError ret = aclmdlSetConfigOpt(handle, ACL_MDL_KERNEL_ARGS_SIZET, &kernelArgsSize, 0);
  EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
  ret = aclmdlSetConfigOpt(handle, ACL_MDL_KERNEL_ARGS_SIZET, &kernelArgsSize, sizeof(kernelArgsSize));
  EXPECT_EQ(ret, ACL_SUCCESS);
  aclmdlDestroyConfigHandle(handle);
}

TEST_F(AclMdlTest, aclmdlSetConfigOpt_staticTaskPtr) {
  aclmdlConfigHandle *handle = aclmdlCreateConfigHandle();
  void *p = (void *)0x0004;
  // valueSize invalid
  aclError ret = aclmdlSetConfigOpt(handle, ACL_MDL_STATIC_TASK_PTR, &p, 0);
  EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
  ret = aclmdlSetConfigOpt(handle, ACL_MDL_STATIC_TASK_PTR, &p, sizeof(p));
  EXPECT_EQ(ret, ACL_SUCCESS);
  EXPECT_EQ(handle->exeOMInfo.staticTaskPtr, p);
  aclmdlDestroyConfigHandle(handle);
}

TEST_F(AclMdlTest, aclmdlSetConfigOpt_staticTaskSize) {
  aclmdlConfigHandle *handle = aclmdlCreateConfigHandle();
  size_t staticTaskSize = 1;
  // valueSize invalid
  aclError ret = aclmdlSetConfigOpt(handle, ACL_MDL_STATIC_TASK_SIZET, &staticTaskSize, 0);
  EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
  ret = aclmdlSetConfigOpt(handle, ACL_MDL_STATIC_TASK_SIZET, &staticTaskSize, sizeof(staticTaskSize));
  EXPECT_EQ(ret, ACL_SUCCESS);
  aclmdlDestroyConfigHandle(handle);
}

TEST_F(AclMdlTest, aclmdlSetConfigOpt_dynamicTaskPtr) {
  aclmdlConfigHandle *handle = aclmdlCreateConfigHandle();
  void *p = (void *)0x0004;
  // valueSize invalid
  aclError ret = aclmdlSetConfigOpt(handle, ACL_MDL_DYNAMIC_TASK_PTR, &p, 0);
  EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
  ret = aclmdlSetConfigOpt(handle, ACL_MDL_DYNAMIC_TASK_PTR, &p, sizeof(p));
  EXPECT_EQ(ret, ACL_SUCCESS);
  EXPECT_EQ(handle->exeOMInfo.dynamicTaskPtr, p);
  aclmdlDestroyConfigHandle(handle);
}

TEST_F(AclMdlTest, aclmdlSetConfigOpt_dynamicTaskSize) {
  aclmdlConfigHandle *handle = aclmdlCreateConfigHandle();
  size_t dynamicTaskSize = 1;
  // valueSize invalid
  aclError ret = aclmdlSetConfigOpt(handle, ACL_MDL_DYNAMIC_TASK_SIZET, &dynamicTaskSize, 0);
  EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
  ret = aclmdlSetConfigOpt(handle, ACL_MDL_DYNAMIC_TASK_SIZET, &dynamicTaskSize, sizeof(dynamicTaskSize));
  EXPECT_EQ(ret, ACL_SUCCESS);
  aclmdlDestroyConfigHandle(handle);
}

TEST_F(AclMdlTest, aclmdlSetConfigOpt_memType) {
  aclmdlConfigHandle *handle = aclmdlCreateConfigHandle();
  size_t memPolicy = (size_t)ACL_MEM_MALLOC_NORMAL_ONLY_P2P + 1;
  // memPolicy invalid
  aclError ret = aclmdlSetConfigOpt(handle, ACL_MDL_MEM_MALLOC_POLICY_SIZET, &memPolicy, sizeof(memPolicy));
  EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
  memPolicy = (size_t)ACL_MEM_MALLOC_HUGE_FIRST;
  ret = aclmdlSetConfigOpt(handle, ACL_MDL_MEM_MALLOC_POLICY_SIZET, &memPolicy, sizeof(memPolicy));
  EXPECT_EQ(ret, ACL_SUCCESS);
  EXPECT_EQ(handle->memType, RT_MEMORY_POLICY_HUGE_PAGE_FIRST | RT_MEMORY_DEFAULT);
  aclmdlDestroyConfigHandle(handle);
}

TEST_F(AclMdlTest, aclmdlSetConfigOpt_fifoPtr) {
  aclmdlConfigHandle *handle = aclmdlCreateConfigHandle();
  void *p = (void *)0x0004;
  // valueSize invalid
  aclError ret = aclmdlSetConfigOpt(handle, ACL_MDL_FIFO_PTR, &p, 0);
  EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
  ret = aclmdlSetConfigOpt(handle, ACL_MDL_FIFO_PTR, &p, sizeof(p));
  EXPECT_EQ(ret, ACL_SUCCESS);
  EXPECT_EQ(handle->exeOMInfo.fifoTaskPtr, p);
  aclmdlDestroyConfigHandle(handle);
}

TEST_F(AclMdlTest, aclmdlSetConfigOpt_fifoSize) {
  aclmdlConfigHandle *handle = aclmdlCreateConfigHandle();
  size_t fifoSize = 1;
  // valueSize invalid
  aclError ret = aclmdlSetConfigOpt(handle, ACL_MDL_FIFO_SIZET, &fifoSize, 0);
  EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
  ret = aclmdlSetConfigOpt(handle, ACL_MDL_FIFO_SIZET, &fifoSize, sizeof(fifoSize));
  EXPECT_EQ(ret, ACL_SUCCESS);
  EXPECT_EQ(handle->exeOMDesc.fifoTaskSize, fifoSize);
  aclmdlDestroyConfigHandle(handle);
}

TEST_F(AclMdlTest, aclmdlLoadWithConfig_paramNull) {
  uint32_t modelId;
  aclmdlConfigHandle handle;
  EXPECT_EQ(aclmdlLoadWithConfig(nullptr, &modelId), ACL_ERROR_INVALID_PARAM);
  EXPECT_EQ(aclmdlLoadWithConfig(&handle, nullptr), ACL_ERROR_INVALID_PARAM);
}

TEST_F(AclMdlTest, aclmdlLoadWithConfig_handleErr_loadTypeError) {
  uint32_t modelId;
  aclmdlConfigHandle *handle = aclmdlCreateConfigHandle();
  // 1. load type is not set in handle
  EXPECT_EQ(aclmdlLoadWithConfig(handle, &modelId), ACL_ERROR_INVALID_PARAM);

  // 2. mdl load type is invalid
  size_t type = ACL_MDL_LOAD_FROM_MEM;
  aclError ret = aclmdlSetConfigOpt(handle, ACL_MDL_LOAD_TYPE_SIZET, &type, sizeof(type));
  EXPECT_EQ(ret, ACL_SUCCESS);
  handle->mdlLoadType = ACL_MDL_LOAD_FROM_FILE_WITH_Q;
  ret = aclmdlLoadWithConfig(handle, &modelId);
  EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
  aclmdlDestroyConfigHandle(handle);
}

TEST_F(AclMdlTest, aclmdlLoadWithConfig_fromMem_handleErr_memErr) {
  aclmdlConfigHandle *handle = aclmdlCreateConfigHandle();
  size_t type = ACL_MDL_LOAD_FROM_MEM;
  aclError ret = aclmdlSetConfigOpt(handle, ACL_MDL_LOAD_TYPE_SIZET, &type, sizeof(type));
  EXPECT_EQ(ret, ACL_SUCCESS);
  uint32_t modelId;
  //1. model mem ptr is not set in handle
  ret = aclmdlLoadWithConfig(handle, &modelId);
  EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);

  //2. model mem size is not set in handle
  void *p = (void *)0x0004;
  ret = aclmdlSetConfigOpt(handle, ACL_MDL_MEM_ADDR_PTR, &p, sizeof(p));
  EXPECT_EQ(ret, ACL_SUCCESS);

  ret = aclmdlLoadWithConfig(handle, &modelId);
  EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
  aclmdlDestroyConfigHandle(handle);
}

TEST_F(AclMdlTest, aclmdlLoadWithConfig_fromMem_handleErr_memSizeErr) {
  aclmdlConfigHandle *handle = aclmdlCreateConfigHandle();
  size_t type = ACL_MDL_LOAD_FROM_MEM;
  void *p = (void *)0x0004;
  aclmdlSetConfigOpt(handle, ACL_MDL_LOAD_TYPE_SIZET, &type, sizeof(type));
  aclmdlSetConfigOpt(handle, ACL_MDL_MEM_ADDR_PTR, &p, sizeof(p));
  //modelSize is 0
  size_t modelSize = 0;
  aclmdlSetConfigOpt(handle, ACL_MDL_MEM_SIZET, &modelSize, sizeof(modelSize));
  uint32_t modelId;
  aclError ret = aclmdlLoadWithConfig(handle, &modelId);
  EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
  aclmdlDestroyConfigHandle(handle);
}

TEST_F(AclMdlTest, aclmdlLoadWithConfig_fromMem_abnormal_GeLoadModelFromDataErr) {
  // when loadFromMem, loadFromMemwithMem, handle must set loadtype, memptr, memsize
  aclmdlConfigHandle *handle = aclmdlCreateConfigHandle();
  size_t type = ACL_MDL_LOAD_FROM_MEM_WITH_MEM;
  aclmdlSetConfigOpt(handle, ACL_MDL_LOAD_TYPE_SIZET, &type, sizeof(type));
  void *p = (void *)0x0002;
  aclmdlSetConfigOpt(handle, ACL_MDL_MEM_ADDR_PTR, &p, sizeof(p));
  size_t modelSize = 1;
  aclmdlSetConfigOpt(handle, ACL_MDL_MEM_SIZET, &modelSize, sizeof(modelSize));

  uint32_t modelId;
  EXPECT_CALL(GeExecutorStubMock::GetInstance(), GeLoadModelFromData(_,_)).Times(1).WillOnce(Return(ACL_ERROR_GE_PARAM_INVALID));
  aclError ret = aclmdlLoadWithConfig(handle, &modelId);
  EXPECT_EQ(ret, ACL_ERROR_GE_PARAM_INVALID);
  aclmdlDestroyConfigHandle(handle);
}

TEST_F(AclMdlTest, aclmdlLoadWithConfig_fromMem_normal) {
  // when loadFromMem, loadFromMemwithMem, handle must set loadtype, memptr, memsize
  aclmdlConfigHandle *handle = aclmdlCreateConfigHandle();
  size_t type = ACL_MDL_LOAD_FROM_MEM_WITH_MEM;
  aclmdlSetConfigOpt(handle, ACL_MDL_LOAD_TYPE_SIZET, &type, sizeof(type));
  void *p = (void *)0x0002;
  aclmdlSetConfigOpt(handle, ACL_MDL_MEM_ADDR_PTR, &p, sizeof(p));
  size_t modelSize = 1;
  aclmdlSetConfigOpt(handle, ACL_MDL_MEM_SIZET, &modelSize, sizeof(modelSize));

  uint32_t modelId;
  EXPECT_CALL(GeExecutorStubMock::GetInstance(), GeLoadModelFromData(_,_)).Times(1).WillOnce(Return(SUCCESS));
  aclError ret = aclmdlLoadWithConfig(handle, &modelId);
  EXPECT_EQ(ret, ACL_SUCCESS);
  aclmdlDestroyConfigHandle(handle);
}

TEST_F(AclMdlTest, aclmdlLoadWithConfig_fromFile_handleErr) {
  aclmdlConfigHandle *handle = aclmdlCreateConfigHandle();
  size_t type = ACL_MDL_LOAD_FROM_FILE;
  aclError ret = aclmdlSetConfigOpt(handle, ACL_MDL_LOAD_TYPE_SIZET, &type, sizeof(type));
  EXPECT_EQ(ret, ACL_SUCCESS);
  uint32_t modelId;
  // 1. model path is not set in handle
  ret = aclmdlLoadWithConfig(handle, &modelId);
  EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
  aclmdlDestroyConfigHandle(handle);
}

TEST_F(AclMdlTest, aclmdlLoadWithConfig_fromFile_abnormal_LoadDataFromFilelErr) {
  // when loadFromFile, loadFromFilewithMem, handle must set loadtype, mdlPath
  aclmdlConfigHandle *handle = aclmdlCreateConfigHandle();
  size_t type = ACL_MDL_LOAD_FROM_FILE;
  aclmdlSetConfigOpt(handle, ACL_MDL_LOAD_TYPE_SIZET, &type, sizeof(type));
  const char *path = "/home";
  aclmdlSetConfigOpt(handle, ACL_MDL_PATH_PTR, &path, sizeof(path));
  uint32_t modelId;
  EXPECT_CALL(MockFunctionTest::aclStubInstance(), LoadDataFromFile(_,_)).Times(1).WillOnce(Return(ACL_ERROR_GE_PARAM_INVALID));
  aclError ret = aclmdlLoadWithConfig(handle, &modelId);
  EXPECT_EQ(ret, ACL_ERROR_GE_PARAM_INVALID);
  aclmdlDestroyConfigHandle(handle);
}

TEST_F(AclMdlTest, aclmdlLoadWithConfig_fromFile_abnormal_GeLoadModelFromDataErr) {
  // when loadFromFile, loadFromFilewithMem, handle must set loadtype, mdlPath
  aclmdlConfigHandle *handle = aclmdlCreateConfigHandle();
  size_t type = ACL_MDL_LOAD_FROM_FILE;
  aclmdlSetConfigOpt(handle, ACL_MDL_LOAD_TYPE_SIZET, &type, sizeof(type));
  const char *path = "/home";
  aclmdlSetConfigOpt(handle, ACL_MDL_PATH_PTR, &path, sizeof(path));
  uint32_t modelId;
  EXPECT_CALL(MockFunctionTest::aclStubInstance(), LoadDataFromFile(_,_)).Times(1).WillOnce(Return(SUCCESS));
  EXPECT_CALL(GeExecutorStubMock::GetInstance(), GeLoadModelFromData(_,_)).Times(1).WillOnce(Return(ACL_ERROR_GE_PARAM_INVALID));
  aclError ret = aclmdlLoadWithConfig(handle, &modelId);
  EXPECT_EQ(ret, ACL_ERROR_GE_PARAM_INVALID);
  aclmdlDestroyConfigHandle(handle);
}


TEST_F(AclMdlTest, aclmdlLoadWithConfig_fromFile_normal) {
  // when loadFromFile, loadFromFilewithMem, handle must set loadtype, mdlPath
  aclmdlConfigHandle *handle = aclmdlCreateConfigHandle();
  size_t type = ACL_MDL_LOAD_FROM_FILE;
  aclmdlSetConfigOpt(handle, ACL_MDL_LOAD_TYPE_SIZET, &type, sizeof(type));
  const char *path = "/home";
  aclmdlSetConfigOpt(handle, ACL_MDL_PATH_PTR, &path, sizeof(path));
  uint32_t modelId;
  EXPECT_CALL(MockFunctionTest::aclStubInstance(), LoadDataFromFile(_,_)).Times(1).WillOnce(Return(SUCCESS));
  EXPECT_CALL(GeExecutorStubMock::GetInstance(), GeLoadModelFromData(_,_)).Times(1).WillOnce(Return(SUCCESS));
  aclError ret = aclmdlLoadWithConfig(handle, &modelId);
  EXPECT_EQ(ret, ACL_SUCCESS);
  aclmdlDestroyConfigHandle(handle);
}

TEST_F(AclMdlTest, aclmdlUnload) {
  EXPECT_CALL(GeExecutorStubMock::GetInstance(), UnloadModel(_)).Times(1).WillOnce(Return(SUCCESS));
  aclError ret = aclmdlUnload(0);
  EXPECT_EQ(ret, ACL_SUCCESS);

  EXPECT_CALL(GeExecutorStubMock::GetInstance(), UnloadModel(_)).Times(1).WillOnce(Return(ACL_ERROR_GE_PARAM_INVALID));
  ret = aclmdlUnload(0);
  EXPECT_NE(ret, ACL_SUCCESS);
}

TEST_F(AclMdlTest, DataBuffer) {
  aclmdlDataset *dataset = aclmdlCreateDataset();
  EXPECT_NE(dataset, nullptr);

  DataSet *geDataSet = (DataSet *)dataset;
  geDataSet->io_addr_host = (uint64_t*)mmMalloc(sizeof(uint64_t));
  EXPECT_NE(geDataSet->io_addr_host, nullptr);
  aclError ret = aclmdlAddDatasetBuffer(dataset, nullptr);
  EXPECT_NE(ret, ACL_SUCCESS);

  ret = aclmdlAddDatasetBuffer(dataset, (aclDataBuffer *)0x01);
  EXPECT_EQ(ret, ACL_SUCCESS);

  size_t size = aclmdlGetDatasetNumBuffers(nullptr);
  EXPECT_EQ(size, 0);

  size = aclmdlGetDatasetNumBuffers(dataset);
  EXPECT_EQ(size, 1);

  aclDataBuffer *dataBuffer = aclmdlGetDatasetBuffer(dataset, 1);
  EXPECT_EQ(dataBuffer, nullptr);

  dataBuffer = aclmdlGetDatasetBuffer(dataset, 0);
  EXPECT_NE(dataBuffer, nullptr);

  ret = aclmdlAddDatasetBuffer(nullptr, nullptr);
  EXPECT_NE(ret, ACL_SUCCESS);

  ret = aclmdlDestroyDataset(nullptr);
  EXPECT_NE(ret, ACL_SUCCESS);

  ret = aclmdlDestroyDataset(dataset);
  EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(AclMdlTest, aclCreateDataBuffer_mmMallocErr) {
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmMalloc(_)).WillOnce(Invoke(mmMalloc_Abnormal_Invoke));
  aclDataBuffer *buffer = aclCreateDataBuffer((void*)0x1, 1);
  EXPECT_EQ(buffer, nullptr);
}

TEST_F(AclMdlTest, aclGetDataBufferSizeV2_Test) {
  aclDataBuffer *buffer = aclCreateDataBuffer((void*)0x1, 1);
  EXPECT_EQ(aclGetDataBufferSizeV2(nullptr), 0);
  EXPECT_EQ(aclGetDataBufferSizeV2(buffer), 1);
  aclDestroyDataBuffer(buffer);
}

TEST_F(AclMdlTest, aclDestroyDataBuffer_abnormal) {
  aclDataBuffer *buffer = NULL;
  EXPECT_EQ(aclDestroyDataBuffer(buffer), ACL_ERROR_INVALID_PARAM);
}

TEST_F(AclMdlTest, aclGetDataBufferAddr) {
  aclDataBuffer *dataBuffer = nullptr;
  EXPECT_EQ(aclGetDataBufferAddr(dataBuffer), nullptr);
  dataBuffer = aclCreateDataBuffer((void *)0x1, 1);
  EXPECT_NE(aclGetDataBufferAddr(dataBuffer), nullptr);
  aclDestroyDataBuffer(dataBuffer);
}

TEST_F(AclMdlTest, aclmdlExecConfigAttr)
{
    aclmdlExecConfigAttr configExecAttr;
    configExecAttr = (aclmdlExecConfigAttr)2;
    EXPECT_EQ(configExecAttr, ACL_MDL_WORK_ADDR_PTR);

    configExecAttr = (aclmdlExecConfigAttr)3;
    EXPECT_EQ(configExecAttr, ACL_MDL_WORK_SIZET);

    configExecAttr = (aclmdlExecConfigAttr)4;
    EXPECT_EQ(configExecAttr, ACL_MDL_MPAIMID_SIZET);

    configExecAttr = (aclmdlExecConfigAttr)5;
    EXPECT_EQ(configExecAttr, ACL_MDL_AICQOS_SIZET);

    configExecAttr = (aclmdlExecConfigAttr)6;
    EXPECT_EQ(configExecAttr, ACL_MDL_AICOST_SIZET);

    configExecAttr = (aclmdlExecConfigAttr)7;
    EXPECT_EQ(configExecAttr, ACL_MDL_MEC_TIMETHR_SIZET);
}