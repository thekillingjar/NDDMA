/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <securec.h>

#include <iostream>
#include <string>
#include <vector>


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

using namespace std;
using namespace testing;

class AclMdlDescTest : public testing::Test {
 protected:
  void SetUp(){
    ON_CALL(MmpaStubMock::GetInstance(), mmMalloc(_))
        .WillByDefault(Invoke(mmMalloc_Normal_Invoke));
  }

  void TearDown() {
    Mock::VerifyAndClearExpectations(&MmpaStubMock::GetInstance());
    Mock::VerifyAndClearExpectations(&MockFunctionTest::aclStubInstance());
  }
};

static void ModelInfoDestroyStub(void *base) {
  ModelInOutTensorDesc *config = (ModelInOutTensorDesc *)base;
  if (config->name != NULL) {
    free(config->name);
    config->name = NULL;
  }
  DeInitVector(&config->dims);
}

static void ModelTensorDescDestroyStub(void *base) {
  aclmdlTensorDesc *config = (aclmdlTensorDesc *)base;
  if (config->name != NULL) {
    free(config->name);
    config->name = NULL;
  }
  DeInitVector(&config->dims);
}

Status GetModelDescInfo_Invoke(uint32_t modelId, ModelInOutInfo *info) {
  (void)modelId;

  InitVector(&info->input_desc, sizeof(ModelInOutTensorDesc));
  SetVectorDestroyItem(&info->input_desc, ModelInfoDestroyStub);
  ModelInOutTensorDesc tDesc1;
  const char *name1 = "1";
  tDesc1.name = (char *)malloc(strlen(name1) + 1);
  strcpy_s(tDesc1.name, strlen(name1) + 1, name1);
  tDesc1.size = 2;
  tDesc1.format = FORMAT_NHWC;
  tDesc1.dataType = DT_FLOAT;
  InitVector(&(tDesc1.dims), sizeof(int64_t));
  int64_t dim1 = 1;
  int64_t dim2 = 2;
  EmplaceBackVector(&(tDesc1.dims), &dim1);
  EmplaceBackVector(&(tDesc1.dims), &dim2);
  EmplaceBackVector(&info->input_desc, &tDesc1);

  InitVector(&info->output_desc, sizeof(ModelInOutTensorDesc));
  SetVectorDestroyItem(&info->output_desc, ModelInfoDestroyStub);
  ModelInOutTensorDesc tDesc2;
  const char *name2 = "2";
  tDesc2.name = (char *)malloc(strlen(name2) + 1);
  strcpy_s(tDesc2.name, strlen(name2) + 1, name2);
  tDesc2.size = 2;
  tDesc2.format = FORMAT_NHWC;
  tDesc2.dataType = DT_FLOAT;
  InitVector(&(tDesc2.dims), sizeof(int64_t));
  int64_t dim3 = 3;
  int64_t dim4 = 4;
  EmplaceBackVector(&(tDesc2.dims), &dim3);
  EmplaceBackVector(&(tDesc2.dims), &dim4);
  EmplaceBackVector(&info->output_desc, &tDesc2);

  ModelInOutTensorDesc tDesc3;
  const char *name3 = "3";
  InitVector(&(tDesc3.dims), sizeof(int64_t));
  int64_t dim5 = 3;
  tDesc3.name = (char *)malloc(strlen(name3) + 1);
  strcpy_s(tDesc3.name, strlen(name3) + 1, name3);
  tDesc3.size = 1;
  EmplaceBackVector(&(tDesc3.dims), &dim5);
  EmplaceBackVector(&info->input_desc, &tDesc3);

  //dims over than ACL_MAX_DIM_CNT
  ModelInOutTensorDesc tDesc4;
  const char *name = "33";
  tDesc4.name = (char *)malloc(strlen(name) + 1);
  strcpy_s(tDesc4.name, strlen(name) + 1, name);
  tDesc4.size = 2;
  tDesc4.format = FORMAT_NHWC;
  tDesc4.dataType = DT_FLOAT;
  InitVector(&(tDesc4.dims), sizeof(int64_t));
  int64_t dim = 1;
  for (int64_t i = 0; i < 1000; i++) {
    dim += 1;
    EmplaceBackVector(&(tDesc4.dims), &dim);
  }
  EmplaceBackVector(&info->input_desc, &tDesc4);

  return SUCCESS;
}

Status GetModelDescInfo_Fail_Invoke(uint32_t modelId, ModelInOutInfo *info) {
  (void)modelId;
  (void)info;
  return ACL_ERROR_GE_FAILURE;
}

Status LoadDataFromFile_Fail_Invoke(const char *modelPath, ModelData *data) {
  (void)modelPath;
  (void)data;
  return ACL_ERROR_GE_FAILURE;
}

Status GetModelDescInfoFromMem_Invoke(const ModelData *modelData,
                                      ModelInOutInfo *info) {
  (void)modelData;
  InitVector(&info->input_desc, sizeof(ModelInOutTensorDesc));
  SetVectorDestroyItem(&info->input_desc, ModelInfoDestroyStub);
  ModelInOutTensorDesc desc1;
  const char *name1 = "1";
  desc1.name = (char *)malloc(strlen(name1) + 1);
  strcpy_s(desc1.name, strlen(name1) + 1, name1);
  desc1.size = 2;
  desc1.format = FORMAT_CHWN;
  desc1.dataType = DT_FLOAT;
  InitVector(&(desc1.dims), sizeof(int64_t));
  int64_t dim1 = 1;
  int64_t dim2 = 2;
  EmplaceBackVector(&(desc1.dims), &dim1);
  EmplaceBackVector(&(desc1.dims), &dim2);
  EmplaceBackVector(&info->input_desc, &desc1);

  ModelInOutTensorDesc desc2;
  const char *name2 = "2";
  desc2.name = (char *)malloc(strlen(name2) + 1);
  strcpy_s(desc2.name, strlen(name2) + 1, name2);
  desc2.size = 1;
  desc2.format = FORMAT_CHWN;
  desc2.dataType = DT_FLOAT;
  InitVector(&(desc2.dims), sizeof(int64_t));
  int64_t dim = 1;
  EmplaceBackVector(&(desc2.dims), &dim);
  EmplaceBackVector(&info->input_desc, &desc2);

  InitVector(&info->output_desc, sizeof(ModelInOutTensorDesc));
  SetVectorDestroyItem(&info->output_desc, ModelInfoDestroyStub);
  ModelInOutTensorDesc desc3;
  const char *name3 = "3";
  desc3.name = (char *)malloc(strlen(name3) + 1);
  strcpy_s(desc3.name, strlen(name3) + 1, name3);
  desc3.size = 1;
  desc3.format = FORMAT_NHWC;
  desc3.dataType = DT_FLOAT;
  InitVector(&(desc3.dims), sizeof(int64_t));
  int64_t dim3 = 3;
  EmplaceBackVector(&(desc3.dims), &dim3);
  EmplaceBackVector(&info->output_desc, &desc3);
  return SUCCESS;
}

Status GetModelDescInfoFromMem_Fail_Invoke(const ModelData *modelData,
                                           ModelInOutInfo *info) {
  (void)modelData;
  (void)info;
  return ACL_ERROR_GE_FAILURE;
}

TEST_F(AclMdlDescTest, aclmdlCreateDesc_mmMallocErr) {
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmMalloc(_)).WillOnce(Invoke(mmMalloc_Abnormal_Invoke));
  aclmdlDesc *desc = aclmdlCreateDesc();
  EXPECT_EQ(desc, nullptr);
}

TEST_F(AclMdlDescTest, TestAclMdlDesc_Success) {
  aclmdlDesc *desc = aclmdlCreateDesc();
  EXPECT_NE(desc, nullptr);
  aclError ret = aclmdlDestroyDesc(desc);
  EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(AclMdlDescTest, TestAclMdlDesc_Fail) {
  aclError ret = aclmdlDestroyDesc(nullptr);
  EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
}

TEST_F(AclMdlDescTest, Test_aclmdlGetDesc_Sussess) {
  aclmdlDesc *desc = aclmdlCreateDesc();
  EXPECT_NE(desc, nullptr);

  EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetModelDescInfo(_, _))
      .WillOnce(Invoke(GetModelDescInfo_Invoke));
  aclError ret = aclmdlGetDesc(desc, 1);
  aclmdlTensorDesc *tensorDesc1 =
      (aclmdlTensorDesc *)VectorAt(&desc->inputDesc, 0);
  aclmdlTensorDesc *tensorDesc2 =
      (aclmdlTensorDesc *)VectorAt(&desc->outputDesc, 0);
  const char *name1 = "1";
  const char *name2 = "2";
  EXPECT_EQ(*name1, *(tensorDesc1->name));
  EXPECT_EQ(*name2, *(tensorDesc2->name));
  EXPECT_EQ(2, tensorDesc1->size);
  EXPECT_EQ(ACL_FORMAT_NHWC, tensorDesc1->format);
  EXPECT_EQ(ACL_FLOAT, tensorDesc1->dataType);
  EXPECT_EQ(1, *((int64_t *)VectorAt(&tensorDesc1->dims, 0)));
  EXPECT_EQ(ret, ACL_SUCCESS);
  aclError acl_ret = aclmdlDestroyDesc(desc);
  EXPECT_EQ(acl_ret, ACL_SUCCESS);
}

TEST_F(AclMdlDescTest, Test_aclmdlGetDesc_Fail) {
  aclError ret = aclmdlGetDesc(nullptr, 1);
  EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
}

TEST_F(AclMdlDescTest, Test_aclmdlGetDesc_GE_GetModelDescInfo_Fail) {
  aclmdlDesc *desc = aclmdlCreateDesc();
  EXPECT_NE(desc, nullptr);

  EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetModelDescInfo(_, _))
      .WillOnce(Invoke(GetModelDescInfo_Fail_Invoke));
  aclError ret = aclmdlGetDesc(desc, 1);
  EXPECT_EQ(ret, ACL_ERROR_GE_FAILURE);
  aclError acl_ret = aclmdlDestroyDesc(desc);
  EXPECT_EQ(acl_ret, ACL_SUCCESS);
}

TEST_F(AclMdlDescTest, Test_aclmdlGetDescFromFile_Sussess) {
  aclmdlDesc *desc = aclmdlCreateDesc();
  EXPECT_NE(desc, nullptr);
  const char *path = "/home";

  EXPECT_CALL(MockFunctionTest::aclStubInstance(), LoadDataFromFile(_, _))
      .WillOnce(Return(SUCCESS));
  EXPECT_CALL(MockFunctionTest::aclStubInstance(),
              GetModelDescInfoFromMem(_, _))
      .WillOnce(Invoke(GetModelDescInfoFromMem_Invoke));
  aclError ret = aclmdlGetDescFromFile(desc, path);
  aclmdlTensorDesc *tensorDesc1 =
      (aclmdlTensorDesc *)VectorAt(&desc->inputDesc, 0);
  const char *name1 = "1";
  EXPECT_EQ(*name1, *(tensorDesc1->name));
  EXPECT_EQ(2, tensorDesc1->size);
  EXPECT_EQ(FORMAT_CHWN, tensorDesc1->format);
  EXPECT_EQ(DT_FLOAT, tensorDesc1->dataType);
  EXPECT_EQ(1, *((int64_t *)VectorAt(&tensorDesc1->dims, 0)));
  EXPECT_EQ(ret, ACL_SUCCESS);
  aclError acl_ret = aclmdlDestroyDesc(desc);
  EXPECT_EQ(acl_ret, ACL_SUCCESS);
}

TEST_F(AclMdlDescTest, Test_aclmdlGetDescFromFile_Fail) {
  const char *path = "/home";
  aclError ret = aclmdlGetDescFromFile(nullptr, path);
  EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
}

TEST_F(AclMdlDescTest, Test_aclmdlGetDescFromMem_Sussess) {
  aclmdlDesc *desc = aclmdlCreateDesc();
  EXPECT_NE(desc, nullptr);
  const int modelValue = 1;
  const int *model = &modelValue;

  EXPECT_CALL(MockFunctionTest::aclStubInstance(),
              GetModelDescInfoFromMem(_, _))
      .WillOnce(Invoke(GetModelDescInfoFromMem_Invoke));
  aclError ret = aclmdlGetDescFromMem(desc, &model, 1024);
  aclmdlTensorDesc *tensorDesc1 =
      (aclmdlTensorDesc *)VectorAt(&desc->inputDesc, 0);
  const char *name1 = "1";
  EXPECT_EQ(*name1, *(tensorDesc1->name));
  EXPECT_EQ(2, tensorDesc1->size);
  EXPECT_EQ(FORMAT_CHWN, tensorDesc1->format);
  EXPECT_EQ(DT_FLOAT, tensorDesc1->dataType);
  EXPECT_EQ(1, *((int64_t *)VectorAt(&tensorDesc1->dims, 0)));
  EXPECT_EQ(ret, ACL_SUCCESS);
  aclError acl_ret = aclmdlDestroyDesc(desc);
  EXPECT_EQ(acl_ret, ACL_SUCCESS);
}

TEST_F(AclMdlDescTest, Test_aclmdlGetDescFromMem_Fail) {
  const int modelValue = 1;
  const int *model = &modelValue;
  aclError ret = aclmdlGetDescFromMem(nullptr, &model, 1024);
  EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
}

TEST_F(AclMdlDescTest, Test_GE_LoadDataFromFile_Fail) {
  aclmdlDesc *desc = aclmdlCreateDesc();
  EXPECT_NE(desc, nullptr);
  const char *path = "/home";

  EXPECT_CALL(MockFunctionTest::aclStubInstance(), LoadDataFromFile(_, _))
      .WillOnce(Invoke(LoadDataFromFile_Fail_Invoke));
  aclError ret = aclmdlGetDescFromFile(desc, path);
  EXPECT_EQ(ret, ACL_ERROR_GE_FAILURE);
  aclError acl_ret = aclmdlDestroyDesc(desc);
  EXPECT_EQ(acl_ret, ACL_SUCCESS);
}

TEST_F(AclMdlDescTest, Test_GE_GetModelDescInfoFromMem_Fail) {
  aclmdlDesc *desc = aclmdlCreateDesc();
  EXPECT_NE(desc, nullptr);
  const char *path = "/home";

  EXPECT_CALL(MockFunctionTest::aclStubInstance(), LoadDataFromFile(_, _))
      .WillOnce(Return(SUCCESS));
  EXPECT_CALL(MockFunctionTest::aclStubInstance(),
              GetModelDescInfoFromMem(_, _))
      .WillOnce(Invoke(GetModelDescInfoFromMem_Fail_Invoke));
  aclError file_ret = aclmdlGetDescFromFile(desc, path);
  EXPECT_EQ(file_ret, ACL_ERROR_GE_FAILURE);
  aclError acl_file_ret = aclmdlDestroyDesc(desc);
  EXPECT_EQ(acl_file_ret, ACL_SUCCESS);

  aclmdlDesc *desc1 = aclmdlCreateDesc();
  EXPECT_NE(desc1, nullptr);
  const int modelValue = 1;
  const int *model = &modelValue;
  EXPECT_CALL(MockFunctionTest::aclStubInstance(),
              GetModelDescInfoFromMem(_, _))
      .WillOnce(Invoke(GetModelDescInfoFromMem_Fail_Invoke));
  aclError mem_ret = aclmdlGetDescFromMem(desc1, &model, 1024);
  EXPECT_EQ(mem_ret, ACL_ERROR_GE_FAILURE);
  aclError acl_mem_ret = aclmdlDestroyDesc(desc1);
  EXPECT_EQ(acl_mem_ret, ACL_SUCCESS);
}

TEST_F(AclMdlDescTest, Test_aclmdlGetNumInputs_Success) {
  aclmdlDesc *desc = aclmdlCreateDesc();
  EXPECT_NE(desc, nullptr);
  size_t nums = aclmdlGetNumInputs(desc);
  EXPECT_NE(nums, ACL_ERROR_INVALID_PARAM);
  aclError ret = aclmdlDestroyDesc(desc);
  EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(AclMdlDescTest, Test_aclmdlGetNumInputs_Fail) {
  size_t ret = aclmdlGetNumInputs(nullptr);
  EXPECT_EQ(ret, 0);
}

TEST_F(AclMdlDescTest, Test_aclmdlGetNumOutputs_Success) {
  aclmdlDesc *desc = aclmdlCreateDesc();
  EXPECT_NE(desc, nullptr);
  size_t nums = aclmdlGetNumOutputs(desc);
  EXPECT_NE(nums, ACL_ERROR_INVALID_PARAM);
  aclError ret = aclmdlDestroyDesc(desc);
  EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(AclMdlDescTest, Test_aclmdlGetNumOutputs_Fail) {
  size_t ret = aclmdlGetNumOutputs(nullptr);
  EXPECT_EQ(ret, 0);
}

TEST_F(AclMdlDescTest, Test_aclmdlGetInputSizeByIndex_Success) {
  aclmdlDesc *desc = aclmdlCreateDesc();
  EXPECT_NE(desc, nullptr);
  InitVector(&desc->inputDesc, sizeof(aclmdlTensorDesc));
  SetVectorDestroyItem(&desc->inputDesc, ModelTensorDescDestroyStub);
  aclmdlTensorDesc tDesc;
  const char *name1 = "1";
  tDesc.name = (char *)malloc(strlen(name1) + 1);
  strcpy_s(tDesc.name, strlen(name1) + 1, name1);
  tDesc.size = 2;
  tDesc.format = ACL_FORMAT_NHWC;
  tDesc.dataType = ACL_FLOAT;
  InitVector(&(tDesc.dims), sizeof(int64_t));
  int64_t dim1 = 1;
  int64_t dim2 = 2;
  EmplaceBackVector(&(tDesc.dims), &dim1);
  EmplaceBackVector(&(tDesc.dims), &dim2);
  EmplaceBackVector(&desc->inputDesc, &tDesc);
  size_t size = aclmdlGetInputSizeByIndex(desc, 0);
  EXPECT_EQ(size, 2);
  aclError ret = aclmdlDestroyDesc(desc);
  EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(AclMdlDescTest, Test_aclmdlGetInputSizeByIndex_Fail) {
  //1. modelDesc is NULL
  size_t ret = aclmdlGetInputSizeByIndex(nullptr, 0);
  EXPECT_EQ(ret, 0);
  //2. only one input, index is invalid
  aclmdlDesc *desc = aclmdlCreateDesc();
  EXPECT_NE(desc, nullptr);
  InitVector(&desc->inputDesc, sizeof(aclmdlTensorDesc));
  SetVectorDestroyItem(&desc->inputDesc, ModelTensorDescDestroyStub);
  aclmdlTensorDesc tDesc;
  const char *name1 = "1";
  tDesc.name = (char *)malloc(strlen(name1) + 1);
  strcpy_s(tDesc.name, strlen(name1) + 1, name1);
  tDesc.size = 2;
  tDesc.format = ACL_FORMAT_NHWC;
  tDesc.dataType = ACL_FLOAT;
  InitVector(&(tDesc.dims), sizeof(int64_t));
  int64_t dim1 = 1;
  int64_t dim2 = 2;
  EmplaceBackVector(&(tDesc.dims), &dim1);
  EmplaceBackVector(&(tDesc.dims), &dim2);
  EmplaceBackVector(&desc->inputDesc, &tDesc);
  ret = aclmdlGetInputSizeByIndex(desc, 1);
  EXPECT_EQ(ret, 0);
  ret = aclmdlDestroyDesc(desc);
  EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(AclMdlDescTest, Test_aclmdlGetOutputSizeByIndex_Success) {
  aclmdlDesc *desc = aclmdlCreateDesc();
  EXPECT_NE(desc, nullptr);
  InitVector(&desc->outputDesc, sizeof(aclmdlTensorDesc));
  SetVectorDestroyItem(&desc->outputDesc, ModelTensorDescDestroyStub);
  aclmdlTensorDesc tDesc;
  const char *name1 = "1";
  tDesc.name = (char *)malloc(strlen(name1) + 1);
  strcpy_s(tDesc.name, strlen(name1) + 1, name1);
  tDesc.size = 2;
  tDesc.format = ACL_FORMAT_NHWC;
  tDesc.dataType = ACL_FLOAT;
  InitVector(&(tDesc.dims), sizeof(int64_t));
  int64_t dim1 = 1;
  int64_t dim2 = 2;
  EmplaceBackVector(&(tDesc.dims), &dim1);
  EmplaceBackVector(&(tDesc.dims), &dim2);
  EmplaceBackVector(&desc->outputDesc, &tDesc);
  size_t size = aclmdlGetOutputSizeByIndex(desc, 0);
  EXPECT_EQ(size, 2);
  aclError ret = aclmdlDestroyDesc(desc);
  EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(AclMdlDescTest, Test_aclmdlGetOutputSizeByIndex_Fail) {
  size_t ret = aclmdlGetOutputSizeByIndex(nullptr, 0);
  EXPECT_EQ(ret, 0);
}

TEST_F(AclMdlDescTest, Test_aclmdlGetInputNameByIndex_Success) {
  aclmdlDesc *desc = aclmdlCreateDesc();
  EXPECT_NE(desc, nullptr);
  InitVector(&desc->inputDesc, sizeof(aclmdlTensorDesc));
  SetVectorDestroyItem(&desc->inputDesc, ModelTensorDescDestroyStub);
  aclmdlTensorDesc tDesc;
  const char *name1 = "1";
  tDesc.name = (char *)malloc(strlen(name1) + 1);
  strcpy_s(tDesc.name, strlen(name1) + 1, name1);
  tDesc.size = 2;
  tDesc.format = ACL_FORMAT_NHWC;
  tDesc.dataType = ACL_FLOAT;
  InitVector(&(tDesc.dims), sizeof(int64_t));
  int64_t dim1 = 1;
  int64_t dim2 = 2;
  EmplaceBackVector(&(tDesc.dims), &dim1);
  EmplaceBackVector(&(tDesc.dims), &dim2);
  EmplaceBackVector(&desc->inputDesc, &tDesc);
  string name = aclmdlGetInputNameByIndex(desc, 0);
  EXPECT_EQ(name, "1");
  aclError ret = aclmdlDestroyDesc(desc);
  EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(AclMdlDescTest, Test_aclmdlGetInputNameByIndex_Fail) {
  string name = aclmdlGetInputNameByIndex(nullptr, 0);
  EXPECT_EQ(name, "");
}

TEST_F(AclMdlDescTest, Test_aclmdlGetOutputNameByIndex_Success) {
  aclmdlDesc *desc = aclmdlCreateDesc();
  EXPECT_NE(desc, nullptr);
  InitVector(&desc->outputDesc, sizeof(aclmdlTensorDesc));
  SetVectorDestroyItem(&desc->outputDesc, ModelTensorDescDestroyStub);
  aclmdlTensorDesc tDesc;
  const char *name1 = "1";
  tDesc.name = (char *)malloc(strlen(name1) + 1);
  strcpy_s(tDesc.name, strlen(name1) + 1, name1);
  tDesc.size = 2;
  tDesc.format = ACL_FORMAT_NHWC;
  tDesc.dataType = ACL_FLOAT;
  InitVector(&(tDesc.dims), sizeof(int64_t));
  int64_t dim1 = 1;
  int64_t dim2 = 2;
  EmplaceBackVector(&(tDesc.dims), &dim1);
  EmplaceBackVector(&(tDesc.dims), &dim2);
  EmplaceBackVector(&desc->outputDesc, &tDesc);
  string name = aclmdlGetOutputNameByIndex(desc, 0);
  EXPECT_EQ(name, "1");
  aclError ret = aclmdlDestroyDesc(desc);
  EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(AclMdlDescTest, Test_aclmdlGetOutputNameByIndex_Fail) {
  string name = aclmdlGetOutputNameByIndex(nullptr, 0);
  EXPECT_EQ(name, "");
}

TEST_F(AclMdlDescTest, Test_aclmdlGetInputFormat_Success) {
  aclmdlDesc *desc = aclmdlCreateDesc();
  EXPECT_NE(desc, nullptr);
  InitVector(&desc->inputDesc, sizeof(aclmdlTensorDesc));
  SetVectorDestroyItem(&desc->inputDesc, ModelTensorDescDestroyStub);
  aclmdlTensorDesc tDesc;
  const char *name1 = "1";
  tDesc.name = (char *)malloc(strlen(name1) + 1);
  strcpy_s(tDesc.name, strlen(name1) + 1, name1);
  tDesc.size = 2;
  tDesc.format = ACL_FORMAT_NHWC;
  tDesc.dataType = ACL_FLOAT;
  InitVector(&(tDesc.dims), sizeof(int64_t));
  int64_t dim1 = 1;
  int64_t dim2 = 2;
  EmplaceBackVector(&(tDesc.dims), &dim1);
  EmplaceBackVector(&(tDesc.dims), &dim2);
  EmplaceBackVector(&desc->inputDesc, &tDesc);
  size_t format = aclmdlGetInputFormat(desc, 0);
  EXPECT_EQ(format, ACL_FORMAT_NHWC);
  aclError ret = aclmdlDestroyDesc(desc);
  EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(AclMdlDescTest, Test_aclmdlGetInputFormat_Fail) {
  size_t ret = aclmdlGetInputFormat(nullptr, 0);
  EXPECT_EQ(ret, ACL_FORMAT_UNDEFINED);
}

TEST_F(AclMdlDescTest, Test_aclmdlGetOutputFormat_Success) {
  aclmdlDesc *desc = aclmdlCreateDesc();
  EXPECT_NE(desc, nullptr);
  InitVector(&desc->outputDesc, sizeof(aclmdlTensorDesc));
  SetVectorDestroyItem(&desc->outputDesc, ModelTensorDescDestroyStub);
  aclmdlTensorDesc tDesc;
  const char *name1 = "1";
  tDesc.name = (char *)malloc(strlen(name1) + 1);
  strcpy_s(tDesc.name, strlen(name1) + 1, name1);
  tDesc.size = 2;
  tDesc.format = ACL_FORMAT_NHWC;
  tDesc.dataType = ACL_FLOAT;
  InitVector(&(tDesc.dims), sizeof(int64_t));
  int64_t dim1 = 1;
  int64_t dim2 = 2;
  EmplaceBackVector(&(tDesc.dims), &dim1);
  EmplaceBackVector(&(tDesc.dims), &dim2);
  EmplaceBackVector(&desc->outputDesc, &tDesc);
  size_t format = aclmdlGetOutputFormat(desc, 0);
  EXPECT_EQ(format, ACL_FORMAT_NHWC);
  aclError ret = aclmdlDestroyDesc(desc);
  EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(AclMdlDescTest, Test_aclmdlGetOutputFormat_Fail) {
  size_t ret = aclmdlGetOutputFormat(nullptr, 0);
  EXPECT_EQ(ret, ACL_FORMAT_UNDEFINED);
}

TEST_F(AclMdlDescTest, Test_aclmdlGetInputDataType_Success) {
  aclmdlDesc *desc = aclmdlCreateDesc();
  EXPECT_NE(desc, nullptr);
  InitVector(&desc->inputDesc, sizeof(aclmdlTensorDesc));
  SetVectorDestroyItem(&desc->inputDesc, ModelTensorDescDestroyStub);
  aclmdlTensorDesc tDesc;
  const char *name1 = "1";
  tDesc.name = (char *)malloc(strlen(name1) + 1);
  strcpy_s(tDesc.name, strlen(name1) + 1, name1);
  tDesc.size = 2;
  tDesc.format = ACL_FORMAT_NHWC;
  tDesc.dataType = ACL_FLOAT;
  InitVector(&(tDesc.dims), sizeof(int64_t));
  int64_t dim1 = 1;
  int64_t dim2 = 2;
  EmplaceBackVector(&(tDesc.dims), &dim1);
  EmplaceBackVector(&(tDesc.dims), &dim2);
  EmplaceBackVector(&desc->inputDesc, &tDesc);
  size_t dataType = aclmdlGetInputDataType(desc, 0);
  EXPECT_EQ(dataType, ACL_FLOAT);
  aclError ret = aclmdlDestroyDesc(desc);
  EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(AclMdlDescTest, Test_aclmdlGetInputDataType_Fail) {
  size_t ret = aclmdlGetInputDataType(nullptr, 0);
  EXPECT_EQ(ret, ACL_DT_UNDEFINED);
}

TEST_F(AclMdlDescTest, Test_aclmdlGetOutputDataType_Success) {
  aclmdlDesc *desc = aclmdlCreateDesc();
  EXPECT_NE(desc, nullptr);
  InitVector(&desc->outputDesc, sizeof(aclmdlTensorDesc));
  SetVectorDestroyItem(&desc->outputDesc, ModelTensorDescDestroyStub);
  aclmdlTensorDesc tDesc;
  const char *name1 = "1";
  tDesc.name = (char *)malloc(strlen(name1) + 1);
  strcpy_s(tDesc.name, strlen(name1) + 1, name1);
  tDesc.size = 2;
  tDesc.format = ACL_FORMAT_NHWC;
  tDesc.dataType = ACL_FLOAT;
  InitVector(&(tDesc.dims), sizeof(int64_t));
  int64_t dim1 = 1;
  int64_t dim2 = 2;
  EmplaceBackVector(&(tDesc.dims), &dim1);
  EmplaceBackVector(&(tDesc.dims), &dim2);
  EmplaceBackVector(&desc->outputDesc, &tDesc);
  size_t dataType = aclmdlGetOutputDataType(desc, 0);
  EXPECT_EQ(dataType, ACL_FLOAT);
  aclError ret = aclmdlDestroyDesc(desc);
  EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(AclMdlDescTest, Test_aclmdlGetOutputDataType_Fail) {
  size_t ret = aclmdlGetOutputDataType(nullptr, 0);
  EXPECT_EQ(ret, ACL_DT_UNDEFINED);
}

TEST_F(AclMdlDescTest, Test_aclmdlGetInputDims_paramNull) {
  aclmdlIODims dims;
  aclError ret = aclmdlGetInputDims(nullptr, 3, &dims);
  EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
}

TEST_F(AclMdlDescTest, Test_aclmdlGetInputDims_paramErr) {
  EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetModelDescInfo(_, _))
      .WillOnce(Invoke(GetModelDescInfo_Invoke));
  aclmdlDesc *desc = aclmdlCreateDesc();
  EXPECT_NE(desc, nullptr);

  aclError ret = aclmdlGetDesc(desc, 1);
  EXPECT_EQ(ret, ACL_SUCCESS);

  aclmdlIODims dims;
  // dimSize greater than ACL_MAX_DIM_CNT
  ret = aclmdlGetInputDims(desc, 2, &dims);
  EXPECT_EQ(ret, ACL_ERROR_STORAGE_OVER_LIMIT);

  // index is out of bounds
  ret = aclmdlGetInputDims(desc, 3, &dims);
  EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);

  ret = aclmdlDestroyDesc(desc);
  EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(AclMdlDescTest, Test_aclmdlGetInputDims_normal) {
  EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetModelDescInfo(_, _))
      .WillOnce(Invoke(GetModelDescInfo_Invoke));
  aclmdlDesc *desc = aclmdlCreateDesc();
  EXPECT_NE(desc, nullptr);

  aclError ret = aclmdlGetDesc(desc, 1);
  EXPECT_EQ(ret, ACL_SUCCESS);

  aclmdlIODims dims;
  ret = aclmdlGetInputDims(desc, 0, &dims);
  EXPECT_EQ(ret, ACL_SUCCESS);
  EXPECT_EQ(dims.dims[0], 1);
  EXPECT_EQ(dims.dims[1], 2);

  ret = aclmdlDestroyDesc(desc);
  EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(AclMdlDescTest, Test_aclmdlGetOutputDims) {
  EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetModelDescInfo(_, _))
      .WillOnce(Invoke(GetModelDescInfo_Invoke));
  aclmdlDesc *desc = aclmdlCreateDesc();
  EXPECT_NE(desc, nullptr);

  aclError ret = aclmdlGetDesc(desc, 1);
  EXPECT_EQ(ret, ACL_SUCCESS);

  aclmdlIODims dims;
  ret = aclmdlGetOutputDims(desc, 0, &dims);
  EXPECT_EQ(ret, ACL_SUCCESS);
  EXPECT_EQ(dims.dims[0], 3);
  EXPECT_EQ(dims.dims[1], 4);

  ret = aclmdlGetOutputDims(desc, 2, &dims);
  EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);

  ret = aclmdlDestroyDesc(desc);
  EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(AclMdlDescTest, Test_aclmdlGetRealTensorName_paramNull) {
  const char *str = aclmdlGetTensorRealName(nullptr, "abc");
  EXPECT_EQ(str, nullptr);
}

aclmdlDesc *ModelDescConstruct_For_GetRealTensorName() {
  aclmdlDesc *mdlDesc = aclmdlCreateDesc();  // mdlDesc include three inputs and one output.
  EXPECT_NE(mdlDesc, nullptr);
  aclmdlTensorDesc tDesc1;
  const char *name1 = "acl_modelId_0_input_2";
  tDesc1.name = (char *)malloc(strlen(name1) + 1);
  if (tDesc1.name == nullptr) {
    aclmdlDestroyDesc(mdlDesc);
    return nullptr;
  }
  strcpy_s(tDesc1.name, strlen(name1) + 1, name1);
  InitVector(&(tDesc1.dims), sizeof(int64_t));
  EmplaceBackVector(&mdlDesc->inputDesc, &tDesc1);

  aclmdlTensorDesc tDesc2;
  const char *name2 = "a6872_1bu_idc";
  tDesc2.name = (char *)malloc(strlen(name2) + 1);
  if (tDesc2.name == nullptr) {
    aclmdlDestroyDesc(mdlDesc);
    return nullptr;
  }
  strcpy_s(tDesc2.name, strlen(name2) + 1, name2);
  InitVector(&(tDesc2.dims), sizeof(int64_t));
  EmplaceBackVector(&mdlDesc->inputDesc, &tDesc2);

  aclmdlTensorDesc tDesc3;
  const char *name3 =
      "dhsdhasiodhsaiodhsiashdisdhsiahdisahdisoahisahdihdisahdaoidhaihdsaihdsai"
      "hdsahdishaodhsiahihdoiahdsioadhisahdasidhsaidashdiaoiahdisohdosahdsahdia"
      "soidashoidaoidhahdaoidahioadhiahdsahdiahdaiodaidahdhdahidahdaoda";
  tDesc3.name = (char *)malloc(strlen(name3) + 1);
  if (tDesc3.name == nullptr) {
    aclmdlDestroyDesc(mdlDesc);
    return nullptr;
  }
  strcpy_s(tDesc3.name, strlen(name3) + 1, name3);
  InitVector(&(tDesc3.dims), sizeof(int64_t));
  EmplaceBackVector(&mdlDesc->inputDesc, &tDesc3);

  aclmdlTensorDesc tDesc4;
  const char *name4 =
      "output_dhsdhasiodhsaiodhsiashdisdhsiahdisahdisoahisahdihdisahdaoidhaihdsaihdsai"
      "hdsahdishaodhsiahihdoiahdsioadhisahdasidhsaidashdiaoiahdisohdosahdsahdia"
      "soidashoidaoidhahdaoidahioadhiahdsahdiahdaiodaidahdhdahidahdaoda";
  tDesc4.name = (char *)malloc(strlen(name4) + 1);
  if (tDesc4.name == nullptr) {
    aclmdlDestroyDesc(mdlDesc);
    return nullptr;
  }
  strcpy_s(tDesc4.name, strlen(name4) + 1, name4);
  InitVector(&(tDesc4.dims), sizeof(int64_t));
  EmplaceBackVector(&mdlDesc->outputDesc, &tDesc4);
  return mdlDesc;
}

TEST_F(AclMdlDescTest, Test_aclmdlGetInputDims_normal_ConvertTensorNameLegal) {
  aclmdlDesc *mdlDesc = ModelDescConstruct_For_GetRealTensorName();
  EXPECT_NE(mdlDesc, nullptr);
  mdlDesc->modelId = 0;
  aclmdlIODims dims;
  aclError ret = aclmdlGetOutputDims(mdlDesc, 0, &dims);
  EXPECT_EQ(ret, ACL_SUCCESS);
  EXPECT_STREQ(dims.name, "acl_modelId_0_output_0");
  ret = aclmdlDestroyDesc(mdlDesc);
  EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(AclMdlDescTest, Test_aclmdlGetInputDims_normal_TransConvertTensorNameToLegal) {
  aclmdlDesc *mdlDesc = ModelDescConstruct_For_GetRealTensorName();
  EXPECT_NE(mdlDesc, nullptr);
  mdlDesc->modelId = 0;
  aclmdlIODims dims;
  // The first input tensorName is acl_modelId_0_input_2, can be found in inputdesc tensor.
  aclError ret = aclmdlGetInputDims(mdlDesc, 0, &dims);
  EXPECT_EQ(ret, ACL_SUCCESS);
  EXPECT_STREQ(dims.name, "acl_modelId_0_input_2");

  aclmdlIODims dims1;
  /*
  The third input tensorName is over than 128, use convertName=acl_modelId_0_input_2,
  but conflicts with existing name, so need to transconvert tensorName to legal.
  */
  ret = aclmdlGetInputDims(mdlDesc, 2, &dims1);
  EXPECT_EQ(ret, ACL_SUCCESS);
  EXPECT_STREQ(dims1.name, "acl_modelId_0_input_2_aaa");
  ret = aclmdlDestroyDesc(mdlDesc);
  EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(AclMdlDescTest, Test_aclmdlGetRealTensorName_GetRealTensorNamesSucc_input) {
  aclmdlDesc *mdlDesc = ModelDescConstruct_For_GetRealTensorName();
  EXPECT_NE(mdlDesc, nullptr);
  aclmdlIODims dims0;
  aclmdlIODims dims1;
  //aclmdlIODims dims2;
  mdlDesc->modelId = 0;
  aclmdlGetInputDims(mdlDesc, 0, &dims0);
  aclmdlGetInputDims(mdlDesc, 1, &dims1);
  EXPECT_STREQ(dims0.name, "acl_modelId_0_input_2");
  EXPECT_STREQ(dims1.name, "a6872_1bu_idc");
  const char *str = aclmdlGetTensorRealName(mdlDesc, dims0.name);
  EXPECT_STREQ(str, "acl_modelId_0_input_2");

  str = aclmdlGetTensorRealName(mdlDesc, dims1.name);
  EXPECT_STREQ(str, "a6872_1bu_idc");

  aclError ret = aclmdlDestroyDesc(mdlDesc);
  EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(AclMdlDescTest, Test_aclmdlGetRealTensorName_GetRealTensorNamesSucc_output) {
  aclmdlDesc *mdlDesc = ModelDescConstruct_For_GetRealTensorName();
  EXPECT_NE(mdlDesc, nullptr);

  const char *name = "output_dhsdhasiodhsaiodhsiashdisdhsiahdisahdisoahisahdihdisahdaoidhaihdsaihdsai"
  "hdsahdishaodhsiahihdoiahdsioadhisahdasidhsaidashdiaoiahdisohdosahdsahdia"
  "soidashoidaoidhahdaoidahioadhiahdsahdiahdaiodaidahdhdahidahdaoda";

  const char *str = aclmdlGetTensorRealName(mdlDesc, name);
  EXPECT_STREQ(str, name);
  aclError ret = aclmdlDestroyDesc(mdlDesc);
  EXPECT_EQ(ret, ACL_SUCCESS);
  EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(AclMdlDescTest, Test_aclmdlGetRealTensorName_abnormal_TransTensorNameToRealErr_nameInvalidFormat) {
  aclmdlDesc *mdlDesc = ModelDescConstruct_For_GetRealTensorName();
  EXPECT_NE(mdlDesc, nullptr);
  mdlDesc->modelId = 0;
  // cannot find name in tensorDesc->name, and name shoule incdlue _input_ or _output_.
  const char* str = aclmdlGetTensorRealName(mdlDesc, "xxxdwdfefesdasd");
  EXPECT_EQ(str, nullptr);

  // failed before index judgement, strlen(name) <= the strlen of acl_modelId_xx_input(output)_,
  str = aclmdlGetTensorRealName(mdlDesc, "modelId_0_input_x0x");
  EXPECT_EQ(str, nullptr);
  aclError ret = aclmdlDestroyDesc(mdlDesc);
  EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(AclMdlDescTest, Test_aclmdlGetRealTensorName_abnormal_TransTensorNameToRealErr_indexNotDigit) {
  aclmdlDesc *mdlDesc = ModelDescConstruct_For_GetRealTensorName();  // mdlDesc include three inputs and one output.
  EXPECT_NE(mdlDesc, nullptr);
  mdlDesc->modelId = 0;
  // when called TransTensorNameToReal, str after acl_modelId_0_input_ is not digit
  const char* str = aclmdlGetTensorRealName(mdlDesc, "acl_modelId_0_input_xxxx");
  EXPECT_EQ(str, nullptr);

  aclError ret = aclmdlDestroyDesc(mdlDesc);
  EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(AclMdlDescTest, Test_aclmdlGetRealTensorName_abnormal_TransTensorNameToRealErr_prefixErr) {
  aclmdlDesc *mdlDesc = ModelDescConstruct_For_GetRealTensorName();
  EXPECT_NE(mdlDesc, nullptr);
  mdlDesc->modelId = 0;
  // index is digit, but prefix acl_modelId_x_input_ is consistent with generated format(acl_modelId_x_input_).
  const char* str = aclmdlGetTensorRealName(mdlDesc, "acl_modelId_x_input_0");
  EXPECT_EQ(str, nullptr);
  aclError ret = aclmdlDestroyDesc(mdlDesc);
  EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(AclMdlDescTest, Test_aclmdlGetRealTensorName_abnormal_TransTensorNameToRealErr_indexNotExist) {
  aclmdlDesc *mdlDesc = ModelDescConstruct_For_GetRealTensorName();
  EXPECT_NE(mdlDesc, nullptr);
  mdlDesc->modelId = 0;
  //name format is right, index not found in real inputdesc or outputdesc
  const char *str = aclmdlGetTensorRealName(mdlDesc, "acl_modelId_0_output_11");
  EXPECT_EQ(str, nullptr);

  aclError ret = aclmdlDestroyDesc(mdlDesc);
  EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(AclMdlDescTest, Test_aclmdlGetRealTensorName_TransTensorNameToRealSucc) {
  aclmdlDesc *mdlDesc = ModelDescConstruct_For_GetRealTensorName();
  EXPECT_NE(mdlDesc, nullptr);
  mdlDesc->modelId = 0;
  const char *str = aclmdlGetTensorRealName(mdlDesc, "acl_modelId_0_output_0");
  EXPECT_STREQ(str, "output_dhsdhasiodhsaiodhsiashdisdhsiahdisahdisoahisahdihdisahdaoidhaihdsaihdsai"
      "hdsahdishaodhsiahihdoiahdsioadhisahdasidhsaidashdiaoiahdisohdosahdsahdia"
      "soidashoidaoidhahdaoidahioadhiahdsahdiahdaiodaidahdhdahidahdaoda");
  aclError ret = aclmdlDestroyDesc(mdlDesc);
  EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(AclMdlDescTest, Test_aclmdlGetInputIndexByName_abnormal) {
  aclmdlDesc *desc = aclmdlCreateDesc();
  EXPECT_NE(desc, nullptr);

  size_t idx = 0;
  //1. modelDesc/name/index must not be null
  aclError ret = aclmdlGetInputIndexByName(NULL, "2", &idx);
  EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);

  //2. no input named "2"
  ret = aclmdlGetInputIndexByName(desc, "2", &idx);
  EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);

  ret = aclmdlDestroyDesc(desc);
  EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(AclMdlDescTest, Test_aclmdlGetInputIndexByName_normal) {
  EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetModelDescInfo(_, _))
      .WillOnce(Invoke(GetModelDescInfo_Invoke));
  aclmdlDesc *desc = aclmdlCreateDesc();
  EXPECT_NE(desc, nullptr);

  aclError ret = aclmdlGetDesc(desc, 1);
  EXPECT_EQ(ret, ACL_SUCCESS);

  size_t idx = 0;
  ret = aclmdlGetInputIndexByName(desc, "1", &idx);
  EXPECT_EQ(idx, 0);
  EXPECT_EQ(ret, ACL_SUCCESS);

  ret = aclmdlGetInputIndexByName(desc, "3", &idx);
  EXPECT_EQ(idx, 1);
  EXPECT_EQ(ret, ACL_SUCCESS);

  ret = aclmdlDestroyDesc(desc);
  EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(AclMdlDescTest, Test_aclmdlGetOutputIndexByName) {
  EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetModelDescInfo(_, _))
      .WillOnce(Invoke(GetModelDescInfo_Invoke));
  aclmdlDesc *desc = aclmdlCreateDesc();
  EXPECT_NE(desc, nullptr);

  aclError ret = aclmdlGetDesc(desc, 1);
  EXPECT_EQ(ret, ACL_SUCCESS);

  size_t idx = 0;
  ret = aclmdlGetOutputIndexByName(desc, "2", &idx);
  EXPECT_EQ(idx, 0);
  EXPECT_EQ(ret, ACL_SUCCESS);

  ret = aclmdlGetOutputIndexByName(desc, "3", &idx);
  EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);

  ret = aclmdlDestroyDesc(desc);
  EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(AclMdlDescTest, Test_aclmdlGetInputDims_NumToStr) {
  aclmdlDesc *mdlDesc = aclmdlCreateDesc();

  aclmdlTensorDesc tDesc0;
  const char *name0 =
      "dhsdhasiodhsaiodhsiashdisdhsiahdisahdisoahisahdihdisahdaoidhaihdsaihdsai"
      "hdsahdishaodhsiahihdoiahdsioadhisahdasidhsaidashdiaoiahdisohdosahdsahdia"
      "soidashoidaoidhahdaoidahioadhiahdsahdiahdaiodaidahdhdahidahdaoda";
  tDesc0.name = (char *)malloc(strlen(name0) + 1);
  strcpy_s(tDesc0.name, strlen(name0) + 1, name0);
  InitVector(&(tDesc0.dims), sizeof(int64_t));
  EmplaceBackVector(&mdlDesc->inputDesc, &tDesc0);

  aclmdlIODims dims0;
  mdlDesc->modelId = 10;
  aclError ret = aclmdlGetInputDims(mdlDesc, 0, &dims0);
  EXPECT_EQ(ret, ACL_SUCCESS);
  EXPECT_STREQ(dims0.name, "acl_modelId_10_input_0");

  ret = aclmdlDestroyDesc(mdlDesc);
  EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(AclMdlDescTest, Test_aclmdlGetRealTensorName_strncmp_valid_and_invalid_name) {
  aclmdlDesc *mdlDesc = aclmdlCreateDesc();

  aclmdlTensorDesc tDesc1;
  const char *name1 =
      "dhsdhasiodhsaiodhsiashdisdhsiahdisahdisoahisahdihdisahdaoidhaihdsaihdsai"
      "hdsahdishaodhsiahihdoiahdsioadhisahdasidhsaidashdiaoiahdisohdosahdsahdia"
      "soidashoidaoidhahdaoidahioadhiahdsahdiahdaiodaidahdhdahidahdaoda";
  tDesc1.name = (char *)malloc(strlen(name1) + 1);
  strcpy_s(tDesc1.name, strlen(name1) + 1, name1);
  InitVector(&(tDesc1.dims), sizeof(int64_t));
  EmplaceBackVector(&mdlDesc->inputDesc, &tDesc1);

  aclmdlIODims dims0;
  mdlDesc->modelId = 0;
  aclError ret = aclmdlGetInputDims(mdlDesc, 0, &dims0);
  EXPECT_EQ(ret, ACL_SUCCESS);
  EXPECT_STREQ(dims0.name, "acl_modelId_0_input_0");

  const char *str = aclmdlGetTensorRealName(mdlDesc, dims0.name);
  EXPECT_STREQ(name1, str);

  str = aclmdlGetTensorRealName(mdlDesc, "modelId_0_input_x0x_1");
  EXPECT_EQ(str, nullptr);

  ret = aclmdlDestroyDesc(mdlDesc);
  EXPECT_EQ(ret, ACL_SUCCESS);
}