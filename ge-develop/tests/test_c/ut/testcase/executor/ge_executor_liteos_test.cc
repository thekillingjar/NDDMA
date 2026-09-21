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
#include <memory>
#include <fstream>
#include <string>
#include <cstdio>
#include "ge_executor.h"
#include "exe_model_builder.h"
#include "types.h"
#include "model_desc.h"
#include "model_loader.h"
#include "maintain_manager.h"
#include "ge/ge_error_codes.h"
#include "gmock_stub/ascend_rt_stub.h"
#include "vector.h"
#include "mmpa_api.h"
#include <securec.h>
using namespace testing;
using ::testing::Return;
using ::testing::Eq;
using ::testing::NotNull;
using ::testing::Invoke;
using ::testing::_;
using namespace ge;

class UtestGEExecutorLiteOSTest : public testing::Test {
 protected:
  void SetUp() {
    ON_CALL(RtStubMock::GetInstance(), rtMalloc(_,_,_,_)).WillByDefault(Invoke(rtMalloc_Normal_Invoke));
    ON_CALL(RtStubMock::GetInstance(), rtNanoModelLoad(_,_)).WillByDefault(Return(0));
    ON_CALL(RtStubMock::GetInstance(), rtNanoModelDestroy(_)).WillByDefault(Return(0));
    ON_CALL(RtStubMock::GetInstance(), rtStreamGetSqid(_,_)).WillByDefault(Return(0));
    ON_CALL(RtStubMock::GetInstance(), rtNanoModelExecute(_)).WillByDefault(Return(0));
    ON_CALL(MmpaStubMock::GetInstance(), mmTellFile(_)).WillByDefault(Invoke(mmTellFile_Normal_Invoke));
    ON_CALL(MmpaStubMock::GetInstance(), mmReadFile(_,_,_,_)).WillByDefault(Invoke(mmReadFile_Normal_Invoke));
    ON_CALL(MmpaStubMock::GetInstance(), mmMalloc(_)).WillByDefault(Invoke(mmMalloc_Normal_Invoke));
  }
  void TearDown() {
    Mock::VerifyAndClearExpectations(&MmpaStubMock::GetInstance());
    Mock::VerifyAndClearExpectations(&RtStubMock::GetInstance());
  }
};

TEST_F(UtestGEExecutorLiteOSTest, GeDbgStaticInit) {
  EXPECT_CALL(RtStubMock::GetInstance(), rtMemcpy(_,_,_,_,_)).WillRepeatedly(Invoke(rtMemcpy_Normal_Invoke));
  EXPECT_CALL(RtStubMock::GetInstance(), rtMalloc(_, _, _, _)).WillRepeatedly(Invoke(rtMalloc_Normal_Invoke));
  const char *configPath = nullptr;
  Status ret = GeDbgInit(configPath);
  ASSERT_EQ(ret, SUCCESS);
  ret = GeDbgDeInit();
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestGEExecutorLiteOSTest, GeDbgStaticProfiling) {
  uint32_t chipId = 0;
  uint32_t deviceId = 0;
  Status ret = GeNofifySetDevice(chipId, deviceId);
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestGEExecutorLiteOSTest, GeDbgStaticProfEnable) {
  bool ret = GetProfEnable();
  ASSERT_EQ(ret, false);
}

TEST_F(UtestGEExecutorLiteOSTest, GeDbgStaticPreProcess) {
  rtMdlLoad_t *mdlLoad = nullptr;
  size_t taskDescSize = 0;
  char *modelName = nullptr;
  void *dbgHandle = nullptr;
  Status ret = LoadModelPreProcess(mdlLoad, taskDescSize, modelName, dbgHandle);
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestGEExecutorLiteOSTest, GeDbgStaticPostProcess) {
  uint32_t modelId = 0;
  char *modelName = 0;
  uint64_t *stepIdAddr = nullptr;
  void *dbgHandle = nullptr;
  Status ret = LoadModelPostProcess(modelId, modelName, stepIdAddr, dbgHandle);
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestGEExecutorLiteOSTest, GeDbgStaticStepIdCounterPlus) {
  void *dbgHandle = nullptr;
  StepIdConuterPlus(dbgHandle);
  EXPECT_EQ(dbgHandle, nullptr);
}

TEST_F(UtestGEExecutorLiteOSTest, GeDbgStaticFreeLoadDumpInfo) {
  void *dbgHandle = nullptr;
  DataFreeLoadDumpInfo(dbgHandle);
  EXPECT_EQ(dbgHandle, nullptr);
}

TEST_F(UtestGEExecutorLiteOSTest, GeDbgStaticDbgHandle) {
  void *ptr = CreatModelDbgHandle();
  ASSERT_EQ(ptr, nullptr);
}

TEST_F(UtestGEExecutorLiteOSTest, LoadOfflineModelFromFile_Abnormal) {
  std::string test_smap = "/tmp/" + std::to_string(getpid()) + "_maps";
  std::string self_smap = "/proc/" + std::to_string(getpid()) + "/maps";
  std::string copy_smap = "cp -f " + self_smap + " " + test_smap;
  EXPECT_EQ(system(copy_smap.c_str()), 0);
  ModelData modelData;
  (void)memset_s(&modelData, sizeof(ModelData), 0, sizeof(ModelData));
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmTellFile(_)).WillRepeatedly(Invoke(mmTellFile_Normal_Invoke));
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmMalloc(_)).WillRepeatedly(Invoke(mmMalloc_Normal_Invoke));
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmReadFile(_,_,_,_)).WillRepeatedly(Invoke(mmReadFile_Normal_Invoke));
  EXPECT_EQ(LoadOfflineModelFromFile(test_smap.c_str(), &modelData), ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID);
  EXPECT_EQ(modelData.modelData, nullptr);
}

static void GenerateEleForModel(uint8_t a, int size, std::vector<uint8_t> &model)
{
  for (int i = 0; i < size; ++i) {
    model.push_back(a++);
  }
}

TEST_F(UtestGEExecutorLiteOSTest, LoadDataFromFile_Normal) {
  ExeModelBuilder modelBuilder;
  std::vector<uint64_t> dims1 = {1, 2, 3};
  std::vector<uint64_t> dims2 = {};
  std::vector<std::pair<uint64_t, uint64_t>> shapeRange = {{1, 2}, {2, 5}};
  modelBuilder.AddPartition(PRE_MODEL_DESC, [] (std::vector<uint8_t> &model) {
    ModelDesc desc = {
      .task_num = 1,
      .workspace_size = 100,
      .weight_size = 103,
      .weight_type = 0,
      .profile_enable = 0,
      .model_interrupt = 0,
    };

  std::copy_n((uint8_t*)(&desc), sizeof(ModelDesc), std::back_inserter(model));
    }).AddPartition(DYNAMIC_TASK_DESC, [] (std::vector<uint8_t> &model) {
      GenerateEleForModel(0, 100, model);
    }).AddPartition(TBE_KERNELS, [] (std::vector<uint8_t> &model) {
      GenerateEleForModel(101, 101, model);
    }).AddPartition(TASK_PARAM, [] (std::vector<uint8_t> &model) {
      GenerateEleForModel(201, 102, model);
    }).AddPartition(WEIGHTS_DATA, [] (std::vector<uint8_t> &model) {
      GenerateEleForModel(31, 103, model);
    }).AddPartition(STATIC_TASK_DESC, [] (std::vector<uint8_t> &model) {
      GenerateEleForModel(45, 301, model);
    }).AddInputDesc("op1", 10, FORMAT_CHWN, DT_FLOAT, dims1, dims2, shapeRange
     ).AddOutputDesc("op2", 8, FORMAT_CHWN, DT_FLOAT, dims1, dims2, shapeRange).Build();

  // write modelBuilder to file
  std::string fileName;
  fileName.append("./test").append(".exeom");
  FILE *f = fopen(fileName.c_str(), "wb+");
  fwrite(modelBuilder.ModelData(), modelBuilder.ModelLen(), 1, f);
  fclose(f);

  GeInitialize();
  ModelData modelData;
  (void)memset_s(&modelData, sizeof(ModelData), 0, sizeof(ModelData));
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmMalloc(_)).WillRepeatedly(Invoke(mmMalloc_Normal_Invoke));
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmReadFile(_,_,_,_)).WillRepeatedly(Invoke(mmReadFile_Normal_Invoke));
  EXPECT_EQ(LoadDataFromFile(fileName.c_str(), &modelData), SUCCESS);
  FreeModelData(&modelData);
  GeFinalize();
}

TEST_F(UtestGEExecutorLiteOSTest, LoadDataFromFile_LoadModelFromData_WithoutUserData_Normal) {
  ExeModelBuilder modelBuilder;
  std::vector<uint64_t> dims1 = {1, 2, 3};
  std::vector<uint64_t> dims2 = {};
  std::vector<std::pair<uint64_t, uint64_t>> shapeRange = {{1, 2}, {2, 5}};
  modelBuilder.AddPartition(PRE_MODEL_DESC, [] (std::vector<uint8_t> &model) {
    ModelDesc desc = {
      .task_num = 1,
      .workspace_size = 100,
      .weight_size = 103,
      .weight_type = 0,
      .profile_enable = 0,
      .model_interrupt = 0,
    };

  std::copy_n((uint8_t*)(&desc), sizeof(ModelDesc), std::back_inserter(model));
    }).AddPartition(DYNAMIC_TASK_DESC, [] (std::vector<uint8_t> &model) {
      GenerateEleForModel(0, 100, model);
    }).AddPartition(TBE_KERNELS, [] (std::vector<uint8_t> &model) {
      GenerateEleForModel(101, 101, model);
    }).AddPartition(TASK_PARAM, [] (std::vector<uint8_t> &model) {
      GenerateEleForModel(201, 102, model);
    }).AddPartition(WEIGHTS_DATA, [] (std::vector<uint8_t> &model) {
      GenerateEleForModel(31, 103, model);
    }).AddPartition(STATIC_TASK_DESC, [] (std::vector<uint8_t> &model) {
      GenerateEleForModel(45, 301, model);
    }).AddInputDesc("op1", 10, FORMAT_CHWN, DT_FLOAT, dims1, dims2, shapeRange
     ).AddOutputDesc("op2", 8, FORMAT_CHWN, DT_FLOAT, dims1, dims2, shapeRange).Build();

  // write modelBuilder to file
  std::string fileName;
  fileName.append("./test").append(".exeom");
  FILE *f = fopen(fileName.c_str(), "wb+");
  fwrite(modelBuilder.ModelData(), modelBuilder.ModelLen(), 1, f);
  fclose(f);

  GeInitialize();
  ModelData modelData;
  (void)memset_s(&modelData, sizeof(ModelData), 0, sizeof(ModelData));
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmMalloc(_)).WillRepeatedly(Invoke(mmMalloc_Normal_Invoke));
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmReadFile(_,_,_,_)).Times(1).WillOnce(Invoke(mmReadFile_Abnormal_Invoke));
  // covers first read fail in LoadOfflineModelFromFile
  EXPECT_NE(LoadDataFromFile(fileName.c_str(), &modelData), SUCCESS);
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmReadFile(_,_,_,_)).Times(2).WillOnce(Invoke(mmReadFile_Normal_Invoke)).WillOnce(Invoke(mmReadFile_Abnormal_Invoke));
  // covers second read fail in LoadOfflineModelFromFile
  EXPECT_NE(LoadDataFromFile(fileName.c_str(), &modelData), SUCCESS);

  EXPECT_CALL(MmpaStubMock::GetInstance(), mmReadFile(_,_,_,_)).WillRepeatedly(Invoke(mmReadFile_Normal_Invoke));
  EXPECT_EQ(LoadDataFromFile(fileName.c_str(), &modelData), SUCCESS);
  uint32_t modelId = 0;
  EXPECT_EQ(GeLoadModelFromData(&modelId, &modelData), SUCCESS);
  FreeModelData(&modelData);
  GeFinalize();
}

static void RtFree(uint8_t **ptr) {
  if (*ptr != nullptr) {
    (void)rtFree(*ptr);
    *ptr = nullptr;
  }
}

TEST_F(UtestGEExecutorLiteOSTest, LoadDataFromFile_LoadModelFromData_WithuserData_Normal) {
  ExeModelBuilder modelBuilder;
  std::vector<uint64_t> dims1 = {1, 2, 3};
  std::vector<uint64_t> dims2 = {};
  std::vector<std::pair<uint64_t, uint64_t>> shapeRange = {{1, 2}, {2, 5}};
  modelBuilder.AddPartition(PRE_MODEL_DESC, [] (std::vector<uint8_t> &model) {
    ModelDesc desc = {
      .task_num = 1,
      .workspace_size = 100,
      .weight_size = 103,
      .weight_type = 0,
      .profile_enable = 0,
      .model_interrupt = 0,
    };

  std::copy_n((uint8_t*)(&desc), sizeof(ModelDesc), std::back_inserter(model));
    }).AddPartition(DYNAMIC_TASK_DESC, [] (std::vector<uint8_t> &model) {
      GenerateEleForModel(0, 100, model);
    }).AddPartition(TBE_KERNELS, [] (std::vector<uint8_t> &model) {
      GenerateEleForModel(101, 101, model);
    }).AddPartition(TASK_PARAM, [] (std::vector<uint8_t> &model) {
      GenerateEleForModel(201, 102, model);
    }).AddPartition(WEIGHTS_DATA, [] (std::vector<uint8_t> &model) {
      GenerateEleForModel(31, 103, model);
    }).AddPartition(STATIC_TASK_DESC, [] (std::vector<uint8_t> &model) {
      GenerateEleForModel(45, 301, model);
    }).AddInputDesc("op1", 10, FORMAT_CHWN, DT_FLOAT, dims1, dims2, shapeRange
     ).AddOutputDesc("op2", 8, FORMAT_CHWN, DT_FLOAT, dims1, dims2, shapeRange).Build();

  // write modelBuilder to file
  std::string fileName;
  fileName.append("./test").append(".exeom");
  FILE *f = fopen(fileName.c_str(), "wb+");
  fwrite(modelBuilder.ModelData(), modelBuilder.ModelLen(), 1, f);
  fclose(f);

  GeInitialize();
  uint8_t *workspace = nullptr;
  (void)rtMalloc((void **)&workspace, sizeof(uint8_t) * 100, RT_MEMORY_DEFAULT, 0);
  size_t weightSize = 103;
  uint8_t *weightPtr = nullptr;
  (void)rtMalloc((void **)&weightPtr, sizeof(uint8_t) * weightSize, RT_MEMORY_DEFAULT, 0);

  size_t modelDescSize = sizeof(ModelDesc);
  uint8_t *modelDescPtr = nullptr;
  (void)rtMalloc((void **)&modelDescPtr, sizeof(uint8_t) * modelDescSize, RT_MEMORY_DEFAULT, 0);

  size_t kernelSize = 101;
  uint8_t *kernelPtr = nullptr;
  (void)rtMalloc((void **)&kernelPtr, sizeof(uint8_t) * kernelSize, RT_MEMORY_DEFAULT, 0);

  size_t paramSize = 102;
  uint8_t *paramPtr = nullptr;
  (void)rtMalloc((void **)&paramPtr, sizeof(uint8_t) * paramSize, RT_MEMORY_DEFAULT, 0);

  size_t taskSize = 301;
  uint8_t *taskPtr = nullptr;
  (void)rtMalloc((void **)&taskPtr, sizeof(uint8_t) * taskSize, RT_MEMORY_DEFAULT, 0);

  size_t dynTaskSize = 100;
  uint8_t *dynTaskPtr = nullptr;
  (void)rtMalloc((void **)&dynTaskPtr, sizeof(uint8_t) * dynTaskSize, RT_MEMORY_DEFAULT, 0);

  ModelData modelData;
  (void)memset_s(&modelData, sizeof(ModelData), 0, sizeof(ModelData));
  modelData.part.weightPtr = weightPtr;
  modelData.part.weightSize = weightSize;

  modelData.part.modelDescPtr = modelDescPtr;
  modelData.part.modelDescSize = modelDescSize;

  modelData.part.kernelPtr = kernelPtr;
  modelData.part.kernelSize = kernelSize;

  modelData.part.paramPtr = paramPtr;
  modelData.part.paramSize = paramSize;

  modelData.part.taskPtr = taskPtr;
  modelData.part.taskSize = taskSize;

  modelData.part.dynTaskPtr = dynTaskPtr;
  modelData.part.dynTaskSize = dynTaskSize;
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmTellFile(_)).Times(1).WillOnce(Invoke(mmTellFile_Abnormal_Invoke));
  // covers tell fail in LoadOfflineModelFromFile
  EXPECT_EQ(LoadDataFromFile(fileName.c_str(), &modelData), ACL_ERROR_GE_EXEC_MODEL_PATH_INVALID);
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmTellFile(_)).WillRepeatedly(Invoke(mmTellFile_Normal_Invoke));

  EXPECT_CALL(MmpaStubMock::GetInstance(), mmMalloc(_)).Times(1).WillOnce(Invoke(mmMalloc_Abnormal_Invoke));
  // covers malloc fail in LoadOfflineModelFromFile
  EXPECT_EQ(LoadDataFromFile(fileName.c_str(), &modelData), ACL_ERROR_GE_MEMORY_ALLOCATION);

  EXPECT_CALL(MmpaStubMock::GetInstance(), mmMalloc(_)).WillRepeatedly(Invoke(mmMalloc_Normal_Invoke));
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmReadFile(_,_,_,_)).WillRepeatedly(Invoke(mmReadFile_Normal_Invoke));
  EXPECT_EQ(LoadDataFromFile(fileName.c_str(), &modelData), SUCCESS);
  uint32_t modelId = 0;
  EXPECT_EQ(GeLoadModelFromData(&modelId, &modelData), SUCCESS);
  FreeModelData(&modelData);
  ExecHandleDesc execDesc;
  execDesc.workPtr = workspace;
  execDesc.workSize = 100;
   InputData inputData;
  InitVector(&inputData.blobs, sizeof(DataBuffer));
  DataBlob inputBlob;
  DataBuffer input;
  uint32_t inputAddr = 0x1005;
  input.data = (void *)(&inputAddr);
  input.length = 50;
  inputBlob.dataBuffer = &input;
  EmplaceBackVector(&inputData.blobs, &inputBlob);
  OutputData outputData;
  InitVector(&outputData.blobs, sizeof(DataBuffer));
  outputData.io_addr = NULL;
  outputData.io_addr_host = NULL;
  outputData.ioa_size = 1;
  DataBlob outputBlob;
  DataBuffer output;
  uint32_t outputAddr = 0x1004;
  output.data = (void *)(&outputAddr);
  output.length = 100;
  outputBlob.dataBuffer = &output;
  EmplaceBackVector(&outputData.blobs, &outputBlob);
  EXPECT_EQ(ExecModel(modelId, &execDesc, true, &inputData, &outputData), SUCCESS);
  EXPECT_EQ(UnloadModel(modelId), SUCCESS);
  RtFree(&workspace);
  RtFree(&weightPtr);
  RtFree(&modelDescPtr);
  RtFree(&kernelPtr);
  RtFree(&paramPtr);
  RtFree(&taskPtr);
  RtFree(&dynTaskPtr);
  DeInitVector(&inputData.blobs);
  DeInitVector(&outputData.blobs);
  if (outputData.io_addr != NULL) {
    rtFree(outputData.io_addr);
  }
  if (outputData.io_addr_host != NULL) {
    mmFree(outputData.io_addr_host);
  }
  outputData.ioa_size = 0;
  GeFinalize();
}

TEST_F(UtestGEExecutorLiteOSTest, LoadDataFromFile_fileLen_Abnormal) {
  ExeModelBuilder modelBuilder;
  std::vector<uint64_t> dims1 = {1, 2, 3};
  std::vector<uint64_t> dims2 = {};
  std::vector<std::pair<uint64_t, uint64_t>> shapeRange = {{1, 2}, {2, 5}};
  modelBuilder.AddPartition(PRE_MODEL_DESC, [] (std::vector<uint8_t> &model) {
    ModelDesc desc = {
      .task_num = 1,
      .workspace_size = 100,
      .weight_size = 103,
      .weight_type = 0,
      .profile_enable = 0,
      .model_interrupt = 0,
    };

  std::copy_n((uint8_t*)(&desc), sizeof(ModelDesc), std::back_inserter(model));
    }).AddPartition(DYNAMIC_TASK_DESC, [] (std::vector<uint8_t> &model) {
      GenerateEleForModel(0, 100, model);
    }).AddPartition(TBE_KERNELS, [] (std::vector<uint8_t> &model) {
      GenerateEleForModel(101, 101, model);
    }).AddPartition(TASK_PARAM, [] (std::vector<uint8_t> &model) {
      GenerateEleForModel(201, 102, model);
    }).AddPartition(WEIGHTS_DATA, [] (std::vector<uint8_t> &model) {
      GenerateEleForModel(31, 103, model);
    }).AddPartition(STATIC_TASK_DESC, [] (std::vector<uint8_t> &model) {
      GenerateEleForModel(45, 301, model);
    }).AddInputDesc("op1", 10, FORMAT_CHWN, DT_FLOAT, dims1, dims2, shapeRange
     ).AddOutputDesc("op2", 8, FORMAT_CHWN, DT_FLOAT, dims1, dims2, shapeRange).Build();

  // write modelBuilder to file
  std::string fileName;
  fileName.append("./test").append(".exeom");
  FILE *f = fopen(fileName.c_str(), "wb+");
  fwrite(modelBuilder.ModelData(), 1, 1, f);
  fclose(f);

  GeInitialize();
  ModelData modelData;
  (void)memset_s(&modelData, sizeof(ModelData), 0, sizeof(ModelData));
  EXPECT_EQ(LoadDataFromFile(fileName.c_str(), &modelData), ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID);
  FreeModelData(&modelData);
  GeFinalize();
}

TEST_F(UtestGEExecutorLiteOSTest, MODEL_INOUT_INFO_Parse_size_Abnormal) {
  ExeModelBuilder modelBuilder;
  std::vector<uint64_t> dims1 = {1, 2, 3};
  std::vector<uint64_t> dims2 = {};
  std::vector<std::pair<uint64_t, uint64_t>> shapeRange = {{1, 2}, {2, 5}};
  modelBuilder.AddPartition(PRE_MODEL_DESC, [] (std::vector<uint8_t> &model) {
    ModelDesc desc = {
      .task_num = 1,
      .workspace_size = 100,
      .weight_size = 103,
      .weight_type = 0,
      .profile_enable = 0,
      .model_interrupt = 0,
    };

  std::copy_n((uint8_t*)(&desc), sizeof(ModelDesc), std::back_inserter(model));
    }).AddPartition(MODEL_INOUT_INFO, [] (std::vector<uint8_t> &model) {
      GenerateEleForModel(0, 3, model);
    }).Build();

  // write modelBuilder to file
  std::string fileName;
  fileName.append("./test").append(".exeom");
  FILE *f = fopen(fileName.c_str(), "wb+");
  fwrite(modelBuilder.ModelData(), modelBuilder.ModelLen(), 1, f);
  fclose(f);

  GeInitialize();
  ModelData modelData;
  (void)memset_s(&modelData, sizeof(ModelData), 0, sizeof(ModelData));
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmMalloc(_)).WillRepeatedly(Invoke(mmMalloc_Normal_Invoke));
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmReadFile(_,_,_,_)).WillRepeatedly(Invoke(mmReadFile_Normal_Invoke));
  EXPECT_EQ(LoadDataFromFile(fileName.c_str(), &modelData), SUCCESS);
  uint32_t modelId = 0;
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmMalloc(_)).Times(2).WillOnce(Invoke(mmMalloc_Normal_Invoke)).WillOnce(Invoke(mmMalloc_Abnormal_Invoke));
  EXPECT_EQ(LoadOfflineModelFromData(&modelId, &modelData), ACL_ERROR_GE_MEMORY_ALLOCATION); // covers malloc fail in ParserPartionFromFd
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmMalloc(_)).WillRepeatedly(Invoke(mmMalloc_Normal_Invoke));
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmReadFile(_,_,_,_)).Times(2).WillOnce(Invoke(mmReadFile_Normal_Invoke)).WillOnce(Invoke(mmReadFile_Abnormal_Invoke));
  EXPECT_EQ(LoadOfflineModelFromData(&modelId, &modelData), ACL_ERROR_GE_LOAD_MODEL); // covers read fail in ParserPartionFromFd

  EXPECT_CALL(MmpaStubMock::GetInstance(), mmReadFile(_,_,_,_)).WillRepeatedly(Invoke(mmReadFile_Normal_Invoke));
  EXPECT_EQ(GeLoadModelFromData(&modelId, &modelData), ACL_ERROR_GE_INTERNAL_ERROR);

  ModelInOutInfo info;
  EXPECT_EQ(GetModelDescInfoFromMem(&modelData, &info), ACL_ERROR_GE_INTERNAL_ERROR);
  FreeModelData(&modelData);
  GeFinalize();
}

void StubExtendPartitionNormalLiteOS(std::vector<uint8_t> &model) {
  static uint8_t stub[] = {
      // uint8_t[] data
      0x48, 0x4D, 0x4F, 0x44,                          // magic
      0x24, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // len 36
      // uint8_t[] tlv
      0x00, 0x00, 0x00, 0x00,                          // type 0
      0x1C, 0x00, 0x00, 0x00,                          // len 28
      0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // total_size 16
      0x02, 0x00, 0x00, 0x00,                          // num
      0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // mem_size 0
      0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // mem_size 1
  };
  size_t total_size =
      sizeof(TlvHead) + sizeof(struct ModelExtendHead) + sizeof(struct ModelGlobalDataInfo) + 2 * sizeof(uint64_t);  // mem_size
  std::copy_n(stub, total_size, std::back_inserter(model));
}

void StubExtendPartitionAbnormal01LiteOS(std::vector<uint8_t> &model) {
  static uint8_t stub[] = {
      // uint8_t[] data
      0x48, 0x4D, 0x4F, 0x44,  // magic
  };
  size_t total_size = 4;
  std::copy_n(stub, total_size, std::back_inserter(model));
}

TEST_F(UtestGEExecutorLiteOSTest, GetPartitionSizeExtendInfoAbnormal) {
  ExeModelBuilder modelBuilder;
  modelBuilder.AddPartition(PRE_MODEL_DESC, [] (std::vector<uint8_t> &model) {
    ModelDesc desc = {
      .task_num = 1,
      .workspace_size = 100,
      .weight_size = 103,
      .weight_type = 0,
      .profile_enable = 0,
      .model_interrupt = 0,
    };

  std::copy_n((uint8_t*)(&desc), sizeof(ModelDesc), std::back_inserter(model));
    }).AddPartition(PRE_MODEL_DESC_EXTEND, [](std::vector<uint8_t> &model) { StubExtendPartitionAbnormal01LiteOS(model); }).Build();

  // write modelBuilder to file
  std::string fileName;
  fileName.append("./test").append(".exeom");
  FILE *f = fopen(fileName.c_str(), "wb+");
  fwrite(modelBuilder.ModelData(), modelBuilder.ModelLen(), 1, f);
  fclose(f);

  GeInitialize();
  GePartitionSize parSize;
  EXPECT_EQ(GetPartitionSize(fileName.c_str(), &parSize), ACL_ERROR_GE_LOAD_MODEL);
  GeFinalize();
}

TEST_F(UtestGEExecutorLiteOSTest, GetModelFifoSizeGetPartInfoFromModelAbnormal) {
  ExeModelBuilder modelBuilder;
  modelBuilder.AddPartition(PRE_MODEL_DESC, [] (std::vector<uint8_t> &model) {
    ModelDesc desc = {
      .task_num = 1,
      .workspace_size = 100,
      .weight_size = 103,
      .weight_type = 0,
      .profile_enable = 0,
      .model_interrupt = 0,
    };

  std::copy_n((uint8_t*)(&desc), sizeof(ModelDesc), std::back_inserter(model));
    }).AddPartition(PRE_MODEL_DESC_EXTEND, [](std::vector<uint8_t> &model) { StubExtendPartitionNormalLiteOS(model); }).Build();

  // write modelBuilder to file
  std::string fileName;
  fileName.append("./test").append(".exeom");
  FILE *f = fopen(fileName.c_str(), "wb+");
  fwrite(modelBuilder.ModelData(), modelBuilder.ModelLen(), 1, f);
  fclose(f);

  GeInitialize();
  GePartitionSize parSize;
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmMalloc(_))
      .Times(3)
      .WillOnce(Invoke(mmMalloc_Normal_Invoke))
      .WillOnce(Invoke(mmMalloc_Normal_Invoke))
      .WillOnce(Invoke(mmMalloc_Abnormal_Invoke));
  EXPECT_EQ(GetPartitionSize(fileName.c_str(), &parSize), ACL_ERROR_GE_MEMORY_OPERATE_FAILED);
  GeFinalize();
}

TEST_F(UtestGEExecutorLiteOSTest, QueryPartitionSize) {
  ExeModelBuilder modelBuilder;
  std::vector<uint64_t> dims1 = {1, 2, 3};
  std::vector<uint64_t> dims2 = {};
  std::vector<std::pair<uint64_t, uint64_t>> shapeRange = {{1, 2}, {2, 5}};
  modelBuilder.AddPartition(PRE_MODEL_DESC, [] (std::vector<uint8_t> &model) {
    ModelDesc desc = {
      .task_num = 1,
      .workspace_size = 100,
      .weight_size = 103,
      .weight_type = 0,
      .profile_enable = 0,
      .model_interrupt = 0,
    };

  std::copy_n((uint8_t*)(&desc), sizeof(ModelDesc), std::back_inserter(model));
    }).AddPartition(PRE_MODEL_DESC_EXTEND, [](std::vector<uint8_t> &model) { StubExtendPartitionNormalLiteOS(model); })
      .AddPartition(DYNAMIC_TASK_DESC, [] (std::vector<uint8_t> &model) {
      GenerateEleForModel(0, 100, model);
    }).AddPartition(TBE_KERNELS, [] (std::vector<uint8_t> &model) {
      GenerateEleForModel(101, 101, model);
    }).AddPartition(TASK_PARAM, [] (std::vector<uint8_t> &model) {
      GenerateEleForModel(201, 102, model);
    }).AddPartition(WEIGHTS_DATA, [] (std::vector<uint8_t> &model) {
      GenerateEleForModel(31, 103, model);
    }).AddPartition(STATIC_TASK_DESC, [] (std::vector<uint8_t> &model) {
      GenerateEleForModel(45, 301, model);
    }).AddInputDesc("op1", 10, FORMAT_CHWN, DT_FLOAT, dims1, dims2, shapeRange
     ).AddOutputDesc("op2", 8, FORMAT_CHWN, DT_FLOAT, dims1, dims2, shapeRange).Build();

  // write modelBuilder to file
  std::string fileName;
  fileName.append("./test").append(".exeom");
  FILE *f = fopen(fileName.c_str(), "wb+");
  fwrite(modelBuilder.ModelData(), modelBuilder.ModelLen(), 1, f);
  fclose(f);

  GeInitialize();
  size_t mem_size = 0;
  size_t weight_size = 0;
  EXPECT_EQ(GetMemAndWeightSize(fileName.c_str(), &mem_size, &weight_size), SUCCESS);
  EXPECT_EQ(mem_size, 100);
  EXPECT_EQ(weight_size, 103);
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmReadFile(_,_,_,_)).Times(3).WillOnce(Invoke(mmReadFile_Normal_Invoke)).WillOnce(Invoke(mmReadFile_Normal_Invoke)).WillOnce(Invoke(mmReadFile_Abnormal_Invoke));
  GePartitionSize parSizeTmp;
  // covers first read fail in GetPartitonSizeFromFd
  EXPECT_NE(GetPartitionSize(fileName.c_str(), &parSizeTmp), SUCCESS);
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmReadFile(_,_,_,_)).Times(3).WillOnce(Invoke(mmReadFile_Normal_Invoke)).WillOnce(Invoke(mmReadFile_Normal_Invoke)).WillOnce(Invoke(mmReadFile_Abnormal_Invoke));
  // covers second read fail in GetPartitonSizeFromFd
  EXPECT_NE(GetPartitionSize(fileName.c_str(), &parSizeTmp), SUCCESS);
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmReadFile(_,_,_,_)).WillRepeatedly(Invoke(mmReadFile_Normal_Invoke));
  GePartitionSize parSize;
  EXPECT_EQ(GetPartitionSize(fileName.c_str(), &parSize), SUCCESS);
  EXPECT_EQ(parSize.modelDescSize, 26);
  EXPECT_EQ(parSize.dynamicTaskSize, 100);
  EXPECT_EQ(parSize.kernelSize, 101);
  EXPECT_EQ(parSize.kernelArgsSize, 102);
  EXPECT_EQ(parSize.weightSize, 103);
  EXPECT_EQ(parSize.workSize, 100);
  EXPECT_EQ(parSize.staticTaskSize, 301);
  EXPECT_EQ(parSize.fifoSize, 16);

  ModelData modelData;
  (void)memset_s(&modelData, sizeof(ModelData), 0, sizeof(ModelData));
  ModelInOutInfo info;
  EXPECT_EQ(LoadDataFromFile(fileName.c_str(), &modelData), SUCCESS);
  // covers malloc fail in GetModelInOutDesc
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmMalloc(_))
      .Times(3)
      .WillOnce(Invoke(mmMalloc_Normal_Invoke))
      .WillOnce(Invoke(mmMalloc_Normal_Invoke))
      .WillOnce(Invoke(mmMalloc_Abnormal_Invoke));
  EXPECT_EQ(GetModelDescInfoFromMem(&modelData, &info), ACL_ERROR_GE_MEMORY_OPERATE_FAILED);

  EXPECT_CALL(MmpaStubMock::GetInstance(), mmMalloc(_)).WillRepeatedly(Invoke(mmMalloc_Normal_Invoke));
  EXPECT_EQ(GetModelDescInfoFromMem(&modelData, &info), SUCCESS);
  DestroyModelInOutInfo(&info);
  FreeModelData(&modelData);
  GeFinalize();
}

TEST_F(UtestGEExecutorLiteOSTest, GeExecutorCaseInvalidOm) {
  GeInitialize();

  // Exeom head magic is wrong
  std::string test_smap = "/tmp/" + std::to_string(getpid()) + "_maps";
  std::string self_smap = "/proc/" + std::to_string(getpid()) + "/maps";
  std::string copy_smap = "cp -f " + self_smap + " " + test_smap;
  EXPECT_EQ(system(copy_smap.c_str()), 0);
  size_t mem_size = 0;
  size_t weight_size = 0;
  GePartitionSize parSize;
  EXPECT_NE(GetMemAndWeightSize(test_smap.c_str(), &mem_size, &weight_size), SUCCESS);
  EXPECT_NE(GetPartitionSize(test_smap.c_str(), &parSize), SUCCESS);

  // Exeom not exist
  std::string filePath = "contact.exeom";
  EXPECT_EQ(GetMemAndWeightSize(filePath.c_str(), &mem_size, &weight_size), ACL_ERROR_GE_EXEC_MODEL_PATH_INVALID);
  EXPECT_EQ(GetPartitionSize(filePath.c_str(), &parSize), ACL_ERROR_GE_EXEC_MODEL_PATH_INVALID);
  GeFinalize();
}

TEST_F(UtestGEExecutorLiteOSTest, GeExecutorPartitionSizeInvalid) {
  ExeModelBuilder modelBuilder;
  modelBuilder.AddPartition(DYNAMIC_TASK_DESC, [] (std::vector<uint8_t> &model) {
    (void)model;
    }).AddPartition(TBE_KERNELS, [] (std::vector<uint8_t> &model) {
      GenerateEleForModel(101, 101, model);
    }).Build();

  // write modelBuilder to file
  std::string fileName;
  fileName.append("./test").append(".exeom");
  FILE *f = fopen(fileName.c_str(), "wb+");
  fwrite(modelBuilder.ModelData(), modelBuilder.ModelLen(), 1, f);
  fclose(f);

  GeInitialize();
  ModelData modelData;
  (void)memset_s(&modelData, sizeof(ModelData), 0, sizeof(ModelData));
  EXPECT_EQ(LoadDataFromFile(fileName.c_str(), &modelData), SUCCESS);
  GePartitionSize parSize;
  EXPECT_EQ(GetModelPartitionSize(&modelData, &parSize), ACL_ERROR_GE_INTERNAL_ERROR);
  FreeModelData(&modelData);
  GeFinalize();
}

TEST_F(UtestGEExecutorLiteOSTest, LoadDataFromFile_pathLen_Abnormal) {
  ExeModelBuilder modelBuilder;
  std::vector<uint64_t> dims1 = {1, 2, 3};
  std::vector<uint64_t> dims2 = {};
  std::vector<std::pair<uint64_t, uint64_t>> shapeRange = {{1, 2}, {2, 5}};
  modelBuilder.AddPartition(PRE_MODEL_DESC, [] (std::vector<uint8_t> &model) {
    ModelDesc desc = {
      .task_num = 1,
      .workspace_size = 100,
      .weight_size = 103,
      .weight_type = 0,
      .profile_enable = 0,
      .model_interrupt = 0,
    };

  std::copy_n((uint8_t*)(&desc), sizeof(ModelDesc), std::back_inserter(model));
    }).Build();

  // write modelBuilder to file
  std::string fileName;
  fileName
      .append(
          "./"
          "testtesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttest"
          "testtesttesttesttesttesttesttesttesttesttesttesttesttesttesttesttest"
          "testtesttesttesttesttesttesttesttesttesttesttesttesttesttesttest")
      .append(".exeom");
  FILE *f = fopen(fileName.c_str(), "wb+");
  fwrite(modelBuilder.ModelData(), modelBuilder.ModelLen(), 1, f);
  fclose(f);

  GeInitialize();
  ModelData modelData;
  (void)memset_s(&modelData, sizeof(ModelData), 0, sizeof(ModelData));
  EXPECT_EQ(LoadDataFromFile(fileName.c_str(), &modelData), ACL_ERROR_GE_EXEC_MODEL_PATH_INVALID);
  FreeModelData(&modelData);
  GeFinalize();
}

TEST_F(UtestGEExecutorLiteOSTest, ReadDataFromFd_Abnormal) {
  int a = 20;
  void *dst = &a;
  EXPECT_EQ(ReadDataFromFd(nullptr, 1, 1, dst), ACL_ERROR_GE_PARAM_INVALID);

  ExeModelBuilder modelBuilder;
  std::vector<uint64_t> dims1 = {1, 2, 3};
  std::vector<uint64_t> dims2 = {};
  std::vector<std::pair<uint64_t, uint64_t>> shapeRange = {{1, 2}, {2, 5}};
  modelBuilder.AddPartition(PRE_MODEL_DESC, [] (std::vector<uint8_t> &model) {
    ModelDesc desc = {
      .task_num = 1,
      .workspace_size = 100,
      .weight_size = 103,
      .weight_type = 0,
      .profile_enable = 0,
      .model_interrupt = 0,
    };

  std::copy_n((uint8_t*)(&desc), sizeof(ModelDesc), std::back_inserter(model));
    }).Build();

  // write modelBuilder to file
  std::string fileName;
  fileName.append("./test").append(".exeom");
  FILE *f = fopen(fileName.c_str(), "wb+");
  fwrite(modelBuilder.ModelData(), modelBuilder.ModelLen(), 1, f);
  fclose(f);

  mmFileHandle *fd = mmOpenFile(fileName.c_str(), FILE_READ_BIN);
  EXPECT_EQ(ReadDataFromFd(fd, 1, 0, dst), ACL_ERROR_GE_LOAD_MODEL);
  mmCloseFile(fd);
}

TEST_F(UtestGEExecutorLiteOSTest, GeExecutorCaseParseModelDescExtendNormal) {
  ExeModelBuilder modelBuilder;
  modelBuilder
      .AddPartition(PRE_MODEL_DESC,
                    [](std::vector<uint8_t> &model) {
                      ModelDesc desc = {
                          .task_num = 1,
                          .workspace_size = 100,
                          .weight_size = 103,
                          .weight_type = 0,
                          .profile_enable = 0,
                          .model_interrupt = 0,
                      };
                      std::copy_n((uint8_t *)(&desc), sizeof(ModelDesc), std::back_inserter(model));
                    })
      .AddPartition(PRE_MODEL_DESC_EXTEND, [](std::vector<uint8_t> &model) { StubExtendPartitionNormalLiteOS(model); })
      .Build();

  // write modelBuilder to file
  std::string fileName;
  fileName.append("./test").append(".exeom");
  FILE *f = fopen(fileName.c_str(), "wb+");
  fwrite(modelBuilder.ModelData(), modelBuilder.ModelLen(), 1, f);
  fclose(f);

  GeInitialize();
  ModelData modelData;
  (void)memset_s(&modelData, sizeof(ModelData), 0, sizeof(ModelData));
  EXPECT_EQ(LoadDataFromFile(fileName.c_str(), &modelData), SUCCESS);
  uint32_t modelId = 0;
  EXPECT_CALL(RtStubMock::GetInstance(), rtNanoModelLoad(_, _)).Times(AnyNumber()).WillOnce(Return(RT_ERROR_NONE));
  EXPECT_CALL(RtStubMock::GetInstance(), rtNanoModelDestroy(_)).Times(AnyNumber()).WillOnce(Return(RT_ERROR_NONE));
  EXPECT_EQ(GeLoadModelFromData(&modelId, &modelData), SUCCESS);
  FreeModelData(&modelData);
  GeFinalize();
}

TEST_F(UtestGEExecutorLiteOSTest, GeExecutorCaseParseModelDescExtendAbnormalInvalidModelExtendHead) {
  ExeModelBuilder modelBuilder;
  modelBuilder
      .AddPartition(PRE_MODEL_DESC,
                    [](std::vector<uint8_t> &model) {
                      ModelDesc desc = {
                          .task_num = 1,
                          .workspace_size = 100,
                          .weight_size = 103,
                          .weight_type = 0,
                          .profile_enable = 0,
                          .model_interrupt = 0,
                      };
                      std::copy_n((uint8_t *)(&desc), sizeof(ModelDesc), std::back_inserter(model));
                    })
      .AddPartition(PRE_MODEL_DESC_EXTEND, [](std::vector<uint8_t> &model) { StubExtendPartitionAbnormal01LiteOS(model); })
      .Build();

  // write modelBuilder to file
  std::string fileName;
  fileName.append("./test").append(".exeom");
  FILE *f = fopen(fileName.c_str(), "wb+");
  fwrite(modelBuilder.ModelData(), modelBuilder.ModelLen(), 1, f);
  fclose(f);

  GeInitialize();
  ModelData modelData;
  (void)memset_s(&modelData, sizeof(ModelData), 0, sizeof(ModelData));
  EXPECT_EQ(LoadDataFromFile(fileName.c_str(), &modelData), SUCCESS);
  uint32_t modelId = 0;
  EXPECT_EQ(GeLoadModelFromData(&modelId, &modelData), ACL_ERROR_GE_LOAD_MODEL);
  FreeModelData(&modelData);
  GeFinalize();
}

TEST_F(UtestGEExecutorLiteOSTest, GetPartInfoFromModel_mmMallocAbnormal) {
  ExeModelBuilder modelBuilder;
  modelBuilder.AddPartition(PRE_MODEL_DESC, [] (std::vector<uint8_t> &model) {
    ModelDesc desc = {
      .task_num = 1,
      .workspace_size = 100,
      .weight_size = 103,
      .weight_type = 0,
      .profile_enable = 0,
      .model_interrupt = 0,
    };

  std::copy_n((uint8_t*)(&desc), sizeof(ModelDesc), std::back_inserter(model));
    }).Build();

  GeInitialize();
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmMalloc(_)).WillRepeatedly(Invoke(mmMalloc_Abnormal_Invoke));
  ModelData model_data;
  (void)memset_s(&model_data , sizeof(ModelData), 0, sizeof(ModelData));
  model_data.modelData = modelBuilder.ModelData();
  model_data.modelLen = modelBuilder.ModelLen();
  model_data.flag = NEED_READ_FROM_FD;
  ModelPartition partition;
  partition.type = PRE_MODEL_DESC;
  EXPECT_EQ(GetPartInfoFromModel(&model_data, &partition), ACL_ERROR_GE_MEMORY_OPERATE_FAILED);
  GeFinalize();
}

TEST_F(UtestGEExecutorLiteOSTest, GetPartInfoFromModel_mmReadAbnormal) {
  ExeModelBuilder modelBuilder;
  modelBuilder.AddPartition(PRE_MODEL_DESC, [] (std::vector<uint8_t> &model) {
    ModelDesc desc = {
      .task_num = 1,
      .workspace_size = 100,
      .weight_size = 103,
      .weight_type = 0,
      .profile_enable = 0,
      .model_interrupt = 0,
    };

  std::copy_n((uint8_t*)(&desc), sizeof(ModelDesc), std::back_inserter(model));
    }).Build();

  // write modelBuilder to file
  std::string fileName;
  fileName.append("./test").append(".exeom");
  FILE *f = fopen(fileName.c_str(), "wb+");
  fwrite(modelBuilder.ModelData(), modelBuilder.ModelLen(), 1, f);
  fclose(f);

  GeInitialize();
  ModelData model_data;
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmMalloc(_)).WillRepeatedly(Invoke(mmMalloc_Normal_Invoke));
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmReadFile(_,_,_,_)).WillRepeatedly(Invoke(mmReadFile_Normal_Invoke));
  EXPECT_EQ(LoadDataFromFile(fileName.c_str(), &model_data), SUCCESS);

  ModelPartition partition;
  partition.type = PRE_MODEL_DESC;
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmReadFile(_,_,_,_)).WillRepeatedly(Invoke(mmReadFile_Abnormal_Invoke));
  EXPECT_EQ(GetPartInfoFromModel(&model_data, &partition), ACL_ERROR_GE_LOAD_MODEL);
  FreeModelData(&model_data);
  GeFinalize();
}
