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
#include "ge_executor.h"
#include "exe_model_builder.h"
#include "types.h"
#include "model_desc.h"
#include "maintain_manager.h"
#include "ge/ge_error_codes.h"
#include "gmock_stub/ascend_rt_stub.h"
#include "vector.h"
#include "model_loader.h"
#include "common/tlv/tlv.h"
#include "common/tlv/nano_common_desc.h"
#include <securec.h>
using namespace testing;
using ::testing::Return;
using ::testing::Eq;
using ::testing::NotNull;
using ::testing::Invoke;
using ::testing::_;
using namespace ge;

class UtestGEExecutorTest : public testing::Test {
 protected:
  void SetUp() {
    ON_CALL(RtStubMock::GetInstance(), access(_,_)).WillByDefault(Return(0));
    ON_CALL(RtStubMock::GetInstance(), rtNanoModelLoad(_,_)).WillByDefault(Return(0));
    ON_CALL(RtStubMock::GetInstance(), rtNanoModelDestroy(_)).WillByDefault(Return(0));
    ON_CALL(RtStubMock::GetInstance(), rtMemcpy(_,_,_,_,_)).WillByDefault(Invoke(rtMemcpy_Normal_Invoke));
    ON_CALL(RtStubMock::GetInstance(), rtMalloc(_,_,_,_)).WillByDefault(Invoke(rtMalloc_Normal_Invoke));
    ON_CALL(MmpaStubMock::GetInstance(), mmMalloc(_)).WillByDefault(Invoke(mmMalloc_Normal_Invoke));
  }
  void TearDown() {
    Mock::VerifyAndClearExpectations(&MmpaStubMock::GetInstance());
    Mock::VerifyAndClearExpectations(&RtStubMock::GetInstance());
  }
};

TEST_F(UtestGEExecutorTest, GeExecutorCaseBasicInit) {
  EXPECT_CALL(RtStubMock::GetInstance(), rtMemcpy(_,_,_,_,_)).WillRepeatedly(Invoke(rtMemcpy_Normal_Invoke));
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmTellFile(_)).WillRepeatedly(Invoke(mmTellFile_Normal_Invoke));
  EXPECT_CALL(RtStubMock::GetInstance(), rtMalloc(_, _, _, _)).WillRepeatedly(Invoke(rtMalloc_Normal_Invoke));
  Status ret = GeInitialize();
  ASSERT_EQ(ret, SUCCESS);
  ret = GeInitialize();
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestGEExecutorTest, GeExecutorCaseBasicGeFinalize) {
  GeInitialize();
  Status ret = GeFinalize();
  ASSERT_EQ(ret, SUCCESS);
  ret = GeFinalize();
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestGEExecutorTest, GeExecutorCaseUnGeInitialize) {
  std::string test_smap = "/tmp/" + std::to_string(getpid()) + "_maps";
  std::string self_smap = "/proc/" + std::to_string(getpid()) + "/maps";
  std::string copy_smap = "cp -f " + self_smap + " " + test_smap;
  EXPECT_EQ(system(copy_smap.c_str()), 0);
  uint32_t model_id = 0;
  size_t mem_size = 0;
  size_t weight_size = 0;
  ExecHandleDesc execDesc;
  InputData inputData;
  OutputData outputData;
  ModelData model_data;
  (void)memset_s(&model_data, sizeof(ModelData), 0, sizeof(ModelData));
  GePartitionSize parSize;
  EXPECT_EQ(LoadDataFromFile(test_smap.c_str(), &model_data), ACL_ERROR_GE_EXEC_NOT_INIT);
  EXPECT_EQ(GetMemAndWeightSize(test_smap.c_str(), &mem_size, &weight_size), ACL_ERROR_GE_EXEC_NOT_INIT);
  EXPECT_EQ(GetPartitionSize(test_smap.c_str(), &parSize), ACL_ERROR_GE_EXEC_NOT_INIT);
  EXPECT_EQ(GeLoadModelFromData(&model_id, &model_data), ACL_ERROR_GE_EXEC_NOT_INIT);
  EXPECT_EQ(ExecModel(model_id, &execDesc, false, &inputData, &outputData), ACL_ERROR_GE_EXEC_NOT_INIT);
  EXPECT_EQ(UnloadModel(model_id), ACL_ERROR_GE_EXEC_NOT_INIT);
  EXPECT_EQ(model_data.modelData, nullptr);
}

TEST_F(UtestGEExecutorTest, GeExecutorCaseInvalidOm) {
  GeInitialize();

  // Exeom head magic is wrong
  std::string test_smap = "/tmp/" + std::to_string(getpid()) + "_maps";
  std::string self_smap = "/proc/" + std::to_string(getpid()) + "/maps";
  std::string copy_smap = "cp -f " + self_smap + " " + test_smap;
  EXPECT_EQ(system(copy_smap.c_str()), 0);
  size_t mem_size = 0;
  size_t weight_size = 0;
  GePartitionSize parSize;
  EXPECT_EQ(GetMemAndWeightSize(test_smap.c_str(), &mem_size, &weight_size), ACL_ERROR_GE_INTERNAL_ERROR);
  EXPECT_EQ(GetPartitionSize(test_smap.c_str(), &parSize), ACL_ERROR_GE_INTERNAL_ERROR);
  ModelData model_data;
  (void)memset_s(&model_data, sizeof(ModelData), 0, sizeof(ModelData));
  ModelInOutInfo info;
  EXPECT_EQ(LoadDataFromFile(test_smap.c_str(), &model_data), SUCCESS);
  EXPECT_EQ(GetModelDescInfoFromMem(&model_data, &info), ACL_ERROR_GE_INTERNAL_ERROR);
  FreeModelData(&model_data);
  RefObj obj;
  EXPECT_EQ(FuncNull(&obj), nullptr);

  // Exeom not exist
  std::string filePath = "contact.exeom";
  EXPECT_EQ(GetMemAndWeightSize(filePath.c_str(), &mem_size, &weight_size), ACL_ERROR_GE_EXEC_MODEL_PATH_INVALID);
  EXPECT_EQ(GetPartitionSize(filePath.c_str(), &parSize), ACL_ERROR_GE_EXEC_MODEL_PATH_INVALID);
  GeFinalize();
}

TEST_F(UtestGEExecutorTest, GeExecutorCaseLoadDataFromFile) {
  GeInitialize();
  std::string test_smap = "/tmp/" + std::to_string(getpid()) + "_maps";
  std::string self_smap = "/proc/" + std::to_string(getpid()) + "/maps";
  std::string copy_smap = "cp -f " + self_smap + " " + test_smap;
  EXPECT_EQ(system(copy_smap.c_str()), 0);
  ModelData model_data;
  (void)memset_s(&model_data, sizeof(ModelData), 0, sizeof(ModelData));
  EXPECT_EQ(LoadDataFromFile(test_smap.c_str(), &model_data), SUCCESS);
  EXPECT_NE(model_data.modelData, nullptr);
  std::string filePath = "contact.exeom";
  EXPECT_NE(LoadDataFromFile(filePath.c_str(), &model_data), SUCCESS);
  FreeModelData(&model_data);
  GeFinalize();
}

void *CreatModelDbgHandleMock() {
  return NULL;
}

Status LoadModelPreProcessMock(void *taskDescBaseAddr, char *modelName, void *dbgHandle) {
  (void)taskDescBaseAddr;
  (void)modelName;
  (void)dbgHandle;
  return SUCCESS;
}

void StepIdConuterPlusMock(void *dbgHandle) {
  (void)dbgHandle;
  return;
}

void DataFreeLoadDumpInfoMock(void *dbgHandle) {
  (void)dbgHandle;
  return;
}

Status DbgInitMock(const char *configPath) {
  (void)configPath;
  return SUCCESS;
}

Status DbgDeInitMock() {
  return SUCCESS;
}

Status DbgNotifySetDeviceMock(uint32_t chipId, uint32_t deviceId)
{
  (void)chipId;
  (void)deviceId;
  return 0;
}

bool GetProfEnableMock() {
  return 0;
}

Status DbgLoadModelPostProcessMock(uint32_t modelId, char *om, uint64_t *stepIdAddr, void *dbgHandle) {
  (void)modelId;
  (void)om;
  (void)stepIdAddr;
  (void)dbgHandle;
  return SUCCESS;
}

static void GenerateEleForModel(uint8_t a, int size, std::vector<uint8_t> &model)
{
  for (int i = 0; i < size; ++i) {
    model.push_back(a++);
  }
}

TEST_F(UtestGEExecutorTest, GeExecutorCaseLoadModelFromData) {
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

  EXPECT_CALL(MmpaStubMock::GetInstance(), mmMalloc(_)).WillRepeatedly(Invoke(mmMalloc_Normal_Invoke));
  GeInitialize();
  int test = 1;
  void *ptr = &test;
  EXPECT_CALL(RtStubMock::GetInstance(), dlopen(_,_)).Times(1).WillOnce(Return(ptr));
  EXPECT_CALL(RtStubMock::GetInstance(), dlclose(_)).Times(1).WillOnce(Return(0));
  EXPECT_CALL(RtStubMock::GetInstance(), dlsym(_,_)).Times(9).WillOnce(Return((void *)DbgInitMock))
            .WillOnce(Return((void *)DbgDeInitMock))
            .WillOnce(Return((void *)DbgNotifySetDeviceMock))
            .WillOnce(Return((void *)GetProfEnableMock))
            .WillOnce(Return((void *)DbgLoadModelPostProcessMock))
            .WillOnce(Return((void *)CreatModelDbgHandleMock))
            .WillOnce(Return((void *)LoadModelPreProcessMock))
            .WillOnce(Return((void *)StepIdConuterPlusMock))
            .WillOnce(Return((void *)DataFreeLoadDumpInfoMock));
  const char *configPath = nullptr;
  GeDbgInit(configPath);
  uint32_t model_id = 0;
  size_t mem_size = 0;
  size_t weight_size = 0;
  uint8_t *weightspace = nullptr;
  (void)rtMalloc((void **)&weightspace, sizeof(uint8_t) * 103, RT_MEMORY_DEFAULT, 0);
  ModelData model_data;
  (void)memset_s(&model_data, sizeof(ModelData), 0, sizeof(ModelData));
  model_data.modelData = modelBuilder.ModelData();
  model_data.modelLen = modelBuilder.ModelLen();
  EXPECT_EQ(GetModelMemAndWeightSize(&model_data, &mem_size, &weight_size), SUCCESS);
  EXPECT_EQ(mem_size, 100);
  EXPECT_EQ(weight_size, 103);
  EXPECT_CALL(RtStubMock::GetInstance(), rtNanoModelLoad(_,_)).Times(1).WillOnce(Return(RT_ERROR_NONE));
  EXPECT_CALL(RtStubMock::GetInstance(), rtNanoModelDestroy(_)).Times(1).WillOnce(Return(RT_ERROR_NONE));
  EXPECT_EQ(GeLoadModelFromData(&model_id, nullptr), ACL_ERROR_GE_PARAM_INVALID);
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmMalloc(_)).Times(1).WillOnce(Invoke(mmMalloc_Abnormal_Invoke));
  // covers malloc fail in LoadOfflineModelFromData
  EXPECT_EQ(GeLoadModelFromData(&model_id, &model_data), ACL_ERROR_GE_MEMORY_ALLOCATION);
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmMalloc(_)).WillRepeatedly(Invoke(mmMalloc_Normal_Invoke));
  EXPECT_EQ(GeLoadModelFromData(&model_id, &model_data), SUCCESS);
  ModelInOutInfo info;
  EXPECT_EQ(GetModelDescInfo(model_id, &info), SUCCESS);
  EXPECT_EQ(VectorSize(&info.input_desc), 1);
  EXPECT_EQ(VectorSize(&info.output_desc), 1);
  DestroyModelInOutInfo(&info);
  EXPECT_EQ(UnloadModel(model_id), SUCCESS);
  EXPECT_CALL(RtStubMock::GetInstance(), rtNanoModelLoad(_,_)).Times(1).WillOnce(Return(1));
  EXPECT_EQ(GeLoadModelFromData(&model_id, &model_data), 1);
  EXPECT_CALL(RtStubMock::GetInstance(), rtNanoModelLoad(_,_)).Times(1).WillOnce(Return(RT_ERROR_NONE));
  EXPECT_CALL(RtStubMock::GetInstance(), rtNanoModelDestroy(_)).Times(1).WillOnce(Return(RT_ERROR_NONE));
  EXPECT_EQ(GeLoadModelFromData(&model_id, &model_data), SUCCESS);
  EXPECT_EQ(UnloadModel(model_id), SUCCESS);
  FreeModelData(&model_data);
  rtFree(weightspace);
  GeDbgDeInit();
  GeFinalize();
}

void StubExtendPartitionNormal(std::vector<uint8_t> &model) {
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

TEST_F(UtestGEExecutorTest, GeExecutorCaseModelExec) {
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
    }).AddPartition(PRE_MODEL_DESC_EXTEND, [](std::vector<uint8_t> &model) { StubExtendPartitionNormal(model); })
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

  GeInitialize();
  int test = 1;
  void *ptr = &test;
  EXPECT_CALL(RtStubMock::GetInstance(), dlopen(_,_)).Times(1).WillOnce(Return(ptr));
  EXPECT_CALL(RtStubMock::GetInstance(), dlsym(_,_)).Times(9).WillOnce(Return((void *)DbgInitMock))
            .WillOnce(Return((void *)DbgDeInitMock))
            .WillOnce(Return((void *)DbgNotifySetDeviceMock))
            .WillOnce(Return((void *)GetProfEnableMock))
            .WillOnce(Return((void *)DbgLoadModelPostProcessMock))
            .WillOnce(Return((void *)CreatModelDbgHandleMock))
            .WillOnce(Return((void *)LoadModelPreProcessMock))
            .WillOnce(Return((void *)StepIdConuterPlusMock))
            .WillOnce(Return((void *)DataFreeLoadDumpInfoMock));
  const char *configPath = nullptr;
  GeDbgInit(configPath);

  uint32_t model_id = 0;
  size_t mem_size = 0;
  size_t weight_size = 0;
  uint8_t *workspace = nullptr;
  uint8_t *weightspace = nullptr;
  (void)rtMalloc((void **)&workspace, sizeof(uint8_t) * 100, RT_MEMORY_DEFAULT, 0);
  (void)rtMalloc((void **)&weightspace, sizeof(uint8_t) * 103, RT_MEMORY_DEFAULT, 0);
  ModelData model_data;
  (void)memset_s(&model_data, sizeof(ModelData), 0, sizeof(ModelData));
  model_data.modelData = modelBuilder.ModelData();
  model_data.modelLen = modelBuilder.ModelLen();
  ModelInOutInfo info;
  EXPECT_EQ(GetModelDescInfoFromMem(&model_data, &info), SUCCESS);
  DestroyModelInOutInfo(&info);

  EXPECT_EQ(GetModelMemAndWeightSize(&model_data, &mem_size, &weight_size), SUCCESS);
  EXPECT_EQ(mem_size, 100);
  EXPECT_EQ(weight_size, 103);
  EXPECT_CALL(RtStubMock::GetInstance(), rtNanoModelLoad(_,_)).Times(1).WillOnce(Return(RT_ERROR_NONE));
  EXPECT_CALL(RtStubMock::GetInstance(), rtNanoModelDestroy(_)).Times(1).WillOnce(Return(RT_ERROR_NONE));
  EXPECT_EQ(GeLoadModelFromData(&model_id, &model_data), SUCCESS);
  ExecHandleDesc execDesc;
  execDesc.workPtr = workspace;
  execDesc.workSize = 100;
  InputData inputData;
  InitVector(&inputData.blobs, sizeof(DataBuffer));
  DataBlob inputBlob;
  DataBuffer input;
  uint32_t inputAddr = 0x1006;
  input.data = (void *)(&inputAddr);
  input.length = 50;
  inputBlob.dataBuffer = &input;
  EmplaceBackVector(&inputData.blobs, &inputBlob);
  OutputData outputData;
  InitVector(&outputData.blobs, sizeof(DataBuffer));
  outputData.io_addr = NULL;
  outputData.io_addr_host = NULL;
  outputData.ioa_size = 0;
  DataBlob outputBlob;
  DataBuffer output;
  uint32_t outputAddr = 0x1008;
  output.data = (void *)(&outputAddr);
  output.length = 100;
  outputBlob.dataBuffer = &output;
  EmplaceBackVector(&outputData.blobs, &outputBlob);
  EXPECT_CALL(RtStubMock::GetInstance(), rtStreamGetSqid(_,_)).Times(1).WillOnce(Return(RT_ERROR_NONE));
  EXPECT_CALL(RtStubMock::GetInstance(), rtNanoModelExecute(_)).Times(1).WillOnce(Return(RT_ERROR_NONE));
  EXPECT_EQ(ExecModel(model_id, &execDesc, true, &inputData, &outputData), SUCCESS);
  EXPECT_CALL(RtStubMock::GetInstance(), rtNanoModelExecute(_)).Times(1).WillOnce(Return(RT_ERROR_NONE));
  EXPECT_CALL(RtStubMock::GetInstance(), rtStreamGetSqid(_,_)).Times(1).WillOnce(Return(RT_ERROR_NONE));
  EXPECT_EQ(ExecModel(model_id, &execDesc, false, &inputData, &outputData), SUCCESS);
  EXPECT_CALL(RtStubMock::GetInstance(), rtNanoModelExecute(_)).Times(1).WillOnce(Return(1));
  EXPECT_CALL(RtStubMock::GetInstance(), rtStreamGetSqid(_,_)).Times(1).WillOnce(Return(RT_ERROR_NONE));
  EXPECT_EQ(ExecModel(model_id, &execDesc, true, &inputData, &outputData), 1);
  EXPECT_CALL(RtStubMock::GetInstance(), rtNanoModelExecute(_)).Times(1).WillOnce(Return(1));
  EXPECT_CALL(RtStubMock::GetInstance(), rtStreamGetSqid(_,_)).Times(1).WillOnce(Return(RT_ERROR_NONE));
  EXPECT_EQ(ExecModel(model_id, &execDesc, false, &inputData, &outputData), 1);
  EXPECT_CALL(RtStubMock::GetInstance(), rtNanoModelExecute(_)).Times(1).WillOnce(Return(RT_ERROR_NONE));
  EXPECT_CALL(RtStubMock::GetInstance(), rtStreamGetSqid(_,_)).Times(1).WillOnce(Return(RT_ERROR_NONE));
  EXPECT_EQ(ExecModel(model_id, &execDesc, false, &inputData, &outputData), SUCCESS);

  EXPECT_CALL(RtStubMock::GetInstance(), rtStreamGetSqid(_,_)).Times(1).WillOnce(Return(1));
  EXPECT_EQ(ExecModel(model_id, &execDesc, false, &inputData, &outputData), 1);

  OutputData outputData1;
  InitVector(&outputData1.blobs, sizeof(DataBuffer));
  outputData1.io_addr = NULL;
  outputData1.io_addr_host = NULL;
  outputData1.ioa_size = 1;
  EmplaceBackVector(&outputData1.blobs, &outputBlob);
  EXPECT_CALL(RtStubMock::GetInstance(), rtNanoModelExecute(_)).Times(1).WillOnce(Return(RT_ERROR_NONE));
  EXPECT_CALL(RtStubMock::GetInstance(), rtStreamGetSqid(_,_)).Times(1).WillOnce(Return(RT_ERROR_NONE));
  EXPECT_EQ(ExecModel(model_id, &execDesc, false, &inputData, &outputData1), SUCCESS);

  EXPECT_EQ(UnloadModel(model_id), SUCCESS);
  rtFree(workspace);
  rtFree(weightspace);
  DeInitVector(&inputData.blobs);
  DeInitVector(&outputData.blobs);
  DeInitVector(&outputData1.blobs);
  if (outputData.io_addr != NULL) {
    rtFree(outputData.io_addr);
  }
  if (outputData.io_addr_host != NULL) {
    mmFree(outputData.io_addr_host);
  }
  outputData.ioa_size = 0;
  if (outputData1.io_addr != NULL) {
    rtFree(outputData1.io_addr);
  }
  if (outputData1.io_addr_host != NULL) {
    mmFree(outputData1.io_addr_host);
  }
  outputData1.ioa_size = 0;

  EXPECT_CALL(RtStubMock::GetInstance(), dlclose(_)).Times(1).WillOnce(Return(0));
  GeDbgDeInit();
  GeFinalize();
}

TEST_F(UtestGEExecutorTest, GeExecutorCaseModelExecErr) {
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

  GeInitialize();
  uint32_t model_id = 0;
  size_t mem_size = 0;
  size_t weight_size = 0;
  uint8_t *workspace = nullptr;
  uint8_t *weightspace = nullptr;
  (void)rtMalloc((void **)&workspace, sizeof(uint8_t) * 100, RT_MEMORY_DEFAULT, 0);
  (void)rtMalloc((void **)&weightspace, sizeof(uint8_t) * 103, RT_MEMORY_DEFAULT, 0);
  ModelData model_data;
  (void)memset_s(&model_data, sizeof(ModelData), 0, sizeof(ModelData));
  model_data.modelData = modelBuilder.ModelData();
  model_data.modelLen = modelBuilder.ModelLen();
  ModelInOutInfo info;
  EXPECT_EQ(GetModelDescInfoFromMem(&model_data, &info), SUCCESS);
  DestroyModelInOutInfo(&info);

  EXPECT_EQ(GetModelMemAndWeightSize(&model_data, &mem_size, &weight_size), SUCCESS);
  EXPECT_EQ(mem_size, 100);
  EXPECT_EQ(weight_size, 103);
  EXPECT_CALL(RtStubMock::GetInstance(), rtNanoModelLoad(_,_)).Times(1).WillOnce(Return(RT_ERROR_NONE));
  EXPECT_CALL(RtStubMock::GetInstance(), rtNanoModelDestroy(_)).Times(1).WillOnce(Return(RT_ERROR_NONE));
  EXPECT_EQ(GeLoadModelFromData(&model_id, &model_data), SUCCESS);
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
  model_id = 1000;
  EXPECT_EQ(ExecModel(model_id, &execDesc, true, &inputData, &outputData), ACL_ERROR_GE_EXEC_MODEL_ID_INVALID);
  EXPECT_EQ(UnloadModel(model_id), ACL_ERROR_GE_EXEC_MODEL_ID_INVALID);
  rtFree(workspace);
  rtFree(weightspace);
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

TEST_F(UtestGEExecutorTest, GeExecutorCaseModelExecErr_InvalidInputNums) {
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
     ).AddInputDesc("op2", 10, FORMAT_CHWN, DT_FLOAT, dims1, dims2, shapeRange
     ).AddOutputDesc("op3", 8, FORMAT_CHWN, DT_FLOAT, dims1, dims2, shapeRange).Build();

  GeInitialize();
  uint32_t model_id = 0;
  uint8_t *workspace = nullptr;
  uint8_t *weightspace = nullptr;
  (void)rtMalloc((void **)&workspace, sizeof(uint8_t) * 100, RT_MEMORY_DEFAULT, 0);
  (void)rtMalloc((void **)&weightspace, sizeof(uint8_t) * 103, RT_MEMORY_DEFAULT, 0);
  ModelData model_data;
  (void)memset_s(&model_data, sizeof(ModelData), 0, sizeof(ModelData));
  model_data.modelData = modelBuilder.ModelData();
  model_data.modelLen = modelBuilder.ModelLen();
  EXPECT_CALL(RtStubMock::GetInstance(), rtNanoModelLoad(_,_)).Times(1).WillOnce(Return(RT_ERROR_NONE));
  EXPECT_CALL(RtStubMock::GetInstance(), rtNanoModelDestroy(_)).Times(1).WillOnce(Return(RT_ERROR_NONE));
  EXPECT_EQ(GeLoadModelFromData(&model_id, &model_data), SUCCESS);
  ExecHandleDesc execDesc;
  execDesc.workPtr = workspace;
  execDesc.workSize = 100;
  InputData inputData;
  InitVector(&inputData.blobs, sizeof(DataBuffer));
  DataBlob inputBlob;
  DataBuffer input;
  uint32_t inputAddr = 0x1006;
  input.data = (void *)(&inputAddr);
  input.length = 50;
  inputBlob.dataBuffer = &input;
  EmplaceBackVector(&inputData.blobs, &inputBlob);
  OutputData outputData;
  InitVector(&outputData.blobs, sizeof(DataBuffer));
  outputData.io_addr = NULL;
  outputData.io_addr_host = NULL;
  outputData.ioa_size = 0;
  DataBlob outputBlob;
  DataBuffer output;
  uint32_t outputAddr = 0x1008;
  output.data = (void *)(&outputAddr);
  output.length = 100;
  outputBlob.dataBuffer = &output;
  EmplaceBackVector(&outputData.blobs, &outputBlob);
  EXPECT_CALL(RtStubMock::GetInstance(), rtStreamGetSqid(_,_)).Times(1).WillOnce(Return(RT_ERROR_NONE));
  EXPECT_EQ(ExecModel(model_id, &execDesc, false, &inputData, &outputData), ACL_ERROR_GE_PARAM_INVALID);
  EXPECT_EQ(UnloadModel(model_id), SUCCESS);
  rtFree(workspace);
  rtFree(weightspace);
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

TEST_F(UtestGEExecutorTest, GeExecutorCaseModelExecErr_ExecModelFail) {
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
     ).AddInputDesc("op2", 10, FORMAT_CHWN, DT_FLOAT, dims1, dims2, shapeRange
     ).AddOutputDesc("op3", 8, FORMAT_CHWN, DT_FLOAT, dims1, dims2, shapeRange).Build();

  GeInitialize();
  uint32_t model_id = 0;
  uint8_t *workspace = nullptr;
  uint8_t *weightspace = nullptr;
  (void)rtMalloc((void **)&workspace, sizeof(uint8_t) * 100, RT_MEMORY_DEFAULT, 0);
  (void)rtMalloc((void **)&weightspace, sizeof(uint8_t) * 103, RT_MEMORY_DEFAULT, 0);
  ModelData model_data;
  (void)memset_s(&model_data, sizeof(ModelData), 0, sizeof(ModelData));
  model_data.modelData = modelBuilder.ModelData();
  model_data.modelLen = modelBuilder.ModelLen();
  EXPECT_CALL(RtStubMock::GetInstance(), rtNanoModelLoad(_,_)).Times(1).WillOnce(Return(RT_ERROR_NONE));
  EXPECT_CALL(RtStubMock::GetInstance(), rtNanoModelDestroy(_)).Times(1).WillOnce(Return(RT_ERROR_NONE));
  EXPECT_EQ(GeLoadModelFromData(&model_id, &model_data), SUCCESS);
  ExecHandleDesc execDesc;
  execDesc.workPtr = workspace;
  execDesc.workSize = 100;
  InputData inputData;
  InitVector(&inputData.blobs, sizeof(DataBuffer));
  DataBlob inputBlob1;
  DataBuffer input1;
  uint32_t inputAddr1 = 0x1006;
  input1.data = (void *)(&inputAddr1);
  input1.length = 50;
  inputBlob1.dataBuffer = &input1;
  DataBlob inputBlob2;
  DataBuffer input2;
  uint32_t inputAddr2 = 0x1010;
  input2.data = (void *)(&inputAddr2);
  input2.length = 50;
  inputBlob2.dataBuffer = &input2;
  EmplaceBackVector(&inputData.blobs, &inputBlob1);
  EmplaceBackVector(&inputData.blobs, &inputBlob2);

  OutputData outputData;
  InitVector(&outputData.blobs, sizeof(DataBuffer));
  outputData.io_addr = NULL;
  outputData.io_addr_host = NULL;
  outputData.ioa_size = 0;
  DataBlob outputBlob;
  DataBuffer output;
  uint32_t outputAddr = 0x1008;
  output.data = (void *)(&outputAddr);
  output.length = 100;
  outputBlob.dataBuffer = &output;
  EmplaceBackVector(&outputData.blobs, &outputBlob);
  EXPECT_CALL(RtStubMock::GetInstance(), rtStreamGetSqid(_,_)).Times(1).WillOnce(Return(RT_ERROR_NONE));
  EXPECT_EQ(ExecModel(model_id, &execDesc, false, &inputData, &outputData), ACL_ERROR_GE_MEMORY_ALLOCATION);
  EXPECT_EQ(UnloadModel(model_id), SUCCESS);
  rtFree(workspace);
  rtFree(weightspace);
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


TEST_F(UtestGEExecutorTest, GeExecutorCase_ModelExecuteInner) {
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
    }).AddInputDesc("op1", 10, FORMAT_CHWN, DT_FLOAT, dims1, dims2, shapeRange
     ).AddOutputDesc("op2", 8, FORMAT_CHWN, DT_FLOAT, dims1, dims2, shapeRange).Build();

  GeInitialize();
  int test = 1;
  void *ptr = &test;
  EXPECT_CALL(RtStubMock::GetInstance(), dlopen(_,_)).Times(1).WillOnce(Return(ptr));
  EXPECT_CALL(RtStubMock::GetInstance(), dlsym(_,_)).Times(9).WillOnce(Return((void *)DbgInitMock))
            .WillOnce(Return((void *)DbgDeInitMock))
            .WillOnce(Return((void *)DbgNotifySetDeviceMock))
            .WillOnce(Return((void *)GetProfEnableMock))
            .WillOnce(Return((void *)DbgLoadModelPostProcessMock))
            .WillOnce(Return((void *)CreatModelDbgHandleMock))
            .WillOnce(Return((void *)LoadModelPreProcessMock))
            .WillOnce(Return((void *)StepIdConuterPlusMock))
            .WillOnce(Return((void *)DataFreeLoadDumpInfoMock));
  const char *configPath = nullptr;
  GeDbgInit(configPath);

  uint32_t model_id = 0;
  size_t mem_size = 0;
  size_t weight_size = 0;
  uint8_t *workspace = nullptr;
  uint8_t *weightspace = nullptr;
  (void)rtMalloc((void **)&workspace, sizeof(uint8_t) * 100, RT_MEMORY_DEFAULT, 0);
  (void)rtMalloc((void **)&weightspace, sizeof(uint8_t) * 103, RT_MEMORY_DEFAULT, 0);
  ModelData model_data;
  (void)memset_s(&model_data, sizeof(ModelData), 0, sizeof(ModelData));
  model_data.modelData = modelBuilder.ModelData();
  model_data.modelLen = modelBuilder.ModelLen();
  ModelInOutInfo info;
  EXPECT_EQ(GetModelDescInfoFromMem(&model_data, &info), SUCCESS);
  DestroyModelInOutInfo(&info);

  EXPECT_EQ(GetModelMemAndWeightSize(&model_data, &mem_size, &weight_size), SUCCESS);
  EXPECT_EQ(mem_size, 100);
  EXPECT_EQ(weight_size, 103);
  EXPECT_CALL(RtStubMock::GetInstance(), rtNanoModelLoad(_,_)).Times(1).WillOnce(Return(RT_ERROR_NONE));
  EXPECT_CALL(RtStubMock::GetInstance(), rtNanoModelDestroy(_)).Times(1).WillOnce(Return(RT_ERROR_NONE));
  EXPECT_EQ(GeLoadModelFromData(&model_id, &model_data), SUCCESS);
  ExecHandleDesc execDesc;
  execDesc.workPtr = workspace;
  execDesc.workSize = 100;
  InputData inputData;
  InitVector(&inputData.blobs, sizeof(DataBuffer));
  DataBlob inputBlob;
  DataBuffer input;
  uint32_t inputAddr = 0x1006;
  input.data = (void *)(&inputAddr);
  input.length = 50;
  inputBlob.dataBuffer = &input;
  EmplaceBackVector(&inputData.blobs, &inputBlob);
  OutputData outputData;
  InitVector(&outputData.blobs, sizeof(DataBuffer));
  outputData.io_addr = NULL;
  outputData.io_addr_host = NULL;
  outputData.ioa_size = 0;
  DataBlob outputBlob;
  DataBuffer output;
  uint32_t outputAddr = 0x1008;
  output.data = (void *)(&outputAddr);
  output.length = 100;
  outputBlob.dataBuffer = &output;
  EmplaceBackVector(&outputData.blobs, &outputBlob);
  EXPECT_CALL(RtStubMock::GetInstance(), rtStreamGetSqid(_,_)).Times(2).WillOnce(Return(RT_ERROR_NONE)).WillOnce(Return(RT_ERROR_NONE));
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmMalloc(_)).Times(2).WillOnce(Invoke(mmMalloc_Abnormal_Invoke)).WillOnce(Invoke(mmMalloc_Normal_Invoke));
  EXPECT_EQ(ExecModel(model_id, &execDesc, false, &inputData, &outputData), ACL_ERROR_GE_MEMORY_ALLOCATION);

  EXPECT_CALL(RtStubMock::GetInstance(), rtMemcpy(_,_,_,_,_)).Times(1).WillOnce(Return(ACL_ERROR_GE_MEMORY_ALLOCATION));
  EXPECT_EQ(ExecModel(model_id, &execDesc, false, &inputData, &outputData), ACL_ERROR_GE_MEMORY_ALLOCATION);

  EXPECT_EQ(UnloadModel(model_id), SUCCESS);
  rtFree(workspace);
  rtFree(weightspace);
  DeInitVector(&inputData.blobs);
  DeInitVector(&outputData.blobs);;
  if (outputData.io_addr != NULL) {
    rtFree(outputData.io_addr);
  }
  if (outputData.io_addr_host != NULL) {
    mmFree(outputData.io_addr_host);
  }
  outputData.ioa_size = 0;

  EXPECT_CALL(RtStubMock::GetInstance(), dlclose(_)).Times(1).WillOnce(Return(0));
  GeDbgDeInit();
  GeFinalize();
}

TEST_F(UtestGEExecutorTest, GeExecutorCase1ModelDescInfo) {
  ModelInOutInfo info;
  uint32_t model_id = 0U;
  EXPECT_NE(GetModelDescInfo(model_id, &info), SUCCESS);
  GeInitialize();
  EXPECT_NE(GetModelDescInfo(model_id, &info), SUCCESS);
  GeFinalize();
}


TEST_F(UtestGEExecutorTest, GeExecutorCase2LoadModelFromData) {
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
     ).AddOutputDesc("op2", 8, FORMAT_CHWN, DT_FLOAT, dims1, dims2, shapeRange).BuildMdlErr();

  GeInitialize();
  uint32_t model_id = 0;
  ModelData model_data;
  (void)memset_s(&model_data, sizeof(ModelData), 0, sizeof(ModelData));
  model_data.modelData = modelBuilder.ModelData();
  model_data.modelLen = modelBuilder.ModelLen();
  EXPECT_NE(GeLoadModelFromData(&model_id, &model_data), SUCCESS);
  GeFinalize();
}

Status LoadModelPreProcessMock1(void *taskDescBaseAddr, char *modelName, void *dbgHandle) {
  (void)taskDescBaseAddr;
  (void)modelName;
  (void)dbgHandle;
  return ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
}

Status DbgLoadModelPostProcessMock1(uint32_t modelId, char *om, uint64_t *stepIdAddr, void *dbgHandle) {
  (void)modelId;
  (void)om;
  (void)stepIdAddr;
  (void)dbgHandle;
  return ACL_ERROR_GE_MEMORY_ALLOCATION;
}

TEST_F(UtestGEExecutorTest, GeExecutorCase1LoadModelFromDataErr) {
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

  GeInitialize();
  int test = 1;
  void *ptr = &test;
  EXPECT_CALL(RtStubMock::GetInstance(), dlopen(_,_)).Times(1).WillOnce(Return(ptr));
  EXPECT_CALL(RtStubMock::GetInstance(), dlclose(_)).Times(1).WillOnce(Return(0));
  EXPECT_CALL(RtStubMock::GetInstance(), dlsym(_,_)).Times(9).WillOnce(Return((void *)DbgInitMock))
            .WillOnce(Return((void *)DbgDeInitMock))
            .WillOnce(Return((void *)DbgNotifySetDeviceMock))
            .WillOnce(Return((void *)GetProfEnableMock))
            .WillOnce(Return((void *)DbgLoadModelPostProcessMock))
            .WillOnce(Return((void *)CreatModelDbgHandleMock))
            .WillOnce(Return((void *)LoadModelPreProcessMock1))
            .WillOnce(Return((void *)StepIdConuterPlusMock))
            .WillOnce(Return((void *)DataFreeLoadDumpInfoMock));
  const char *configPath = nullptr;
  GeDbgInit(configPath);
  uint32_t model_id = 0;
  size_t mem_size = 0;
  size_t weight_size = 0;
  uint8_t *weightspace = nullptr;
  (void)rtMalloc((void **)&weightspace, sizeof(uint8_t) * 103, RT_MEMORY_DEFAULT, 0);
  ModelData model_data;
  (void)memset_s(&model_data, sizeof(ModelData), 0, sizeof(ModelData));
  model_data.modelData = modelBuilder.ModelData();
  model_data.modelLen = modelBuilder.ModelLen();
  EXPECT_EQ(GetModelMemAndWeightSize(&model_data, &mem_size, &weight_size), SUCCESS);
  EXPECT_EQ(mem_size, 100);
  EXPECT_EQ(weight_size, 103);
  EXPECT_EQ(GeLoadModelFromData(&model_id, &model_data), ACL_ERROR_GE_MEMORY_OPERATE_FAILED);
  EXPECT_EQ(UnloadModel(model_id), ACL_ERROR_GE_EXEC_MODEL_ID_INVALID);
  rtFree(weightspace);
  GeDbgDeInit();
  GeFinalize();
}

TEST_F(UtestGEExecutorTest, GeExecutorCase2LoadModelFromDataErr) {
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

  GeInitialize();
  int test = 1;
  void *ptr = &test;
  EXPECT_CALL(RtStubMock::GetInstance(), dlopen(_,_)).Times(1).WillOnce(Return(ptr));
  EXPECT_CALL(RtStubMock::GetInstance(), dlclose(_)).Times(1).WillOnce(Return(0));
  EXPECT_CALL(RtStubMock::GetInstance(), dlsym(_,_)).Times(9).WillOnce(Return((void *)DbgInitMock))
            .WillOnce(Return((void *)DbgDeInitMock))
            .WillOnce(Return((void *)DbgNotifySetDeviceMock))
            .WillOnce(Return((void *)GetProfEnableMock))
            .WillOnce(Return((void *)DbgLoadModelPostProcessMock1))
            .WillOnce(Return((void *)CreatModelDbgHandleMock))
            .WillOnce(Return((void *)LoadModelPreProcessMock))
            .WillOnce(Return((void *)StepIdConuterPlusMock))
            .WillOnce(Return((void *)DataFreeLoadDumpInfoMock));
  const char *configPath = nullptr;
  GeDbgInit(configPath);
  uint32_t model_id = 0;
  size_t mem_size = 0;
  size_t weight_size = 0;
  uint8_t *weightspace = nullptr;
  (void)rtMalloc((void **)&weightspace, sizeof(uint8_t) * 103, RT_MEMORY_DEFAULT, 0);
  ModelData model_data;
  (void)memset_s(&model_data, sizeof(ModelData), 0, sizeof(ModelData));
  model_data.modelData = modelBuilder.ModelData();
  model_data.modelLen = modelBuilder.ModelLen();
  EXPECT_EQ(GetModelMemAndWeightSize(&model_data, &mem_size, &weight_size), SUCCESS);
  EXPECT_EQ(mem_size, 100);
  EXPECT_EQ(weight_size, 103);
  EXPECT_CALL(RtStubMock::GetInstance(), rtNanoModelLoad(_,_)).Times(AnyNumber()).WillOnce(Return(RT_ERROR_NONE));
  EXPECT_CALL(RtStubMock::GetInstance(), rtNanoModelDestroy(_)).Times(AnyNumber()).WillOnce(Return(RT_ERROR_NONE));
  EXPECT_EQ(GeLoadModelFromData(&model_id, &model_data), ACL_ERROR_GE_MEMORY_ALLOCATION);
  EXPECT_EQ(UnloadModel(model_id), ACL_ERROR_GE_EXEC_MODEL_ID_INVALID);
  rtFree(weightspace);
  GeDbgDeInit();
  GeFinalize();
}

TEST_F(UtestGEExecutorTest, GeDbgInitErrorCase) {
  void *ptr = nullptr;
  EXPECT_CALL(RtStubMock::GetInstance(), dlopen(_,_)).Times(1).WillOnce(Return(ptr));
  EXPECT_CALL(RtStubMock::GetInstance(), dlerror()).Times(1).WillOnce(Return(nullptr));
  const char *configPath = nullptr;
  EXPECT_EQ(GeDbgInit(configPath), ACL_ERROR_GE_PARAM_INVALID);
}

TEST_F(UtestGEExecutorTest, GeFreeLoadDumpInfoCase) {
  int test = 1;
  void *ptr = &test;
  EXPECT_CALL(RtStubMock::GetInstance(), dlopen(_,_)).Times(1).WillOnce(Return(ptr));
  EXPECT_CALL(RtStubMock::GetInstance(), dlclose(_)).Times(1).WillOnce(Return(0));
  EXPECT_CALL(RtStubMock::GetInstance(), dlsym(_,_)).Times(9).WillOnce(Return((void *)DbgInitMock))
            .WillOnce(Return((void *)DbgDeInitMock))
            .WillOnce(Return((void *)DbgNotifySetDeviceMock))
            .WillOnce(Return((void *)GetProfEnableMock))
            .WillOnce(Return((void *)DbgLoadModelPostProcessMock))
            .WillOnce(Return((void *)CreatModelDbgHandleMock))
            .WillOnce(Return((void *)LoadModelPreProcessMock))
            .WillOnce(Return((void *)StepIdConuterPlusMock))
            .WillOnce(Return((void *)DataFreeLoadDumpInfoMock));
  const char *configPath = nullptr;
  Status ret = GeDbgInit(configPath);
  EXPECT_EQ(ret, SUCCESS);
  uint32_t chipId = 0;
  uint32_t deviceId = 0;
  ret = GeNofifySetDevice(chipId, deviceId);
  EXPECT_EQ(ret, SUCCESS);
  void *dbgHandle = nullptr;
  DataFreeLoadDumpInfo(dbgHandle);
  ret = GeDbgDeInit();
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestGEExecutorTest, LoadDataFromFile_fileLen_Abnormal) {
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
  (void)memset_s(&modelData , sizeof(ModelData), 0, sizeof(ModelData));
  EXPECT_EQ(LoadDataFromFile(fileName.c_str(), &modelData), ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID);
  GeFinalize();
}

TEST_F(UtestGEExecutorTest, LoadDataFromFile_file_path_Abnormal) {
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
  (void)memset_s(&modelData , sizeof(ModelData), 0, sizeof(ModelData));
  EXPECT_CALL(RtStubMock::GetInstance(), access(_,_)).Times(1).WillOnce(Return(1));
  EXPECT_EQ(LoadDataFromFile(fileName.c_str(), &modelData), ACL_ERROR_GE_EXEC_MODEL_PATH_INVALID);
  GeFinalize();
}

TEST_F(UtestGEExecutorTest, PRE_MODEL_DESC_rtMalloc_Abnormal) {
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

  std::copy_n((uint8_t*)(&desc), MALLOC_SIZE_INVALID, std::back_inserter(model));
    }).Build();

  // write modelBuilder to file
  std::string fileName;
  fileName.append("./test").append(".exeom");
  FILE *f = fopen(fileName.c_str(), "wb+");
  fwrite(modelBuilder.ModelData(), modelBuilder.ModelLen(), 1, f);
  fclose(f);

  GeInitialize();
  ModelData modelData;
  (void)memset_s(&modelData , sizeof(ModelData), 0, sizeof(ModelData));
  EXPECT_CALL(RtStubMock::GetInstance(), access(_,_)).Times(1).WillOnce(Return(0));
  EXPECT_EQ(LoadDataFromFile(fileName.c_str(), &modelData), SUCCESS);
  uint32_t modelId = 0;
  EXPECT_EQ(GeLoadModelFromData(&modelId, &modelData), ACL_ERROR_GE_MEMORY_OPERATE_FAILED);
  FreeModelData(&modelData);
  GeFinalize();
}

TEST_F(UtestGEExecutorTest, PRE_MODEL_DESC_rtMemcpy_Abnormal) {
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

  std::copy_n((uint8_t*)(&desc), MEMCPY_SIZE_INVALID, std::back_inserter(model));
    }).Build();

  // write modelBuilder to file
  std::string fileName;
  fileName.append("./test").append(".exeom");
  FILE *f = fopen(fileName.c_str(), "wb+");
  fwrite(modelBuilder.ModelData(), modelBuilder.ModelLen(), 1, f);
  fclose(f);

  GeInitialize();
  ModelData modelData;
  (void)memset_s(&modelData , sizeof(ModelData), 0, sizeof(ModelData));
  EXPECT_CALL(RtStubMock::GetInstance(), access(_,_)).Times(1).WillOnce(Return(0));
  EXPECT_EQ(LoadDataFromFile(fileName.c_str(), &modelData), SUCCESS);
  uint32_t modelId = 0;
  EXPECT_EQ(GeLoadModelFromData(&modelId, &modelData), ACL_ERROR_GE_MEMORY_OPERATE_FAILED);
  FreeModelData(&modelData);
  GeFinalize();
}

TEST_F(UtestGEExecutorTest, DYNAMIC_TASK_DESC_rtMalloc_Abnormal) {
  ExeModelBuilder modelBuilder;
  std::vector<uint64_t> dims1 = {1, 2, 3};
  std::vector<uint64_t> dims2 = {};
  std::vector<std::pair<uint64_t, uint64_t>> shapeRange = {{1, 2}, {2, 5}};
  modelBuilder.AddPartition(DYNAMIC_TASK_DESC, [] (std::vector<uint8_t> &model) {
      GenerateEleForModel(0, MALLOC_SIZE_INVALID, model);
    }).Build();

  // write modelBuilder to file
  std::string fileName;
  fileName.append("./test").append(".exeom");
  FILE *f = fopen(fileName.c_str(), "wb+");
  fwrite(modelBuilder.ModelData(), modelBuilder.ModelLen(), 1, f);
  fclose(f);

  GeInitialize();
  ModelData modelData;
  (void)memset_s(&modelData , sizeof(ModelData), 0, sizeof(ModelData));
  EXPECT_CALL(RtStubMock::GetInstance(), access(_,_)).Times(1).WillOnce(Return(0));
  EXPECT_EQ(LoadDataFromFile(fileName.c_str(), &modelData), SUCCESS);
  uint32_t modelId = 0;
  EXPECT_EQ(GeLoadModelFromData(&modelId, &modelData), ACL_ERROR_GE_MEMORY_OPERATE_FAILED);
  FreeModelData(&modelData);
  GeFinalize();
}

TEST_F(UtestGEExecutorTest, DYNAMIC_TASK_DESC_rtMemcpy_Abnormal) {
  ExeModelBuilder modelBuilder;
  std::vector<uint64_t> dims1 = {1, 2, 3};
  std::vector<uint64_t> dims2 = {};
  std::vector<std::pair<uint64_t, uint64_t>> shapeRange = {{1, 2}, {2, 5}};
  modelBuilder.AddPartition(DYNAMIC_TASK_DESC, [] (std::vector<uint8_t> &model) {
      GenerateEleForModel(0, MEMCPY_SIZE_INVALID, model);
    }).Build();

  // write modelBuilder to file
  std::string fileName;
  fileName.append("./test").append(".exeom");
  FILE *f = fopen(fileName.c_str(), "wb+");
  fwrite(modelBuilder.ModelData(), modelBuilder.ModelLen(), 1, f);
  fclose(f);

  GeInitialize();
  ModelData modelData;
  (void)memset_s(&modelData , sizeof(ModelData), 0, sizeof(ModelData));
  EXPECT_CALL(RtStubMock::GetInstance(), access(_,_)).Times(1).WillOnce(Return(0));
  EXPECT_EQ(LoadDataFromFile(fileName.c_str(), &modelData), SUCCESS);
  uint32_t modelId = 0;
  EXPECT_EQ(GeLoadModelFromData(&modelId, &modelData), ACL_ERROR_GE_MEMORY_OPERATE_FAILED);
  FreeModelData(&modelData);
  GeFinalize();
}

TEST_F(UtestGEExecutorTest, TBE_KERNELS_rtMalloc_Abnormal) {
  ExeModelBuilder modelBuilder;
  std::vector<uint64_t> dims1 = {1, 2, 3};
  std::vector<uint64_t> dims2 = {};
  std::vector<std::pair<uint64_t, uint64_t>> shapeRange = {{1, 2}, {2, 5}};
  modelBuilder.AddPartition(TBE_KERNELS, [] (std::vector<uint8_t> &model) {
      GenerateEleForModel(101, MALLOC_SIZE_INVALID, model);
    }).Build();

  // write modelBuilder to file
  std::string fileName;
  fileName.append("./test").append(".exeom");
  FILE *f = fopen(fileName.c_str(), "wb+");
  fwrite(modelBuilder.ModelData(), modelBuilder.ModelLen(), 1, f);
  fclose(f);

  GeInitialize();
  ModelData modelData;
  (void)memset_s(&modelData , sizeof(ModelData), 0, sizeof(ModelData));
  EXPECT_CALL(RtStubMock::GetInstance(), access(_,_)).Times(1).WillOnce(Return(0));
  EXPECT_EQ(LoadDataFromFile(fileName.c_str(), &modelData), SUCCESS);
  uint32_t modelId = 0;
  EXPECT_EQ(GeLoadModelFromData(&modelId, &modelData), ACL_ERROR_GE_MEMORY_OPERATE_FAILED);
  FreeModelData(&modelData);
  GeFinalize();
}

TEST_F(UtestGEExecutorTest, TBE_KERNELS_rtMemcpy_Abnormal) {
  ExeModelBuilder modelBuilder;
  std::vector<uint64_t> dims1 = {1, 2, 3};
  std::vector<uint64_t> dims2 = {};
  std::vector<std::pair<uint64_t, uint64_t>> shapeRange = {{1, 2}, {2, 5}};
  modelBuilder.AddPartition(TBE_KERNELS, [] (std::vector<uint8_t> &model) {
      GenerateEleForModel(101, MEMCPY_SIZE_INVALID, model);
    }).Build();

  // write modelBuilder to file
  std::string fileName;
  fileName.append("./test").append(".exeom");
  FILE *f = fopen(fileName.c_str(), "wb+");
  fwrite(modelBuilder.ModelData(), modelBuilder.ModelLen(), 1, f);
  fclose(f);

  GeInitialize();
  ModelData modelData;
  (void)memset_s(&modelData , sizeof(ModelData), 0, sizeof(ModelData));
  EXPECT_CALL(RtStubMock::GetInstance(), access(_,_)).Times(1).WillOnce(Return(0));
  EXPECT_EQ(LoadDataFromFile(fileName.c_str(), &modelData), SUCCESS);
  uint32_t modelId = 0;
  EXPECT_EQ(GeLoadModelFromData(&modelId, &modelData), ACL_ERROR_GE_MEMORY_OPERATE_FAILED);
  FreeModelData(&modelData);
  GeFinalize();
}

TEST_F(UtestGEExecutorTest, TASK_PARAM_rtMalloc_Abnormal) {
  ExeModelBuilder modelBuilder;
  std::vector<uint64_t> dims1 = {1, 2, 3};
  std::vector<uint64_t> dims2 = {};
  std::vector<std::pair<uint64_t, uint64_t>> shapeRange = {{1, 2}, {2, 5}};
  modelBuilder.AddPartition(TASK_PARAM, [] (std::vector<uint8_t> &model) {
      GenerateEleForModel(201, MALLOC_SIZE_INVALID, model);
    }).Build();

  // write modelBuilder to file
  std::string fileName;
  fileName.append("./test").append(".exeom");
  FILE *f = fopen(fileName.c_str(), "wb+");
  fwrite(modelBuilder.ModelData(), modelBuilder.ModelLen(), 1, f);
  fclose(f);

  GeInitialize();
  ModelData modelData;
  (void)memset_s(&modelData , sizeof(ModelData), 0, sizeof(ModelData));
  EXPECT_CALL(RtStubMock::GetInstance(), access(_,_)).Times(1).WillOnce(Return(0));
  EXPECT_EQ(LoadDataFromFile(fileName.c_str(), &modelData), SUCCESS);
  uint32_t modelId = 0;
  EXPECT_EQ(GeLoadModelFromData(&modelId, &modelData), ACL_ERROR_GE_MEMORY_OPERATE_FAILED);
  FreeModelData(&modelData);
  GeFinalize();
}

TEST_F(UtestGEExecutorTest, TASK_PARAM_rtMemcpy_Abnormal) {
  ExeModelBuilder modelBuilder;
  std::vector<uint64_t> dims1 = {1, 2, 3};
  std::vector<uint64_t> dims2 = {};
  std::vector<std::pair<uint64_t, uint64_t>> shapeRange = {{1, 2}, {2, 5}};
  modelBuilder.AddPartition(TASK_PARAM, [] (std::vector<uint8_t> &model) {
      GenerateEleForModel(201, MEMCPY_SIZE_INVALID, model);
    }).Build();

  // write modelBuilder to file
  std::string fileName;
  fileName.append("./test").append(".exeom");
  FILE *f = fopen(fileName.c_str(), "wb+");
  fwrite(modelBuilder.ModelData(), modelBuilder.ModelLen(), 1, f);
  fclose(f);

  GeInitialize();
  ModelData modelData;
  (void)memset_s(&modelData , sizeof(ModelData), 0, sizeof(ModelData));
  EXPECT_CALL(RtStubMock::GetInstance(), access(_,_)).Times(1).WillOnce(Return(0));
  EXPECT_EQ(LoadDataFromFile(fileName.c_str(), &modelData), SUCCESS);
  uint32_t modelId = 0;
  EXPECT_EQ(GeLoadModelFromData(&modelId, &modelData), ACL_ERROR_GE_MEMORY_OPERATE_FAILED);
  FreeModelData(&modelData);
  GeFinalize();
}

TEST_F(UtestGEExecutorTest, WEIGHTS_DATA_rtMalloc_Abnormal) {
  ExeModelBuilder modelBuilder;
  std::vector<uint64_t> dims1 = {1, 2, 3};
  std::vector<uint64_t> dims2 = {};
  std::vector<std::pair<uint64_t, uint64_t>> shapeRange = {{1, 2}, {2, 5}};
  modelBuilder.AddPartition(WEIGHTS_DATA, [] (std::vector<uint8_t> &model) {
      GenerateEleForModel(31, MALLOC_SIZE_INVALID, model);
    }).Build();

  // write modelBuilder to file
  std::string fileName;
  fileName.append("./test").append(".exeom");
  FILE *f = fopen(fileName.c_str(), "wb+");
  fwrite(modelBuilder.ModelData(), modelBuilder.ModelLen(), 1, f);
  fclose(f);

  GeInitialize();
  ModelData modelData;
  (void)memset_s(&modelData , sizeof(ModelData), 0, sizeof(ModelData));
  EXPECT_CALL(RtStubMock::GetInstance(), access(_,_)).Times(1).WillOnce(Return(0));
  EXPECT_EQ(LoadDataFromFile(fileName.c_str(), &modelData), SUCCESS);
  uint32_t modelId = 0;
  EXPECT_EQ(GeLoadModelFromData(&modelId, &modelData), ACL_ERROR_GE_MEMORY_OPERATE_FAILED);
  FreeModelData(&modelData);
  GeFinalize();
}

TEST_F(UtestGEExecutorTest, WEIGHTS_DATA_rtMemcpy_Abnormal) {
  ExeModelBuilder modelBuilder;
  std::vector<uint64_t> dims1 = {1, 2, 3};
  std::vector<uint64_t> dims2 = {};
  std::vector<std::pair<uint64_t, uint64_t>> shapeRange = {{1, 2}, {2, 5}};
  modelBuilder.AddPartition(WEIGHTS_DATA, [] (std::vector<uint8_t> &model) {
      GenerateEleForModel(31, MEMCPY_SIZE_INVALID, model);
    }).Build();

  // write modelBuilder to file
  std::string fileName;
  fileName.append("./test").append(".exeom");
  FILE *f = fopen(fileName.c_str(), "wb+");
  fwrite(modelBuilder.ModelData(), modelBuilder.ModelLen(), 1, f);
  fclose(f);

  GeInitialize();
  ModelData modelData;
  (void)memset_s(&modelData , sizeof(ModelData), 0, sizeof(ModelData));
  EXPECT_CALL(RtStubMock::GetInstance(), access(_,_)).Times(1).WillOnce(Return(0));
  EXPECT_EQ(LoadDataFromFile(fileName.c_str(), &modelData), SUCCESS);
  uint32_t modelId = 0;
  EXPECT_EQ(GeLoadModelFromData(&modelId, &modelData), ACL_ERROR_GE_MEMORY_OPERATE_FAILED);
  FreeModelData(&modelData);
  GeFinalize();
}

TEST_F(UtestGEExecutorTest, STATIC_TASK_DESC_rtMalloc_Abnormal) {
  ExeModelBuilder modelBuilder;
  std::vector<uint64_t> dims1 = {1, 2, 3};
  std::vector<uint64_t> dims2 = {};
  std::vector<std::pair<uint64_t, uint64_t>> shapeRange = {{1, 2}, {2, 5}};
  modelBuilder.AddPartition(STATIC_TASK_DESC, [] (std::vector<uint8_t> &model) {
      GenerateEleForModel(45, MALLOC_SIZE_INVALID, model);
    }).Build();

  // write modelBuilder to file
  std::string fileName;
  fileName.append("./test").append(".exeom");
  FILE *f = fopen(fileName.c_str(), "wb+");
  fwrite(modelBuilder.ModelData(), modelBuilder.ModelLen(), 1, f);
  fclose(f);

  GeInitialize();
  ModelData modelData;
  (void)memset_s(&modelData , sizeof(ModelData), 0, sizeof(ModelData));
  EXPECT_CALL(RtStubMock::GetInstance(), access(_,_)).Times(1).WillOnce(Return(0));
  EXPECT_EQ(LoadDataFromFile(fileName.c_str(), &modelData), SUCCESS);
  uint32_t modelId = 0;
  EXPECT_EQ(GeLoadModelFromData(&modelId, &modelData), ACL_ERROR_GE_MEMORY_OPERATE_FAILED);
  FreeModelData(&modelData);
  GeFinalize();
}

TEST_F(UtestGEExecutorTest, STATIC_TASK_DESC_rtMemcpy_Abnormal) {
  ExeModelBuilder modelBuilder;
  std::vector<uint64_t> dims1 = {1, 2, 3};
  std::vector<uint64_t> dims2 = {};
  std::vector<std::pair<uint64_t, uint64_t>> shapeRange = {{1, 2}, {2, 5}};
  modelBuilder.AddPartition(STATIC_TASK_DESC, [] (std::vector<uint8_t> &model) {
      GenerateEleForModel(45, MEMCPY_SIZE_INVALID, model);
    }).Build();

  // write modelBuilder to file.
  std::string fileName;
  fileName.append("./test").append(".exeom");
  FILE *f = fopen(fileName.c_str(), "wb+");
  fwrite(modelBuilder.ModelData(), modelBuilder.ModelLen(), 1, f);
  fclose(f);

  GeInitialize();
  ModelData modelData;
  (void)memset_s(&modelData , sizeof(ModelData), 0, sizeof(ModelData));
  EXPECT_CALL(RtStubMock::GetInstance(), access(_,_)).Times(AnyNumber()).WillOnce(Return(0));
  EXPECT_EQ(LoadDataFromFile(fileName.c_str(), &modelData), SUCCESS);
  uint32_t modelId = 0;
  EXPECT_EQ(GeLoadModelFromData(&modelId, &modelData), ACL_ERROR_GE_MEMORY_OPERATE_FAILED);
  FreeModelData(&modelData);
  GeFinalize();
}

TEST_F(UtestGEExecutorTest, GeExecutorCaseGetPartitionSize) {
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
  GePartitionSize parSize;
  EXPECT_EQ(GetPartitionSize(fileName.c_str(), &parSize), SUCCESS);
  EXPECT_EQ(parSize.modelDescSize, 26);
  EXPECT_EQ(parSize.dynamicTaskSize, 100);
  EXPECT_EQ(parSize.kernelSize, 101);
  EXPECT_EQ(parSize.kernelArgsSize, 102);
  EXPECT_EQ(parSize.weightSize, 103);
  EXPECT_EQ(parSize.workSize, 100);
  EXPECT_EQ(parSize.staticTaskSize, 301);

  ModelData model_data;
  (void)memset_s(&model_data, sizeof(ModelData), 0, sizeof(ModelData));
  ModelInOutInfo info;
  EXPECT_EQ(LoadDataFromFile(fileName.c_str(), &model_data), SUCCESS);
  EXPECT_EQ(GetModelDescInfoFromMem(&model_data, &info), SUCCESS);
  DestroyModelInOutInfo(&info);
  FreeModelData(&model_data);
  GeFinalize();
}

TEST_F(UtestGEExecutorTest, GeExecutorNoPartition) {
  ExeModelBuilder modelBuilder;
  modelBuilder.AddPartition(DYNAMIC_TASK_DESC, [] (std::vector<uint8_t> &model) {
      GenerateEleForModel(0, 100, model);
    }).Build();

  // write modelBuilder to file
  std::string fileName;
  fileName.append("./test").append(".exeom");
  FILE *f = fopen(fileName.c_str(), "wb+");
  fwrite(modelBuilder.ModelData(), modelBuilder.ModelLen(), 1, f);
  fclose(f);

  GeInitialize();
  ModelData modelData;
  (void)memset_s(&modelData , sizeof(ModelData), 0, sizeof(ModelData));
  ModelInOutInfo info;
  EXPECT_EQ(LoadDataFromFile(fileName.c_str(), &modelData), SUCCESS);
  size_t workSize = 0;
  size_t weightSize = 0;
  EXPECT_EQ(GetModelMemAndWeightSize(&modelData, &workSize, &weightSize), ACL_ERROR_GE_INTERNAL_ERROR);
  EXPECT_EQ(GetModelDescInfoFromMem(&modelData, &info), ACL_ERROR_GE_INTERNAL_ERROR);
  FreeModelData(&modelData);
  GeFinalize();
}

TEST_F(UtestGEExecutorTest, GeExecutorPartitionSizeInvalid) {
  ExeModelBuilder modelBuilder;
  modelBuilder.AddPartition(DYNAMIC_TASK_DESC, [] (std::vector<uint8_t> &model) {
    (void)model;
    }).Build();

  // write modelBuilder to file
  std::string fileName;
  fileName.append("./test").append(".exeom");
  FILE *f = fopen(fileName.c_str(), "wb+");
  fwrite(modelBuilder.ModelData(), modelBuilder.ModelLen(), 1, f);
  fclose(f);

  GeInitialize();
  ModelData modelData;
  (void)memset_s(&modelData , sizeof(ModelData), 0, sizeof(ModelData));
  ModelInOutInfo info;
  EXPECT_EQ(LoadDataFromFile(fileName.c_str(), &modelData), SUCCESS);
  size_t workSize = 0;
  size_t weightSize = 0;
  EXPECT_EQ(GetModelMemAndWeightSize(&modelData, &workSize, &weightSize), ACL_ERROR_GE_INTERNAL_ERROR);
  GePartitionSize parSize;
  EXPECT_EQ(GetModelPartitionSize(&modelData, &parSize), ACL_ERROR_GE_INTERNAL_ERROR);
  EXPECT_EQ(GetModelDescInfoFromMem(&modelData, &info), ACL_ERROR_GE_INTERNAL_ERROR);
  FreeModelData(&modelData);
  ModelData modelData2;
  (void)memset_s(&modelData2 , sizeof(ModelData), 0, sizeof(ModelData));
  EXPECT_EQ(GetModelPartitionSize(&modelData2, &parSize), ACL_ERROR_GE_INTERNAL_ERROR);
  GeFinalize();
}

TEST_F(UtestGEExecutorTest, CheckLenValid_tableLen_Abnormal) {
  ModelFileHeader head;
  (void)memset_s(&head , sizeof(ModelFileHeader), 0, sizeof(ModelFileHeader));
  head.magic = MODEL_FILE_MAGIC_NUM;

  std::string fileName;
  fileName.append("./test").append(".exeom");
  FILE *f = fopen(fileName.c_str(), "wb+");
  fwrite(&head, sizeof(ModelFileHeader) + 1, 1, f);
  fclose(f);

  GeInitialize();
  ModelData modelData;
  (void)memset_s(&modelData , sizeof(ModelData), 0, sizeof(ModelData));
  EXPECT_CALL(RtStubMock::GetInstance(), access(_,_)).Times(AnyNumber()).WillOnce(Return(0));
  EXPECT_EQ(LoadDataFromFile(fileName.c_str(), &modelData), SUCCESS);
  uint32_t modelId = 0;
  EXPECT_EQ(GeLoadModelFromData(&modelId, &modelData), ACL_ERROR_GE_LOAD_MODEL);
  FreeModelData(&modelData);
  GeFinalize();
}

TEST_F(UtestGEExecutorTest, ModelPartitionTable_num_Abnormal) {
  ModelFileHeader head;
  (void)memset_s(&head , sizeof(ModelFileHeader), 0, sizeof(ModelFileHeader));
  head.magic = MODEL_FILE_MAGIC_NUM;
  ModelPartitionTable table;
  (void)memset_s(&table , sizeof(ModelPartitionTable), 0, sizeof(ModelPartitionTable));
  table.num = MAX_PARTITION_NUM + 1;

  std::string fileName;
  fileName.append("./test").append(".exeom");
  FILE *f = fopen(fileName.c_str(), "wb+");
  fwrite(&head, sizeof(ModelFileHeader), 1, f);
  fwrite(&table, sizeof(ModelPartitionTable), 1, f);
  fclose(f);

  GeInitialize();
  ModelData modelData;
  (void)memset_s(&modelData , sizeof(ModelData), 0, sizeof(ModelData));
  EXPECT_CALL(RtStubMock::GetInstance(), access(_,_)).Times(AnyNumber()).WillOnce(Return(0));
  EXPECT_EQ(LoadDataFromFile(fileName.c_str(), &modelData), SUCCESS);
  uint32_t modelId = 0;
  EXPECT_EQ(GeLoadModelFromData(&modelId, &modelData), ACL_ERROR_GE_EXEC_LOAD_MODEL_PARTITION_FAILED);
  FreeModelData(&modelData);
  GeFinalize();
}

TEST_F(UtestGEExecutorTest, PartitionInfo_mem_size_is_zero) {
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

  std::copy_n((uint8_t*)(&desc), 0, std::back_inserter(model));
    }).Build();

  // write modelBuilder to file
  std::string fileName;
  fileName.append("./test").append(".exeom");
  FILE *f = fopen(fileName.c_str(), "wb+");
  fwrite(modelBuilder.ModelData(), modelBuilder.ModelLen(), 1, f);
  fclose(f);

  GeInitialize();
  ModelData modelData;
  (void)memset_s(&modelData , sizeof(ModelData), 0, sizeof(ModelData));
  EXPECT_CALL(RtStubMock::GetInstance(), access(_,_)).Times(AnyNumber()).WillOnce(Return(0));
  EXPECT_EQ(LoadDataFromFile(fileName.c_str(), &modelData), SUCCESS);
  uint32_t modelId = 0;
  EXPECT_EQ(GeLoadModelFromData(&modelId, &modelData), ACL_ERROR_GE_EXEC_LOAD_MODEL_PARTITION_FAILED);
  FreeModelData(&modelData);
  GeFinalize();
}

TEST_F(UtestGEExecutorTest, PartitionInfo_type_not_exist) {
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
    }).AddPartition(CUST_AICPU_KERNELS, [] (std::vector<uint8_t> &model) {
      GenerateEleForModel(0, 100, model);
    }).Build();

  // write modelBuilder to file
  std::string fileName;
  fileName.append("./test").append(".exeom");
  FILE *f = fopen(fileName.c_str(), "wb+");
  fwrite(modelBuilder.ModelData(), modelBuilder.ModelLen(), 1, f);
  fclose(f);

  GeInitialize();
  ModelData modelData;
  (void)memset_s(&modelData , sizeof(ModelData), 0, sizeof(ModelData));
  EXPECT_CALL(RtStubMock::GetInstance(), access(_,_)).Times(AnyNumber()).WillOnce(Return(0));
  EXPECT_EQ(LoadDataFromFile(fileName.c_str(), &modelData), SUCCESS);
  uint32_t modelId = 0;
  EXPECT_EQ(GeLoadModelFromData(&modelId, &modelData), SUCCESS);
  FreeModelData(&modelData);
  GeFinalize();
}

void StubModelInOutPartitionAbnormal1(std::vector<uint8_t> &model) {
  static uint8_t stub[] = {
      // uint8_t[] ModelDescTlvConfig
      0x00, 0x00, 0x00, 0x00,                          // type
      0x04, 0x00, 0x00, 0x00,                          // len
      // uint8_t[] descNum
      0x01, 0x00, 0x00, 0x00,                          // tensor num
      // uint8_t[] ModelTensorDescBaseInfo
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // size
      0x1C, 0x00, 0x00, 0x00,                          // format
      0x10, 0x00, 0x00, 0x00,                          // dt
      0x02, 0x00, 0x00, 0x00,                          // name_len
      0x08, 0x00, 0x00, 0x00,                          // dims_len
      0x08, 0x00, 0x00, 0x00,                          // dimsV2_len
      0x08, 0x00, 0x00, 0x00,                          // shape_range_len
  };
  size_t total_size = sizeof(ModelDescTlvConfig) + sizeof(uint32_t) + sizeof(ModelTensorDescBaseInfo);
  std::copy_n(stub, total_size, std::back_inserter(model));
}

TEST_F(UtestGEExecutorTest, ParseModelIoDescInfo_WithMemoryErr) {
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
    }).AddPartition(MODEL_INOUT_INFO, [] (std::vector<uint8_t> &model) { StubModelInOutPartitionAbnormal1(model); })
      .Build();

  std::string fileName;
  fileName.append("./test").append(".exeom");
  FILE *f = fopen(fileName.c_str(), "wb+");
  fwrite(modelBuilder.ModelData(), modelBuilder.ModelLen(), 1, f);
  fclose(f);

  GeInitialize();
  ModelData modelData;
  (void)memset_s(&modelData , sizeof(ModelData), 0, sizeof(ModelData));
  EXPECT_CALL(RtStubMock::GetInstance(), access(_,_)).Times(AnyNumber()).WillOnce(Return(0));
  EXPECT_EQ(LoadDataFromFile(fileName.c_str(), &modelData), SUCCESS);
  uint32_t modelId = 0;
  EXPECT_EQ(GeLoadModelFromData(&modelId, &modelData), ACL_ERROR_GE_MEMORY_OPERATE_FAILED);
  FreeModelData(&modelData);
  GeFinalize();
}

void StubModelInOutPartitionAbnormal2(std::vector<uint8_t> &model) {
  static uint8_t stub[] = {
    // uint8_t[] ModelDescTlvConfig
    0x03, 0x00, 0x00, 0x00,                          // type
    0x04, 0x00, 0x00, 0x00,                          // len
    // uint8_t[] descNum
    0x01, 0x00, 0x00, 0x00,                          // tensor num
    // uint8_t[] ModelTensorDescBaseInfo
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // size
    0x1C, 0x00, 0x00, 0x00,                          // format
    0x10, 0x00, 0x00, 0x00,                          // dt
    0x02, 0x00, 0x00, 0x00,                          // name_len
    0x08, 0x00, 0x00, 0x00,                          // dims_len
    0x08, 0x00, 0x00, 0x00,                          // dimsV2_len
    0x08, 0x00, 0x00, 0x00,                          // shape_range_len
  };
  size_t total_size = sizeof(ModelDescTlvConfig) + sizeof(uint32_t) + sizeof(ModelTensorDescBaseInfo);
  std::copy_n(stub, total_size, std::back_inserter(model));
}

TEST_F(UtestGEExecutorTest, ParseModelIoDescInfo_WithInternalErr) {
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
    }).AddPartition(MODEL_INOUT_INFO, [] (std::vector<uint8_t> &model) { StubModelInOutPartitionAbnormal2(model); })
      .Build();

  std::string fileName;
  fileName.append("./test").append(".exeom");
  FILE *f = fopen(fileName.c_str(), "wb+");
  fwrite(modelBuilder.ModelData(), modelBuilder.ModelLen(), 1, f);
  fclose(f);

  GeInitialize();
  ModelData modelData;
  (void)memset_s(&modelData , sizeof(ModelData), 0, sizeof(ModelData));
  EXPECT_CALL(RtStubMock::GetInstance(), access(_,_)).Times(AnyNumber()).WillOnce(Return(0));
  EXPECT_EQ(LoadDataFromFile(fileName.c_str(), &modelData), SUCCESS);
  uint32_t modelId = 0;
  EXPECT_EQ(GeLoadModelFromData(&modelId, &modelData), ACL_ERROR_GE_INTERNAL_ERROR);
  FreeModelData(&modelData);
  GeFinalize();
}

void *CreatModelDbgHandleMock1() {
  return (void *)ACL_ERROR_GE_INTERNAL_ERROR;
}

TEST_F(UtestGEExecutorTest, CreateModelDbgHandle_NotNULL) {
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
  int test = 1;
  void *ptr = &test;
  EXPECT_CALL(RtStubMock::GetInstance(), dlopen(_,_)).Times(1).WillOnce(Return(ptr));
  EXPECT_CALL(RtStubMock::GetInstance(), dlclose(_)).Times(1).WillOnce(Return(0));
  EXPECT_CALL(RtStubMock::GetInstance(), dlsym(_,_)).Times(9).WillOnce(Return((void *)DbgInitMock))
            .WillOnce(Return((void *)DbgDeInitMock))
            .WillOnce(Return((void *)DbgNotifySetDeviceMock))
            .WillOnce(Return((void *)GetProfEnableMock))
            .WillOnce(Return((void *)DbgLoadModelPostProcessMock))
            .WillOnce(Return((void *)CreatModelDbgHandleMock1))
            .WillOnce(Return((void *)LoadModelPreProcessMock))
            .WillOnce(Return((void *)StepIdConuterPlusMock))
            .WillOnce(Return((void *)DataFreeLoadDumpInfoMock));
  const char *configPath = nullptr;
  GeDbgInit(configPath);
  uint32_t model_id = 0;
  ModelData model_data;
  (void)memset_s(&model_data, sizeof(ModelData), 0, sizeof(ModelData));
  model_data.modelData = modelBuilder.ModelData();
  model_data.modelLen = modelBuilder.ModelLen();
  EXPECT_CALL(RtStubMock::GetInstance(), rtNanoModelLoad(_,_)).Times(1).WillOnce(Return(RT_ERROR_NONE));
  EXPECT_CALL(RtStubMock::GetInstance(), rtNanoModelDestroy(_)).Times(1).WillOnce(Return(RT_ERROR_NONE));
  EXPECT_EQ(GeLoadModelFromData(&model_id, &model_data), SUCCESS);
  EXPECT_EQ(UnloadModel(model_id), SUCCESS);
  GeDbgDeInit();
  GeFinalize();
}

TEST_F(UtestGEExecutorTest, Parse_IO_Size_WithSmallerSize) {
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
  }).AddPartition(MODEL_INOUT_INFO, [] (std::vector<uint8_t> &model) {
    GenerateEleForModel(0, 6, model);
  }).Build();

  GeInitialize();
  ModelData modelData;
  (void)memset_s(&modelData , sizeof(ModelData), 0, sizeof(ModelData));
  modelData.modelData = modelBuilder.ModelData();
  modelData.modelLen = modelBuilder.ModelLen();
  ModelInOutInfo info;
  EXPECT_EQ(GetModelDescInfoFromMem(&modelData, &info), ACL_ERROR_GE_INTERNAL_ERROR);
  GeFinalize();
}

TEST_F(UtestGEExecutorTest, Parse_IO_Size_Abnormal_WithBiggerSize) {
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
  }).AddPartition(MODEL_INOUT_INFO, [] (std::vector<uint8_t> &model) {
    GenerateEleForModel(0, 206, model);
  }).Build();

  GeInitialize();
  ModelData modelData;
  (void)memset_s(&modelData , sizeof(ModelData), 0, sizeof(ModelData));
  modelData.modelData = modelBuilder.ModelData();
  modelData.modelLen = modelBuilder.ModelLen();
  ModelInOutInfo info;
  EXPECT_EQ(GetModelDescInfoFromMem(&modelData, &info), ACL_ERROR_GE_INTERNAL_ERROR);
  GeFinalize();
}

TEST_F(UtestGEExecutorTest, GeExecutorCaseParseModelDescExtendNormal) {
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
      .AddPartition(PRE_MODEL_DESC_EXTEND, [](std::vector<uint8_t> &model) { StubExtendPartitionNormal(model); })
      .Build();

  GeInitialize();
  uint32_t model_id = 0;
  ModelData model_data;
  (void)memset_s(&model_data, sizeof(ModelData), 0, sizeof(ModelData));
  model_data.modelData = modelBuilder.ModelData();
  model_data.modelLen = modelBuilder.ModelLen();
  EXPECT_CALL(RtStubMock::GetInstance(), rtNanoModelLoad(_, _)).Times(AnyNumber()).WillOnce(Return(RT_ERROR_NONE));
  EXPECT_CALL(RtStubMock::GetInstance(), rtNanoModelDestroy(_)).Times(AnyNumber()).WillOnce(Return(RT_ERROR_NONE));
  EXPECT_EQ(GeLoadModelFromData(&model_id, &model_data), SUCCESS);
  EXPECT_EQ(UnloadModel(model_id), SUCCESS);
  GeFinalize();
}

void StubExtendPartitionAbnormal01(std::vector<uint8_t> &model) {
  static uint8_t stub[] = {
      // uint8_t[] data
      0x48, 0x4D, 0x4F, 0x44,  // magic
  };
  size_t total_size = 4;
  std::copy_n(stub, total_size, std::back_inserter(model));
}

TEST_F(UtestGEExecutorTest, GeExecutorCaseParseModelDescExtendInvalidModelExtendHead) {
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
      .AddPartition(PRE_MODEL_DESC_EXTEND, [](std::vector<uint8_t> &model) { StubExtendPartitionAbnormal01(model); })
      .Build();

  GeInitialize();
  uint32_t model_id = 0;
  ModelData model_data;
  (void)memset_s(&model_data, sizeof(ModelData), 0, sizeof(ModelData));
  model_data.modelData = modelBuilder.ModelData();
  model_data.modelLen = modelBuilder.ModelLen();
  EXPECT_EQ(GeLoadModelFromData(&model_id, &model_data), ACL_ERROR_GE_LOAD_MODEL);
  GeFinalize();
}

void StubExtendPartitionAbnormal02(std::vector<uint8_t> &model) {
  static uint8_t stub[] = {
      // uint8_t[] data
      0x00, 0x00, 0x00, 0x01,                          // magic
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // len 0
  };
  size_t total_size = 12;
  std::copy_n(stub, total_size, std::back_inserter(model));
}

TEST_F(UtestGEExecutorTest, GeExecutorCaseParseModelDescExtendAbnormalInvalidMagicNumber) {
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
      .AddPartition(PRE_MODEL_DESC_EXTEND, [](std::vector<uint8_t> &model) { StubExtendPartitionAbnormal02(model); })
      .Build();

  GeInitialize();
  uint32_t model_id = 0;
  ModelData model_data;
  (void)memset_s(&model_data, sizeof(ModelData), 0, sizeof(ModelData));
  model_data.modelData = modelBuilder.ModelData();
  model_data.modelLen = modelBuilder.ModelLen();
  EXPECT_EQ(GeLoadModelFromData(&model_id, &model_data), ACL_ERROR_GE_LOAD_MODEL);
  GeFinalize();
}

void StubExtendPartitionAbnormal03(std::vector<uint8_t> &model) {
  static uint8_t stub[] = {
      // uint8_t[] data
      0x48, 0x4D, 0x4F, 0x44,                          // magic
      0x24, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // len 36
      // uint8_t[] tlv
      0x00, 0x00, 0x00, 0x00,                          // type 0
      0x1C, 0x00, 0x00, 0x00,                          // len 28
      0x09, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // total_size 9
      0x02, 0x00, 0x00, 0x00,                          // num
      0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // mem_size 0
      0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // mem_size 1
  };
  size_t total_size =
      sizeof(TlvHead) + sizeof(struct ModelExtendHead) + sizeof(struct ModelGlobalDataInfo) + 2 * sizeof(uint64_t);  // mem_size
  std::copy_n(stub, total_size, std::back_inserter(model));
}

TEST_F(UtestGEExecutorTest, GeExecutorCaseParseModelDescExtendAbnormalInvalidTotalSize) {
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
      .AddPartition(PRE_MODEL_DESC_EXTEND, [](std::vector<uint8_t> &model) { StubExtendPartitionAbnormal03(model); })
      .Build();

  GeInitialize();
  uint32_t model_id = 0;
  ModelData model_data;
  (void)memset_s(&model_data, sizeof(ModelData), 0, sizeof(ModelData));
  model_data.modelData = modelBuilder.ModelData();
  model_data.modelLen = modelBuilder.ModelLen();
  EXPECT_EQ(GeLoadModelFromData(&model_id, &model_data), ACL_ERROR_GE_LOAD_MODEL);
  GeFinalize();
}

void StubExtendPartitionAbnormal04(std::vector<uint8_t> &model) {
  static uint8_t stub[] = {
      // uint8_t[] data
      0x48, 0x4D, 0x4F, 0x44,                          // magic
      0x24, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // len 36
      // uint8_t[] tlv
      0x00, 0x00, 0x00, 0x00,                          // type 0
      0x02, 0x00, 0x00, 0x00,                          // len 2
      0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // total_size 16
      0x02, 0x00, 0x00, 0x00,                          // num
      0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // mem_size 0
      0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // mem_size 1
  };
  size_t total_size =
      sizeof(TlvHead) + sizeof(struct ModelExtendHead) + sizeof(struct ModelGlobalDataInfo) + 2 * sizeof(uint64_t);  // mem_size
  std::copy_n(stub, total_size, std::back_inserter(model));
}

TEST_F(UtestGEExecutorTest, GeExecutorCaseParseModelDescExtendAbnormalInvalidTlvLen) {
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
      .AddPartition(PRE_MODEL_DESC_EXTEND, [](std::vector<uint8_t> &model) { StubExtendPartitionAbnormal04(model); })
      .Build();

  GeInitialize();
  uint32_t model_id = 0;
  ModelData model_data;
  (void)memset_s(&model_data, sizeof(ModelData), 0, sizeof(ModelData));
  model_data.modelData = modelBuilder.ModelData();
  model_data.modelLen = modelBuilder.ModelLen();
  EXPECT_EQ(GeLoadModelFromData(&model_id, &model_data), ACL_ERROR_GE_LOAD_MODEL);
  GeFinalize();
}

void StubExtendPartitionAbnormal05(std::vector<uint8_t> &model) {
  static uint8_t stub[] = {
      // uint8_t[] data
      0x48, 0x4D, 0x4F, 0x44,                          // magic
      0x25, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // len 37
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

TEST_F(UtestGEExecutorTest, GeExecutorCaseParseModelDescExtendAbnormalInvalidTlvListLen) {
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
      .AddPartition(PRE_MODEL_DESC_EXTEND, [](std::vector<uint8_t> &model) { StubExtendPartitionAbnormal05(model); })
      .Build();

  GeInitialize();
  uint32_t model_id = 0;
  ModelData model_data;
  (void)memset_s(&model_data, sizeof(ModelData), 0, sizeof(ModelData));
  model_data.modelData = modelBuilder.ModelData();
  model_data.modelLen = modelBuilder.ModelLen();
  EXPECT_EQ(GeLoadModelFromData(&model_id, &model_data), ACL_ERROR_GE_LOAD_MODEL);
  GeFinalize();
}

void StubExtendPartitionAbnormal06(std::vector<uint8_t> &model) {
  static uint8_t stub[] = {
      // uint8_t[] data
      0x48, 0x4D, 0x4F, 0x44,                          // magic
      0x24, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // len 36
      // uint8_t[] tlv
      0x00, 0x00, 0x00, 0x00,                          // type 0
      0x1C, 0x00, 0x00, 0x00,                          // len 28
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // total_size 0
      0x02, 0x00, 0x00, 0x00,                          // num
      0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // mem_size 0
      0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // mem_size 1
  };
  size_t total_size =
      sizeof(TlvHead) + sizeof(struct ModelExtendHead) + sizeof(struct ModelGlobalDataInfo) + 2 * sizeof(uint64_t);  // mem_size
  std::copy_n(stub, total_size, std::back_inserter(model));
}

TEST_F(UtestGEExecutorTest, GeExecutorCaseParseModelDescExtendAbnormalInvalidTotalSizeZero) {
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
      .AddPartition(PRE_MODEL_DESC_EXTEND, [](std::vector<uint8_t> &model) { StubExtendPartitionAbnormal06(model); })
      .Build();

  GeInitialize();
  uint32_t model_id = 0;
  ModelData model_data;
  (void)memset_s(&model_data, sizeof(ModelData), 0, sizeof(ModelData));
  model_data.modelData = modelBuilder.ModelData();
  model_data.modelLen = modelBuilder.ModelLen();
  EXPECT_EQ(GeLoadModelFromData(&model_id, &model_data), ACL_ERROR_GE_LOAD_MODEL);
  GeFinalize();
}

void StubExtendPartitionAbnormal07(std::vector<uint8_t> &model) {
  static uint8_t stub[] = {
      // uint8_t[] data
      0x48, 0x4D, 0x4F, 0x44,                          // magic
      0x34, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  // len 52
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

TEST_F(UtestGEExecutorTest, GeExecutorCaseParseModelDescExtendAbnormalInvalidLen) {
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
      .AddPartition(PRE_MODEL_DESC_EXTEND, [](std::vector<uint8_t> &model) { StubExtendPartitionAbnormal07(model); })
      .Build();

  GeInitialize();
  uint32_t model_id = 0;
  ModelData model_data;
  (void)memset_s(&model_data, sizeof(ModelData), 0, sizeof(ModelData));
  model_data.modelData = modelBuilder.ModelData();
  model_data.modelLen = modelBuilder.ModelLen();
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmMalloc(_)).Times(2).WillOnce(Invoke(mmMalloc_Normal_Invoke)).WillOnce(Invoke(mmMalloc_Abnormal_Invoke));
  EXPECT_EQ(GeLoadModelFromData(&model_id, &model_data), ACL_ERROR_GE_LOAD_MODEL); //covers malloc fail in ProcFifoInfo

  EXPECT_CALL(MmpaStubMock::GetInstance(), mmMalloc(_)).WillRepeatedly(Invoke(mmMalloc_Normal_Invoke));
  EXPECT_EQ(GeLoadModelFromData(&model_id, &model_data), ACL_ERROR_GE_LOAD_MODEL);
  GeFinalize();
}

TEST_F(UtestGEExecutorTest, GeExecutor_mmMallocFailed) {
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
    }).AddInputDesc("op1", 10, FORMAT_CHWN, DT_FLOAT, dims1, dims2, shapeRange
     ).AddOutputDesc("op2", 8, FORMAT_CHWN, DT_FLOAT, dims1, dims2, shapeRange).Build();
  GeInitialize();
  uint32_t model_id = 0;
  ModelData model_data;
  (void)memset_s(&model_data, sizeof(ModelData), 0, sizeof(ModelData));
  model_data.modelData = modelBuilder.ModelData();
  model_data.modelLen = modelBuilder.ModelLen();
  EXPECT_CALL(RtStubMock::GetInstance(), rtNanoModelLoad(_,_)).Times(1).WillOnce(Return(RT_ERROR_NONE));
  EXPECT_CALL(RtStubMock::GetInstance(), rtNanoModelDestroy(_)).Times(1).WillOnce(Return(RT_ERROR_NONE));
  EXPECT_EQ(GeLoadModelFromData(&model_id, &model_data), SUCCESS);

  ModelInOutInfo info;
  (void)memset_s(&info, sizeof(ModelInOutInfo), 0, sizeof(ModelInOutInfo));
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmMalloc(_)).WillRepeatedly(Invoke(mmMalloc_Abnormal_Invoke));
  EXPECT_EQ(GetModelDescInfo(model_id, &info), ACL_ERROR_GE_DEVICE_MEMORY_OPERATE_FAILED);
  EXPECT_CALL(MmpaStubMock::GetInstance(), mmMalloc(_)).Times(2)
    .WillOnce(Invoke(mmMalloc_Normal_Invoke))
    .WillOnce(Invoke(mmMalloc_Abnormal_Invoke));
  EXPECT_EQ(GetModelDescInfo(model_id, &info), ACL_ERROR_GE_MEMORY_ALLOCATION);
  EXPECT_EQ(UnloadModel(model_id), SUCCESS);
  GeFinalize();
}

TEST_F(UtestGEExecutorTest, ForDump_StaticTaskSize_Set_Normal) {
  ExeModelBuilder modelBuilder;
  modelBuilder.AddPartition(STATIC_TASK_DESC, [] (std::vector<uint8_t> &model) {
    GenerateEleForModel(45, 301, model);
  }).Build();

  // write modelBuilder to file
  std::string fileName;
  fileName.append("./test").append(".exeom");
  FILE *f = fopen(fileName.c_str(), "wb+");
  fwrite(modelBuilder.ModelData(), modelBuilder.ModelLen(), 1, f);
  fclose(f);

  GeInitialize();
  ModelData modelData;
  (void)memset_s(&modelData , sizeof(ModelData), 0, sizeof(ModelData));
  EXPECT_EQ(LoadDataFromFile(fileName.c_str(), &modelData), SUCCESS);
  GeModelDesc desc;
  (void)memset_s(&desc, sizeof(GeModelDesc), 0, sizeof(GeModelDesc));
  int a = 1;
  modelData.part.taskPtr = &a;
  modelData.part.taskSize = 302;
  EXPECT_EQ(MdlPartitionParse(&modelData, &desc), SUCCESS);
  EXPECT_EQ(desc.part.taskSize, 301);
  FreeModelData(&modelData);
  GeFinalize();
}
