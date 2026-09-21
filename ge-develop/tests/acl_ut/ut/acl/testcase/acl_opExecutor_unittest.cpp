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

#define protected public
#define private public
#include "single_op/op_executor.h"
#include "executor/stream_executor.h"
#undef private
#undef protected

#define protected public
#define private public
#include "model/acl_resource_manager.h"
#include "single_op/acl_op_resource_manager.h"
#include "single_op/shape_range_utils.h"
#undef private
#undef protected

#define protected public
#define private public
#include "utils/acl_op_map.h"
#undef private
#undef protected

#include "acl/acl.h"
#include "acl/acl_mdl.h"
#include "single_op/op_model_parser.h"
#include "single_op/executor/resource_manager.h"
#include "utils/math_utils.h"
#include "common/ge_types.h"
#include "acl_stub.h"

using namespace std;
using namespace testing;
using namespace acl;
using namespace ge;
namespace {
static aclError SelectAclopBatchNorm(int numInputs, const aclTensorDesc *const inputDesc[], int numOutputs,
    const aclTensorDesc *const outputDesc[], const aclopAttr *opAttr, aclopKernelDesc *aclopKernelDes)
{
    (void) numInputs;
    (void) inputDesc;
    (void) numOutputs;
    (void) outputDesc;
    (void) opAttr;
    aclopKernelDes->kernelId = "kernel1";
    return ACL_SUCCESS;
}

struct AllocatorStub : public ge::Allocator {
    ge::MemBlock *Malloc(size_t size) override
    {
        (void) size;
        return nullptr;
    }

    void Free(ge::MemBlock *block) override
    {
        (void) block;
    }
    ~AllocatorStub()
    {
    }
};

std::unique_ptr<ge::Allocator> CreateAllocators_succ(const gert::TensorPlacement &placement)
{
    (void) placement;
    AllocatorStub *ptr = new (std::nothrow) AllocatorStub();
    return std::unique_ptr<ge::Allocator>(ptr);
}

ge::Status LoadDynamicSingleOpV2_Invok(
    const std::string& model_name,
    const ge::ModelData& model_data,
    void* stream,
    ge::DynamicSingleOp** single_op,
    uint64_t model_id)
{
    (void) model_name;
    (void) model_data;
    (void) stream;
    (void) model_id;
    *single_op = (DynamicSingleOp *)0x12345678;
    return SUCCESS;
}
} // namespace

class OpExecutorTest : public testing::Test {
protected:
    void SetUp()
    {
        MockFunctionTest::aclStubInstance().ResetToDefaultMock();
        int64_t shape[] = {16, 16};
        inputs_[0] = aclCreateDataBuffer((void *) 0x12345, 1024);
        inputs_[1] = aclCreateDataBuffer((void *) 0x12345, 1024);
        outputs_[0] = aclCreateDataBuffer((void *) 0x12345, 1024);
        input_desc_[0] = aclCreateTensorDesc(ACL_FLOAT, 2, shape, ACL_FORMAT_ND);
        input_desc_[1] = aclCreateTensorDesc(ACL_FLOAT, 2, shape, ACL_FORMAT_ND);
        input_desc_[2] = nullptr;
        output_desc_[0] = aclCreateTensorDesc(ACL_FLOAT, 2, shape, ACL_FORMAT_ND);
        output_desc_[1] = nullptr;
    }

    void TearDown()
    {
        Mock::VerifyAndClear((void *)(&MockFunctionTest::aclStubInstance()));
        aclDestroyDataBuffer(inputs_[0]);
        aclDestroyDataBuffer(inputs_[1]);
        aclDestroyDataBuffer(outputs_[0]);
        aclDestroyTensorDesc(input_desc_[0]);
        aclDestroyTensorDesc(input_desc_[1]);
        aclDestroyTensorDesc(output_desc_[0]);
    }

    const aclDataBuffer *inputs_[2];
    aclDataBuffer *outputs_[1];
    const aclTensorDesc *input_desc_[3];
    const aclTensorDesc *output_desc_[2];
};

Status LoadSingleOpMock(const std::string &modelName,
                        const ge::ModelData &modelData,
                        void *stream,
                        SingleOp **single_op,
                        const uint64_t model_id)
{
    (void) modelName;
    (void) modelData;
    (void) stream;
    (void) model_id;
    *single_op = (SingleOp *)0x12345678;
    return SUCCESS;
}

Status LoadDynamicSingleOpMock(const std::string &modelName,
                                const ge::ModelData &modelData,
                                void *stream,
                                DynamicSingleOp **single_op,
                                const uint64_t model_id)
{
    (void) modelName;
    (void) modelData;
    (void) stream;
    (void) model_id;
    *single_op = (DynamicSingleOp *)0x12345678;
    return SUCCESS;
}

std::unique_ptr<gert::StreamExecutor> LoadStreamExecutorFromModelDataReturnSuccess(const ge::ModelData &model_data,
                                                                                   const gert::LoweringOption &optimize_option,
                                                                                   ge::graphStatus &error_code)
{
    (void) model_data;
    (void) optimize_option;
    std::unique_ptr<gert::StreamExecutor> executor = std::unique_ptr<gert::StreamExecutor>(new gert::StreamExecutor(nullptr));
    error_code = ge::GRAPH_SUCCESS;
    return executor;
}

std::unique_ptr<gert::StreamExecutor> LoadStreamExecutorFromModelDataCheckOption(const ge::ModelData &model_data,
                                                                                 const gert::LoweringOption &optimize_option,
                                                                                 ge::graphStatus &error_code)
{
    (void) model_data;
    if (optimize_option.trust_shape_on_out_tensor) {
      error_code = ge::GRAPH_FAILED;
      return nullptr;
    }
    std::unique_ptr<gert::StreamExecutor> executor = std::unique_ptr<gert::StreamExecutor>(new gert::StreamExecutor(nullptr));
    error_code = ge::GRAPH_SUCCESS;
    return executor;
}

std::unique_ptr<gert::StreamExecutor> LoadStreamExecutorFromModelDataReturnFailed(const ge::ModelData &model_data,
                                                                                  const gert::LoweringOption &optimize_option,
                                                                                  ge::graphStatus &error_code)
{
    (void) model_data;
    (void) optimize_option;
    error_code = ge::GRAPH_FAILED;
    auto exe = std::unique_ptr<gert::StreamExecutor>(new (std::nothrow) gert::StreamExecutor(nullptr));
    return exe;
}

Status SetAllocatorStub(void *const stream, ge::Allocator *const external_allocator) {
    (void) stream;
    (void) external_allocator;
    return FAILED;
}

gert::ModelV2Executor *GetOrCreateLoadedNotNullptr(rtStream_t stream, const gert::ModelExecuteArg &arg)
{
    (void) stream;
    auto executor = new (std::nothrow) gert::ModelV2Executor;
    executor->Load(arg);
    return executor;
}

gert::ModelV2Executor *CreateAndLoadNotNullptr(rtStream_t stream, const gert::ModelExecuteArg &arg)
{
    (void) stream;
    static gert::ModelV2Executor executor;
    executor.Load(arg);
    return &executor;
}

gert::ModelV2Executor *CreateAndLoadReturnNullptr(rtStream_t stream, const gert::ModelExecuteArg &arg)
{
    (void) stream;
    (void) arg;
    return nullptr;
}

ge::graphStatus ExecuteFailed(const gert::ModelExecuteArg &arg,
                        gert::Tensor **inputs, size_t input_num,
                        gert::Tensor **outputs, size_t output_num)
{
    (void) arg;
    (void) inputs;
    (void) input_num;
    (void) outputs;
    (void) output_num;
    return FAILED;
}

std::unique_ptr<gert::ModelV2Executor> LoadExecutorFromModelDataReturnSuccess(const ge::ModelData &model_data,
                                                                              ge::graphStatus &error_code)
{
    (void) model_data;
    auto executor = std::unique_ptr<gert::ModelV2Executor>(new (std::nothrow) gert::ModelV2Executor);
    error_code = ge::GRAPH_SUCCESS;
    return executor;
}

std::unique_ptr<gert::ModelV2Executor> LoadExecutorFromModelDataReturnFailed(const ge::ModelData &model_data,
                                                                             ge::graphStatus &error_code)
{
    (void) model_data;
    error_code = ge::GRAPH_FAILED;
    return nullptr;
}

rtError_t rtCtxGetCurrentDefaultStreamSuccess(rtStream_t *stm)
{
    int tmp = 0x1;
    *stm = (rtStream_t)(&tmp);
    return RT_ERROR_NONE;
}

rtError_t rtGetDeviceSuccess(int32_t *device)
{
    *device = 0;
    return RT_ERROR_NONE;
}

rtError_t rtGetDeviceFailed(int32_t *device)
{
    (void) device;
    return ACL_ERROR_GE_FAILURE;
}

TEST_F(OpExecutorTest, TestLoadSingleOp)
{
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), LoadSingleOpV2(_, _,_,_,_))
        .WillOnce(Return(RT_FAILED))
        .WillRepeatedly(Invoke(LoadSingleOpMock));

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), LoadDynamicSingleOpV2(_, _,_,_,_))
        .WillOnce(Return(RT_FAILED))
        .WillRepeatedly(Invoke(LoadDynamicSingleOpMock));

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), Create(_)).WillRepeatedly(Invoke(CreateAllocators_succ));

    OpModel opModel;
    auto aclStream = (aclrtStream) 0x1234;
    auto *expectedSingleOp = (SingleOp *) 0x12345678;
    auto *expectedDynamicSingleOp = (DynamicSingleOp *) 0x12345678;

    SingleOp *SingleOp = nullptr;
    // first call will return failed
    ASSERT_NE(OpExecutor::LoadSingleOp(opModel, aclStream, &SingleOp), ACL_SUCCESS);
    // other call will return success
    ASSERT_EQ(OpExecutor::LoadSingleOp(opModel, aclStream, &SingleOp), ACL_SUCCESS);
    ASSERT_EQ(SingleOp, expectedSingleOp);

    DynamicSingleOp *dynamicSingleOp = nullptr;
    ASSERT_NE(OpExecutor::LoadDynamicSingleOp(opModel, aclStream, &dynamicSingleOp), ACL_SUCCESS);
    ASSERT_EQ(OpExecutor::LoadDynamicSingleOp(opModel, aclStream, &dynamicSingleOp), ACL_SUCCESS);
    ASSERT_EQ(dynamicSingleOp, expectedDynamicSingleOp);
}

TEST_F(OpExecutorTest, DoExecuteAsyncTest)
{
    auto *singleOp = (SingleOp *) 0x12345678;
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), ExecuteAsync(_,_,_))
        .WillOnce(Return(PARAM_INVALID))
        .WillRepeatedly(Return(SUCCESS));

    AclOp aclOp;
    aclOp.opType = "Add";
    aclOp.numInputs = 2;
    aclOp.numOutputs = 1;
    aclTensorDesc *inputDesc[2];
    aclTensorDesc *outputDesc[1];
    int64_t shape[]{16, 16};
    inputDesc[0] = aclCreateTensorDesc(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    inputDesc[1] = aclCreateTensorDesc(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    outputDesc[0] = aclCreateTensorDesc(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    aclOp.inputDesc = inputDesc;
    aclOp.outputDesc = outputDesc;
    bool executeWithExactModel = true;
    ASSERT_NE(OpExecutor::DoExecuteAsync(singleOp, aclOp, inputs_, outputs_, executeWithExactModel), ACL_SUCCESS);
    ASSERT_EQ(OpExecutor::DoExecuteAsync(singleOp, aclOp, inputs_, outputs_, executeWithExactModel), ACL_SUCCESS);

    aclSetTensorPlaceMent(inputDesc[0], aclMemType::ACL_MEMTYPE_HOST);
    ASSERT_EQ(OpExecutor::DoExecuteAsync(singleOp, aclOp, inputs_, outputs_, executeWithExactModel), ACL_SUCCESS);

    aclSetTensorPlaceMent(outputDesc[0], aclMemType::ACL_MEMTYPE_HOST);
    ASSERT_EQ(OpExecutor::DoExecuteAsync(singleOp, aclOp, inputs_, outputs_, executeWithExactModel), ACL_SUCCESS);

    aclDestroyTensorDesc(inputDesc[0]);
    aclDestroyTensorDesc(inputDesc[1]);
    aclDestroyTensorDesc(outputDesc[0]);
}

TEST_F(OpExecutorTest, CheckConstTensor)
{
    int64_t shape[]{16, -1};
    aclTensorDesc *desc = aclCreateTensorDesc(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    aclTensorDesc desc2;
    desc2 = *desc;
    ASSERT_EQ(desc2.CheckConstTensor(false), false);
    desc->isConst = true;
    ASSERT_EQ(desc->CheckConstTensor(false), true);
    desc->isConst = false;
    ASSERT_EQ(desc->CheckConstTensor(true), false);
    desc->memtype = ACL_MEMTYPE_HOST;
    ASSERT_EQ(desc->CheckConstTensor(true), true);
    aclDestroyTensorDesc(desc);
}

TEST_F(OpExecutorTest, DoExecuteAsyncDynamicSuccessTest)
{
    auto *dynamicSingleOp = (DynamicSingleOp *) 0x12345678;
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), ExecuteAsync(_,_,_,_,_))
        .WillOnce(Return(PARAM_INVALID))
        .WillRepeatedly(Return(SUCCESS));
    AclOp aclOp;
    aclOp.opType = "Add";
    aclOp.numInputs = 2;
    aclOp.numOutputs = 1;
    int64_t shape[]{16, -1};
    ge::SmallVector<int64_t, ge::kDefaultMaxInputNum> storageDims;
    storageDims.push_back(16);
    storageDims.push_back(-1);
    aclopAttr *opAttr = aclopCreateAttr();
    aclTensorDesc *inputDesc[2];
    aclTensorDesc *outputDesc[1];
    inputDesc[0] = aclCreateTensorDesc(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    const_cast<aclTensorDesc *>(inputDesc[0])->storageDims = storageDims;
    inputDesc[0]->IsDynamicTensor();
    inputDesc[1] = aclCreateTensorDesc(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    const_cast<aclTensorDesc *>(inputDesc[1])->storageDims = storageDims;
    outputDesc[0] = aclCreateTensorDesc(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    const_cast<aclTensorDesc *>(outputDesc[0])->storageDims = storageDims;
    aclOp.inputDesc = inputDesc;
    aclOp.outputDesc = outputDesc;
    aclOp.opAttr = opAttr;
    ASSERT_NE(OpExecutor::DoExecuteAsync(dynamicSingleOp, aclOp, inputs_, outputs_), ACL_SUCCESS);
    ASSERT_EQ(OpExecutor::DoExecuteAsync(dynamicSingleOp, aclOp, inputs_, outputs_), ACL_SUCCESS);
    
    bool executeWithExactModel = true;   
    aclSetTensorStorageFormat(const_cast<aclTensorDesc *>(aclOp.inputDesc[0]), ACL_FORMAT_ND);
    ASSERT_EQ(OpExecutor::DoExecuteAsync(dynamicSingleOp, aclOp, inputs_, outputs_, executeWithExactModel), ACL_SUCCESS);

    aclSetTensorStorageFormat(const_cast<aclTensorDesc *>(aclOp.outputDesc[0]), ACL_FORMAT_ND);
    ASSERT_EQ(OpExecutor::DoExecuteAsync(dynamicSingleOp, aclOp, inputs_, outputs_, executeWithExactModel), ACL_SUCCESS);

    aclOp.exeucteType = ACL_OP_EXECUTE_V2;
    ASSERT_EQ(OpExecutor::DoExecuteAsync(dynamicSingleOp, aclOp, inputs_, outputs_, executeWithExactModel), ACL_SUCCESS);

    aclOp.exeucteType = ACL_OP_EXECUTE_REFRESH_OUTPUT_ORI_SHAPE;
    ASSERT_EQ(OpExecutor::DoExecuteAsync(dynamicSingleOp, aclOp, inputs_, outputs_, executeWithExactModel), ACL_SUCCESS);

    aclSetTensorPlaceMent(inputDesc[0], aclMemType::ACL_MEMTYPE_HOST);
    ASSERT_EQ(OpExecutor::DoExecuteAsync(dynamicSingleOp, aclOp, inputs_, outputs_, executeWithExactModel), ACL_SUCCESS);

    aclSetTensorPlaceMent(outputDesc[0], aclMemType::ACL_MEMTYPE_HOST);
    ASSERT_EQ(OpExecutor::DoExecuteAsync(dynamicSingleOp, aclOp, inputs_, outputs_, executeWithExactModel), ACL_SUCCESS);

    aclDestroyTensorDesc(inputDesc[0]);
    aclDestroyTensorDesc(inputDesc[1]);
    aclDestroyTensorDesc(outputDesc[0]);
    aclopDestroyAttr(opAttr);
}

TEST_F(OpExecutorTest, TestCaseExecuteAsync)
{
    aclopUnregisterCompileFunc("BatchNorm");
    aclopRegisterCompileFunc("BatchNorm", SelectAclopBatchNorm);
    ASSERT_EQ(aclopCreateKernel("BatchNorm", "kernel1", "kernel1", (void *)0x1000, 1024, ACL_ENGINE_AICORE, nullptr),
              ACL_SUCCESS);
    ASSERT_EQ(aclopUpdateParams("BatchNorm", 1, input_desc_, 1, output_desc_, nullptr),
	      ACL_SUCCESS);
    int res1 = 0;
    size_t res2 = 0;
    ASSERT_NE(CheckIntAddOverflow(2147483647, 1, res1), ACL_SUCCESS);
    ASSERT_NE(CheckSizeTAddOverflow(0xffffffffffffffff, 1, res2), ACL_SUCCESS);
    ASSERT_NE(CheckSizeTMultiOverflow(0xffffffffffffffff, 2, res2), ACL_SUCCESS);
    aclopUnregisterCompileFunc("BatchNorm");
}

TEST_F(OpExecutorTest, TestCaseaclopExecWithHandle)
{
    aclopRegisterCompileFunc("BatchNorm_Test1", SelectAclopBatchNorm);
    ASSERT_EQ(aclopCreateKernel("BatchNorm_Test1", "kernel1", "kernel1", (void *)0x1000, 1024, ACL_ENGINE_AICORE, nullptr),
              ACL_SUCCESS);
    ASSERT_EQ(aclopUpdateParams("BatchNorm_Test1", 1, input_desc_, 1, output_desc_, nullptr),
	      ACL_SUCCESS);
    aclopHandle *handle = nullptr;
    ASSERT_EQ(aclopCreateHandle("BatchNorm_Test1", 1, input_desc_, 1, output_desc_, nullptr, &handle), ACL_SUCCESS);
    ASSERT_EQ(aclopExecWithHandle(handle, 1, inputs_, 1, outputs_, nullptr), ACL_SUCCESS);
    aclTensorDesc **const input_desc_tmp = const_cast<aclTensorDesc **>(const_cast<aclTensorDesc *const *>(input_desc_));
    aclDataBuffer **const inputs_tmp = const_cast<aclDataBuffer **>(const_cast<aclDataBuffer *const *>(inputs_));
    aclTensorDesc **const output_desc_tmp = const_cast<aclTensorDesc **>(const_cast<aclTensorDesc *const *>(output_desc_));
    aclDataBuffer **const outputs_tmp = const_cast<aclDataBuffer **>(const_cast<aclDataBuffer *const *>(outputs_));

    ASSERT_EQ(aclopExecuteV2("BatchNorm_Test1", 1, input_desc_tmp, inputs_tmp, 1, output_desc_tmp, outputs_tmp, nullptr, nullptr), ACL_SUCCESS);
    
    ASSERT_EQ(aclopExecWithHandle(handle, 0, inputs_, 1, outputs_, nullptr), ACL_ERROR_OP_INPUT_NOT_MATCH);
    
    aclopDestroyHandle(handle);
    aclopUnregisterCompileFunc("BatchNorm_Test1");
}


TEST_F(OpExecutorTest, TestCaseaclopUpdateParamsAging)
{
    ASSERT_EQ(acl::OpKernelSelector::GetInstance().kernelDescMap_.maxOpNum, UINT64_MAX);
    acl::OpKernelSelector::GetInstance().kernelDescMap_.hashMap_.clear();
    acl::OpKernelSelector::GetInstance().kernelDescMap_.maxOpNum = 1;
    acl::OpKernelSelector::GetInstance().kernelDescMap_.cnt = 0;

    aclopRegisterCompileFunc("BatchNorm_Test3", SelectAclopBatchNorm);
    ASSERT_EQ(aclopCreateKernel("BatchNorm_Test3", "kernel1", "kernel1", (void *)0x1000, 1024, ACL_ENGINE_AICORE, nullptr),
              ACL_SUCCESS);
    ASSERT_EQ(aclopUpdateParams("BatchNorm_Test3", 1, input_desc_, 1, output_desc_, nullptr), ACL_SUCCESS);

    ASSERT_EQ(acl::OpKernelSelector::GetInstance().kernelDescMap_.hashMap_.size(), 1);
    uint64_t timestamp_1 = acl::OpKernelSelector::GetInstance().kernelDescMap_.hashMap_.begin()->second[0]->timestamp;
    ASSERT_EQ(aclopUpdateParams("BatchNorm_Test3", 1, input_desc_, 1, output_desc_, nullptr), ACL_SUCCESS);
    uint64_t timestamp_2 = acl::OpKernelSelector::GetInstance().kernelDescMap_.hashMap_.begin()->second[0]->timestamp;
    ASSERT_TRUE(timestamp_2 > timestamp_1);
    aclopUnregisterCompileFunc("BatchNorm_Test3");

    aclopRegisterCompileFunc("BatchNorm_Test2", SelectAclopBatchNorm);
    ASSERT_EQ(aclopCreateKernel("BatchNorm_Test2", "kernel1", "kernel1", (void *)0x1000, 1024, ACL_ENGINE_AICORE, nullptr),
              ACL_SUCCESS);
    ASSERT_EQ(aclopUpdateParams("BatchNorm_Test2", 1, input_desc_, 1, output_desc_, nullptr),
	      ACL_SUCCESS);
    ASSERT_EQ(acl::OpKernelSelector::GetInstance().kernelDescMap_.hashMap_.size(), 1);
    aclopUnregisterCompileFunc("BatchNorm_Test2");
    acl::OpKernelSelector::GetInstance().kernelDescMap_.maxOpNum = UINT64_MAX;
}

TEST_F(OpExecutorTest, TestCaseaclopSetMaxOpQueueNumAging)
{
    ASSERT_EQ(acl::OpKernelSelector::GetInstance().kernelDescMap_.maxOpNum, UINT64_MAX);
    ASSERT_EQ(acl::AclOpResourceManager::GetInstance().opModels_.maxOpNum, DEFAULT_MAX_OPQUEUE_NUM);
    acl::OpKernelSelector::GetInstance().kernelDescMap_.hashMap_.clear();
    ASSERT_EQ(aclopSetMaxOpQueueNum(0), ACL_ERROR_INVALID_PARAM);
    ASSERT_EQ(aclopSetMaxOpQueueNum(1), ACL_SUCCESS);
    ASSERT_EQ(acl::OpKernelSelector::GetInstance().kernelDescMap_.maxOpNum, 1UL);
    ASSERT_EQ(acl::AclOpResourceManager::GetInstance().opModels_.maxOpNum, 1UL);
    acl::OpKernelSelector::GetInstance().kernelDescMap_.maxOpNum = UINT64_MAX;
    acl::AclOpResourceManager::GetInstance().opModels_.maxOpNum = DEFAULT_MAX_OPQUEUE_NUM;
}

TEST_F(OpExecutorTest, TestResourceManager)
{
    ResourceManager resource_manager((void*)0x1234);
    const char* str = "str";
    ASSERT_EQ(resource_manager.GetMemory((void**)&str, 10),
               ACL_SUCCESS);
}

TEST_F(OpExecutorTest, TestInitTbeTask)
{
    OpKernelDesc desc;
    ResourceManager *mng = new ResourceManager((void *)0x1234);
    TbeOpTask task(nullptr, 0);
    desc.workspaceSizes.resize(18);
    StreamExecutor se(mng, nullptr);
    ASSERT_EQ(se.InitTbeTask(desc, 0, 0, task), ACL_ERROR_INVALID_PARAM);
    desc.workspaceSizes.clear();
    desc.workspaceSizes.push_back(32);
    ASSERT_EQ(se.InitTbeTask(desc, 0, 0, task), ACL_SUCCESS);
}

TEST_F(OpExecutorTest, TestAllocateWorkspaces)
{
    std::vector<size_t> workspaceSizes;
    vector<uintptr_t> workspaces;
    workspaceSizes.resize(18);
    StreamExecutor se(nullptr, nullptr);
    auto ret = se.AllocateWorkspaces(workspaceSizes, workspaces);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
}

TEST_F(OpExecutorTest, TestAllocateWorkspacesOverSize)
{
    std::vector<size_t> workspaceSizes;
    vector<uintptr_t> workspaces;
    workspaceSizes.push_back(UINT64_MAX);
    StreamExecutor se(nullptr, nullptr);
    auto ret = se.AllocateWorkspaces(workspaceSizes, workspaces);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
}

TEST_F(OpExecutorTest, TestExecutors)
{
    StreamExecutor * exec = Executors::GetOrCreate(nullptr, nullptr);
    ASSERT_NE(exec, nullptr);
    ASSERT_EQ(exec, Executors::GetOrCreate(nullptr, nullptr));

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), aclrtCtxGetCurrentDefaultStream(_))
        .WillRepeatedly(Return(ACL_ERROR_INVALID_PARAM));
    exec = Executors::GetOrCreate(nullptr, nullptr);
    ASSERT_EQ(exec, nullptr);
    Executors::RemoveExecutor(nullptr);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), aclrtCtxGetCurrentDefaultStream(_))
        .WillOnce(Return(ACL_RT_SUCCESS));
    Executors::RemoveExecutor(nullptr);
}

TEST_F(OpExecutorTest, StreamExecuteAsyncTest)
{
    AclOp aclOpDesc;
    aclDataBuffer *inputs = aclCreateDataBuffer(nullptr, 0);
    aclDataBuffer *outputs = aclCreateDataBuffer(nullptr, 0);
    ResourceManager *mng = new ResourceManager((void *)0x1234);
    StreamExecutor executor(mng, nullptr);
    aclError ret = executor.ExecuteAsync(aclOpDesc, &inputs, &outputs);
    EXPECT_EQ(ret, ACL_ERROR_OP_NOT_FOUND);
    aclDestroyDataBuffer(inputs);
    aclDestroyDataBuffer(outputs);
}

TEST_F(OpExecutorTest, StreamExecuteAsyncTest1)
{
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), Create(_)).WillRepeatedly(Invoke(CreateAllocators_succ));
    AclOp aclOpDesc;
    aclDataBuffer *const inputs[2]{0};
    aclDataBuffer *const outputs[2]{0};
    aclOpDesc.isMatched = true;
    const aclrtStream stream = (aclrtStream) 0x1234;
    auto ret = OpExecutor::ExecuteAsync(aclOpDesc, inputs, outputs, stream);
    EXPECT_EQ(ret, ACL_SUCCESS);
    /// EXPECT_EQ(aclrtDestroyStream(stream), ACL_SUCCESS);
}

TEST_F(OpExecutorTest, StreamExecuteAsyncTest02)
{
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), Create(_)).WillRepeatedly(Invoke(CreateAllocators_succ));
    OpKernelDesc kernelDesc;
    for (size_t i = 0; i < 18; ++i) {
        kernelDesc.workspaceSizes.push_back(1);
    }

    int32_t numInputs = 1;
    int32_t numOutputs = 1;
    aclDataBuffer *inputs = aclCreateDataBuffer(nullptr, 0);
    aclDataBuffer *outputs = aclCreateDataBuffer(nullptr, 0);
    ResourceManager *mng = new ResourceManager((void *)0x1234);
    StreamExecutor executor(mng, nullptr);
    aclError ret = executor.ExecuteAsync(kernelDesc, numInputs, &inputs, numOutputs, &outputs);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
    aclDestroyDataBuffer(inputs);
    aclDestroyDataBuffer(outputs);
}


TEST_F(OpExecutorTest, TestStaticExecuteAsyncWithOpHandle)
{
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), LoadSingleOpV2(_,_,_,_,_))
        .WillOnce(Return(ACL_ERROR_GE_LOAD_MODEL))
        .WillRepeatedly(Return(SUCCESS));
    
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), ExecuteAsync(_,_,_,_,_))
        .WillRepeatedly(Return(SUCCESS));
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), Create(_)).WillRepeatedly(Invoke(CreateAllocators_succ));
    // aclOp.isCompile = false;
    aclopHandle *handle = nullptr;

    OpModelDef modelDef;
    modelDef.opType = "AAA";
    modelDef.modelPath = "BBB";

    int64_t shape[] = {16, 16};
    modelDef.inputDescArr.emplace_back(ACL_FLOAT, 2, shape, ACL_FORMAT_ND);
    modelDef.inputDescArr.emplace_back(ACL_FLOAT, 2, shape, ACL_FORMAT_ND);
    modelDef.outputDescArr.emplace_back(ACL_FLOAT, 2, shape, ACL_FORMAT_ND);

    OpModel opModel;

    char_t *data = new char_t[32];
    const std::shared_ptr<void> p_data(data, [](const char_t *const p) { delete[]p; });
    opModel.data = p_data;
    opModel.size = 32U;
    opModel.name = "path";

    auto &instance = AclOpResourceManager::GetInstance();
    instance.modelCache_.Add(modelDef.opModelId, opModel);
    bool isDeduplicate = false;
    EXPECT_EQ(instance.RegisterModel(std::move(modelDef), AclOpResourceManager::GetInstance().opModels_, false, isDeduplicate), ACL_SUCCESS);

    aclopAttr *attr = aclopCreateAttr();
    aclopCreateHandle("AAA", 2, input_desc_, 1, output_desc_, attr, &handle);
    const aclrtStream stream = (aclrtStream) 0x1234;
    EXPECT_NE(aclopExecWithHandle(handle, 2, inputs_, 1, outputs_, stream), ACL_SUCCESS);
    EXPECT_EQ(aclopExecWithHandle(handle, 2, inputs_, 1, outputs_, stream), ACL_SUCCESS);
    EXPECT_EQ(aclopExecWithHandle(handle, 2, inputs_, 0, outputs_, stream), ACL_ERROR_OP_OUTPUT_NOT_MATCH);
    aclopDestroyAttr(attr);
    aclopDestroyHandle(handle);
}

TEST_F(OpExecutorTest, TestDynamicExecuteAsyncWithOpHandle1)
{
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), LoadDynamicSingleOpV2(_, _, _, _, _))
        .WillRepeatedly(Invoke(LoadDynamicSingleOpV2_Invok));

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), ExecuteAsync(_,_,_,_,_))
        .WillRepeatedly(Return(SUCCESS));

    aclopHandle *handle = nullptr;

    OpModelDef modelDef;
    modelDef.opType = "dynamic_op";
    modelDef.modelPath = "BBB";

    int64_t shape[] = {-1, -1};
    aclTensorDesc desc(ACL_FLOAT, 2, shape, ACL_FORMAT_ND);
    desc.shapeRange.emplace_back(std::pair<int64_t, int64_t>(1, -1));
    desc.shapeRange.emplace_back(std::pair<int64_t, int64_t>(1, -1));

    modelDef.inputDescArr.emplace_back(desc);
    modelDef.inputDescArr.emplace_back(desc);
    modelDef.outputDescArr.emplace_back(desc);

    OpModel opModel;

    char_t *data = new char_t[32];
    const std::shared_ptr<void> p_data(data, [](const char_t *const p) { delete[]p; });
    opModel.data = p_data;
    opModel.size = 32U;
    opModel.name = "path";

    auto &instance = AclOpResourceManager::GetInstance();
    instance.modelCache_.Add(modelDef.opModelId, opModel);
    bool isDeduplicate = false;
    EXPECT_EQ(instance.RegisterModel(std::move(modelDef), AclOpResourceManager::GetInstance().opModels_, true, isDeduplicate),
              ACL_SUCCESS);

    aclopAttr *attr = aclopCreateAttr();
    aclopCreateHandle("dynamic_op", 2, input_desc_, 1, output_desc_, attr, &handle);
    const aclrtStream stream = (aclrtStream) 0x1234;
    EXPECT_EQ(aclopExecWithHandle(handle, 2, inputs_, 1, outputs_, stream), ACL_SUCCESS);

    instance.modelCache_.Delete(modelDef, false);
    aclopDestroyAttr(attr);
    aclopDestroyHandle(handle);
}

TEST_F(OpExecutorTest, TestDynamicExecuteAsyncWithOpHandle2)
{
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), LoadDynamicSingleOpV2(_,_,_,_,_))
        .WillOnce(Return(ACL_ERROR_GE_LOAD_MODEL))
        .WillRepeatedly(Return(SUCCESS));
    
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), ExecuteAsync(_,_,_,_,_))
        .WillRepeatedly(Return(SUCCESS));

    aclopHandle *handle = nullptr;

    OpModelDef modelDef;
    modelDef.opType = "dynamic_op";
    modelDef.modelPath = "BBB";

    int64_t shape[] = {-1, -1};
    aclTensorDesc desc(ACL_FLOAT, 2, shape, ACL_FORMAT_ND);
    desc.shapeRange.emplace_back(std::pair<int64_t, int64_t>(1, -1));
    desc.shapeRange.emplace_back(std::pair<int64_t, int64_t>(1, -1));

    modelDef.inputDescArr.emplace_back(desc);
    modelDef.inputDescArr.emplace_back(desc);
    modelDef.outputDescArr.emplace_back(desc);

    OpModel opModel;

    char_t *data = new char_t[32];
    const std::shared_ptr<void> p_data(data, [](const char_t *const p) { delete[]p; });
    opModel.data = p_data;
    opModel.size = 32U;
    opModel.name = "path";

    auto &instance = AclOpResourceManager::GetInstance();
    instance.modelCache_.Add(modelDef.opModelId, opModel);
    bool isDeduplicate = false;
    EXPECT_EQ(instance.RegisterModel(std::move(modelDef), AclOpResourceManager::GetInstance().opModels_, true, isDeduplicate),
              ACL_SUCCESS);

    aclopAttr *attr = aclopCreateAttr();
    aclopCreateHandle("dynamic_op", 2, input_desc_, 1, output_desc_, attr, &handle);
    const aclrtStream stream = (aclrtStream) 0x1234;
    EXPECT_NE(aclopExecWithHandle(handle, 2, inputs_, 1, outputs_, stream), ACL_SUCCESS);
    EXPECT_EQ(aclopExecWithHandle(handle, 2, inputs_, 1, outputs_, stream), ACL_SUCCESS);
    
    instance.modelCache_.Delete(modelDef, false);
    aclopDestroyAttr(attr);
    aclopDestroyHandle(handle);
}

TEST_F(OpExecutorTest, DoRTV2ExecuteAsyncTest)
{
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), Create(_)).WillRepeatedly(Invoke(CreateAllocators_succ));
    AclOp aclOp;
    aclOp.opType = "Add";
    aclOp.numInputs = 2;
    aclOp.numOutputs = 1;
    aclTensorDesc *inputDesc[2];
    aclTensorDesc *outputDesc[1];
    int64_t shape[]{16, 16};
    inputDesc[0] = aclCreateTensorDesc(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    inputDesc[1] = aclCreateTensorDesc(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    outputDesc[0] = aclCreateTensorDesc(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    aclDataBuffer *inputs[2];
    aclDataBuffer *outputs[1];
    inputs[0] = aclCreateDataBuffer((void *)0x12345, 16);
    inputs[1] = aclCreateDataBuffer((void *)0x12345, 16);
    outputs[0] = aclCreateDataBuffer((void *)0x12345, 16);

    aclOp.inputDesc = inputDesc;
    aclOp.outputDesc = outputDesc;
    aclOp.inputs = inputs;
    aclOp.outputs = outputs;
    aclOp.isMatched = true;
    aclOp.isDynamic = false;

    OpModelDef modelDef;
    modelDef.opType = "Add";
    modelDef.modelPath = "BBB";

    int64_t shape_model[] = {-1, -1};
    aclTensorDesc desc(ACL_FLOAT, 2, shape_model, ACL_FORMAT_ND);
    desc.shapeRange.emplace_back(std::pair<int64_t, int64_t>(1, -1));
    desc.shapeRange.emplace_back(std::pair<int64_t, int64_t>(1, -1));

    modelDef.inputDescArr.emplace_back(desc);
    modelDef.inputDescArr.emplace_back(desc);
    modelDef.outputDescArr.emplace_back(desc);

    OpModel opModel;

    char_t *data = new char_t[32];
    const std::shared_ptr<void> p_data(data, [](const char_t *const p) {
        delete[] p;
    });
    opModel.data = p_data;
    opModel.size = 32U;
    opModel.name = "path";

    auto &instance = AclOpResourceManager::GetInstance();
    instance.modelCache_.Add(modelDef.opModelId, opModel);
    bool isDeduplicate = false;
    EXPECT_EQ(
        instance.RegisterModel(std::move(modelDef), AclOpResourceManager::GetInstance().opModels_, false, isDeduplicate),
        ACL_SUCCESS);

    const aclrtStream stream = (aclrtStream)0x1234;
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), LoadStreamExecutorFromModelData(_, _, _))
        .WillRepeatedly(Invoke(LoadStreamExecutorFromModelDataReturnSuccess));
    auto ret = OpExecutor::ExecuteAsync(aclOp, inputs, outputs, stream);
    EXPECT_EQ(ret, ACL_SUCCESS); // go with rt1

    aclOp.isDynamic = true;
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), LoadStreamExecutorFromModelData(_, _, _))
        .WillOnce(Invoke(LoadStreamExecutorFromModelDataReturnFailed))
        .WillRepeatedly(Invoke(LoadStreamExecutorFromModelDataReturnSuccess));

    ret = OpExecutor::ExecuteAsync(aclOp, inputs, outputs, stream);
    EXPECT_EQ(ret, ACL_ERROR_GE_FAILURE); // go with rt2 LoadStreamExecutorFromModelDataReturnFailed

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), CreateAndLoad(_, _))
        .WillRepeatedly(Invoke(CreateAndLoadReturnNullptr));
    ret = OpExecutor::ExecuteAsync(aclOp, inputs, outputs, stream);
    EXPECT_NE(ret, ACL_SUCCESS);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), CreateAndLoad(_, _))
        .WillRepeatedly(Invoke(CreateAndLoadNotNullptr));
    ret = OpExecutor::ExecuteAsync(aclOp, inputs, outputs, stream);
    EXPECT_EQ(ret, ACL_SUCCESS); // go with rt2 LoadStreamExecutorFromModelDataReturnSuccess

    aclOp.isDynamic = true;
    aclOp.exeucteType = ACL_OP_EXECUTE_V2;
    ret = OpExecutor::ExecuteAsync(aclOp, inputs, outputs, stream);
    EXPECT_EQ(ret, ACL_SUCCESS);

    aclOp.exeucteType = ACL_OP_EXECUTE_REFRESH_OUTPUT_ORI_SHAPE;
    ret = OpExecutor::ExecuteAsync(aclOp, inputs, outputs, stream);
    EXPECT_EQ(ret, ACL_SUCCESS);

    aclOp.isDynamic = true;
    const aclrtStream streamIsNullptr = nullptr;

    ret = OpExecutor::ExecuteAsync(aclOp, inputs, outputs, streamIsNullptr);
    EXPECT_EQ(ret, ACL_SUCCESS); // go with rt2 rtCtxGetCurrentDefaultStream

    ret = OpExecutor::ExecuteAsync(aclOp, inputs, outputs, streamIsNullptr);
    EXPECT_EQ(ret, ACL_SUCCESS); // go with rt2 rtGetDeviceSuccess

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), Execute(_, _, _, _, _)).WillOnce(Invoke(ExecuteFailed));
    ret = OpExecutor::ExecuteAsync(aclOp, inputs, outputs, stream);
    EXPECT_NE(ret, ACL_SUCCESS); // go with rt2 execute failed

    instance.modelCache_.Delete(modelDef, false);

    aclDestroyTensorDesc(inputDesc[0]);
    aclDestroyTensorDesc(inputDesc[1]);
    aclDestroyTensorDesc(outputDesc[0]);
    aclDestroyDataBuffer(inputs[0]);
    aclDestroyDataBuffer(inputs[1]);
    aclDestroyDataBuffer(outputs[0]);
    /// EXPECT_EQ(aclrtDestroyStream(stream), ACL_SUCCESS);
}

TEST_F(OpExecutorTest, AclRT2StorageShapeEqualShape)
{
    int64_t shape[4]{16, -1, 0, 0};
    aclTensorDesc desc1(ACL_FLOAT16, 4, shape, ACL_FORMAT_ND);
    EXPECT_EQ(desc1.storageDims.at(0), shape[0]);
    EXPECT_EQ(desc1.storageDims.at(1), shape[1]);

    aclTensorDesc desc2(ACL_FLOAT16, {1, 2}, ACL_FORMAT_ND);
    EXPECT_EQ(desc2.storageDims.at(0), 1);
    EXPECT_EQ(desc2.storageDims.at(1), 2);
}

TEST_F(OpExecutorTest, ExecuteAsyncCheckTrustShapeOnOutTensorFalse)
{
    AclOp aclOp;
    aclOp.opType = "Add";
    aclOp.numInputs = 2;
    aclOp.numOutputs = 1;
    aclTensorDesc *inputDesc[2];
    aclTensorDesc *outputDesc[1];
    int64_t shape[]{4, 64, 16, 16};
    int64_t storage_shape[]{4, 4, 16, 16, 16};
    inputDesc[0] = aclCreateTensorDesc(ACL_FLOAT16, 4, shape, ACL_FORMAT_NCHW);
    inputDesc[1] = aclCreateTensorDesc(ACL_FLOAT16, 4, shape, ACL_FORMAT_NCHW);
    outputDesc[0] = aclCreateTensorDesc(ACL_FLOAT16, 4, shape, ACL_FORMAT_NCHW);
    aclSetTensorFormat(inputDesc[0], ACL_FORMAT_NC1HWC0);
    aclSetTensorShape(inputDesc[0], 5, storage_shape);

    aclSetTensorFormat(inputDesc[1], ACL_FORMAT_NC1HWC0);
    aclSetTensorShape(inputDesc[1], 5, storage_shape);

    aclDataBuffer *inputs[2];
    aclDataBuffer *outputs[1];
    inputs[0] = aclCreateDataBuffer((void *) 0x12345, 65536);
    inputs[1] = aclCreateDataBuffer((void *) 0x12345, 65536);
    outputs[0] = aclCreateDataBuffer((void *) 0x12345, 65536);

    aclOp.inputDesc = inputDesc;
    aclOp.outputDesc = outputDesc;
    aclOp.inputs = inputs;
    aclOp.outputs = outputs;
    aclOp.isMatched = true;
    aclOp.isDynamic = false;
    const aclrtStream stream = (aclrtStream) 0x1234;

    aclOp.isDynamic = true;

    OpModelDef modelDef;
    modelDef.opType = "Add";
    modelDef.modelPath = "BBB";

    int64_t shape_model[] = {-1, -1};
    aclTensorDesc desc(ACL_FLOAT, 2, shape_model, ACL_FORMAT_ND);
    desc.shapeRange.emplace_back(std::pair<int64_t, int64_t>(1, -1));
    desc.shapeRange.emplace_back(std::pair<int64_t, int64_t>(1, -1));

    modelDef.inputDescArr.emplace_back(desc);
    modelDef.inputDescArr.emplace_back(desc);
    modelDef.outputDescArr.emplace_back(desc);

    OpModel opModel;

    char_t *data = new char_t[32];
    const std::shared_ptr<void> p_data(data, [](const char_t *const p) { delete[]p; });
    opModel.data = p_data;
    opModel.size = 32U;
    opModel.name = "path";

    auto &instance = AclOpResourceManager::GetInstance();
    instance.modelCache_.Add(modelDef.opModelId, opModel);
    bool isDeduplicate = false;
    EXPECT_EQ(instance.RegisterModel(std::move(modelDef), AclOpResourceManager::GetInstance().opModels_, false, isDeduplicate), ACL_SUCCESS);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), LoadStreamExecutorFromModelData(_, _, _))
        .WillRepeatedly(Invoke(LoadStreamExecutorFromModelDataCheckOption));

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), CreateAndLoad(_, _))
        .WillRepeatedly(Invoke(CreateAndLoadNotNullptr));

    aclOp.isDynamic = true;
    aclOp.exeucteType = ACL_OP_EXECUTE_REFRESH_OUTPUT_ORI_SHAPE;
    auto ret = OpExecutor::ExecuteAsync(aclOp, inputs, outputs, stream);
    EXPECT_EQ(ret, ACL_SUCCESS);

    instance.modelCache_.Delete(modelDef, false);

    aclDestroyTensorDesc(inputDesc[0]);
    aclDestroyTensorDesc(inputDesc[1]);
    aclDestroyTensorDesc(outputDesc[0]);
    aclDestroyDataBuffer(inputs[0]);
    aclDestroyDataBuffer(inputs[1]);
    aclDestroyDataBuffer(outputs[0]);
}

TEST_F(OpExecutorTest, DoRTV2ExecuteAsyncTestWithStorageSetByUser)
{
  AclOp aclOp;
  aclOp.opType = "Add";
  aclOp.numInputs = 2;
  aclOp.numOutputs = 1;
  aclTensorDesc *inputDesc[2];
  aclTensorDesc *outputDesc[1];
  int64_t shape[]{4, 64, 16, 16};
  int64_t storage_shape[]{4, 4, 16, 16, 16};
  inputDesc[0] = aclCreateTensorDesc(ACL_FLOAT16, 4, shape, ACL_FORMAT_NCHW);
  inputDesc[1] = aclCreateTensorDesc(ACL_FLOAT16, 4, shape, ACL_FORMAT_NCHW);
  outputDesc[0] = aclCreateTensorDesc(ACL_FLOAT16, 4, shape, ACL_FORMAT_NCHW);
  aclSetTensorFormat(inputDesc[0], ACL_FORMAT_NC1HWC0);
  aclSetTensorShape(inputDesc[0], 5, storage_shape);

  aclSetTensorFormat(inputDesc[1], ACL_FORMAT_NC1HWC0);
  aclSetTensorShape(inputDesc[1], 5, storage_shape);

  aclDataBuffer *inputs[2];
  aclDataBuffer *outputs[1];
  inputs[0] = aclCreateDataBuffer((void *) 0x12345, 65536);
  inputs[1] = aclCreateDataBuffer((void *) 0x12345, 65536);
  outputs[0] = aclCreateDataBuffer((void *) 0x12345, 65536);

  aclOp.inputDesc = inputDesc;
  aclOp.outputDesc = outputDesc;
  aclOp.inputs = inputs;
  aclOp.outputs = outputs;
  aclOp.isMatched = true;
  aclOp.isDynamic = false;
  const aclrtStream stream = (aclrtStream) 0x1234;

  OpModelDef modelDef;
  modelDef.opType = "Add";
  modelDef.modelPath = "BBB";

  int64_t shape_model[] = {-1, -1};
  aclTensorDesc desc(ACL_FLOAT, 2, shape_model, ACL_FORMAT_ND);
  desc.shapeRange.emplace_back(std::pair<int64_t, int64_t>(1, -1));
  desc.shapeRange.emplace_back(std::pair<int64_t, int64_t>(1, -1));

  modelDef.inputDescArr.emplace_back(desc);
  modelDef.inputDescArr.emplace_back(desc);
  modelDef.outputDescArr.emplace_back(desc);

  OpModel opModel;

  char_t *data = new char_t[32];
  const std::shared_ptr<void> p_data(data, [](const char_t *const p) { delete[]p; });
  opModel.data = p_data;
  opModel.size = 32U;
  opModel.name = "path";

  auto &instance = AclOpResourceManager::GetInstance();
  instance.modelCache_.Add(modelDef.opModelId, opModel);
  bool isDeduplicate = false;
  EXPECT_EQ(instance.RegisterModel(std::move(modelDef), AclOpResourceManager::GetInstance().opModels_, false, isDeduplicate), ACL_SUCCESS);

  aclOp.isDynamic = true;
  EXPECT_CALL(MockFunctionTest::aclStubInstance(), LoadStreamExecutorFromModelData(_, _, _))
      .WillRepeatedly(Invoke(LoadStreamExecutorFromModelDataReturnSuccess));

  EXPECT_CALL(MockFunctionTest::aclStubInstance(), CreateAndLoad(_, _))
      .WillRepeatedly(Invoke(CreateAndLoadNotNullptr));

  aclOp.isDynamic = true;
  aclOp.exeucteType = ACL_OP_EXECUTE_V2;
  auto ret = OpExecutor::ExecuteAsync(aclOp, inputs, outputs, stream);
  EXPECT_EQ(ret, ACL_SUCCESS);

  aclOp.exeucteType = ACL_OP_EXECUTE_REFRESH_OUTPUT_ORI_SHAPE;
  ret = OpExecutor::ExecuteAsync(aclOp, inputs, outputs, stream);
  EXPECT_EQ(ret, ACL_SUCCESS);

  instance.modelCache_.Delete(modelDef, false);

  aclDestroyTensorDesc(inputDesc[0]);
  aclDestroyTensorDesc(inputDesc[1]);
  aclDestroyTensorDesc(outputDesc[0]);
  aclDestroyDataBuffer(inputs[0]);
  aclDestroyDataBuffer(inputs[1]);
  aclDestroyDataBuffer(outputs[0]);
}

TEST_F(OpExecutorTest, GetCacheMutexAndRT2ExecutorFail)
{
  auto tmpMu = AclOpResourceManager::GetInstance().GetCacheMutex(1);
  EXPECT_EQ(tmpMu, nullptr);
  auto tmpExecutor = AclOpResourceManager::GetInstance().GetRT2Executor(1);
  EXPECT_EQ(tmpExecutor, nullptr);
}

TEST_F(OpExecutorTest, aclRecoverAllHcclTasks)
{
    auto ret = aclRecoverAllHcclTasks(0);
    EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(OpExecutorTest, DoRTV2ExecuteAsyncTestFailed)
{
    AclOp aclOp;
    aclOp.opType = "Add";
    aclOp.numInputs = 2;
    aclOp.numOutputs = 1;
    aclTensorDesc *inputDesc[2];
    aclTensorDesc *outputDesc[1];
    int64_t shape[]{16, 16};
    inputDesc[0] = aclCreateTensorDesc(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    inputDesc[1] = aclCreateTensorDesc(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    outputDesc[0] = aclCreateTensorDesc(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    aclDataBuffer *inputs[2];
    aclDataBuffer *outputs[1];
    inputs[0] = aclCreateDataBuffer((void *) 0x12345, 16);
    inputs[1] = aclCreateDataBuffer((void *) 0x12345, 16);
    outputs[0] = aclCreateDataBuffer((void *) 0x12345, 16);

    aclOp.inputDesc = inputDesc;
    aclOp.outputDesc = outputDesc;
    aclOp.inputs = inputs;
    aclOp.outputs = outputs;
    aclOp.isMatched = true;
    aclOp.isDynamic = false;

    OpModelDef modelDef;
    modelDef.opType = "Add";
    modelDef.modelPath = "BBB";

    int64_t shape_model[] = {-1, -1};
    aclTensorDesc desc(ACL_FLOAT, 2, shape_model, ACL_FORMAT_ND);
    desc.shapeRange.emplace_back(std::pair<int64_t, int64_t>(1, -1));
    desc.shapeRange.emplace_back(std::pair<int64_t, int64_t>(1, -1));

    modelDef.inputDescArr.emplace_back(desc);
    modelDef.inputDescArr.emplace_back(desc);
    modelDef.outputDescArr.emplace_back(desc);

    OpModel opModel;

    char_t *data = new char_t[32];
    const std::shared_ptr<void> p_data(data, [](const char_t *const p) { delete[]p; });
    opModel.data = p_data;
    opModel.size = 32U;
    opModel.name = "path";

    auto &instance = AclOpResourceManager::GetInstance();
    instance.modelCache_.Add(modelDef.opModelId, opModel);
    bool isDeduplicate = false;
    EXPECT_EQ(instance.RegisterModel(std::move(modelDef), AclOpResourceManager::GetInstance().opModels_, false, isDeduplicate), ACL_SUCCESS);

    const aclrtStream stream = (aclrtStream) 0x1234;
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), LoadStreamExecutorFromModelData(_, _, _))
    .WillRepeatedly(Invoke(LoadStreamExecutorFromModelDataReturnSuccess));
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), SetAllocator(_,_)).WillRepeatedly(Invoke(SetAllocatorStub));
    auto ret = OpExecutor::ExecuteAsync(aclOp, inputs, outputs, stream);
    EXPECT_NE(ret, ACL_SUCCESS); // go with rt1

    instance.modelCache_.Delete(modelDef, false);

    aclDestroyTensorDesc(inputDesc[0]);
    aclDestroyTensorDesc(inputDesc[1]);
    aclDestroyTensorDesc(outputDesc[0]);
    aclDestroyDataBuffer(inputs[0]);
    aclDestroyDataBuffer(inputs[1]);
    aclDestroyDataBuffer(outputs[0]);
}

namespace acl {
    extern aclError AclOpExecutorInitCallbackFunc(const char *configBuffer, size_t bufferSize, void *userData);
    extern aclError AclOpResourceInitCallbackFunc(const char *configBuffer, size_t bufferSize, void *userData);
    extern aclError AclOpResourceFinalizeCallbackFunc(void *userData);
}

TEST_F(OpExecutorTest, AclInitCallbackFunc)
{
    EXPECT_EQ(AclOpExecutorInitCallbackFunc(nullptr, 0, nullptr), ACL_SUCCESS);
    const char *configBuffer = "{\"op_executor\":{\"max_op_queue_num\":1024}}";
    EXPECT_EQ(AclOpExecutorInitCallbackFunc(configBuffer, strlen(configBuffer), nullptr), ACL_SUCCESS);
    const char *configBufferInvalid = "{\"op_executor\":{\"max_op_queue_num\":\"invalid\"}";
    EXPECT_EQ(AclOpExecutorInitCallbackFunc(configBufferInvalid, strlen(configBufferInvalid), nullptr), ACL_ERROR_INVALID_PARAM);

    EXPECT_EQ(AclOpResourceInitCallbackFunc(nullptr, 0, nullptr), ACL_SUCCESS);
    const char *configBufferRes = "{\"op_resource\":{\"max_op_queue_num\":2048}}";
    EXPECT_EQ(AclOpResourceInitCallbackFunc(configBufferRes, strlen(configBufferRes), nullptr), ACL_SUCCESS);
    const char *configBufferResInvalid = "{\"op_resource\":{\"max_op_queue_num\":\"invalid\"}";
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), aclrtRegStreamStateCallback(_,_,_))
        .WillOnce(Return(ACL_ERROR_INVALID_PARAM));
    EXPECT_EQ(AclOpResourceInitCallbackFunc(configBufferResInvalid, strlen(configBufferResInvalid), nullptr), ACL_ERROR_INVALID_PARAM);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), aclrtRegStreamStateCallback(_,_,_))
        .WillOnce(Return(ACL_SUCCESS));
    EXPECT_EQ(AclOpResourceFinalizeCallbackFunc(nullptr), ACL_SUCCESS);
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), aclrtRegStreamStateCallback(_,_,_))
        .WillOnce(Return(ACL_ERROR_INVALID_PARAM));
    EXPECT_EQ(AclOpResourceFinalizeCallbackFunc(nullptr), ACL_ERROR_INVALID_PARAM);

    AclOp aclOp;
    aclTensorDesc *inputDesc[1];
    aclTensorDesc *outputDesc[1];
    int64_t shape[]{16, 16};
    inputDesc[0] = aclCreateTensorDesc(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    outputDesc[0] = aclCreateTensorDesc(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    aclOp.inputDesc = inputDesc;
    aclOp.outputDesc = outputDesc;
    aclOp.numInputs = 1;
    aclOp.numOutputs = 1;
    aclOp.BackupDimsAndShapeRanges();
    aclOp.RecoverDimsAndShapeRanges();
    aclDestroyTensorDesc(inputDesc[0]);
    aclDestroyTensorDesc(outputDesc[0]);

    acl::AclOpResourceManager::GetInstance().HandleReleaseSourceByDevice(0, ACL_RT_DEVICE_STATE_RESET_PRE, nullptr);
    acl::AclOpResourceManager::GetInstance().HandleReleaseSourceByStream(0, ACL_RT_STREAM_STATE_DESTROY_PRE, nullptr);
}

TEST_F(OpExecutorTest, TestAclopExecute)
{
    aclTensorDesc *inputDesc[1];
    aclTensorDesc *outputDesc[1];
    int64_t shape[]{16, 16};
    inputDesc[0] = aclCreateTensorDesc(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    outputDesc[0] = aclCreateTensorDesc(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    aclDataBuffer *inputs[1];
    aclDataBuffer *outputs[1];
    EXPECT_EQ(aclopExecute("BatchNorm_Test1", 1, inputDesc, inputs, 1, outputDesc, outputs, nullptr, nullptr), ACL_ERROR_OP_NOT_FOUND);
    aclDestroyTensorDesc(inputDesc[0]);
    aclDestroyTensorDesc(outputDesc[0]);
}