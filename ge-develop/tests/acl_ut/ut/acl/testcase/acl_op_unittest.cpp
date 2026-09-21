/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <stdlib.h>
#include <vector>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "framework/common/ge_format_util.h"
#include "graph/operator_factory.h"
#include "acl/acl.h"
#include "acl/acl_op.h"
#include "acl/acl_op_compiler.h"
#include "model/acl_resource_manager.h"
#include "single_op/op_executor.h"
#include "single_op/compile/op_kernel_registry.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/operator.h"
#include "graph/opsproto_manager.h"
#include "acl_stub.h"

using namespace std;
using namespace testing;
using namespace acl;

namespace acl {
    extern void SetLoadOpsProtoFlag(bool value);
}; // namespace acl

static aclError SelectAclopBatchNorm(int numInputs,
                            const aclTensorDesc *const inputDesc[],
                            int numOutputs,
                            const aclTensorDesc *const outputDesc[],
                            const aclopAttr *opAttr,
                            aclopKernelDesc *aclopKernelDes)
{
    (void) numInputs;
    (void) inputDesc;
    (void) numOutputs;
    (void) outputDesc;
    (void) opAttr;
    (void) aclopKernelDes;
    return ACL_SUCCESS;
}

class OpApiTest : public testing::Test {
protected:
    void SetUp()
    {
        MockFunctionTest::aclStubInstance().ResetToDefaultMock();
        int64_t shape[] = {16, 16};
        int64_t empty_shape[] = {0, 16};
        input_desc_[0] = aclCreateTensorDesc(ACL_FLOAT, 2, shape, ACL_FORMAT_ND);
        input_desc_[1] = aclCreateTensorDesc(ACL_FLOAT, 2, shape, ACL_FORMAT_ND);
        input_desc_[2] = nullptr;

        v2_input_desc_[0] = aclCreateTensorDesc(ACL_FLOAT, 2, shape, ACL_FORMAT_ND);
        v2_input_desc_[1] = aclCreateTensorDesc(ACL_FLOAT, 2, shape, ACL_FORMAT_ND);
        v2_input_desc_[2] = nullptr;

        output_desc_[0] = aclCreateTensorDesc(ACL_FLOAT, 2, shape, ACL_FORMAT_ND);
        output_desc_[1] = nullptr;

        v2_output_desc_[0] = aclCreateTensorDesc(ACL_FLOAT, 2, shape, ACL_FORMAT_ND);
        v2_output_desc_[1] = nullptr;

        empty_output_desc_[0] = aclCreateTensorDesc(ACL_FLOAT, 2, empty_shape, ACL_FORMAT_ND);
        empty_output_desc_[1] = aclCreateTensorDesc(ACL_FLOAT, 2, empty_shape, ACL_FORMAT_ND);

        v2_empty_output_desc_[0] = aclCreateTensorDesc(ACL_FLOAT, 2, empty_shape, ACL_FORMAT_ND);
        v2_empty_output_desc_[1] = aclCreateTensorDesc(ACL_FLOAT, 2, empty_shape, ACL_FORMAT_ND);

        inputs_[0] = aclCreateDataBuffer((void *) 0x1000, 1024);
        inputs_[1] = aclCreateDataBuffer((void *) 0x1000, 1024);
        inputs_[2] = nullptr;

        v2_inputs_[0] = aclCreateDataBuffer((void *) 0x1000, 1024);
        v2_inputs_[1] = aclCreateDataBuffer((void *) 0x1000, 1024);
        v2_inputs_[2] = nullptr;

        outputs_[0] = aclCreateDataBuffer((void *) 0x1000, 1024);
        outputs_[1] = nullptr;

        empty_outputs_[0] = aclCreateDataBuffer((void *) 0x1000, 0);
        empty_outputs_[1] = aclCreateDataBuffer((void *) 0x1000, 0);

        opAttr = aclopCreateAttr();
    }

    void TearDown()
    {
        aclDestroyTensorDesc(input_desc_[0]);
        aclDestroyTensorDesc(input_desc_[1]);
        aclDestroyTensorDesc(v2_input_desc_[0]);
        aclDestroyTensorDesc(v2_input_desc_[1]);
        aclDestroyTensorDesc(output_desc_[0]);
        aclDestroyTensorDesc(v2_output_desc_[0]);
        aclDestroyTensorDesc(empty_output_desc_[0]);
        aclDestroyTensorDesc(empty_output_desc_[1]);
        aclDestroyTensorDesc(v2_empty_output_desc_[0]);
        aclDestroyTensorDesc(v2_empty_output_desc_[1]);

        aclDestroyDataBuffer(inputs_[0]);
        aclDestroyDataBuffer(inputs_[1]);
        aclDestroyDataBuffer(outputs_[0]);
        aclDestroyDataBuffer(v2_inputs_[0]);
        aclDestroyDataBuffer(v2_inputs_[1]);
        aclDestroyDataBuffer(empty_outputs_[0]);
        aclDestroyDataBuffer(empty_outputs_[1]);

        aclopDestroyAttr(opAttr);
        Mock::VerifyAndClear((void *)(&MockFunctionTest::aclStubInstance()));
    }

    const aclTensorDesc *input_desc_[3];
    const aclTensorDesc *output_desc_[2];
    const aclTensorDesc *empty_output_desc_[2];

    aclTensorDesc *v2_input_desc_[3];
    aclTensorDesc *v2_output_desc_[2];
    aclTensorDesc *v2_empty_output_desc_[2];

    const aclDataBuffer *inputs_[3];
    aclDataBuffer *v2_inputs_[3];
    aclDataBuffer *outputs_[2];
    aclDataBuffer *empty_outputs_[2];

    aclopAttr *opAttr;
};

TEST_F(OpApiTest, TestTensorDesc)
{
    auto *desc = aclCreateTensorDesc(ACL_FLOAT16, 0, nullptr, ACL_FORMAT_ND);
    aclSetTensorDescName(nullptr, "hello"); // test not core
    aclSetTensorDescName(desc, "hello");
    const char *name = aclGetTensorDescName(nullptr);
    ASSERT_STREQ(name, "");
    name = aclGetTensorDescName(desc);
    ASSERT_STREQ(name, "hello");
    aclSetTensorDescName(desc, nullptr);
    ASSERT_STREQ(name, "");
    int constData = 4;
    EXPECT_EQ(aclSetTensorConst(desc, (void *)&constData, 4), ACL_SUCCESS);
    int64_t valueRange[2][2] = {{1,2},{2,3}};
    EXPECT_EQ(aclSetTensorValueRange(nullptr, 2, valueRange), ACL_ERROR_INVALID_PARAM);
    EXPECT_EQ(aclSetTensorValueRange(desc, 2, valueRange), ACL_SUCCESS);
    aclDestroyTensorDesc(desc);
    desc = nullptr;
    aclDestroyTensorDesc(desc);
    auto *desc1 = aclCreateTensorDesc(ACL_FLOAT16, -1, nullptr, ACL_FORMAT_ND);
    desc1 = nullptr;
    aclDestroyTensorDesc(desc1);
}

TEST_F(OpApiTest, TestOpHandleCreate)
{
    aclopHandle *opHandle = nullptr;
    EXPECT_NE(aclopCreateHandle(nullptr, 0, nullptr, 0, nullptr, nullptr, &opHandle), ACL_SUCCESS);
    EXPECT_NE(aclopCreateHandle("opType", 0, nullptr, 1, nullptr, nullptr, &opHandle), ACL_SUCCESS);
    aclopDestroyHandle(opHandle);
    EXPECT_NE(aclopCreateHandle("opType", 3, input_desc_, 0, nullptr, nullptr, &opHandle), ACL_SUCCESS);
    aclopDestroyHandle(opHandle);
    EXPECT_NE(aclopCreateHandle("opType", 0, nullptr, 2, output_desc_, nullptr, &opHandle), ACL_SUCCESS);
    aclopDestroyHandle(opHandle);
    EXPECT_NE(aclopCreateHandle("opType", 2, input_desc_, 1, output_desc_, opAttr, &opHandle), ACL_SUCCESS);
    aclopDestroyHandle(opHandle);
    aclSetTensorPlaceMent(const_cast<aclTensorDesc *>(input_desc_[0]), ACL_MEMTYPE_HOST);
    EXPECT_NE(aclopCreateHandle("opType", 2, input_desc_, 1, output_desc_, opAttr, &opHandle), ACL_SUCCESS);
    aclSetTensorPlaceMent(const_cast<aclTensorDesc *>(input_desc_[0]), ACL_MEMTYPE_HOST_COMPILE_INDEPENDENT);
    EXPECT_NE(aclopCreateHandle("opType", 2, input_desc_, 1, output_desc_, opAttr, &opHandle), ACL_SUCCESS);
    aclSetTensorPlaceMent(const_cast<aclTensorDesc *>(input_desc_[0]), ACL_MEMTYPE_DEVICE);
    aclSetTensorPlaceMent(const_cast<aclTensorDesc *>(output_desc_[0]), ACL_MEMTYPE_HOST);
    EXPECT_NE(aclopCreateHandle("opType", 2, input_desc_, 1, output_desc_, opAttr, &opHandle), ACL_SUCCESS);
}

// TODO: aclopCompile
TEST_F(OpApiTest, OpCompileTest)
{
    aclopCompileType compileFlag = (aclCompileType)(3);
    aclSetTensorPlaceMent(const_cast<aclTensorDesc *>(input_desc_[0]), ACL_MEMTYPE_HOST);
    EXPECT_NE(aclopCompile("opType", 2, input_desc_, 1, output_desc_, opAttr, ACL_ENGINE_SYS, ACL_COMPILE_SYS, nullptr), ACL_SUCCESS);
    aclSetTensorPlaceMent(const_cast<aclTensorDesc *>(input_desc_[0]), ACL_MEMTYPE_HOST_COMPILE_INDEPENDENT);
    EXPECT_NE(aclopCompile("opType", 2, input_desc_, 1, output_desc_, opAttr, ACL_ENGINE_SYS, ACL_COMPILE_SYS, nullptr), ACL_SUCCESS);
    aclSetTensorPlaceMent(const_cast<aclTensorDesc *>(input_desc_[0]), ACL_MEMTYPE_DEVICE);
    aclSetTensorPlaceMent(const_cast<aclTensorDesc *>(output_desc_[0]), ACL_MEMTYPE_HOST);
    EXPECT_NE(aclopCompile("opType", 2, input_desc_, 1, output_desc_, opAttr, ACL_ENGINE_SYS, ACL_COMPILE_SYS, nullptr), ACL_SUCCESS);
    EXPECT_EQ(aclopCompile("opType", 2, input_desc_, 1, output_desc_, opAttr, ACL_ENGINE_SYS, compileFlag, nullptr), ACL_ERROR_API_NOT_SUPPORT);
}

TEST_F(OpApiTest, TestGetTensorSize)
{
    EXPECT_EQ(aclGetTensorDescSize(nullptr), 0);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), aclDataTypeSize(_))
        .WillOnce(Return((sizeof(int8_t))));
    int64_t shape[] = {16, 16};
    auto desc = aclCreateTensorDesc(ACL_BOOL, 2, shape, ACL_FORMAT_ND);
    EXPECT_EQ(aclGetTensorDescSize(desc), 256);
    aclDestroyTensorDesc(desc);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), aclDataTypeSize(_))
        .WillOnce(Return((sizeof(int16_t))));
    desc = aclCreateTensorDesc(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    EXPECT_EQ(aclGetTensorDescSize(desc), 512);
    aclDestroyTensorDesc(desc);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), aclDataTypeSize(_))
        .WillOnce(Return((sizeof(int32_t))));
    desc = aclCreateTensorDesc(ACL_FLOAT, 2, shape, ACL_FORMAT_ND);
    EXPECT_EQ(aclGetTensorDescSize(desc), 1024);
    aclDestroyTensorDesc(desc);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), aclDataTypeSize(_))
        .WillOnce(Return((sizeof(int64_t))));
    desc = aclCreateTensorDesc(ACL_INT64, 2, shape, ACL_FORMAT_ND);
    EXPECT_EQ(aclGetTensorDescSize(desc), 2048);
    aclDestroyTensorDesc(desc);
}

TEST_F(OpApiTest, TestGetTensorDescFormat)
{
    EXPECT_EQ(aclGetTensorDescFormat(nullptr), ACL_FORMAT_UNDEFINED);
    auto desc = aclCreateTensorDesc(ACL_BOOL, 0, nullptr, ACL_FORMAT_ND);
    EXPECT_EQ(aclGetTensorDescFormat(desc), ACL_FORMAT_ND);
    aclDestroyTensorDesc(desc);

    desc = aclCreateTensorDesc(ACL_BOOL, 0, nullptr, ACL_FORMAT_NC1HWC0);
    EXPECT_EQ(aclGetTensorDescFormat(desc), ACL_FORMAT_NC1HWC0);
    aclDestroyTensorDesc(desc);
}

TEST_F(OpApiTest, TestGetTensorDescDim)
{
    EXPECT_EQ(aclGetTensorDescDim(nullptr, 0), -1);
    int64_t dimSize;
    EXPECT_EQ(aclGetTensorDescDimV2(nullptr, 0, &dimSize), ACL_ERROR_INVALID_PARAM);
    auto desc = aclCreateTensorDesc(ACL_BOOL, 0, nullptr, ACL_FORMAT_ND);
    EXPECT_EQ(aclGetTensorDescDim(desc, 0), -1);
    EXPECT_EQ(aclGetTensorDescDimV2(desc, 0, &dimSize), ACL_ERROR_INVALID_PARAM);
    aclDestroyTensorDesc(desc);

    int64_t shape[] = {16, 32};
    desc = aclCreateTensorDesc(ACL_BOOL, 2, shape, ACL_FORMAT_NC1HWC0);
    EXPECT_EQ(aclGetTensorDescDim(desc, 0), 16);
    EXPECT_EQ(aclGetTensorDescDim(desc, 1), 32);
    EXPECT_EQ(aclGetTensorDescDim(desc, 2), -1);
    EXPECT_EQ(aclGetTensorDescDimV2(desc, 0, &dimSize), ACL_SUCCESS);
    aclDestroyTensorDesc(desc);
}

TEST_F(OpApiTest, TestGetDimRange)
{
    int64_t dimRange[2];
    EXPECT_EQ(aclGetTensorDescDimRange(nullptr, 0, 2, dimRange), ACL_ERROR_INVALID_PARAM);

    int64_t shape[] = {16, -1};
    int64_t range[2][2] = {{16, 16}, {1, 16}};
    auto desc = aclCreateTensorDesc(ACL_BOOL, 2, shape, ACL_FORMAT_NC1HWC0);
    aclSetTensorShapeRange(desc, 2, range);
    EXPECT_EQ(aclGetTensorDescDimRange((const aclTensorDesc*)desc, 1, 2, dimRange), ACL_SUCCESS);
    EXPECT_EQ(aclGetTensorDescDimRange((const aclTensorDesc*)desc, 2, 2, dimRange), ACL_ERROR_INVALID_PARAM);
    aclDestroyTensorDesc(desc);

    int64_t shape2[] = {-2};
    desc = aclCreateTensorDesc(ACL_BOOL, 1, shape2, ACL_FORMAT_NC1HWC0);
    EXPECT_EQ(aclGetTensorDescNumDims((const aclTensorDesc*)desc), ACL_UNKNOWN_RANK);
    aclDestroyTensorDesc(desc);
}

TEST_F(OpApiTest, TestAclOpDebugString)
{
    AclOp aclOp;
    aclOp.opType = "opType";
    ASSERT_FALSE(aclOp.DebugString().empty());

    aclOp.numInputs = 2;
    aclOp.inputDesc = input_desc_;
    ASSERT_FALSE(aclOp.DebugString().empty());

    aclOp.numOutputs = 1;
    aclOp.outputDesc = output_desc_;
    ASSERT_FALSE(aclOp.DebugString().empty());

    aclOp.opAttr = opAttr;
    ASSERT_FALSE(aclOp.DebugString().empty());

    int64_t shape[] = {1, 3, 224, 224};
    aclTensorDesc *desc_tmp = aclCreateTensorDesc(ACL_FLOAT16, 4, shape, ACL_FORMAT_NCHW);
    aclTensorDesc *input_desc_tmp[10];
    for (size_t i = 0; i < 10; ++i) {
        input_desc_tmp[i] = desc_tmp;
    }
    aclOp.inputDesc = input_desc_tmp;
    aclOp.numInputs = 10;
    ASSERT_TRUE(aclOp.DebugString().length() > 800);
    aclDestroyTensorDesc(desc_tmp);
}

TEST_F(OpApiTest, NullsTest)
{
    ASSERT_EQ(aclCreateTensorDesc(ACL_FLOAT16, 1, nullptr, ACL_FORMAT_ND), nullptr);
    ASSERT_EQ(aclGetTensorDescType(nullptr), ACL_DT_UNDEFINED);
    ASSERT_EQ(aclGetTensorDescFormat(nullptr), ACL_FORMAT_UNDEFINED);
    ASSERT_EQ(aclGetTensorDescElementCount(nullptr), 0);
    int64_t dims[1] = {-1};
    aclTensorDesc desc(ACL_INT64, 1, dims, ACL_FORMAT_NCHW);
    ASSERT_EQ(aclGetTensorDescElementCount(&desc), 0);
    int64_t dims1[2] = {0x7fffffffffffffff, 0x7fffffffffffffff};
    aclTensorDesc desc1(ACL_INT64, 2, dims1, ACL_FORMAT_NCHW);
    ASSERT_EQ(aclGetTensorDescElementCount(&desc1), 0);
    ASSERT_EQ(aclGetTensorDescNumDims(nullptr), 0);
}

TEST_F(OpApiTest, TestSetModelDir)
{
    auto ret = aclopSetModelDir("op_model");
    ASSERT_EQ(ret, ACL_ERROR_READ_MODEL_FAILURE);
    ASSERT_EQ(aclopSetModelDir("./"), ACL_SUCCESS);
    ASSERT_EQ(aclopSetModelDir("./"), ACL_ERROR_REPEAT_INITIALIZE);
}

TEST_F(OpApiTest, TestAclopLoad)
{
    int modelSize = 10;
    char* model = new(std::nothrow) char[modelSize]();
    ASSERT_NE(aclopLoad((void *)model, modelSize), ACL_SUCCESS);
    ASSERT_NE(aclopLoad((void *)model, 0), ACL_SUCCESS);
    delete []model;
}

TEST_F(OpApiTest, TestTransTensorDescFormat)
{
    int64_t shape[] = {1, 3, 224, 224};
    aclTensorDesc *srcDesc = aclCreateTensorDesc(ACL_FLOAT16, 4, shape, ACL_FORMAT_NCHW);
    aclTensorDesc *dstDesc = nullptr;
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), TransShape(_, _, _))
        .WillRepeatedly(Return(PARAM_INVALID));
    ASSERT_NE(aclTransTensorDescFormat(srcDesc, ACL_FORMAT_NC1HWC0, &dstDesc), ACL_SUCCESS);
    aclDestroyTensorDesc(srcDesc);
    Mock::VerifyAndClear((void *)(&MockFunctionTest::aclStubInstance()));

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), TransShape(_, _, _))
        .WillRepeatedly(Return(SUCCESS));
    srcDesc = aclCreateTensorDesc(ACL_FLOAT16, -1, shape, ACL_FORMAT_NCHW);
    ASSERT_NE(aclTransTensorDescFormat(srcDesc, ACL_FORMAT_NC1HWC0, &dstDesc), ACL_SUCCESS);
    aclDestroyTensorDesc(srcDesc);
    aclDestroyTensorDesc(dstDesc);
}

TEST_F(OpApiTest, AclOpCompileTest)
{
    aclopUnregisterCompileFunc("BatchNorm");
    aclopRegisterCompileFunc("BatchNorm", SelectAclopBatchNorm);
    ASSERT_NE(aclopUpdateParams(nullptr, 1, input_desc_, 1, output_desc_, opAttr),
	      ACL_SUCCESS);
    ASSERT_NE(aclopUpdateParams("BatchNorm", 1, nullptr, 1, output_desc_, opAttr),
              ACL_SUCCESS);
    ASSERT_NE(aclopUpdateParams("BatchNorm", 1, input_desc_, 1, nullptr, opAttr),
            ACL_SUCCESS);
    aclopUnregisterCompileFunc("BatchNorm");
    ASSERT_NE(aclopUpdateParams("opType", 1, input_desc_, 1, output_desc_, opAttr), ACL_SUCCESS);
}

TEST_F(OpApiTest, TestRepeatRegisterSelectKernelFunc)
{
  aclopUnregisterCompileFunc("BatchNorm");
  EXPECT_EQ(aclopRegisterCompileFunc("BatchNorm", SelectAclopBatchNorm), ACL_SUCCESS);
  EXPECT_EQ(aclopRegisterCompileFunc("BatchNorm", SelectAclopBatchNorm), ACL_ERROR_BIN_SELECTOR_ALREADY_REGISTERED);
  aclopUnregisterCompileFunc("BatchNorm");
}

TEST_F(OpApiTest, TestAclOpSetKernelArgs)
{
    aclopKernelDesc kernel_desc;
    const char* kernel_id = "kernelId";
    string arg = "arg";
    EXPECT_EQ(aclopSetKernelArgs(&kernel_desc, kernel_id, 3, arg.c_str(), arg.size()),
              ACL_SUCCESS);
}

TEST_F(OpApiTest, TestAclOpSetKernelWorkspaceSizes)
{
    aclopKernelDesc kernel_desc;
    size_t workspace_sizes[2] = {1, 2};
    EXPECT_EQ(aclopSetKernelWorkspaceSizes(&kernel_desc, 2, workspace_sizes), ACL_SUCCESS);
}

TEST_F(OpApiTest, TestAclOpUnregisterSelectKernelFunc)
{
    EXPECT_EQ(aclopUnregisterCompileFunc("opType"), ACL_SUCCESS);
}

TEST_F(OpApiTest, aclopCreateKernelTest)
{
void (*deallocator)(void *data, size_t length) = nullptr;
    EXPECT_EQ(aclopCreateKernel("opType", "kernelId", "kernelName", (void *) 0x1000, 1024, ACL_ENGINE_AICORE, deallocator),
        ACL_SUCCESS);
    EXPECT_EQ(aclopCreateKernel(nullptr, "kernelId", "kernelName", (void *) 0x1000, 1024, ACL_ENGINE_AICORE, deallocator),
        ACL_ERROR_INVALID_PARAM);
    EXPECT_EQ(aclopCreateKernel("opType", nullptr, "kernelName", (void *) 0x1000, 1024, ACL_ENGINE_AICORE, deallocator),
        ACL_ERROR_INVALID_PARAM);
    EXPECT_EQ(aclopCreateKernel("opType", "kernelId", nullptr, (void *) 0x1000, 1024, ACL_ENGINE_AICORE, deallocator),
        ACL_ERROR_INVALID_PARAM);
    EXPECT_EQ(aclopCreateKernel("opType", "kernelId", "kernelName", nullptr, 1024, ACL_ENGINE_AICORE, deallocator),
        ACL_ERROR_INVALID_PARAM);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), rtDevBinaryRegister(_, _))
        .WillOnce(Return((ACL_ERROR_RT_PARAM_INVALID)))
        .WillOnce(Return(RT_ERROR_NONE))
        .WillOnce(Return(RT_ERROR_NONE))
        .WillOnce(Return(RT_ERROR_NONE));

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), rtFunctionRegister(_, _,_,_,_))
        .WillOnce(Return((ACL_ERROR_RT_PARAM_INVALID)))
        .WillOnce(Return(RT_ERROR_NONE))
        .WillOnce(Return(RT_ERROR_NONE));

    EXPECT_NE(aclopCreateKernel("opType2", "kernelId", "kernelName", (void *) 0x1000, 1024, ACL_ENGINE_AICORE, deallocator),
                ACL_SUCCESS);
    EXPECT_NE(aclopCreateKernel("opType2", "kernelId", "kernelName", (void *) 0x1000, 1024, ACL_ENGINE_AICORE, deallocator),
                ACL_SUCCESS);
    EXPECT_EQ(aclopCreateKernel("opType2", "kernelId", "kernelName", (void *) 0x1000, 1024, ACL_ENGINE_AICORE, deallocator),
                ACL_SUCCESS);
    EXPECT_EQ(aclopCreateKernel("opType2", "kernelId", "kernelName", (void *) 0x1000, 1024, ACL_ENGINE_AICORE, deallocator),
                ACL_ERROR_KERNEL_ALREADY_REGISTERED);
    EXPECT_EQ(aclopCreateKernel("opType3", "kernelId", "kernelName", (void *) 0x1000, 1024, ACL_ENGINE_VECTOR, deallocator),
                ACL_SUCCESS);
}

TEST_F(OpApiTest, TestAclSetTensorStorage)
{
    int64_t *dims = new int64_t[2];
    aclTensorDesc desc(ACL_FLOAT, 2, dims, ACL_FORMAT_NCHW);
    EXPECT_EQ(aclSetTensorStorageFormat(&desc, ACL_FORMAT_NCHW), ACL_SUCCESS);
    EXPECT_EQ(aclSetTensorStorageShape(&desc, 2, dims), ACL_SUCCESS);
    delete[] dims;
}

TEST_F(OpApiTest, TestAclOpInferShape_fail)
{
    int32_t mmRet = 0;
    MM_SYS_SET_ENV(MM_ENV_ASCEND_OPP_PATH, "", 1, mmRet);
    (void)mmRet;
    int8_t *buf1 = new int8_t[16];
    int8_t *buf2 = new int8_t[16];
    aclDataBuffer *inputs[2];
    inputs[0] = aclCreateDataBuffer((void *)buf1, 16);
    inputs[1] = aclCreateDataBuffer((void *)buf2, 16);
    aclError ret = aclopInferShape("opType", 2, (aclTensorDesc **)input_desc_, inputs, 2, v2_empty_output_desc_, opAttr);
    aclDestroyDataBuffer(inputs[0]);
    aclDestroyDataBuffer(inputs[1]);
    delete []buf1;
    delete []buf2;
    EXPECT_EQ(ret, ACL_ERROR_INVALID_OPP_PATH);
}

TEST_F(OpApiTest, TestAclOpInferShapeFail)
{
    int32_t mmRet = 0;
    MM_SYS_SET_ENV(MM_ENV_ASCEND_OPP_PATH, "/home/jc/Ascend/invalidPath", 1, mmRet);
    (void)mmRet;
    int8_t *buf1 = new int8_t[16];
    int8_t *buf2 = new int8_t[16];
    aclDataBuffer *inputs[2];
    inputs[0] = aclCreateDataBuffer((void *)buf1, 16);
    inputs[1] = aclCreateDataBuffer((void *)buf2, 16);
    std::string filePath = "";
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), RealPath(_))
        .WillRepeatedly(Return((filePath)));

    aclError ret = aclopInferShape("opType", 2, (aclTensorDesc **)input_desc_, inputs, 2, v2_empty_output_desc_, opAttr);
    aclDestroyDataBuffer(inputs[0]);
    aclDestroyDataBuffer(inputs[1]);
    delete []buf1;
    delete []buf2;
    EXPECT_EQ(ret, ACL_ERROR_INVALID_OPP_PATH);
}

TEST_F(OpApiTest, TestLoadOpsProtoFail)
{
    int32_t mmRet = 0;
    MM_SYS_SET_ENV(MM_ENV_ASCEND_OPP_PATH, "/usr/local/Ascend/oppPath", 1, mmRet);
    (void)mmRet;
    int8_t *buf1 = new int8_t[16];
    int8_t *buf2 = new int8_t[16];
    aclDataBuffer *inputs[2];
    inputs[0] = aclCreateDataBuffer((void *)buf1, 16);
    inputs[1] = aclCreateDataBuffer((void *)buf2, 16);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), OpsProtoManager_Initialize(_))
        .WillRepeatedly(Return(false));

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetOpsTypeList(_))
        .WillRepeatedly(Return((GRAPH_SUCCESS)));

    EXPECT_NE(aclopInferShape("opType", 2, (aclTensorDesc **)input_desc_, inputs, 2, v2_empty_output_desc_, opAttr),
              ACL_SUCCESS);
    aclDestroyDataBuffer(inputs[0]);
    aclDestroyDataBuffer(inputs[1]);
    delete []buf1;
    delete []buf2;
}

TEST_F(OpApiTest, AclOpInferShapeTest)
{
    int32_t mmRet = 0;
    MM_SYS_SET_ENV(MM_ENV_ASCEND_OPP_PATH, "/usr/local/Ascend/opp", 1, mmRet);
    (void)mmRet;
    int8_t *buf1 = new int8_t[16];
    int8_t *buf2 = new int8_t[16];
    aclDataBuffer *inputs[2];
    inputs[0] = aclCreateDataBuffer((void *)buf1, 16);
    inputs[1] = aclCreateDataBuffer((void *)buf2, 16);
    aclSetTensorDescName((aclTensorDesc *)input_desc_[0], "x0");
    EXPECT_NE(aclopInferShape("opType", 2, (aclTensorDesc **)input_desc_, inputs, 2, v2_empty_output_desc_, opAttr), ACL_SUCCESS);

    aclDataBuffer *inputsV2[2];
    inputsV2[0] = aclCreateDataBuffer((void *)buf1, 16);
    inputsV2[1] = aclCreateDataBuffer((void *)buf2, 16);
    EXPECT_NE(aclopInferShape("opType", 2, (aclTensorDesc **)input_desc_, inputsV2, 2, v2_empty_output_desc_, opAttr), ACL_SUCCESS);

    string filePath = "/path";
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), RealPath(_))
        .WillRepeatedly(Return((filePath)));
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), OpsProtoManager_Initialize(_))
        .WillRepeatedly(Return(true));
    EXPECT_EQ(aclopInferShape("opType", 2, (aclTensorDesc **)input_desc_, inputsV2, 2, v2_empty_output_desc_, opAttr), ACL_SUCCESS);

    aclDestroyDataBuffer(inputs[0]);
    aclDestroyDataBuffer(inputs[1]);
    aclDestroyDataBuffer(inputsV2[0]);
    aclDestroyDataBuffer(inputsV2[1]);
    delete []buf1;
    delete []buf2;
}

TEST_F(OpApiTest, AclOpInferShapeTest02)
{
    std::string opp_path = __FILE__;
    opp_path = opp_path.substr(0, opp_path.rfind("/")) + "/opp_path";
    std::string path_builtin = opp_path + "/built-in";
    std::string path_vendors = opp_path + "/vendors";
    int32_t mmRet = 0;
    MM_SYS_SET_ENV(MM_ENV_ASCEND_OPP_PATH, opp_path.c_str(), 1, mmRet);
    (void)mmRet;
    (void)!system(("mkdir -p " + path_builtin).c_str());
    (void)!system(("mkdir -p " + path_vendors).c_str());
    int8_t *buf1 = new int8_t[16];
    int8_t *buf2 = new int8_t[16];
    aclDataBuffer *inputs[2];
    inputs[0] = aclCreateDataBuffer((void *)buf1, 16);
    inputs[1] = aclCreateDataBuffer((void *)buf2, 16);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), RealPath(_))
        .WillRepeatedly(Return(opp_path));
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), OpsProtoManager_Initialize(_))
        .WillRepeatedly(Return(true));
    acl::SetLoadOpsProtoFlag(false);
    EXPECT_EQ(aclopInferShape("opType", 2, (aclTensorDesc **)input_desc_, inputs, 2, v2_empty_output_desc_, opAttr), ACL_SUCCESS);

    aclDestroyDataBuffer(inputs[0]);
    aclDestroyDataBuffer(inputs[1]);
    delete []buf1;
    delete []buf2;
    (void)!system(("rm -rf " + opp_path).c_str());
}

TEST_F(OpApiTest, AclOpInferShapeTest03)
{
    std::string opp_path = __FILE__;
    opp_path = opp_path.substr(0, opp_path.rfind("/")) + "/opp_path";
    std::string path_builtin = opp_path + "/built-in";
    std::string path_vendors = opp_path + "/vendors";
    std::string path_config = path_vendors + "/config.ini";
    int32_t mmRet = 0;
    MM_SYS_SET_ENV(MM_ENV_ASCEND_OPP_PATH, opp_path.c_str(), 1, mmRet);
    (void)mmRet;
    (void)!system(("mkdir -p " + path_builtin).c_str());
    (void)!system(("mkdir -p " + path_vendors).c_str());
    (void)!system(("echo '' > " + path_config).c_str());
    int8_t *buf1 = new int8_t[16];
    int8_t *buf2 = new int8_t[16];
    aclDataBuffer *inputs[2];
    inputs[0] = aclCreateDataBuffer((void *)buf1, 16);
    inputs[1] = aclCreateDataBuffer((void *)buf2, 16);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), RealPath(_))
        .WillRepeatedly(Return(opp_path));
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), OpsProtoManager_Initialize(_))
        .WillRepeatedly(Return(true));
    acl::SetLoadOpsProtoFlag(false);
    EXPECT_EQ(aclopInferShape("opType", 2, (aclTensorDesc **)input_desc_, inputs, 2, v2_empty_output_desc_, opAttr), ACL_SUCCESS);

    aclDestroyDataBuffer(inputs[0]);
    aclDestroyDataBuffer(inputs[1]);
    delete []buf1;
    delete []buf2;
    (void)!system(("rm -rf " + opp_path).c_str());
}

TEST_F(OpApiTest, AclOpInferShapeTest04)
{
    std::string opp_path = __FILE__;
    opp_path = opp_path.substr(0, opp_path.rfind("/")) + "/opp_path";
    std::string path_builtin = opp_path + "/built-in";
    std::string path_vendors = opp_path + "/vendors";
    std::string path_config = path_vendors + "/config.ini";
    int32_t mmRet = 0;
    MM_SYS_SET_ENV(MM_ENV_ASCEND_OPP_PATH, opp_path.c_str(), 1, mmRet);
    (void)mmRet;
    (void)!system(("mkdir -p " + path_builtin).c_str());
    (void)!system(("mkdir -p " + path_vendors).c_str());
    (void)!system(("rm -rf " + path_config).c_str());
    int8_t *buf1 = new int8_t[16];
    int8_t *buf2 = new int8_t[16];
    aclDataBuffer *inputs[2];
    inputs[0] = aclCreateDataBuffer((void *)buf1, 16);
    inputs[1] = aclCreateDataBuffer((void *)buf2, 16);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), RealPath(_))
        .WillRepeatedly(Return(opp_path));
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), OpsProtoManager_Initialize(_))
        .WillRepeatedly(Return(true));
    acl::SetLoadOpsProtoFlag(false);
    EXPECT_EQ(aclopInferShape("opType", 2, (aclTensorDesc **)input_desc_, inputs, 2, v2_empty_output_desc_, opAttr), ACL_SUCCESS);

    aclDestroyDataBuffer(inputs[0]);
    aclDestroyDataBuffer(inputs[1]);
    delete []buf1;
    delete []buf2;
    (void)!system(("rm -rf " + opp_path).c_str());
}

TEST_F(OpApiTest, AclOpInferShapeTest05)
{
    std::string opp_path = __FILE__;
    opp_path = opp_path.substr(0, opp_path.rfind("/")) + "/opp_path";
    std::string custom_opp_path = opp_path + "/custom_opp_path";
    std::string path_builtin = opp_path + "/built-in";
    std::string path_vendors = opp_path + "/vendors";
    int32_t mmRet = 0;
    MM_SYS_SET_ENV(MM_ENV_ASCEND_OPP_PATH, opp_path.c_str(), 1, mmRet);
    (void)mmRet;
    MM_SYS_SET_ENV(MM_ENV_ASCEND_CUSTOM_OPP_PATH, custom_opp_path.c_str(), 1, mmRet);
    (void)mmRet;
    (void)!system(("mkdir -p " + path_builtin).c_str());
    (void)!system(("mkdir -p " + path_vendors).c_str());
    (void)!system(("mkdir -p " + custom_opp_path + "/op_proto").c_str());
    int8_t *buf1 = new int8_t[16];
    int8_t *buf2 = new int8_t[16];
    aclDataBuffer *inputs[2];
    inputs[0] = aclCreateDataBuffer((void *)buf1, 16);
    inputs[1] = aclCreateDataBuffer((void *)buf2, 16);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), RealPath(_))
        .WillRepeatedly(Return(opp_path));
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), OpsProtoManager_Initialize(_))
        .WillRepeatedly(Return(true));
    acl::SetLoadOpsProtoFlag(false);
    EXPECT_EQ(aclopInferShape("opType", 2, (aclTensorDesc **)input_desc_, inputs, 2, v2_empty_output_desc_, opAttr), ACL_SUCCESS);

    aclDestroyDataBuffer(inputs[0]);
    aclDestroyDataBuffer(inputs[1]);
    delete []buf1;
    delete []buf2;
    (void)!system(("rm -rf " + opp_path).c_str());
}

TEST_F(OpApiTest, AclOpInferShapeTest06)
{
    std::string opp_path = __FILE__;
    opp_path = opp_path.substr(0, opp_path.rfind("/")) + "/opp_path";
    std::string custom_opp_path = opp_path + "/custom_opp_path";
    std::string path_builtin = opp_path + "/built-in";
    std::string path_vendors = opp_path + "/vendors";
    int32_t mmRet = 0;
    MM_SYS_SET_ENV(MM_ENV_ASCEND_OPP_PATH, opp_path.c_str(), 1, mmRet);
    (void)mmRet;
    MM_SYS_SET_ENV(MM_ENV_ASCEND_CUSTOM_OPP_PATH, "", 1, mmRet);
    (void)mmRet;
    (void)!system(("mkdir -p " + path_builtin).c_str());
    (void)!system(("mkdir -p " + path_vendors).c_str());
    (void)!system(("mkdir -p " + custom_opp_path + "/op_proto").c_str());
    int8_t *buf1 = new int8_t[16];
    int8_t *buf2 = new int8_t[16];
    aclDataBuffer *inputs[2];
    inputs[0] = aclCreateDataBuffer((void *)buf1, 16);
    inputs[1] = aclCreateDataBuffer((void *)buf2, 16);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), RealPath(_))
        .WillRepeatedly(Return(opp_path));
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), OpsProtoManager_Initialize(_))
        .WillRepeatedly(Return(true));
    acl::SetLoadOpsProtoFlag(false);
    EXPECT_EQ(aclopInferShape("opType", 2, (aclTensorDesc **)input_desc_, inputs, 2, v2_empty_output_desc_, opAttr), ACL_SUCCESS);

    aclDestroyDataBuffer(inputs[0]);
    aclDestroyDataBuffer(inputs[1]);
    delete []buf1;
    delete []buf2;
    (void)!system(("rm -rf " + opp_path).c_str());
}

TEST_F(OpApiTest, AclOpInferShapeTest07)
{
    std::string opp_path = __FILE__;
    opp_path = opp_path.substr(0, opp_path.rfind("/")) + "/opp_path";
    std::string custom_opp_path_01 = opp_path + "/custom_opp_path_01";
    std::string custom_opp_path_02 = opp_path + "/custom_opp_path_02";
    std::string path_builtin = opp_path + "/built-in";
    std::string path_vendors = opp_path + "/vendors";
    int32_t mmRet = 0;
    MM_SYS_SET_ENV(MM_ENV_ASCEND_OPP_PATH, opp_path.c_str(), 1, mmRet);
    (void)mmRet;
    MM_SYS_SET_ENV(MM_ENV_ASCEND_CUSTOM_OPP_PATH, (custom_opp_path_01 + ":" + custom_opp_path_02).c_str(), 1, mmRet);
    (void)mmRet;
    (void)!system(("mkdir -p " + path_builtin).c_str());
    (void)!system(("mkdir -p " + path_vendors).c_str());
    (void)!system(("mkdir -p " + custom_opp_path_01 + "/op_proto").c_str());
    (void)!system(("mkdir -p " + custom_opp_path_02 + "/op_proto").c_str());
    int8_t *buf1 = new int8_t[16];
    int8_t *buf2 = new int8_t[16];
    aclDataBuffer *inputs[2];
    inputs[0] = aclCreateDataBuffer((void *)buf1, 16);
    inputs[1] = aclCreateDataBuffer((void *)buf2, 16);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), RealPath(_))
        .WillRepeatedly(Return(opp_path));
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), OpsProtoManager_Initialize(_))
        .WillRepeatedly(Return(true));
    acl::SetLoadOpsProtoFlag(false);
    EXPECT_EQ(aclopInferShape("opType", 2, (aclTensorDesc **)input_desc_, inputs, 2, v2_empty_output_desc_, opAttr), ACL_SUCCESS);

    aclDestroyDataBuffer(inputs[0]);
    aclDestroyDataBuffer(inputs[1]);
    delete []buf1;
    delete []buf2;
    (void)!system(("rm -rf " + opp_path).c_str());
}

TEST_F(OpApiTest, AclOpInferShapeTest08)
{
    std::string opp_path = __FILE__;
    opp_path = opp_path.substr(0, opp_path.rfind("/")) + "/opp_path";
    std::string custom_opp_path = opp_path + "/custom_opp_path";
    std::string path_builtin = opp_path + "/built-in";
    std::string path_vendors = opp_path + "/vendors";
    int32_t mmRet = 0;
    MM_SYS_SET_ENV(MM_ENV_ASCEND_OPP_PATH, opp_path.c_str(), 1, mmRet);
    (void)mmRet;
    MM_SYS_SET_ENV(MM_ENV_ASCEND_CUSTOM_OPP_PATH, custom_opp_path.c_str(), 1, mmRet);
    (void)mmRet;
    (void)!system(("mkdir -p " + path_builtin).c_str());
    (void)!system(("mkdir -p " + path_vendors).c_str());
    (void)!system(("mkdir -p " + custom_opp_path + "/op_proto").c_str());
    int8_t *buf1 = new int8_t[16];
    int8_t *buf2 = new int8_t[16];
    aclDataBuffer *inputs[2];
    inputs[0] = aclCreateDataBuffer((void *)buf1, 16);
    inputs[1] = aclCreateDataBuffer((void *)buf2, 16);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), RealPath(_))
        .WillRepeatedly(Return(opp_path));
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), OpsProtoManager_Initialize(_))
        .WillRepeatedly(Return(true));
    acl::SetLoadOpsProtoFlag(false);
    EXPECT_EQ(aclopInferShape("opType", 2, (aclTensorDesc **)input_desc_, inputs, 2, v2_empty_output_desc_, opAttr), ACL_SUCCESS);

    aclDestroyDataBuffer(inputs[0]);
    aclDestroyDataBuffer(inputs[1]);
    delete []buf1;
    delete []buf2;
    (void)!system(("rm -rf " + opp_path).c_str());
}

TEST_F(OpApiTest, AclOpInferShapeTest09)
{
    std::string opp_path = __FILE__;
    opp_path = opp_path.substr(0, opp_path.rfind("/")) + "/opp_path";
    std::string custom_opp_path_01 = opp_path + "/custom_opp_path_01";
    std::string custom_opp_path_invalid_01 = opp_path + "/custom_opp_path_invalid_01";
    std::string custom_opp_path_empty = "";
    std::string custom_opp_path_02 = opp_path + "/custom_opp_path_02";
    std::string custom_opp_path_invalid_02 = opp_path + "/custom_opp_path_invalid_02";
    std::string path_builtin = opp_path + "/built-in";
    std::string path_vendors = opp_path + "/vendors";
    int32_t mmRet = 0;
    MM_SYS_SET_ENV(MM_ENV_ASCEND_OPP_PATH, opp_path.c_str(), 1, mmRet);
    (void)mmRet;
    MM_SYS_SET_ENV(MM_ENV_ASCEND_CUSTOM_OPP_PATH, (custom_opp_path_01 + ":" +
                                      custom_opp_path_invalid_01 + ":" +
                                      custom_opp_path_empty + ":" +
                                      custom_opp_path_02 + ":" +
                                      custom_opp_path_invalid_02
                                     ).c_str(), 1, mmRet);
    (void)mmRet;
    (void)!system(("mkdir -p " + path_builtin).c_str());
    (void)!system(("mkdir -p " + path_vendors).c_str());
    (void)!system(("mkdir -p " + custom_opp_path_01 + "/op_proto").c_str());
    (void)!system(("mkdir -p " + custom_opp_path_02 + "/op_proto").c_str());
    int8_t *buf1 = new int8_t[16];
    int8_t *buf2 = new int8_t[16];
    aclDataBuffer *inputs[2];
    inputs[0] = aclCreateDataBuffer((void *)buf1, 16);
    inputs[1] = aclCreateDataBuffer((void *)buf2, 16);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), RealPath(_))
        .WillRepeatedly(Return(opp_path));
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), OpsProtoManager_Initialize(_))
        .WillRepeatedly(Return(true));
    acl::SetLoadOpsProtoFlag(false);
    EXPECT_EQ(aclopInferShape("opType", 2, (aclTensorDesc **)input_desc_, inputs, 2, v2_empty_output_desc_, opAttr), ACL_SUCCESS);

    aclDestroyDataBuffer(inputs[0]);
    aclDestroyDataBuffer(inputs[1]);
    delete []buf1;
    delete []buf2;
    (void)!system(("rm -rf " + opp_path).c_str());
}

TEST_F(OpApiTest, AclOpInferShapeTest10)
{
    std::string opp_path = __FILE__;
    opp_path = opp_path.substr(0, opp_path.rfind("/")) + "/opp_path";
    std::string path_builtin = opp_path + "/built-in";
    std::string path_vendors = opp_path + "/vendors";
    int32_t mmRet = 0;
    MM_SYS_SET_ENV(MM_ENV_ASCEND_OPP_PATH, opp_path.c_str(), 1, mmRet);
    (void)mmRet;
    (void)!system(("mkdir -p " + path_builtin).c_str());
    (void)!system(("mkdir -p " + path_vendors).c_str());
    int8_t *buf1 = new int8_t[16];
    int8_t *buf2 = new int8_t[16];
    aclDataBuffer *inputs[2];
    inputs[0] = aclCreateDataBuffer((void *)buf1, 16);
    inputs[1] = aclCreateDataBuffer((void *)buf2, 16);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), RealPath(_))
        .WillRepeatedly(Return(opp_path));
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), OpsProtoManager_Initialize(_))
        .WillRepeatedly(Return(true));
    acl::SetLoadOpsProtoFlag(false);
    EXPECT_EQ(aclopInferShape("opType", 2, (aclTensorDesc **)input_desc_, inputs, 2, v2_empty_output_desc_, opAttr), ACL_SUCCESS);

    aclDestroyDataBuffer(inputs[0]);
    aclDestroyDataBuffer(inputs[1]);
    delete []buf1;
    delete []buf2;
    (void)!system(("rm -rf " + opp_path).c_str());
}

TEST_F(OpApiTest, AclSetTensorDynamicInputTest)
{
    int64_t *dims = new int64_t[2];
    aclTensorDesc desc(ACL_FLOAT, 2, dims, ACL_FORMAT_NCHW);
    EXPECT_EQ(aclSetTensorDynamicInput(&desc, "x"), ACL_SUCCESS);
    delete[] dims;
}

TEST_F(OpApiTest, AclSetTensorFormatTest)
{
    int64_t *dims = new int64_t[2];
    aclTensorDesc desc(ACL_FLOAT, 2, dims, ACL_FORMAT_NCHW);
    EXPECT_EQ(aclSetTensorFormat(&desc, ACL_FORMAT_NCHW), ACL_SUCCESS);
    EXPECT_EQ(aclSetTensorShape(&desc, 2, dims), ACL_SUCCESS);
    EXPECT_EQ(aclSetTensorOriginFormat(&desc, ACL_FORMAT_NCHW), ACL_SUCCESS);
    EXPECT_EQ(aclSetTensorOriginShape(&desc, 2, dims), ACL_SUCCESS);
    delete[] dims;
}

TEST_F(OpApiTest, AclSetTensorConstTest)
{
    aclTensorDesc *desc = aclCreateTensorDesc(ACL_FLOAT16, 0, nullptr, ACL_FORMAT_ND);
    void *dataBuffer = (void *)malloc(5);
    size_t length = 0;
    EXPECT_EQ(aclSetTensorConst(desc, dataBuffer, length), ACL_ERROR_INVALID_PARAM);

    length = 1;
    EXPECT_NE(aclSetTensorConst(desc, dataBuffer, length), ACL_ERROR_FAILURE);
    aclDestroyTensorDesc(desc);
    free(dataBuffer);
}

TEST_F(OpApiTest, OpKernelRegistryFailedTest)
{
    const std::string opType;
    std::string kernelId;
    EXPECT_EQ(acl::OpKernelRegistry::GetInstance().GetStubFunc(opType, kernelId), nullptr);
}

TEST_F(OpApiTest, aclopExecuteV2DataIsNull)
{
    const char *opType = "Add";
    int numInputs = 2;

    int64_t shape[]{16, -1};
    aclTensorDesc *inputDesc[2];
    aclTensorDesc *outputDesc[1];
    inputDesc[0] = aclCreateTensorDesc(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    inputDesc[1] = aclCreateTensorDesc(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);

    outputDesc[0] = aclCreateTensorDesc(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);

    aclDataBuffer *inputs[2];
    char tmpBuf[1024] = {0};
    inputs[0] = aclCreateDataBuffer(nullptr, 1024);
    inputs[1] = aclCreateDataBuffer((void *) tmpBuf, 1024);

    aclDataBuffer *outputs[1];
    outputs[0] = aclCreateDataBuffer((void *) tmpBuf, 1024);

    int numOutputs = 1;

    aclopAttr attr;
    string str = "";
    attr.SetAttr("unregst _attrlist", str);
    auto ret = aclopExecuteV2(opType, numInputs, inputDesc, inputs,
                   numOutputs, outputDesc, outputs, &attr, nullptr);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
    aclDestroyDataBuffer(inputs[0]);
    aclDestroyDataBuffer(inputs[1]);
    aclDestroyDataBuffer(outputs[0]);
    inputs[0] = nullptr;
    inputs[1] = nullptr;
    ret = aclopExecuteV2(opType, numInputs, inputDesc, inputs,
                   numOutputs, outputDesc, outputs, &attr, nullptr);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);

    aclDestroyTensorDesc(inputDesc[0]);
    aclDestroyTensorDesc(inputDesc[1]);
    aclDestroyTensorDesc(outputDesc[0]);
}

TEST_F(OpApiTest, AclIsSameTensorNull)
{
    aclTensorDesc *desc = aclCreateTensorDesc(ACL_FLOAT16, 0, nullptr, ACL_FORMAT_ND);
    EXPECT_NE(desc, nullptr);
    EXPECT_EQ(desc->IsSameTensor(nullptr), false);
    bool ret = (desc == nullptr);
    EXPECT_EQ(ret, false);
    aclDestroyTensorDesc(desc);
}

TEST_F(OpApiTest, AclAclopSetAttrDataType)
{
    aclopAttr attr;
    EXPECT_EQ(aclopSetAttrDataType(&attr, "attrName", ACL_FLOAT), ACL_SUCCESS);
    EXPECT_EQ(aclopSetAttrDataType(&attr, "attrName", ACL_DT_UNDEFINED), ACL_SUCCESS);
}

TEST_F(OpApiTest, AclAclopSetAttrListDataType)
{
    aclopAttr attr;
    int numValues = 1;
    const aclDataType values[] { ACL_FLOAT };
    EXPECT_EQ(aclopSetAttrListDataType(&attr, "attrName", numValues, values), ACL_SUCCESS);
    const aclDataType valuesUndefined[] { ACL_DT_UNDEFINED };
    EXPECT_EQ(aclopSetAttrListDataType(&attr, "attrName", numValues, valuesUndefined), ACL_SUCCESS);
}
