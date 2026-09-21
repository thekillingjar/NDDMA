/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "acl/acl.h"
#include "acl/acl_op_compiler.h"
#include "runtime/dev.h"

#include <string>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#define protected public
#define private public

#include "single_op/compile/local_compiler.h"
#include "single_op/compile/op_compile_processor.h"
#include "single_op/compile/op_compile_service.h"
#include "model/acl_resource_manager.h"
#include "utils/array_utils.h"
#include "utils/string_utils.h"
#include "acl_stub.h"

#undef private
#undef protected

#include "framework/generator/ge_generator.h"
#include "ge/ge_api.h"
#include "framework/executor/ge_executor.h"
#include "common/common_inner.h"
#include "graph/ge_context.h"

using namespace testing;
using namespace std;
using namespace acl;
using namespace ge;

using OpDescPtr = std::shared_ptr<OpDesc>;

namespace {
rtError_t  JitEnableFakeFunc(const char *label, const char *key, char *value, const uint32_t maxLen) {
  (void)label;
  (void)key;
  (void)strcpy_s(value, maxLen, "1001");
  return RT_ERROR_NONE;
}

aclError GeFinalizeCallbackFunc() {
    return ACL_SUCCESS;
}

aclError GeFinalizeCallbackFuncFailed() {
    return ACL_ERROR_INVALID_PARAM;
}
}

namespace acl {
    extern GeFinalizeCallback GetGeFinalizeCallback();
    extern void SetGeFinalizeCallback(const GeFinalizeCallback func);
    extern aclError AclOpCompilerFinalizeCallbackFunc(void *userData);
}

class UTEST_ACL_OpCompiler : public testing::Test {
public:
    void SetUp() override
    {
        MockFunctionTest::aclStubInstance().ResetToDefaultMock();
    }
    void TearDown() override
    {
        Mock::VerifyAndClear((void *) (&MockFunctionTest::aclStubInstance()));
    }
};

TEST_F(UTEST_ACL_OpCompiler, InitLocalCompilerTest)
{
    std::map<string, string> options;
    LocalCompiler compiler;
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GEInitialize(_))
        .WillOnce(Return((ge::PARAM_INVALID)))
        .WillRepeatedly(Return(ge::SUCCESS));
    EXPECT_EQ(compiler.Init(options), ACL_ERROR_GE_FAILURE);
    EXPECT_EQ(compiler.Init(options), ACL_SUCCESS);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), Finalize())
        .WillRepeatedly(Return((ge::PARAM_INVALID)));

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GEFinalize())
        .WillRepeatedly(Return((ge::PARAM_INVALID)));
}

TEST_F(UTEST_ACL_OpCompiler, InitLocalCompiler_SetOptionNameMap) {
    std::map<string, string> options;
    LocalCompiler compiler;
    EXPECT_EQ(compiler.Init(options), ACL_SUCCESS);
    EXPECT_EQ(ge::GEContext().GetReadableName(ge::PRECISION_MODE), "ACL_PRECISION_MODE");
}

TEST_F(UTEST_ACL_OpCompiler, OnlineCompileTest)
{
    LocalCompiler compiler;
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), BuildSingleOpModel(_,_,_,_,_,_))
        .WillOnce(Return(ge::PARAM_INVALID))
        .WillRepeatedly(Return(ge::SUCCESS));
    CompileParam param;
    std::shared_ptr<void> modelData = nullptr;
    size_t modelSize = 0;
    EXPECT_NE(compiler.OnlineCompileAndDump(param, modelData, modelSize, nullptr, nullptr), ACL_SUCCESS);
    EXPECT_EQ(compiler.OnlineCompileAndDump(param, modelData, modelSize, nullptr, nullptr), ACL_SUCCESS);
}

TEST_F(UTEST_ACL_OpCompiler, TestMakeCompileParamV3)
{
    CompileParam param;
    AclOp aclOp;
    aclOp.opType = "Add";
    aclOp.numInputs = 2;
    aclOp.numOutputs = 1;
    int64_t shape[]{16, -1};
    const aclTensorDesc *inputDesc[2];
    const aclTensorDesc *outputDesc[1];
    inputDesc[0] = aclCreateTensorDesc(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    inputDesc[1] = aclCreateTensorDesc(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    outputDesc[0] = aclCreateTensorDesc(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    aclMemType memType = ACL_MEMTYPE_HOST;
    aclSetTensorPlaceMent(const_cast<aclTensorDesc *>(inputDesc[0]), memType);
    aclSetTensorPlaceMent(const_cast<aclTensorDesc *>(outputDesc[0]), memType);
    inputDesc[0]->IsHostMemTensor();
    aclOp.inputDesc = inputDesc;
    aclOp.outputDesc = outputDesc;
    aclDataBuffer *inputs[2];
    inputs[0] = aclCreateDataBuffer((void *) 0x1000, 1024);
    inputs[1] = aclCreateDataBuffer((void *) 0x1000, 1024);
    aclDataBuffer *outputs[1];
    outputs[0] = aclCreateDataBuffer((void *) 0x1000, 1024);
    aclOp.inputs = inputs;
    aclOp.outputs = nullptr;
    aclOpCompileFlag compileFlag = ACL_OP_COMPILE_DEFAULT;
    ASSERT_NE(OpCompiler::MakeCompileParam(aclOp, param, compileFlag), ACL_SUCCESS);
    aclDestroyTensorDesc(inputDesc[0]);
    aclDestroyTensorDesc(inputDesc[1]);
    aclDestroyTensorDesc(outputDesc[0]);
    aclDestroyDataBuffer(inputs[0]);
    aclDestroyDataBuffer(inputs[1]);
    aclDestroyDataBuffer(outputs[0]);
}

TEST_F(UTEST_ACL_OpCompiler, MakeCompileParamTest_InvalidDynamicInputName)
{
    CompileParam param;
    AclOp aclOp;
    aclOp.opType = "TransData";
    // invalid dynamic input desc: dims of shape is not equal to dims of shape range
    std::vector<int64_t> dims1 = {-1, 2, 3};
    aclTensorDesc desc1(ACL_FLOAT16, dims1.size(), dims1.data(), ACL_FORMAT_ND);
    int64_t shapeRange1[3][2] = {{1,2}, {2,2}, {3, 3}};
    aclSetTensorShapeRange(&desc1, 3, shapeRange1);
    aclSetTensorDescName(&desc1, "input0");
    aclSetTensorDynamicInput(&desc1, "input0");

    std::vector<int64_t> dims2 = {-1, 2, 3};
    aclTensorDesc desc2(ACL_FLOAT16, dims1.size(), dims1.data(), ACL_FORMAT_ND);
    int64_t shapeRange2[3][2] = {{1,2}, {2,2}, {3, 3}};
    aclSetTensorShapeRange(&desc2, 3, shapeRange2);
    aclSetTensorDescName(&desc2, "input1");
    aclSetTensorDynamicInput(&desc2, "input1");

    std::vector<int64_t> dims3 = {-1, 2, 3};
    aclTensorDesc desc3(ACL_FLOAT16, dims1.size(), dims1.data(), ACL_FORMAT_ND);
    int64_t shapeRange3[3][2] = {{1,2}, {2,2}, {3, 3}};
    aclSetTensorShapeRange(&desc3, 3, shapeRange3);
    aclSetTensorDescName(&desc3, "input2");
    aclSetTensorDynamicInput(&desc3, "input0");

    aclTensorDesc* descArr1[] = {&desc1, &desc2, &desc3};
    aclTensorDesc* descArr2[] = {&desc1};
    aclOp.inputDesc = descArr1;
    aclOp.outputDesc = descArr2;
    aclOp.numInputs = 3;
    aclOp.numOutputs = 1;
    aclOpCompileFlag compileFlag = ACL_OP_COMPILE_DEFAULT;
    EXPECT_EQ(OpCompiler::MakeCompileParam(aclOp, param, compileFlag), ACL_ERROR_INVALID_PARAM);
}

TEST_F(UTEST_ACL_OpCompiler, MakeCompileParamTest)
{
    CompileParam param;
    AclOp aclOp;
    aclOp.opType = "TransData";
    std::vector<int64_t> dims2 = {-2};
    aclTensorDesc desc2(ACL_FLOAT16, dims2.size(), dims2.data(), ACL_FORMAT_ND);
    std::vector<int64_t> dims3 = {-1, 16};
    aclTensorDesc desc3(ACL_FLOAT16, dims3.size(), dims3.data(), ACL_FORMAT_ND);

    aclTensorDesc desc(ACL_FLOAT16, 0, nullptr, ACL_FORMAT_ND);
    int64_t valRange[1][2] = {{1,10}};
    aclSetTensorValueRange(&desc, 1, valRange);
    aclSetTensorDescName(&desc, "xxx");
    aclTensorDesc* descArr[] = {&desc};
    aclOp.inputDesc = descArr;
    aclOp.outputDesc = descArr;
    aclOp.numInputs = 1;
    aclOp.numOutputs = 1;
    aclOpCompileFlag compileFlag = ACL_OP_COMPILE_DEFAULT;
    EXPECT_EQ(OpCompiler::MakeCompileParam(aclOp, param, compileFlag), ACL_SUCCESS);

    aclTensorDesc* descArr1[] = {&desc2};
    aclOp.inputDesc = descArr1;
    EXPECT_EQ(OpCompiler::MakeCompileParam(aclOp, param, compileFlag), ACL_SUCCESS);

    aclTensorDesc* descArr2[] = {&desc3};
    aclOp.inputDesc = descArr2;
    EXPECT_NE(OpCompiler::MakeCompileParam(aclOp, param, compileFlag), ACL_SUCCESS);

    desc.storageFormat = ACL_FORMAT_NCHW;
    aclOp.inputDesc = descArr;
    aclOp.outputDesc = descArr;
    EXPECT_EQ(OpCompiler::MakeCompileParam(aclOp, param, compileFlag), ACL_SUCCESS);

    aclOp.compileType = OP_COMPILE_UNREGISTERED;
    aclopAttr attr;
    string str = "";
    attr.SetAttr("unregst _attrlist", str);
    aclOp.opAttr = &attr;
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), SetShapeRange(_))
            .WillOnce(Return(GRAPH_SUCCESS))
            .WillOnce(Return(GRAPH_SUCCESS))
            .WillOnce(Return(GRAPH_PARAM_INVALID))
            .WillOnce(Return(GRAPH_SUCCESS))
            .WillOnce(Return(GRAPH_PARAM_INVALID))
            .WillRepeatedly(Return(GRAPH_SUCCESS));
    EXPECT_EQ(OpCompiler::MakeCompileParam(aclOp, param, compileFlag), ACL_SUCCESS);
    EXPECT_EQ(OpCompiler::MakeCompileParam(aclOp, param, compileFlag), ACL_ERROR_GE_FAILURE);
    EXPECT_EQ(OpCompiler::MakeCompileParam(aclOp, param, compileFlag), ACL_ERROR_GE_FAILURE);
}

TEST_F(UTEST_ACL_OpCompiler, MakeCompileParamOptionalInputTest)
{
    GeTensorDescPtr desc = nullptr;
    CompileParam param;
    AclOp aclOp;
    aclOp.opType = "testOp";
    aclTensorDesc desc1(ACL_FLOAT16, 0, nullptr, ACL_FORMAT_ND);
    aclTensorDesc desc2(ACL_DT_UNDEFINED, 0, nullptr, ACL_FORMAT_UNDEFINED);
    aclTensorDesc desc3(ACL_FLOAT16, 0, nullptr, ACL_FORMAT_ND);
    aclSetTensorDynamicInput(&desc1, "x");
    aclSetTensorDynamicInput(&desc3, "y");
    aclTensorDesc* descArr[] = {&desc1, &desc2, &desc3};
    aclOp.inputDesc = descArr;
    aclOp.outputDesc = descArr;
    aclOp.numInputs = 3;
    aclOp.numOutputs = 1;
    aclOpCompileFlag compileFlag = ACL_OP_COMPILE_DEFAULT;
    EXPECT_EQ(OpCompiler::MakeCompileParam(aclOp, param, compileFlag), ACL_SUCCESS);

    desc1.storageFormat = ACL_FORMAT_NCHW;
    aclOp.inputDesc = descArr;
    aclOp.outputDesc = descArr;
    EXPECT_EQ(OpCompiler::MakeCompileParam(aclOp, param, compileFlag), ACL_SUCCESS);

    aclOp.compileType = OP_COMPILE_UNREGISTERED;
    aclopAttr attr;
    string str = "";
    attr.SetAttr("unregst _attrlist", str);
    aclOp.opAttr = &attr;
    EXPECT_EQ(OpCompiler::MakeCompileParam(aclOp, param, compileFlag), ACL_SUCCESS);
}

TEST_F(UTEST_ACL_OpCompiler, MakeCompileParamTransTest)
{
    CompileParam param;
    AclOp aclOp;
    aclOp.opType = "TransData";
    std::vector<int64_t> dims2 = {1,2,3,4};
    aclTensorDesc desc2(ACL_FLOAT16, dims2.size(), dims2.data(), ACL_FORMAT_NCHW);
    std::vector<int64_t> dims3 = {1,2,3,16};
    aclTensorDesc desc3(ACL_FLOAT16, dims3.size(), dims3.data(), ACL_FORMAT_NCHW);

    aclTensorDesc* descArrIn[] = {&desc2};
    aclTensorDesc* descArrOut[] = {&desc3};
    aclOp.inputDesc = descArrIn;
    aclOp.outputDesc = descArrOut;
    aclOp.numInputs = 1;
    aclOp.numOutputs = 1;
    aclOpCompileFlag compileFlag = ACL_OP_COMPILE_DEFAULT;
    EXPECT_EQ(OpCompiler::MakeCompileParam(aclOp, param, compileFlag), ACL_SUCCESS);

    aclSetTensorOriginFormat(&desc2, ACL_FORMAT_NC1HWC0);
    EXPECT_EQ(OpCompiler::MakeCompileParam(aclOp, param, compileFlag), ACL_SUCCESS);
}

TEST_F(UTEST_ACL_OpCompiler, MakeCompileParamHostMemTest)
{
    CompileParam param;
    AclOp aclOp;
    aclOp.opType = "TransData";
    std::vector<int64_t> dims2 = {1,2,3,4};
    aclTensorDesc desc1(ACL_FLOAT16, dims2.size(), dims2.data(), ACL_FORMAT_NCHW);
    desc1.memtype = ACL_MEMTYPE_HOST;

    aclTensorDesc desc2(ACL_FLOAT16, dims2.size(), dims2.data(), ACL_FORMAT_NCHW);
    desc2.memtype = ACL_MEMTYPE_HOST_COMPILE_INDEPENDENT;

    std::vector<int64_t> dims3 = {1,2,3,16};
    aclTensorDesc desc3(ACL_FLOAT16, dims3.size(), dims3.data(), ACL_FORMAT_NCHW);

    aclTensorDesc* descArrIn[] = {&desc1, &desc2};
    aclTensorDesc* descArrOut[] = {&desc3};

    aclDataBuffer *inputs[2];
    inputs[0] = aclCreateDataBuffer((void *) 0x0001, 1);
    inputs[1] = aclCreateDataBuffer((void *) 0x0002, 1);
    aclOp.inputs = inputs;
    aclOp.inputDesc = descArrIn;
    aclOp.outputDesc = descArrOut;
    aclOp.numInputs = 2;
    aclOp.numOutputs = 1;
    aclOpCompileFlag compileFlag = ACL_OP_COMPILE_DEFAULT;
    aclError ret1 = OpCompiler::MakeCompileParam(aclOp, param, compileFlag);
    compileFlag = ACL_OP_COMPILE_FUZZ;
    aclError ret2 = OpCompiler::MakeCompileParam(aclOp, param, compileFlag);
    EXPECT_EQ(ret1, ACL_SUCCESS);
    EXPECT_EQ(ret2, ACL_SUCCESS);
    aclDestroyDataBuffer(inputs[0]);
    aclDestroyDataBuffer(inputs[1]);
}

TEST_F(UTEST_ACL_OpCompiler, SetCompileStrategyTest)
{
    std::map<std::string, std::string> options;
    OpCompileService service;
    service.creators_.clear();
    EXPECT_EQ(service.SetCompileStrategy(NATIVE_COMPILER, options), ACL_ERROR_COMPILER_NOT_REGISTERED);

    CompileStrategy strategy = CompileStrategy(3);
    EXPECT_EQ(service.SetCompileStrategy(strategy, options), ACL_ERROR_INVALID_PARAM);
}

TEST_F(UTEST_ACL_OpCompiler, aclSetCompileoptTest)
{
    aclCompileOpt opt1 = ACL_OP_COMPILER_CACHE_DIR;
    char value1[10] = "111";
    EXPECT_EQ(aclSetCompileopt(opt1, value1), ACL_SUCCESS);
}

TEST_F(UTEST_ACL_OpCompiler, aclSetCompileoptTest1)
{
    acl::OpCompileProcessor::GetInstance().globalCompileOpts.clear();
    aclCompileOpt opt1 = ACL_OP_JIT_COMPILE;
    char value1[10] = "0";
    EXPECT_EQ(aclSetCompileopt(opt1, value1), ACL_SUCCESS);

    std::map<std::string, std::string> currentOptions;
    acl::OpCompileProcessor::GetInstance().GetGlobalCompileOpts(currentOptions);
    std::string shape_generalized = "ge.shape_generalized";
    std::string Valuestr =
        currentOptions.find(shape_generalized) != currentOptions.cend() ? currentOptions[shape_generalized] : "";
    EXPECT_EQ(Valuestr, "1");
    EXPECT_EQ(GetGlobalJitCompileFlag(), 0);
}

TEST_F(UTEST_ACL_OpCompiler, aclSetCompileoptTest2)
{
    aclCompileOpt opt1 = ACL_AUTO_TUNE_MODE;
    char value1[10] = "111";
    EXPECT_NE(aclSetCompileopt(opt1, value1), ACL_SUCCESS);
}

TEST_F(UTEST_ACL_OpCompiler, aclSetCompileoptTest3)
{
    acl::OpCompileProcessor::GetInstance().globalCompileOpts.clear();
    aclCompileOpt opt1 = ACL_OP_DETERMINISTIC;
    char value1[10] = "1";
    EXPECT_EQ(aclSetCompileopt(opt1, value1), ACL_SUCCESS);

    std::map<std::string, std::string> currentOptions;
    acl::OpCompileProcessor::GetInstance().GetGlobalCompileOpts(currentOptions);
    std::string Valuestr =
        currentOptions.find("ge.deterministic") != currentOptions.cend() ? currentOptions["ge.deterministic"] : "";
    EXPECT_EQ(Valuestr, "1");
}

TEST_F(UTEST_ACL_OpCompiler, aclGetCompileoptSizeTest)
{
    EXPECT_EQ(aclGetCompileoptSize(ACL_OP_JIT_COMPILE), 256);
    // case: opt is invalid
    EXPECT_EQ(aclGetCompileoptSize((aclCompileOpt)233), 0);
    // case: opt value is not set
    EXPECT_EQ(aclGetCompileoptSize(ACL_OP_DEBUG_OPTION), 0);
}

TEST_F(UTEST_ACL_OpCompiler, aclGetCompileoptNullPtrFailed)
{
    EXPECT_EQ(aclGetCompileopt(ACL_OP_JIT_COMPILE, nullptr, 256), ACL_ERROR_INVALID_PARAM);
}

TEST_F(UTEST_ACL_OpCompiler, aclGetCompileopt_Ok)
{
    std::unique_ptr<char[]> buff(new char[256]);
    // case: opt is not set
    EXPECT_EQ(aclGetCompileopt(ACL_OP_DEBUG_OPTION, buff.get(), 256), ACL_ERROR_API_NOT_SUPPORT);
    // case: opt is set
    std::string optStr = "fake_option_length123";
    EXPECT_EQ(aclSetCompileopt(ACL_OP_DEBUG_OPTION, optStr.c_str()), ACL_SUCCESS);
    auto length = aclGetCompileoptSize(ACL_OP_DEBUG_OPTION);
    EXPECT_EQ(length, optStr.size() + 1);
    EXPECT_EQ(aclGetCompileopt(ACL_OP_DEBUG_OPTION, buff.get(), length), ACL_SUCCESS);
    // case: strncpy_s failed
    size_t maxStrLen = 0x7fffffffUL;
    EXPECT_EQ(aclGetCompileopt(ACL_OP_DEBUG_OPTION, buff.get(), maxStrLen + 1UL), ACL_ERROR_FAILURE);
}

TEST_F(UTEST_ACL_OpCompiler, aclGetCompileoptLengthCheckFailed)
{
    char value;
    EXPECT_EQ(aclGetCompileopt(ACL_OP_JIT_COMPILE, &value, 1), ACL_ERROR_FAILURE);
}

TEST_F(UTEST_ACL_OpCompiler, aclGetCompileoptNotSupportFailed)
{
    char value;
    EXPECT_EQ(aclGetCompileopt(ACL_AICORE_NUM, &value, 256), ACL_ERROR_API_NOT_SUPPORT);
}

TEST_F(UTEST_ACL_OpCompiler, aclGetCompileopt_Ok_GetJitCompileDefaultValue)
{
    std::vector<char> value(256);
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), rtGetSocSpec(_, _, _, _))
        .WillOnce(Return(-1))
        .WillOnce(Invoke(JitEnableFakeFunc));
    // 第一次rtGetSocSpec失败, 返回disable
    EXPECT_EQ(aclGetCompileopt(ACL_OP_JIT_COMPILE, value.data(), 256), ACL_SUCCESS);
    EXPECT_EQ(std::string(value.data()), "disable");

    // 第二次获取到arch=1001, 默认开启enbale
    value.clear();
    EXPECT_EQ(aclGetCompileopt(ACL_OP_JIT_COMPILE, value.data(), 256), ACL_SUCCESS);
    EXPECT_EQ(std::string(value.data()), "enable");
}

TEST_F(UTEST_ACL_OpCompiler, aclGetCompileoptTestDisable)
{
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), aclrtGetSocName())
        .WillRepeatedly(Return("Ascend910B1"));
    const auto size = aclGetCompileoptSize(ACL_OP_JIT_COMPILE);
    char *value_buff = new char[size];
    EXPECT_EQ(aclGetCompileopt(ACL_OP_JIT_COMPILE, value_buff, size), ACL_SUCCESS);
    EXPECT_STREQ(value_buff, "disable");
    delete[] value_buff;
}

TEST_F(UTEST_ACL_OpCompiler, TestSetOptionFail)
{
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), aclrtGetSocName())
        .WillOnce(Return(nullptr));
    EXPECT_EQ(acl::OpCompileProcessor::GetInstance().SetOption(), 500000);
}

TEST_F(UTEST_ACL_OpCompiler, aclSetCompileoptTest4)
{
    acl::OpCompileProcessor::GetInstance().globalCompileOpts.clear();
    aclCompileOpt opt1 = ACL_CUSTOMIZE_DTYPES;
    char value1[10] = "1";
    EXPECT_EQ(aclSetCompileopt(opt1, value1), ACL_SUCCESS);

    std::map<std::string, std::string> currentOptions;
    acl::OpCompileProcessor::GetInstance().GetGlobalCompileOpts(currentOptions);
    std::string Valuestr =
        currentOptions.find("ge.customizeDtypes") != currentOptions.cend() ? currentOptions["ge.customizeDtypes"] : "";
    EXPECT_EQ(Valuestr, "1");
}

TEST_F(UTEST_ACL_OpCompiler, ReadOpModelFromFileTest)
{
    const std::string path;
    OpModel opModel;

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), ReadBytesFromBinaryFile(_, _, _))
        .WillOnce(Return((false)));
    EXPECT_NE(ReadOpModelFromFile(path, opModel), ACL_SUCCESS);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), ReadBytesFromBinaryFile(_, _, _))
        .WillOnce(Return((true)));
    EXPECT_EQ(ReadOpModelFromFile(path, opModel), ACL_SUCCESS);
}

TEST_F(UTEST_ACL_OpCompiler, aclopSetCompileFlagTest)
{
    EXPECT_EQ(aclopSetCompileFlag(ACL_OP_COMPILE_DEFAULT), ACL_SUCCESS);
}

TEST_F(UTEST_ACL_OpCompiler, TestOnlineCompile02)
{
    LocalCompiler compiler;
    CompileParam param;
    std::shared_ptr<void> modelData = nullptr;
    size_t modelSize = 0;

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), Initialize(_, _))
        .WillOnce(Return(ge::PARAM_INVALID));
    EXPECT_NE(compiler.OnlineCompileAndDump(param, modelData, modelSize, nullptr, nullptr), ACL_SUCCESS);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), Initialize(_, _))
        .WillOnce(Return(ge::SUCCESS));
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), BuildSingleOpModel(_,_,_,_,_,_))
        .WillOnce(Return(ge::SUCCESS));
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), Ge_Generator_Finalize())
        .WillOnce(Return(ge::PARAM_INVALID))
        .WillOnce(Return(ge::SUCCESS));
    EXPECT_NE(compiler.OnlineCompileAndDump(param, modelData, modelSize, nullptr, nullptr), ACL_SUCCESS);
}

TEST_F(UTEST_ACL_OpCompiler, aclopCompile_Check_JitCompileDefaultValue)
{
    const char *opType = "Add";
    int numInputs = 2;

    int64_t shape[]{16, 2};
    aclTensorDesc *inputDesc[2];
    aclTensorDesc *outputDesc[1];
    inputDesc[0] = aclCreateTensorDesc(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    inputDesc[1] = aclCreateTensorDesc(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    outputDesc[0] = aclCreateTensorDesc(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);

    inputDesc[0]->shapeRange.emplace_back(std::make_pair(16, 16));
    inputDesc[0]->shapeRange.emplace_back(std::make_pair(2, 2));
    inputDesc[1]->shapeRange.emplace_back(std::make_pair(16, 16));
    inputDesc[1]->shapeRange.emplace_back(std::make_pair(2, 2));

    int numOutputs = 1;

    aclopAttr attr;
    string str = "";
    attr.SetAttr("unregst _attrlist", str);

    const char *opPath = "tests/";
    aclopEngineType engineType = ACL_ENGINE_AICORE;
    aclopCompileType compileFlag = ACL_COMPILE_UNREGISTERED;

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), Initialize(_, _))
    .WillRepeatedly(Return(ge::SUCCESS));
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), BuildSingleOpModel(_,_,_,_,_,_))
    .WillRepeatedly(Return(ge::SUCCESS));
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), Ge_Generator_Finalize())
    .WillRepeatedly(Return(ge::SUCCESS));

    aclError ret = aclopCompile(opType, numInputs, inputDesc,
                                numOutputs, outputDesc, &attr, engineType, compileFlag,
                                opPath);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);

    auto options = ge::GetThreadLocalContext().GetAllGraphOptions();
    ASSERT_NE(options.find("ge.jit_compile"), options.cend());
    EXPECT_STREQ(options["ge.jit_compile"].c_str(), "0");
    EXPECT_EQ(GetGlobalCompileFlag(), 0);
    ASSERT_NE(options.find("ge.shape_generalized"), options.cend());
    EXPECT_STREQ(options["ge.shape_generalized"].c_str(), "1");

    compileFlag = ACL_COMPILE_UNREGISTERED;
    opPath = nullptr;
    ret = aclopCompile(opType, numInputs, inputDesc,
                       numOutputs, outputDesc, &attr, engineType, compileFlag,
                       opPath);
    ASSERT_NE(options.find("ge.jit_compile"), options.cend());
    EXPECT_STREQ(options["ge.jit_compile"].c_str(), "0");
    EXPECT_EQ(GetGlobalCompileFlag(), 0);
    ASSERT_NE(options.find("ge.shape_generalized"), options.cend());
    EXPECT_STREQ(options["ge.shape_generalized"].c_str(), "1");
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
    aclDestroyTensorDesc(inputDesc[0]);
    aclDestroyTensorDesc(inputDesc[1]);
    aclDestroyTensorDesc(outputDesc[0]);
}

TEST_F(UTEST_ACL_OpCompiler, aclopCompile_Check_UseUserSetJitCompileValue)
{
    // acl::InitSocVersion();
    const char *opType = "Add";
    int numInputs = 2;

    int64_t shape[]{16, 2};
    aclTensorDesc *inputDesc[2];
    aclTensorDesc *outputDesc[1];
    inputDesc[0] = aclCreateTensorDesc(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    inputDesc[1] = aclCreateTensorDesc(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    outputDesc[0] = aclCreateTensorDesc(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);

    inputDesc[0]->shapeRange.emplace_back(std::make_pair(16, 16));
    inputDesc[0]->shapeRange.emplace_back(std::make_pair(2, 2));
    inputDesc[1]->shapeRange.emplace_back(std::make_pair(16, 16));
    inputDesc[1]->shapeRange.emplace_back(std::make_pair(2, 2));

    int numOutputs = 1;

    aclopAttr attr;
    string str = "";
    attr.SetAttr("unregst _attrlist", str);

    const char *opPath = "tests/";
    aclopEngineType engineType = ACL_ENGINE_AICORE;
    aclopCompileType compileFlag = ACL_COMPILE_UNREGISTERED;

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), Initialize(_, _))
    .WillRepeatedly(Return(ge::SUCCESS));
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), BuildSingleOpModel(_,_,_,_,_,_))
    .WillRepeatedly(Return(ge::SUCCESS));
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), Ge_Generator_Finalize())
    .WillRepeatedly(Return(ge::SUCCESS));

    aclSetCompileopt(ACL_OP_JIT_COMPILE, "disalbe");
    aclError ret = aclopCompile(opType, numInputs, inputDesc,
                                numOutputs, outputDesc, &attr, engineType, compileFlag,
                                opPath);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);

    auto options = ge::GetThreadLocalContext().GetAllGraphOptions();
    ASSERT_NE(options.find("ge.jit_compile"), options.cend());
    EXPECT_STREQ(options["ge.jit_compile"].c_str(), "0");
    for (const auto desc : inputDesc) {
        aclDestroyTensorDesc(desc);
    }
    aclDestroyTensorDesc(outputDesc[0]);
}

TEST_F(UTEST_ACL_OpCompiler, TestInitLocalCompiler02)
{
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GEInitialize(_))
        .WillOnce(Return(ge::SUCCESS));

    std::map<string, string> options;
    LocalCompiler compiler;
    ASSERT_EQ(compiler.Init(options), ACL_SUCCESS);
    ASSERT_NE(compiler.Init(options), ACL_SUCCESS);
}

TEST_F(UTEST_ACL_OpCompiler, AclIsDigitTest)
{
    std::string str = "";
    EXPECT_EQ(acl::StringUtils::IsDigit(str), false);
}

TEST_F(UTEST_ACL_OpCompiler, aclopCompileAndExecuteTest)
{
    const char *opType = "Add";
    int numInputs = 2;

    int64_t shape[]{16, -1};
    const aclTensorDesc *inputDesc[2];
    const aclTensorDesc *outputDesc[1];
    inputDesc[0] = aclCreateTensorDesc(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    inputDesc[1] = aclCreateTensorDesc(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);

    outputDesc[0] = aclCreateTensorDesc(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);

    aclDataBuffer *inputs[2];
    char tmpBuf[1024] = {0};
    inputs[0] = aclCreateDataBuffer((void *) tmpBuf, 1024);
    inputs[1] = aclCreateDataBuffer((void *) tmpBuf, 1024);

    aclDataBuffer *outputs[1];
    outputs[0] = aclCreateDataBuffer((void *) tmpBuf, 1024);

    int numOutputs = 1;

    aclopAttr attr;
    string str = "";
    attr.SetAttr("unregst _attrlist", str);
    aclopEngineType engineType = ACL_ENGINE_AICORE;
    aclopCompileType compileFlag = (aclCompileType)(3);
    const char *opPath = "./test";
    aclrtStream stream = nullptr;

    aclError ret = aclopCompileAndExecute(opType, numInputs, inputDesc, inputs,
                numOutputs, outputDesc, outputs, &attr, engineType, compileFlag,
                opPath, stream);
    EXPECT_EQ(ret, ACL_ERROR_API_NOT_SUPPORT);

    compileFlag = ACL_COMPILE_UNREGISTERED;
    ret = aclopCompileAndExecute(opType, numInputs, inputDesc, inputs,
                numOutputs, outputDesc, outputs, &attr, engineType, compileFlag,
                opPath, stream);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);

    compileFlag = ACL_COMPILE_SYS;
    ret = aclopCompileAndExecute(opType, numInputs, inputDesc, inputs,
                numOutputs, outputDesc, outputs, &attr, engineType, compileFlag,
                opPath, stream);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);

    opPath = "tests/ge/ut/";
    ret = aclopCompileAndExecute(opType, numInputs, inputDesc, inputs,
                numOutputs, outputDesc, outputs, &attr, engineType, compileFlag,
                opPath, stream);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
    aclDestroyTensorDesc(inputDesc[0]);
    aclDestroyTensorDesc(inputDesc[1]);
    aclDestroyTensorDesc(outputDesc[0]);
    aclDestroyDataBuffer(inputs[0]);
    aclDestroyDataBuffer(inputs[1]);
    aclDestroyDataBuffer(outputs[0]);
}

TEST_F(UTEST_ACL_OpCompiler, aclopCompileAndExecuteCheckInput)
{
    const char *opType = "Add";
    int numInputs = 2;

    int64_t shape[]{16, -1};
    const aclTensorDesc *inputDesc[2];
    const aclTensorDesc *outputDesc[1];
    inputDesc[0] = aclCreateTensorDesc(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    inputDesc[1] = aclCreateTensorDesc(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);

    outputDesc[0] = aclCreateTensorDesc(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);

    aclDataBuffer *inputs[2];
    char tmpBuf[1024] = {0};
    inputs[0] = aclCreateDataBuffer((void *) tmpBuf, 1024);
    inputs[1] = aclCreateDataBuffer((void *) tmpBuf, 1024);

    aclDataBuffer *outputs[1];
    outputs[0] = aclCreateDataBuffer((void *) tmpBuf, 1024);

    int numOutputs = 1;

    aclopAttr attr;
    string str = "";
    attr.SetAttr("unregst _attrlist", str);
    aclopEngineType engineType = ACL_ENGINE_AICORE;
    aclopCompileType compileFlag = ACL_COMPILE_UNREGISTERED;
    const char *opPath = "path1";
    aclrtStream stream = nullptr;

    aclError ret = aclopCompileAndExecute(opType, numInputs, nullptr, inputs,
                numOutputs, outputDesc, outputs, &attr, engineType, compileFlag,
                opPath, stream);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);

    ret = aclopCompileAndExecute(opType, numInputs, inputDesc, nullptr,
                numOutputs, outputDesc, outputs, &attr, engineType, compileFlag,
                opPath, stream);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
    aclDestroyTensorDesc(inputDesc[0]);
    aclDestroyTensorDesc(inputDesc[1]);
    aclDestroyTensorDesc(outputDesc[0]);
    aclDestroyDataBuffer(inputs[0]);
    aclDestroyDataBuffer(inputs[1]);
    aclDestroyDataBuffer(outputs[0]);
}


TEST_F(UTEST_ACL_OpCompiler, aclopCompileAndExecuteV2Test)
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
    inputs[0] = aclCreateDataBuffer((void *) tmpBuf, 1024);
    inputs[1] = aclCreateDataBuffer((void *) tmpBuf, 1024);

    aclDataBuffer *outputs[1];
    outputs[0] = aclCreateDataBuffer((void *) tmpBuf, 1024);

    int numOutputs = 1;

    aclopAttr attr;
    string str = "";
    attr.SetAttr("unregst _attrlist", str);
    aclopEngineType engineType = ACL_ENGINE_AICORE;
    aclopCompileType compileFlag = (aclCompileType)(3);
    const char *opPath = "path1";
    aclrtStream stream = nullptr;

    aclError ret = aclopCompileAndExecuteV2(opType, numInputs, inputDesc, inputs,
                numOutputs, outputDesc, outputs, &attr, engineType, compileFlag,
                opPath, stream);
    EXPECT_EQ(ret, ACL_ERROR_API_NOT_SUPPORT);

    compileFlag = ACL_COMPILE_UNREGISTERED;
    ret = aclopCompileAndExecuteV2(opType, numInputs, inputDesc, inputs,
                numOutputs, outputDesc, outputs, &attr, engineType, compileFlag,
                opPath, stream);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);

    compileFlag = ACL_COMPILE_SYS;
    ret = aclopCompileAndExecuteV2(opType, numInputs, inputDesc, inputs,
                numOutputs, outputDesc, outputs, &attr, engineType, compileFlag,
                opPath, stream);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);

    opPath = "tests/ge/ut/";
    ret = aclopCompileAndExecuteV2(opType, numInputs, inputDesc, inputs,
                numOutputs, outputDesc, outputs, &attr, engineType, compileFlag,
                opPath, stream);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
    aclDestroyTensorDesc(inputDesc[0]);
    aclDestroyTensorDesc(inputDesc[1]);
    aclDestroyTensorDesc(outputDesc[0]);
    aclDestroyDataBuffer(inputs[0]);
    aclDestroyDataBuffer(inputs[1]);
    aclDestroyDataBuffer(outputs[0]);
}

TEST_F(UTEST_ACL_OpCompiler, aclopCompileAndExecuteV2CheckInput)
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
    inputs[0] = aclCreateDataBuffer((void *) tmpBuf, 1024);
    inputs[1] = aclCreateDataBuffer((void *) tmpBuf, 1024);

    aclDataBuffer *outputs[1];
    outputs[0] = aclCreateDataBuffer((void *) tmpBuf, 1024);

    int numOutputs = 1;

    aclopAttr attr;
    string str = "";
    attr.SetAttr("unregst _attrlist", str);
    aclopEngineType engineType = ACL_ENGINE_AICORE;
    aclopCompileType compileFlag = ACL_COMPILE_UNREGISTERED;
    const char *opPath = "path1";
    aclrtStream stream = nullptr;

    aclError ret = aclopCompileAndExecuteV2(opType, numInputs, nullptr, inputs,
                numOutputs, outputDesc, outputs, &attr, engineType, compileFlag,
                opPath, stream);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);

    ret = aclopCompileAndExecuteV2(opType, numInputs, inputDesc, nullptr,
                numOutputs, outputDesc, outputs, &attr, engineType, compileFlag,
                opPath, stream);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
    aclDestroyTensorDesc(inputDesc[0]);
    aclDestroyTensorDesc(inputDesc[1]);
    aclDestroyTensorDesc(outputDesc[0]);
    aclDestroyDataBuffer(inputs[0]);
    aclDestroyDataBuffer(inputs[1]);
    aclDestroyDataBuffer(outputs[0]);
}

TEST_F(UTEST_ACL_OpCompiler, aclGenGraphAndDumpForOpTest)
{
    int numInputs = 2;

    int64_t shape[]{16, 1};
    const aclTensorDesc *inputDesc[2];
    const aclTensorDesc *outputDesc[1];
    inputDesc[0] = aclCreateTensorDesc(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    inputDesc[1] = aclCreateTensorDesc(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);

    outputDesc[0] = aclCreateTensorDesc(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);

    aclDataBuffer *inputs[2];
    inputs[0] = aclCreateDataBuffer((void *) 0x1000, 1024);
    inputs[1] = aclCreateDataBuffer((void *) 0x1000, 1024);

    aclDataBuffer *outputs[1];
    outputs[0] = aclCreateDataBuffer((void *) 0x1000, 1024);

    int numOutputs = 1;

    aclopAttr attr;
    string str = "";
    attr.SetAttr("unregst _attrlist", str);
    aclopEngineType engineType = ACL_ENGINE_AICORE;
    aclGraphDumpOption *opt = aclCreateGraphDumpOpt();
    EXPECT_NE(opt, nullptr);
    aclError ret = aclGenGraphAndDumpForOp("Add", numInputs, inputDesc, inputs,
                numOutputs, outputDesc, outputs, &attr, engineType, "./", opt);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), aclrtGetCurrentContext(_))
        .WillRepeatedly(Return(ACL_SUCCESS));
    ret = aclGenGraphAndDumpForOp("Add", numInputs, inputDesc, inputs,
                numOutputs, outputDesc, outputs, &attr, engineType, "./", nullptr);
    EXPECT_EQ(ret, ACL_SUCCESS);
    aclDestroyGraphDumpOpt(opt);
    aclDestroyTensorDesc(inputDesc[0]);
    aclDestroyTensorDesc(inputDesc[1]);
    aclDestroyTensorDesc(outputDesc[0]);
    aclDestroyDataBuffer(inputs[0]);
    aclDestroyDataBuffer(inputs[1]);
    aclDestroyDataBuffer(outputs[0]);
}

TEST_F(UTEST_ACL_OpCompiler, aclopCompile)
{
    const char *opType = "Add";
    int numInputs = 2;

    int64_t shape[]{16, -1};
    const aclTensorDesc *inputDesc[2];
    const aclTensorDesc *outputDesc[1];
    inputDesc[0] = aclCreateTensorDesc(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    inputDesc[1] = aclCreateTensorDesc(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    outputDesc[0] = aclCreateTensorDesc(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);

    int numOutputs = 1;

    aclopAttr attr;
    string str = "";
    attr.SetAttr("unregst _attrlist", str);

    const char *opPath = "tests/";
    aclopEngineType engineType = ACL_ENGINE_AICORE;
    aclopCompileType compileFlag = ACL_COMPILE_UNREGISTERED;

    aclError ret = aclopCompile(opType, numInputs, inputDesc,
                numOutputs, outputDesc, &attr, engineType, compileFlag,
                opPath);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);

    compileFlag = ACL_COMPILE_UNREGISTERED;
    opPath = nullptr;
    ret = aclopCompile(opType, numInputs, inputDesc,
                numOutputs, outputDesc, &attr, engineType, compileFlag,
                opPath);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
    aclDestroyTensorDesc(inputDesc[0]);
    aclDestroyTensorDesc(inputDesc[1]);
    aclDestroyTensorDesc(outputDesc[0]);
}

TEST_F(UTEST_ACL_OpCompiler, aclInitCallbackTest)
{
    SetGeFinalizeCallback(nullptr);
    EXPECT_EQ(AclOpCompilerFinalizeCallbackFunc(nullptr), ACL_SUCCESS);

    SetGeFinalizeCallback(GeFinalizeCallbackFunc);
    EXPECT_EQ(AclOpCompilerFinalizeCallbackFunc(nullptr), ACL_SUCCESS);

    SetGeFinalizeCallback(GeFinalizeCallbackFuncFailed);
    EXPECT_EQ(AclOpCompilerFinalizeCallbackFunc(nullptr), ACL_SUCCESS);
}