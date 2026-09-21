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

#include "acl/acl.h"
#include "types/op_attr.h"
#include "graph/ge_attr_value.h"
#include "utils/attr_utils.h"
#include "acl_stub.h"

using namespace testing;
using namespace acl;
using namespace ge;

namespace acl {
    namespace attr_utils {
        extern void GeAttrValueToDigest(size_t &digest, const ge::GeAttrValue &val);
        extern std::string GeAttrValueToString(const ge::GeAttrValue &val);
    }
}

class UTEST_ACL_OpAttr : public testing::Test {
protected:
    virtual void SetUp() {
        MockFunctionTest::aclStubInstance().ResetToDefaultMock();
    }

    virtual void TearDown() {
        Mock::VerifyAndClear((void *)(&MockFunctionTest::aclStubInstance()));
    }
};

TEST_F(UTEST_ACL_OpAttr, OpAttrEqualsTest)
{
    aclopAttr *opAttr1 = aclopCreateAttr();
    ASSERT_FALSE(attr_utils::OpAttrEquals(nullptr, opAttr1));
    EXPECT_EQ(aclopSetAttrString(opAttr1, "string", "string"), ACL_SUCCESS);
    ASSERT_TRUE(attr_utils::OpAttrEquals(opAttr1, opAttr1));
    aclopAttr *opAttr2 = aclopCreateAttr();
    ASSERT_FALSE(attr_utils::OpAttrEquals(opAttr1, opAttr2));
    EXPECT_EQ(aclopSetAttrString(opAttr2, "string1", "string1"), ACL_SUCCESS);
    ASSERT_FALSE(attr_utils::OpAttrEquals(opAttr1, opAttr2));
    aclopDestroyAttr(opAttr1);
    aclopDestroyAttr(opAttr2);
}

TEST_F(UTEST_ACL_OpAttr, SetScalarAttrTest)
{
    aclopAttr *opAttr = aclopCreateAttr();
    EXPECT_EQ(aclopSetAttrString(opAttr, "string", "string"), ACL_SUCCESS);
    ASSERT_FALSE(opAttr->DebugString().empty());
    EXPECT_EQ(aclopSetAttrInt(opAttr, "666", 666), ACL_SUCCESS);
    ASSERT_FALSE(opAttr->DebugString().empty());
    EXPECT_EQ(aclopSetAttrInt(opAttr, "666666", 666666), ACL_SUCCESS);
    ASSERT_FALSE(opAttr->DebugString().empty());
    EXPECT_EQ(aclopSetAttrBool(opAttr, "false", false), ACL_SUCCESS);
    ASSERT_FALSE(opAttr->DebugString().empty());
    EXPECT_EQ(aclopSetAttrBool(opAttr, "true", true), ACL_SUCCESS);
    ASSERT_FALSE(opAttr->DebugString().empty());
    EXPECT_EQ(aclopSetAttrFloat(opAttr, "float", 1.0), ACL_SUCCESS);
    ASSERT_FALSE(opAttr->DebugString().empty());

    for (auto &it : opAttr->Attrs()) {
        ASSERT_TRUE(attr_utils::AttrValueEquals(it.second, it.second));
    }
    aclopDestroyAttr(opAttr);
}

TEST_F(UTEST_ACL_OpAttr, SetListAttrTest)
{
    aclopAttr *opAttr = aclopCreateAttr();
    const char *string1 = "string1";
    const char *string2 = "string2";
    const char *argv[2] = {string1, string2};

    int64_t intList[3]{1, 2, 3};
    uint8_t boolList[2]{false, true};
    float floatList[2]{1.0, 0.0};
    EXPECT_EQ(aclopSetAttrListString(opAttr, "stringList", 2, argv), ACL_SUCCESS);
    ASSERT_FALSE(opAttr->DebugString().empty());
    EXPECT_EQ(aclopSetAttrListBool(opAttr, "boolList", 2, boolList), ACL_SUCCESS);
    ASSERT_FALSE(opAttr->DebugString().empty());
    EXPECT_EQ(aclopSetAttrListInt(opAttr, "intList", 3, intList), ACL_SUCCESS);
    ASSERT_FALSE(opAttr->DebugString().empty());
    EXPECT_EQ(aclopSetAttrListFloat(opAttr, "floatList", 2, floatList), ACL_SUCCESS);
    ASSERT_FALSE(opAttr->DebugString().empty());

    for (auto &it : opAttr->Attrs()) {
        ASSERT_TRUE(attr_utils::AttrValueEquals(it.second, it.second));
    }
    aclopDestroyAttr(opAttr);
}

TEST_F(UTEST_ACL_OpAttr, SetListListAttrTest)
{
    aclopAttr *opAttr = aclopCreateAttr();

    int64_t value1[2] = {1, 2};
    int64_t value2[3] = {4, 5, 6};
    const int64_t *values[2] = {value1, value2};
    int numValues[2] = {2, 3};

    EXPECT_EQ(aclopSetAttrListListInt(opAttr, "ListListInt", 2, numValues, values), ACL_SUCCESS);
    ASSERT_FALSE(opAttr->DebugString().empty());

    for (auto &it : opAttr->Attrs()) {
        ASSERT_TRUE(attr_utils::AttrValueEquals(it.second, it.second));
    }

    aclopDestroyAttr(opAttr);
}

TEST_F(UTEST_ACL_OpAttr, IsSameValueTest)
{
    ge::GeAttrValue tensorValue;
    ge::GeTensor geTensor;

    tensorValue.SetValue<ge::GeTensor>(geTensor);
    std::map<AttrRangeType, ge::GeAttrValue> attrMap;
    std::map<AttrRangeType, ge::GeAttrValue> attrMap2;
    auto ret = acl::attr_utils::IsSameValueRange(attrMap, attrMap2);

    // size of map is not equal;
    attrMap[AttrRangeType::VALUE_TYPE] = tensorValue;
    ret = acl::attr_utils::IsSameValueRange(attrMap, attrMap2);
    EXPECT_EQ(ret, false);

    attrMap2[AttrRangeType::VALUE_TYPE] = tensorValue;
    ret = acl::attr_utils::IsSameValueRange(attrMap, attrMap2);
    EXPECT_EQ(ret, true);

    std::vector<std::vector<int64_t>> range = {{1,2}, {1,2}};
    tensorValue.SetValue(range);

    attrMap[AttrRangeType::RANGE_TYPE] = tensorValue;
    attrMap2[AttrRangeType::RANGE_TYPE] = tensorValue;
    ret = acl::attr_utils::IsSameValueRange(attrMap, attrMap2);
    EXPECT_EQ(ret, true);
}

TEST_F(UTEST_ACL_OpAttr, ValueRangeCheckTest)
{
    ge::GeAttrValue tensorValue;
    ge::GeTensor geTensor;

    tensorValue.SetValue<ge::GeTensor>(geTensor);
    std::map<AttrRangeType, ge::GeAttrValue> attrMap;

    attrMap[AttrRangeType::VALUE_TYPE] = tensorValue;
    uint8_t *data[8] = {0};
    aclDataBuffer *dataBuffer = aclCreateDataBuffer(reinterpret_cast<void *>(data), 8);
    auto ret = acl::attr_utils::ValueRangeCheck(attrMap, dataBuffer, ACL_UINT8);
    EXPECT_EQ(ret, true);

    dataBuffer->length = 10;
    ret = acl::attr_utils::ValueRangeCheck(attrMap, dataBuffer, ACL_UINT8);
    EXPECT_EQ(ret, false);
    dataBuffer->data = nullptr;

    ret = acl::attr_utils::ValueRangeCheck(attrMap, dataBuffer, ACL_UINT8);
    EXPECT_EQ(ret, true);
    EXPECT_EQ(aclDestroyDataBuffer(dataBuffer), ACL_SUCCESS);
}

TEST_F(UTEST_ACL_OpAttr, ValueRangeCheckTest1)
{
    ge::GeAttrValue tensorValue;
    ge::GeTensor geTensor;

    tensorValue.SetValue<ge::GeTensor>(geTensor);
    std::map<AttrRangeType, ge::GeAttrValue> attrMap;
    std::vector<std::vector<int64_t>> range = {{1,2}, {1,2}};
    tensorValue.SetValue(range);
    // attrMap[VALUE_TYPE] = tensorValue;
    attrMap[AttrRangeType::RANGE_TYPE] = tensorValue;
    uint8_t *data[8] = {0};
    aclDataBuffer *dataBuffer = aclCreateDataBuffer(reinterpret_cast<void *>(data), 8);
    auto ret = acl::attr_utils::ValueRangeCheck(attrMap, dataBuffer, ACL_UINT8);
    EXPECT_EQ(ret, false);
    std::vector<std::vector<float32_t>> range1 = {{1.0, 2.0}, {1.0, 2.0}};
    tensorValue.SetValue(range1);
    attrMap[AttrRangeType::RANGE_TYPE] = tensorValue;
    ret = acl::attr_utils::ValueRangeCheck(attrMap, dataBuffer, ACL_FLOAT);
    EXPECT_EQ(ret, false);
    std::vector<int64_t> range2 = {1, 2};
    tensorValue.SetValue(range2);
    attrMap[AttrRangeType::RANGE_TYPE] = tensorValue;
    ret = acl::attr_utils::ValueRangeCheck(attrMap, dataBuffer, ACL_UINT8);
    EXPECT_EQ(ret, false);
    std::vector<float32_t> range3 = {1.0, 2.0};
    tensorValue.SetValue(range3);
    attrMap[AttrRangeType::RANGE_TYPE] = tensorValue;
    ret = acl::attr_utils::ValueRangeCheck(attrMap, dataBuffer, ACL_FLOAT);
    EXPECT_EQ(ret, false);
    std::vector<std::vector<int64_t>> range4 = {{-1, 2}, {-1 ,2}, {-1, 2}, {-1, 2},
                                               {-1, 2}, {-1, 2}, {-1, 2}, {-1, 2}};
    tensorValue.SetValue(range4);
    attrMap[AttrRangeType::RANGE_TYPE] = tensorValue;
    ret = acl::attr_utils::ValueRangeCheck(attrMap, dataBuffer, ACL_UINT8);
    EXPECT_EQ(ret, true);
    EXPECT_EQ(aclDestroyDataBuffer(dataBuffer), ACL_SUCCESS);
}

TEST_F(UTEST_ACL_OpAttr, ValueRangeCheckTest2)
{
    ge::GeAttrValue tensorValue;
    ge::GeTensor geTensor;

    tensorValue.SetValue<ge::GeTensor>(geTensor);
    std::map<AttrRangeType, ge::GeAttrValue> attrMap;
    attrMap[AttrRangeType::VALUE_TYPE] = tensorValue;

    uint16_t *data[8] = {0};
    aclDataBuffer *dataBuffer = aclCreateDataBuffer(reinterpret_cast<void *>(data), 8);
    auto ret = acl::attr_utils::ValueRangeCheck(attrMap, dataBuffer, ACL_UINT16);
    EXPECT_EQ(ret, true);
    dataBuffer->length = 10;
    ret = acl::attr_utils::ValueRangeCheck(attrMap, dataBuffer, ACL_UINT16);
    EXPECT_EQ(ret, false);
    dataBuffer->data = nullptr;
    ret = acl::attr_utils::ValueRangeCheck(attrMap, dataBuffer, ACL_UINT16);
    EXPECT_EQ(ret, true);
    EXPECT_EQ(aclDestroyDataBuffer(dataBuffer), ACL_SUCCESS);

    int32_t *data1[8] = {0};
    aclDataBuffer *dataBuffer1 = aclCreateDataBuffer(reinterpret_cast<void *>(data1), 8);
    ret = acl::attr_utils::ValueRangeCheck(attrMap, dataBuffer1, ACL_INT32);
    EXPECT_EQ(ret, true);
    dataBuffer1->length = 10;
    ret = acl::attr_utils::ValueRangeCheck(attrMap, dataBuffer1, ACL_INT32);
    EXPECT_EQ(ret, false);
    dataBuffer1->data = nullptr;
    ret = acl::attr_utils::ValueRangeCheck(attrMap, dataBuffer1, ACL_INT32);
    EXPECT_EQ(ret, true);
    EXPECT_EQ(aclDestroyDataBuffer(dataBuffer1), ACL_SUCCESS);

    uint32_t *data2[8] = {0};
    aclDataBuffer *dataBuffer2 = aclCreateDataBuffer(reinterpret_cast<void *>(data2), 8);
    ret = acl::attr_utils::ValueRangeCheck(attrMap, dataBuffer2, ACL_UINT32);
    EXPECT_EQ(ret, true);
    dataBuffer2->length = 10;
    ret = acl::attr_utils::ValueRangeCheck(attrMap, dataBuffer2, ACL_UINT32);
    EXPECT_EQ(ret, false);
    dataBuffer2->data = nullptr;
    ret = acl::attr_utils::ValueRangeCheck(attrMap, dataBuffer2, ACL_UINT32);
    EXPECT_EQ(ret, true);
    EXPECT_EQ(aclDestroyDataBuffer(dataBuffer2), ACL_SUCCESS);

    int64_t *data3[8] = {0};
    aclDataBuffer *dataBuffer3 = aclCreateDataBuffer(reinterpret_cast<void *>(data3), 8);
    ret = acl::attr_utils::ValueRangeCheck(attrMap, dataBuffer3, ACL_INT64);
    EXPECT_EQ(ret, true);
    dataBuffer3->length = 10;
    ret = acl::attr_utils::ValueRangeCheck(attrMap, dataBuffer3, ACL_INT64);
    EXPECT_EQ(ret, false);
    dataBuffer3->data = nullptr;
    ret = acl::attr_utils::ValueRangeCheck(attrMap, dataBuffer3, ACL_INT64);
    EXPECT_EQ(ret, true);
    EXPECT_EQ(aclDestroyDataBuffer(dataBuffer3), ACL_SUCCESS);

    uint64_t *data4[8] = {0};
    aclDataBuffer *dataBuffer4 = aclCreateDataBuffer(reinterpret_cast<void *>(data4), 8);
    ret = acl::attr_utils::ValueRangeCheck(attrMap, dataBuffer4, ACL_UINT64);
    EXPECT_EQ(ret, true);
    dataBuffer4->length = 10;
    ret = acl::attr_utils::ValueRangeCheck(attrMap, dataBuffer4, ACL_UINT64);
    EXPECT_EQ(ret, false);
    dataBuffer4->data = nullptr;
    ret = acl::attr_utils::ValueRangeCheck(attrMap, dataBuffer4, ACL_UINT64);
    EXPECT_EQ(ret, true);
    EXPECT_EQ(aclDestroyDataBuffer(dataBuffer4), ACL_SUCCESS);
}

TEST_F(UTEST_ACL_OpAttr, GeAttrValueToDigestTest)
{
    size_t digest = 0;
    ge::GeAttrValue value;
    value.SetValue<string>("hello");
    attr_utils::GeAttrValueToDigest(digest, value);

    value.SetValue<bool>(false);
    attr_utils::GeAttrValueToDigest(digest, value);
    value.SetValue<bool>(true);
    attr_utils::GeAttrValueToDigest(digest, value);

    value.SetValue<int64_t>(666);
    attr_utils::GeAttrValueToDigest(digest, value);

    value.SetValue(DT_FLOAT);
    value.SetValue<float>(1.11f);
    attr_utils::GeAttrValueToDigest(digest, value);

    value.SetValue<ge::GeAttrValue::DATA_TYPE>(DT_FLOAT);
    attr_utils::GeAttrValueToDigest(digest, value);

    vector<string> val1{"hello", "world"};
    vector<bool> val2{false, true};
    vector<int64_t> val3{666, 444};
    vector<vector<int64_t>> val4;
    val4.emplace_back(vector<int64_t>{1,2});
    val4.emplace_back(vector<int64_t>{3,4,5});
    vector<float> val5{1.0f, 2.0f};

    digest = 0;
    value.SetValue(val1);
    attr_utils::GeAttrValueToDigest(digest, value);
    EXPECT_EQ(digest, 4469517942762953920UL);
    value.SetValue(val2);
    attr_utils::GeAttrValueToDigest(digest, value);
    EXPECT_EQ(digest, 6147301992897187629UL);
    value.SetValue(val3);
    attr_utils::GeAttrValueToDigest(digest, value);
    EXPECT_EQ(digest, 1159624198106721570UL);
    value.SetValue(val4);
    attr_utils::GeAttrValueToDigest(digest, value);
    EXPECT_EQ(digest, 3178425737445201206UL);
    value.SetValue(val5);
    attr_utils::GeAttrValueToDigest(digest, value);
    EXPECT_EQ(digest, 6041060120762474175UL);

    map<string, ge::GeAttrValue> attr;
    attr.emplace("alpha", value);
    digest = attr_utils::AttrMapToDigest(attr);
    EXPECT_EQ(digest, 14603419155467829796UL);
}

TEST_F(UTEST_ACL_OpAttr, ListFloatEqualsTest)
{
    vector<float> lhsValue;
    vector<float> rhsValue;
    ASSERT_TRUE(attr_utils::IsListFloatEquals(lhsValue, rhsValue));

    lhsValue.push_back(1.0000001f);
    ASSERT_FALSE(attr_utils::IsListFloatEquals(lhsValue, rhsValue));

    rhsValue.push_back(1.0000002f);
    ASSERT_TRUE(attr_utils::IsListFloatEquals(lhsValue, rhsValue));

    rhsValue[0] = 1.0002f;
    ASSERT_FALSE(attr_utils::IsListFloatEquals(lhsValue, rhsValue));
}


TEST_F(UTEST_ACL_OpAttr, TestSaveConstToAttr)
{
    OpModelDef modelDef1;

    modelDef1.opType = "test";
    int64_t shape[4]{32, 2, 16, 16};
    aclTensorDesc desc1(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    desc1.isConst = true;
    desc1.constDataBuf = nullptr;
    desc1.constDataLen = 4;
    modelDef1.inputDescArr.emplace_back(desc1);
    ASSERT_FALSE(attr_utils::SaveConstToAttr(modelDef1));

    OpModelDef modelDef2;
    modelDef2.opType = "test";
    auto *data = new (std::nothrow) int[4];
    std::shared_ptr<void> modelData;
    modelData.reset(data, [](const int *p) { delete[]p; });
    aclTensorDesc desc2(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    desc2.constDataBuf = modelData;
    desc2.isConst = true;
    desc2.constDataLen = 0;
    modelDef2.inputDescArr.emplace_back(desc2);
    ASSERT_FALSE(attr_utils::SaveConstToAttr(modelDef2));

    OpModelDef modelDef3;
    modelDef3.opType = "test";
    aclTensorDesc desc3(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    desc3.constDataBuf = modelData;
    desc3.isConst = true;
    desc3.constDataLen = 4;
    modelDef3.inputDescArr.emplace_back(desc3);
    ASSERT_TRUE(attr_utils::SaveConstToAttr(modelDef3));
}

TEST_F(UTEST_ACL_OpAttr, SetAttrInfNanTest)
{
    aclopAttr *opAttr = aclopCreateAttr();
    const float inf = std::numeric_limits<float>::infinity();
    const float nan1 = std::nanf("1");
    const float nan2 = std::nanf("2"); 
    float floatList1[2]{1.0, 0.0};
    float floatList2[2]{inf, inf};
    float floatList3[2]{nan1, nan2};
    EXPECT_EQ(aclopSetAttrFloat(opAttr, "float1", inf), ACL_SUCCESS);
    ASSERT_FALSE(opAttr->DebugString().empty());
    EXPECT_EQ(aclopSetAttrFloat(opAttr, "float2", nan1), ACL_SUCCESS);
    ASSERT_FALSE(opAttr->DebugString().empty());
    EXPECT_EQ(aclopSetAttrFloat(opAttr, "float3", nan2), ACL_SUCCESS);
    ASSERT_FALSE(opAttr->DebugString().empty());
    EXPECT_EQ(aclopSetAttrListFloat(opAttr, "floatlist1", 2, floatList1), ACL_SUCCESS);
    ASSERT_FALSE(opAttr->DebugString().empty());
    EXPECT_EQ(aclopSetAttrListFloat(opAttr, "floatlist2", 2, floatList2), ACL_SUCCESS);
    ASSERT_FALSE(opAttr->DebugString().empty());
    EXPECT_EQ(aclopSetAttrListFloat(opAttr, "floatlist3", 2, floatList3), ACL_SUCCESS);
    ASSERT_FALSE(opAttr->DebugString().empty());

    for (auto &it : opAttr->Attrs()) {
        ASSERT_TRUE(attr_utils::AttrValueEquals(it.second, it.second));
    }
    aclopDestroyAttr(opAttr);
}

TEST_F(UTEST_ACL_OpAttr, GeAttrValueToString_string)
{
    std::string strValue = "test";
    const auto attrValues = ge::GeAttrValue::CreateFrom<std::string>(strValue);
    std::string str = attr_utils::GeAttrValueToString(attrValues);

    ASSERT_EQ(str, std::string("test"));
}

TEST_F(UTEST_ACL_OpAttr, GeAttrValueToString_bool_true)
{
    bool value = true;
    const auto attrValues = ge::GeAttrValue::CreateFrom<bool>(value);
    std::string str = attr_utils::GeAttrValueToString(attrValues);

    ASSERT_EQ(str, std::string("True"));
}

TEST_F(UTEST_ACL_OpAttr, GeAttrValueToString_bool_false)
{
    bool value = false;
    const auto attrValues = ge::GeAttrValue::CreateFrom<bool>(value);
    std::string str = attr_utils::GeAttrValueToString(attrValues);

    ASSERT_EQ(str, std::string("False"));
}

TEST_F(UTEST_ACL_OpAttr, GeAttrValueToString_int64)
{
    int64_t value = 123;
    const auto attrValues = ge::GeAttrValue::CreateFrom<int64_t>(value);
    std::string str = attr_utils::GeAttrValueToString(attrValues);

    ASSERT_EQ(str, std::string("123"));
}

TEST_F(UTEST_ACL_OpAttr, GeAttrValueToString_float32)
{
    float32_t value = 123.456f;
    const auto attrValues = ge::GeAttrValue::CreateFrom<float32_t>(value);
    std::string str = attr_utils::GeAttrValueToString(attrValues);

    ASSERT_EQ(str, std::string("123.456"));
}

#if 1
TEST_F(UTEST_ACL_OpAttr, GeAttrValueToString_data_type)
{
    ge::GeAttrValue::DATA_TYPE value = DT_VARIANT;
    const auto attrValues = ge::GeAttrValue::CreateFrom<ge::GeAttrValue::DATA_TYPE>(value);
    std::string str = attr_utils::GeAttrValueToString(attrValues);

    ASSERT_EQ(str, std::to_string(value));
}

TEST_F(UTEST_ACL_OpAttr, GeAttrValueToString_list_string)
{
    std::vector<std::string> value;
    for (int32_t i = 0; i < 2; ++i) {
        value.emplace_back(std::to_string(i));
    }
    const auto attrValues = ge::GeAttrValue::CreateFrom<std::vector<std::string>>(value);
    std::string str = attr_utils::GeAttrValueToString(attrValues);

    ASSERT_EQ(str, "[0, 1]");
}

TEST_F(UTEST_ACL_OpAttr, GeAttrValueToString_list_bool)
{
    std::vector<bool> value;
    value.emplace_back(true);
    value.emplace_back(false);

    const auto attrValues = ge::GeAttrValue::CreateFrom<std::vector<bool>>(value);
    std::string str = attr_utils::GeAttrValueToString(attrValues);

    ASSERT_EQ(str, "[True, False]");
}

TEST_F(UTEST_ACL_OpAttr, GeAttrValueToString_list_int64)
{
    std::vector<int64_t> value;
    for (int64_t i = 0; i < 3; ++i) {
        value.emplace_back(i);
    }
    const auto attrValues = ge::GeAttrValue::CreateFrom<std::vector<int64_t>>(value);
    std::string str = attr_utils::GeAttrValueToString(attrValues);

    ASSERT_EQ(str, "[0, 1, 2]");
}

TEST_F(UTEST_ACL_OpAttr, GeAttrValueToString_list_float)
{
    std::vector<float32_t> value;
    value.emplace_back(1.2f);
    value.emplace_back(3.4f);
    const auto attrValues = ge::GeAttrValue::CreateFrom<std::vector<float32_t>>(value);
    std::string str = attr_utils::GeAttrValueToString(attrValues);

    ASSERT_EQ(str, "[1.2, 3.4]");
}

TEST_F(UTEST_ACL_OpAttr, GeAttrValueToString_list_data_type)
{
    std::vector<ge::GeAttrValue::DATA_TYPE> value;
    value.emplace_back(DT_UINT2);
    value.emplace_back(DT_COMPLEX128);
    const auto attrValues = ge::GeAttrValue::CreateFrom<std::vector<ge::GeAttrValue::DATA_TYPE>>(value);
    std::string str = attr_utils::GeAttrValueToString(attrValues);

    ASSERT_EQ(str, "[32, 17]");
}

TEST_F(UTEST_ACL_OpAttr, GeAttrValueToString_list_list_int)
{
    std::vector<std::vector<int64_t>> value;
    for (int64_t i = 0; i < 2; ++i) {
        std::vector<int64_t> v = {i, i+1};
        value.emplace_back(v);

    }
    const auto attrValues = ge::GeAttrValue::CreateFrom<std::vector<std::vector<int64_t>>>(value);
    std::string str = attr_utils::GeAttrValueToString(attrValues);

    ASSERT_EQ(str, "[[0, 1], [1, 2]]");
}

TEST_F(UTEST_ACL_OpAttr, GeAttrValueToString_default)
{
    int32_t value = 789;
    const auto attrValues = ge::GeAttrValue::CreateFrom<int32_t>(value);
    std::string str = attr_utils::GeAttrValueToString(attrValues);

    ASSERT_EQ(str, std::string("Unknown type 0"));
}
#endif
