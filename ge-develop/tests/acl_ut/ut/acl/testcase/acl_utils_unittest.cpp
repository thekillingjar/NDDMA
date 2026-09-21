#include <gtest/gtest.h>
#include "utils/string_utils.h"
#include "utils/math_utils.h"
#include "utils/hash_utils.h"
#include "utils/file_utils.h"
#include "utils/attr_utils.h"
#include "acl_stub.h"
#include "types/acl_op_inner.h"

using namespace testing;
using namespace std;
using namespace acl;

class UTEST_ACL_Utils : public testing::Test {
protected:
    virtual void SetUp() {
        MockFunctionTest::aclStubInstance().ResetToDefaultMock();
    }
    virtual void TearDown() {
        Mock::VerifyAndClear((void *)(&MockFunctionTest::aclStubInstance()));
    }
};

TEST_F(UTEST_ACL_Utils, AclStringUtilsTest)
{
    std::string str;
    acl::StringUtils::Trim(str);
    acl::StringUtils::Strip(str, str);
    std::vector<std::string> elems;
    acl::StringUtils::Split(str, '#', elems);
    str = " customize#";
    acl::StringUtils::Trim(str);
    EXPECT_EQ(str, "customize#");
    acl::StringUtils::Split(str, '#', elems);
    EXPECT_EQ(elems[0], "customize");
}

TEST_F(UTEST_ACL_Utils, AclMathUtilsTest)
{
    const size_t a = 1;
    const size_t b = 2;
    size_t res;
    EXPECT_EQ(acl::CheckSizeTMultiOverflow(a, b, res), ACL_SUCCESS);
    EXPECT_EQ(res, b);
    const size_t c_max = SIZE_MAX;
    EXPECT_EQ(acl::CheckSizeTMultiOverflow(c_max, c_max, res), ACL_ERROR_FAILURE);

    size_t ret = 3;
    EXPECT_EQ(acl::CheckSizeTAddOverflow(a, b, res), ACL_SUCCESS);
    EXPECT_EQ(res, ret);
    EXPECT_EQ(acl::CheckSizeTAddOverflow(c_max, c_max, res), ACL_ERROR_FAILURE);

    const int32_t a_int = 0;
    const int32_t b_int = 1;
    int32_t res_int;
    EXPECT_EQ(acl::CheckIntAddOverflow(a_int, b_int, res_int), ACL_SUCCESS);
    EXPECT_EQ(res_int, b_int);
    const int32_t a_int_max = INT_MAX;
    const int32_t b_int_max = INT_MAX;
    EXPECT_EQ(acl::CheckIntAddOverflow(a_int_max, b_int_max, res_int), ACL_ERROR_FAILURE);
}

TEST_F(UTEST_ACL_Utils, AclHashUtilsTest)
{
    OpModelDef modelDef;
    modelDef.opType = "acltest";
    int64_t shape[]{16, 16};
    modelDef.inputDescArr.emplace_back(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    modelDef.inputDescArr.emplace_back(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    modelDef.outputDescArr.emplace_back(ACL_FLOAT16, 2, shape, ACL_FORMAT_ND);
    modelDef.opAttr.SetAttr<string>("testAttr", "attrValue");
    auto modelDefPtr = shared_ptr<OpModelDef>(new (std::nothrow)OpModelDef(std::move(modelDef)));

    AclOp aclOp;
    aclOp.opType = "acltesterror";
    aclopAttr *opAttr = nullptr;
    const uint64_t seq = 0;
    EXPECT_EQ(hash_utils::CheckModelAndAttrMatchDynamic(aclOp, opAttr, modelDefPtr, seq), false);

    auto modelDefNullPtr = shared_ptr<OpModelDef>(nullptr);
    EXPECT_EQ(hash_utils::CheckModelAndAttrMatchDynamic(aclOp, opAttr, modelDefNullPtr, seq), false);

    size_t seed = 0;
    EXPECT_EQ(hash_utils::GetTensorDescHash(1, nullptr, seed), ACL_ERROR_FAILURE);
    EXPECT_EQ(hash_utils::GetTensorDescHashDynamic(1, nullptr, seed), ACL_ERROR_FAILURE);
}

TEST_F(UTEST_ACL_Utils, AclFileUtilsTest)
{
    const int32_t maxDepth = 3;
    std::vector<std::string> dummyNames;
    EXPECT_EQ(file_utils::ListFiles("./not_exist_dir", nullptr, dummyNames, maxDepth), ACL_ERROR_READ_MODEL_FAILURE);
}

TEST_F(UTEST_ACL_Utils, AclAttrUtilsTest)
{
    std::map<AttrRangeType, ge::GeAttrValue> valueRange;
    void *data = malloc(4);
    aclDataBuffer *dataBuffer = aclCreateDataBuffer(data, 4);
    aclDataType dataType = ACL_FLOAT;
    auto ret = attr_utils::ValueRangeCheck(valueRange, dataBuffer, dataType);
    EXPECT_EQ(ret, true);

    *static_cast<uint16_t*>(data) = 0x0000;
    dataType = ACL_FLOAT16;
    ret = attr_utils::ValueRangeCheck(valueRange, dataBuffer, dataType);
    EXPECT_EQ(ret, true);

    *static_cast<uint16_t*>(data) = 0x0001;
    ret = attr_utils::ValueRangeCheck(valueRange, dataBuffer, dataType);
    EXPECT_EQ(ret, true);

    free(data);
    aclDestroyDataBuffer(dataBuffer);
}

TEST_F(UTEST_ACL_Utils, AclStringUtilsTrimTest)
{
    std::string str;
    str = " customize";
    acl::StringUtils::Trim(str);
    EXPECT_EQ(str, "customize");

    str = "   customize";
    acl::StringUtils::Trim(str);
    EXPECT_EQ(str, "customize");

    str = "customize ";
    acl::StringUtils::Trim(str);
    EXPECT_EQ(str, "customize");

    str = "customize   ";
    acl::StringUtils::Trim(str);
    EXPECT_EQ(str, "customize");

    str = " customize ";
    acl::StringUtils::Trim(str);
    EXPECT_EQ(str, "customize");

    str = "   customize   ";
    acl::StringUtils::Trim(str);
    EXPECT_EQ(str, "customize");

    str = "   customize vendor   ";
    acl::StringUtils::Trim(str);
    EXPECT_EQ(str, "customize vendor");
}
