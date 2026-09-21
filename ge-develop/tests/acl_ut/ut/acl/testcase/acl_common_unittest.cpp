#include <gtest/gtest.h>
#include "common/common_inner.h"
#include "common/log_inner.h"
#include "common/json_parser.h"
#include "common/prof_api_reg.h"
#include "acl_stub.h"

using namespace testing;
using namespace std;
using namespace acl;

class UTEST_ACL_Common : public testing::Test {
protected:
    virtual void SetUp() {
        MockFunctionTest::aclStubInstance().ResetToDefaultMock();
    }
    virtual void TearDown() {
        Mock::VerifyAndClear((void *)(&MockFunctionTest::aclStubInstance()));
    }
};

TEST_F(UTEST_ACL_Common, GetCurLogLevel1)
{
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), dlog_getlevel(_,_))
        .WillOnce(Return((DLOG_ERROR)));
    uint32_t log_level = AclLog::GetCurLogLevel();
    EXPECT_EQ(log_level, ACL_ERROR);
}

TEST_F(UTEST_ACL_Common, GetCurLogLevel2)
{
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), dlog_getlevel(_,_))
        .WillOnce(Return((DLOG_WARN)));
    uint32_t log_level = AclLog::GetCurLogLevel();
    EXPECT_EQ(log_level, ACL_WARNING);
}

TEST_F(UTEST_ACL_Common, GetCurLogLevel3)
{
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), dlog_getlevel(_,_))
        .WillOnce(Return((DLOG_INFO)));
    uint32_t log_level = AclLog::GetCurLogLevel();
    EXPECT_EQ(log_level, ACL_INFO);
}

TEST_F(UTEST_ACL_Common, GetCurLogLevel4)
{
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), dlog_getlevel(_,_))
        .WillOnce(Return((DLOG_DEBUG)));
    uint32_t log_level = AclLog::GetCurLogLevel();
    EXPECT_EQ(log_level, ACL_DEBUG);
}

TEST_F(UTEST_ACL_Common, GetCurLogLevel5)
{
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), dlog_getlevel(_,_))
        .WillOnce(Return((DLOG_NULL)));
    uint32_t log_level = AclLog::GetCurLogLevel();
    EXPECT_EQ(log_level, ACL_INFO);
}

extern int g_logLevel;
TEST_F(UTEST_ACL_Common, ACLSaveLog)
{
    const char *strLog = "hello, acl";
    AclLog::ACLSaveLog(ACL_DEBUG, strLog);
    EXPECT_EQ(g_logLevel, ACL_DEBUG);
    AclLog::ACLSaveLog(ACL_INFO, strLog);
    EXPECT_EQ(g_logLevel, ACL_INFO);
    AclLog::ACLSaveLog(ACL_WARNING, strLog);
    EXPECT_EQ(g_logLevel, ACL_WARNING);
    AclLog::ACLSaveLog(ACL_ERROR, strLog);
    EXPECT_EQ(g_logLevel, ACL_ERROR);
    AclLog::ACLSaveLog(ACL_ERROR, nullptr);
}

TEST_F(UTEST_ACL_Common, jsonParserUtilsTest)
{
    nlohmann::json js;
    auto ret = JsonParser::ParseJson("{", js);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
    ret = JsonParser::ParseJson("{}", js);
    EXPECT_EQ(ret, ACL_RT_SUCCESS);
    std::string strJsonCtx;
    std::string subStrKey = "key";
    bool found = false;
    ret = JsonParser::GetJsonCtxByKeyFromBuffer("{", strJsonCtx, subStrKey, found);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
    ret = JsonParser::GetJsonCtxByKeyFromBuffer("{\"key\":\"value\"}", strJsonCtx, subStrKey, found);
    EXPECT_EQ(ret, ACL_RT_SUCCESS);
    EXPECT_EQ(found, true);
    EXPECT_EQ(strJsonCtx, "\"value\"");
    ret = JsonParser::GetJsonCtxByKeyFromBuffer("{\"key1\":\"value\"}", strJsonCtx, subStrKey, found);
    EXPECT_EQ(ret, ACL_RT_SUCCESS);
    EXPECT_EQ(found, false);
}

TEST_F(UTEST_ACL_Common, GetJsonCtxByKey_failed)
{
    nlohmann::json js;
    const char_t* fileName = ACL_BASE_DIR "/tests/ut/acl/json/illegal_format.json";
    std::string strJsonCtx = "123";
    std::string subStrKey;
    bool found = false;
    aclError ret = acl::JsonParser::GetJsonCtxByKeyFromBuffer(fileName, strJsonCtx, subStrKey, found);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
    EXPECT_FALSE(found);
}

TEST_F(UTEST_ACL_Common, AclProfilingReporter)
{
    acl::AclProfilingReporter reporter(acl::AclProfType::AclmdlSetDynamicHWSize);
    EXPECT_EQ(reporter.aclApi_, acl::AclProfType::AclmdlSetDynamicHWSize);
    reporter.~AclProfilingReporter();
    acl::AclProfilingReporter reporter1(acl::AclProfType::AclmdlExecute);
    EXPECT_EQ(reporter1.aclApi_, acl::AclProfType::AclmdlExecute);
    reporter1.~AclProfilingReporter();
}

TEST_F(UTEST_ACL_Common, FormatStr_failed_1100bytes)
{
    std::string str(1100, 'a');
    std::string value = acl::AclErrorLogManager::FormatStr(str.c_str());
    EXPECT_TRUE(value.empty());

    value = acl::AclErrorLogManager::FormatStr(nullptr);
    EXPECT_TRUE(value.empty());
}
