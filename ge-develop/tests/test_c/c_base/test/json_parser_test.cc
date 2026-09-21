/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <memory>
#include <fstream>
#include <string>
#include "json_parser.h"

using namespace testing;

class UtestJsonParseTest : public testing::Test {
 protected:
  void SetUp() {
  }

  void TearDown() {
  }
};

TEST_F(UtestJsonParseTest, JsonParseCaseBasicNull) {
  const char *json = "null";
  CJsonObj *jsonObj = CJsonParse(json, strlen(json));
  ASSERT_TRUE((jsonObj != NULL));
  ASSERT_TRUE(CJsonIsNull(jsonObj));
  FreeCJsonObj(jsonObj);
}

TEST_F(UtestJsonParseTest, JsonParseCaseBasicTrue) {
  const char *json = "true";
  CJsonObj *jsonObj = CJsonParse(json, strlen(json));
  ASSERT_TRUE((jsonObj != NULL));
  ASSERT_TRUE(CJsonIsBool(jsonObj));
  ASSERT_TRUE(GetCJsonBool(jsonObj));
  FreeCJsonObj(jsonObj);
}

TEST_F(UtestJsonParseTest, JsonParseCaseBasicFalse) {
  const char *json = "false";
  CJsonObj *jsonObj = CJsonParse(json, strlen(json));
  ASSERT_TRUE((jsonObj != NULL));
  ASSERT_TRUE(CJsonIsBool(jsonObj));
  ASSERT_FALSE(GetCJsonBool(jsonObj));
  FreeCJsonObj(jsonObj);
}

TEST_F(UtestJsonParseTest, JsonParseCaseBasicNumber) {
  const char *json = "123";
  CJsonObj *jsonObj = CJsonParse(json, strlen(json));
  ASSERT_TRUE((jsonObj != NULL));
  ASSERT_TRUE(CJsonIsInt(jsonObj));
  EXPECT_EQ(GetCJsonInt(jsonObj), 123);
  FreeCJsonObj(jsonObj);
}

TEST_F(UtestJsonParseTest, JsonParseCaseBasicNegativeNumber) {
  const char *json = "-123";
  CJsonObj *jsonObj = CJsonParse(json, strlen(json));
  ASSERT_TRUE((jsonObj != NULL));
  ASSERT_TRUE(CJsonIsInt(jsonObj));
  EXPECT_EQ(GetCJsonInt(jsonObj), -123);
  FreeCJsonObj(jsonObj);
}

TEST_F(UtestJsonParseTest, JsonParseCaseBasicPositiveNumber) {
  const char *json = "+123";
  CJsonObj *jsonObj = CJsonParse(json, strlen(json));
  ASSERT_TRUE((jsonObj != NULL));
  ASSERT_TRUE(CJsonIsInt(jsonObj));
  EXPECT_EQ(GetCJsonInt(jsonObj), 123);
  FreeCJsonObj(jsonObj);
}

TEST_F(UtestJsonParseTest, JsonParseCaseBasicDecimal) {
  const char *json = "1.23";
  CJsonObj *jsonObj = CJsonParse(json, strlen(json));
  ASSERT_TRUE((jsonObj != NULL));
  ASSERT_TRUE(CJsonIsDouble(jsonObj));
  EXPECT_EQ(GetCJsonDouble(jsonObj), 1.23);
  FreeCJsonObj(jsonObj);
}

TEST_F(UtestJsonParseTest, JsonParseCaseBasicNegativeExp) {
  const char *json = "1.23e-2";
  CJsonObj *jsonObj = CJsonParse(json, strlen(json));
  ASSERT_TRUE((jsonObj != NULL));
  ASSERT_TRUE(CJsonIsDouble(jsonObj));
  double value = GetCJsonDouble(jsonObj);
  EXPECT_TRUE(((value > 0.0122) && (value < 0.0124)));
  FreeCJsonObj(jsonObj);
}


TEST_F(UtestJsonParseTest, JsonParseCaseBasicPositiveExp) {
  const char *json = "1.23e2";
  CJsonObj *jsonObj = CJsonParse(json, strlen(json));
  ASSERT_TRUE((jsonObj != NULL));
  ASSERT_TRUE(CJsonIsDouble(jsonObj));
  double value = GetCJsonDouble(jsonObj);
  EXPECT_TRUE(((value > 122) && (value < 124)));
  FreeCJsonObj(jsonObj);
}

TEST_F(UtestJsonParseTest, JsonParseCaseBasicUpperExp) {
  const char *json = "1.23E2";
  CJsonObj *jsonObj = CJsonParse(json, strlen(json));
  ASSERT_TRUE((jsonObj != NULL));
  ASSERT_TRUE(CJsonIsDouble(jsonObj));
  double value = GetCJsonDouble(jsonObj);
  EXPECT_TRUE(((value > 122) && (value < 124)));
  FreeCJsonObj(jsonObj);
}

TEST_F(UtestJsonParseTest, JsonParseCaseBasicString) {
  const char *json = "\"123\"";
  CJsonObj *jsonObj = CJsonParse(json, strlen(json));
  ASSERT_TRUE((jsonObj != NULL));
  ASSERT_TRUE(CJsonIsString(jsonObj));
  ASSERT_STREQ(GetCJsonString(jsonObj), "123");
  FreeCJsonObj(jsonObj);
}

TEST_F(UtestJsonParseTest, JsonParseCaseBasicInvalidString) {
  const char *json = "\"\\123\"";
  CJsonObj *jsonObj = CJsonParse(json, strlen(json));
  ASSERT_TRUE((jsonObj == NULL));
}

TEST_F(UtestJsonParseTest, JsonParseCaseBasicStringWithFrontQuotation) {
  const char *json = "\"\\\"123\"";
  CJsonObj *jsonObj = CJsonParse(json, strlen(json));
  ASSERT_TRUE((jsonObj != NULL));
  ASSERT_TRUE(CJsonIsString(jsonObj));
  ASSERT_STREQ(GetCJsonString(jsonObj), "\"123");
  FreeCJsonObj(jsonObj);
}

TEST_F(UtestJsonParseTest, JsonParseCaseBasicInvalidWithBackQuotation) {
  const char *json = "123\"";
  CJsonObj *jsonObj = CJsonParse(json, strlen(json));
  ASSERT_TRUE((jsonObj == NULL));
}

TEST_F(UtestJsonParseTest, JsonParseCaseBasicInvalidWithFrontBracket) {
  const char *json = "[123";
  CJsonObj *jsonObj = CJsonParse(json, strlen(json));
  ASSERT_TRUE((jsonObj == NULL));
}

TEST_F(UtestJsonParseTest, JsonParseCaseBasicInvalidWithFrontCurlyBraces) {
  const char *json = "{123";
  CJsonObj *jsonObj = CJsonParse(json, strlen(json));
  ASSERT_TRUE((jsonObj == NULL));
}

TEST_F(UtestJsonParseTest, JsonParseCaseBasicInvalidWithSlash) {
  const char *json = "\\123";
  CJsonObj *jsonObj = CJsonParse(json, strlen(json));
  ASSERT_TRUE((jsonObj == NULL));
}
#if 0
TEST_F(UtestJsonParseTest, JsonParseCaseBasicInvalidWithDecimalPoint) {
  const char *json = ".123";
  CJsonObj *jsonObj = CJsonParse(json, strlen(json));
  ASSERT_TRUE((jsonObj == NULL));
}
TEST_F(UtestJsonParseTest, JsonParseCaseBasicInvalidWithMinusSign) {
  const char *json = "1-23";
  CJsonObj *jsonObj = CJsonParse(json, strlen(json));
  ASSERT_TRUE((jsonObj == NULL));
}

TEST_F(UtestJsonParseTest, JsonParseCaseBasicInvalidWithPlusSign) {
  const char *json = "1+23";
  CJsonObj *jsonObj = CJsonParse(json, strlen(json));
  ASSERT_TRUE((jsonObj == NULL));
}
#endif

TEST_F(UtestJsonParseTest, JsonParseCaseBasicObj) {
  const char *json = "{\"a\" : true, \"b\" : false, \"c\" : 123, \"d\" : \"123\"}";
  CJsonObj *jsonObj = CJsonParse(json, strlen(json));
  ASSERT_TRUE((jsonObj != NULL));
  ASSERT_TRUE(CJsonIsObj(jsonObj));
  CJsonObj *jsonObjValue = GetCJsonSubObj(jsonObj, "a");
  ASSERT_TRUE((jsonObjValue != NULL));
  ASSERT_TRUE(CJsonIsBool(jsonObjValue));
  ASSERT_TRUE(GetCJsonBool(jsonObjValue));

  jsonObjValue = GetCJsonSubObj(jsonObj, "b");
  ASSERT_TRUE((jsonObjValue != NULL));
  ASSERT_TRUE(CJsonIsBool(jsonObjValue));
  ASSERT_FALSE(GetCJsonBool(jsonObjValue));

  jsonObjValue = GetCJsonSubObj(jsonObj, "c");
  ASSERT_TRUE((jsonObjValue != NULL));
  ASSERT_TRUE(CJsonIsInt(jsonObjValue));
  ASSERT_EQ(GetCJsonInt(jsonObjValue), 123);

  jsonObjValue = GetCJsonSubObj(jsonObj, "d");
  ASSERT_TRUE((jsonObjValue != NULL));
  ASSERT_TRUE(CJsonIsString(jsonObjValue));
  ASSERT_STREQ(GetCJsonString(jsonObjValue), "123");

  FreeCJsonObj(jsonObj);
}

TEST_F(UtestJsonParseTest, JsonParseCaseBasicObjInvalid) {
  const char *json = "{\"a\" : true; \"b\" : false, \"c\" : 123, \"d\" : \"123\"}";
  CJsonObj *jsonObj = CJsonParse(json, strlen(json));
  ASSERT_TRUE((jsonObj == NULL));
}

TEST_F(UtestJsonParseTest, JsonParseCaseBasicObjInvalidWithComma) {
  const char *json = "{\"a\" : true, \"b\" : false, \"c\" : 123, \"d\" : \"123\",}";
  CJsonObj *jsonObj = CJsonParse(json, strlen(json));
  ASSERT_TRUE((jsonObj == NULL));
}

TEST_F(UtestJsonParseTest, JsonParseCaseBasicArray) {
  const char *json = "[0, 1, 2]";
  CJsonObj *jsonObj = CJsonParse(json, strlen(json));
  ASSERT_TRUE((jsonObj != NULL));
  ASSERT_TRUE(CJsonIsArray(jsonObj));
  ASSERT_EQ(GetCJsonArraySize(jsonObj), 3);

  for (int i = 0; i < 3; i++) {
    CJsonObj *jsonObjValue = CJsonArrayAt(jsonObj, i);
    ASSERT_TRUE((jsonObjValue != NULL));
    ASSERT_TRUE(CJsonIsInt(jsonObjValue));
    ASSERT_EQ(GetCJsonInt(jsonObjValue), i);
  }

  CJsonObj *jsonObjValue = CJsonArrayAt(jsonObj, 4);
  ASSERT_TRUE((jsonObjValue == NULL));

  FreeCJsonObj(jsonObj);
}

TEST_F(UtestJsonParseTest, JsonParseCaseBasicArrayObj) {
  const char *json = "[{\"a\" : true, \"b\" : false, \"c\" : 123, \"d\" : \"123\"}, "
                      "{\"a\" : true, \"b\" : false, \"c\" : 123, \"d\" : \"123\"}, "
                      "{\"a\" : true, \"b\" : false, \"c\" : 123, \"d\" : \"123\"}]";
  CJsonObj *jsonObj = CJsonParse(json, strlen(json));
  ASSERT_TRUE((jsonObj != NULL));
  ASSERT_TRUE(CJsonIsArray(jsonObj));
  ASSERT_EQ(GetCJsonArraySize(jsonObj), 3);

  for (int i = 0; i < 3; i++) {
    CJsonObj *jsonObjIt = CJsonArrayAt(jsonObj, i);
    CJsonObj *jsonObjValue = GetCJsonSubObj(jsonObjIt, "a");
    ASSERT_TRUE((jsonObjValue != NULL));
    ASSERT_TRUE(CJsonIsBool(jsonObjValue));
    ASSERT_TRUE(GetCJsonBool(jsonObjValue));

    jsonObjValue = GetCJsonSubObj(jsonObjIt, "b");
    ASSERT_TRUE((jsonObjValue != NULL));
    ASSERT_TRUE(CJsonIsBool(jsonObjValue));
    ASSERT_FALSE(GetCJsonBool(jsonObjValue));

    jsonObjValue = GetCJsonSubObj(jsonObjIt, "c");
    ASSERT_TRUE((jsonObjValue != NULL));
    ASSERT_TRUE(CJsonIsInt(jsonObjValue));
    ASSERT_EQ(GetCJsonInt(jsonObjValue), 123);

    jsonObjValue = GetCJsonSubObj(jsonObjIt, "d");
    ASSERT_TRUE((jsonObjValue != NULL));
    ASSERT_TRUE(CJsonIsString(jsonObjValue));
    ASSERT_STREQ(GetCJsonString(jsonObjValue), "123");
  }

  CJsonObj *jsonObjValue = CJsonArrayAt(jsonObj, 4);
  ASSERT_TRUE((jsonObjValue == NULL));

  FreeCJsonObj(jsonObj);
}


TEST_F(UtestJsonParseTest, JsonParseCaseBasicObjArray) {
  const char *json = "{\"a\" : [0, 1, 2]}";
  CJsonObj *jsonObj = CJsonParse(json, strlen(json));
  ASSERT_TRUE((jsonObj != NULL));
  ASSERT_TRUE(CJsonIsObj(jsonObj));
  CJsonObj *jsonObjValue = GetCJsonSubObj(jsonObj, "a");
  ASSERT_TRUE((jsonObjValue != NULL));

  ASSERT_TRUE(CJsonIsArray(jsonObjValue));
  ASSERT_EQ(GetCJsonArraySize(jsonObjValue), 3);

  for (int i = 0; i < 3; i++) {
    CJsonObj *jsonObjIt = CJsonArrayAt(jsonObjValue, i);
    ASSERT_TRUE((jsonObjIt != NULL));
    ASSERT_TRUE(CJsonIsInt(jsonObjIt));
    ASSERT_EQ(GetCJsonInt(jsonObjIt), i);
  }

  FreeCJsonObj(jsonObj);
}

TEST_F(UtestJsonParseTest, JsonParseCaseJsonFile) {
  const char *json = "{\"a\" : [{\"a\" : true, \"b\" : false, \"c\" : 123, \"d\" : \"123\"}, "
                                "{\"a\" : true, \"b\" : false, \"c\" : 123, \"d\" : \"123\"}, "
                                "{\"a\" : true, \"b\" : false, \"c\" : 123, \"d\" : \"123\"}]}";
  const char *filePath = "test.json";
  std::ofstream stream(filePath);
  stream << json;
  stream.close();

  CJsonObj *jsonObj = CJsonFileParse(filePath);
  ASSERT_TRUE((jsonObj != NULL));
  ASSERT_TRUE(CJsonIsObj(jsonObj));
  CJsonObj *jsonObjA1 = GetCJsonSubObj(jsonObj, "a");
  ASSERT_TRUE((jsonObjA1 != NULL));

  ASSERT_TRUE(CJsonIsArray(jsonObjA1));
  ASSERT_EQ(GetCJsonArraySize(jsonObjA1), 3);

  for (int i = 0; i < 3; i++) {
    CJsonObj *jsonObjIt = CJsonArrayAt(jsonObjA1, i);
    CJsonObj *jsonObjValue = GetCJsonSubObj(jsonObjIt, "a");
    ASSERT_TRUE((jsonObjValue != NULL));
    ASSERT_TRUE(CJsonIsBool(jsonObjValue));
    ASSERT_TRUE(GetCJsonBool(jsonObjValue));

    jsonObjValue = GetCJsonSubObj(jsonObjIt, "b");
    ASSERT_TRUE((jsonObjValue != NULL));
    ASSERT_TRUE(CJsonIsBool(jsonObjValue));
    ASSERT_FALSE(GetCJsonBool(jsonObjValue));

    jsonObjValue = GetCJsonSubObj(jsonObjIt, "c");
    ASSERT_TRUE((jsonObjValue != NULL));
    ASSERT_TRUE(CJsonIsInt(jsonObjValue));
    ASSERT_EQ(GetCJsonInt(jsonObjValue), 123);

    jsonObjValue = GetCJsonSubObj(jsonObjIt, "d");
    ASSERT_TRUE((jsonObjValue != NULL));
    ASSERT_TRUE(CJsonIsString(jsonObjValue));
    ASSERT_STREQ(GetCJsonString(jsonObjValue), "123");
  }

  FreeCJsonObj(jsonObj);
}

TEST_F(UtestJsonParseTest, JsonParseCaseJsonKey) {
  const char *json = "{\"a\" : [{\"a\" : true, \"b\" : false, \"c\" : 123, \"d\" : \"123\"}, "
                                "{\"a\" : true, \"b\" : false, \"c\" : 123, \"d\" : \"123\"}, "
                                "{\"a\" : true, \"b\" : false, \"c\" : 123, \"d\" : \"123\"}]}";
  size_t offset;
  size_t len = CJsonParseKeyPosition(json, strlen(json), "a", &offset);
  ASSERT_EQ(offset, 7);
  ASSERT_EQ(len, strlen("[{\"a\" : true, \"b\" : false, \"c\" : 123, \"d\" : \"123\"}, "
                        "{\"a\" : true, \"b\" : false, \"c\" : 123, \"d\" : \"123\"}, "
                        "{\"a\" : true, \"b\" : false, \"c\" : 123, \"d\" : \"123\"}]"));
}

TEST_F(UtestJsonParseTest, JsonParseCaseJsonFileKey) {
  const char *json = "{\"a\" : [{\"a\" : true, \"b\" : false, \"c\" : 123, \"d\" : \"123\"}, "
                    "{\"a\" : true, \"b\" : false, \"c\" : 123, \"d\" : \"123\"}, "
                    "{\"a\" : true, \"b\" : false, \"c\" : 123, \"d\" : \"123\"}]}";
  const char *filePath = "test.json";
  std::ofstream stream(filePath);
  stream << json;
  stream.close();

  char *subJson = CJsonFileParseKey(filePath, "a");
  ASSERT_TRUE((subJson != NULL));
  ASSERT_STREQ(subJson, "[{\"a\" : true, \"b\" : false, \"c\" : 123, \"d\" : \"123\"}, "
                      "{\"a\" : true, \"b\" : false, \"c\" : 123, \"d\" : \"123\"}, "
                      "{\"a\" : true, \"b\" : false, \"c\" : 123, \"d\" : \"123\"}]");
  free(subJson);
}

TEST_F(UtestJsonParseTest, JsonParseCaseJsonFileKeyWithTwoKey) {
  const char *json = "{\"a\" : 123, \"b\": [{\"a\" : true, \"b\" : false, \"c\" : 123, \"d\" : \"123\"}, "
                    "{\"a\" : true, \"b\" : false, \"c\" : 123, \"d\" : \"123\"}, "
                    "{\"a\" : true, \"b\" : false, \"c\" : 123, \"d\" : \"123\"}]}";
  const char *filePath = "test.json";
  std::ofstream stream(filePath);
  stream << json;
  stream.close();

  char *subJson = CJsonFileParseKey(filePath, "a");
  ASSERT_TRUE((subJson != NULL));
  ASSERT_STREQ(subJson, "123");
  free(subJson);

  subJson = CJsonFileParseKey(filePath, "b");
  ASSERT_TRUE((subJson != NULL));
  ASSERT_STREQ(subJson, "[{\"a\" : true, \"b\" : false, \"c\" : 123, \"d\" : \"123\"}, "
                      "{\"a\" : true, \"b\" : false, \"c\" : 123, \"d\" : \"123\"}, "
                      "{\"a\" : true, \"b\" : false, \"c\" : 123, \"d\" : \"123\"}]");
  free(subJson);
}

TEST_F(UtestJsonParseTest, JsonParseCaseJsonFileKeyFail) {
  const char *json = "{\"a\" : 123, \"b\": [{\"a\" : true, \"b\" : false, \"c\" : 123, \"d\" : \"123\"}, "
                    "{\"a\" : true, \"b\" : false; \"c\" : 123, \"d\" : \"123\"}, "
                    "{\"a\" : true, \"b\" : false, \"c\" : 123, \"d\" : \"123\"}]}";
  const char *filePath = "test.json";
  std::ofstream stream(filePath);
  stream << json;
  stream.close();

  char *subJson = CJsonFileParseKey(filePath, "a");
  ASSERT_TRUE((subJson == NULL));

  subJson = CJsonFileParseKey(filePath, "b");
  ASSERT_TRUE((subJson == NULL));
}

TEST_F(UtestJsonParseTest, JsonParseCaseJsonFileKeyFailNull) {
  const char *json = "null";
  const char *filePath = "test.json";
  std::ofstream stream(filePath);
  stream << json;
  stream.close();

  char *subJson = CJsonFileParseKey(filePath, "null");
  ASSERT_TRUE((subJson == NULL));
}

TEST_F(UtestJsonParseTest, JsonParseCaseJsonFileKeyFailNumber) {
  const char *json = "123";
  const char *filePath = "test.json";
  std::ofstream stream(filePath);
  stream << json;
  stream.close();

  char *subJson = CJsonFileParseKey(filePath, "123");
  ASSERT_TRUE((subJson == NULL));
}

TEST_F(UtestJsonParseTest, JsonParseCaseJsonFileKeyFailString) {
  const char *json = "\"123\"";
  const char *filePath = "test.json";
  std::ofstream stream(filePath);
  stream << json;
  stream.close();

  char *subJson = CJsonFileParseKey(filePath, "123");
  ASSERT_TRUE((subJson == NULL));
}
