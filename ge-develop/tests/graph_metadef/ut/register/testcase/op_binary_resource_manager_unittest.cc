/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <gtest/gtest.h>
#include "register/op_binary_resource_manager.h"

class OpBinaryResourceManagerUT : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}

    nnopbase::OpBinaryResourceManager &manager = nnopbase::OpBinaryResourceManager::GetInstance();
};

TEST_F(OpBinaryResourceManagerUT, SaveFunc) {
  int i;
  manager.AddOpFuncHandle("AddTik2", {(void *)&i});
  EXPECT_EQ(manager.resourceHandle_.size(), 1);
  EXPECT_EQ(manager.resourceHandle_["AddTik2"][0], &i);
}

std::string AddTik2Json = "{"
                          "  \"binList\": ["
                          "    {"
                          "      \"simplifiedKey\": ["
                          "        \"AddTik2/d=0,p=0/1,2/1,2/1,2\","
                          "        \"AddTik2/d=1,p=0/1,2/1,2/1,2\""
                          "      ],"
                          "      \"binInfo\": {"
                          "        \"jsonFilePath\": \"ascend910/add_tik2/Add_Tik2_01.json\""
                          "      }"
                          "    },"
                          "    {"
                          "      \"simplifiedKey\": ["
                          "        \"AddTik2/d=0,p=0/1,2/0,2/0,2\","
                          "        \"AddTik2/d=1,p=0/1,2/0,2/0,2\""
                          "      ],"
                          "      \"binInfo\": {"
                          "        \"jsonFilePath\": \"ascend910/add_tik2/Add_Tik2_02.json\""
                          "      }"
                          "    }"
                          "  ]"
                          "}";
std::string AddTik201Json = "{"
                            "  \"filePath\": \"ascend910/add_tik2/Add_Tik2_01.json\","
                            "  \"supportInfo\": {"
                            "    \"simplifiedKey\": ["
                            "      \"AddTik2/d=0,p=0/1,2/1,2/1,2\","
                            "      \"AddTik2/d=1,p=0/1,2/1,2/1,2\""
                            "    ]"
                            "  }"
                            "}";
std::string AddTik201Bin = "01";
std::string AddTik202Json = "{"
                            "  \"filePath\": \"ascend910/add_tik2/Add_Tik2_02.json\","
                            "  \"supportInfo\": {"
                            "    \"simplifiedKey\": ["
                            "      \"AddTik2/d=0,p=0/1,2/0,2/0,2\","
                            "      \"AddTik2/d=1,p=0/1,2/0,2/0,2\""
                            "    ]"
                            "  }"
                            "}";
std::string AddTik202Bin = "02";

std::vector<std::tuple<const uint8_t*, const uint8_t*>> addTik2OpBinary(
  {{(const uint8_t*)AddTik2Json.c_str(), (const uint8_t*)AddTik2Json.c_str() + AddTik2Json.size()},
   {(const uint8_t*)AddTik201Json.c_str(), (const uint8_t*)AddTik201Json.c_str() + AddTik201Json.size()},
   {(const uint8_t*)AddTik201Bin.c_str(), (const uint8_t*)AddTik201Bin.c_str() + AddTik201Bin.size()},
   {(const uint8_t*)AddTik202Json.c_str(), (const uint8_t*)AddTik202Json.c_str() + AddTik202Json.size()},
   {(const uint8_t*)AddTik202Bin.c_str(), (const uint8_t*)AddTik202Bin.c_str() + AddTik202Bin.size()}});

TEST_F(OpBinaryResourceManagerUT, SaveBinary) {
  EXPECT_EQ(manager.AddBinary("AddTik2", addTik2OpBinary), ge::GRAPH_SUCCESS);
  EXPECT_EQ(manager.opBinaryDesc_.size(), 1);
  auto it = manager.opBinaryDesc_.find("AddTik2");
  ASSERT_NE(it, manager.opBinaryDesc_.end());
  auto list = it->second;
  EXPECT_EQ(list.size(), 1);

  auto binIter = manager.pathToBinary_.find("ascend910/add_tik2/Add_Tik2_01.json");
  ASSERT_NE(binIter, manager.pathToBinary_.end());
  auto binJson = std::get<0U>(binIter->second);
  auto bin = std::get<1U>(binIter->second);
  auto filePath = binJson["filePath"].get<std::string>();
  EXPECT_EQ(filePath, "ascend910/add_tik2/Add_Tik2_01.json");
  EXPECT_EQ(bin.content, (const uint8_t*)AddTik201Bin.c_str());
  EXPECT_EQ(bin.len, AddTik201Bin.size());

  EXPECT_EQ(manager.keyToPath_["AddTik2/d=0,p=0/1,2/1,2/1,2"], "ascend910/add_tik2/Add_Tik2_01.json");
  EXPECT_EQ(manager.keyToPath_["AddTik2/d=1,p=0/1,2/1,2/1,2"], "ascend910/add_tik2/Add_Tik2_01.json");
  EXPECT_EQ(manager.keyToPath_["AddTik2/d=0,p=0/1,2/0,2/0,2"], "ascend910/add_tik2/Add_Tik2_02.json");
  EXPECT_EQ(manager.keyToPath_["AddTik2/d=1,p=0/1,2/0,2/0,2"], "ascend910/add_tik2/Add_Tik2_02.json");
}

TEST_F(OpBinaryResourceManagerUT, BinaryForJson) {
  nlohmann::json binDesc;
  EXPECT_EQ(manager.GetOpBinaryDesc("AddTik2", binDesc), ge::GRAPH_SUCCESS);
  auto keys = binDesc["binList"][0]["simplifiedKey"].get<std::vector<std::string>>();
  EXPECT_EQ(keys[0], "AddTik2/d=0,p=0/1,2/1,2/1,2");
  EXPECT_EQ(keys[1], "AddTik2/d=1,p=0/1,2/1,2/1,2");
  auto jsonFilePath = binDesc["binList"][0]["binInfo"]["jsonFilePath"].get<std::string>();
  EXPECT_EQ(jsonFilePath, "ascend910/add_tik2/Add_Tik2_01.json");

  keys = binDesc["binList"][1]["simplifiedKey"].get<std::vector<std::string>>();
  EXPECT_EQ(keys[0], "AddTik2/d=0,p=0/1,2/0,2/0,2");
  EXPECT_EQ(keys[1], "AddTik2/d=1,p=0/1,2/0,2/0,2");
  jsonFilePath = binDesc["binList"][1]["binInfo"]["jsonFilePath"].get<std::string>();
  EXPECT_EQ(jsonFilePath, "ascend910/add_tik2/Add_Tik2_02.json");
}

TEST_F(OpBinaryResourceManagerUT, KeyToBinary) {
  std::tuple<nlohmann::json, nnopbase::Binary> binInfo;
  EXPECT_EQ(manager.GetOpBinaryDescByKey("AddTik2/d=1,p=0/1,2/1,2/1,2", binInfo), ge::GRAPH_SUCCESS);
  auto binJson = std::get<0U>(binInfo);
  auto bin = std::get<1U>(binInfo);
  auto filePath = binJson["filePath"].get<std::string>();
  EXPECT_EQ(filePath, "ascend910/add_tik2/Add_Tik2_01.json");
  EXPECT_EQ(bin.content, (const uint8_t*)AddTik201Bin.c_str());
  EXPECT_EQ(bin.len, AddTik201Bin.size());

  EXPECT_EQ(manager.GetOpBinaryDescByKey("AddTik2/d=1,p=0/1,2/0,2/0,2", binInfo), ge::GRAPH_SUCCESS);
  binJson = std::get<0U>(binInfo);
  bin = std::get<1U>(binInfo);
  filePath = binJson["filePath"].get<std::string>();
  EXPECT_EQ(filePath, "ascend910/add_tik2/Add_Tik2_02.json");
  EXPECT_EQ(bin.content, (const uint8_t*)AddTik202Bin.c_str());
  EXPECT_EQ(bin.len, AddTik202Bin.size());
}

TEST_F(OpBinaryResourceManagerUT, PathToBinary) {
  std::tuple<nlohmann::json, nnopbase::Binary> binInfo;
  EXPECT_EQ(manager.GetOpBinaryDescByPath("ascend910/add_tik2/Add_Tik2_01.json", binInfo), ge::GRAPH_SUCCESS);
  auto binJson = std::get<0U>(binInfo);
  auto bin = std::get<1U>(binInfo);
  auto filePath = binJson["filePath"].get<std::string>();
  EXPECT_EQ(filePath, "ascend910/add_tik2/Add_Tik2_01.json");
  EXPECT_EQ(bin.content, (const uint8_t*)AddTik201Bin.c_str());
  EXPECT_EQ(bin.len, AddTik201Bin.size());

  EXPECT_EQ(manager.GetOpBinaryDescByPath("ascend910/add_tik2/Add_Tik2_02.json", binInfo), ge::GRAPH_SUCCESS);
  binJson = std::get<0U>(binInfo);
  bin = std::get<1U>(binInfo);
  filePath = binJson["filePath"].get<std::string>();
  EXPECT_EQ(filePath, "ascend910/add_tik2/Add_Tik2_02.json");
  EXPECT_EQ(bin.content, (const uint8_t*)AddTik202Bin.c_str());
  EXPECT_EQ(bin.len, AddTik202Bin.size());
}

TEST_F(OpBinaryResourceManagerUT, BinaryAllDesc) {
  auto &map = manager.GetAllOpBinaryDesc();
  EXPECT_EQ(map.size(), 1);
  auto it = map.find("AddTik2");
  ASSERT_NE(it, map.end());

  auto keys = (it->second)["binList"][0]["simplifiedKey"].get<std::vector<std::string>>();
  EXPECT_EQ(keys[0], "AddTik2/d=0,p=0/1,2/1,2/1,2");
  EXPECT_EQ(keys[1], "AddTik2/d=1,p=0/1,2/1,2/1,2");
  auto jsonFilePath = (it->second)["binList"][0]["binInfo"]["jsonFilePath"].get<std::string>();
  EXPECT_EQ(jsonFilePath, "ascend910/add_tik2/Add_Tik2_01.json");

  keys = (it->second)["binList"][1]["simplifiedKey"].get<std::vector<std::string>>();
  EXPECT_EQ(keys[0], "AddTik2/d=0,p=0/1,2/0,2/0,2");
  EXPECT_EQ(keys[1], "AddTik2/d=1,p=0/1,2/0,2/0,2");
  jsonFilePath = (it->second)["binList"][1]["binInfo"]["jsonFilePath"].get<std::string>();
  EXPECT_EQ(jsonFilePath, "ascend910/add_tik2/Add_Tik2_02.json");
}

std::string AddTik2KbRuntime = "1234";
std::vector<std::tuple<const uint8_t*, const uint8_t*>> addTik2RuntimeKb(
  {{(const uint8_t*)AddTik2KbRuntime.c_str(), (const uint8_t*)AddTik2KbRuntime.c_str() + AddTik2KbRuntime.size()}});

TEST_F(OpBinaryResourceManagerUT, RuntimeKB) {
  EXPECT_EQ(manager.AddRuntimeKB("AddTik2", addTik2RuntimeKb), ge::GRAPH_SUCCESS);
  // 重复添加正常
  EXPECT_EQ(manager.AddRuntimeKB("AddTik2", addTik2RuntimeKb), ge::GRAPH_SUCCESS);
  std::vector<ge::AscendString> kbList;
  EXPECT_EQ(manager.GetOpRuntimeKB("AddTik2", kbList), ge::GRAPH_SUCCESS);
  EXPECT_EQ(kbList.size(), 1);
  EXPECT_EQ(AddTik2KbRuntime, kbList[0].GetString());
}

TEST_F(OpBinaryResourceManagerUT, Error) {
  nlohmann::json binDesc;
  EXPECT_EQ(manager.GetOpBinaryDesc("AddTik2Invalid",binDesc), ge::GRAPH_PARAM_INVALID);

  std::tuple<nlohmann::json, nnopbase::Binary> binInfo;
  EXPECT_EQ(manager.GetOpBinaryDescByPath("AddTik2Invalid", binInfo), ge::GRAPH_PARAM_INVALID);
  EXPECT_EQ(manager.GetOpBinaryDescByKey("AddTik2Invalid", binInfo), ge::GRAPH_PARAM_INVALID);

  std::vector<ge::AscendString> kbList;
  EXPECT_EQ(manager.GetOpRuntimeKB("AddTik2Invalid", kbList), ge::GRAPH_PARAM_INVALID);
}
