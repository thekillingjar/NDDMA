/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <regex>
#include <map>
#include "gtest/gtest.h"
#include "tiling_code_generator.h"
#include "test_common_utils.h"

using namespace att;
namespace {
constexpr const char *kTilingPath = "./test/sample/gen_data_st/";
constexpr const char *kTilingDataSuffix = "tiling_data.h";
constexpr const char *kTilingFuncSuffix = "tiling_func.cpp";
constexpr const char *kInputJsonFileSuffix = "input_shapes.json";
constexpr const char *kTilingSummaryKey = "The value of ";
constexpr const char *kGenDataFile = "./result.log";
constexpr const char *kGenDataLog = "./tiling_func.log";
constexpr const char kSampleDir[] = "./test/sample/";
constexpr const char kInstallDir[] = "./tests/autofuse/st/att/testcase/sample/";

std::string ReadFile(const std::string& filename) {
  std::ifstream file(filename, std::ios::binary);
  std::string content;

  if (file) {
    std::ostringstream ss;
    ss << file.rdbuf();
    content = ss.str();
  }
  file.close();
  return content;
}

std::map<std::string, std::string> ParseKeyValuePairs(const std::string &log, const std::string &split_key) {
  std::map<std::string, std::string> key_val_pairs;
  std::regex key_val_pair_regex(std::string("(\\w+)") + split_key + std::string("([\\w]+)"));  // 正则表达式匹配键值对
  std::sregex_iterator begin(log.begin(), log.end(), key_val_pair_regex);
  std::sregex_iterator end;

  for (std::sregex_iterator i = begin; i != end; ++i) {
    std::smatch match = *i;
    std::string key = match.str(1);
    std::string value = match.str(2);
    key_val_pairs[key] = value;
  }
  return key_val_pairs;
}

bool FileExists(const std::string &directory, const std::string &filename_suffix) {
  // 遍历 tmp 目录下的所有文件
  bool ret = false;
  for (const auto &entry : std::filesystem::directory_iterator(directory)) {
    std::string filename = entry.path().filename().string();
    if (filename.size() >= sizeof(filename_suffix) &&
        (filename.find(filename_suffix) != std::string::npos)) {
      std::cout << "Found file: " << filename << std::endl;
      ret = true;
    }
  }
  return ret;
}

std::string GetGenCodeCmd(const std::string &op_type, const std::string &scence = "tool") {
  std::string build_cmd(std::string(kSampleDir) + "./build.sh --build_type=gen --target=code");
  build_cmd.append(" --example=").append(op_type);
  build_cmd.append(" --scence=").append(scence);
#ifdef ASCEND_INSTALL_PATH
  build_cmd.append(" --ascend_install_path=").append(ASCEND_INSTALL_PATH);
#endif
  std::cout << "build_cmd=" << build_cmd << std::endl;
  return build_cmd;
}

std::string GetGenCodeWithCtxCmd(const std::string &op_type) {
  std::string build_cmd(std::string(kSampleDir) + "./build.sh --build_type=gen --target=code_with_ctx");
  build_cmd.append(" --example=").append(op_type);
#ifdef ASCEND_INSTALL_PATH
  build_cmd.append(" --ascend_install_path=").append(ASCEND_INSTALL_PATH);
#endif
  std::cout << "build_cmd=" << build_cmd << std::endl;
  return build_cmd;
}

std::string GetGenDataCmd() {
  std::string build_cmd(std::string(kSampleDir) + "./build.sh --build_type=gen --target=data --log=0");
#ifdef ASCEND_INSTALL_PATH
  build_cmd.append(" --ascend_install_path=").append(ASCEND_INSTALL_PATH);
#endif
  std::cout << "build_cmd=" << build_cmd << std::endl;
  return build_cmd;
}
}
class TestAttE2e : public ::testing::Test {
 public:
  static void TearDownTestCase()
  {
    std::cout << "Test end." << std::endl;
  }
  static void SetUpTestCase()
  {
    std::cout << "Test begin." << std::endl;
  }
  static bool IsGenDataSuccess() {
    constexpr int64_t kMaxBlockDim = 64L;
    constexpr int64_t kMaxUBSize = 240 * 1024;
    constexpr int64_t kMaxHbmSize = 180 * 1024 * 1024;
    auto gen_data_cmd = GetGenDataCmd().append(" > ").append(kGenDataLog);
    std::cout << "gen_data_cmd=" << gen_data_cmd << std::endl;
    auto ret = std::system(gen_data_cmd.c_str());
    // 执行失败
    if (ret != 0) {
      auto cat_cmd = std::string("cat ").append(kGenDataLog);
      std::cout << "cat_cmd=" << cat_cmd << std::endl;
      (void)std::system(cat_cmd.c_str());
      return false;
    }
    auto cat_cmd = std::string("cat ")
                       .append(kGenDataLog)
                       .append(" | grep '")
                       .append(kTilingSummaryKey)
                       .append("'>")
                       .append(kGenDataFile);
    std::cout << "cat_cmd=" << cat_cmd << std::endl;
    (void)std::system(cat_cmd.c_str());
    auto read_string = ReadFile(kGenDataFile);
    std::map<std::string, std::string> result = ParseKeyValuePairs(read_string, " is ");
    std::map<std::string, std::pair<int64_t, int64_t>> expect_res = {
        {"ub_size", std::pair<int64_t, int64_t>(kMaxUBSize * 0.8, kMaxUBSize)},
        {"hbm_size", std::pair<int64_t, int64_t>(0, kMaxHbmSize)},
        {"block_dim", std::pair<int64_t, int64_t>(1, kMaxBlockDim)},
    };
    for (const auto &expect : expect_res) {
      int64_t res = std::stoi(result[expect.first]);
      EXPECT_TRUE(expect.second.first <= res);
      EXPECT_TRUE(res <= expect.second.second);
    }
    return true;
  }
  void SetUp() override {
    (void)std::system((std::string("rm ") + kGenDataFile).c_str());
    (void)std::system((std::string("rm ") + kGenDataFile).c_str());
    // (void)std::system((std::string(kSampleDir) + "./build.sh --build_type=clean").c_str());
    (void)std::system((std::string(kSampleDir) + "build.sh --build_type=clean").c_str());
    (void)system(("rm -rf " + std::string(kSampleDir)).c_str());
    // Code here will be called immediately after the constructor (right
    // before each test).
    auto install_cmd = std::string("").append(TOP_DIR).append(std::string(kInstallDir) + "/install.sh ./test/");
    std::cout << "install_cmd=" << install_cmd << std::endl;
    (void)system(install_cmd.c_str());
  }

  void TearDown() override {
    // 清理测试生成的临时文件
    autofuse::test::CleanupTestArtifacts();
    // before the destructor).
  }
};
#if 0
TEST_F(TestAttE2e, test_elementwise_gelu_003) {
  auto ret = std::system(GetGenCodeCmd("gelu").c_str());
  system("pwd");
  EXPECT_EQ(ret, 0);
  ASSERT_TRUE(FileExists(kTilingPath, kTilingDataSuffix));
  ASSERT_TRUE(FileExists(kTilingPath, kTilingFuncSuffix));
  ret = std::system(R"(sed -i 's|// TODO 用户可根据期望获取的数据，打印不同输出的值|printf(\"Get tiling data:get_b0_size=%u, get_b1_size=%u, get_b2_size=%u, get_b3_size=%u, get_b4_size=%u, get_ndb_size=%u, get_ndbt_size=%u, get_ND=%u, get_block_dim=%u\", tiling_data.get_b0_size(), tiling_data.get_b1_size(), tiling_data.get_b2_size(), tiling_data.get_b3_size(), tiling_data.get_b4_size(), tiling_data.get_ndb_size(), tiling_data.get_ndbt_size(), tiling_data.get_ND(), tiling_data.get_block_dim());\n|' ./test/sample/gen_data_st/tiling_func_main.cpp)");
  EXPECT_EQ(ret, 0);
  ASSERT_TRUE(FileExists(kTilingPath, kInputJsonFileSuffix));
  std::string input_json_str = R"(
{
    "input_shapes": [
        {
            "axes": [
                [
                    "ND",
                    1024
                ]
            ],
            "tiling_key": 0
        }
    ]
}
      )";
  std::ofstream input_json(std::string(kTilingPath) + std::string(kInputJsonFileSuffix));
  ASSERT_TRUE(input_json.is_open());
  input_json << input_json_str;
  input_json.close();
  ret = std::system(GetGenDataCmd().append(" | grep 'Get tiling data' >").append(kGenDataFile).c_str());
  EXPECT_EQ(ret, 0);
  auto read_string = ReadFile(kGenDataFile);
  std::map<std::string, std::string> result = ParseKeyValuePairs(read_string, "=");
  std::map<std::string, int32_t> expect_min_res = {
      {"get_b0_size", 4096},
      {"get_b1_size", 4096},
      {"get_b2_size", 4096},
      {"get_b3_size", 4096},
      {"get_b4_size", 4096},
      {"get_ndb_size", 1},
      {"get_ndbt_size", 1},
      {"get_ND", 1},
      {"get_block_dim", 1}
  };
  for (const auto &pair : expect_min_res) {
    EXPECT_TRUE(pair.second <= std::stoi(result[pair.first]));
  }
  ret = std::system((std::string("rm ") + kGenDataFile).c_str());
}

TEST_F(TestAttE2e, test_elementwise_gelu_v2_006) {
  setenv("ASCEND_GLOBAL_LOG_LEVEL", "4", 1);
  auto ret = std::system(GetGenCodeCmd("gelu_v2").c_str());
  EXPECT_EQ(ret, 0);
  ASSERT_TRUE(FileExists(kTilingPath, kTilingDataSuffix));
  ASSERT_TRUE(FileExists(kTilingPath, kTilingFuncSuffix));
  ret = std::system("sed -i 's|// TODO 用户可根据期望获取的数据，打印不同输出的值|"
                    "printf(\"Get tiling data:get_ndb_size=%u, get_ndbt_size=%u, get_ND=%u, get_block_dim=%u\", tiling_data.get_ndb_size(), tiling_data.get_ndbt_size(), tiling_data.get_ND(), tiling_data.get_block_dim());|' ./test/sample/gen_data_st/tiling_func_main.cpp");
  EXPECT_EQ(ret, 0);
  ASSERT_TRUE(FileExists(kTilingPath, kInputJsonFileSuffix));
  std::string input_json_str = R"(
  {
      "input_shapes": [
          {
              "axes": [
                  [
                      "ND",
                      1024
                  ]
              ],
              "tiling_key": 0
          }
      ]
  }
        )";
  std::ofstream input_json(std::string(kTilingPath) + std::string(kInputJsonFileSuffix));
  ASSERT_TRUE(input_json.is_open());
  input_json << input_json_str;
  input_json.close();
  ret = std::system(GetGenDataCmd().append(" | grep 'Get tiling data' > ").append(kGenDataFile).c_str());
  EXPECT_EQ(ret, 0);
  auto read_string = ReadFile(kGenDataFile);
  std::map<std::string, std::string> result = ParseKeyValuePairs(read_string, "=");
  std::map<std::string, int32_t> expect_min_res = {
      {"get_ndb_size", 1},
      {"get_ndbt_size", 1},
      {"get_ND", 1},
      {"get_block_dim", 1}
  };
  for (const auto &pair : expect_min_res) {
    EXPECT_TRUE(pair.second <= std::stoi(result[pair.first]));
  }
  ret = std::system(std::string("rm ").append(kGenDataFile).c_str());
  unsetenv("ASCEND_GLOBAL_LOG_LEVEL");
}


TEST_F(TestAttE2e, test_elementwise_gelu_007_autofuse) {
  setenv("ASCEND_GLOBAL_LOG_LEVEL", "1", 1);
  auto ret = std::system(GetGenCodeCmd("gelu", "autofuse").c_str());
  EXPECT_EQ(ret, 0);
  ASSERT_TRUE(FileExists(kTilingPath, kTilingDataSuffix));
  ASSERT_TRUE(FileExists(kTilingPath, kTilingFuncSuffix));
  ret = std::system("sed -i 's|// TODO 用户可根据期望获取的数据，打印不同输出的值|"
      "printf(\"Get tiling data:get_ndb_size=%u, get_ndbt_size=%u, get_ND=%u, get_block_dim=%u\", tiling_data.get_ndb_size(), tiling_data.get_ndbt_size(), tiling_data.get_ND(), tiling_data.get_block_dim());|' ./test/sample/gen_data_st/tiling_func_main.cpp");
  EXPECT_EQ(ret, 0);
  ASSERT_TRUE(FileExists(kTilingPath, kInputJsonFileSuffix));
  std::string input_json_str = R"(
{
    "input_shapes": [
        {
            "axes": [
                [
                    "ND",
                    1024
                ]
            ],
            "tiling_key": 0
        }
    ]
}
      )";
  std::ofstream input_json(std::string(kTilingPath) + std::string(kInputJsonFileSuffix));
  ASSERT_TRUE(input_json.is_open());
  input_json << input_json_str;
  input_json.close();
  ret = std::system(GetGenDataCmd().append(" | grep 'Get tiling data' >").append(kGenDataFile).c_str());
  EXPECT_EQ(ret, 0);
  auto read_string = ReadFile(kGenDataFile);
  std::map<std::string, std::string> result = ParseKeyValuePairs(read_string, "=");
  std::map<std::string, int32_t> expect_min_res = {
      {"get_ndb_size", 1},
      {"get_ndbt_size", 1},
      {"get_ND", 1},
      {"get_block_dim", 1}
  };
  for (const auto &pair : expect_min_res) {
    EXPECT_TRUE(pair.second <= std::stoi(result[pair.first]));
  }
  ret = std::system((std::string("rm ") + kGenDataFile).c_str());
  unsetenv("ASCEND_GLOBAL_LOG_LEVEL");
}
#endif