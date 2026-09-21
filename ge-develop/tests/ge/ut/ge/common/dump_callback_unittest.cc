/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#include "common/dump/dump_callback.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include <fstream>
#include <cstring>

using namespace testing;
using ge::DumpConfig;
using ge::ModelDumpConfig;
using ge::DumpConfigValidator;
using ge::DumpCallbackManager;

// Mock GeExecutor 类
class MockGeExecutor {
public:
    MOCK_METHOD(ge::Status, SetDump, (const ge::DumpConfig& config));
};

// 测试数据
namespace ge{

const char* kValidDumpConfig = R"({
    "dump": {
        "dump_path": "/tmp/dump_test",
        "dump_mode": "all",
        "dump_level": "op",
        "dump_step": "1|3-5",
        "dump_list": [
            {
                "model_name": "test_model",
                "layers": ["layer1", "layer2"],
                "watcher_nodes": ["node1", "node2"],
                "optype_blacklist": [
                    {
                        "name": "Const",
                        "pos": ["input"]
                    }
                ],
                "opname_blacklist": [
                    {
                        "name": "node_to_skip",
                        "pos": ["output"]
                    }
                ],
                "op_name_range": [
                    {
                        "begin": "conv1",
                        "end": "conv5"
                    }
                ]
            }
        ]
    }
})";

const char* kInvalidDumpConfig = R"({
    "dump": {
        "dump_mode": "all"
    }
})";

const char* kDebugDumpConfig = R"({
    "dump": {
        "dump_path": "/tmp/debug_dump",
        "dump_debug": "on"
    }
})";

const char* kEmptyDumpConfig = R"({})";

const char* kDisableDumpConfig = R"({
    "dump": {
        "dump_status": "off",
        "dump_path": "/tmp/dummy_dump_path"
    }
})";

const std::string kDumpConfigWithLongPath = R"({
    "dump": {
        "dump_path": "/tmp/)" + std::string(MAX_DUMP_PATH_LENGTH + 10, 'a') + R"("
    }
})";
} // namespace

// DumpConfigValidator 测试类
class DumpConfigValidatorTest : public Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// 测试有效的dump配置验证
TEST_F(DumpConfigValidatorTest, IsValidDumpConfig_ValidConfig_ReturnsTrue) {
    nlohmann::json config = nlohmann::json::parse(ge::kValidDumpConfig);
    EXPECT_TRUE(DumpConfigValidator::IsValidDumpConfig(config));
}

// 测试无效的dump配置验证 - 缺少dump_path
TEST_F(DumpConfigValidatorTest, IsValidDumpConfig_InvalidConfig_ReturnsFalse) {
    nlohmann::json config = nlohmann::json::parse(ge::kInvalidDumpConfig);
    EXPECT_FALSE(DumpConfigValidator::IsValidDumpConfig(config));
}

// 测试空配置
TEST_F(DumpConfigValidatorTest, IsValidDumpConfig_EmptyConfig_ReturnsTrue) {
    nlohmann::json config = nlohmann::json::parse(ge::kEmptyDumpConfig);
    EXPECT_TRUE(DumpConfigValidator::IsValidDumpConfig(config));
}

// 测试debug dump配置
TEST_F(DumpConfigValidatorTest, IsValidDumpConfig_DebugConfig_ReturnsTrue) {
    nlohmann::json config = nlohmann::json::parse(ge::kDebugDumpConfig);
    EXPECT_TRUE(DumpConfigValidator::IsValidDumpConfig(config));
}

// 测试dump路径验证 - 有效路径
TEST_F(DumpConfigValidatorTest, CheckDumpPath_ValidPath_ReturnsTrue) {
    nlohmann::json config = nlohmann::json::parse(ge::kValidDumpConfig);

    // 确保传递正确的 dump 配置部分
    nlohmann::json dumpConfig = config["dump"];
    EXPECT_TRUE(ge::DumpConfigValidator::CheckDumpPath(dumpConfig));
}

// 测试dump路径验证 - 空路径
TEST_F(DumpConfigValidatorTest, CheckDumpPath_EmptyPath_ReturnsFalse) {
    nlohmann::json config = nlohmann::json::parse(R"({
        "dump": {
            "dump_path": ""
        }
    })");
    nlohmann::json dumpConfig = config["dump"];
    EXPECT_FALSE(DumpConfigValidator::CheckDumpPath(dumpConfig));
}

// 测试dump步骤验证 - 有效步骤
TEST_F(DumpConfigValidatorTest, CheckDumpStep_ValidSteps_ReturnsTrue) {
    nlohmann::json config = nlohmann::json::parse(ge::kValidDumpConfig);
    nlohmann::json dumpConfig = config["dump"];
    EXPECT_TRUE(DumpConfigValidator::CheckDumpStep(dumpConfig));
}

// 测试dump步骤验证 - 无效步骤格式
TEST_F(DumpConfigValidatorTest, CheckDumpStep_InvalidSteps_ReturnsFalse) {
    nlohmann::json config = nlohmann::json::parse(R"({
        "dump": {
            "dump_path": "/tmp/dump",
            "dump_step": "1-2-3"
        }
    })");
    nlohmann::json dumpConfig = config["dump"];
    EXPECT_FALSE(DumpConfigValidator::CheckDumpStep(dumpConfig));
}

// 测试dump步骤验证 - 包含非数字字符
TEST_F(DumpConfigValidatorTest, CheckDumpStep_StepsWithNonDigit_ReturnsFalse) {
    nlohmann::json config = nlohmann::json::parse(R"({
        "dump": {
            "dump_path": "/tmp/dump",
            "dump_step": "1|abc"
        }
    })");
    nlohmann::json dumpConfig = config["dump"];
    EXPECT_FALSE(DumpConfigValidator::CheckDumpStep(dumpConfig));
}

TEST_F(DumpConfigValidatorTest, NeedDump_ConfigNeedsDump_ReturnsTrue) {
    DumpConfig config;

    // 测试场景1: dump_op_switch = "on"
    config.dump_op_switch = "on";
    config.dump_debug = "off";
    config.dump_exception = "";
    config.dump_list.clear();
    EXPECT_TRUE(DumpConfigValidator::NeedDump(config));

    // 测试场景2: dump_debug = "on"
    config.dump_op_switch = "off";
    config.dump_debug = "on";
    EXPECT_TRUE(DumpConfigValidator::NeedDump(config));

    // 测试场景3: dump_list非空
    config.dump_debug = "off";
    config.dump_list.push_back(ModelDumpConfig());
    EXPECT_TRUE(DumpConfigValidator::NeedDump(config));

    // 测试场景4: dump_exception非空
    config.dump_list.clear();
    config.dump_exception = "exception_dump";
    EXPECT_TRUE(DumpConfigValidator::NeedDump(config));

    // 测试场景5: 大小写混合的"on" (如果比较是大小写敏感的)
    config.dump_exception = "";
    config.dump_op_switch = "on";
    EXPECT_TRUE(DumpConfigValidator::NeedDump(config));

    // 测试场景6: 多个条件同时满足
    config.dump_op_switch = "on";
    config.dump_debug = "on";
    config.dump_exception = "exception_dump";
    config.dump_list.push_back(ModelDumpConfig());
    EXPECT_TRUE(DumpConfigValidator::NeedDump(config));
}

// 测试NeedDump功能 - 不需要dump的情况
TEST_F(DumpConfigValidatorTest, NeedDump_ConfigNoNeedDump_ReturnsFalse) {
    DumpConfig config;
    config.dump_op_switch = "off";
    config.dump_debug = "off";
    EXPECT_FALSE(DumpConfigValidator::NeedDump(config));
}

// 测试JSON解析功能 - 有效配置
TEST_F(DumpConfigValidatorTest, ParseDumpConfig_ValidJson_ParsesCorrectly) {
    std::string configStr = ge::kValidDumpConfig;
    DumpConfig config;

    bool result = DumpConfigValidator::ParseDumpConfig(
        configStr.c_str(), configStr.size(), config);

    EXPECT_TRUE(result);
    EXPECT_EQ(config.dump_path, "/tmp/dump_test");
    EXPECT_EQ(config.dump_mode, "all");
    EXPECT_EQ(config.dump_level, "op");
    EXPECT_EQ(config.dump_step, "1|3-5");
    EXPECT_FALSE(config.dump_list.empty());
}

// 测试JSON解析功能 - 空数据
TEST_F(DumpConfigValidatorTest, ParseDumpConfig_NullData_ReturnsFalse) {
    DumpConfig config;
    bool result = DumpConfigValidator::ParseDumpConfig(nullptr, 10, config);
    EXPECT_FALSE(result);
}

// 测试JSON解析功能 - 无效数据大小
TEST_F(DumpConfigValidatorTest, ParseDumpConfig_InvalidSize_ReturnsFalse) {
    std::string configStr = ge::kValidDumpConfig;
    DumpConfig config;
    bool result = DumpConfigValidator::ParseDumpConfig(
        configStr.c_str(), 0, config);
    EXPECT_FALSE(result);

    result = DumpConfigValidator::ParseDumpConfig(
        configStr.c_str(), -1, config);
    EXPECT_FALSE(result);
}

// 测试JSON解析功能 - 无效JSON
TEST_F(DumpConfigValidatorTest, ParseDumpConfig_InvalidJson_ReturnsFalse) {
    const char* invalidJson = "{ invalid json }";
    DumpConfig config;
    bool result = DumpConfigValidator::ParseDumpConfig(
        invalidJson, strlen(invalidJson), config);
    EXPECT_FALSE(result);
}

// 测试路径有效性检查 - 有效路径
TEST_F(DumpConfigValidatorTest, IsDumpPathValid_ValidPath_ReturnsTrue) {
    EXPECT_TRUE(DumpConfigValidator::IsDumpPathValid("/tmp/valid_path"));
    EXPECT_TRUE(DumpConfigValidator::IsDumpPathValid("/tmp/path-with_underscore"));
    EXPECT_TRUE(DumpConfigValidator::IsDumpPathValid("/tmp/path.with.dots"));
}

// 测试路径有效性检查 - 无效路径
TEST_F(DumpConfigValidatorTest, IsDumpPathValid_InvalidPath_ReturnsFalse) {
    EXPECT_FALSE(DumpConfigValidator::IsDumpPathValid(""));  // 空路径
    EXPECT_FALSE(DumpConfigValidator::IsDumpPathValid("/tmp/invalid path"));  // 空格
    EXPECT_FALSE(DumpConfigValidator::IsDumpPathValid("/tmp/invalid`path"));  // 反引号
    EXPECT_FALSE(DumpConfigValidator::IsDumpPathValid("/tmp/invalid~path"));  // 波浪号
    EXPECT_FALSE(DumpConfigValidator::IsDumpPathValid("/tmp/invalid<path"));  // 小于号
    EXPECT_FALSE(DumpConfigValidator::IsDumpPathValid("/tmp/invalid>path"));  // 大于号
    EXPECT_FALSE(DumpConfigValidator::IsDumpPathValid("/tmp/invalid|path"));  // 管道符
}

// 测试路径有效性检查 - 路径过长
TEST_F(DumpConfigValidatorTest, IsDumpPathValid_PathTooLong_ReturnsFalse) {
    std::string longPath(ge::MAX_DUMP_PATH_LENGTH + 10, 'a');
    EXPECT_FALSE(DumpConfigValidator::IsDumpPathValid(longPath));
}

// 测试目录字符串有效性检查
TEST_F(DumpConfigValidatorTest, IsValidDirStr_ValidChars_ReturnsTrue) {
    EXPECT_TRUE(DumpConfigValidator::IsValidDirStr("/tmp/valid-path_123"));
    EXPECT_TRUE(DumpConfigValidator::IsValidDirStr("./relative/path"));
    EXPECT_TRUE(DumpConfigValidator::IsValidDirStr("simple_name"));
}

// 测试目录字符串有效性检查 - 无效字符
TEST_F(DumpConfigValidatorTest, IsValidDirStr_InvalidChars_ReturnsFalse) {
    EXPECT_FALSE(DumpConfigValidator::IsValidDirStr("/tmp/invalid~path"));  // 波浪号
    EXPECT_FALSE(DumpConfigValidator::IsValidDirStr("/tmp/invalid<path"));  // 小于号
    EXPECT_FALSE(DumpConfigValidator::IsValidDirStr("/tmp/invalid>path"));  // 大于号
    EXPECT_FALSE(DumpConfigValidator::IsValidDirStr("/tmp/invalid|path"));  // 管道符
}

// 测试dump模式验证
TEST_F(DumpConfigValidatorTest, IsValidDumpConfig_InvalidDumpMode_ReturnsFalse) {
    nlohmann::json config = nlohmann::json::parse(R"({
        "dump": {
            "dump_path": "/tmp/dump",
            "dump_mode": "invalid_mode"
        }
    })");
    EXPECT_FALSE(DumpConfigValidator::IsValidDumpConfig(config));
}

// 测试dump级别验证
TEST_F(DumpConfigValidatorTest, IsValidDumpConfig_InvalidDumpLevel_ReturnsFalse) {
    nlohmann::json config = nlohmann::json::parse(R"({
        "dump": {
            "dump_path": "/tmp/dump",
            "dump_level": "invalid_level"
        }
    })");
    EXPECT_FALSE(DumpConfigValidator::IsValidDumpConfig(config));
}

// 测试dump操作开关验证
TEST_F(DumpConfigValidatorTest, CheckDumpOpSwitch_ValidSwitch_ReturnsTrue) {
    EXPECT_TRUE(DumpConfigValidator::CheckDumpOpSwitch("on"));
    EXPECT_TRUE(DumpConfigValidator::CheckDumpOpSwitch("off"));
}

TEST_F(DumpConfigValidatorTest, CheckDumpOpSwitch_InvalidSwitch_ReturnsFalse) {
    EXPECT_FALSE(DumpConfigValidator::CheckDumpOpSwitch("invalid"));
    EXPECT_FALSE(DumpConfigValidator::CheckDumpOpSwitch(""));
}

// 测试Split函数
TEST_F(DumpConfigValidatorTest, Split_ValidString_ReturnsCorrectVector) {
    std::vector<std::string> result = DumpConfigValidator::Split("a|b|c", "|");
    EXPECT_EQ(result.size(), 3U);
    EXPECT_EQ(result[0], "a");
    EXPECT_EQ(result[1], "b");
    EXPECT_EQ(result[2], "c");
}

TEST_F(DumpConfigValidatorTest, Split_EmptyString_ReturnsEmptyVector) {
    std::vector<std::string> result = DumpConfigValidator::Split("", "|");
    EXPECT_TRUE(result.empty());
}

TEST_F(DumpConfigValidatorTest, Split_NullDelimiter_ReturnsEmptyVector) {
    std::vector<std::string> result = DumpConfigValidator::Split("a|b|c", nullptr);
    EXPECT_TRUE(result.empty());
}

// DumpCallbackManager 测试类
class DumpCallbackManagerTest : public Test {
protected:
    void SetUp() override {
        // 可以在这里初始化测试环境
    }

    void TearDown() override {
        // 清理测试环境
    }
};

// 测试单例模式
TEST_F(DumpCallbackManagerTest, GetInstance_ReturnsSameInstance) {
    DumpCallbackManager& instance1 = DumpCallbackManager::GetInstance();
    DumpCallbackManager& instance2 = DumpCallbackManager::GetInstance();
    EXPECT_EQ(&instance1, &instance2);
}

// 测试启用dump回调 - 有效配置
TEST_F(DumpCallbackManagerTest, EnableDumpCallback_ValidConfig_ReturnsSuccess) {
    std::string configStr = ge::kValidDumpConfig;
    int32_t result = DumpCallbackManager::EnableDumpCallback(
        1, const_cast<char*>(configStr.c_str()), configStr.size());
    EXPECT_EQ(result, 0);
}

// 测试启用dump回调 - 无效数据
TEST_F(DumpCallbackManagerTest, EnableDumpCallback_InvalidData_ReturnsError) {
    int32_t result = DumpCallbackManager::EnableDumpCallback(1, nullptr, 10);
    EXPECT_EQ(result, ge::ADUMP_SUCCESS);
}

// 测试禁用dump回调
TEST_F(DumpCallbackManagerTest, DisableDumpCallback_ReturnsSuccess) {
    int32_t result = DumpCallbackManager::DisableDumpCallback(0, nullptr, 0);
    EXPECT_EQ(result, 0);
}

// 测试JSON结构体解析
TEST_F(DumpConfigValidatorTest, FromJson_DumpBlacklist_ParsesCorrectly) {
    nlohmann::json json = R"({
        "name": "test_blacklist",
        "pos": ["input", "output"]
    })"_json;

    ge::DumpBlacklist blacklist;
    ge::from_json(json, blacklist);

    EXPECT_EQ(blacklist.name, "test_blacklist");
    EXPECT_EQ(blacklist.pos.size(), 2U);
    EXPECT_EQ(blacklist.pos[0], "input");
    EXPECT_EQ(blacklist.pos[1], "output");
}

// 测试ModelDumpConfig JSON解析
TEST_F(DumpConfigValidatorTest, FromJson_ModelDumpConfig_ParsesCorrectly) {
    nlohmann::json json = R"({
        "model_name": "test_model",
        "layer": ["layer1", "layer2"],
        "watcher_nodes": ["node1", "node2"],
        "optype_blacklist": [{"name": "Const", "pos": ["input"]}],
        "opname_blacklist": [{"name": "Node", "pos": ["output"]}],
        "opname_range": [{"begin": "start", "end": "end"}]
    })"_json;

    ge::ModelDumpConfig modelConfig;
    ge::from_json(json, modelConfig);

    EXPECT_EQ(modelConfig.model_name, "test_model");
    EXPECT_EQ(modelConfig.layers.size(), 2U);
    EXPECT_EQ(modelConfig.watcher_nodes.size(), 2U);
    EXPECT_EQ(modelConfig.optype_blacklist.size(), 1U);
    EXPECT_EQ(modelConfig.opname_blacklist.size(), 1U);
    EXPECT_EQ(modelConfig.dump_op_ranges.size(), 1U);
}

// 测试ContainKey辅助函数
TEST_F(DumpConfigValidatorTest, ContainKey_KeyExists_ReturnsTrue) {
    nlohmann::json json = R"({"key": "value"})"_json;
    EXPECT_TRUE(ge::ContainKey(json, "key"));
    EXPECT_FALSE(ge::ContainKey(json, "nonexistent"));
}

// 测试GetCfgStrByKey辅助函数
TEST_F(DumpConfigValidatorTest, GetCfgStrByKey_KeyExists_ReturnsValue) {
    nlohmann::json json = R"({"key": "value"})"_json;
    EXPECT_EQ(ge::GetCfgStrByKey(json, "key"), "value");
}

TEST_F(DumpConfigValidatorTest, TestLongPath) {
    // 使用 kDumpConfigWithLongPath
    DumpConfig dumpConfig;
    bool result = DumpConfigValidator::ParseDumpConfig(
        ge::kDumpConfigWithLongPath.c_str(),
        ge::kDumpConfigWithLongPath.size(),
        dumpConfig
    );
    EXPECT_FALSE(result);  // 应该失败，因为路径太长
}

TEST_F(DumpConfigValidatorTest, TestDisableDump) {
    // 使用 kDisableDumpConfig
    DumpConfig dumpConfig;
    bool result = DumpConfigValidator::ParseDumpConfig(
        ge::kDisableDumpConfig,
        strlen(ge::kDisableDumpConfig),
        dumpConfig
    );
    EXPECT_TRUE(result);
}

// 测试有效的IP地址
TEST_F(DumpConfigValidatorTest, CheckIpAddress_ValidIp_ReturnsTrue) {
    DumpConfig config;
    config.dump_path = "192.168.1.1:/tmp/dump";
    EXPECT_TRUE(DumpConfigValidator::CheckIpAddress(config));
}

// 测试无效的IP地址格式
TEST_F(DumpConfigValidatorTest, CheckIpAddress_InvalidIpFormat_ReturnsFalse) {
    DumpConfig config;
    config.dump_path = "192.168.1:/tmp/dump";  // 不完整的IP
    EXPECT_FALSE(DumpConfigValidator::CheckIpAddress(config));
}

// 测试IP地址范围超出
TEST_F(DumpConfigValidatorTest, CheckIpAddress_IpOutOfRange_ReturnsFalse) {
    DumpConfig config;
    config.dump_path = "256.168.1.1:/tmp/dump";  // IP超出范围
    EXPECT_FALSE(DumpConfigValidator::CheckIpAddress(config));
}

// 测试IP地址后无路径
TEST_F(DumpConfigValidatorTest, CheckIpAddress_NoPathAfterIp_ReturnsFalse) {
    DumpConfig config;
    config.dump_path = "192.168.1.1:";  // 冒号后无路径
    EXPECT_FALSE(DumpConfigValidator::CheckIpAddress(config));
}

// 测试DumpDebug检查 - 有效配置
TEST_F(DumpConfigValidatorTest, DumpDebugCheck_ValidConfig_ReturnsTrue) {
    nlohmann::json config = nlohmann::json::parse(R"({
        "dump": {
            "dump_path": "/tmp/debug_dump",
            "dump_debug": "on"
        }
    })");
    nlohmann::json dumpConfig = config["dump"];
    EXPECT_TRUE(DumpConfigValidator::DumpDebugCheck(dumpConfig));
}

// 测试DumpDebug检查 - 包含dump_list
TEST_F(DumpConfigValidatorTest, DumpDebugCheck_WithDumpList_ReturnsFalse) {
    nlohmann::json config = nlohmann::json::parse(R"({
        "dump": {
            "dump_path": "/tmp/debug_dump",
            "dump_debug": "on",
            "dump_list": [{"model_name": "test_model"}]
        }
    })");
    nlohmann::json dumpConfig = config["dump"];
    EXPECT_FALSE(DumpConfigValidator::DumpDebugCheck(dumpConfig));
}

// 测试DumpDebug检查 - 无dump_path
TEST_F(DumpConfigValidatorTest, DumpDebugCheck_NoDumpPath_ReturnsFalse) {
    nlohmann::json config = nlohmann::json::parse(R"({
        "dump": {
            "dump_debug": "on"
        }
    })");
    nlohmann::json dumpConfig = config["dump"];
    EXPECT_FALSE(DumpConfigValidator::DumpDebugCheck(dumpConfig));
}

// 测试DumpStats检查 - 有效配置
TEST_F(DumpConfigValidatorTest, DumpStatsCheck_ValidConfig_ReturnsTrue) {
    nlohmann::json config = nlohmann::json::parse(R"({
        "dump": {
            "dump_path": "/tmp/dump",
            "dump_data": "stats",
            "dump_stats": ["stat1", "stat2"]
        }
    })");
    nlohmann::json dumpConfig = config["dump"];
    EXPECT_TRUE(DumpConfigValidator::DumpStatsCheck(dumpConfig));
}

// 测试DumpStats检查 - 空dump_stats
TEST_F(DumpConfigValidatorTest, DumpStatsCheck_EmptyDumpStats_ReturnsFalse) {
    nlohmann::json config = nlohmann::json::parse(R"({
        "dump": {
            "dump_path": "/tmp/dump",
            "dump_stats": []
        }
    })");
    nlohmann::json dumpConfig = config["dump"];
    EXPECT_FALSE(DumpConfigValidator::DumpStatsCheck(dumpConfig));
}

// 测试DumpStats检查 - dump_data不是stats
TEST_F(DumpConfigValidatorTest, DumpStatsCheck_DumpDataNotStats_ReturnsFalse) {
    nlohmann::json config = nlohmann::json::parse(R"({
        "dump": {
            "dump_path": "/tmp/dump",
            "dump_data": "tensor",
            "dump_stats": ["stat1"]
        }
    })");
    nlohmann::json dumpConfig = config["dump"];
    EXPECT_FALSE(DumpConfigValidator::DumpStatsCheck(dumpConfig));
}

// 测试空dump_list
TEST_F(DumpConfigValidatorTest, CheckDumplist_EmptyDumpList_ReturnsTrue) {
    nlohmann::json config = nlohmann::json::parse(R"({
        "dump": {
            "dump_path": "/tmp/dump",
            "dump_list": []
        }
    })");
    nlohmann::json dumpConfig = config["dump"];
    EXPECT_TRUE(DumpConfigValidator::CheckDumplist(dumpConfig, "op"));
}

// 测试包含空对象的dump_list
TEST_F(DumpConfigValidatorTest, CheckDumplist_EmptyObjects_ReturnsTrue) {
    nlohmann::json config = nlohmann::json::parse(R"({
        "dump": {
            "dump_path": "/tmp/dump",
            "dump_list": [{}]
        }
    })");
    nlohmann::json dumpConfig = config["dump"];
    EXPECT_TRUE(DumpConfigValidator::CheckDumplist(dumpConfig, "op"));
}

// 测试操作名范围验证 - 无效dump_level
TEST_F(DumpConfigValidatorTest, CheckDumpOpNameRange_InvalidDumpLevel_ReturnsFalse) {
    std::vector<ModelDumpConfig> dumpList;
    ModelDumpConfig modelConfig;
    modelConfig.model_name = "test_model";
    modelConfig.dump_op_ranges.emplace_back("start", "end");
    dumpList.push_back(modelConfig);

    EXPECT_FALSE(DumpConfigValidator::CheckDumpOpNameRange(dumpList, "kernel"));
}

// 测试操作名范围验证 - 空模型名
TEST_F(DumpConfigValidatorTest, CheckDumpOpNameRange_EmptyModelName_ReturnsFalse) {
    std::vector<ModelDumpConfig> dumpList;
    ModelDumpConfig modelConfig;
    modelConfig.dump_op_ranges.emplace_back("start", "end");
    dumpList.push_back(modelConfig);

    EXPECT_FALSE(DumpConfigValidator::CheckDumpOpNameRange(dumpList, "op"));
}

// 测试操作名范围验证 - 不完整的范围
TEST_F(DumpConfigValidatorTest, CheckDumpOpNameRange_IncompleteRange_ReturnsFalse) {
    std::vector<ModelDumpConfig> dumpList;
    ModelDumpConfig modelConfig;
    modelConfig.model_name = "test_model";
    modelConfig.dump_op_ranges.emplace_back("", "end");  // 空的begin
    dumpList.push_back(modelConfig);

    EXPECT_FALSE(DumpConfigValidator::CheckDumpOpNameRange(dumpList, "op"));
}

// 测试JSON解析异常
TEST_F(DumpConfigValidatorTest, ParseDumpConfig_JsonException_ReturnsFalse) {
    // 创建一个会导致JSON解析异常的字符串
    const char* malformedJson = "{\"dump\": { \"dump_path\": \"/tmp/test\" }";  // 缺少闭合括号
    DumpConfig config;
    bool result = DumpConfigValidator::ParseDumpConfig(
        malformedJson, strlen(malformedJson), config);
    EXPECT_FALSE(result);
}

// 测试包含IP地址的路径验证
TEST_F(DumpConfigValidatorTest, CheckDumpPath_WithValidIp_ReturnsTrue) {
    nlohmann::json config = nlohmann::json::parse(R"({
        "dump": {
            "dump_path": "192.168.1.1:/tmp/valid_path"
        }
    })");
    nlohmann::json dumpConfig = config["dump"];
    EXPECT_TRUE(DumpConfigValidator::CheckDumpPath(dumpConfig));
}

// 测试包含无效字符的IP路径
TEST_F(DumpConfigValidatorTest, CheckDumpPath_WithInvalidIpPath_ReturnsFalse) {
    nlohmann::json config = nlohmann::json::parse(R"({
        "dump": {
            "dump_path": "192.168.1.1:/tmp/invalid<path"
        }
    })");
    nlohmann::json dumpConfig = config["dump"];
    EXPECT_FALSE(DumpConfigValidator::CheckDumpPath(dumpConfig));
}

// 测试最大dump_step数量
TEST_F(DumpConfigValidatorTest, CheckDumpStep_MaxSteps_ReturnsTrue) {
    nlohmann::json config = nlohmann::json::parse(R"({
        "dump": {
            "dump_path": "/tmp/dump",
            "dump_step": "1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|18|19|20|21|22|23|24|25|26|27|28|29|30|31|32|33|34|35|36|37|38|39|40|41|42|43|44|45|46|47|48|49|50|51|52|53|54|55|56|57|58|59|60|61|62|63|64|65|66|67|68|69|70|71|72|73|74|75|76|77|78|79|80|81|82|83|84|85|86|87|88|89|90|91|92|93|94|95|96|97|98|99|100"
        }
    })");
    nlohmann::json dumpConfig = config["dump"];
    EXPECT_TRUE(DumpConfigValidator::CheckDumpStep(dumpConfig));
}

// 测试超过最大dump_step数量
TEST_F(DumpConfigValidatorTest, CheckDumpStep_ExceedMaxSteps_ReturnsFalse) {
    nlohmann::json config = nlohmann::json::parse(R"({
        "dump": {
            "dump_path": "/tmp/dump",
            "dump_step": "1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|18|19|20|21|22|23|24|25|26|27|28|29|30|31|32|33|34|35|36|37|38|39|40|41|42|43|44|45|46|47|48|49|50|51|52|53|54|55|56|57|58|59|60|61|62|63|64|65|66|67|68|69|70|71|72|73|74|75|76|77|78|79|80|81|82|83|84|85|86|87|88|89|90|91|92|93|94|95|96|97|98|99|100|101"
        }
    })");
    nlohmann::json dumpConfig = config["dump"];
    EXPECT_FALSE(DumpConfigValidator::CheckDumpStep(dumpConfig));
}

// 1. 异常dump场景基本测试
TEST_F(DumpConfigValidatorTest, ValidateExceptionDumpConfig_ValidExceptionDump_ReturnsTrue) {
    // 测试lite_exception
    nlohmann::json config1 = nlohmann::json::parse(R"({
        "dump": {
            "dump_scene": "lite_exception"
        }
    })");
    EXPECT_TRUE(DumpConfigValidator::IsValidDumpConfig(config1));

    // 测试aic_err_brief_dump
    nlohmann::json config2 = nlohmann::json::parse(R"({
        "dump": {
            "dump_scene": "aic_err_brief_dump"
        }
    })");
    EXPECT_TRUE(DumpConfigValidator::IsValidDumpConfig(config2));
}

TEST_F(DumpConfigValidatorTest, ValidateExceptionDumpConfig_InvalidExceptionType_ReturnsFalse) {
    nlohmann::json config = nlohmann::json::parse(R"({
        "dump": {
            "dump_scene": "invalid_exception_type"
        }
    })");
    EXPECT_FALSE(DumpConfigValidator::IsValidDumpConfig(config));
}

TEST_F(DumpConfigValidatorTest, ValidateExceptionDumpConfig_BothExceptionAndDebug_ReturnsFalse) {
    nlohmann::json config = nlohmann::json::parse(R"({
        "dump": {
            "dump_scene": "lite_exception",
            "dump_debug": "on"
        }
    })");
    EXPECT_FALSE(DumpConfigValidator::IsValidDumpConfig(config));
}

TEST_F(DumpConfigValidatorTest, ParseBasicConfigurations_WithDumpScene_ParsesCorrectly) {
    nlohmann::json config = nlohmann::json::parse(R"({
        "dump_path": "/tmp/dump",
        "dump_scene": "lite_exception"
    })");

    DumpConfig dumpCfg;
    DumpConfigValidator::ParseBasicConfigurations(config, dumpCfg);

    EXPECT_EQ(dumpCfg.dump_exception, "lite_exception");
    EXPECT_EQ(dumpCfg.dump_path, "/tmp/dump");
}

TEST_F(DumpConfigValidatorTest, ParseBasicConfigurations_WithoutDumpScene_DefaultsToEmpty) {
    nlohmann::json config = nlohmann::json::parse(R"({
        "dump_path": "/tmp/dump"
    })");

    DumpConfig dumpCfg;
    DumpConfigValidator::ParseBasicConfigurations(config, dumpCfg);

    EXPECT_TRUE(dumpCfg.dump_exception.empty());
}

TEST_F(DumpCallbackManagerTest, HandleEnableDump_ExceptionDumpConfig_Success) {
    std::string exceptionConfig = R"({
        "dump": {
            "dump_path": "/tmp/exception",
            "dump_scene": "lite_exception"
        }
    })";

    int32_t result = DumpCallbackManager::EnableDumpCallback(
        1, const_cast<char*>(exceptionConfig.c_str()), exceptionConfig.size());

    EXPECT_EQ(result, 0);
}

TEST_F(DumpCallbackManagerTest, HandleEnableDump_DebugDumpConfig_Success) {
    std::string debugConfig = R"({
        "dump": {
            "dump_path": "/tmp/debug",
            "dump_debug": "on",
            "dump_step": "1-10"
        }
    })";

    int32_t result = DumpCallbackManager::EnableDumpCallback(
        1, const_cast<char*>(debugConfig.c_str()), debugConfig.size());

    EXPECT_EQ(result, 0);
}

TEST_F(DumpCallbackManagerTest, HandleEnableDump_NormalDumpConfig_Success) {
    std::string normalConfig = R"({
        "dump": {
            "dump_path": "/tmp/normal",
            "dump_mode": "output",
            "dump_level": "all"
        }
    })";

    int32_t result = DumpCallbackManager::EnableDumpCallback(
        1, const_cast<char*>(normalConfig.c_str()), normalConfig.size());

    EXPECT_EQ(result, 0);
}

TEST_F(DumpCallbackManagerTest, HandleEnableDump_ExceptionPriorityOverNormal) {
    // 即使有normal dump配置，只要指定了exception scene，就按exception处理
    std::string mixedConfig = R"({
        "dump": {
            "dump_path": "/tmp/mixed",
            "dump_scene": "aic_err_norm_dump",
            "dump_mode": "output"
        }
    })";

    int32_t result = DumpCallbackManager::EnableDumpCallback(
        1, const_cast<char*>(mixedConfig.c_str()), mixedConfig.size());

    EXPECT_EQ(result, 0);
}

TEST_F(DumpConfigValidatorTest, ValidateExceptionDumpConfig_EmptyScene_ReturnsFalse) {
    nlohmann::json config = nlohmann::json::parse(R"({
        "dump": {
            "dump_path": "/tmp/dump",
            "dump_scene": ""
        }
    })");
    EXPECT_FALSE(DumpConfigValidator::IsValidDumpConfig(config));
}

TEST_F(DumpConfigValidatorTest, IsDumpDebugEnabled_ValidDebugConfig_ReturnsTrue) {
    nlohmann::json config = nlohmann::json::parse(R"({
        "dump": {
            "dump_path": "/tmp/debug",
            "dump_debug": "on"
        }
    })");

    nlohmann::json dumpConfig = config["dump"];
    EXPECT_TRUE(DumpConfigValidator::IsDumpDebugEnabled(dumpConfig));
}

TEST_F(DumpConfigValidatorTest, DumpDebugCheck_ValidDebugConfig_ReturnsTrue) {
    nlohmann::json config = nlohmann::json::parse(R"({
        "dump": {
            "dump_path": "/tmp/debug",
            "dump_debug": "on"
        }
    })");

    nlohmann::json dumpConfig = config["dump"];
    EXPECT_TRUE(DumpConfigValidator::DumpDebugCheck(dumpConfig));
}

TEST_F(DumpConfigValidatorTest, ParseDumpConfig_ExceptionDump_ParsesCorrectly) {
    std::string configStr = R"({
        "dump": {
            "dump_path": "/tmp/exception",
            "dump_scene": "aic_err_detail_dump"
        }
    })";

    DumpConfig config;
    bool result = DumpConfigValidator::ParseDumpConfig(
        configStr.c_str(), configStr.size(), config);

    EXPECT_TRUE(result);
    EXPECT_EQ(config.dump_exception, "aic_err_detail_dump");
}

TEST_F(DumpConfigValidatorTest, GetConfigWithDefault_DumpSceneHandling) {
    // 存在dump_scene
    nlohmann::json config1 = nlohmann::json::parse(R"({
        "dump_scene": "lite_exception"
    })");
    std::string value1 = DumpConfigValidator::GetConfigWithDefault(config1, "dump_scene", "");
    EXPECT_EQ(value1, "lite_exception");

    // 不存在dump_scene
    nlohmann::json config2 = nlohmann::json::parse(R"({})");
    std::string value2 = DumpConfigValidator::GetConfigWithDefault(config2, "dump_scene", "");
    EXPECT_TRUE(value2.empty());
}

TEST_F(DumpConfigValidatorTest, ParseDumpConfig_SetDumpStatus_BasedOnDumpLevel) {
    // 测试场景1: dump_level = "op" -> dump_status = "on"
    {
        std::string configStr = R"({
            "dump": {
                "dump_path": "/tmp/test",
                "dump_level": "op"
            }
        })";

        DumpConfig config;
        bool result = DumpConfigValidator::ParseDumpConfig(
            configStr.c_str(), configStr.size(), config);

        EXPECT_TRUE(result);
        EXPECT_EQ(config.dump_level, "op");
        EXPECT_EQ(config.dump_status, "on");
    }

    // 测试场景2: dump_level = "all" -> dump_status = "on"
    {
        std::string configStr = R"({
            "dump": {
                "dump_path": "/tmp/test",
                "dump_level": "all"
            }
        })";

        DumpConfig config;
        bool result = DumpConfigValidator::ParseDumpConfig(
            configStr.c_str(), configStr.size(), config);

        EXPECT_TRUE(result);
        EXPECT_EQ(config.dump_level, "all");
        EXPECT_EQ(config.dump_status, "on");
    }

    // 测试场景3: dump_level = "kernel" -> dump_status = "off"
    {
        std::string configStr = R"({
            "dump": {
                "dump_path": "/tmp/test",
                "dump_level": "kernel"
            }
        })";

        DumpConfig config;
        bool result = DumpConfigValidator::ParseDumpConfig(
            configStr.c_str(), configStr.size(), config);

        EXPECT_TRUE(result);
        EXPECT_EQ(config.dump_level, "kernel");
        EXPECT_EQ(config.dump_status, "off");
    }
}

TEST_F(DumpConfigValidatorTest, ParseModelDumpConfig_EmptyModelName_ReturnsFalse) {
    nlohmann::json modelJson = R"({
        "model_name": "",
        "layer": ["layer1", "layer2"]
    })"_json;
    
    ge::ModelDumpConfig modelConfig;
    bool result = DumpConfigValidator::ParseModelDumpConfig(modelJson, modelConfig);
    
    EXPECT_FALSE(result);
}

TEST_F(DumpConfigValidatorTest, ParseModelDumpConfig_EmptyLayerArray_ReturnsFalse) {
    nlohmann::json modelJson = R"({
        "model_name": "test_model",
        "layer": []
    })"_json;
    
    ge::ModelDumpConfig modelConfig;
    bool result = DumpConfigValidator::ParseModelDumpConfig(modelJson, modelConfig);
    
    EXPECT_FALSE(result);
}

TEST_F(DumpConfigValidatorTest, IsEnableExceptionDumpBySwitch_ValidBits_ReturnsTrue) {
    EXPECT_TRUE(DumpCallbackManager::IsEnableExceptionDumpBySwitch(0x4));

    EXPECT_TRUE(DumpCallbackManager::IsEnableExceptionDumpBySwitch(0x8));

    EXPECT_TRUE(DumpCallbackManager::IsEnableExceptionDumpBySwitch(0xC));

    EXPECT_TRUE(DumpCallbackManager::IsEnableExceptionDumpBySwitch(0xF));
    EXPECT_TRUE(DumpCallbackManager::IsEnableExceptionDumpBySwitch(0x14));
}

TEST_F(DumpConfigValidatorTest, IsEnableExceptionDumpBySwitch_NoExceptionBits_ReturnsFalse) {
    EXPECT_FALSE(DumpCallbackManager::IsEnableExceptionDumpBySwitch(0x0));
    EXPECT_FALSE(DumpCallbackManager::IsEnableExceptionDumpBySwitch(0x1));
    EXPECT_FALSE(DumpCallbackManager::IsEnableExceptionDumpBySwitch(0x2));
    EXPECT_FALSE(DumpCallbackManager::IsEnableExceptionDumpBySwitch(0x10));
    EXPECT_FALSE(DumpCallbackManager::IsEnableExceptionDumpBySwitch(0x100));
}

TEST_F(DumpConfigValidatorTest, BuildExceptionDumpJsonBySwitch_NormDumpBit_ReturnsCorrectJson) {
    std::string jsonStr = DumpCallbackManager::BuildExceptionDumpJsonBySwitch(0x8);

    nlohmann::json js = nlohmann::json::parse(jsonStr);
    EXPECT_TRUE(js.contains(ge::GE_DUMP));
    EXPECT_TRUE(js[ge::GE_DUMP].contains(ge::GE_DUMP_SCENE));
    EXPECT_EQ(js[ge::GE_DUMP][ge::GE_DUMP_SCENE].get<std::string>(), "aic_err_norm_dump");
    EXPECT_TRUE(js[ge::GE_DUMP].contains(ge::GE_DUMP_PATH));
    EXPECT_TRUE(js[ge::GE_DUMP][ge::GE_DUMP_PATH].get<std::string>().empty());
}

TEST_F(DumpConfigValidatorTest, BuildExceptionDumpJsonBySwitch_BriefDumpBit_ReturnsCorrectJson) {
    std::string jsonStr = DumpCallbackManager::BuildExceptionDumpJsonBySwitch(0x4);
    
    nlohmann::json js = nlohmann::json::parse(jsonStr);
    EXPECT_TRUE(js.contains(ge::GE_DUMP));
    EXPECT_TRUE(js[ge::GE_DUMP].contains(ge::GE_DUMP_SCENE));
    EXPECT_EQ(js[ge::GE_DUMP][ge::GE_DUMP_SCENE].get<std::string>(), "aic_err_brief_dump");
    EXPECT_TRUE(js[ge::GE_DUMP].contains(ge::GE_DUMP_PATH));
    EXPECT_TRUE(js[ge::GE_DUMP][ge::GE_DUMP_PATH].get<std::string>().empty());
}

TEST_F(DumpConfigValidatorTest, BuildExceptionDumpJsonBySwitch_BothBits_NormDumpTakesPriority) {
    std::string jsonStr = DumpCallbackManager::BuildExceptionDumpJsonBySwitch(0xC);
    
    nlohmann::json js = nlohmann::json::parse(jsonStr);
    EXPECT_EQ(js[ge::GE_DUMP][ge::GE_DUMP_SCENE].get<std::string>(), "aic_err_norm_dump");
}

TEST_F(DumpConfigValidatorTest, BuildExceptionDumpJsonBySwitch_NoExceptionBits_ReturnsEmpty) {
    std::string jsonStr = DumpCallbackManager::BuildExceptionDumpJsonBySwitch(0x0);
    EXPECT_TRUE(jsonStr.empty());
    
    jsonStr = DumpCallbackManager::BuildExceptionDumpJsonBySwitch(0x1);
    EXPECT_TRUE(jsonStr.empty());
}

TEST_F(DumpConfigValidatorTest, ProcessExceptionDumpBySwitch_ValidSwitch_ReturnsTrue) {
    EXPECT_TRUE(DumpCallbackManager::ProcessExceptionDumpBySwitch(0x4));

    EXPECT_TRUE(DumpCallbackManager::ProcessExceptionDumpBySwitch(0x8));

    EXPECT_TRUE(DumpCallbackManager::ProcessExceptionDumpBySwitch(0xC));
}

TEST_F(DumpConfigValidatorTest, ProcessExceptionDumpBySwitch_InvalidSwitch_ReturnsFalse) {
    EXPECT_FALSE(DumpCallbackManager::ProcessExceptionDumpBySwitch(0x0));
    EXPECT_FALSE(DumpCallbackManager::ProcessExceptionDumpBySwitch(0x1));
    EXPECT_FALSE(DumpCallbackManager::ProcessExceptionDumpBySwitch(0x2));
    EXPECT_FALSE(DumpCallbackManager::ProcessExceptionDumpBySwitch(0x10));
}

TEST_F(DumpCallbackManagerTest, EnableDumpCallback_EmptyDataWithNormDumpBit_ReturnsSuccess) {
    uint64_t dumpSwitch = 0x8;
    
    int32_t result = DumpCallbackManager::EnableDumpCallback(dumpSwitch, nullptr, 0);
    EXPECT_EQ(result, ge::ADUMP_SUCCESS);
}

TEST_F(DumpCallbackManagerTest, EnableDumpCallback_EmptyDataWithBriefDumpBit_ReturnsSuccess) {
    uint64_t dumpSwitch = 0x4;
    
    int32_t result = DumpCallbackManager::EnableDumpCallback(dumpSwitch, nullptr, 0);
    EXPECT_EQ(result, ge::ADUMP_SUCCESS);
}

TEST_F(DumpCallbackManagerTest, EnableDumpCallback_EmptyDataWithBothExceptionBits_ReturnsSuccess) {
    uint64_t dumpSwitch = 0xC;
    
    int32_t result = DumpCallbackManager::EnableDumpCallback(dumpSwitch, nullptr, 0);
    EXPECT_EQ(result, ge::ADUMP_SUCCESS);
}

TEST_F(DumpCallbackManagerTest, EnableDumpCallback_EmptyDataWithNoExceptionBits_ReturnsSuccess) {
    uint64_t dumpSwitch = 0x0;
    
    int32_t result = DumpCallbackManager::EnableDumpCallback(dumpSwitch, nullptr, 0);
    EXPECT_EQ(result, ge::ADUMP_SUCCESS);
}

TEST_F(DumpCallbackManagerTest, EnableDumpCallback_EmptyDataWithOtherBits_ReturnsSuccess) {
    uint64_t dumpSwitch = 0x1;
    
    int32_t result = DumpCallbackManager::EnableDumpCallback(dumpSwitch, nullptr, 0);
    EXPECT_EQ(result, ge::ADUMP_SUCCESS);
}

TEST_F(DumpCallbackManagerTest, EnableDumpCallback_ValidDataWithExceptionBits_IgnoresSwitchBits) {
    uint64_t dumpSwitch = 0xC; 
    
    std::string validConfig = R"({
        "dump": {
            "dump_path": "/tmp/test",
            "dump_mode": "output",
            "dump_level": "op"
        }
    })";
    
    int32_t result = DumpCallbackManager::EnableDumpCallback(
        dumpSwitch, const_cast<char*>(validConfig.c_str()), validConfig.size());

    EXPECT_EQ(result, ge::ADUMP_SUCCESS);
}

TEST_F(DumpCallbackManagerTest, EnableDumpCallback_EmptyDataWithLargeSize_ReturnsSuccess) {
    uint64_t dumpSwitch = 0x8;
    
    int32_t result = DumpCallbackManager::EnableDumpCallback(dumpSwitch, nullptr, 10);
    EXPECT_EQ(result, ge::ADUMP_SUCCESS);
}

TEST_F(DumpCallbackManagerTest, EnableDumpCallback_InvalidDataWithExceptionBits_ReturnsSuccess) {
    uint64_t dumpSwitch = 0x8;
    
    const char* dummyData = "";
    int32_t result = DumpCallbackManager::EnableDumpCallback(dumpSwitch, dummyData, 0);
    EXPECT_EQ(result, ge::ADUMP_SUCCESS);
}

TEST_F(DumpCallbackManagerTest, EnableDumpCallback_MixedSwitchBits_ReturnsSuccess) {
    uint64_t dumpSwitch = 0x9;
    
    int32_t result = DumpCallbackManager::EnableDumpCallback(dumpSwitch, nullptr, 0);
    EXPECT_EQ(result, ge::ADUMP_SUCCESS);
    
    dumpSwitch = 0x14;
    result = DumpCallbackManager::EnableDumpCallback(dumpSwitch, nullptr, 0);
    EXPECT_EQ(result, ge::ADUMP_SUCCESS);
}

TEST_F(DumpConfigValidatorTest, BuiltJsonConfig_CanBeParsedCorrectly) {
    uint64_t dumpSwitch = 0x8; 
    
    std::string jsonStr = DumpCallbackManager::BuildExceptionDumpJsonBySwitch(dumpSwitch);
    ASSERT_FALSE(jsonStr.empty());

    DumpConfig config;
    bool result = DumpConfigValidator::ParseDumpConfig(
        jsonStr.c_str(), jsonStr.size(), config);
    
    EXPECT_TRUE(result);
    EXPECT_EQ(config.dump_exception, "aic_err_norm_dump");
    EXPECT_TRUE(config.dump_path.empty());
}

TEST_F(DumpConfigValidatorTest, BuiltJsonConfig_ContainsRequiredFields) {

    std::string jsonStr = DumpCallbackManager::BuildExceptionDumpJsonBySwitch(0x4);
    nlohmann::json js = nlohmann::json::parse(jsonStr);
    
    EXPECT_TRUE(js.is_object());
    EXPECT_TRUE(js.contains(ge::GE_DUMP));
    EXPECT_TRUE(js[ge::GE_DUMP].is_object());
    EXPECT_TRUE(js[ge::GE_DUMP].contains(ge::GE_DUMP_SCENE));
    EXPECT_TRUE(js[ge::GE_DUMP].contains(ge::GE_DUMP_PATH));

    EXPECT_FALSE(js[ge::GE_DUMP].contains(ge::GE_DUMP_MODE));
    EXPECT_FALSE(js[ge::GE_DUMP].contains(ge::GE_DUMP_LEVEL));
    EXPECT_FALSE(js[ge::GE_DUMP].contains(ge::GE_DUMP_LIST));
}

// 主函数
int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}