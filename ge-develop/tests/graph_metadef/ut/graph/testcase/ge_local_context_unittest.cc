/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "graph/ge_local_context.h"

namespace ge {
class UtestGeLocalContext : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestGeLocalContext, GetAllGraphOptionsTest) {
    GEThreadLocalContext ge_local_context;
    std::map<std::string, std::string> graph_maps;
    std::string key1 = "333";
    std::string value1 = "cccc";
    std::string key2 = "444";
    std::string value2 = "ddd";
    graph_maps.insert(std::make_pair(key1, value1));
    graph_maps.insert(std::make_pair(key2, value2));
    ge_local_context.SetGraphOption(graph_maps);

    std::map<std::string, std::string> graph_options_;
    graph_options_ = ge_local_context.GetAllGraphOptions();
    std::string ret_value1 = graph_options_[key1];
    EXPECT_EQ(ret_value1, "cccc");
    std::string ret_value2 = graph_options_[key2];
    EXPECT_EQ(ret_value2, "ddd");
}

TEST_F(UtestGeLocalContext, GetAllOptionsTest) {
    GEThreadLocalContext ge_local_context;
    std::map<std::string, std::string> global_maps;
    std::string global_key1 = "111";
    std::string global_value1 = "aaa";
    std::string global_key2 = "222";
    std::string global_value2 = "bbb";
    global_maps.insert(std::make_pair(global_key1, global_value1));
    global_maps.insert(std::make_pair(global_key2, global_value2));
    ge_local_context.SetGlobalOption(global_maps);

    std::map<std::string, std::string> session_maps;
    std::string session_key1 = "333";
    std::string session_value1 = "ccc";
    std::string session_key2 = "444";
    std::string session_value2 = "ddd";
    session_maps.insert(std::make_pair(session_key1, session_value1));
    session_maps.insert(std::make_pair(session_key2, session_value2));
    ge_local_context.SetSessionOption(session_maps);

    std::map<std::string, std::string> graph_maps;
    std::string graph_key1 = "555";
    std::string graph_value1 = "eee";
    std::string graph_key2 = "666";
    std::string graph_value2 = "fff";
    graph_maps.insert(std::make_pair(graph_key1, graph_value1));
    graph_maps.insert(std::make_pair(graph_key2, graph_value2));
    ge_local_context.SetGraphOption(graph_maps);

    std::map<std::string, std::string> options_all;
    options_all = ge_local_context.GetAllOptions();
    std::string ret_value1 = options_all["222"];
    EXPECT_EQ(ret_value1, "bbb");
    std::string ret_value2 = options_all["444"];
    EXPECT_EQ(ret_value2, "ddd");
    std::string ret_value3 = options_all["555"];
    EXPECT_EQ(ret_value3, "eee");
}

TEST_F(UtestGeLocalContext, GetGraphOptionSuccess) {
    GEThreadLocalContext ge_local_context;
    std::map<std::string, std::string> graph_maps;
    std::string graph_key1 = "test1";
    std::string graph_value1 = "node1";
    std::string graph_key2 = "test2";
    std::string graph_value2 = "node2";
    graph_maps.insert(std::make_pair(graph_key1, graph_value1));
    graph_maps.insert(std::make_pair(graph_key2, graph_value2));
    ge_local_context.SetGraphOption(graph_maps);

    std::string find_key = "test1";
    std::string option;
    int ret =  ge_local_context.GetOption(find_key, option);
    EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestGeLocalContext, GetGlobalOptionSuccess) {
    GEThreadLocalContext ge_local_context;
    std::map<std::string, std::string> graph_maps;
    std::string global_key1 = "global1";
    std::string global_value1 = "node1";
    std::string global_key2 = "global2";
    std::string global_value2 = "node2";
    graph_maps.insert(std::make_pair(global_key1, global_value1));
    graph_maps.insert(std::make_pair(global_key2, global_value2));
    ge_local_context.SetGlobalOption(graph_maps);

    std::string find_key = "global1";
    std::string option;
    int ret =  ge_local_context.GetOption(find_key, option);
    EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestGeLocalContext, StreamSyncTimeoutIsInvalid) {
  std::map<std::string, std::string> global_options;
  global_options.insert(std::make_pair("stream_sync_timeout", "aaaaaaaaaa"));
  GetThreadLocalContext().SetGlobalOption(global_options);
  EXPECT_EQ(GetThreadLocalContext().StreamSyncTimeout(), -1);
}

TEST_F(UtestGeLocalContext, GetReableNameSuccess) {
  GetThreadLocalContext() = GEThreadLocalContext();
  std::map<std::string, std::string> global_options;
  std::string ir_option = "ge.Test";
  std::string show_name = "--test";
  std::string json = "{\"" + ir_option + "\": \"" + show_name +"\"\n}";
  global_options["ge.optionNameMap"] = json;
  GetThreadLocalContext().SetGlobalOption(global_options);
  auto readable_name = GetThreadLocalContext().GetReadableName(ir_option);
  EXPECT_EQ(readable_name, show_name);
}
} // namespace ge
