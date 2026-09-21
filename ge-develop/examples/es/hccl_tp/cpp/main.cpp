/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "src/es_showcase.h"
#include <iostream>
#include <map>
#include <functional>
#include "ge/ge_api.h"
#include "graph/graph.h"
#include "graph/tensor.h"
#include "utils.h"

int main(int argc, char **argv) {
  if (argc < 2) {
    es_showcase::MakePfaHcomGraphByEsAndDump();
    return 0;
  }

  std::string command = argv[1];

  if (command == "dump") {
    es_showcase::MakePfaHcomGraphByEsAndDump();
    return 0;
  } else if (command == "run") {
    // 先读取必需的 RANK_ID
    const char* rank_id_env = std::getenv("RANK_ID");
    if (rank_id_env == nullptr) {
      std::cerr << "[Error] RANK_ID environment variable not set\n";
      return -1;
    }
    ge::AscendString rank_id = rank_id_env;

    // 读取 DEVICE_ID，如果不存在则使用 RANK_ID
    const char* device_id_env = std::getenv("DEVICE_ID");
    if (device_id_env == nullptr) {
      device_id_env = rank_id_env;  
    }
    ge::AscendString device_id = device_id_env;

    ge::AscendString rank_table_file = std::getenv("RANK_TABLE_FILE");

    std::map<ge::AscendString, ge::AscendString> config = {
      {"ge.exec.deviceId", device_id},
      {"ge.graphRunMode", "0"},
      {"ge.exec.rankTableFile", rank_table_file},
      {"ge.exec.rankId", rank_id}
    };
    auto ret = ge::GEInitialize(config);
    if (ret != ge::SUCCESS) {
      std::cerr << "GE 初始化失败\n";
      return -1;
    }
    int result = -1;
    if (es_showcase::MakePfaHcomGraphByEsAndRun() == 0) {
      result = 0;
    }
    ge::GEFinalize();
    std::cout << "执行结束" << std::endl;
    return result;
  } else {
    std::cout << "错误: 未知命令 '" << command << "'" << std::endl;
    return -1;
  }
}
