/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_HETEROGENEOUS_FLOWRM_RANK_TABLE_BUILDER
#define AIR_RUNTIME_HETEROGENEOUS_FLOWRM_RANK_TABLE_BUILDER

#include <map>
#include <string>
#include "nlohmann/json.hpp"
#include "ge/ge_api_error_codes.h"
#include "framework/common/debug/log.h"

namespace ge {
struct HcomRank {
  std::string device_id;
  std::string port;
  std::string rank_id;
  DeviceType device_type = NPU;
  int32_t hcom_device_id;
};

struct HcomNode {
  std::string node_addr;
  std::vector<HcomRank> node_ranks;
};

struct HcomRankTable {
  std::string collective_id;
  std::vector<HcomNode> node_list;
  std::string status;
  std::string version;
};

void to_json(nlohmann::json &j, const HcomRank &rank);
void to_json(nlohmann::json &j, const HcomNode &node);
void to_json(nlohmann::json &j, const HcomRankTable &rank_table);

void from_json(const nlohmann::json &j, HcomRank &rank);
void from_json(const nlohmann::json &j, HcomNode &node);
void from_json(const nlohmann::json &j, HcomRankTable &rank_table);

class RankTableBuilder {
 public:
  void SetRankTable(const HcomRankTable &hcom_rank_table);
  bool GetRankTable(std::string &rank_table);
  bool GetRankIdByDeviceId(const std::string &ip, int32_t hcom_device_id, int32_t &rank_id);

 private:
  bool InitRanksMap();
  bool InitRankId();
  void FilterRankTableByDeviceId(const HcomRankTable &hcom_rank_table);

 private:
  HcomRankTable hcom_rank_table_;
  HcomRankTable filtered_rank_table_;
  std::map<std::string, HcomRank *> ip_rank_map_;
};
}  // namespace ge
#endif  // INC_RANK_TABLE_BUILDER
