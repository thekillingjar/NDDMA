/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rank_table_builder.h"
namespace ge {
void to_json(nlohmann::json &j, const HcomRank &rank) {
  j = nlohmann::json{};
  if (!rank.device_id.empty() && rank.device_type == NPU) {
    j["device_id"] = rank.device_id;
  }
  if (!rank.rank_id.empty()) {
    j["rank_id"] = rank.rank_id;
  }
  if (!rank.port.empty()) {
    j["port"] = rank.port;
  }
}

void to_json(nlohmann::json &j, const HcomNode &node) {
  j = nlohmann::json{};
  if (!node.node_addr.empty()) {
    j["node_addr"] = node.node_addr;
  }
  j["ranks"] = node.node_ranks;
}

void to_json(nlohmann::json &j, const HcomRankTable &rank_table) {
  j = nlohmann::json{};
  if (!rank_table.collective_id.empty()) {
    j["collective_id"] = rank_table.collective_id;
  }
  j["node_list"] = rank_table.node_list;
  if (!rank_table.status.empty()) {
    j["status"] = rank_table.status;
  }
  if (!rank_table.version.empty()) {
    j["version"] = rank_table.version;
  }
}

void from_json(const nlohmann::json &j, HcomRank &rank) {
  if (j.find("device_id") != j.end()) {
    j.at("device_id").get_to(rank.device_id);
  }
  if (j.find("rank_id") != j.end()) {
    j.at("rank_id").get_to(rank.rank_id);
  }
  if (j.find("port") != j.end()) {
    j.at("port").get_to(rank.port);
  }
}

void from_json(const nlohmann::json &j, HcomNode &node) {
  if (j.find("node_addr") != j.end()) {
    j.at("node_addr").get_to(node.node_addr);
  }
  if (j.find("ranks") != j.end()) {
    j.at("ranks").get_to(node.node_ranks);
  }
}

void from_json(const nlohmann::json &j, HcomRankTable &rank_table) {
  if (j.find("collective_id") != j.end()) {
    j.at("collective_id").get_to(rank_table.collective_id);
  }
  if (j.find("node_list") != j.end()) {
    j.at("node_list").get_to(rank_table.node_list);
  }
  if (j.find("status") != j.end()) {
    j.at("status").get_to(rank_table.status);
  }
  if (j.find("version") != j.end()) {
    j.at("version").get_to(rank_table.version);
  }
}

void RankTableBuilder::FilterRankTableByDeviceId(const HcomRankTable &hcom_rank_table) {
  hcom_rank_table_.node_list.clear();
  hcom_rank_table_.collective_id = hcom_rank_table.collective_id;
  hcom_rank_table_.status = hcom_rank_table.status;
  hcom_rank_table_.version = hcom_rank_table.version;
  for (const auto &node : hcom_rank_table.node_list) {
    HcomNode filtered_node = {};
    filtered_node.node_addr = node.node_addr;
    std::set<std::string> device_set;
    for (const auto &rank : node.node_ranks) {
      auto device_info = node.node_addr + "_" + std::to_string(rank.hcom_device_id);
      const auto &it = device_set.find(device_info);
      if (it != device_set.cend()) {
        continue;
      }
      filtered_node.node_ranks.emplace_back(rank);
      device_set.emplace(device_info);
    }
    hcom_rank_table_.node_list.emplace_back(filtered_node);
  }
}

void RankTableBuilder::SetRankTable(const HcomRankTable &hcom_rank_table) {
  FilterRankTableByDeviceId(hcom_rank_table);
  ip_rank_map_.clear();
}

bool RankTableBuilder::GetRankTable(std::string &rank_table) {
  if (hcom_rank_table_.collective_id.empty()) {
    GELOGE(FAILED, "collective_id is empty");
    return false;
  }
  if (ip_rank_map_.empty()) {
    if (!InitRanksMap()) {
      GELOGE(FAILED, "get ranks map failed");
      return false;
    }
  }
  nlohmann::json j = hcom_rank_table_;
  try {
    rank_table = j.dump();
  } catch (const nlohmann::json::exception &e) {
    GELOGE(FAILED, "GetRankTable failed, exception message: %s", e.what());
    return false;
  }
  return true;
}

bool RankTableBuilder::GetRankIdByDeviceId(const std::string &ip, int32_t hcom_device_id, int32_t &rank_id) {
  if (ip_rank_map_.empty()) {
    if (!InitRanksMap()) {
      GELOGW("get ranks map failed");
      return false;
    }
  }

  const std::string key = ip + ":" + std::to_string(hcom_device_id);
  const HcomRank *const rank_ptr = ip_rank_map_[key];
  if (rank_ptr == nullptr) {
    GELOGW("rank_ptr is nullptr, key is %s", key.c_str());
    return false;
  }

  try {
    rank_id = std::stoi(rank_ptr->rank_id);
  } catch (...) {
    GELOGW("rank_id %s illegal.", rank_ptr->rank_id.c_str());
    return false;
  }
  return true;
}

bool RankTableBuilder::InitRanksMap() {
  if (hcom_rank_table_.node_list.empty()) {
    GELOGE(FAILED, "hcomRankTable_ nodeList is empty");
    return false;
  }
  for (HcomNode &node : hcom_rank_table_.node_list) {
    if (node.node_addr.empty()) {
      GELOGE(FAILED, "node addr is empty.");
      return false;
    }
    const std::string &ip = node.node_addr;
    for (HcomRank &rank : node.node_ranks) {
      if (rank.port.empty()) {
        GELOGE(FAILED, "node addr is %s, but port is empty.", ip.c_str());
        return false;
      }
      const std::string ip_port = ip + ":" + std::to_string(rank.hcom_device_id);
      ip_rank_map_[ip_port] = &rank;
      GELOGI("Insert rank success, key = %s", ip_port.c_str());
    }
  }
  return InitRankId();
}

bool RankTableBuilder::InitRankId() {
  if (ip_rank_map_.empty()) {
    GELOGE(FAILED, "ipRankMap_ is empty");
    return false;
  }
  int32_t rank_id = 0;
  for (auto item : ip_rank_map_) {
    if (item.second == nullptr) {
      GELOGE(FAILED, "rankptr is nullptr, key is %s", item.first.c_str());
      return false;
    }
    item.second->rank_id = std::to_string(rank_id);
    GELOGI("Device[%s] use rank[%s]", item.second->device_id.c_str(), item.second->rank_id.c_str());
    rank_id++;
  }
  return true;
}
}  // namespace ge
