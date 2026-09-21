/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_EXTERNAL_WEIGHT_MANAGER_H_
#define GE_GRAPH_EXTERNAL_WEIGHT_MANAGER_H_

#include <map>
#include <set>
#include <mutex>
#include <vector>
#include <algorithm>
#include "ge/ge_api_types.h"
#include "nlohmann/json.hpp"
#include "graph/any_value.h"

namespace ge {
struct FileConstantInfo {
  std::string weight_path;
  size_t weight_offset;
  size_t weight_length;
};

struct FileConstantMeta {
  std::map<std::string, std::string> hash_to_weight_file;
  std::map<std::string, size_t> hash_to_weight_offset;
};

void from_json(const nlohmann::json &j, FileConstantMeta &meta);
void to_json(nlohmann::json &j, const FileConstantMeta &meta);

class ExternalWeightManager {
 public:
  explicit ExternalWeightManager(const uint64_t session_id);

  ~ExternalWeightManager() = default;

  Status CreateWeightPath();

  bool CheckAndSetWeightLoaded(const std::string &file_name, const uint32_t device_id);

  void Finalize() noexcept;

  void SetWeightPath(const std::string &weight_path);

  const std::string &GetWeightPath();

  FileConstantMeta &MutableMetaFile() { return meta_file_; };

  void SaveSlicedFileConstantInfo(const std::string &shard_info, const FileConstantInfo &weight_info);

  bool TryGetSlicedFileConstantInfo(const std::string &shard_info, FileConstantInfo &weight_info);

  static std::string GetWeightPathFromOption();

 private:
  std::mutex mutex_;
  uint64_t session_id_;
  std::string weight_path_;
  FileConstantMeta meta_file_;
  std::map<uint32_t, std::set<std::string>> loaded_external_weight_files_;
  std::map<std::string, FileConstantInfo> shard_info_to_fileconstant_info_;
};

using ExternalWeightManagerPtr = std::shared_ptr<ExternalWeightManager>;

class ExternalWeightManagerPool {
 public:
  static ExternalWeightManagerPool &Instance();

  ~ExternalWeightManagerPool();

  ExternalWeightManagerPtr GetManager(const uint64_t session_id);

  void RemoveManager(const uint64_t session_id);

  void Destroy() noexcept;

 private:
  ExternalWeightManagerPool() = default;
  std::mutex mutex_;
  std::map<uint64_t, ExternalWeightManagerPtr> session_id_to_manager_;
};
}  // namespace ge
#endif  // GE_GRAPH_EXTERNAL_WEIGHT_MANAGER_H_
