/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_BUILD_MODEL_CACHE_H_
#define GE_GRAPH_BUILD_MODEL_CACHE_H_

#include <mutex>
#include "ge/ge_api_types.h"
#include "common/model/ge_root_model.h"
#include "nlohmann/json.hpp"
#include "graph/manager/graph_var_manager.h"
#include "graph/manager/util/graph_rebuild_state_ctrl.h"

namespace ge {
struct CacheFileIdx {
  std::string graph_key;
  std::string cache_file_name;
  std::string var_desc_file_name;
};

struct VarDescCache {
  std::unordered_map<std::string, ge::GeTensorDesc> var_desc_map;
  std::unordered_map<std::string, std::vector<TransNodeInfo>> var_trans_roads_map;
  std::vector<std::string> changed_var_names;
  std::unordered_map<std::string, ge::GeTensorDesc> staged_var_desc_map;
};

struct CacheConfig {
  bool cache_manual_check = false;
  bool cache_debug_mode = false;
};

class ModelCache {
 public:
   ModelCache() = default;
   ~ModelCache();

  /**
   * @brief Init root model cache.
   * @return Status SUCCESS:init cache success.
   *                Other:failed.
   */
  Status Init(const ComputeGraphPtr &root_graph, GraphRebuildStateCtrl *ctrl);

  static std::string GetCacheDirFromContext();
  static std::string GetGraphKeyFromContext();

  /**
   * @brief try load model from cache.
   * if return success, but ge_root_model is null, means no cache.
   * @param ge_root_model loaded model.
   * @return Status SUCCESS:load cache success or no cache.
   *                Other:failed.
   */
  Status TryLoadModelFromCache(const ComputeGraphPtr &root_graph, GeRootModelPtr &ge_root_model);

  /**
   * @brief try cache model.
   * @param ge_root_model cache model.
   * @return Status SUCCESS:cache success or no need cache
   *                Other: failed.
   */
  Status TryCacheModel(GeRootModelPtr &ge_root_model);

  static bool CheckFileExist(const std::string &file_path);
  static Status ReadJsonFile(const std::string &file_path, nlohmann::json &json_obj);
  static Status WriteJsonFile(const std::string &file_path, const nlohmann::json &json_obj);

 private:
  static Status SaveModelToGeRootModel(GeRootModelPtr &ge_root_model, const std::string &cache_file_name);
  static Status SerializeModel(const GeRootModelPtr &ge_root_model, ModelBufferData &model_buff);
  static bool IsMatchFileName(const std::string &str);
  static Status CreateIndexFile(const std::string &index_file);
  Status ReadIndex(const std::string &index_file, std::vector<CacheFileIdx> &cache_file_list) const;
  Status InitCacheFileInfo();
  Status InitCacheFileByIdx(const std::string &cache_path);
  Status SaveCacheIndexFile() const;
  Status GetRealFileName(std::string &cache_file_name) const;
  void GenerateCacheFile();
  Status RefreshVariableDesc(const ComputeGraphPtr &root_graph, const VarDescCache &update_var_desc);
  static void InitVarDescFromProto(const deployer::VarDescInfo &desc_proto, VarDescCache &var_desc);
  static bool CompareVarDesc(uint64_t session_id, const VarDescCache &var_descs);
  static Status MatchVariableDesc(uint64_t session_id,
                                  const std::string &var_desc_file_name,
                                  bool &matched,
                                  VarDescCache &update_var_desc);
  Status SaveVarDescToFile();
  Status TryMatchVarDescWithCache(bool &is_matched, VarDescCache &update_var_desc);
  Status RenewVarDesc(uint64_t session_id, const VarDescCache &update_var_desc) const;
  Status RenewVarDesc(uint64_t session_id, const std::string &var_name, const VarTransRoad &fusion_road) const;
  static std::string NormalizeDirPath(const std::string &dir_path);
  static Status CreateDir(const std::string &dir_path);
  static Status LoadToGeRootModel(const std::string &model_path, GeRootModelPtr &ge_root_model);
  static Status AssignConstantVarMem(const GeRootModelPtr &ge_root_model,
      const std::string &model_path, const uint64_t session_id, const uint32_t graph_id);
  static Status UpdateGeModelSessionId(const GeRootModelPtr &ge_root_model, const uint64_t session_id);
  static Status UpdateSessionGraphId(const GeRootModelPtr &ge_root_model, const std::string &session_graph_id);
  Status CheckCacheFile(const ComputeGraphPtr &root_graph, bool &need_load);
  static Status ReadCacheConfig(const std::string &config_file, CacheConfig &cache_config);
  bool cache_enable_ = false;
  std::string cache_dir_;
  CacheFileIdx cache_index_;
  std::string cache_file_prefix_;
  std::string cache_file_time_str_;
  std::string index_file_;
  std::vector<CacheFileIdx> matched_file_list_;
  deployer::VarDescInfo var_desc_before_compile_;
  uint64_t session_id_ = UINT64_MAX;
  uint64_t graph_id_ = UINT32_MAX;
  int32_t lock_file_fd_ = -1;
  GraphRebuildStateCtrl *var_accelerate_ctrl_ = nullptr;

  std::string build_info_path_;
  ComputeGraphPtr root_graph_ = nullptr;
  bool cache_debug_mode_ = false;
  bool cache_manual_check_ = false;
};
void from_json(const nlohmann::json &json_obj, CacheFileIdx &cache_file_index);
void to_json(nlohmann::json &json_obj, const CacheFileIdx &cache_file_index);
}  // namespace ge

#endif  // GE_GRAPH_BUILD_MODEL_CACHE_H_
