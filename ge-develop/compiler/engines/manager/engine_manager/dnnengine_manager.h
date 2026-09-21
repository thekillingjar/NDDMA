/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_ENGINE_MANAGER_DNNENGINE_MANAGER_H_
#define GE_ENGINE_MANAGER_DNNENGINE_MANAGER_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <mutex>

#include "nlohmann/json.hpp"

#include "graph_metadef/common/plugin/plugin_manager.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/engine/dnnengine.h"
#include "graph/op_desc.h"
#include "graph/node.h"
#include "engines/manager/opskernel_manager/ops_kernel_manager.h"

using JsonHandle = void *;
namespace ge {
using nlohmann::json;

// Engine configuration
struct EngineConf {
  std::string id;                       // engine ID
  std::string name;                     // engine name
  bool independent{false};         // independent stream
  bool attach{false};              // attach stream
  bool skip_assign_stream{false};  // not assign stream
  std::string scheduler_id;             // scheduler ID
};
using EngineConfPtr = std::shared_ptr<EngineConf>;

// Configuration information of schedule unit
struct SchedulerConf {
  std::string id;                               // scheduler ID
  std::string name;                             // scheduler name
  std::string ex_attrs;                         // extra information
  std::map<std::string, EngineConfPtr> cal_engines;  // engine information
};

using DNNEnginePtr = std::shared_ptr<DNNEngine>;

class DNNEngineManager {
 public:
  friend class GELib;
  static DNNEngineManager &GetInstance();
  std::shared_ptr<ge::DNNEngine> GetEngine(const std::string &name) const;
  const std::map<std::string, DNNEnginePtr> &GetAllEngines() const { return engines_map_; }
  bool IsEngineRegistered(const std::string &name);
  // If can't find appropriate engine name, return "", report error
  static void GetExcludeEngines(std::set<std::string> &exclude_engines);
  static void GetExcludeEngines(const std::string &option, std::set<std::string> &engine_names);
  static void UpdateOpDescsWithOpInfos(const std::map<NodePtr, OpInfo> &nodes_op_infos);
  static void UpdateOpDescWithOpInfo(const OpDescPtr &op_desc, const OpInfo &op_info);
  std::string GetDNNEngineName(const ge::NodePtr &node_ptr);
  std::string GetDNNEngineName(const ge::NodePtr &node_ptr, const std::set<std::string> &exclude_engines,
                               OpInfo &matched_op_info);
  std::string GetCompositeEngineName(const ge::NodePtr &node_ptr, uint32_t recursive_depth = 1);
  std::string GetCompositeEngineName(const std::string &atomic_engine_name);
  std::string GetCompositeEngineKernelLibName(const std::string &composite_engine_name) const;
  const std::map<std::string, SchedulerConf> &GetSchedulers() const;
  void LogCheckSupportCost() const;
  void InitPerformanceStatistic();
  bool IsNoTask(const NodePtr &node);
  bool IsStreamAssignSkip(const NodePtr &node);
  bool IsStreamAssignSkip(const std::string &engine_name);

 private:
  DNNEngineManager();
  ~DNNEngineManager();
  Status Initialize(const std::map<std::string, std::string> &options);
  Status Finalize();
  Status ReadJsonFile(const std::string &file_path, JsonHandle handle);
  Status ParserJsonFile();
  Status ParserEngineMessage(const json engines_json, const std::string &scheduler_mark,
                             std::map<std::string, EngineConfPtr> &engines) const;
  Status CheckJsonFile() const;
  std::string GetHostCpuEngineName(const std::vector<OpInfo> &op_infos, const OpDescPtr &op_desc,
                                   OpInfo &matched_op_info) const;
  void GetOpInfos(std::vector<OpInfo> &op_infos, const OpDescPtr &op_desc,
		  bool &is_op_specified_engine) const;
  void InitAtomicCompositeMapping();
  std::string GetCompositeEngine(const NodePtr &node);
  std::string GetCompositeEngine(const NodePtr &func_node, uint32_t recursive_depth);
  std::string GetCompositeEngine(const ComputeGraphPtr &subgraph, uint32_t recursive_depth);

  PluginManager plugin_mgr_;
  std::map<std::string, DNNEnginePtr> engines_map_;
  std::map<std::string, ge::DNNEngineAttribute> engines_attrs_map_;
  std::map<std::string, SchedulerConf> schedulers_;
  std::map<std::string, uint64_t> checksupport_cost_;
  // {atomic_engine, composite_engine}
  std::map<std::string, std::string> atomic_2_composite_{};
  bool init_flag_;
  mutable std::mutex mutex_;
};
}  // namespace ge

#endif  // GE_ENGINE_MANAGER_DNNENGINE_MANAGER_H_
