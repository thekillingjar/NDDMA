/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "inc/ffts_utils.h"
#include <mutex>
#include <sys/time.h>
#include <sstream>
#include <climits>
#include <cmath>
#include <numeric>
#include "inc/ffts_error_codes.h"
#include "engine/engine_manager.h"
#include "base/err_msg.h"
#include "platform/platform_info.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "runtime/rt.h"
#include "framework/common/types.h"
#include "common/sgt_slice_type.h"


namespace ffts {
std::mutex g_report_error_msg_mutex;

FFTSMode GetPlatformFFTSMode() {
  string soc_version = EngineManager::Instance(kFFTSPlusCoreName).GetSocVersion();
  fe::PlatFormInfos platform_infos;
  fe::OptionalInfos opti_compilation_infos;
  auto ret = fe::PlatformInfoManager::Instance().GetPlatformInfos(soc_version, platform_infos,
                                                                  opti_compilation_infos);
  if (ret != SUCCESS) {
    FFTS_LOGW("Failed to retrieve platform information using SOC version [%s].", soc_version.c_str());
    return FFTSMode::FFTS_MODE_NO_FFTS;
  }
  string ffts_mode = "";
  platform_infos.GetPlatformRes(kSocInfo, kFFTSMode, ffts_mode);
  FFTS_LOGD("GetPlatformFFTSMode [%s].", ffts_mode.c_str());
  if (ffts_mode == kFFTSPlus) {
    return FFTSMode::FFTS_MODE_FFTS_PLUS;
  } else if (ffts_mode == kFFTS) {
    return FFTSMode::FFTS_MODE_FFTS;
  }
  return FFTSMode::FFTS_MODE_NO_FFTS;
}

int64_t GetMicroSecondTime() {
  struct timeval tv = {0, 0};
  int ret = gettimeofday(&tv, nullptr);
  if (ret != 0) {
    return 0;
  }
  if (tv.tv_sec < 0 || tv.tv_usec < 0) {
    return 0;
  }
  int64_t micro_multiples = 1000000;
  int64_t second = tv.tv_sec;
  FFTS_INT64_MULCHECK(second, micro_multiples);
  int64_t second_to_micro = second * micro_multiples;
  FFTS_INT64_ADDCHECK(second_to_micro, tv.tv_usec);
  return second_to_micro + tv.tv_usec;
}


void LogErrorMessage(std::string error_code, const std::map<std::string, std::string> &args_map) {
  std::vector<const char*> args_keys;
  std::vector<const char*> args_values;
  for (const auto &item : args_map) {
    args_keys.push_back(item.first.c_str());
    args_values.push_back(item.second.c_str());
  }
  int result = REPORT_PREDEFINED_ERR_MSG(error_code.c_str(), args_keys, args_values);

  FFTS_LOGE_IF(result != 0, "Faild to call ReportErrMessage.");
}

std::string RealPath(const std::string &path) {
  if (path.empty()) {
    FFTS_LOGI("The path string is a nullptr.");
    return "";
  }
  if (path.size() >= PATH_MAX) {
    FFTS_LOGI("file path %s is too long!", path.c_str());
    return "";
  }

  // PATH_MAX is the system marco, indicate the maximum length for file path
  // pclint check one param in stack can not exceed 1K bytes
  char resoved_path[PATH_MAX] = {0x00};

  std::string res;

  // path not exists or not allowed to read return nullptr
  // path exists and readable, return the resoved path
  if (realpath(path.c_str(), resoved_path) != nullptr) {
    res = resoved_path;
  } else {
    FFTS_LOGI("Path '%s' does not exist.", path.c_str());
  }
  return res;
}

void DumpGraph(const ge::ComputeGraph &graph, const std::string &suffix) {
  std::shared_ptr<ge::ComputeGraph> compute_graph_ptr = FFTSComGraphMakeShared<ge::ComputeGraph>(graph);
  ge::GraphUtils::DumpGEGraph(compute_graph_ptr, suffix);
}

bool IsSubGraphData(const ge::OpDescPtr &op_desc_ptr) {
  if (op_desc_ptr == nullptr) {
    return false;
  }
  if (op_desc_ptr->GetType() == CONSTANTOP || op_desc_ptr->GetType() == CONSTANT) {
    return true;
  }
  if (op_desc_ptr->GetType() != DATA) {
    return false;
  }
  return op_desc_ptr->HasAttr(ge::ATTR_NAME_PARENT_NODE_INDEX);
}

bool IsSubGraphDataWithControlEdge(const ge::NodePtr &node, uint32_t recurise_cnt) {
  if (recurise_cnt == 0) {
    FFTS_LOGI("Recursive count reduced to 0, will not continue processing.");
    return false;
  }

  ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
  if (op_desc_ptr == nullptr) {
    return false;
  }

  if (IsPhonyOp(op_desc_ptr) || op_desc_ptr->GetType() == CONSTANTOP || op_desc_ptr->GetType() == CONSTANT ||
      op_desc_ptr->GetType() == REFDATA) {
    FFTS_LOGD("Op type: %s, name: %s, recursive_cnt is %u.", op_desc_ptr->GetType().c_str(),
              op_desc_ptr->GetName().c_str(), recurise_cnt);

    for (const auto &in_node : node->GetInAllNodes()) {
      FFTS_CHECK(in_node == nullptr, FFTS_LOGW("in_node is null"), return false);
      auto in_node_desc = in_node->GetOpDesc();
      if (in_node_desc == nullptr) {
        return false;
      }
      FFTS_LOGD("Get node type: %s, name: %s, from node type: %s, name: %s.", op_desc_ptr->GetType().c_str(),
                op_desc_ptr->GetName().c_str(), in_node_desc->GetType().c_str(), in_node_desc->GetName().c_str());

      bool res = IsSubGraphDataWithControlEdge(in_node, recurise_cnt - 1);
      if (!res) {
        return false;
      }
    }
    return true;
  }

  if (op_desc_ptr->GetType() != DATA) {
    return false;
  }
  return op_desc_ptr->HasAttr(ge::ATTR_NAME_PARENT_NODE_INDEX);
}

bool IsPhonyOp(const ge::OpDescPtr &op_desc_ptr) {
  if (op_desc_ptr == nullptr) {
    return false;
  }
  if (op_desc_ptr->HasAttr(ge::ATTR_NAME_NOTASK)) {
    return true;
  }
  return (kPhonyTypes.count(op_desc_ptr->GetType()) != 0);
}

Status IsNoEdge(const ge::NodePtr &parant_node, const ge::NodePtr &child_node) {
  for (auto &out_node : parant_node->GetOutAllNodes()) {
    if (out_node == child_node) {
       FFTS_LOGD("out_node: %s, child_node: %s.",
                 out_node->GetOpDesc()->GetName().c_str(), child_node->GetOpDesc()->GetName().c_str());
      return false;
    }
  }
  return true;
}

bool IsSubGraphNetOutput(const ge::OpDescPtr &op_desc_ptr) {
  if (op_desc_ptr == nullptr || op_desc_ptr->GetType() != NETOUTPUT) {
    return false;
  }
  for (auto &tensor : op_desc_ptr->GetAllInputsDescPtr()) {
    if (ge::AttrUtils::HasAttr(tensor, ge::ATTR_NAME_PARENT_NODE_INDEX)) {
      return true;
    }
  }
  return false;
}

void GenerateSameAtomicNodesMap(std::vector<ge::NodePtr> &graph_nodes,
                                SameAtomicNodeMap &same_memset_nodes_map) {
  for (auto &node : graph_nodes) {
    ThreadSliceMapPtr slice_info_ptr = nullptr;
    FFTS_CHECK(node == nullptr, FFTS_LOGW("node is null"), return);
    slice_info_ptr = node->GetOpDesc()->TryGetExtAttr(ffts::kAttrSgtStructInfo, slice_info_ptr);
    if (slice_info_ptr == nullptr) {
      continue;
    }
    if (slice_info_ptr->same_atomic_clean_nodes.empty()) {
      continue;
    }
    auto same_memset_nodes = slice_info_ptr->same_atomic_clean_nodes;
    if (same_memset_nodes_map.find(same_memset_nodes) == same_memset_nodes_map.end()) {
      same_memset_nodes_map[same_memset_nodes] = {node};
    } else {
      same_memset_nodes_map[same_memset_nodes].emplace_back(node);
    }
  }
}

void SetBitOne(const uint32_t pos, uint32_t &bm) {
  if (pos > sizeof(uint32_t) * kOneByteBitNum) {
    FFTS_LOGW("Set bit at position %u within the uint32_t range.", pos);
  }
  bm |= static_cast<uint32_t>(0x1U << pos);
}

void SetBitOne(const uint32_t pos, uint64_t &bm) {
  if (pos > sizeof(uint64_t) * kOneByteBitNum) {
    FFTS_LOGW("Set bit to position %u over uint64_t range.", pos);
  }
  bm |= static_cast<uint64_t>(0x1UL << pos);
}

void DeleteNode(ge::NodePtr &node) {
  PrintNode(node);
  (void)ge::GraphUtils::IsolateNode(node, {});
  (void)ge::GraphUtils::RemoveNodeWithoutRelink(node->GetOwnerComputeGraph(), node);
}

void PrintNode(const ge::NodePtr &node) {
  if (CheckLogLevel(static_cast<int32_t>(FFTS), DLOG_DEBUG) != 1) {
    return;
  }
  FFTS_LOGD("===============================printcurrentnode start=================================");
  if (node == nullptr) {
    return;
  }
  FFTS_LOGD("Node name = %s, type = %s.", node->GetName().c_str(), node->GetType().c_str());
  FFTS_LOGD("===============================printcurrentnode end=================================");
}

void PrintNodeAttrExtNode(const ge::NodePtr &node, std::string attrname) {
  if (CheckLogLevel(static_cast<int32_t>(FFTS), DLOG_DEBUG) != 1) {
    return;
  }
  FFTS_LOGD("===============================printnodeattrext start=================================");
  if (node == nullptr) {
    return;
  }
  ge::OpDescPtr op_desc = node->GetOpDesc();
  FFTS_LOGD("Node name = %s, type = %s attrname = %s.", node->GetName().c_str(), node->GetType().c_str(),
            attrname.c_str());
  ge::NodePtr attrnode = nullptr;
  attrnode = op_desc->TryGetExtAttr(attrname, attrnode);
  if (attrnode == nullptr) {
    FFTS_LOGD("Node has not dst attr node");
  } else {
    PrintNode(attrnode);
  }
  FFTS_LOGD("===============================printnodeattrext end=================================");
}

void PrintNodeAttrExtNodes(const ge::NodePtr &node, std::string attrname) {
  if (CheckLogLevel(static_cast<int32_t>(FFTS), DLOG_DEBUG) != 1) {
    return;
  }
  FFTS_LOGD("===============================printnodeattrext start=================================");
  if (node == nullptr) {
    return;
  }
  ge::OpDescPtr op_desc = node->GetOpDesc();
  FFTS_LOGD("Node name = %s, type = %s attrname = %s.", node->GetName().c_str(), node->GetType().c_str(),
            attrname.c_str());
  std::shared_ptr<std::vector<ge::NodePtr>> attrnodes = nullptr;
  attrnodes = op_desc->TryGetExtAttr(attrname, attrnodes);
  if (attrnodes != nullptr && (*attrnodes).size() != 0) {
    for (const auto &attr_node : (*attrnodes)) {
      PrintNode(attr_node);
    }
  } else {
     FFTS_LOGD("Node does not have dst attr nodes");
  }
  FFTS_LOGD("===============================printnodeattrext end=================================");
}

Status UnfoldPartionCallOnlyOneDepth(ge::ComputeGraph& graph, const std::string &node_type) {
  FFTS_LOGD("UnfoldPartionCallOnlyOneDepth graphname = %s node_type = %s", graph.GetName().c_str(), node_type.c_str());
  for (auto &node : graph.GetDirectNode()) {
    FFTS_CHECK_NOTNULL(node);
    if (node->GetType() != node_type) {
      continue;
    }
    FFTS_LOGD("UnfoldPartionCallOnlyOneDepth node_name = %s.", node->GetName().c_str());
    auto func_graph = node->GetOwnerComputeGraph();
    FFTS_CHECK_NOTNULL(func_graph);
    ge::ComputeGraphPtr src_graph = func_graph->TryGetExtAttr("part_src_graph", ge::ComputeGraphPtr());
    FFTS_CHECK_NOTNULL(src_graph);
    auto root_graph = ge::GraphUtils::FindRootGraph(src_graph);
    FFTS_CHECK_NOTNULL(root_graph);
    std::vector<ge::ComputeGraphPtr> sub_graphs = root_graph->GetAllSubgraphs();
    for (const auto &sub_graph : sub_graphs) {
      FFTS_CHECK_NOTNULL(sub_graph);
      if (sub_graph->GetParentNode()->GetName() == node->GetName() &&
          ge::GraphUtils::UnfoldGraph(sub_graph, func_graph, node, nullptr) != ge::GRAPH_SUCCESS) {
        REPORT_FFTS_ERROR("[OptimizeFusedGraph][UnfoldSubGraph] UnfoldSubGraph failed.");
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

bool IsMemoryEmpty(const ge::GeTensorDesc &tensor_desc) {
  auto memory_size_calc_type = static_cast<int64_t>(ge::MemorySizeCalcType::NORMAL);
  (void)ge::AttrUtils::GetInt(tensor_desc, ge::ATTR_NAME_MEMORY_SIZE_CALC_TYPE, memory_size_calc_type);
  return memory_size_calc_type == static_cast<int64_t>(ge::MemorySizeCalcType::ALWAYS_EMPTY);
}
}  // namespace ffts
