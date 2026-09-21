/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fusion_rule_manager/fusion_rule_parser/fusion_rule_json_pattern.h"
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include "common/op_info_common.h"
#include "ops_store/ops_kernel_manager.h"
#include "fusion_rule_manager/fusion_rule_parser/fusion_rule_parser_utils.h"

using std::map;
using std::string;
using std::vector;

namespace fe {
namespace {
bool CheckNodeSupported(const vector<string> &types) {
  std::ostringstream oss;
  for (const auto &op_type : types) {
    if (IsPlaceOrEnd(op_type)) {
      return true;
    }

    std::string engine_name = FusionRuleParserUtils::Instance()->GetEngineName();
    FE_CHECK(engine_name.empty(), REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ChkNdSpt] The engine_name is empty."),
             return ILLEGAL_JSON);
    OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(engine_name).GetHighPrioOpKernelInfoPtr(op_type);
    if (op_kernel_info_ptr != nullptr) {
      return true;
    }
    oss << (op_type + ", ");
  }
  FE_LOGD("Can't find op type:[%s] in OpKernelInfoStore.", oss.str().c_str());
  return false;
}
} // namespace

Status FusionRuleJsonPattern::ParseToJsonPattern(const nlohmann::json &json_object) {
  FE_LOGD("Start parsing fusion rule.");
  FusionRuleJsonStruct fusion_rule_json_struct;

  Status ret = LoadJson(json_object, fusion_rule_json_struct);
  if (ret != SUCCESS) {
    FE_LOGW("Load json to single fusion rule not successfully");
    return ret;
  }

  ret = ParseToOuterInput(fusion_rule_json_struct.outer_inputs, input_infos_);
  if (ret != SUCCESS) {
    FE_LOGW("Parse json fusion rule:%s OuterInputs not successfully.", name_.c_str());
    return ret;
  }

  ret = ParseToOuterOutput(fusion_rule_json_struct.outer_outputs, output_infos_);
  if (ret != SUCCESS) {
    FE_LOGW("Parse json fusion rule:%s OuterOutputs not successfully.", name_.c_str());
    return ret;
  }

  ret = ParseToOriginGraph(fusion_rule_json_struct.origin_json_graph, origin_graph_);
  if (ret != SUCCESS) {
    FE_LOGW("Parse json fusion rule:%s OriginGraph not successfully.", name_.c_str());
    return ret;
  }

  ret = ParseToFusionGraph(fusion_rule_json_struct.fusion_json_graph, fusion_graph_);
  if (ret != SUCCESS) {
    FE_LOGW("Parse json fusion rule:%s FusionGraph not successfully.", name_.c_str());
    return ret;
  }

  if (fusion_graph_->HasAttrs()) {
    // To parse fusion graph attr assign expression, need to gather all node
    // info in graph
    map<string, vector<string>> node_map;
    origin_graph_->GatherNode(node_map);
    fusion_graph_->GatherNode(node_map);
    ret = fusion_graph_->ParseAllAttrValue(node_map);
    if (ret != SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ParToJsPtn] Parse fusion rule:%s FusionGraph.Attrs failed.",
                      name_.c_str());
      return ret;
    }
  }
  return SUCCESS;
}

Status FusionRuleJsonPattern::LoadJson(const nlohmann::json &json_object,
                                       FusionRuleJsonStruct &fusion_rule_json_struct) {
  if (!json_object.is_object()) {
    REPORT_FE_ERROR(
        "[GraphOpt][FusionRuleInit][LdJson] Single fusion rule json item should be object, actually is %s",
        GetJsonType(json_object).c_str());
    return ILLEGAL_JSON;
  }

  map<string, bool> check_map{
      {RULE_NAME, false}, {OUTER_INPUTS, false}, {OUTER_OUTPUTS, false}, {ORIGIN_GRAPH, false}, {FUSION_GRAPH, false}};

  for (const auto &item : json_object.items()) {
    if (item.key() == RULE_NAME) {
      // json content should be "RuleName": "*******"
      Status ret = GetFromJson(item.value(), name_);
      if (ret != SUCCESS) {
        REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][LdJson] Get %s from json failed.", RULE_NAME.c_str());
        return ret;
      }
      check_map[RULE_NAME] = true;
    } else if (item.key() == OUTER_INPUTS) {
      // json content should be "OuterInputs": [{"name": "****", "src": "****"},
      // ...]
      if (!item.value().is_array()) {
        REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][LdJson] Type of %s should be array, actually is %s",
                        OUTER_INPUTS.c_str(), GetJsonType(item.value()).c_str());
        return ILLEGAL_JSON;
      }
      for (const auto &iter : item.value()) {
        fusion_rule_json_struct.outer_inputs.push_back(iter);
      }
      check_map[OUTER_INPUTS] = true;
    } else if (item.key() == OUTER_OUTPUTS) {
      // json content should be "OuterOutputs": [{"name": "****"}, ...]
      if (!item.value().is_array()) {
        REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][LdJson] Type of %s should be array, actually is %s",
                        OUTER_OUTPUTS.c_str(), GetJsonType(item.value()).c_str());
        return ILLEGAL_JSON;
      }
      for (const auto &iter : item.value()) {
        fusion_rule_json_struct.outer_outputs.push_back(iter);
      }
      check_map[OUTER_OUTPUTS] = true;
    } else if (item.key() == ORIGIN_GRAPH) {
      // json content should be "OriginGraph": {"Nodes":[...],...}
      fusion_rule_json_struct.origin_json_graph = item.value();
      check_map[ORIGIN_GRAPH] = true;
    } else if (item.key() == FUSION_GRAPH) {
      // json content should be "FusionGraph": {"Nodes":[...],...}
      fusion_rule_json_struct.fusion_json_graph = item.value();
      check_map[FUSION_GRAPH] = true;
    } else {
      REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][LdJson] Not support key:%s in fusion rule", item.key().c_str());
      return ILLEGAL_JSON;
    }
  }
  if (!AnalyseCheckMap(check_map)) {
    REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][LdJson] Doing single fusion rule info key words check failed.");
    return ILLEGAL_JSON;
  }

  return SUCCESS;
}

Status FusionRuleJsonPattern::ParseToOuterInput(const std::vector<nlohmann::json> &outer_inputs,
                                                std::vector<FusionRuleJsonOuterPtr> &input_infos) {
  for (const auto &iter : outer_inputs) {
    FusionRuleJsonOuterPtr input_info = nullptr;
    FE_MAKE_SHARED(input_info = make_shared<FusionRuleJsonOuter>(), return INTERNAL_ERROR);

    Status ret = input_info->ParseToJsonOuter(iter);
    if (ret != SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ParseToOutIn] Parse outer inputs failed.");
      return ret;
    }
    input_infos.push_back(input_info);
  }

  return SUCCESS;
}

Status FusionRuleJsonPattern::ParseToOuterOutput(const std::vector<nlohmann::json> &outer_outputs,
                                                 std::vector<FusionRuleJsonOuterPtr> &output_infos) {
  for (const auto &iter : outer_outputs) {
    FusionRuleJsonOuterPtr output_info = nullptr;
    FE_MAKE_SHARED(output_info = make_shared<FusionRuleJsonOuter>(), return INTERNAL_ERROR);

    Status ret = output_info->ParseToJsonOuter(iter);
    if (ret != SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ParseToOutOut] Parse outer outputs failed.");
      return ret;
    }

    if (output_info->HasSrc()) {
      REPORT_FE_ERROR(
          "[GraphOpt][FusionRuleInit][ParseToOutOut] Parse outer outputs failed, outer outputs should not define src.");
      return ILLEGAL_JSON;
    }
    output_infos.push_back(output_info);
  }

  return SUCCESS;
}

Status FusionRuleJsonPattern::ParseToOriginGraph(const nlohmann::json &json_object,
                                                 FusionRuleJsonGraphPtr &origin_graph) {
  if (origin_graph != nullptr) {
    REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ParseToOrgnGph] Parse OriginGraph input graph ptr is not nullptr");
    return ILLEGAL_JSON;
  }

  FusionRuleJsonGraphPtr graph = nullptr;
  FE_MAKE_SHARED(graph = make_shared<FusionRuleJsonGraph>(), return INTERNAL_ERROR);

  Status ret = graph->ParseToJsonGraph(json_object);
  if (ret != SUCCESS) {
    if (ret != INVALID_RULE) {
      REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ParseToOrgnGph] Parse origin graph failed!");
    }
    return ret;
  }
  if (graph->HasAttrs()) {
    REPORT_FE_ERROR(
        "[GraphOpt][FusionRuleInit][ParseToOrgnGph] Parse origin graph failed, origin graph should not define Attrs");
    return ILLEGAL_JSON;
  }
  origin_graph = graph;

  return SUCCESS;
}

Status FusionRuleJsonPattern::ParseToFusionGraph(const nlohmann::json &json_object,
                                                 FusionRuleJsonGraphPtr &fusion_graph) {
  if (fusion_graph != nullptr) {
    REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ParseToFusGph] Parse FusionGraph input graph ptr is not nullptr");
    return ILLEGAL_JSON;
  }
  FusionRuleJsonGraphPtr graph = nullptr;
  FE_MAKE_SHARED(graph = make_shared<FusionRuleJsonGraph>(), return INTERNAL_ERROR);

  Status ret = graph->ParseToJsonGraph(json_object);
  if (ret != SUCCESS) {
    if (ret == INVALID_RULE) {
      FE_LOGD("Exist OpKernelInfoStore not supported item in fusion graph.");
    } else {
      REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ParseToFusGph] Parse fusion graph failed!");
    }
    return ret;
  }
  fusion_graph = graph;
  return SUCCESS;
}

template <typename T>
Status ParseItem(const nlohmann::detail::iteration_proxy_value<nlohmann::detail::iter_impl<const nlohmann::json>> &item,
                 std::vector<std::shared_ptr<T>> &storage_vec, const string &key) {
  if (!item.value().is_array()) {
    REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ParseItem] Type of graph.%s should be array, actually is %s",
                    key.c_str(), GetJsonType(item.value()).c_str());
    return ILLEGAL_JSON;
  }
  for (const auto &iter : item.value()) {
    std::shared_ptr<T> local_item = nullptr;
    FE_MAKE_SHARED(local_item = make_shared<T>(), return INTERNAL_ERROR);
    Status ret = local_item->ParseJson(iter);
    if (ret != SUCCESS) {
      if (ret == INVALID_RULE) {
        FE_LOGW("Exist OpKernelInfoStore not supported %s in graph.", key.c_str());
      } else {
        REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ParseItem] Parse %s from json failed.", key.c_str());
      }
      return ret;
    }
    storage_vec.push_back(local_item);
  }
  return SUCCESS;
}

Status FusionRuleJsonGraph::ParseToJsonGraph(const nlohmann::json &json_object) {
  if (!json_object.is_object()) {
    REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ParseToJsGph] Type of graph should be object, actually is %s",
                    GetJsonType(json_object).c_str());
    return ILLEGAL_JSON;
  }

  map<string, bool> check_map{{NODES, false}, {EDGES, false}};
  for (const auto &item : json_object.items()) {
    if (item.key() == NODES) {
      Status ret = ParseItem(item, nodes_, item.key());
      if (ret != SUCCESS) {
        return ret;
      }
      check_map[NODES] = true;
    } else if (item.key() == EDGES) {
      Status ret = ParseItem(item, edges_, item.key());
      if (ret != SUCCESS) {
        return ret;
      }
      check_map[EDGES] = true;
    } else if (item.key() == ATTRS) {
      Status ret = ParseItem(item, attr_assigns_, item.key());
      if (ret != SUCCESS) {
        return ret;
      }
      has_attrs_ = true;
    } else {
      REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ParseToJsGph] Not support key:%s in graph.", item.key().c_str());
      return ILLEGAL_JSON;
    }
  }

  if (!AnalyseCheckMap(check_map)) {
    REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ParseToJsGph] Doing Graph info key words check failed.");
    return ILLEGAL_JSON;
  }

  return SUCCESS;
}

Status FusionRuleJsonGraph::ParseAllAttrValue(const std::map<string, std::vector<string>> &node_map) {
  for (const auto &iter : attr_assigns_) {
    Status ret = iter->ParseToAttrValue(node_map);
    if (ret != SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ParseAllAttrVal] Get attr assignment expression from json failed.");
      return ret;
    }
  }
  return SUCCESS;
}

Status FusionRuleJsonGraph::GatherNode(map<string, vector<string>> &node_map) {
  for (const auto &iter : nodes_) {
    if (node_map.find(iter->GetName()) != node_map.end()) {
      continue;
    }
    node_map[iter->GetName()] = iter->GetTypes();
  }
  return SUCCESS;
}

Status FusionRuleJsonEdge::ParseJson(const nlohmann::json &json_object) {
  if (!json_object.is_object()) {
    REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ParseJs] Type of graph.edge should be object, actually is %s",
                    GetJsonType(json_object).c_str());
    return ILLEGAL_JSON;
  }

  map<string, bool> check_map{{SRC, false}, {DST, false}};
  for (const auto &item : json_object.items()) {
    if (item.key() == SRC) {
      FusionRuleJsonAnchorPtr anchor = nullptr;
      FE_MAKE_SHARED(anchor = make_shared<FusionRuleJsonAnchor>(), return INTERNAL_ERROR);
      Status ret = anchor->ParseToJsonAnchor(item.value());
      if (ret != SUCCESS) {
        REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ParseJs] Get edge.%s.anchor from json failed.", SRC.c_str());
        return ret;
      }

      src_ = anchor;
      check_map[SRC] = true;
    } else if (item.key() == DST) {
      FusionRuleJsonAnchorPtr anchor = nullptr;
      FE_MAKE_SHARED(anchor = make_shared<FusionRuleJsonAnchor>(), return INTERNAL_ERROR);
      Status ret = anchor->ParseToJsonAnchor(item.value());
      if (ret != SUCCESS) {
        REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ParseJs] Get edge.%s.anchor from json failed.", DST.c_str());
        return ret;
      }

      dst_ = anchor;
      check_map[DST] = true;
    } else {
      REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ParseJs] Not support key:%s in Edge.", item.key().c_str());
      return ILLEGAL_JSON;
    }
  }
  if (!AnalyseCheckMap(check_map)) {
    REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ParseJs] Doing Edge info key words check failed.");
    return ILLEGAL_JSON;
  }
  return SUCCESS;
}

Status FusionRuleJsonEdge::ParseToJsonEdge(const nlohmann::json &json_object) {
  if (!json_object.is_object()) {
    REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ParseToJsEdge] Type of graph.edge should be object, actually is %s",
                    GetJsonType(json_object).c_str());
    return ILLEGAL_JSON;
  }

  map<string, bool> check_map{{SRC, false}, {DST, false}};
  for (const auto &item : json_object.items()) {
    if (item.key() == SRC) {
      FusionRuleJsonAnchorPtr anchor = nullptr;
      FE_MAKE_SHARED(anchor = make_shared<FusionRuleJsonAnchor>(), return INTERNAL_ERROR);
      Status ret = anchor->ParseToJsonAnchor(item.value());
      if (ret != SUCCESS) {
        REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ParseToJsEdge] Get edge.%s.anchor from json failed.", SRC.c_str());
        return ret;
      }

      src_ = anchor;
      check_map[SRC] = true;
    } else if (item.key() == DST) {
      FusionRuleJsonAnchorPtr anchor = nullptr;
      FE_MAKE_SHARED(anchor = make_shared<FusionRuleJsonAnchor>(), return INTERNAL_ERROR);
      Status ret = anchor->ParseToJsonAnchor(item.value());
      if (ret != SUCCESS) {
        REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ParseToJsEdge] Get edge.%s.anchor from json failed.", DST.c_str());
        return ret;
      }

      dst_ = anchor;
      check_map[DST] = true;
    } else {
      REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ParseToJsEdge] Not support key:%s in Edge.", item.key().c_str());
      return ILLEGAL_JSON;
    }
  }
  if (!AnalyseCheckMap(check_map)) {
    REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ParseToJsEdge] Doing Edge info key words check failed.");
    return ILLEGAL_JSON;
  }
  return SUCCESS;
}

Status FusionRuleJsonNode::ParseJson(const nlohmann::json &json_object) {
  if (!json_object.is_object()) {
    REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ParseJs] Type of graph.nodes should be object, but actually is [%s]",
                    GetJsonType(json_object).c_str());
    return ILLEGAL_JSON;
  }

  map<string, bool> check_map{{NAME, false}, {TYPE, false}};
  for (const auto &item : json_object.items()) {
    if (item.key() == NAME) {
      Status ret = GetFromJson(item.value(), name_);
      if (ret != SUCCESS) {
        REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ParseJs] Get Node[%s] from json failed.", NAME.c_str());
        return ret;
      }
      check_map[NAME] = true;
    } else if (item.key() == TYPE) {
      if (!item.value().is_array()) {
        REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ParseJs] Type of Node[%s] should be array, actually is [%s]",
                        TYPE.c_str(), GetJsonType(item.value()).c_str());
        return ILLEGAL_JSON;
      }
      if (item.value().empty()) {
        REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ParseJs] Each node should have at least specify one op type.");
        return ILLEGAL_JSON;
      }
      // Not supported more than one type, now
      if (item.value().size() > 1) {
        REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ParseJs] Not support more than one op type now.");
        return ILLEGAL_JSON;
      }

      for (const auto &iter : item.value()) {
        string type;
        Status ret = GetFromJson(iter, type);
        if (ret != SUCCESS) {
          REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ParseJs] Get Node[%s] from json failed", TYPE.c_str());
          return ret;
        }
        types_.push_back(type);
      }
      // each node should at least have one OpKernelInfoStore supported op type
      if (!CheckNodeSupported(types_)) {
        return INVALID_RULE;
      }
      check_map[TYPE] = true;
    } else {
      REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ParseJs] Not support key:%s in Nodes.", item.key().c_str());
      return ILLEGAL_JSON;
    }
  }
  if (!AnalyseCheckMap(check_map)) {
    REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ParseJs] Doing Node info key words check failed.");
    return ILLEGAL_JSON;
  }
  return SUCCESS;
}

Status FusionRuleJsonNode::ParseToJsonNode(const nlohmann::json &json_object) {
  if (!json_object.is_object()) {
    REPORT_FE_ERROR(
        "[GraphOpt][FusionRuleInit][ParseToJsNd] Type of graph.nodes should be object, but actually is [%s]",
        GetJsonType(json_object).c_str());
    return ILLEGAL_JSON;
  }

  map<string, bool> check_map{{NAME, false}, {TYPE, false}};
  for (const auto &item : json_object.items()) {
    if (item.key() == NAME) {
      Status ret = GetFromJson(item.value(), name_);
      if (ret != SUCCESS) {
        REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ParseToJsNd] Get Node[%s] from json failed.", NAME.c_str());
        return ret;
      }
      check_map[NAME] = true;
    } else if (item.key() == TYPE) {
      if (!item.value().is_array()) {
        REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ParseToJsNd] Type of Node[%s] should be array, actually is [%s]",
                        TYPE.c_str(), GetJsonType(item.value()).c_str());
        return ILLEGAL_JSON;
      }
      if (item.value().empty()) {
        REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ParseToJsNd] Each node should have at least specify one op type.");
        return ILLEGAL_JSON;
      }
      // Not supported more than one type, now
      if (item.value().size() > 1) {
        REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ParseToJsNd] Not support more than one op type now.");
        return ILLEGAL_JSON;
      }

      for (const auto &iter : item.value()) {
        string type;
        Status ret = GetFromJson(iter, type);
        if (ret != SUCCESS) {
          REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ParseToJsNd] Get Node[%s] from json failed", TYPE.c_str());
          return ret;
        }
        types_.push_back(type);
      }
      // each node should at least have one OpKernelInfoStore supported op type
      if (!CheckNodeSupported(types_)) {
        return INVALID_RULE;
      }
      check_map[TYPE] = true;
    } else {
      REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ParseToJsNd] Not support key:%s in Nodes.", item.key().c_str());
      return ILLEGAL_JSON;
    }
  }
  if (!AnalyseCheckMap(check_map)) {
    REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ParseToJsNd] Doing Node info key words check failed.");
    return ILLEGAL_JSON;
  }
  return SUCCESS;
}

Status FusionRuleJsonOuter::ParseToJsonOuter(const nlohmann::json &json_object) {
  if (!json_object.is_object()) {
    REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ParseToJsOut] Outer item's type should be object, actually is %s",
                    GetJsonType(json_object).c_str());
    return ILLEGAL_JSON;
  }

  map<string, bool> check_map{
      {NAME, false},
  };

  has_src_ = false;
  for (const auto &item : json_object.items()) {
    if (item.key() == NAME) {
      Status ret = GetFromJson(item.value(), name_);
      if (ret != SUCCESS) {
        REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ParseToJsOut] Get Outer.%s from json failed", NAME.c_str());
        return ret;
      }

      check_map[NAME] = true;
    } else if (item.key() == SRC) {
      string str;
      Status ret = GetFromJson(item.value(), str);
      if (ret != SUCCESS) {
        REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ParseToJsOut] Get Outer.%s from json failed", SRC.c_str());
        return ret;
      }

      string src_node;
      string src_index;
      if (SplitKeyValueByColon(str, src_node, src_index) != SUCCESS) {
        src_node_ = str;
        src_index_ = 0;  // default index is 0
      } else {
        ret = StringToInt(src_index, src_index_);
        if (ret != SUCCESS) {
          REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ParseToJsOut] Convert %s to index failed.", src_index.c_str());
          return ret;
        }
        src_node_ = src_node;
      }
      has_src_ = true;
    } else {
      REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ParseToJsOut] Not support key:%s in Outer define.",
                      item.key().c_str());
      return ILLEGAL_JSON;
    }
  }
  if (!AnalyseCheckMap(check_map)) {
    REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ParseToJsOut] Doing Outer info key words check failed.");
    return ILLEGAL_JSON;
  }

  return SUCCESS;
}

Status FusionRuleJsonAnchor::ParseToJsonAnchor(const nlohmann::json &json_object) {
  string str;
  Status ret = GetFromJson(json_object, str);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ParseToJsAch] Get Edges.Anchor from json failed.");
    return ret;
  }

  string src_node;
  string src_index;
  if (SplitKeyValueByColon(str, src_node, src_index) != SUCCESS) {
    name_ = str;
    has_index_ = false;
    src_node_ = "";
    src_index_ = 0;
  } else {
    ret = StringToInt(src_index, src_index_);
    if (ret != SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ParseToJsAch] Convert %s to index failed.", src_index.c_str());
      return ret;
    }
    src_node_ = src_node;
  }

  return SUCCESS;
}
}  // namespace fe
