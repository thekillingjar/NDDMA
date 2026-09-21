/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flow_graph/process_point.h"
#include "common/checker.h"
#include "common/util/mem_utils.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/serialization/attr_serializer_registry.h"
#include "proto/dflow.pb.h"
#include "flow_graph/flow_graph.h"
#include "flow_graph_utils.h"
#include "base/err_msg.h"

namespace ge {
namespace dflow {
class ProcessPointImpl {
public:
  ProcessPointImpl(const char_t *pp_name, ProcessPointType pp_type)
      : pp_name_(pp_name), pp_type_(pp_type), json_file_path_() {}
  ~ProcessPointImpl() = default;

  ProcessPointType GetProcessPointType() const {
    return pp_type_;
  }

  const char_t *GetProcessPointName() const {
    return pp_name_.c_str();
  }

  void SetCompileConfig(const char_t *json_file_path) {
    json_file_path_ = json_file_path;
  }

  const char_t *GetCompileConfig() const {
    return json_file_path_.c_str();
  }

private:
  std::string pp_name_;
  const ProcessPointType pp_type_;
  std::string json_file_path_;
};

ProcessPoint::ProcessPoint(const char_t *pp_name, ProcessPointType pp_type) {
  if (pp_name == nullptr) {
    impl_ = nullptr;
    GELOGE(GRAPH_FAILED, "ProcessPoint name is nullptr.");
  } else {
    impl_ = MakeShared<ProcessPointImpl>(pp_name, pp_type);
    if (impl_ == nullptr) {
      GELOGE(GRAPH_FAILED, "ProcessPointImpl make shared failed.");
    }
  }
}
ProcessPoint::~ProcessPoint() {}

ProcessPointType ProcessPoint::GetProcessPointType() const {
  if (impl_ == nullptr) {
    GELOGE(GRAPH_FAILED, "[Check][Param] ProcessPointImpl is nullptr, check failed");
    REPORT_INNER_ERR_MSG("E18888", "GetProcessPointType failed: ProcessPoint can not be used, impl is nullptr.");
    return ProcessPointType::INVALID;
  }
  return impl_->GetProcessPointType();
}

const char_t *ProcessPoint::GetProcessPointName() const {
  if (impl_ == nullptr) {
    GELOGE(GRAPH_FAILED, "[Check][Param] ProcessPointImpl is nullptr, check failed");
    REPORT_INNER_ERR_MSG("E18888", "GetProcessPointName failed: ProcessPoint can not be used, impl is nullptr.");
    return nullptr;
  }
  return impl_->GetProcessPointName();
}

void ProcessPoint::SetCompileConfigFile(const char_t *json_file_path) {
  if (impl_ == nullptr) {
    GELOGE(GRAPH_PARAM_INVALID, "[Check][Param] ProcessPointImpl is nullptr, check failed");
    REPORT_INNER_ERR_MSG("E18888", "SetCompileConfigFile failed: ProcessPoint can not be used, impl is nullptr.");
    return;
  }
  if (json_file_path == nullptr) {
    GELOGE(GRAPH_PARAM_INVALID, "[Check][Param] ProcessPoint(%s)'s compile config json is nullptr.",
           this->GetProcessPointName());
    REPORT_INNER_ERR_MSG("E18888",
                       "SetCompileConfigFile failed: [Check][Param] ProcessPoint(%s)'s compile config json is nullptr.",
                       this->GetProcessPointName());
    return;
  }
  return impl_->SetCompileConfig(json_file_path);
}

const char_t *ProcessPoint::GetCompileConfig() const {
  if (impl_ == nullptr) {
    GELOGE(GRAPH_FAILED, "[Check][Param] ProcessPointImpl is nullptr, check failed");
    return nullptr;
  }
  return impl_->GetCompileConfig();
}

class GraphPpImpl {
public:
  GraphPpImpl(const char_t *pp_name, const GraphBuilder &builder) : pp_name_(pp_name), builder_(builder) {}
  ~GraphPpImpl() = default;

  GraphBuilder GetGraphBuilder() const {
    auto builder = builder_;
    auto pp_name = pp_name_;
    GraphBuilder graph_builder = [builder, pp_name]() {
      auto graph = builder();
      auto compute_graph = ge::GraphUtilsEx::GetComputeGraph(graph);
      if (compute_graph == nullptr) {
        GELOGE(GRAPH_FAILED, "graph is invalid.");
        return graph;
      }
      compute_graph->SetName(pp_name);
      return graph;
    };
    return graph_builder;
  }
private:
  std::string pp_name_;
  GraphBuilder builder_;
};

GraphPp::GraphPp(const char_t *pp_name, const GraphBuilder &builder) : ProcessPoint(pp_name, ProcessPointType::GRAPH) {
  if (pp_name == nullptr) {
    GELOGE(GRAPH_FAILED, "GraphPp pp_name is null.");
    impl_ = nullptr;
  } else if (builder == nullptr) {
    GELOGE(GRAPH_FAILED, "GraphPp(%s) graph builder is null.", pp_name);
    impl_ = nullptr;
  } else {
    impl_ = MakeShared<GraphPpImpl>(pp_name, builder);
    if (impl_ == nullptr) {
      GELOGE(GRAPH_FAILED, "GraphPpImpl make shared failed.");
    }
  }
}
GraphPp::~GraphPp() = default;

GraphPp &GraphPp::SetCompileConfig(const char_t *json_file_path) {
  if (impl_ == nullptr) {
    GELOGE(GRAPH_FAILED, "[Check][Param] GraphPpImpl is nullptr, check failed");
    REPORT_INNER_ERR_MSG("E18888", "SetCompileConfig failed: GraphPp can not be used, impl is nullptr.");
    return *this;
  }
  if (json_file_path == nullptr) {
    GELOGE(GRAPH_FAILED, "[Check][Param] ProcessPoint(%s)'s compile config json is nullptr.",
           this->GetProcessPointName());
    REPORT_INNER_ERR_MSG("E18888",
                       "SetCompileConfig failed: [Check][Param] ProcessPoint(%s)'s compile config json is nullptr.",
                       this->GetProcessPointName());
    return *this;
  }

  ProcessPoint::SetCompileConfigFile(json_file_path);
  GELOGI("SetCompileConfig, json_file_path=%s.", json_file_path);
  return *this;
}

void GraphPp::Serialize(ge::AscendString &str) const {
  if (impl_ == nullptr) {
    GELOGE(GRAPH_PARAM_INVALID, "[Check][Param] GraphPpImpl is nullptr, check failed");
    REPORT_INNER_ERR_MSG("E18888", "Serialize failed: GraphPp can not be used, impl is nullptr.");
    return;
  }
  dataflow::ProcessPoint process_point;
  process_point.set_name(this->GetProcessPointName());
  process_point.set_type(dataflow::ProcessPoint_ProcessPointType_GRAPH);
  process_point.set_compile_cfg_file(this->GetCompileConfig());
  process_point.add_graphs(this->GetProcessPointName());
  std::string target_str;
  process_point.SerializeToString(&target_str);
  str = ge::AscendString(target_str.c_str(), target_str.length());
  return;
}

GraphBuilder GraphPp::GetGraphBuilder() const {
  if (impl_ == nullptr) {
    GELOGE(GRAPH_FAILED, "[Check][Param] GraphPpImpl is nullptr, check failed");
    REPORT_INNER_ERR_MSG("E18888", "GetGraphBuilder failed: GraphPp can not be used, impl is nullptr.");
    return nullptr;
  }
  return impl_->GetGraphBuilder();
}

class FlowGraphPpImpl {
public:
  FlowGraphPpImpl(const char_t *pp_name, const FlowGraphBuilder &builder) : pp_name_(pp_name), builder_(builder) {}
  ~FlowGraphPpImpl() = default;

  GraphBuilder GetGraphBuilder() const {
    auto builder = builder_;
    auto pp_name = pp_name_;
    GraphBuilder graph_build = [builder, pp_name]() {
      auto flow_graph = builder();
      const auto &graph = flow_graph.ToGeGraph();
      auto compute_graph = ge::GraphUtilsEx::GetComputeGraph(graph);
      if (compute_graph == nullptr) {
        GELOGE(GRAPH_FAILED, "graph is invalid.");
        return graph;
      }
      compute_graph->SetName(pp_name);
      return graph;
    };
    return graph_build;
  }
private:
  std::string pp_name_;
  FlowGraphBuilder builder_;
};

FlowGraphPp::FlowGraphPp(const char_t *pp_name, const FlowGraphBuilder &builder)
    : ProcessPoint(pp_name, ProcessPointType::FLOW_GRAPH) {
  if (pp_name == nullptr) {
    GELOGE(GRAPH_FAILED, "FlowGraphPp pp_name is null.");
    impl_ = nullptr;
  } else if (builder == nullptr) {
    GELOGE(GRAPH_FAILED, "FlowGraphPp(%s) graph builder is null.", pp_name);
    impl_ = nullptr;
  } else {
    impl_ = MakeShared<FlowGraphPpImpl>(pp_name, builder);
    if (impl_ == nullptr) {
      GELOGE(GRAPH_FAILED, "FlowGraphPpImpl make shared failed.");
    }
  }
}

FlowGraphPp::~FlowGraphPp() = default;

FlowGraphPp &FlowGraphPp::SetCompileConfig(const char_t *json_file_path) {
  if (impl_ == nullptr) {
    GELOGE(GRAPH_FAILED, "[Check][Param] FlowGraphPpImpl is nullptr, check failed");
    REPORT_INNER_ERR_MSG("E18888", "SetCompileConfig failed: FlowGraphPp can not be used, impl is nullptr.");
    return *this;
  }
  if (json_file_path == nullptr) {
    GELOGE(GRAPH_FAILED, "[Check][Param] ProcessPoint(%s)'s compile config json is nullptr.",
           this->GetProcessPointName());
    REPORT_INNER_ERR_MSG("E18888",
                       "SetCompileConfig failed: [Check][Param] ProcessPoint(%s)'s compile config json is nullptr.",
                       this->GetProcessPointName());
    return *this;
  }

  ProcessPoint::SetCompileConfigFile(json_file_path);
  GELOGI("SetCompileConfig, json_file_path=%s.", json_file_path);
  return *this;
}

void FlowGraphPp::Serialize(ge::AscendString &str) const {
  if (impl_ == nullptr) {
    GELOGE(GRAPH_PARAM_INVALID, "[Check][Param] FlowGraphPpImpl is nullptr, check failed");
    REPORT_INNER_ERR_MSG("E18888", "Serialize failed: FlowGraphPpImpl can not be used, impl is nullptr.");
    return;
  }
  dataflow::ProcessPoint process_point;
  process_point.set_name(this->GetProcessPointName());
  process_point.set_type(dataflow::ProcessPoint_ProcessPointType_FLOW_GRAPH);
  process_point.set_compile_cfg_file(this->GetCompileConfig());
  process_point.add_graphs(this->GetProcessPointName());
  std::string target_str;
  process_point.SerializeToString(&target_str);
  str = ge::AscendString(target_str.c_str(), target_str.length());
  return;
}

GraphBuilder FlowGraphPp::GetGraphBuilder() const {
  if (impl_ == nullptr) {
    GELOGE(GRAPH_FAILED, "[Check][Param] FlowGraphPpImpl is nullptr, check failed");
    REPORT_INNER_ERR_MSG("E18888", "GetGraphBuilder failed: FlowGraphPp can not be used, impl is nullptr.");
    return nullptr;
  }
  return impl_->GetGraphBuilder();
}

class FunctionPpImpl {
public:
  FunctionPpImpl() = default;
  ~FunctionPpImpl() = default;

  graphStatus AddInvokedClosure(const char_t *name, const GraphPp &graph_pp) {
    const graphStatus check_ret = CheckInvokeName(name);
    if (check_ret != GRAPH_SUCCESS) {
      GELOGE(check_ret, "[Check][Param] check invoke name failed.");
      return check_ret;
    }
    if (graph_pp.GetGraphBuilder() == nullptr) {
      GELOGE(GRAPH_PARAM_INVALID, "[Check][Param] AddInvokedClosure failed for graphpp graph builder is nullptr.");
      return GRAPH_PARAM_INVALID;
    }
    (void) invoked_closures_.emplace(name, graph_pp);
    GELOGI("AddInvokedClosure key(%s), pp name(%s).", name, graph_pp.GetProcessPointName());
    return GRAPH_SUCCESS;
  }

  graphStatus AddInvokedClosure(const char_t *name, const FlowGraphPp &flow_graph_pp) {
    const graphStatus check_ret = CheckInvokeName(name);
    if (check_ret != GRAPH_SUCCESS) {
      GELOGE(check_ret, "[Check][Param] check invoke name failed.");
      return check_ret;
    }
    if (flow_graph_pp.GetGraphBuilder() == nullptr) {
      GELOGE(GRAPH_PARAM_INVALID, "[Check][Param] AddInvokedClosure failed for flowgraphpp graph builder is nullptr.");
      return GRAPH_PARAM_INVALID;
    }
    (void) invoked_flow_closures_.emplace(name, flow_graph_pp);
    GELOGI("AddInvokedClosure key(%s), pp name(%s).", name, flow_graph_pp.GetProcessPointName());
    return GRAPH_SUCCESS;
  }

  graphStatus AddInvokedClosure(const char_t *name, const ProcessPoint &pp) {
    const graphStatus check_ret = CheckInvokeName(name);
    if (check_ret != GRAPH_SUCCESS) {
      GELOGE(check_ret, "[Check][Param] check invoke name failed.");
      return check_ret;
    }
    if (pp.GetProcessPointType() == ProcessPointType::GRAPH) {
      try {
        const auto &graph_pp = dynamic_cast<const GraphPp &>(pp);
        return AddInvokedClosure(name, graph_pp);
      } catch (const std::exception &e) {
        GELOGE(GRAPH_PARAM_INVALID, "ProcessPointType is Graph %s, but dynamic_cast to GraphPP exception, error=%s.", name,
               e.what());
        return GRAPH_PARAM_INVALID;
      }
    }

    if (pp.GetProcessPointType() == ProcessPointType::FLOW_GRAPH) {
      try {
        const auto &flow_graph_pp = dynamic_cast<const FlowGraphPp &>(pp);
        return AddInvokedClosure(name, flow_graph_pp);
      } catch (const std::exception &e) {
        GELOGE(GRAPH_PARAM_INVALID, "ProcessPointType is FlowGraph %s, but dynamic_cast to FlowGraphPP exception, error=%s.",
               name, e.what());
        return GRAPH_PARAM_INVALID;
      }
    }

    if (pp.GetProcessPointType() != ProcessPointType::INNER) {
      GELOGE(GRAPH_PARAM_INVALID, "AddInvokedClosure failed, as ProcessPointType=%d is not support.",
             static_cast<int32_t>(pp.GetProcessPointType()));
      return GRAPH_PARAM_INVALID;
    }
    ge::AscendString serialize_str;
    pp.Serialize(serialize_str);
    other_invoked_closures_[name] = std::string(serialize_str.GetString(), serialize_str.GetLength());
    GELOGI("AddInvokedClosure key(%s), pp name(%s), type(%d).", name, pp.GetProcessPointName(),
           static_cast<int32_t>(pp.GetProcessPointType()));
    return GRAPH_SUCCESS;
  }

  const std::map<const std::string, const GraphPp> &GetInvokedClosures() const {
    return invoked_closures_;
  }

  const std::map<const std::string, const FlowGraphPp> &GetInvokedFlowClosures() const {
    return invoked_flow_closures_;
  }

  template<typename T>
  bool SetAttrValue(const char_t *name, T &&value) {
    if (name == nullptr) {
      GELOGE(GRAPH_FAILED, "name is null.");
      return false;
    }
    return attrs_.SetByName(name, std::forward<T>(value));
  }

  const ge::AttrStore &GetAttrMap() const {
    return attrs_;
  }

  void UpdataProcessPoint(dataflow::ProcessPoint &process_point) {
    AddInvokedPps(process_point);
    AddFunctionPpInitPara(process_point);
  }
private:
  void AddInvokedPps(dataflow::ProcessPoint &process_point);
  void AddFunctionPpInitPara(dataflow::ProcessPoint &process_point);
  graphStatus CheckInvokeName(const char_t *name) const;
  ge::AttrStore attrs_;
  std::map<const std::string, const GraphPp> invoked_closures_;
  std::map<const std::string, const FlowGraphPp> invoked_flow_closures_;
  std::map<std::string, std::string> other_invoked_closures_;
};

graphStatus FunctionPpImpl::CheckInvokeName(const char_t *name) const {
  if (name == nullptr) {
    GELOGE(GRAPH_PARAM_INVALID, "[Check][Param] check invoke name failed as name is nullptr.");
    return GRAPH_PARAM_INVALID;
  }
  if ((invoked_closures_.find(name) != invoked_closures_.end()) ||
      (invoked_flow_closures_.find(name) != invoked_flow_closures_.end()) ||
      (other_invoked_closures_.find(name) != other_invoked_closures_.end())) {
    GELOGE(GRAPH_PARAM_INVALID, "check invoke name[%s] failed as duplicate.", name);
    return GRAPH_PARAM_INVALID;
  }
  return GRAPH_SUCCESS;
}

void FunctionPpImpl::AddInvokedPps(dataflow::ProcessPoint &process_point) {
  auto invoke_pps = process_point.mutable_invoke_pps();
  GE_RT_VOID_CHECK_NOTNULL(invoke_pps);
  for (auto iter = invoked_closures_.cbegin(); iter != invoked_closures_.cend(); ++iter) {
    const GraphPp &graph_pp = iter->second;
    dataflow::ProcessPoint invoked_pp;
    invoked_pp.set_name(graph_pp.GetProcessPointName());
    invoked_pp.set_type(dataflow::ProcessPoint_ProcessPointType_GRAPH);
    invoked_pp.set_compile_cfg_file(graph_pp.GetCompileConfig());
    const auto builder = graph_pp.GetGraphBuilder();
    invoked_pp.add_graphs(graph_pp.GetProcessPointName());
    (*invoke_pps)[iter->first] = std::move(invoked_pp);
    GELOGI("Add invoke graph pp success. key:%s, invoked pp name:%s", (iter->first).c_str(),
           graph_pp.GetProcessPointName());
  }
  for (auto iter = invoked_flow_closures_.cbegin(); iter != invoked_flow_closures_.cend(); ++iter) {
    const FlowGraphPp &flow_graph_pp = iter->second;
    dataflow::ProcessPoint invoked_pp;
    invoked_pp.set_name(flow_graph_pp.GetProcessPointName());
    invoked_pp.set_type(dataflow::ProcessPoint_ProcessPointType_FLOW_GRAPH);
    invoked_pp.set_compile_cfg_file(flow_graph_pp.GetCompileConfig());
    invoked_pp.add_graphs(flow_graph_pp.GetProcessPointName());
    (*invoke_pps)[iter->first] = std::move(invoked_pp);
    GELOGI("Add invoke graph pp success. key:%s, invoked pp name:%s", (iter->first).c_str(),
           flow_graph_pp.GetProcessPointName());
  }
  for (const auto &other_invoked_closure : other_invoked_closures_) {
    const auto &serialize_str = other_invoked_closure.second;
    dataflow::ProcessPoint invoked_pp;
    if (!invoked_pp.ParseFromString(serialize_str)) {
      GELOGE(GRAPH_FAILED, "parse process point failed, key:%s.", other_invoked_closure.first.c_str());
      return;
    }
    (*invoke_pps)[other_invoked_closure.first] = std::move(invoked_pp);
    GELOGI("Add invoke pp success. key:%s", other_invoked_closure.first.c_str());
  }
  return;
}

void FunctionPpImpl::AddFunctionPpInitPara(dataflow::ProcessPoint &process_point) {
  auto init_attr = process_point.mutable_attrs();
  const auto attrs = attrs_.GetAllAttrs();
  for (const auto &attr : attrs) {
    const AnyValue attr_value = attr.second;
    const auto serializer = AttrSerializerRegistry::GetInstance().GetSerializer(attr_value.GetValueTypeId());
    GE_RT_VOID_CHECK_NOTNULL(serializer);

    proto::AttrDef attr_def;
    if (serializer->Serialize(attr_value, attr_def) != GRAPH_SUCCESS) {
      GELOGE(GRAPH_FAILED, "Attr serialized failed, name:[%s].", attr.first.c_str());
      return;
    }
    (*init_attr)[attr.first] = attr_def;
  }
  return;
}

FunctionPp::FunctionPp(const char_t *pp_name) : ProcessPoint(pp_name, ProcessPointType::FUNCTION) {
  if (pp_name == nullptr) {
    GELOGE(GRAPH_FAILED, "[Check][Param] pp_name is nullptr.");
    impl_ = nullptr;
  } else {
    impl_ = MakeShared<FunctionPpImpl>();
    if (impl_ == nullptr) {
      GELOGE(GRAPH_FAILED, "FunctionPpImpl make shared failed.");
    }
  }
}
FunctionPp::~FunctionPp() = default;

FunctionPp &FunctionPp::SetCompileConfig(const char_t *json_file_path) {
  if (impl_ == nullptr) {
    GELOGE(GRAPH_FAILED, "[Check][Param] FunctionPpImpl is nullptr, check failed.");
    REPORT_INNER_ERR_MSG("E18888", "SetCompileConfig failed: FunctionPp can not be used, impl is nullptr.");
    return *this;
  }
  if (json_file_path == nullptr) {
    GELOGE(GRAPH_FAILED, "[Check][Param] ProcessPoint(%s)'s compile config json is nullptr.",
           this->GetProcessPointName());
    REPORT_INNER_ERR_MSG("E18888",
                       "SetCompileConfig failed: [Check][Param] ProcessPoint(%s)'s compile config json is nullptr.",
                       this->GetProcessPointName());
    return *this;
  }

  ProcessPoint::SetCompileConfigFile(json_file_path);
  GELOGI("SetCompileConfig, json_file_path=%s.", json_file_path);
  return *this;
}

FunctionPp &FunctionPp::SetInitParam(const char_t *attr_name, const ge::AscendString &value) {
  if (impl_ == nullptr) {
    GELOGE(GRAPH_FAILED, "[Check][Param] FunctionPpImpl is nullptr, check failed.");
    REPORT_INNER_ERR_MSG("E18888", "SetInitParam failed: FunctionPp can not be used, impl is nullptr.");
    return *this;
  }

  std::string str_value(value.GetString());
  if (!impl_->SetAttrValue(attr_name, str_value)) {
    const std::string name = (attr_name == nullptr) ? "nullptr" : attr_name;
    GELOGE(GRAPH_FAILED, "set attr name(%s) failed.", name.c_str());
    REPORT_INNER_ERR_MSG("E18888", "SetInitParam failed: set attr name(%s) failed.", name.c_str());
  }

  return *this;
}

FunctionPp &FunctionPp::SetInitParam(const char_t *attr_name, const char_t *value) {
  if (impl_ == nullptr) {
    GELOGE(GRAPH_FAILED, "[Check][Param] FunctionPpImpl is nullptr, check failed.");
    REPORT_INNER_ERR_MSG("E18888", "SetInitParam failed: FunctionPp can not be used, impl is nullptr.");
    return *this;
  }

  if (value == nullptr) {
    GELOGE(GRAPH_FAILED, "[Check][Param] Set init param value is nullptr, check failed.");
    REPORT_INNER_ERR_MSG("E18888", "SetInitParam failed: [Check][Param] Set init param value is nullptr, check failed.");
    return *this;
  }

  std::string str_value(value);
  if (!impl_->SetAttrValue(attr_name, str_value)) {
    const std::string name = (attr_name == nullptr) ? "nullptr" : attr_name;
    GELOGE(GRAPH_FAILED, "set attr name(%s) failed.", name.c_str());
    REPORT_INNER_ERR_MSG("E18888", "SetInitParam failed: set attr name(%s) failed.", name.c_str());
  }

  return *this;
}

FunctionPp &FunctionPp::SetInitParam(const char_t *attr_name, const std::vector<ge::AscendString> &value) {
  if (impl_ == nullptr) {
    GELOGE(GRAPH_FAILED, "[Check][Param] FunctionPpImpl is nullptr, check failed.");
    REPORT_INNER_ERR_MSG("E18888", "SetInitParam failed: FunctionPp can not be used, impl is nullptr.");
    return *this;
  }

  std::vector<std::string> vec_value;
  for (auto iter = value.cbegin(); iter != value.cend(); iter++) {
    vec_value.emplace_back(iter->GetString());
  }
  if (!impl_->SetAttrValue(attr_name, vec_value)) {
    const std::string name = (attr_name == nullptr) ? "nullptr" : attr_name;
    GELOGE(GRAPH_FAILED, "set attr name(%s) failed.", name.c_str());
    REPORT_INNER_ERR_MSG("E18888", "SetInitParam failed: set attr name(%s) failed.", name.c_str());
  }

  return *this;
}

FunctionPp &FunctionPp::SetInitParam(const char_t *attr_name, const int64_t &value) {
  if (impl_ == nullptr) {
    GELOGE(GRAPH_FAILED, "[Check][Param] FunctionPpImpl is nullptr, check failed.");
    REPORT_INNER_ERR_MSG("E18888", "SetInitParam failed: FunctionPp can not be used, impl is nullptr.");
    return *this;
  }

  if (!impl_->SetAttrValue(attr_name, value)) {
    const std::string name = (attr_name == nullptr) ? "nullptr" : attr_name;
    GELOGE(GRAPH_FAILED, "set attr name(%s) failed.", name.c_str());
    REPORT_INNER_ERR_MSG("E18888", "SetInitParam failed: set attr name(%s) failed.", name.c_str());
  }

  return *this;
}

FunctionPp &FunctionPp::SetInitParam(const char_t *attr_name, const std::vector<int64_t> &value) {
  if (impl_ == nullptr) {
    GELOGE(GRAPH_FAILED, "[Check][Param] FunctionPpImpl is nullptr, check failed.");
    REPORT_INNER_ERR_MSG("E18888", "SetInitParam failed: FunctionPp can not be used, impl is nullptr.");
    return *this;
  }

  if (!impl_->SetAttrValue(attr_name, value)) {
    const std::string name = (attr_name == nullptr) ? "nullptr" : attr_name;
    GELOGE(GRAPH_FAILED, "set attr name(%s) failed.", name.c_str());
    REPORT_INNER_ERR_MSG("E18888", "SetInitParam failed: set attr name(%s) failed.", name.c_str());
  }

  return *this;
}

FunctionPp &FunctionPp::SetInitParam(const char_t *attr_name, const std::vector<std::vector<int64_t>> &value) {
  if (impl_ == nullptr) {
    GELOGE(GRAPH_FAILED, "[Check][Param] FunctionPpImpl is nullptr, check failed.");
    REPORT_INNER_ERR_MSG("E18888", "SetInitParam failed: FunctionPp can not be used, impl is nullptr.");
    return *this;
  }

  if (!impl_->SetAttrValue(attr_name, value)) {
    const std::string name = (attr_name == nullptr) ? "nullptr" : attr_name;
    GELOGE(GRAPH_FAILED, "set attr name(%s) failed.", name.c_str());
    REPORT_INNER_ERR_MSG("E18888", "SetInitParam failed: set attr name(%s) failed.", name.c_str());
  }

  return *this;
}

FunctionPp &FunctionPp::SetInitParam(const char_t *attr_name, const float &value) {
  if (impl_ == nullptr) {
    GELOGE(GRAPH_FAILED, "[Check][Param] FunctionPpImpl is nullptr, check failed.");
    REPORT_INNER_ERR_MSG("E18888", "SetInitParam failed: FunctionPp can not be used, impl is nullptr.");
    return *this;
  }

  if (!impl_->SetAttrValue(attr_name, value)) {
    const std::string name = (attr_name == nullptr) ? "nullptr" : attr_name;
    GELOGE(GRAPH_FAILED, "set attr name(%s) failed.", name.c_str());
    REPORT_INNER_ERR_MSG("E18888", "SetInitParam failed: set attr name(%s) failed.", name.c_str());
  }

  return *this;
}

FunctionPp &FunctionPp::SetInitParam(const char_t *attr_name, const std::vector<float> &value) {
  if (impl_ == nullptr) {
    GELOGE(GRAPH_FAILED, "[Check][Param] FunctionPpImpl is nullptr, check failed.");
    REPORT_INNER_ERR_MSG("E18888", "SetInitParam failed: FunctionPp can not be used, impl is nullptr.");
    return *this;
  }

  if (!impl_->SetAttrValue(attr_name, value)) {
    const std::string name = (attr_name == nullptr) ? "nullptr" : attr_name;
    GELOGE(GRAPH_FAILED, "set attr name(%s) failed.", name.c_str());
    REPORT_INNER_ERR_MSG("E18888", "SetInitParam failed: set attr name(%s) failed.", name.c_str());
  }

  return *this;
}

FunctionPp &FunctionPp::SetInitParam(const char_t *attr_name, const bool &value) {
  if (impl_ == nullptr) {
    GELOGE(GRAPH_FAILED, "[Check][Param] FunctionPpImpl is nullptr, check failed.");
    REPORT_INNER_ERR_MSG("E18888", "SetInitParam failed: FunctionPp can not be used, impl is nullptr.");
    return *this;
  }

  if (!impl_->SetAttrValue(attr_name, value)) {
    const std::string name = (attr_name == nullptr) ? "nullptr" : attr_name;
    GELOGE(GRAPH_FAILED, "set attr name(%s) failed.", name.c_str());
    REPORT_INNER_ERR_MSG("E18888", "SetInitParam failed: set attr name(%s) failed.", name.c_str());
  }

  return *this;
}

FunctionPp &FunctionPp::SetInitParam(const char_t *attr_name, const std::vector<bool> &value) {
  if (impl_ == nullptr) {
    GELOGE(GRAPH_FAILED, "[Check][Param] FunctionPpImpl is nullptr, check failed.");
    REPORT_INNER_ERR_MSG("E18888", "SetInitParam failed: FunctionPp can not be used, impl is nullptr.");
    return *this;
  }

  if (!impl_->SetAttrValue(attr_name, value)) {
    const std::string name = (attr_name == nullptr) ? "nullptr" : attr_name;
    GELOGE(GRAPH_FAILED, "set attr name(%s) failed.", name.c_str());
    REPORT_INNER_ERR_MSG("E18888", "SetInitParam failed: set attr name(%s) failed.", name.c_str());
  }

  return *this;
}

FunctionPp &FunctionPp::SetInitParam(const char_t *attr_name, const ge::DataType &value) {
  if (impl_ == nullptr) {
    GELOGE(GRAPH_FAILED, "[Check][Param] FunctionPpImpl is nullptr, check failed.");
    REPORT_INNER_ERR_MSG("E18888", "SetInitParam failed: FunctionPp can not be used, impl is nullptr.");
    return *this;
  }

  if (!impl_->SetAttrValue(attr_name, value)) {
    const std::string name = (attr_name == nullptr) ? "nullptr" : attr_name;
    GELOGE(GRAPH_FAILED, "set attr name(%s) failed.", name.c_str());
    REPORT_INNER_ERR_MSG("E18888", "SetInitParam failed: set attr name(%s) failed.", name.c_str());
  }

  return *this;
}

FunctionPp &FunctionPp::SetInitParam(const char_t *attr_name, const std::vector<ge::DataType> &value) {
  if (impl_ == nullptr) {
    GELOGE(GRAPH_FAILED, "[Check][Param] FunctionPpImpl is nullptr, check failed.");
    REPORT_INNER_ERR_MSG("E18888", "SetInitParam failed: FunctionPp can not be used, impl is nullptr.");
    return *this;
  }

  if (!impl_->SetAttrValue(attr_name, value)) {
    const std::string name = (attr_name == nullptr) ? "nullptr" : attr_name;
    GELOGE(GRAPH_FAILED, "set attr name(%s) failed.", name.c_str());
    REPORT_INNER_ERR_MSG("E18888", "SetInitParam failed: set attr name(%s) failed.", name.c_str());
  }

  return *this;
}

FunctionPp &FunctionPp::AddInvokedClosure(const char_t *name, const GraphPp &graph_pp) {
  if (impl_ == nullptr) {
    GELOGE(GRAPH_FAILED, "[Check][Param] FunctionPpImpl is nullptr, check failed.");
    REPORT_INNER_ERR_MSG("E18888", "AddInvokedClosure failed: FunctionPp can not be used, impl is nullptr.");
    return *this;
  }

  if (impl_->AddInvokedClosure(name, graph_pp) != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E18888", "AddInvokedClosure failed.");
  }
  return *this;
}

FunctionPp &FunctionPp::AddInvokedClosure(const char_t *name, const FlowGraphPp &graph_pp) {
  if (impl_ == nullptr) {
    GELOGE(GRAPH_FAILED, "[Check][Param] FunctionPpImpl is nullptr, check failed.");
    REPORT_INNER_ERR_MSG("E18888", "AddInvokedClosure failed: FunctionPp can not be used, impl is nullptr.");
    return *this;
  }

  if (impl_->AddInvokedClosure(name, graph_pp) != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E18888", "AddInvokedClosure failed.");
  }
  return *this;
}

FunctionPp &FunctionPp::AddInvokedClosure(const char_t *name, const ProcessPoint &pp) {
  if (impl_ == nullptr) {
    REPORT_INNER_ERR_MSG("E18888", "AddInvokedClosure failed: FunctionPp can not be used, impl is nullptr.");
    return *this;
  }

  if (impl_->AddInvokedClosure(name, pp) != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E18888", "AddInvokedClosure failed.");
  }
  return *this;
}

const std::map<const std::string, const GraphPp> &FunctionPp::GetInvokedClosures() const {
  if (impl_ == nullptr) {
    static std::map<const std::string, const GraphPp> empty_map;
    return empty_map;
  }

  return impl_->GetInvokedClosures();
}

void FunctionPp::Serialize(ge::AscendString &str) const {
  if (impl_ == nullptr) {
    GELOGE(GRAPH_PARAM_INVALID, "[Check][Param] FunctionPpImpl is nullptr, check failed");
    REPORT_INNER_ERR_MSG("E18888", "Serialize failed: FunctionPp can not be used, impl is nullptr.");
    return;
  }
  dataflow::ProcessPoint process_point;
  process_point.set_name(this->GetProcessPointName());
  process_point.set_type(dataflow::ProcessPoint_ProcessPointType_FUNCTION);
  process_point.set_compile_cfg_file(this->GetCompileConfig());
  impl_->UpdataProcessPoint(process_point);
  std::string target_str;
  process_point.SerializeToString(&target_str);
  str = ge::AscendString(target_str.c_str(), target_str.length());
  return;
}

const std::map<const ge::string, const GraphPp> &FlowGraphUtils::GetInvokedClosures(const FunctionPp *func_pp) {
  if (func_pp == nullptr || func_pp->impl_ == nullptr) {
    static std::map<const std::string, const GraphPp> empty_map;
    return empty_map;
  }
    return func_pp->impl_->GetInvokedClosures();
}

const std::map<const ge::string, const FlowGraphPp> &FlowGraphUtils::GetInvokedFlowClosures(const FunctionPp *func_pp) {
  if (func_pp == nullptr || func_pp->impl_ == nullptr) {
    static std::map<const std::string, const FlowGraphPp> empty_map;
    return empty_map;
  }
    return func_pp->impl_->GetInvokedFlowClosures();
}
}  // namespace dflow
}  // namespace ge
