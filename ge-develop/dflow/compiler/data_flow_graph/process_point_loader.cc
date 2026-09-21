/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dflow/compiler/data_flow_graph/process_point_loader.h"
#include <algorithm>
#include <openssl/sha.h>
#include <sstream>
#include "common/compile_profiling/ge_trace_wrapper.h"
#include "common/util/mem_utils.h"
#include "common/checker.h"
#include "framework/common/types.h"
#include "dflow/inc/data_flow/model/pne_model.h"
#include "graph/debug/ge_attr_define.h"
#include "dflow/flow_graph/data_flow_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_type_utils.h"
#include "graph/serialization/attr_serializer_registry.h"
#include "dflow/compiler/data_flow_graph/compile_config_json.h"
#include "dflow/compiler/data_flow_graph/data_flow_graph_utils.h"
#include "dflow/compiler/data_flow_graph/inner_pp_loader.h"
#include "api/aclgrph/option_utils.h"

namespace {
constexpr int64_t kHighestPriority = 0;
constexpr int64_t kLowestPriority = 2;
constexpr const char *kCpuNumResourceType = "cpu";
constexpr const char *kCpuNumAttrName = "__cpu_num";
constexpr const char *kPageTypeNormal = "normal";
constexpr const char *kPageTypeHuge = "huge";
constexpr const char *kBufferConfig = "_user_buf_cfg";
constexpr const size_t kMaxBufCfgNum = 64UL;
constexpr const uint32_t kNormalPageAtomicValue = 4 * 1024;
constexpr const uint32_t kHugePageAtomicValue = 2 * 1024 * 1024;
constexpr const uint32_t kMaxBlkSize = 2 * 1024 * 1024;
constexpr uint32_t kMaxPpNameLen = 128U;
}  // namespace

namespace ge {
constexpr const char *ATTR_NAME_DATA_FLOW_COMPILER_RESULT = "_dflow_compiler_result";
constexpr const char *ATTR_NAME_DATA_FLOW_RUNNING_RESOURCE_INFO = "_dflow_running_resource_info";
constexpr const char *ATTR_NAME_DATA_FLOW_RUNNABLE_RESOURCE = "_dflow_runnable_resource";
constexpr const char *ATTR_NAME_DATA_FLOW_HEAVY_LOAD = "_dflow_heavy_load";
constexpr const char *ATTR_NAME_DATA_FLOW_VISIBLE_DEVICE_ENABLE = "_dflow_visible_device_enable";
constexpr const char *ATTR_NAME_DATA_FLOW_DATA_FLOW_SCOPE = "_dflow_data_flow_scope";
constexpr const char_t *ATTR_NAME_DATA_FLOW_DATA_FLOW_INVOKED_SCOPES = "_dflow_data_flow_invoked_scopes";


Status ProcessPointLoader::LoadProcessPoint(const dataflow::ProcessPoint &process_point,
                                            DataFlowGraph &data_flow_graph, const NodePtr &node) {
  GE_CHECK_NOTNULL(data_flow_graph.root_graph_);
  GE_CHECK_NOTNULL(node);
  GELOGI("load process point[%s] on node[%s], type=%d", process_point.name().c_str(), node->GetNamePtr(),
         process_point.type());
  if (process_point.name().length() > kMaxPpNameLen) {
    GELOGE(PARAM_INVALID, "node[%s] process point name[%s] length is too long, over %u.", node->GetName().c_str(),
           process_point.name().c_str(), kMaxPpNameLen);
    return PARAM_INVALID;
  }
  if (process_point.type() == dataflow::ProcessPoint_ProcessPointType_FUNCTION) {
    return LoadFunctionProcessPoint(process_point, data_flow_graph, node);
  } else if (process_point.type() == dataflow::ProcessPoint_ProcessPointType_GRAPH ||
             process_point.type() == dataflow::ProcessPoint_ProcessPointType_FLOW_GRAPH) {
    return LoadGraphProcessPoint(process_point, data_flow_graph, node);
  } else if (process_point.type() == dataflow::ProcessPoint_ProcessPointType_INNER) {
    // inner pp loader.
    return InnerPpLoader::LoadProcessPoint(process_point, data_flow_graph, node);
  } else {
    GELOGE(FAILED, "The process point type[%d] is not supported.", process_point.type());
  }
  return FAILED;
}

Status ProcessPointLoader::SetAttrBinPathForFlowFunc(const CompileConfigJson::FunctionPpConfig &func_pp_cfg,
                                                     OpDescPtr &flow_func_desc) {
  std::string bin_dir = func_pp_cfg.workspace;
  std::string bin_name = func_pp_cfg.target_bin;
  std::string bin_path = bin_dir + "/" + bin_name;
  GE_CHK_BOOL_RET_STATUS(AttrUtils::SetStr(flow_func_desc, "bin_path", bin_path), FAILED,
                         "Failed to set attr[bin_path], value[%s] for FlowFunc[%s].", bin_path.c_str(),
                         flow_func_desc->GetName().c_str());
  GELOGD("Set attr[bin_path], value[%s] for FlowFunc[%s] successfully.", bin_path.c_str(),
         flow_func_desc->GetName().c_str());
  return SUCCESS;
}

Status ProcessPointLoader::SetAttrFuncsForFlowFunc(const CompileConfigJson::FunctionPpConfig &func_pp_cfg,
                                                   OpDescPtr &flow_func_desc) {
  std::vector<NamedAttrs> funcs_attr;
  for (const auto &func : func_pp_cfg.funcs) {
    NamedAttrs func_attr;
    func_attr.SetName(func.func_name);
    std::vector<int64_t> inputs_idx;
    std::transform(func.inputs_index.cbegin(), func.inputs_index.cend(), std::back_inserter(inputs_idx),
                   [](const uint32_t a) { return static_cast<int64_t>(a); });
    GE_CHK_STATUS_RET(func_attr.SetAttr(dflow::ATTR_NAME_FLOW_FUNC_FUNC_INPUTS_INDEX, AnyValue::CreateFrom(inputs_idx)),
                      "Failed to set attr[%s] for FlowFunc[%s] attr[%s]", dflow::ATTR_NAME_FLOW_FUNC_FUNC_INPUTS_INDEX,
                      flow_func_desc->GetName().c_str(), dflow::ATTR_NAME_FLOW_FUNC_FUNC_LIST);
    std::vector<int64_t> outputs_idx;
    std::transform(func.outputs_index.cbegin(), func.outputs_index.cend(), std::back_inserter(outputs_idx),
                   [](const uint32_t a) { return static_cast<int64_t>(a); });
    GE_CHK_STATUS_RET(
        func_attr.SetAttr(dflow::ATTR_NAME_FLOW_FUNC_FUNC_OUTPUTS_INDEX, AnyValue::CreateFrom(outputs_idx)),
        "Failed to set attr[%s] for FlowFunc[%s] attr[%s]", dflow::ATTR_NAME_FLOW_FUNC_FUNC_OUTPUTS_INDEX,
        flow_func_desc->GetName().c_str(), dflow::ATTR_NAME_FLOW_FUNC_FUNC_LIST);
    GE_CHK_STATUS_RET(
        func_attr.SetAttr(dflow::ATTR_NAME_FLOW_FUNC_FUNC_STREAM_INPUT, AnyValue::CreateFrom(func.stream_input)),
        "Failed to set attr[%s] for FlowFunc[%s] attr[%s]", dflow::ATTR_NAME_FLOW_FUNC_FUNC_STREAM_INPUT,
        flow_func_desc->GetName().c_str(), dflow::ATTR_NAME_FLOW_FUNC_FUNC_LIST);
    funcs_attr.emplace_back(func_attr);
  }
  GE_CHK_BOOL_RET_STATUS(AttrUtils::SetListNamedAttrs(flow_func_desc, dflow::ATTR_NAME_FLOW_FUNC_FUNC_LIST, funcs_attr),
                         FAILED, "Failed to set attr[%s] for FlowFunc[%s].", dflow::ATTR_NAME_FLOW_FUNC_FUNC_LIST,
                         flow_func_desc->GetName().c_str());
  GELOGD("Set attr[%s] for FlowFunc[%s] successfully.", dflow::ATTR_NAME_FLOW_FUNC_FUNC_LIST,
         flow_func_desc->GetName().c_str());
  return SUCCESS;
}

Status ProcessPointLoader::SetCpuNumForFlowFunc(const CompileConfigJson::FunctionPpConfig &func_pp_cfg,
                                                OpDescPtr &flow_func_desc) {
  uint32_t cpu_num = 1U;
  bool is_cpu_num_set = false;
  for (const auto &running_resource_info : func_pp_cfg.running_resources_info) {
    if (running_resource_info.resource_type == kCpuNumResourceType) {
      cpu_num = running_resource_info.resource_num;
      is_cpu_num_set = true;
      break;
    }
  }
  if (is_cpu_num_set) {
    GE_CHK_BOOL_RET_STATUS(AttrUtils::SetInt(flow_func_desc, kCpuNumAttrName, static_cast<int64_t>(cpu_num)), FAILED,
                           "Failed to set attr[%s], value[%u] for FlowFunc[%s].", kCpuNumAttrName, cpu_num,
                           flow_func_desc->GetName().c_str());
    GELOGD("Set attr[%s], value[%u] for FlowFunc[%s] successfully.", kCpuNumAttrName, cpu_num,
           flow_func_desc->GetName().c_str());
  }
  return SUCCESS;
}

Status ProcessPointLoader::SetAttrFuncsForFlowFunc(const dataflow::ProcessPoint &process_point,
                                                   OpDescPtr &flow_func_desc) {
  if (process_point.funcs().empty()) {
    return SUCCESS;
  }
  GE_CHK_BOOL_RET_STATUS(AttrUtils::SetStr(flow_func_desc, "func_name", process_point.funcs(0).name()), FAILED,
                         "Failed to set func_name attr to op[%s].", flow_func_desc->GetName().c_str());
  std::vector<NamedAttrs> funcs_attr;
  for (const auto &func : process_point.funcs()) {
    NamedAttrs func_attr;
    func_attr.SetName(func.name());
    std::vector<int64_t> inputs_idx;
    std::transform(func.in_index().cbegin(), func.in_index().cend(), std::back_inserter(inputs_idx),
                   [](const uint32_t a) { return static_cast<int64_t>(a); });
    GE_CHK_STATUS_RET(func_attr.SetAttr(dflow::ATTR_NAME_FLOW_FUNC_FUNC_INPUTS_INDEX, AnyValue::CreateFrom(inputs_idx)),
                      "Failed to set attr[%s] for FlowFunc[%s] attr[%s]", dflow::ATTR_NAME_FLOW_FUNC_FUNC_INPUTS_INDEX,
                      flow_func_desc->GetName().c_str(), dflow::ATTR_NAME_FLOW_FUNC_FUNC_LIST);
    std::vector<int64_t> outputs_idx;
    std::transform(func.out_index().cbegin(), func.out_index().cend(), std::back_inserter(outputs_idx),
                   [](const uint32_t a) { return static_cast<int64_t>(a); });
    GE_CHK_STATUS_RET(
        func_attr.SetAttr(dflow::ATTR_NAME_FLOW_FUNC_FUNC_OUTPUTS_INDEX, AnyValue::CreateFrom(outputs_idx)),
        "Failed to set attr[%s] for FlowFunc[%s] attr[%s]", dflow::ATTR_NAME_FLOW_FUNC_FUNC_OUTPUTS_INDEX,
        flow_func_desc->GetName().c_str(), dflow::ATTR_NAME_FLOW_FUNC_FUNC_LIST);
    funcs_attr.emplace_back(func_attr);
  }
  GE_CHK_BOOL_RET_STATUS(AttrUtils::SetListNamedAttrs(flow_func_desc, dflow::ATTR_NAME_FLOW_FUNC_FUNC_LIST, funcs_attr),
                         FAILED, "Failed to set attr[%s] for FlowFunc[%s].", dflow::ATTR_NAME_FLOW_FUNC_FUNC_LIST,
                         flow_func_desc->GetName().c_str());
  GELOGD("Set attr[%s] for FlowFunc[%s] successfully.", dflow::ATTR_NAME_FLOW_FUNC_FUNC_LIST,
         flow_func_desc->GetName().c_str());
  return SUCCESS;
}

Status ProcessPointLoader::SetCustomizedAttrsForFlowFunc(const dataflow::ProcessPoint &process_point,
                                                         OpDescPtr &flow_func_desc) {
  for (const auto &it : process_point.attrs()) {
    const auto deserializer = AttrSerializerRegistry::GetInstance().GetDeserializer(it.second.value_case());
    GE_CHK_BOOL_RET_STATUS(deserializer != nullptr, FAILED, "Get deserialize failed, attr type:[%d]",
                           static_cast<int32_t>(it.second.value_case()));
    AnyValue attr_value;
    GE_CHK_STATUS_RET(deserializer->Deserialize(it.second, attr_value), "Attr[%s] deserialized failed.",
                      it.first.c_str());
    GE_CHK_STATUS_RET(flow_func_desc->SetAttr(it.first, attr_value), "Failed to set attr[%s] for FlowFunc[%s].",
                      it.first.c_str(), flow_func_desc->GetName().c_str());
  }
  GELOGD("Set customized attrs for FlowFunc[%s] successfully.", flow_func_desc->GetName().c_str());
  return SUCCESS;
}

Status ProcessPointLoader::CreateFlowFuncOpDescFromProcessPoint(const dataflow::ProcessPoint &process_point,
                                                                const CompileConfigJson::FunctionPpConfig &func_pp_cfg,
                                                                OpDescPtr &op_desc) {
  GE_CHK_STATUS_RET(DataFlowGraphUtils::CreateFlowFuncOpDesc(process_point.name(), func_pp_cfg.input_num,
                                                             func_pp_cfg.output_num, op_desc),
                    "Failed create FlowFunc op desc for process point[%s], inputs num [%zu], outputs num [%zu].",
                    process_point.name().c_str(), func_pp_cfg.input_num, func_pp_cfg.output_num);
  GE_CHK_STATUS_RET_NOLOG(SetAttrBinPathForFlowFunc(func_pp_cfg, op_desc));
  GE_CHK_STATUS_RET_NOLOG(SetAttrFuncsForFlowFunc(func_pp_cfg, op_desc));
  GE_CHK_STATUS_RET_NOLOG(SetCpuNumForFlowFunc(func_pp_cfg, op_desc));
  GE_CHK_STATUS_RET_NOLOG(SetCustomizedAttrsForFlowFunc(process_point, op_desc));
  GELOGD("Create FlowFunc[%s] successfully.", op_desc->GetName().c_str());
  return SUCCESS;
}

Status ProcessPointLoader::AddInputsForFlowFunc(NodePtr &flow_func_node, ComputeGraphPtr &compute_graph) {
  for (size_t i = 0; i < flow_func_node->GetOpDesc()->GetInputsSize(); ++i) {
    const auto data_op = MakeShared<OpDesc>(flow_func_node->GetName() + "_" + DATA + "_" + std::to_string(i), DATA);
    GE_CHECK_NOTNULL(data_op);
    GeTensorDesc tensor_desc;
    GE_CHK_STATUS_RET(data_op->AddInputDesc(tensor_desc), "Failed to add input desc to op[%s].",
                      data_op->GetName().c_str());
    GE_CHK_STATUS_RET(data_op->AddOutputDesc(tensor_desc), "Failed to add output desc to op[%s].",
                      data_op->GetName().c_str());
    GE_CHK_BOOL_RET_STATUS(AttrUtils::SetInt(data_op, ATTR_NAME_INDEX, i), FAILED,
                           "Failed to set attr[%s] for data node[%s].", ATTR_NAME_INDEX.c_str(),
                           data_op->GetName().c_str());
    auto data_node = compute_graph->AddNode(data_op);
    GE_CHECK_NOTNULL(data_node);
    GE_CHK_STATUS_RET(GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), flow_func_node->GetInDataAnchor(i)),
                      "Failed to add edge from node[%s] output[0] to node[%s] input[%zu]", data_node->GetName().c_str(),
                      flow_func_node->GetName().c_str(), i);
    (void)compute_graph->AddInputNode(data_node);
  }
  GELOGD("Add inputs for FlowFunc[%s] successfully.", flow_func_node->GetName().c_str());
  return SUCCESS;
}

Status ProcessPointLoader::AddOutputsForFlowFunc(NodePtr &flow_func_node, ComputeGraphPtr &compute_graph) {
  const auto net_output_op = MakeShared<OpDesc>(flow_func_node->GetName() + "_" + NETOUTPUT, NETOUTPUT);
  GE_CHECK_NOTNULL(net_output_op);
  for (size_t i = 0; i < flow_func_node->GetOpDesc()->GetOutputsSize(); ++i) {
    GeTensorDesc tensor_desc;
    GE_CHK_STATUS_RET(net_output_op->AddInputDesc(tensor_desc), "Failed to add input desc to op[%s].",
                      net_output_op->GetName().c_str());
  }
  auto net_output_node = compute_graph->AddNode(net_output_op);
  GE_CHECK_NOTNULL(net_output_node);
  for (size_t i = 0; i < flow_func_node->GetOpDesc()->GetOutputsSize(); ++i) {
    GE_CHK_STATUS_RET(GraphUtils::AddEdge(flow_func_node->GetOutDataAnchor(i), net_output_node->GetInDataAnchor(i)),
                      "Failed to add edge from node[%s] output[%zu] to node[%s] input[%zu]",
                      flow_func_node->GetName().c_str(), i, net_output_node->GetName().c_str(), i);
    (void)compute_graph->AddOutputNodeByIndex(flow_func_node, i);
  }
  GELOGD("Add outputs for FlowFunc[%s] successfully.", flow_func_node->GetName().c_str());
  return SUCCESS;
}

Status ProcessPointLoader::LoadInvokedProcessPoint(const dataflow::ProcessPoint &process_point,
                                                   DataFlowGraph &data_flow_graph,
                                                   OpDescPtr &flow_func_desc,
                                                   const NodePtr &node,
                                                   bool is_built_in_flow_func) {
  auto invoke_pps = process_point.invoke_pps();
  const auto graph_name = data_flow_graph.IsRootDataFlow() ? process_point.name()
                              : data_flow_graph.GetDataFlowScope() + process_point.name();
  std::vector<std::string> invoked_scopes;
  for (const auto &it : invoke_pps) {
    const auto &invoke_key = it.first;
    const auto &invoke_process_point = it.second;
    if ((invoke_process_point.type() == dataflow::ProcessPoint_ProcessPointType_GRAPH) ||
        (invoke_process_point.type() == dataflow::ProcessPoint_ProcessPointType_INNER) ||
        (invoke_process_point.type() == dataflow::ProcessPoint_ProcessPointType_FLOW_GRAPH)) {
      GE_CHK_STATUS_RET(LoadProcessPoint(it.second, data_flow_graph, node),
                        "Failed to load invoked process point[%s] for process point[%s].",
                        it.second.name().c_str(), process_point.name().c_str());
    } else {
      GELOGE(FAILED, "process point[%s] type[%d] does not support to be invoked,invoke key=%s.",
             invoke_process_point.name().c_str(), invoke_process_point.type(), invoke_key.c_str());
      return FAILED;
    }
    if (invoke_process_point.type() == dataflow::ProcessPoint_ProcessPointType_FLOW_GRAPH) {
      const std::string invoked_scope = data_flow_graph.IsRootDataFlow() ? it.second.name() + "/" + it.first
                                       : data_flow_graph.GetDataFlowScope() + it.second.name() + "/" + it.first;
      invoked_scopes.emplace_back(invoked_scope);
    }
    std::map<std::string, std::string>::const_iterator find_ret = data_flow_graph.invoked_graphs_.find(it.first);
    if ((find_ret != data_flow_graph.invoked_graphs_.cend()) && (find_ret->second != it.second.name())) {
      GELOGE(FAILED, "invoked key[%s] is used by graph[%s], can't be used by graph[%s] again.", it.first.c_str(),
             find_ret->second.c_str(), it.second.name().c_str());
      return FAILED;
    }
    const auto invoke_name = data_flow_graph.IsRootDataFlow() ? it.first
                                                              : data_flow_graph.GetDataFlowScope() + it.first;
    data_flow_graph.invokes_[graph_name].emplace_back(invoke_name);
    data_flow_graph.invoked_keys_[it.second.name()] = invoke_name;
    data_flow_graph.invoked_graphs_[invoke_name] = it.second.name();
    data_flow_graph.invoked_by_built_in_[invoke_name] = is_built_in_flow_func;
    data_flow_graph.invoke_origins_[invoke_name] = invoke_key;
    GELOGD("Load invoked graph process point[%s] for process point[%s] successfully, invoke_key=%s.",
           it.second.name().c_str(), process_point.name().c_str(), invoke_name.c_str());
  }
  if (!data_flow_graph.invokes_[graph_name].empty()) {
    GE_CHK_BOOL_RET_STATUS(AttrUtils::SetListStr(flow_func_desc, dflow::ATTR_NAME_FLOW_FUNC_INVOKE_KEYS,
                                                 data_flow_graph.invokes_[graph_name]),
                           FAILED, "Failed to set attr[%s] for op[%s].", dflow::ATTR_NAME_FLOW_FUNC_INVOKE_KEYS,
                           flow_func_desc->GetName().c_str());
  }
  GE_CHK_STATUS_RET(SetDataFlowInvokedScopes(flow_func_desc, invoked_scopes),
                    "Failed to set data flow invoked scopes attr.");
  return SUCCESS;
}

Status ProcessPointLoader::SetCompileResultToNode(const std::string &pp_name,
                                                  const FunctionCompile::CompileResult &result,
                                                  const NodePtr &node) {
  const auto &node_name = node->GetName();
  const auto &op_desc = node->GetOpDesc();
  if ((result.running_resources_info.empty()) || (result.compile_bin_info.empty())) {
    GELOGI("pp[%s]'s compile result is empty, no need to set compile result to node(%s)", pp_name.c_str(),
           node_name.c_str());
    return SUCCESS;
  }

  ge::NamedAttrs compile_results;
  (void)ge::AttrUtils::GetNamedAttrs(op_desc, ATTR_NAME_DATA_FLOW_COMPILER_RESULT, compile_results);
  ge::NamedAttrs current_compile_result;
  ge::NamedAttrs running_res_info;
  for (const auto &resource_info : result.running_resources_info) {
    running_res_info.SetAttr(resource_info.resource_type,
                             GeAttrValue::CreateFrom<int64_t>(resource_info.resource_num));
  }
  GE_CHK_BOOL_RET_STATUS(
      ge::AttrUtils::SetNamedAttrs(current_compile_result, ATTR_NAME_DATA_FLOW_RUNNING_RESOURCE_INFO, running_res_info),
      FAILED, "Set pp[%s]'s attr[%s] failed.", pp_name.c_str(), ATTR_NAME_DATA_FLOW_RUNNING_RESOURCE_INFO);

  ge::NamedAttrs runnable_resources_info;
  for (const auto &bin_info : result.compile_bin_info) {
    runnable_resources_info.SetAttr(bin_info.first, GeAttrValue::CreateFrom<std::string>(bin_info.second));
    GELOGI("insert pp[%s]'s runnable resource type[%s], value[%s] to node[%s]", pp_name.c_str(), bin_info.first.c_str(),
           bin_info.second.c_str(), node_name.c_str());
  }

  GE_CHK_BOOL_RET_STATUS(ge::AttrUtils::SetNamedAttrs(current_compile_result, ATTR_NAME_DATA_FLOW_RUNNABLE_RESOURCE,
                                                      runnable_resources_info),
                         FAILED, "Set pp[%s]'s attr[%s] failed.", pp_name.c_str(),
                         ATTR_NAME_DATA_FLOW_RUNNABLE_RESOURCE);
  compile_results.SetAttr(pp_name, GeAttrValue::CreateFrom<ge::NamedAttrs>(current_compile_result));
  GE_CHK_BOOL_RET_STATUS(ge::AttrUtils::SetNamedAttrs(op_desc, ATTR_NAME_DATA_FLOW_COMPILER_RESULT, compile_results),
                         FAILED, "Set pp[%s]'s attr[%s] failed.",
                         pp_name.c_str(), ATTR_NAME_DATA_FLOW_COMPILER_RESULT);
  GELOGI("Set pp[%s]'s attr[%s] success.", pp_name.c_str(), ATTR_NAME_DATA_FLOW_COMPILER_RESULT);
  return SUCCESS;
}

void ProcessPointLoader::LoadCompileResultFromCache(const CacheCompileResult &cache_compile_result,
                                                    FunctionCompile::CompileResult &result) {
  for (const auto &it : cache_compile_result.compile_bin_info) {
    result.compile_bin_info[it.first] = it.second;
  }

  for (const auto &it : cache_compile_result.running_resources_info) {
    CompileConfigJson::RunningResourceInfo res_info = {};
    res_info.resource_type = it.first;
    res_info.resource_num = static_cast<uint32_t>(it.second);
    result.running_resources_info.emplace_back(res_info);
  }
}

Status ProcessPointLoader::CompileFunctionProcessPoint(const std::string &pp_name,
                                                       const CompileConfigJson::FunctionPpConfig &func_pp_cfg,
                                                       const NodePtr &node) {
  GE_TRACE_START(CompileFunctionProcessPoint);
  FunctionCompile compile(pp_name, func_pp_cfg);
  GE_CHK_STATUS_RET(compile.CompileAllResourceType(), "Failed to compile function pp[%s]'s all resource type.",
                    pp_name.c_str());
  const auto &compile_result = compile.GetCompileResult();
  GE_CHK_STATUS_RET(SetCompileResultToNode(pp_name, compile_result, node),
                    "Failed to set pp[%s]'s compile result to node[%s]",
                    pp_name.c_str(), node->GetName().c_str());
  const std::string trace_log = "loading func point [" + pp_name + "] in compiling stage";
  GE_COMPILE_TRACE_TIMESTAMP_END(CompileFunctionProcessPoint, trace_log.c_str());
  return SUCCESS;
}

Status ProcessPointLoader::CheckBufConfigIsValid(const std::vector<CompileConfigJson::BufCfg> &user_buf_cfg) {
  if (user_buf_cfg.empty()) {
    GELOGD("User buf config is not set.");
    return SUCCESS;
  }
  GE_ASSERT_TRUE(user_buf_cfg.size() <= kMaxBufCfgNum, "The user buf config num[%zu] is out of range. "
                 "Valid range is (0, %u]", user_buf_cfg.size(), kMaxBufCfgNum);
  uint32_t last_normal_max_buf_size = 0U;
  uint32_t last_huge_max_buf_size = 0U;
  for (const auto &item : user_buf_cfg) {
    GE_ASSERT_TRUE((item.total_size > item.max_buf_size) && (item.max_buf_size >= item.blk_size),
        "The following three params not meet the requirement: total_size[%u] > max_buf_size[%u] >= blk_size[%u]",
        item.total_size, item.max_buf_size, item.blk_size);
    GE_ASSERT_TRUE(item.blk_size != 0, "The blk_size[%u] should not be 0.", item.blk_size);
    GE_ASSERT_TRUE((item.blk_size & (item.blk_size - 1U)) == 0U, "The blk_size[%u] should be 2^n.", item.blk_size);
    GE_ASSERT_TRUE(item.blk_size <= kMaxBlkSize, "The blk_size[%u] should not greater than %u.", kMaxBlkSize);
    GE_ASSERT_TRUE(item.total_size % item.blk_size == 0U, "The total_size[%u] should be multiple of blk_size[%u].",
                   item.total_size, item.blk_size);
    if (item.page_type == kPageTypeNormal) {
      if (item.max_buf_size <= last_normal_max_buf_size) {
        GELOGE(FAILED, "Normal page type should be sorted. But current size[%zu] is less or equal to last gear[%zu].",
               item.max_buf_size, last_normal_max_buf_size);
        return FAILED;
      }
      last_normal_max_buf_size = item.max_buf_size;
      GE_ASSERT_TRUE(item.total_size % kNormalPageAtomicValue == 0UL, "The normal page size[%u] should be multiple of %u",
                     item.total_size, kNormalPageAtomicValue);
    } else if (item.page_type == kPageTypeHuge) {
      if (item.max_buf_size <= last_huge_max_buf_size) {
        GELOGE(FAILED, "Huge page type should be sorted. But current size[%zu] is less or equal to last gear[%zu].",
               item.max_buf_size, last_huge_max_buf_size);
        return FAILED;
      }
      last_huge_max_buf_size = item.max_buf_size;
      GE_ASSERT_TRUE(item.total_size % kHugePageAtomicValue == 0UL, "The huge page size[%u] should be multiple of %u",
                     item.total_size, kHugePageAtomicValue);
    } else {
      GELOGE(FAILED, "The page type[%s] of user buffer config is invalid.", item.page_type.c_str());
      return FAILED;
    }
  }
  return SUCCESS;
}

Status ProcessPointLoader::CheckFunctionPpConfigIsValid(const CompileConfigJson::FunctionPpConfig &func_pp_cfg,
                                                        const std::string &pp_name) {
  std::set<uint32_t> inputs_set;
  std::set<std::string> func_names_set;
  for (const auto &function_desc : func_pp_cfg.funcs) {
    // duplicate func names check
    if (func_names_set.count(function_desc.func_name) > 0) {
      GELOGE(FAILED, "The func name[%s] of pp name[%s] is used by multi funcs", function_desc.func_name.c_str(),
             pp_name.c_str());
      return FAILED;
    }
    func_names_set.insert(function_desc.func_name);

    for (auto input_index : function_desc.inputs_index) {
      if (input_index >= func_pp_cfg.input_num) {
        GELOGE(FAILED, "The input index[%u] of pp name[%s]'s func name[%s] is out of range, valid range is [0, %u]",
               input_index, pp_name.c_str(), function_desc.func_name.c_str(), func_pp_cfg.input_num - 1U);
        return FAILED;
      }
      if (inputs_set.count(input_index) > 0) {
        GELOGE(FAILED, "The input index[%u] of pp name[%s]'s func name[%s] is used by multi funcs",
               input_index, pp_name.c_str(), function_desc.func_name.c_str());
        return FAILED;
      }
      inputs_set.insert(input_index);
    }

    for (auto output_index : function_desc.outputs_index) {
      if (output_index >= func_pp_cfg.output_num) {
        GELOGE(FAILED, "The output index[%u] of pp name[%s]'s func name[%s] is out of range, valid range is [0, %u]",
               output_index, pp_name.c_str(), function_desc.func_name.c_str(), func_pp_cfg.output_num - 1U);
        return FAILED;
      }
    }
  }

  if (inputs_set.empty()) {
    if (func_pp_cfg.funcs.size() > 1U) {
      GELOGE(FAILED, "The pp name[%s] have multi funcs, so it must specify inputs_index.", pp_name.c_str());
      return FAILED;
    }
  } else {
    if (inputs_set.size() != func_pp_cfg.input_num) {
      GELOGE(FAILED, "The input num[%u] of pp name[%s] is not equal to sum of each func's inputs_index nums[%zu].",
             func_pp_cfg.input_num, pp_name.c_str(), inputs_set.size());
      return FAILED;
    }
  }
  return CheckBufConfigIsValid(func_pp_cfg.user_buf_cfg);
}

Status ProcessPointLoader::SetNMappingAttrIfHasStreamInput(const CompileConfigJson::FunctionPpConfig &func_pp_cfg,
                                                           const ComputeGraphPtr &graph) {
  bool contains_stream_input = false;
  for (const auto &func : func_pp_cfg.funcs) {
    if (func.stream_input) {
      contains_stream_input = true;
      break;
    }
  }
  if (contains_stream_input) {
    GE_CHK_STATUS_RET(DataFlowGraphUtils::EnsureNMappingAttr(graph), "Failed to set n-mapping attr for graph[%s]",
                      graph->GetName().c_str());
  }
  return SUCCESS;
}

Status ProcessPointLoader::AddFlowFuncToComputeGraph(const OpDescPtr &flow_func_desc, ComputeGraphPtr &compute_graph) {
  auto flow_func_node = compute_graph->AddNode(flow_func_desc);
  GE_CHECK_NOTNULL(flow_func_node);
  GE_CHK_STATUS_RET(AddInputsForFlowFunc(flow_func_node, compute_graph), "Failed to add inputs to FlowFunc node[%s].",
                    flow_func_node->GetName().c_str());
  GE_CHK_STATUS_RET(AddOutputsForFlowFunc(flow_func_node, compute_graph), "Failed to add outputs to FlowFunc node[%s].",
                    flow_func_node->GetName().c_str());
  return SUCCESS;
}

Status ProcessPointLoader::CompileUserFunctionProcessPoint(const CompileConfigJson::FunctionPpConfig &func_pp_cfg,
    const std::string &pp_name, const ComputeGraphPtr &compute_graph, const NodePtr &node, DataFlowGraph &data_flow_graph) {
    GE_CHECK_NOTNULL(node);

  if (func_pp_cfg.built_in_flow_func) {
    FunctionCompile::CompileResult result;
    GE_CHK_STATUS_RET(FunctionCompile::GetBuiltInFuncCompileResult(result),
                      "Failed to get built-in function compile result for node[%s] pp[%s].", node->GetName().c_str(),
                      pp_name.c_str());
    GE_CHK_STATUS_RET(SetCompileResultToNode(pp_name, result, node),
                      "Failed to set pp[%s]'s compile result to node[%s]", pp_name.c_str(), node->GetName().c_str());
  } else {
    const std::string cache_dir = FlowModelCache::GetCacheDirFromContext();
    const std::string graph_key = FlowModelCache::GetGraphKeyFromContext();
    FlowModelCache sub_flow_model_cache;
    GE_CHECK_NOTNULL(compute_graph);
    GE_CHK_STATUS_RET(sub_flow_model_cache.InitSubmodelCache(compute_graph, cache_dir, graph_key),
                      "Failed to init subgraphs flow model cache.");
    bool match_cache = false;
    CacheCompileResult cache_compile_result = {};
    GELOGI("Set compile result to node, cache enable = %d, cache manual check = %d",
          static_cast<int32_t>(sub_flow_model_cache.EnableCache()),
          static_cast<int32_t>(sub_flow_model_cache.ManualCheck()));
    if (sub_flow_model_cache.EnableCache() && sub_flow_model_cache.ManualCheck()) {
      match_cache = sub_flow_model_cache.TryLoadCompileResultFromCache(cache_compile_result);
    }

    if (match_cache) {
      FunctionCompile::CompileResult compile_result = {};
      LoadCompileResultFromCache(cache_compile_result, compile_result);
      GE_CHK_STATUS_RET(SetCompileResultToNode(pp_name, compile_result, node),
                        "Failed to set pp[%s]'s compile result to node[%s]",
                        pp_name.c_str(), node->GetName().c_str());
      GELOGI("Set compile result to node from cache success");
    } else {
      std::function<Status()> compile_task = [pp_name, func_pp_cfg, node, &sub_flow_model_cache]() {
      GE_CHK_STATUS_RET(CompileFunctionProcessPoint(pp_name, func_pp_cfg, node),
                        "Failed to compile function pp[%s].",
                        pp_name.c_str());
        return SUCCESS;
      };
      GE_CHK_STATUS_RET(data_flow_graph.CommitPreprocessTask(pp_name, compile_task),
                        "Failed to commit compile task, function pp[%s].", pp_name.c_str());
    }
  }
  return SUCCESS;
}

Status ProcessPointLoader::SetUserFunctionProcessPointAttrs(const CompileConfigJson::FunctionPpConfig &func_pp_cfg,
    const ComputeGraphPtr &compute_graph, const NodePtr &node, const OpDescPtr &flow_func_desc) {
    GE_CHECK_NOTNULL(compute_graph);
    GE_CHECK_NOTNULL(node);

  GE_CHK_BOOL_RET_STATUS(AttrUtils::SetBool(compute_graph, ATTR_NAME_DATA_FLOW_HEAVY_LOAD, func_pp_cfg.heavy_load), FAILED,
                         "Failed to set attr[%s] for graph[%s], value=%d.", ATTR_NAME_DATA_FLOW_HEAVY_LOAD,
                         compute_graph->GetName().c_str(), static_cast<int32_t>(func_pp_cfg.heavy_load));
  const auto &op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  GE_CHK_BOOL_RET_STATUS(AttrUtils::SetBool(op_desc, ATTR_NAME_DATA_FLOW_HEAVY_LOAD, func_pp_cfg.heavy_load),
                         FAILED, "Failed to set attr[%s] for node[%s].", ATTR_NAME_DATA_FLOW_HEAVY_LOAD,
                         node->GetName().c_str());
  GELOGD("node[%s] attr[%s] is set to %d", node->GetName().c_str(), ATTR_NAME_DATA_FLOW_HEAVY_LOAD,
         static_cast<int32_t>(func_pp_cfg.heavy_load));
  if (!func_pp_cfg.user_buf_cfg.empty()) {
    GE_CHK_BOOL_RET_STATUS(compute_graph->SetExtAttr(kBufferConfig, func_pp_cfg.user_buf_cfg), FAILED,
                           "Failed to set attr[%s] for graph[%s].", kBufferConfig, compute_graph->GetName().c_str());
  }

  bool is_balance_scatter = false;
  bool is_balance_gather = false;
  (void)AttrUtils::GetBool(op_desc, dflow::ATTR_NAME_BALANCE_SCATTER, is_balance_scatter);
  (void)AttrUtils::GetBool(op_desc, dflow::ATTR_NAME_BALANCE_GATHER, is_balance_gather);

  GE_CHK_BOOL_RET_STATUS(!(is_balance_scatter && is_balance_gather), FAILED,
                         "The attr[%s] and attr[%s] cannot be set on the same node[%s]",
                         dflow::ATTR_NAME_BALANCE_SCATTER, dflow::ATTR_NAME_BALANCE_GATHER, node->GetName().c_str());
  GE_CHECK_NOTNULL(flow_func_desc);
  GE_CHK_BOOL_RET_STATUS(AttrUtils::SetBool(flow_func_desc, ATTR_NAME_DATA_FLOW_VISIBLE_DEVICE_ENABLE,
                         func_pp_cfg.visible_device_enable),
                         FAILED, "Failed to set attr[%s] for func[%s].", ATTR_NAME_DATA_FLOW_VISIBLE_DEVICE_ENABLE,
                         flow_func_desc->GetNamePtr());
  GELOGD("func[%s] attr[%s] is set to %d", flow_func_desc->GetNamePtr(), ATTR_NAME_DATA_FLOW_VISIBLE_DEVICE_ENABLE,
         static_cast<int32_t>(func_pp_cfg.visible_device_enable));
  if (is_balance_scatter) {
    GE_CHK_BOOL_RET_STATUS(AttrUtils::SetBool(flow_func_desc, dflow::ATTR_NAME_BALANCE_SCATTER, true), FAILED,
                           "Failed to set attr[%s] for func[%s].", dflow::ATTR_NAME_BALANCE_SCATTER,
                           flow_func_desc->GetNamePtr());
  }

  if (is_balance_gather) {
    GE_CHK_BOOL_RET_STATUS(AttrUtils::SetBool(flow_func_desc, dflow::ATTR_NAME_BALANCE_GATHER, true), FAILED,
                           "Failed to set attr[%s] for func[%s].", dflow::ATTR_NAME_BALANCE_GATHER,
                           flow_func_desc->GetNamePtr());
  }

  int64_t npu_sched_model = 0;
  constexpr const char *kAttrNpuSchedModel = "_npu_sched_model";
  (void)AttrUtils::GetInt(op_desc, kAttrNpuSchedModel, npu_sched_model);
  if (npu_sched_model != 0) {
    GE_CHK_BOOL_RET_STATUS(AttrUtils::SetInt(compute_graph, kAttrNpuSchedModel, npu_sched_model), FAILED,
                           "Failed to set attr[%s] for graph[%s], value=%ld.", kAttrNpuSchedModel,
                           compute_graph->GetName().c_str(), npu_sched_model);
    GELOGD("Set attr[%s]=%ld to graph[%s] success", kAttrNpuSchedModel, npu_sched_model,
           compute_graph->GetName().c_str());

    constexpr const char *kNpuSchedModelIoPlacement = "device";
    GE_CHK_BOOL_RET_STATUS(
        AttrUtils::SetStr(compute_graph, ATTR_NAME_FLOW_ATTR_IO_PLACEMENT, kNpuSchedModelIoPlacement), FAILED,
        "Failed to set attr[%s]", ATTR_NAME_FLOW_ATTR_IO_PLACEMENT.c_str());
    GELOGD("Set attr[%s]=%s to npu sched model graph[%s] success", ATTR_NAME_FLOW_ATTR_IO_PLACEMENT.c_str(),
           kNpuSchedModelIoPlacement, compute_graph->GetName().c_str());
  }
  return SUCCESS;
}

Status ProcessPointLoader::LoadUserFunctionProcessPoint(const dataflow::ProcessPoint &process_point,
                                                        DataFlowGraph &data_flow_graph, const NodePtr &node) {
  CompileConfigJson::FunctionPpConfig func_pp_cfg = {};
  const std::string pp_name = process_point.name();
  GE_CHK_STATUS_RET(CompileConfigJson::ReadFunctionPpConfigFromJsonFile(process_point.compile_cfg_file(), func_pp_cfg),
                    "Failed to read process point[%s] compile config.", pp_name.c_str());
  GE_CHK_STATUS_RET(CheckFunctionPpConfigIsValid(func_pp_cfg, pp_name),
                    "Process point[%s]'s compile config file[%s] is invalid.", pp_name.c_str(),
                    process_point.compile_cfg_file().c_str());
  GE_CHK_STATUS_RET(SetNMappingAttrIfHasStreamInput(func_pp_cfg, data_flow_graph.root_graph_),
                    "Failed to set n-mapping attr for process point[%s] with stream input", pp_name.c_str());
  ComputeGraphPtr temp_graph;
  GE_CHK_STATUS_RET(CreateFunctionProcessPointComputeGraph(data_flow_graph, pp_name, func_pp_cfg.input_num,
                                                           func_pp_cfg.output_num, temp_graph),
                    "Failed to create compute graph for pp[%s], input num[%zu], output num[%zu]", pp_name.c_str(),
                    func_pp_cfg.input_num, func_pp_cfg.output_num);

  GE_CHK_STATUS_RET(CompileUserFunctionProcessPoint(func_pp_cfg, pp_name, temp_graph, node, data_flow_graph),
                    "Compile user function process point failed.");
  OpDescPtr flow_func_desc;
  GE_CHK_STATUS_RET(CreateFlowFuncOpDescFromProcessPoint(process_point, func_pp_cfg, flow_func_desc),
                    "Failed to create FlowFunc from process point[%s]", process_point.name().c_str());
  GE_CHK_STATUS_RET(AddFlowFuncToComputeGraph(flow_func_desc, temp_graph), "Failed to add FlowFunc[%s] to graph[%s].",
                    flow_func_desc->GetName().c_str(), temp_graph->GetName().c_str());
  GE_CHK_STATUS_RET(SetEschedPriorities(temp_graph, node->GetOpDesc()),
                    "Failed to set esched priorities for graph[%s], node[%s]", temp_graph->GetName().c_str(),
                    node->GetName().c_str());
  GE_CHK_STATUS_RET(SetUserFunctionProcessPointAttrs(func_pp_cfg, temp_graph, node, flow_func_desc),
                    "Failed to set user function process point attrs.");
  GE_CHK_STATUS_RET(SetDataFlowScope(flow_func_desc, data_flow_graph.GetDataFlowScope()),
                    "Failed to set data flow scope attr.");
  GE_CHK_STATUS_RET(LoadInvokedProcessPoint(process_point, data_flow_graph, flow_func_desc, node, func_pp_cfg.built_in_flow_func),
                    "Failed to load invoked graph process point for process point[%s].", process_point.name().c_str());
  data_flow_graph.subgraphs_[process_point.name()] = temp_graph;
  data_flow_graph.node_subgraphs_[node->GetName()].emplace_back(temp_graph);
  GE_DUMP(temp_graph, "LoadFunctionProcessPoint");
  return SUCCESS;
}

Status ProcessPointLoader::CreateFunctionProcessPointComputeGraph(const DataFlowGraph &data_flow_graph,
                                                                  const std::string &graph_name, uint32_t input_num,
                                                                  uint32_t output_num, ComputeGraphPtr &compute_graph) {
  const auto name = data_flow_graph.IsRootDataFlow() ? graph_name
                                                     : data_flow_graph.GetDataFlowScope() + graph_name;
  compute_graph = MakeShared<ComputeGraph>(name);
  GE_CHECK_NOTNULL(compute_graph);
  compute_graph->SetInputSize(input_num);
  compute_graph->SetOutputSize(output_num);
  GE_CHK_BOOL_RET_STATUS(AttrUtils::SetStr(compute_graph, ATTR_NAME_PROCESS_NODE_ENGINE_ID, PNE_ID_UDF), FAILED,
                         "Failed to set attr[%s] for graph[%s].", ATTR_NAME_PROCESS_NODE_ENGINE_ID.c_str(),
                         graph_name.c_str());
  std::string session_graph_id;
  GE_CHK_BOOL_RET_STATUS(AttrUtils::GetStr(data_flow_graph.root_graph_, ATTR_NAME_SESSION_GRAPH_ID, session_graph_id),
                         FAILED, "Failed to get attr[%s] from graph[%s].", ATTR_NAME_SESSION_GRAPH_ID.c_str(),
                         data_flow_graph.root_graph_->GetName().c_str());
  GE_CHK_BOOL_RET_STATUS(AttrUtils::SetStr(compute_graph, ATTR_NAME_SESSION_GRAPH_ID, session_graph_id), FAILED,
                         "Failed to set attr[%s] to graph[%s].", ATTR_NAME_SESSION_GRAPH_ID.c_str(),
                         graph_name.c_str());
  return SUCCESS;
}

Status ProcessPointLoader::LoadBuiltInFunctionProcessPoint(const dataflow::ProcessPoint &process_point,
                                                           DataFlowGraph &data_flow_graph, const NodePtr &node) {
  FunctionCompile::CompileResult result;
  const std::string pp_name = process_point.name();
  GE_CHK_STATUS_RET(FunctionCompile::GetBuiltInFuncCompileResult(result),
                    "Failed to get built-in function compile result for node[%s] pp[%s].", node->GetName().c_str(),
                    pp_name.c_str());
  GE_CHK_STATUS_RET(SetCompileResultToNode(pp_name, result, node), "Failed to set pp[%s]'s compile result to node[%s]",
                    pp_name.c_str(), node->GetName().c_str());

  ComputeGraphPtr temp_graph;
  GE_CHK_STATUS_RET(CreateFunctionProcessPointComputeGraph(data_flow_graph, pp_name, process_point.in_edges().size(),
                                                           process_point.out_edges().size(), temp_graph),
                    "Failed to create compute graph for pp[%s], input num[%zu], output num[%zu]", pp_name.c_str(),
                    process_point.in_edges().size(), process_point.out_edges().size());
  OpDescPtr flow_func_desc;
  GE_CHK_STATUS_RET(DataFlowGraphUtils::CreateFlowFuncOpDesc(pp_name, process_point.in_edges().size(),
                                                             process_point.out_edges().size(), flow_func_desc),
                    "Failed create FlowFunc op desc for process point[%s], inputs num [%zu], outputs num [%zu].",
                    pp_name.c_str(), process_point.in_edges().size(), process_point.out_edges().size());
  GE_CHK_STATUS_RET_NOLOG(SetCustomizedAttrsForFlowFunc(process_point, flow_func_desc));
  GE_CHK_STATUS_RET_NOLOG(SetAttrFuncsForFlowFunc(process_point, flow_func_desc));

  GE_CHK_STATUS_RET(AddFlowFuncToComputeGraph(flow_func_desc, temp_graph), "Failed to add FlowFunc[%s] to graph[%s].",
                    flow_func_desc->GetName().c_str(), temp_graph->GetName().c_str());
  GE_CHK_BOOL_RET_STATUS(temp_graph->SetExtAttr("_is_built_in_for_data_flow", true), FAILED,
                         "Failed to set built in attr for graph[%s].", temp_graph->GetName().c_str());
  data_flow_graph.subgraphs_[process_point.name()] = temp_graph;
  data_flow_graph.node_subgraphs_[node->GetName()].emplace_back(temp_graph);
  GE_DUMP(temp_graph, "LoadFunctionProcessPoint");
  return SUCCESS;
}

Status ProcessPointLoader::LoadFunctionProcessPoint(const dataflow::ProcessPoint &process_point,
                                                    DataFlowGraph &data_flow_graph, const NodePtr &node) {
  GE_TRACE_START(LoadFunctionProcessPoint);
  if (data_flow_graph.subgraphs_.find(process_point.name()) != data_flow_graph.subgraphs_.cend()) {
    GELOGE(FAILED, "The process point [%s] is map more than one node.", process_point.name().c_str());
    return FAILED;
  }
  if (process_point.is_built_in()) {
    GE_CHK_STATUS_RET(LoadBuiltInFunctionProcessPoint(process_point, data_flow_graph, node),
                      "Failed to load process point[%s].", process_point.name().c_str());
  } else {
    GE_CHK_STATUS_RET(LoadUserFunctionProcessPoint(process_point, data_flow_graph, node),
                      "Failed to load process point[%s].", process_point.name().c_str());
  }

  std::string trace_log = "loading func point async [" + process_point.name() + "] on node[" + node->GetName() + "]";
  GE_COMPILE_TRACE_TIMESTAMP_END(LoadFunctionProcessPoint, trace_log.c_str());
  return SUCCESS;
}

Status ProcessPointLoader::UpdateGraphInputsDesc(const CompileConfigJson::GraphPpConfig &graph_pp_cfg,
                                                 ComputeGraphPtr &compute_graph) {
  auto input_nodes = compute_graph->GetInputNodes();
  if (input_nodes.size() != graph_pp_cfg.inputs_tensor_desc_config.size()) {
    GELOGE(FAILED, "The graph[%s] has input num[%zu], but config json set input num[%zu].",
           compute_graph->GetName().c_str(), input_nodes.size(), graph_pp_cfg.inputs_tensor_desc_config.size());
    return FAILED;
  }
  for (size_t i = 0U; i < input_nodes.size(); ++i) {
    const auto &node = input_nodes.at(i);
    GE_CHECK_NOTNULL(node);
    GE_CHK_BOOL_RET_STATUS(OpTypeUtils::IsDataNode(node->GetType()), FAILED, "The node type[%s] is not data node.",
                           node->GetType().c_str());
    const auto &op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    int64_t index = 0;
    GE_CHK_BOOL_RET_STATUS(AttrUtils::GetInt(op_desc, ATTR_NAME_INDEX, index), FAILED,
                           "Failed to get attr[%s] from op[%s].", ATTR_NAME_INDEX.c_str(), op_desc->GetName().c_str());
    GE_CHK_BOOL_RET_STATUS(index < static_cast<int64_t>(graph_pp_cfg.inputs_tensor_desc_config.size()), FAILED,
                           "The op[%s] attr[%s] value[%ld] is out of range [0, %zu).", op_desc->GetName().c_str(),
                           ATTR_NAME_INDEX.c_str(), index, graph_pp_cfg.inputs_tensor_desc_config.size());
    auto input_desc = op_desc->MutableInputDesc(0);
    GE_CHECK_NOTNULL(input_desc);
    if (std::get<0>(graph_pp_cfg.inputs_tensor_desc_config[index].dtype_config)) {
      input_desc->SetDataType(std::get<1>(graph_pp_cfg.inputs_tensor_desc_config[index].dtype_config));
    }
    if (std::get<0>(graph_pp_cfg.inputs_tensor_desc_config[index].format_config)) {
      input_desc->SetFormat(std::get<1>(graph_pp_cfg.inputs_tensor_desc_config[index].format_config));
    }
    if (std::get<0>(graph_pp_cfg.inputs_tensor_desc_config[index].shape_config)) {
      input_desc->SetShape(GeShape(std::get<1>(graph_pp_cfg.inputs_tensor_desc_config[index].shape_config)));
    }
  }
  GELOGD("Update graph[%s] inputs desc successfully.", compute_graph->GetName().c_str());
  return SUCCESS;
}

Status ProcessPointLoader::AddSubGraphsToProcessPointGraph(const ComputeGraphPtr &root_graph,
                                                           const ComputeGraphPtr &pp_graph) {
  std::vector<std::string> move_graph_names;
  for (const auto &subgraph : root_graph->GetAllSubgraphs()) {
    GE_CHECK_NOTNULL(subgraph);
    auto parent_graph = subgraph->GetParentGraph();
    while (parent_graph != nullptr) {
      if (parent_graph == pp_graph) {
        GE_CHK_STATUS_RET(pp_graph->AddSubgraph(subgraph), "Failed to add subgraph[%s] to graph[%s]",
                          subgraph->GetName().c_str(), pp_graph->GetName().c_str());
        move_graph_names.emplace_back(subgraph->GetName());
        break;
      }
      parent_graph = parent_graph->GetParentGraph();
    }
  }

  for (const auto &sub_name : move_graph_names) {
    root_graph->RemoveSubgraph(sub_name);
  }
  GELOGD("[Add][Subgraphs] from root graph:%s to process point graph:%s success.", root_graph->GetName().c_str(),
         pp_graph->GetName().c_str());
  return SUCCESS;
}

Status ProcessPointLoader::SetFlowAttrToGraph(const NodePtr &node, const ComputeGraphPtr &graph) {
  auto op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  if (op_desc->HasAttr(ATTR_NAME_FLOW_ATTR)) {
    bool value = false;
    GE_CHK_BOOL_RET_STATUS(AttrUtils::GetBool(op_desc, ATTR_NAME_FLOW_ATTR, value), FAILED, "Failed to get attr[%s]",
                           ATTR_NAME_FLOW_ATTR.c_str());
    GE_CHK_BOOL_RET_STATUS(AttrUtils::SetBool(graph, ATTR_NAME_FLOW_ATTR, value), FAILED, "Failed to set attr[%s]",
                           ATTR_NAME_FLOW_ATTR.c_str());
    GELOGD("Set attr[%s]=%d to graph[%s] success", ATTR_NAME_FLOW_ATTR.c_str(), static_cast<int32_t>(value),
           graph->GetName().c_str());
  }
  if (op_desc->HasAttr(ATTR_NAME_FLOW_ATTR_DEPTH)) {
    int64_t depth = 0L;
    GE_CHK_BOOL_RET_STATUS(AttrUtils::GetInt(op_desc, ATTR_NAME_FLOW_ATTR_DEPTH, depth), FAILED,
                           "Failed to get attr[%s]", ATTR_NAME_FLOW_ATTR_DEPTH.c_str());
    GE_CHK_BOOL_RET_STATUS(AttrUtils::SetInt(graph, ATTR_NAME_FLOW_ATTR_DEPTH, depth), FAILED,
                           "Failed to set attr[%s]", ATTR_NAME_FLOW_ATTR_DEPTH.c_str());
    GELOGD("Set attr[%s]=%d to graph[%s] success", ATTR_NAME_FLOW_ATTR_DEPTH.c_str(), depth, graph->GetName().c_str());
  }
  if (op_desc->HasAttr(ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY)) {
    std::string policy;
    GE_CHK_BOOL_RET_STATUS(AttrUtils::GetStr(op_desc, ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY, policy), FAILED,
                           "Failed to get attr[%s]", ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY.c_str());
    GE_CHK_BOOL_RET_STATUS(AttrUtils::SetStr(graph, ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY, policy), FAILED,
                           "Failed to set attr[%s]", ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY.c_str());
    GELOGD("Set attr[%s]=%s to graph[%s] success", ATTR_NAME_FLOW_ATTR_ENQUEUE_POLICY.c_str(), policy.c_str(),
           graph->GetName().c_str());
  }
  return SUCCESS;
}

Status ProcessPointLoader::PreProcessSubgraphAttrs(const ComputeGraphPtr &subgraph) {
  for (const auto &node : subgraph->GetDirectNode()) {
    if (node->GetType() == DATA) {
      int64_t parent_node_index = 0;
      if (AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, parent_node_index)) {
        (void) node->GetOpDesc()->DelAttr(ATTR_NAME_PARENT_NODE_INDEX);
        (void) AttrUtils::SetInt(node->GetOpDesc(), ATTR_NAME_INDEX, parent_node_index);
      }
      continue;
    }
    if (node->GetType() == NETOUTPUT) {
      for (size_t index = 0UL; index < node->GetOpDesc()->GetAllInputsSize(); index++) {
        const auto &in_tensor = node->GetOpDesc()->MutableInputDesc(index);
        GE_CHECK_NOTNULL(in_tensor);
        size_t parent_node_index = 0;
        if (AttrUtils::GetInt(in_tensor, ATTR_NAME_PARENT_NODE_INDEX, parent_node_index)) {
          (void) in_tensor->DelAttr(ATTR_NAME_PARENT_NODE_INDEX);
        }
      }
      continue;
    }
  }

  subgraph->SetParentNode(nullptr);
  subgraph->SetParentGraph(nullptr);
  GELOGD("[Subgraph][PreProcess] graph name:%s", subgraph->GetName().c_str());
  return SUCCESS;
}

Status ProcessPointLoader::RemoveGraphFromParent(const ComputeGraphPtr &root_graph, const ComputeGraphPtr &sub_graph) {
  GE_CHECK_NOTNULL(root_graph);
  GE_CHECK_NOTNULL(sub_graph);
  auto parent_node = sub_graph->GetParentNode();
  if ((parent_node != nullptr) && (parent_node->GetOpDesc() != nullptr)) {
    parent_node->GetOpDesc()->RemoveSubgraphInstanceName(sub_graph->GetName());
    GELOGI("Remove subgraph[%s] from node[%s] success.", sub_graph->GetName().c_str(), parent_node->GetNamePtr());
  }
  root_graph->RemoveSubgraph(sub_graph->GetName());
  GE_CHK_STATUS_RET(PreProcessSubgraphAttrs(sub_graph),
                    "Failed to PreProcessSubGraphAttrs failed, graph[%s].", sub_graph->GetName().c_str());
  return SUCCESS;
}

Status ProcessPointLoader::FixGraphOutputNode(const ComputeGraphPtr &graph) {
  if (!graph->GetGraphOutNodesInfo().empty()) {
    return SUCCESS;
  }
  NodePtr netoutput_node = graph->FindFirstNodeMatchType(NETOUTPUT);
  if (netoutput_node == nullptr) {
    GELOGW("Graph[%s] has no output nodes and netoutput node.", graph->GetName().c_str());
    return SUCCESS;
  }
  for (const auto &in_data_anchor : netoutput_node->GetAllInDataAnchorsPtr()) {
    const auto &peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
    GE_CHECK_NOTNULL(peer_out_anchor, "Graph[%s] node[%s] input[%d] peer out anchor is null", graph->GetName().c_str(),
                     netoutput_node->GetNamePtr(), in_data_anchor->GetIdx());
    graph->AddOutputNodeByIndex(peer_out_anchor->GetOwnerNode(), peer_out_anchor->GetIdx());
  }
  GELOGI("Fix graph[%s] out nodes info size=%zu.", graph->GetName().c_str(), graph->GetGraphOutNodesInfo().size());
  return SUCCESS;
}

Status ProcessPointLoader::LoadGraphProcessPoint(const dataflow::ProcessPoint &process_point,
                                                 DataFlowGraph &data_flow_graph, const NodePtr &node) {
  GE_TRACE_START(LoadGraphProcessPoint);
  if (data_flow_graph.subgraphs_.find(process_point.name()) != data_flow_graph.subgraphs_.cend()) {
    GELOGE(FAILED, "The process point [%s] is map more than one node.", process_point.name().c_str());
    return FAILED;
  }
  CompileConfigJson::GraphPpConfig graph_pp_cfg = {};
  GE_CHK_STATUS_RET(CompileConfigJson::ReadGraphPpConfigFromJsonFile(process_point.compile_cfg_file(), graph_pp_cfg),
                    "Failed to read process point[%s] compile config.", process_point.name().c_str());
  GE_CHK_BOOL_RET_STATUS(process_point.graphs_size() == 1, FAILED,
                         "The graph process point[%s] graphs size should be [1], but got [%d].",
                         process_point.name().c_str(), process_point.graphs_size());
  auto temp_graph = data_flow_graph.root_graph_->GetSubgraph(process_point.graphs(0));
  GE_CHECK_NOTNULL(temp_graph);
  GE_CHK_STATUS_RET(RemoveGraphFromParent(data_flow_graph.root_graph_, temp_graph),
                    "Failed to remove graph from parent failed, graph[%s], pp name[%s].", temp_graph->GetName().c_str(),
                    process_point.name().c_str());
  GELOGI("rename graph[%s] to pp name[%s]", temp_graph->GetName().c_str(), process_point.name().c_str());
  // subgraph rename as process point name
  temp_graph->SetName(process_point.name());
  GE_CHK_STATUS_RET(AddSubGraphsToProcessPointGraph(data_flow_graph.root_graph_, temp_graph),
                    "Failed to add subgraphs to process point graph:%s", temp_graph->GetName().c_str());
  bool is_data_flow_graph = false;
  (void)AttrUtils::GetBool(temp_graph, dflow::ATTR_NAME_IS_DATA_FLOW_GRAPH, is_data_flow_graph);
  if (!is_data_flow_graph) {
    GE_CHK_STATUS_RET(FixGraphOutputNode(temp_graph), "Failed to fix graph:%s output node",
                      temp_graph->GetName().c_str());
    // to avoid graph node same, add pp name prefix to node
    AddPrefixForGraphNodeName(temp_graph, process_point.name());
  }
  if (!graph_pp_cfg.inputs_tensor_desc_config.empty()) {
    GE_CHK_STATUS_RET(UpdateGraphInputsDesc(graph_pp_cfg, temp_graph), "Failed to update graph[%s] inputs desc.",
                      temp_graph->GetName().c_str());
  }
  GE_CHK_STATUS_RET(AddGraphInfoForCache(data_flow_graph, temp_graph, graph_pp_cfg.build_options),
                    "Failed to add graph info for cache, graph[%s], data flow graph[%s].",
                    temp_graph->GetName().c_str(), data_flow_graph.GetName().c_str());
  GE_CHK_STATUS_RET(SetEschedPriorities(temp_graph, node->GetOpDesc()),
                    "Failed to set esched priorities for graph[%s], node[%s]", temp_graph->GetName().c_str(),
                    node->GetName().c_str());
  GE_CHK_STATUS_RET(SetFlowAttrToGraph(node, temp_graph), "Failed to set flow attr to graph:%s",
                    temp_graph->GetName().c_str());
  data_flow_graph.subgraphs_[process_point.name()] = temp_graph;
  data_flow_graph.node_subgraphs_[node->GetName()].emplace_back(temp_graph);
  GE_CHK_STATUS_RET(TransferInputShapeToRange(temp_graph, graph_pp_cfg.build_options),
                    "Failed to transfer input shape option to dynamic shape range option.");
  data_flow_graph.graphs_build_options_[process_point.name()] = graph_pp_cfg.build_options;
  GE_DUMP(temp_graph, "LoadGraphProcessPoint");
  std::string trace_log = "loading graph point [" + process_point.name() + "] on node[" + node->GetName() + "]";
  GE_COMPILE_TRACE_TIMESTAMP_END(LoadGraphProcessPoint, trace_log.c_str());
  return SUCCESS;
}

Status ProcessPointLoader::TransferInputShapeToRange(const ComputeGraphPtr &graph,
    std::map<std::string, std::string> &build_options) {
  const auto option_iter = build_options.find(INPUT_SHAPE);
  if ((option_iter == build_options.cend()) || (option_iter->second.empty())) {
    return SUCCESS;
  }
  std::string empty_string;
  std::string input_shape = option_iter->second;
  std::string input_shape_range;
  GE_CHK_STATUS_RET(CheckAndTransferInputShapeToRange(input_shape, input_shape_range,
    empty_string, empty_string, empty_string), "Check and transfer input shape to range failed.");
  GELOGI("Transfer to input shape to range string success. ShapeRange=%s", input_shape_range.c_str());

  build_options[OPTION_EXEC_DYNAMIC_EXECUTE_MODE] = "dynamic_execute";

  if (input_shape_range.find(":") != std::string::npos) {
    // input_shape include node name data0:[xx,xx];data1:[xx,xx]
    std::vector<std::string> shape_range_vec = StringUtils::Split(input_shape_range, ';');
    std::map<std::string, std::string> name_to_range;
    for (const auto &shape_range_item : shape_range_vec) {
      size_t pos = shape_range_item.rfind(":");
      if (pos != std::string::npos) {
        const std::string node_name = shape_range_item.substr(0, pos);
        const std::string range = shape_range_item.substr(pos + 1, shape_range_item.size() - pos);
        GE_CHK_BOOL_RET_STATUS((!node_name.empty()) && (!range.empty()), FAILED,
                               "Name or range is empty, shape_range_item %s", shape_range_item.c_str());
        name_to_range[node_name] = range;
        GELOGD("Parse node name %s range %s", node_name.c_str(), range.c_str());
      }
    }
    std::string temp_shape_range;
    for (const auto &node : graph->GetDirectNode()) {
      GE_CHECK_NOTNULL(node);
      if (OpTypeUtils::IsDataNode(node->GetType())) {
        const auto iter = name_to_range.find(node->GetName());
        if (iter == name_to_range.cend()) {
          GELOGE(FAILED, "Can not find node %s in input range config %s",
                 node->GetName().c_str(), input_shape_range.c_str());
          return FAILED;
        }
        temp_shape_range += iter->second + ",";
      }
    }
    input_shape_range = temp_shape_range.substr(0, temp_shape_range.length() - 1UL);
  }
  build_options[OPTION_EXEC_DATA_INPUTS_SHAPE_RANGE] = input_shape_range;
  GELOGI("Set option %s value %s", OPTION_EXEC_DATA_INPUTS_SHAPE_RANGE, input_shape_range.c_str());
  return SUCCESS;
}

void ProcessPointLoader::AddPrefixForGraphNodeName(ComputeGraphPtr &graph, const std::string &prefix_name) {
  for (const NodePtr &node : graph->GetAllNodes()) {
    if ((node->GetOpDesc() == nullptr) || (node->GetType() == DATA) || (node->GetType() == NETOUTPUT) ||
        OpTypeUtils::IsVarLikeNode(node->GetType())) {
      continue;
    }
    node->GetOpDesc()->SetName(prefix_name + "/" + node->GetName());
  }
}

Status ProcessPointLoader::AddGraphInfoForCache(const DataFlowGraph &data_flow_graph, const ComputeGraphPtr &graph,
                                                const std::map<std::string, std::string> &build_options) {
  if (!data_flow_graph.EnableCache() || data_flow_graph.CacheManualCheck()) {
    GELOGI("Cache is disable, or cache need manual check, no need add graph info[%s] for data flow graph[%s].",
           graph->GetName().c_str(), data_flow_graph.GetName().c_str());
    return SUCCESS;
  }
  std::string session_graph_id;
  (void)AttrUtils::GetStr(*graph, ATTR_NAME_SESSION_GRAPH_ID, session_graph_id);
  (void)graph->DelAttr(ATTR_NAME_SESSION_GRAPH_ID);
  Model model = Model();
  model.SetGraph(graph);
  Buffer buffer;
  GE_CHK_STATUS_RET(model.Save(buffer), "Failed to add graph info[%s] for data flow graph[%s].",
                    graph->GetName().c_str(), data_flow_graph.GetName().c_str());
  (void)AttrUtils::SetStr(*graph, ATTR_NAME_SESSION_GRAPH_ID, session_graph_id);
  unsigned char sha256[SHA256_DIGEST_LENGTH];
  (void)SHA256(PtrToPtr<uint8_t, unsigned char>(buffer.GetData()), buffer.GetSize(), sha256);
  std::string result = "";
  std::stringstream ss;
  for (size_t i = 0U; i < SHA256_DIGEST_LENGTH; ++i) {
    ss << std::hex << int(sha256[i]);
  }
  result = ss.str();
  GE_CHK_BOOL_RET_STATUS(graph->SetExtAttr("_graph_info_for_data_flow_cache", result), FAILED,
                         "Failed to set ext attr[_graph_info_for_data_flow_cache] for graph[%s].",
                         graph->GetName().c_str());
  GELOGI("Set ext attr[_graph_info_for_data_flow_cache] value[%s] to graph[%s] successfully.", result.c_str(),
         graph->GetName().c_str());
  GE_CHK_BOOL_RET_STATUS(graph->SetExtAttr("_graph_build_options_for_data_flow_cache", build_options), FAILED,
                         "Failed to set ext attr[_graph_build_options_for_data_flow_cache] for graph[%s].",
                         graph->GetName().c_str());
  return SUCCESS;
}

Status ProcessPointLoader::SetEschedPriority(const ComputeGraphPtr &graph, const OpDescPtr &op_desc,
                                             const std::string &priority_name) {
  if (AttrUtils::HasAttr(op_desc, priority_name)) {
    int64_t priority = 0;
    GE_CHK_BOOL_RET_STATUS(AttrUtils::GetInt(op_desc, priority_name, priority), FAILED,
                           "Failed to get attr[%s] from op[%s].", priority_name.c_str(), op_desc->GetName().c_str());
    if ((priority < kHighestPriority) || (priority > kLowestPriority)) {
      GELOGE(PARAM_INVALID, "Attr[%s] value should in [%ld, %ld], but got [%ld].", priority_name.c_str(),
             kHighestPriority, kLowestPriority, priority);
      return PARAM_INVALID;
    }
    GELOGD("[ModelEschedPriority]: op name[%s], attr name[%s], value[%ld]", op_desc->GetName().c_str(),
           priority_name.c_str(), priority);
    GE_CHK_BOOL_RET_STATUS(AttrUtils::SetInt(graph, priority_name, priority), FAILED,
                           "Failed to set attr[%s], value[%ld] to graph[%s].", priority_name.c_str(), priority,
                           graph->GetName().c_str());
  }
  return SUCCESS;
}

Status ProcessPointLoader::SetEschedPriorities(const ComputeGraphPtr &graph, const OpDescPtr &op_desc) {
  GE_CHK_STATUS_RET(SetEschedPriority(graph, op_desc, ATTR_NAME_ESCHED_PROCESS_PRIORITY),
                    "Failed to set attr[%s] for op[%s], graph[%s].", ATTR_NAME_ESCHED_PROCESS_PRIORITY.c_str(),
                    op_desc->GetName().c_str(), graph->GetName().c_str());
  GE_CHK_STATUS_RET(SetEschedPriority(graph, op_desc, ATTR_NAME_ESCHED_EVENT_PRIORITY),
                    "Failed to set attr[%s] for op[%s], graph[%s].", ATTR_NAME_ESCHED_EVENT_PRIORITY.c_str(),
                    op_desc->GetName().c_str(), graph->GetName().c_str());
  return SUCCESS;
}

Status ProcessPointLoader::SetDataFlowScope(const OpDescPtr &flow_func_desc, const std::string &data_flow_scope) {
  GE_CHECK_NOTNULL(flow_func_desc);
  GE_CHK_BOOL_RET_STATUS(AttrUtils::SetStr(flow_func_desc, ATTR_NAME_DATA_FLOW_DATA_FLOW_SCOPE, data_flow_scope),
                         FAILED, "Failed to set attr[%s] for func[%s].", ATTR_NAME_DATA_FLOW_DATA_FLOW_SCOPE,
                         flow_func_desc->GetNamePtr());
  GELOGD("func[%s] attr[%s] is set to %s", flow_func_desc->GetNamePtr(), ATTR_NAME_DATA_FLOW_DATA_FLOW_SCOPE,
         data_flow_scope.c_str());
  return SUCCESS;
}

Status ProcessPointLoader::SetDataFlowInvokedScopes(const OpDescPtr &flow_func_desc,
                                                    const std::vector<std::string> &invoked_scopes) {
  GE_CHECK_NOTNULL(flow_func_desc);
  GE_CHK_BOOL_RET_STATUS(AttrUtils::SetListStr(flow_func_desc,
                         ATTR_NAME_DATA_FLOW_DATA_FLOW_INVOKED_SCOPES, invoked_scopes),
                         FAILED, "Failed to set attr[%s] for func[%s].", ATTR_NAME_DATA_FLOW_DATA_FLOW_INVOKED_SCOPES,
                         flow_func_desc->GetNamePtr());
  GELOGD("func[%s] attr[%s] is set, size %zu", flow_func_desc->GetNamePtr(),
         ATTR_NAME_DATA_FLOW_DATA_FLOW_INVOKED_SCOPES, invoked_scopes.size());
  return SUCCESS;
}
}  // namespace ge
