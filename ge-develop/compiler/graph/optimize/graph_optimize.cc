/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/optimize/graph_optimize.h"

#include "graph/ge_context.h"
#include "graph/passes/standard_optimize/constant_folding/dimension_adjust_pass.h"
#include "graph/passes/pass_manager.h"
#include "api/gelib/gelib.h"
#include "graph/partition/engine_place.h"
#include "graph/utils/op_desc_utils_ex.h"
#include "common/util/trace_manager/trace_manager.h"
#include "common/checker.h"
#include "common/helper/model_parser_base.h"
#include "common/helper/model_helper.h"
#include "graph_utils_ex.h"
#include "common/compile_profiling/ge_trace_wrapper.h"
#include "base/err_msg.h"
#include "graph/fusion/pass/fusion_pass_executor.h"

namespace {
const char *const kVectorCore = "VectorCore";
const char *const kVectorEngine = "VectorEngine";
const char *const kAicoreEngine = "AIcoreEngine";
const char *const kHostCpuEngine = "DNN_VM_HOST_CPU";
}  // namespace

namespace ge {
GraphOptimize::GraphOptimize()
    : optimize_type_(domi::FrameworkType::TENSORFLOW),
      cal_config_(""),
      insert_op_config_(""),
      core_type_(""),
      exclude_engines_({kVectorEngine}) {}

void AddNodeInputProperty(const ComputeGraphPtr &compute_graph) {
  for (ge::NodePtr &node : compute_graph->GetDirectNode()) {
    auto node_op_desc = node->GetOpDesc();
    GE_IF_BOOL_EXEC(node_op_desc == nullptr, GELOGW("node_op_desc is nullptr!"); return;);
    auto in_control_anchor = node->GetInControlAnchor();
    std::vector<std::string> src_name_list;
    std::vector<std::string> input_name_list;
    std::vector<int64_t> src_index_list;
    GE_IF_BOOL_EXEC(
      in_control_anchor != nullptr, std::string src_name_temp; for (auto &out_control_anchor
                                                               : in_control_anchor->GetPeerOutControlAnchors()) {
        GE_IF_BOOL_EXEC(out_control_anchor == nullptr, GELOGW("out_control_anchor is nullptr!"); continue);
        ge::NodePtr src_node = out_control_anchor->GetOwnerNode();
        GE_IF_BOOL_EXEC(src_node == nullptr, GELOGW("src_node is nullptr!"); continue);
        src_name_temp = src_name_temp == "" ? src_node->GetName() : src_name_temp + ":" + src_node->GetName();
      } GE_IF_BOOL_EXEC(src_name_temp != "", src_name_list.emplace_back(src_name_temp);
                        node_op_desc->SetSrcName(src_name_list);))

    for (auto &in_data_anchor : node->GetAllInDataAnchors()) {
      GE_IF_BOOL_EXEC(in_data_anchor == nullptr, continue);
      auto peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
      GE_IF_BOOL_EXEC(peer_out_anchor == nullptr, continue);

      ge::NodePtr src_node = peer_out_anchor->GetOwnerNode();
      GE_IF_BOOL_EXEC(src_node == nullptr, continue);
      src_index_list = node_op_desc->GetSrcIndex();
      src_name_list.emplace_back(src_node->GetName());
      src_index_list.emplace_back(peer_out_anchor->GetIdx());
      node_op_desc->SetSrcName(src_name_list);
      node_op_desc->SetSrcIndex(src_index_list);
      GE_IF_BOOL_EXEC(!(node_op_desc->GetType() == NETOUTPUT && GetLocalOmgContext().type == domi::TENSORFLOW),
                      ge::NodePtr peer_owner_node = peer_out_anchor->GetOwnerNode();
                      GE_IF_BOOL_EXEC(peer_owner_node == nullptr, continue);
                      input_name_list.emplace_back(
                        peer_owner_node->GetName() +
                        (peer_out_anchor->GetIdx() == 0 ? "" : ": " + to_string(peer_out_anchor->GetIdx())));
                      node_op_desc->SetInputName(input_name_list);)
    }
  }
}

Status GraphOptimize::OptimizeSubGraph(ComputeGraphPtr &compute_graph, const std::string &engine_name) const {
  if (compute_graph == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Param compute_graph is nullptr, check invalid");
    GELOGE(GE_GRAPH_OPTIMIZE_COMPUTE_GRAPH_NULL, "[Check][Param] compute_graph is nullptr.");
    return GE_GRAPH_OPTIMIZE_COMPUTE_GRAPH_NULL;
  }

  std::vector<GraphOptimizerPtr> graph_optimizer;
  if (DNNEngineManager::GetInstance().IsEngineRegistered(engine_name)) {
    OpsKernelManager::GetInstance().GetGraphOptimizerByEngine(engine_name, graph_optimizer);
    AddNodeInputProperty(compute_graph);

    if (compute_graph->GetDirectNode().size() == 0) {
      GELOGW("[OptimizeSubGraph] compute_graph do not has any node.");
      return SUCCESS;
    }

    if (build_mode_ == BUILD_MODE_TUNING && (build_step_ == BUILD_STEP_AFTER_UB_MATCH
      || build_step_ == BUILD_STEP_AFTER_MERGE)) {
      for (auto iter = graph_optimizer.begin(); iter != graph_optimizer.end(); ++iter) {
        TraceOwnerGuard guard("GE", "OptimizeFusedGraphAfterGraphSlice", compute_graph->GetName());
        Status ret = (*iter)->OptimizeFusedGraphAfterGraphSlice(*(compute_graph));
        if (ret != SUCCESS) {
          REPORT_INNER_ERR_MSG("E19999", "Call OptimizeFusedGraphAfterGraphSlice failed, ret:%u, engine_name:%s, "
                             "graph_name:%s", ret, engine_name.c_str(),
                             compute_graph->GetName().c_str());
          GELOGE(ret, "[Call][OptimizeFusedGraphAfterGraphSlice] failed, ret:%u, engine_name:%s, graph_name:%s",
                 ret, engine_name.c_str(), compute_graph->GetName().c_str());
          return ret;
        }
      }
      return SUCCESS;
    }

    for (auto iter = graph_optimizer.begin(); iter != graph_optimizer.end(); ++iter) {
      TraceOwnerGuard guard("GE", "OptimizeFusedGraph", compute_graph->GetName());
      Status ret = (*iter)->OptimizeFusedGraph(*(compute_graph));
      if (ret != SUCCESS) {
        REPORT_INNER_ERR_MSG("E19999", "Call OptimizeFusedGraph failed, ret:%u, engine_name:%s, "
                           "graph_name:%s", ret, engine_name.c_str(),
                           compute_graph->GetName().c_str());
        GELOGE(ret, "[Optimize][FusedGraph] failed, ret:%u, engine_name:%s, graph_name:%s",
               ret, engine_name.c_str(), compute_graph->GetName().c_str());
        return ret;
      }
    }
  } else {
    GELOGI("Engine:[%s] is not registered. do nothing in subGraph Optimize by ATC.", engine_name.c_str());
  }
  return SUCCESS;
}

Status GraphOptimize::OptimizeGraphInit(const ComputeGraphPtr &compute_graph) const {
  GE_CHECK_NOTNULL(compute_graph);
  std::shared_ptr<GELib> instance_ptr = ge::GELib::GetInstance();
  if (instance_ptr == nullptr || !instance_ptr->InitFlag()) {
    REPORT_INNER_ERR_MSG("E19999", "Gelib not init before, check invalid.");
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "[Get][GELib] Gelib not init before.");
    return GE_CLI_GE_NOT_INITIALIZED;
  }
  // init process for optimize graph every time because options may different in different build process
  // 当前引擎获取编译option是在OptimizeGraphPrepare接口中获取，该接口默认会过滤vector engine。
  // 当前出现问题场景是子图优化阶段因为算子融合直接选择了vector engine的场景，出现了vector engine获取不到编译option导致问题。
  // 当前决策新增OptimizeGraphInit接口，该接口不会过滤引擎，全部调用.这样获取到build option操作就从OptimizeGraphPrepare剥离。
  auto graph_optimizer = instance_ptr->OpsKernelManagerObj().GetAllGraphOptimizerObjsByPriority();
  GELOGD("OptimizeGraphInit by opskernel, num of graph_optimizer is %zu.", graph_optimizer.size());

  for (auto &iter : graph_optimizer) {
    GE_ASSERT_SUCCESS((iter.second)->OptimizeGraphInit(*compute_graph));
  }
  return SUCCESS;
}

Status GraphOptimize::FinalizeSessionInfo(const ComputeGraphPtr &compute_graph) const {
  std::shared_ptr<GELib> instance_ptr = ge::GELib::GetInstance();
  if (instance_ptr == nullptr || !instance_ptr->InitFlag()) {
    GELOGI("instance_ptr: nullptr or not initialized.");
    return SUCCESS;
  }

  auto graph_optimizer = instance_ptr->OpsKernelManagerObj().GetAllGraphOptimizerObjsByPriority();
  GELOGD("FinalizeSessionInfo by graph manager, num of graph_optimizer is %zu.", graph_optimizer.size());

  for (auto &iter : graph_optimizer) {
    Status ret = (iter.second)->FinalizeSessionInfo(*compute_graph);
    if (ret != SUCCESS) {
      GELOGE(ret, "[Optimize][FinalizeSessionInfo] failed, ret:%u, engine_name:%s, graph_name:%s",
             ret, iter.first.c_str(), compute_graph->GetName().c_str());
      return ret;
    }
  }
  return SUCCESS;
}

Status GraphOptimize::OptimizeOriginalGraph(const ComputeGraphPtr &compute_graph) const {
  GE_CHECK_NOTNULL(compute_graph);
  std::vector<std::pair<std::string, GraphOptimizerPtr>> graph_optimizer;
  const bool hostExecFlag = GetContext().GetHostExecFlag();
  Status ret = GetValidGraphOptimizer(graph_optimizer);
  if (ret == SUCCESS) {
    for (auto iter = graph_optimizer.begin(); iter != graph_optimizer.end(); ++iter) {
      if (hostExecFlag && (iter->first != kHostCpuEngine)) {
        // graph exec on host, no need OptimizeOriginalGraph for other engine.
        continue;
      }
      ret = (iter->second)->OptimizeOriginalGraph(*compute_graph);
      if (ret != SUCCESS) {
        REPORT_INNER_ERR_MSG("E19999", "Call OptimizeOriginalGraph failed, ret:%u, engine_name:%s, "
                           "graph_name:%s", ret, iter->first.c_str(),
                           compute_graph->GetName().c_str());
        GELOGE(ret, "[Optimize][OriginalGraph] failed, ret:%u, engine_name:%s, graph_name:%s",
               ret, iter->first.c_str(), compute_graph->GetName().c_str());
        return ret;
      }
    }
  }
  fusion::FusionPassExecutor fusion_pass_executor;
  GE_TRACE_START(RunCustomPassAfterBuiltinFusionPass);
  GE_ASSERT_SUCCESS(fusion_pass_executor.RunPassesWithLegacyCustom(compute_graph, CustomPassStage::kAfterBuiltinFusionPass),
                    "Run after built-in fusion pass for graph [%s] failed.", compute_graph->GetName().c_str());
  GE_COMPILE_TRACE_TIMESTAMP_END(RunCustomPassAfterBuiltinFusionPass, "GraphManager::RunCustomPassAfterBuiltinFusionPass");
  GE_DUMP(compute_graph, "RunCustomPassAfterBuiltinFusionPass");
  return ret;
}

Status GraphOptimize::GetValidGraphOptimizer(
    std::vector<std::pair<std::string, GraphOptimizerPtr>> &valid_optimizer) const {
  std::shared_ptr<GELib> instance_ptr = ge::GELib::GetInstance();
  if (instance_ptr == nullptr || !instance_ptr->InitFlag()) {
    REPORT_INNER_ERR_MSG("E19999", "Gelib not init before, check invalid.");
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "[Get][GELib] Gelib not init before.");
    return GE_CLI_GE_NOT_INITIALIZED;
  }

  auto graph_optimizer = instance_ptr->OpsKernelManagerObj().GetAllGraphOptimizerObjsByPriority();
  GELOGD("optimize by opskernel, num of graph_optimizer is %zu.", graph_optimizer.size());

  for (auto &iter : graph_optimizer) {
    if (IsExclude(iter.first) || (iter.second == nullptr)) {
      GELOGI("[OptimizeOriginalGraphJudgeInsert]: engine type will exclude: %s", iter.first.c_str());
      continue;
    }
    valid_optimizer.emplace_back(iter);
  }
  return SUCCESS;
}

Status GraphOptimize::OptimizeAfterGraphNormalization(const ComputeGraphPtr& compute_graph) const {
  GE_CHECK_NOTNULL(compute_graph);

  std::vector<std::pair<std::string, GraphOptimizerPtr>> graph_optimizer;
  Status ret = GetValidGraphOptimizer(graph_optimizer);
  if (ret != SUCCESS) {
    return ret;
  }

  for (auto iter = graph_optimizer.begin(); iter != graph_optimizer.end(); ++iter) {
    GELOGI("Begin to optimize whole graph by engine %s", iter->first.c_str());
    ret = (iter->second)->OptimizeAfterGraphNormalization(compute_graph);
    if (ret != SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "Call OptimizeAfterGraphNormalization failed, engine_name:%s, graph_name:%s",
                         iter->first.c_str(), compute_graph->GetName().c_str());
      GELOGE(ret, "[Call][OptimizeAfterGraphNormalization] failed, engine_name:%s, graph_name:%s",
             iter->first.c_str(), compute_graph->GetName().c_str());
      return ret;
    }
  }

  return ret;
}

Status GraphOptimize::OptimizeOriginalGraphJudgePrecisionInsert(const ComputeGraphPtr &compute_graph) const {
  GELOGD("OptimizeOriginalGraphJudgePrecisionInsert in");

  GE_CHECK_NOTNULL(compute_graph);
  std::vector<std::pair<std::string, GraphOptimizerPtr>> graph_optimizer;
  const bool host_exec_flag = GetContext().GetHostExecFlag();
  Status ret = GetValidGraphOptimizer(graph_optimizer);
  if (ret == SUCCESS) {
    for (auto iter = graph_optimizer.begin(); iter != graph_optimizer.end(); ++iter) {
      if (host_exec_flag && (iter->first != kHostCpuEngine)) {
        // graph exec on host, no need OptimizeOriginalGraphJudgePrecisionInsert for other engine.
        continue;
      }
      GELOGD("Begin to refine running precision by engine %s for graph %s", iter->first.c_str(),
             compute_graph->GetName().c_str());
      ret = (iter->second)->OptimizeOriginalGraphJudgeInsert(*compute_graph);
      if (ret != SUCCESS) {
        REPORT_INNER_ERR_MSG("E19999", "Call OptimizeOriginalGraphJudgeInsert failed, ret:%u, engine_name:%s, "
                           "graph_name:%s", ret, iter->first.c_str(),
                           compute_graph->GetName().c_str());
        GELOGE(ret, "[Call][OptimizeOriginalGraphJudgeInsert] failed, ret:%u, engine_name:%s, graph_name:%s",
               ret, iter->first.c_str(), compute_graph->GetName().c_str());
        return ret;
      }
    }
  }
  return ret;
}

Status GraphOptimize::OptimizeOriginalGraphJudgeFormatInsert(const ComputeGraphPtr &compute_graph) const {
  GELOGD("OptimizeOriginalGraphJudgeFormatInsert in");
  GE_CHECK_NOTNULL(compute_graph);
  std::vector<std::pair<std::string, GraphOptimizerPtr>> graph_optimizer;
  const bool host_exec_flag = GetContext().GetHostExecFlag();
  Status ret = GetValidGraphOptimizer(graph_optimizer);
  if (ret == SUCCESS) {
    for (auto iter = graph_optimizer.begin(); iter != graph_optimizer.end(); ++iter) {
      if (host_exec_flag && (iter->first != kHostCpuEngine)) {
        // graph exec on host, no need OptimizeOriginalGraphJudgeFormatInsert for other engine.
        continue;
      }
      GELOGD("Begin to refine running format by engine %s for graph %s", iter->first.c_str(),
             compute_graph->GetName().c_str());
      GE_ASSERT_SUCCESS((iter->second)->OptimizeOriginalGraphJudgeFormatInsert(*compute_graph),
          "Call OptimizeOriginalGraphJudgeFormatInsert failed, engine_name:%s, graph_name:%s",
          iter->first.c_str(), compute_graph->GetName().c_str());
    }
  }
  return ret;
}

Status GraphOptimize::OptimizeOriginalGraphForQuantize(const ComputeGraphPtr &compute_graph) const {
  GE_CHECK_NOTNULL(compute_graph);
  std::vector<std::pair<std::string, GraphOptimizerPtr>> graph_optimizer;
  Status ret = GetValidGraphOptimizer(graph_optimizer);
  if (ret == SUCCESS) {
    for (auto iter = graph_optimizer.begin(); iter != graph_optimizer.end(); ++iter) {
      ret = iter->second->OptimizeGraphPrepare(*compute_graph);
      if (ret != SUCCESS) {
        REPORT_INNER_ERR_MSG("E19999", "Call OptimizeGraphPrepare failed, ret:%u, engine_name:%s, "
                           "graph_name:%s", ret, iter->first.c_str(),
                           compute_graph->GetName().c_str());
        GELOGE(ret, "[Call][OptimizeGraphPrepare] failed, ret:%u, engine_name:%s, graph_name:%s",
               ret, iter->first.c_str(), compute_graph->GetName().c_str());
        return ret;
      }
    }
  }
  return ret;
}

Status GraphOptimize::OptimizeGraphBeforeBuild(const ComputeGraphPtr &compute_graph) const {
  if (compute_graph == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Param compute_graph is nullptr, check invalid");
    GELOGE(GE_GRAPH_OPTIMIZE_COMPUTE_GRAPH_NULL, "[Check][Param] compute_graph is nullptr.");
    return GE_GRAPH_OPTIMIZE_COMPUTE_GRAPH_NULL;
  }

  EnginePlacer engine_place(compute_graph);
  Status ret = engine_place.RunAllSubgraphs();
  if (ret != SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Assign atomic engine for graph %s failed", compute_graph->GetName().c_str());
    GELOGE(ret, "[Assign][Engine] Assign atomic engine for graph %s failed", compute_graph->GetName().c_str());
    return ret;
  }
  ret = engine_place.AssignCompositeEngine();
  if (ret != SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Assign composite engine for graph %s failed", compute_graph->GetName().c_str());
    GELOGE(ret, "[Assign][Engine] Assign composite engine for graph %s failed", compute_graph->GetName().c_str());
    return ret;
  }

  std::vector<std::pair<std::string, GraphOptimizerPtr>> graph_optimizer;
  ret = GetValidGraphOptimizer(graph_optimizer);
  if (ret == SUCCESS) {
    for (auto iter = graph_optimizer.begin(); iter != graph_optimizer.end(); ++iter) {
      ret = iter->second->OptimizeGraphBeforeBuild(*compute_graph);
      if (ret != SUCCESS) {
        REPORT_INNER_ERR_MSG("E19999", "Call OptimizeGraphBeforeBuild failed, ret:%u, engine_name:%s, "
                           "graph_name:%s", ret, iter->first.c_str(),
                           compute_graph->GetName().c_str());
        GELOGE(ret, "[Call][OptimizeGraphBeforeBuild] failed, ret:%u, engine_name:%s, graph_name:%s",
               ret, iter->first.c_str(), compute_graph->GetName().c_str());
        return ret;
      }
    }
  }
  return ret;
}

Status GraphOptimize::OptimizeAfterStage1(const ComputeGraphPtr &compute_graph) const {
  GE_CHECK_NOTNULL(compute_graph);
  GELOGD("OptimizeAfterStage1 in");
  if (GetContext().GetHostExecFlag()) {
    // graph exec on host, no need OptimizeAfterStage1
    return SUCCESS;
  }

  std::vector<std::pair<std::string, GraphOptimizerPtr>> graph_optimizer;
  Status ret = GetValidGraphOptimizer(graph_optimizer);
  if (ret == SUCCESS) {
    for (auto iter = graph_optimizer.begin(); iter != graph_optimizer.end(); ++iter) {
      GELOGI("Begin to optimize graph after stage1 by engine %s.", iter->first.c_str());
      ret = (iter->second)->OptimizeAfterStage1(*compute_graph);
      if (ret != SUCCESS) {
        REPORT_INNER_ERR_MSG("E19999", "Call OptimizeAfterStage1 failed, ret:%u, engine_name:%s, "
                           "graph_name:%s.", ret, iter->first.c_str(), compute_graph->GetName().c_str());
        GELOGE(ret, "[OptimizeAfterStage1]: graph optimize failed, ret:%u.", ret);
        return ret;
      }
    }
  }
  return ret;
}

Status GraphOptimize::SetOptions(const ge::GraphManagerOptions &options) {
  if (options.framework_type >= static_cast<int32_t>(domi::FrameworkType::FRAMEWORK_RESERVED)) {
    REPORT_PREDEFINED_ERR_MSG("E10041", std::vector<const char_t *>({"parameter"}),
                       std::vector<const char_t *>({"options.framework_type"}));
    GELOGE(GE_GRAPH_OPTIONS_INVALID, "Optimize Type %d invalid.", options.framework_type);
    return GE_GRAPH_OPTIONS_INVALID;
  }
  optimize_type_ = static_cast<domi::FrameworkType>(options.framework_type);
  cal_config_ = options.calibration_conf_file;
  insert_op_config_ = options.insert_op_file;
  local_fmk_op_flag_ = options.local_fmk_op_flag;
  func_bin_path_ = options.func_bin_path;
  core_type_ = options.core_type;
  build_mode_ = options.build_mode;
  build_step_ = options.build_step;
  exclude_engines_ = options.exclude_engines;
  return SUCCESS;
}

bool GraphOptimize::IsExclude(const std::string &engine_name) const {
  return exclude_engines_.find(engine_name) != exclude_engines_.end();
}

void GraphOptimize::TranFrameOp(const ComputeGraphPtr &compute_graph) const {
  GE_CHECK_NOTNULL_JUST_RETURN(compute_graph);
  std::vector<std::string> local_framework_op_vec = {
    "TensorDataset", "QueueDataset", "DeviceQueueDataset", "ParallelMapDataset", "BatchDatasetV2",
    "IteratorV2",    "MakeIterator", "IteratorGetNext",    "FilterDataset",      "MapAndBatchDatasetV2"};
  for (auto &nodePtr : compute_graph->GetAllNodes()) {
    OpDescPtr op = nodePtr->GetOpDesc();
    GE_IF_BOOL_EXEC(op == nullptr, GELOGW("op is nullptr!"); continue);
    // fwkop black-white sheet
    std::vector<std::string>::iterator iter =
      std::find(local_framework_op_vec.begin(), local_framework_op_vec.end(), op->GetType());
    if (iter != local_framework_op_vec.end()) {
      // set - original_type
      if (!AttrUtils::SetStr(op, ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, op->GetType())) {
        GELOGW("TranFrameOp SetStr ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE failed");
      }
      // set - framework_type
      // [No need to verify return value]
      ge::OpDescUtilsEx::SetType(op, "FrameworkOp");
      if (!AttrUtils::SetInt(op, ATTR_NAME_FRAMEWORK_FWK_TYPE, domi::FrameworkType::TENSORFLOW)) {
        GELOGW("TranFrameOp SetInt ATTR_NAME_FRAMEWORK_FWK_TYPE failed");
      }
    }
  }
}

Status GraphOptimize::IdentifyReference(const ComputeGraphPtr &compute_graph) const {
  for (auto &node : compute_graph->GetAllNodes()) {
    GE_CHECK_NOTNULL(node);
    auto op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    auto input_name_index = op_desc->GetAllInputName();
    bool is_ref = false;
    for (const auto &name_index : input_name_index) {
      const int32_t out_index = op_desc->GetOutputIndexByName(name_index.first);
      if (out_index != -1) {
        const auto &input_desc = op_desc->MutableInputDesc(name_index.second);
        if (input_desc == nullptr) {
          continue;
        }
        input_desc->SetRefPortByIndex({name_index.second});
        GELOGI("SetRefPort: set op[%s] input desc[%u-%s] ref.",
               op_desc->GetName().c_str(), name_index.second, name_index.first.c_str());
        auto output_desc = op_desc->GetOutputDesc(static_cast<uint32_t>(out_index));
        output_desc.SetRefPortByIndex({name_index.second});
        op_desc->UpdateOutputDesc(static_cast<uint32_t>(out_index), output_desc);
        GELOGI("SetRefPort: set op[%s] output desc[%u-%s] ref.",
               op_desc->GetName().c_str(), out_index, name_index.first.c_str());
        is_ref = true;
      }
    }
    if (is_ref) {
      AttrUtils::SetBool(op_desc, ATTR_NAME_REFERENCE, is_ref);
      GELOGI("param [node] %s is reference node, set attribute %s to be true.",
             node->GetName().c_str(), ATTR_NAME_REFERENCE.c_str());
    }
  }
  return SUCCESS;
}
Status GraphOptimize::OptimizeWholeGraph(const ComputeGraphPtr &compute_graph) const {
  GE_CHECK_NOTNULL(compute_graph);
  std::vector<std::pair<std::string, GraphOptimizerPtr>> graph_optimizer;
  Status ret = GetValidGraphOptimizer(graph_optimizer);
  if (ret == SUCCESS) {
    for (auto &iter : graph_optimizer) {
      GELOGI("Begin to optimize whole graph by engine %s", iter.first.c_str());
      ret = iter.second->OptimizeWholeGraph(*compute_graph);
      GE_DUMP(compute_graph, "OptimizeWholeGraph" + iter.first);
      if (ret != SUCCESS) {
        REPORT_INNER_ERR_MSG("E19999", "Call OptimizeWholeGraph failed, ret:%u, engine_name:%s, "
                           "graph_name:%s", ret, iter.first.c_str(),
                           compute_graph->GetName().c_str());
        GELOGE(ret, "[Call][OptimizeWholeGraph] failed, ret:%d, engine_name:%s, graph_name:%s",
               ret, iter.first.c_str(), compute_graph->GetName().c_str());
        return ret;
      }
    }
  }
  return ret;
}

Status GraphOptimize::OptimizeSubgraphProc(ComputeGraph &graph, const bool is_pre) const {
  std::vector<std::pair<std::string, GraphOptimizerPtr>> graph_optimizer;
  Status ret = GetValidGraphOptimizer(graph_optimizer);
  if (ret != SUCCESS) {
    return ret;
  }

  const std::string proc_str = (is_pre ? "pre" : "post");
  for (auto &iter : graph_optimizer) {
    GELOGI("Begin subgraph %s proc by engine %s", proc_str.c_str(), iter.first.c_str());
    if (is_pre) {
      ret = iter.second->OptimizeSubgraphPreProc(graph);
    } else {
      ret = iter.second->OptimizeSubgraphPostProc(graph);
    }
    GE_ASSERT_SUCCESS(ret, "Subgraph %s proc failed, ret:%d, engine_name:%s, graph_name:%s",
                      proc_str.c_str(), ret, iter.first.c_str(), graph.GetName().c_str());
  }
  return ret;
}

Status GraphOptimize::OptimizeSubgraphPreProc(ComputeGraph &graph) const {
  return OptimizeSubgraphProc(graph, true);
}

Status GraphOptimize::OptimizeSubgraphPostProc(ComputeGraph &graph) const {
  return OptimizeSubgraphProc(graph, false);
}
}  // namespace ge
