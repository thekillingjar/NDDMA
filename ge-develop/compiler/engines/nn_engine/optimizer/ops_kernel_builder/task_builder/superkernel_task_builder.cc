/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ops_kernel_builder/task_builder/superkernel_task_builder.h"
#include "common/fe_log.h"
#include "common/scope_allocator.h"
#include "common/fe_op_info_common.h"
#include "common/fe_gentask_utils.h"
#include "common/platform_utils.h"
#include "framework/common/debug/ge_log.h"
#include "ops_kernel_builder/task_builder/task_builder.h"
#include "ops_kernel_builder/task_builder/superkernel_args_format_utils.h"
#include "adapter/common/op_store_adapter_manager.h"
#include "graph_optimizer/json_parser/tbe_json_parse.h"
#include "graph/compute_graph.h"
#include "graph/utils/attr_utils.h"
#include "graph/op_kernel_bin.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/anchor_utils.h"
#include "runtime/stream.h"
#include "graph/args_format_desc.h"
#include "adapter/tbe_adapter/kernel_launch/tbe_kernel_launch.h"
#include "register/op_ext_gentask_registry.h"
#include "framework/common/taskdown_common.h"

namespace fe {
namespace {
const std::string kAttrHasDoneSkp = "_has_done_skp";
const std::string kAttrTaskArgs = "_task_args";
const std::string SPK_REUSED_BINARY = "super_kernel_reuse_binary";
void AddFusionNodes(std::vector<ge::NodePtr> &nodes_vec,
                    std::vector<std::vector<ge::NodePtr>> &fusion_nodes) {
  if (!nodes_vec.empty()) {
    fusion_nodes.push_back(nodes_vec);
    nodes_vec.clear();
  }
}
void ConvertArgsToVec(const void *args, const uint32_t args_size, std::vector<int64_t> &task_args_vec) {
  for (size_t i = 0; i != args_size / sizeof(uint64_t); ++i) {
    uint64_t value = *(reinterpret_cast<uint64_t *>(reinterpret_cast<uintptr_t>(args) + i * sizeof(uint64_t)));
    task_args_vec.push_back(value);
  }
}
void SetFirstOrLastNodeAttr(const ge::OpDescPtr &src_op_desc, const ge::OpDescPtr &dst_op_desc) {
  bool is_first_node = false;
  (void)ge::AttrUtils::GetBool(src_op_desc, ATTR_NAME_IS_FIRST_NODE, is_first_node);
  if (is_first_node) {
    (void)ge::AttrUtils::SetBool(dst_op_desc, ATTR_NAME_IS_FIRST_NODE, true);
  }

  bool is_last_node = false;
  (void)ge::AttrUtils::GetBool(src_op_desc, ATTR_NAME_IS_LAST_NODE, is_last_node);
  if (is_last_node) {
    (void)ge::AttrUtils::SetBool(dst_op_desc, ATTR_NAME_IS_LAST_NODE, true);
  }
}
bool CheckTilingSinkForSK(const ge::Node &node, const bool is_tiling_sink) {
  bool is_superkernel_reuse_binary = false;
  (void)ge::AttrUtils::GetBool(node.GetOpDesc(), SPK_REUSED_BINARY, is_superkernel_reuse_binary);
  FE_LOGD("Node[%s, %s] got attribute super_kernel_reuse_binary[%d].", node.GetNamePtr(), node.GetTypePtr(), is_superkernel_reuse_binary);
  if(is_superkernel_reuse_binary && !is_tiling_sink) {
    return false;
  }
  return true;
}
}

Status SuperkernelTaskBuilder::GenerateKernelTask(const ge::Node &node, const ge::RunContext &context,
                                                  std::vector<domi::TaskDef> &task_defs) {
  int64_t scope_id = 0;
  if (!ScopeAllocator::GetSkpScopeAttr(node.GetOpDesc(), scope_id)) {
    FE_LOGE("Failed to get superkernel plus scope from node[%s, %s].",
            node.GetName().c_str(), node.GetType().c_str());
    return FAILED;
  }
  bool has_skp = false;
  if (ge::AttrUtils::GetBool(node.GetOpDesc(), kAttrHasDoneSkp, has_skp) && has_skp) {
    FE_LOGD("Task has been generated for node[%s, %s].",
            node.GetName().c_str(), node.GetType().c_str());
    return SUCCESS;
  }
  FE_LOGD("Begin to generate superkernel task for node[%s, %s], scope id[%ld].",
          node.GetName().c_str(), node.GetType().c_str(), scope_id);
  std::vector<std::vector<ge::NodePtr>> fusion_nodes;
  if (GetTaskFusionNodes(node, scope_id, fusion_nodes) != SUCCESS) {
    FE_LOGE("Failed to get fusion nodes by node[%s, %s].",
            node.GetName().c_str(), node.GetType().c_str());
    return FAILED;
  }

  TaskBuilderContext task_context;
  if (TaskBuilder::InitTaskContext(context, task_context) != SUCCESS) {
    return FAILED;
  }

  if (GenerateTaskForFusionNodes(fusion_nodes, task_context, task_defs) != SUCCESS) {
    return FAILED;
  }

  FE_LOGD("Finish generating superkernel task for node[%s, %s].", node.GetName().c_str(), node.GetType().c_str());
  return SUCCESS;
}

Status SuperkernelTaskBuilder::GetTaskFusionNodes(const ge::Node &node, const int64_t scope_id,
                                                  std::vector<std::vector<ge::NodePtr>> &fusion_nodes) {
  ge::ComputeGraphPtr graph = node.GetOwnerComputeGraph();
  FE_CHECK_NOTNULL(graph);
  bool has_found_node = false;

  std::vector<ge::NodePtr> nodes_vec;
  for (const ge::NodePtr &node_ptr : graph->GetDirectNode()) {
    if (node_ptr == nullptr) {
      continue;
    }
    if (!has_found_node && node_ptr->GetOpDesc()->GetId() != node.GetOpDesc()->GetId()) {
      continue;
    }
    has_found_node = true;

    int64_t tmp_scopde_id = 0;
    if (ScopeAllocator::GetSkpScopeAttr(node_ptr->GetOpDesc(), tmp_scopde_id)) {
      if (tmp_scopde_id == scope_id) {
        ge::NodePtr memset_node = node_ptr->GetOpDesc()->TryGetExtAttr<ge::NodePtr>(ATTR_NAME_MEMSET_NODE, nullptr);
        if (memset_node != nullptr) {
          // if this node has memset node, this node will not be fusion
          AddFusionNodes(nodes_vec, fusion_nodes);
          nodes_vec.push_back(node_ptr);
          AddFusionNodes(nodes_vec, fusion_nodes);
        } else {
          nodes_vec.push_back(node_ptr);
        }
      } else {
        // the gentask process for another scope id will handle this node
        AddFusionNodes(nodes_vec, fusion_nodes);
      }
    } else {
      if (IsTbeOp(node_ptr->GetOpDesc()) && !IsNoTaskOp(node_ptr)) {
        AddFusionNodes(nodes_vec, fusion_nodes);
      }
    }
  }
  AddFusionNodes(nodes_vec, fusion_nodes);
  return SUCCESS;
}

Status SuperkernelTaskBuilder::GenerateTaskForFusionNodes(const std::vector<std::vector<ge::NodePtr>> &fusion_nodes,
                                                          TaskBuilderContext &task_context,
                                                          std::vector<domi::TaskDef> &task_defs) {
  if (fusion_nodes.empty()) {
    return SUCCESS;
  }

  ScopeNodeIdMap fusion_nodes_map;
  std::map<int64_t, std::vector<domi::TaskDef>> scope_task_defs_map;
  for (const std::vector<ge::NodePtr> &node_vec : fusion_nodes) {
    if (node_vec.empty()) {
      continue;
    }
    if (node_vec.size() == 1) {
      if (TaskBuilder::GenerateKernelTask(*(node_vec[0]), task_context, task_defs) != SUCCESS) {
        FE_LOGE("Failed to generate kernel task for node[%s, %s].", node_vec[0]->GetName().c_str(),
                node_vec[0]->GetType().c_str());
        return FAILED;
      }
      continue;
    }
    // remove the _no_task attr of first node
    (void)node_vec[0]->GetOpDesc()->DelAttr(ge::ATTR_NAME_NOTASK);
    (void)ge::AttrUtils::SetBool(node_vec[0]->GetOpDesc(), kAttrHasDoneSkp, true);
    std::vector<ge::Node *> nodes;
    std::vector<domi::TaskDef> nodes_task_defs;
    for (const ge::NodePtr &node : node_vec) {
      nodes.push_back(node.get());
      SetFirstOrLastNodeAttr(node->GetOpDesc(), node_vec[0]->GetOpDesc());
      domi::TaskDef task_def = {};
      // do gen task for this node, and get task args from task
      if (SetTaskArgsAttr(node, task_context, task_def) != SUCCESS) {
        return FAILED;
      }
      nodes_task_defs.push_back(task_def);
    }
    int64_t scope_id = ScopeAllocator::Instance().AllocateSkpScopeId();
    fusion_nodes_map.emplace(scope_id, nodes);
    scope_task_defs_map.emplace(scope_id, nodes_task_defs);
  }
  if (fusion_nodes_map.empty()) {
    return SUCCESS;
  }

  return DoTaskFusion(fusion_nodes_map, scope_task_defs_map, task_defs);
}

Status SuperkernelTaskBuilder::SetTaskArgsAttr(const ge::NodePtr &node, TaskBuilderContext &task_context,
                                               domi::TaskDef &task_def) {
  task_def.set_stream_id(task_context.stream_id);
  if (TaskBuilder::DoGenerateTask(*node, task_context, task_def) != SUCCESS) {
    return FAILED;
  }
  // set args attr
  vector<int64_t> task_args_vec;
  if (task_def.type() == static_cast<uint32_t>(RT_MODEL_TASK_ALL_KERNEL)) {
    domi::KernelDefWithHandle *kernel_def_with_handle = task_def.mutable_kernel_with_handle();
    FE_CHECK_NOTNULL(kernel_def_with_handle);
    uint32_t args_size = kernel_def_with_handle->args_size();
    const void *args = reinterpret_cast<const void*>(kernel_def_with_handle->args().data());
    FE_CHECK_NOTNULL(args);
    ConvertArgsToVec(args, args_size, task_args_vec);
  } else if (task_def.type() == static_cast<uint32_t>(RT_MODEL_TASK_KERNEL)) {
    domi::KernelDef *kernel_def = task_def.mutable_kernel();
    FE_CHECK_NOTNULL(kernel_def);
    uint32_t args_size = kernel_def->args_size();
    const void *args = reinterpret_cast<const void*>(kernel_def->args().data());
    FE_CHECK_NOTNULL(args);
    ConvertArgsToVec(args, args_size, task_args_vec);
  } else {
    FE_LOGE("The task type[%u] of node[%s, %s] is invalid.", task_def.type(),
            node->GetName().c_str(), node->GetType().c_str());
    return FAILED;
  }

  (void)ge::AttrUtils::SetListInt(node->GetOpDesc(), kAttrTaskArgs, task_args_vec);
  FE_LOGD("Set task args[%s] for node[%s, %s]", StringUtils::IntegerVecToString(task_args_vec).c_str(),
          node->GetName().c_str(), node->GetType().c_str());
  return SUCCESS;
}

Status SuperkernelTaskBuilder::DoTaskFusion(const ScopeNodeIdMap &fusion_nodes_map,
                                            const std::map<int64_t, std::vector<domi::TaskDef>> &scope_task_defs_map,
                                            std::vector<domi::TaskDef> &task_defs) {
  OpStoreAdapterPtr op_store_adapter =
          OpStoreAdapterManager::Instance(AI_CORE_NAME).GetOpStoreAdapter(EN_IMPL_HW_TBE);
  FE_CHECK_NOTNULL(op_store_adapter);
  CompileResultMap compile_ret_map;
  Status status = op_store_adapter->TaskFusion(fusion_nodes_map, compile_ret_map);
  if (status != SUCCESS) {
    return FAILED;
  }
  // parse json file, and set attr on first node
  for (const std::pair<const int64_t, std::vector<ge::Node *>> &nodes_pair : fusion_nodes_map) {
    ge::Node *first_node = nodes_pair.second[0];
    auto iter = compile_ret_map.find(nodes_pair.first);
    if (iter == compile_ret_map.end() || iter->second.empty()) {
      FE_LOGE("Failed to find json file after task fusion for node[%s, %s].",
              first_node->GetName().c_str(), first_node->GetType().c_str());
      return FAILED;
    }

    // clear buffer attr
    for (const ge::Node *node : nodes_pair.second) {
      if (node == nullptr) {
        continue;
      }
      (void)node->GetOpDesc()->DelAttr(ge::ATTR_NAME_TBE_KERNEL_BUFFER);
      (void)node->GetOpDesc()->DelAttr(ge::ATTR_NAME_TBE_KERNEL_NAME);
      (void)node->GetOpDesc()->DelExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL);
    }
    FE_LOGD("Task fusion json file of node[%s, %s] is [%s]",
            first_node->GetName().c_str(), first_node->GetType().c_str(), iter->second[0].json_file_path.c_str());
    TbeJsonFileParsePtr parse_ptr = nullptr;
    FE_MAKE_SHARED(parse_ptr = std::make_shared<TbeJsonFileParse>(*first_node),
            return fe::OP_COMPILER_MAKE_SHARED_FAILED);
    FE_CHECK_NOTNULL(parse_ptr);
    if (parse_ptr->PackageTvmJsonInfo(iter->second[0]) != SUCCESS) {
      FE_LOGE("PackageTvmJsonInfo failed for node[%s, %s].", first_node->GetName().c_str(),
              first_node->GetType().c_str());
      return FAILED;
    }

    auto task_iter = scope_task_defs_map.find(nodes_pair.first);
    if (task_iter == scope_task_defs_map.end()) {
      FE_LOGE("Failed to find task definition after task fusion for node[%s, %s].",
              first_node->GetNamePtr(), first_node->GetTypePtr());
      return FAILED;
    }

    domi::TaskDef task_def = task_iter->second[0];
    TaskBuilder::StartKernelFusion(first_node->GetOpDesc(), task_def.stream_id(), task_defs);
    if (FillupFusionTask(first_node->GetOpDesc(), task_def) != SUCCESS) {
      return FAILED;
    }
    task_defs.push_back(task_def);
    TaskBuilder::EndKernelFusion(first_node->GetOpDesc(), task_def.stream_id(), task_defs);
  }

  return SUCCESS;
}

Status SuperkernelTaskBuilder::FillupFusionTask(const ge::OpDescPtr &op_desc, domi::TaskDef &task_def) {
  std::string attr_val_kernel_name;
  (void)ge::AttrUtils::GetStr(op_desc, kKernelName, attr_val_kernel_name);
  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  FE_CHECK_NOTNULL(kernel_def);
  kernel_def->set_kernel_name(attr_val_kernel_name);
  FE_LOGD("Set kernel_name[%s] for the kernel_def of node[%s, %s]",
          attr_val_kernel_name.c_str(), op_desc->GetName().c_str(), op_desc->GetType().c_str());
  kernel_def->clear_args();
  kernel_def->clear_args_size();
  return SUCCESS;
}

bool IsHcclOp(const ge::NodePtr &node) {
  bool is_hccl = false;
  if (node->GetType().substr(0, 4) == "Hcom") {
    return true;
  }
  return ge::AttrUtils::GetBool(node->GetOpDesc(), "_hccl", is_hccl) && is_hccl;
}

Status SuperkernelTaskBuilder::DoSubKernelCompile(ge::ComputeGraphPtr &sub_graph) {
  std::vector<ge::NodePtr> buff_fus_compile_failed_nodes;
  CompileInfoParam compile_info(buff_fus_compile_failed_nodes);
  OpStoreAdapterPtr op_store_adapter =
        OpStoreAdapterManager::Instance(AI_CORE_NAME).GetOpStoreAdapter(EN_IMPL_HW_TBE);
  FE_CHECK_NOTNULL(op_store_adapter);
  auto sub_nodes = sub_graph->GetAllNodes();
  FE_LOGI("Start DoSubKernelCompile, sub node size[%zu]", sub_nodes.size());
  std::unordered_map<ge::OpDescPtr, std::vector<int64_t>> stored_workspaces;
  std::unordered_map<ge::OpDescPtr, std::vector<int32_t>> stored_tvm_workspaces_types;
  std::vector<int32_t> tvm_workspace_types;
  for (auto &sub_node : sub_nodes) {
    ge::OpDescPtr sub_op_desc = sub_node->GetOpDesc();
    if (IsHcclOp(sub_node)) {
      std::string kernel_meta_dir = op_store_adapter->GetCurKernelMetaDir();
      (void)ge::AttrUtils::SetStr(sub_op_desc, "kernel_meta_parent_dir", kernel_meta_dir);
    }
    int64_t imply_type = -1;
    if (!ge::AttrUtils::GetInt(sub_op_desc, FE_IMPLY_TYPE, imply_type)) {
      FE_LOGD("Node[%s, %s] is not aicore op, imply_type[%ld], will be skipped.",
              sub_node->GetNamePtr(), sub_node->GetTypePtr(), imply_type);
      continue;
    }
    FE_LOGD("Node[%s, %s] need compile as sub kernel.", sub_node->GetNamePtr(), sub_node->GetTypePtr());
    (void)sub_op_desc->DelAttr("_kernelname");
    (void)sub_op_desc->DelAttr("compile_info_json");
    (void)sub_op_desc->DelAttr("compile_info_key");
    std::vector<ge::Node *> sub_node_tmp;
    sub_node_tmp.emplace_back(sub_node.get());
    std::vector<int64_t> workspace = sub_op_desc->GetWorkspaceBytes();
    // store op workspace(after CalcOpParam)
    stored_workspaces.emplace(sub_op_desc, workspace);
    tvm_workspace_types.clear();
    ge::AttrUtils::GetListInt(sub_op_desc, ge::TVM_ATTR_NAME_WORKSPACE_TYPE, tvm_workspace_types);
    stored_tvm_workspaces_types.emplace(sub_op_desc, tvm_workspace_types);
    compile_info.fusion_nodes_map.emplace(ScopeAllocator::Instance().AllocateNegScopeId(), sub_node_tmp);
  }
  FE_LOGI("Start to DoSubKernelCompile, sub nodes size is %zu", sub_nodes.size());
  Status status = op_store_adapter->CompileOp(compile_info);
  if (status != SUCCESS) {
    return FAILED;
  }
  for (auto &fusion_item : compile_info.fusion_nodes_map) {
    int64_t scope_id = fusion_item.first;
    ge::Node &node = *((fusion_item.second)[0]);
    ge::OpDescPtr op_desc_ptr = node.GetOpDesc();
    // find the json file path according to scope id
    CompileResultMap::const_iterator json_iter = compile_info.compile_ret_map.find(scope_id);
    if (json_iter == compile_info.compile_ret_map.cend() || json_iter->second.empty()) {
      REPORT_FE_ERROR("[SPK][Compile][CompOpGetTvmJsInfo] Node[%s, %s]: json file of scopeId %ld is not found.",
                      op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str(), scope_id);
      return FAILED;
    }

    // package tvm json info
    TbeJsonFileParsePtr parse_ptr = nullptr;
    FE_MAKE_SHARED(parse_ptr = std::make_shared<TbeJsonFileParse>(node), return FAILED);
    if (parse_ptr->PackageTvmJsonInfo(json_iter->second[0]) != SUCCESS) {
      REPORT_FE_ERROR("[SPK][Compile][CompOpGetTvmJsInfo] Node[%s, %s]: failed to package tvm json info.",
                      op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str());
      return FAILED;
    }
    // Restore workspace. If we don't do this, the workspace value will be
    // inaccurate because it is not calculated by CalcOpParam.
    if (stored_workspaces.count(op_desc_ptr) > 0) {
      op_desc_ptr->SetWorkspaceBytes(stored_workspaces[op_desc_ptr]);
    }
    if(stored_tvm_workspaces_types.count(op_desc_ptr) > 0) {
        ge::AttrUtils::SetListInt(op_desc_ptr, ge::TVM_ATTR_NAME_WORKSPACE_TYPE, stored_tvm_workspaces_types[op_desc_ptr]);
    }
  }
  return SUCCESS;
}

Status SuperkernelTaskBuilder::DoSuperKernelCompile(const ge::Node &parent_node, const ScopeNodeIdMap &fusion_nodes_map) {
  OpStoreAdapterPtr op_store_adapter =
          OpStoreAdapterManager::Instance(AI_CORE_NAME).GetOpStoreAdapter(EN_IMPL_HW_TBE);
  FE_CHECK_NOTNULL(op_store_adapter);
  FE_LOGD("Start to DoSuperKernelCompile.");
  CompileResultMap compile_ret_map;
  GEEVENT("[FE] Begin to do compile sk %s.", parent_node.GetName().c_str());
  Status status = op_store_adapter->SuperKernelCompile(fusion_nodes_map, compile_ret_map);
  GEEVENT("[FE] End to do compile sk %s.", parent_node.GetName().c_str());
  if (status != SUCCESS) {
    return FAILED;
  }
  // parse json file, and set attr on first node
  for (const std::pair<const int64_t, std::vector<ge::Node *>> &nodes_pair : fusion_nodes_map) {
    auto iter = compile_ret_map.find(nodes_pair.first);
    if (iter == compile_ret_map.end() || iter->second.empty()) {
      FE_LOGE("Failed to find json file after task fusion for node[%s, %s].",
              parent_node.GetName().c_str(), parent_node.GetType().c_str());
      return FAILED;
    }

    // clear buffer attr
    for (const ge::Node *node : nodes_pair.second) {
      if (node == nullptr) {
        continue;
      }
      (void)node->GetOpDesc()->DelAttr(ge::ATTR_NAME_TBE_KERNEL_BUFFER);
      (void)node->GetOpDesc()->DelAttr(ge::ATTR_NAME_TBE_KERNEL_NAME);
      (void)node->GetOpDesc()->DelExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL);
    }
    GEEVENT("[FE] Begin to parse json file for sk %s.", parent_node.GetName().c_str());
    FE_LOGD("Super kernel json file of node[%s, %s] is [%s].",
            parent_node.GetName().c_str(), parent_node.GetType().c_str(), iter->second[0].json_file_path.c_str());
    TbeJsonFileParsePtr parse_ptr = nullptr;
    ge::Node *parent_node_ptr = const_cast<ge::Node *>(&parent_node);
    FE_MAKE_SHARED(parse_ptr = std::make_shared<TbeJsonFileParse>(*parent_node_ptr), return FAILED);
    FE_CHECK_NOTNULL(parse_ptr);
    if (parse_ptr->PackageTvmJsonInfo(iter->second[0]) != SUCCESS) {
      FE_LOGE("PackageTvmJsonInfo failed for node[%s, %s].", parent_node.GetName().c_str(),
              parent_node.GetType().c_str());
      return FAILED;
    }
    GEEVENT("[FE] End to parse json file for sk %s.", parent_node.GetName().c_str());
  }
  return SUCCESS;
}

bool GetSubGraphByNode(const ge::Node &node, ge::ComputeGraphPtr &sub_graph) {
  sub_graph = node.GetOpDesc()->TryGetExtAttr("_sk_sub_graph", sub_graph);
  return sub_graph != nullptr;
}

Status GetArgFormat(std::vector<domi::TaskDef> &tasks, std::string &args_format) {
  args_format = "";
  for (auto &task_temp : tasks) {
    domi::KernelContext *kernel_context = nullptr;
    if (task_temp.type() == RT_MODEL_TASK_KERNEL) {
      auto kernel_def = task_temp.mutable_kernel();
      FE_CHECK_NOTNULL(kernel_def);
      kernel_context = kernel_def->mutable_context();
    } else if (task_temp.type() == RT_MODEL_TASK_ALL_KERNEL) {
      auto kernel_with_handle = task_temp.mutable_kernel_with_handle();
      FE_CHECK_NOTNULL(kernel_with_handle);
      kernel_context = kernel_with_handle->mutable_context();
    } else {
      continue;
    }
    FE_CHECK_NOTNULL(kernel_context);
    args_format = kernel_context->args_format();
    FE_LOGD("GetArgFormat %s.", args_format.c_str());
  }
  return SUCCESS;
}

Status GenerateSubKernelExtTask(const ge::Node &node, ge::RunContext &context, std::vector<domi::TaskDef> &task_defs) {
  bool reg_flag = false;
  FE_TIMECOST_START(GenerateOpExtTask);
  GEEVENT("[FE] Begin to gen ext task for sub node[%s, %s].", node.GetNamePtr(), node.GetTypePtr());
  bool is_tiling_sink = CheckTilingSink(node);
  // For it's designed that all those superkernel ops that has reused binary info must go to the tiling sink step,
  // So here add a block when sk op that has reused binary not going into the tiling sink.
  if(!CheckTilingSinkForSK(node, is_tiling_sink)) {
    FE_LOGE("SuperKernel node[%s, %s] that has reused binary is not going into the tiling sink procedure, SK procedure failed.",
       node.GetNamePtr(), node.GetTypePtr());
    return FAILED;
  }
  if (GenerateOpExtTask(node, is_tiling_sink, task_defs, reg_flag) != SUCCESS) {
    REPORT_FE_ERROR("[GenTask][GenerateOpExtTask] Op[%s][%s] failed gen extra task.", 
                    node.GetNamePtr(), node.GetTypePtr());
    return FAILED;
  }
  if (reg_flag) {
    GEEVENT("[FE] End to gen ext task for sub node[%s, %s].", node.GetNamePtr(), node.GetTypePtr());
    FE_TIMECOST_END(GenerateOpExtTask, "GenerateOpExtTask");
    return SUCCESS;
  }
  auto func = OpExtGenTaskRegistry::GetInstance().FindRegisterFunc(node.GetType());
  if (func == nullptr) {
    GEEVENT("[FE] End to gen ext task for sub node[%s, %s].", node.GetNamePtr(), node.GetTypePtr());
    return SUCCESS;
  }
  FE_LOGD("Node[%s, %s] generate extra task.", node.GetNamePtr(), node.GetTypePtr());
  auto ret = func(node, context, task_defs);
  if (ret != ge::SUCCESS) {
    REPORT_FE_ERROR("[GenExtTask] Op[%s, %s] failed to gen extra task.", node.GetNamePtr(), node.GetTypePtr());
    return FAILED;
  }
  GEEVENT("[FE] End to gen ext task for sub node[%s, %s].", node.GetNamePtr(), node.GetTypePtr());
  FE_TIMECOST_END(GenerateOpExtTask, "GenerateOpExtTask.");
  return SUCCESS;
}

bool IsTilingSinkTask(std::vector<domi::TaskDef> &sub_tasks, const std::string &sub_arg_format) {
  bool has_tiling_task = false;
  for (auto &sub_task : sub_tasks) {
    if (sub_task.type() == RT_MODEL_TASK_PREPROCESS_KERNEL) {
      has_tiling_task = true;
      break;
    }
  }
  if (!has_tiling_task) {
    return false;
  }
  if (sub_arg_format.find("event_addr") != std::string::npos) {
    return true;
  }
  return false;
}

Status SuperkernelTaskBuilder::GenerateSubKernelTask(const ge::ComputeGraphPtr &sub_graph, ge::RunContext &context,
    std::vector<ge::Node *> &sub_nodes, std::vector<std::vector<domi::TaskDef>> &sub_tasks) {
  TaskBuilderContext task_context;
  if (TaskBuilder::InitTaskContext(context, task_context) != SUCCESS) {
    REPORT_FE_ERROR("[GenSubTask] Failed to init context for superkernel.");
    return FAILED;
  }
  for (auto &sub_node : sub_graph->GetAllNodes()) {
    std::vector<domi::TaskDef> tmp_tasks;
    int64_t imply_type = -1;
    if (ge::AttrUtils::GetInt(sub_node->GetOpDesc(), FE_IMPLY_TYPE, imply_type)) {
      FE_LOGD("Node[%s, %s] is aicore op, imply_type[%ld], do aicore generate task.",
              sub_node->GetNamePtr(), sub_node->GetTypePtr(), imply_type);
      GEEVENT("[FE] Begin to gentask for aicore node[%s, %s].", sub_node->GetNamePtr(), sub_node->GetTypePtr());
      if (TaskBuilder::GenerateKernelTask(*(sub_node.get()), task_context, tmp_tasks) != SUCCESS) {
        REPORT_FE_ERROR("[GenSubTask] Op[%s, %s] failed to generate task.", sub_node->GetNamePtr(), sub_node->GetTypePtr());
        return FAILED;
      }
      GEEVENT("[FE] End to gentask for aicore node[%s, %s].", sub_node->GetNamePtr(), sub_node->GetTypePtr());
    } else if (!IsHcclOp(sub_node)) {
      FE_LOGD("Node [%s, %s] is neither an aicore op nor an hccl op, skip aicore gentask.",
              sub_node->GetNamePtr(), sub_node->GetTypePtr());
      continue;
    }
    if (GenerateSubKernelExtTask(*(sub_node.get()), context, tmp_tasks) != SUCCESS) {
      REPORT_FE_ERROR("[GenSubTask] Op[%s, %s] failed to generate sub kernel ext task.", sub_node->GetNamePtr(), sub_node->GetTypePtr());
      return FAILED;
    }
    std::string sub_arg_format;
    GetArgFormat(tmp_tasks, sub_arg_format);
    FE_LOGI("Sub node[%s, %s] finish to gen sub tasks %zu, args_format %s.",
            sub_node->GetNamePtr(), sub_node->GetTypePtr(), tmp_tasks.size(), sub_arg_format.c_str());
    sub_tasks.emplace_back(tmp_tasks);
    sub_nodes.emplace_back(sub_node.get());
    if (IsTilingSinkTask(tmp_tasks, sub_arg_format)) {
      (void)ge::AttrUtils::SetStr(sub_node->GetOpDesc(), "SPK_task_type", "dynamic");
      FE_LOGI("SPK sub node[%s, %s] set task type wait.", sub_node->GetNamePtr(), sub_node->GetTypePtr());
    }
  }
  return SUCCESS;
}

Status SuperkernelTaskBuilder::GenerateSuperKernelTask(const ge::Node &node,
  ge::RunContext &context, std::vector<domi::TaskDef> &tasks) {
  std::string soc_version = PlatformUtils::Instance().GetShortSocVersion();
  GEEVENT("[FE] Begin to compile and gentask for superkernel node[%s, %s].", node.GetNamePtr(), node.GetTypePtr());
  FE_LOGI("Node[%s, %s] begin generate superkernel task for parentnode in soc version[%s].",
          node.GetNamePtr(), node.GetTypePtr(), soc_version.c_str());
  ge::ComputeGraphPtr sub_graph = nullptr;
  if (!GetSubGraphByNode(node, sub_graph)) {
    FE_LOGE("Superkernel node[%s, %s] has no invalid subgraph.",
            node.GetNamePtr(), node.GetTypePtr());
    return FAILED;
  }
  std::vector<ge::Node *> super_kernel_nodes;
  int64_t spk_id = 0;
  for (auto &sub_node : sub_graph->GetAllNodes()) {
    FE_LOGD("Start to set attr _ascendc_superkernel_scope for node");
    int64_t imply_type = -1;
    auto op_desc = sub_node->GetOpDesc();
    if (!ge::AttrUtils::GetInt(op_desc, FE_IMPLY_TYPE, imply_type) && !IsHcclOp(sub_node)) {
      FE_LOGD("[Node %s type %s] is not aicore op, imply_type[%ld], will be skipped.",
        sub_node->GetName().c_str(), sub_node->GetType().c_str(), imply_type);
      continue;
    }
    int64_t spk_scope = static_cast<int64_t>(GetAtomicId());
    (void)ge::AttrUtils::SetInt(op_desc, kAscendcSuperKernelScope, spk_scope);
    (void)ge::AttrUtils::SetInt(op_desc, kAscendcSuperKernelSubId, spk_id);
    FE_LOGW("Node[%s, %s] set attr _ascendc_superkernel_scope[%ld] and _ascendc_sp_sub_id[%ld].",
            op_desc->GetNamePtr(), op_desc->GetTypePtr(), spk_scope, spk_id);
    super_kernel_nodes.emplace_back(sub_node.get());
    ++spk_id;
  }

  for (auto &super_kernel_node : super_kernel_nodes) {
    (void)ge::AttrUtils::SetInt(super_kernel_node->GetOpDesc(), "_ascendc_sp_cnt", spk_id);
  }
  GEEVENT("[FE] Begin to compile sub node.");
  if (DoSubKernelCompile(sub_graph) != SUCCESS) {
    REPORT_FE_ERROR("[GenTask][GenerateTask] Op[%s][%s] compile subkernel failed.",
                    node.GetNamePtr(), node.GetTypePtr());
    return FAILED;
  }
  GEEVENT("[FE] Finish to compile sub node.");
  std::vector<ge::Node *> sub_nodes;
  std::vector<std::vector<domi::TaskDef>> sub_tasks;
  GEEVENT("[FE] Begin to gentask for sub nodes.");
  if (GenerateSubKernelTask(sub_graph, context, sub_nodes, sub_tasks) != SUCCESS) {
    REPORT_FE_ERROR("[GenTask] Op[%s, %s] compile subkernel failed.", node.GetNamePtr(), node.GetTypePtr());
    return FAILED;
  }
  GEEVENT("[FE] Finish to gentask for sub nodes.");
  int64_t spk_scope = -1;
  ScopeAllocator::GetSuperKernelScope(node.GetOpDesc(), spk_scope);
  ScopeNodeIdMap fusion_nodes_map;
  fusion_nodes_map.emplace(spk_scope, sub_nodes);
  GEEVENT("[FE] Begin to compile for sk node.");
  if (DoSuperKernelCompile(node, fusion_nodes_map) != SUCCESS) {
    REPORT_FE_ERROR("[GenTask][GenerateTask] Op[%s][%s] compile superkernel failed.",
                    node.GetNamePtr(), node.GetTypePtr());
    return FAILED;
  }
  GEEVENT("[FE] End to compile for sk node, begin to gentask for sk node.");
  if (fe::GenTaskForSuperKernel(node, sub_tasks, sub_nodes, tasks) != SUCCESS) {
    REPORT_FE_ERROR("[GenTask][GenerateTask] Op[%s][%s] failed to gentask.", node.GetNamePtr(), node.GetTypePtr());
    return FAILED;
  }
  GEEVENT("[FE] End to gentask for sk node.");
  FE_LOGI("Node[%s, %s] finish to generate superkernel task for parentnode.", node.GetNamePtr(), node.GetTypePtr());
  GEEVENT("[FE] Finish to compile and gentask for superkernel node[%s, %s].", node.GetNamePtr(), node.GetTypePtr());
  return SUCCESS;
}
}