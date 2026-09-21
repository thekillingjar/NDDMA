/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/build/task_generator.h"
#include <string>
#include "graph/passes/memory_conflict/atomic_addr_clean_pass.h"
#include "graph/ge_context.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/model_serialize.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/tensor_utils.h"
#include "common/checker.h"
#include "api/gelib/gelib.h"
#include "engines/manager/opskernel_manager/ops_kernel_builder_manager.h"
#include "common/preload/model/pre_model_partition_utils.h"
#include "platform/platform_info.h"
#include "graph/utils/op_type_utils.h"
#include "graph/utils/graph_utils.h"
#include "task_generator_utils.h"
#include "common/preload/model/pre_model_utils.h"
#include "framework/common/tlv/pre_model_desc.h"
#include "common/compile_profiling/ge_trace_wrapper.h"
#include "base/err_mgr.h"
#include "acl/acl_rt.h"

using domi::LogTimeStampDef;
using domi::ModelTaskDef;
using domi::TaskDef;

namespace {
const char *const kIsFirstNode = "is_first_node";
const char *const kIsLastNode = "is_last_node";
const char *const kIsInputVar = "INPUT_IS_VAR";
const char *const kIsOutputVar = "OUTPUT_IS_VAR";
const std::set<std::string> kNanoSocVersion{"Ascend035", "Ascend035A", "Ascend035B"};
const int64_t kHashFactor = 100000;
const int64_t kInvalidGroupId = -1;
const int64_t kInvalidStream = -1;
const int32_t kInvalidDeviceId = -1;
const int64_t kDefaultThreadNum = 1;
const uint64_t invalidAddrType = 0xFFFFFFFFFFFFFFFFULL;
const std::set<std::string> kHcclDvppKernelInfoNames = {"ops_kernel_info_hccl", "ops_kernel_info_hccl_gradtune",
                                                        "hvd_ops_kernel_info", "dvpp_ops_kernel_info_store"};
const std::set<std::string> kAicpuKernelLibs = {"aicpu_ascend_kernel", "aicpu_tf_kernel"};
}  // namespace
namespace ge {
namespace {
auto ffts_filter = [](const Node &node, const char *, const ComputeGraphPtr &) {
  return !(node.GetOpDesc()->HasAttr(ATTR_NAME_FFTS_SUB_GRAPH) ||
           node.GetOpDesc()->HasAttr(ATTR_NAME_FFTS_PLUS_SUB_GRAPH));
};

enum class GenTaskCallKey {
  kAtomicEngine = 0,
  kFftsEngine,
};

const unordered_map<GenTaskCallKey, std::string> key_2_string = {{GenTaskCallKey::kAtomicEngine, "Normal engine"},
                                                                 {GenTaskCallKey::kFftsEngine, "FFTS engine"}};

static const std::set<std::string> kDataOpType{DATA, AIPPDATA, ANN_DATA};
GenTaskCallKey GetKey(const Node *const node) {
  const auto &op_desc = node->GetOpDesc();
  if (op_desc->HasAttr(ATTR_NAME_FFTS_PLUS_SUB_GRAPH)) {
    return GenTaskCallKey::kFftsEngine;
  } else {
    return GenTaskCallKey::kAtomicEngine;
  }
}
using GenTaskCall = std::function<Status(
    TaskGenerator *, Node *, const std::string &, std::vector<domi::TaskDef> &task_def_list_per_node,
    const GEThreadLocalContext &, const error_message::ErrorManagerContext &, int32_t)>;
uint32_t GetThreadNum() {
  const char_t *value = nullptr;
  MM_SYS_GET_ENV(MM_ENV_MAX_COMPILE_CORE_NUMBER, value);
  int64_t thread_num = ((value != nullptr) && (value[0U] != '\0')) ?
    std::strtol(value, nullptr, 10) : kDefaultThreadNum;
  if (thread_num <= 0) {
    GELOGW("Get invalid MAX_COMPILE_CORE_NUMBER env value %s, use default thread number %ld", value, kDefaultThreadNum);
    thread_num = kDefaultThreadNum;
  }
  GELOGI("Thread num is %ld", thread_num);
  return thread_num;
}

// 1. RefreshStreamId的存在是因为历史原因，框架兜底给没有设置taskdef中stream_id字段
//    的引擎补齐stream_id的设置, 因兼容性问题此函数有存在的必要
// 2. 在后续流拆分的流程里如果发生拆流也会刷新stream id
void RefreshStreamId(const OpDescPtr &op_desc, const size_t start_index, std::vector<domi::TaskDef> &task_defs) {
  // 拥有附属stream_id的算子当前会生成多个task_defs, stream_id和附属stream_id作为op_desc上不同
  // 的字段会设置到其生成的不同的task_def的stream_id字段中，因此跳过RefreshStreamId的操作
  if (op_desc->HasValidAttachedStreamId() || op_desc->GetType() == "SuperKernel") {
    GELOGI("Node {%s %s} keep origin stream info, task size %zu.", op_desc->GetNamePtr(), op_desc->GetTypePtr(),
           task_defs.size());
    return;
  }
  for (size_t index = start_index; index < task_defs.size(); ++index) {
    // op_desc must not be null
    // index must be valid
    auto stream_id = op_desc->GetStreamId() == kInvalidStream ? 0 : op_desc->GetStreamId();
    task_defs[index].set_stream_id(static_cast<uint32_t>(stream_id));
  }
}

bool NeedDoFusionTask() {
  std::string buffer_optimize = "off_optimize";
  (void)ge::GetContext().GetOption(BUFFER_OPTIMIZE, buffer_optimize);
  GELOGI("Get buffer option %s with value %s", BUFFER_OPTIMIZE.c_str(), buffer_optimize.c_str());
  return ((buffer_optimize == "l1_optimize") || (buffer_optimize == "l2_optimize"));
}
}  // namespace

TaskGenerator::TaskGenerator(uint8_t *var_mem_base, uint64_t var_mem_size, RunContext *run_context) {
  var_mem_base_ = var_mem_base;
  var_mem_size_ = var_mem_size;
  run_context_ = run_context;
}

TaskGenerator::~TaskGenerator() {}

Status TaskGenerator::AddModelTaskToModel(const ModelTaskDef &model_task_def, uint64_t session_id, ge::Model &model,
                                          RunContext &run_context) const {
  GE_CHK_BOOL_EXEC(
      AttrUtils::SetInt(model, MODEL_ATTR_TASK_GEN_BASE_ADDR, reinterpret_cast<uintptr_t>(run_context.dataMemBase)),
      REPORT_INNER_ERR_MSG("E19999", "Set Attr:%s fail for model:%s", MODEL_ATTR_TASK_GEN_BASE_ADDR.c_str(),
                         model.GetName().c_str());
      GELOGE(FAILED, "[Set][Attr] %s fail for model:%s", MODEL_ATTR_TASK_GEN_BASE_ADDR.c_str(),
             model.GetName().c_str());
      return FAILED);
  GE_CHK_BOOL_EXEC(
      AttrUtils::SetInt(model, MODEL_ATTR_TASK_GEN_WEIGHT_ADDR, reinterpret_cast<uintptr_t>(run_context.weightMemBase)),
      REPORT_INNER_ERR_MSG("E19999", "Set Attr:%s fail for model:%s", MODEL_ATTR_TASK_GEN_WEIGHT_ADDR.c_str(),
                         model.GetName().c_str());
      GELOGE(FAILED, "[Set][Attr] %s fail for model:%s", MODEL_ATTR_TASK_GEN_WEIGHT_ADDR.c_str(),
             model.GetName().c_str());
      return FAILED);
  GE_CHK_BOOL_EXEC(
      AttrUtils::SetInt(model, ATTR_MODEL_TASK_GEN_VAR_ADDR, reinterpret_cast<uintptr_t>(var_mem_base_)),
      REPORT_INNER_ERR_MSG("E19999", "Set Attr:%s fail for model:%s", ATTR_MODEL_TASK_GEN_VAR_ADDR.c_str(),
                         model.GetName().c_str());
      GELOGE(FAILED, "[Set][Attr] %s fail for model:%s", ATTR_MODEL_TASK_GEN_VAR_ADDR.c_str(), model.GetName().c_str());
      return FAILED);
  GE_CHK_BOOL_EXEC(
      AttrUtils::SetInt(model, ATTR_MODEL_VAR_SIZE, var_mem_size_),
      REPORT_INNER_ERR_MSG("E19999", "Set Attr:%s fail for model:%s", ATTR_MODEL_VAR_SIZE.c_str(),
                         model.GetName().c_str());
      GELOGE(FAILED, "[Set][Attr] %s fail for model:%s", ATTR_MODEL_VAR_SIZE.c_str(), model.GetName().c_str());
      return FAILED);
  GE_CHK_BOOL_EXEC(
      AttrUtils::SetInt(model, MODEL_ATTR_SESSION_ID, session_id),
      REPORT_INNER_ERR_MSG("E19999", "Set Attr:%s fail for mode:%s", MODEL_ATTR_SESSION_ID.c_str(),
                         model.GetName().c_str());
      GELOGE(FAILED, "[Set][Attr] %s fail for mode:%s", MODEL_ATTR_SESSION_ID.c_str(), model.GetName().c_str());
      return FAILED);
  std::vector<std::string> task_index_op_name{std::make_move_iterator(op_names_.begin()),
                                              std::make_move_iterator(op_names_.end())};
  GE_ASSERT_TRUE(AttrUtils::SetListStr(model, ATTR_MODEL_TASK_INDEX_OP_NAME, task_index_op_name));
  size_t task_size = model_task_def.ByteSizeLong();
  ge::Buffer serial_buff(task_size);
  google::protobuf::io::ArrayOutputStream array_stream(serial_buff.GetData(),
                                                       static_cast<int32_t>(serial_buff.GetSize()));
  google::protobuf::io::CodedOutputStream output_stream(&array_stream);
  output_stream.SetSerializationDeterministic(true);
  GE_ASSERT_TRUE(model_task_def.SerializeToCodedStream(&output_stream));
  GE_ASSERT_TRUE(AttrUtils::SetZeroCopyBytes(model, MODEL_ATTR_TASKS, std::move(serial_buff)));
  return SUCCESS;
}

Status TaskGenerator::UpdateOpIsVarAttr(const OpDescPtr &op_desc, uint64_t session_id) const {
  GELOGD("Update is var attr, node[name:%s(%s), id:%ld, stream_id:%ld].", op_desc->GetName().c_str(),
         op_desc->GetType().c_str(), op_desc->GetId(), op_desc->GetStreamId());
  std::vector<int64_t> input_offsets = op_desc->GetInputOffset();
  if (!(input_offsets.empty())) {
    std::vector<bool> input_var;
    size_t valid_input_index = 0;
    for (uint32_t i = 0; i < op_desc->GetAllInputsSize(); i++) {
      std::vector<int64_t> output_list;
      auto input_tensor_desc = op_desc->MutableInputDesc(i);
      if (input_tensor_desc == nullptr) {
        continue;
      }
      if (valid_input_index >= input_offsets.size()) {
        break;
      }
      int64_t inner_offset = 0;
      (void)ge::AttrUtils::GetInt(input_tensor_desc, ATTR_NAME_INNER_OFFSET, inner_offset);
      GELOGD("Node[%s] input[%u] has inner_offset[%ld]", op_desc->GetName().c_str(), i, inner_offset);
      input_var.push_back(VarManager::Instance(session_id)->IsVarAddr(input_offsets[valid_input_index] - inner_offset));
      valid_input_index++;
    }
    GE_CHK_BOOL_EXEC(AttrUtils::SetListBool(op_desc, kIsInputVar, input_var),
                     REPORT_INNER_ERR_MSG("E19999", "Set Attr:%s fail for op:%s(%s)", kIsInputVar,
                                        op_desc->GetName().c_str(), op_desc->GetType().c_str());
                     GELOGE(FAILED, "[Set][Attr] %s fail for op:%s(%s)", kIsInputVar, op_desc->GetName().c_str(),
                            op_desc->GetType().c_str());
                     return FAILED);
  }
  std::vector<int64_t> output_offsets = op_desc->GetOutputOffset();
  if (!(output_offsets.empty())) {
    std::vector<bool> output_var;
    size_t valid_output_index = 0;
    for (uint32_t i = 0; i < op_desc->GetAllOutputsDescSize(); i++) {
      std::vector<int64_t> output_list;
      auto output_tensor_desc = op_desc->MutableOutputDesc(i);
      if (output_tensor_desc == nullptr) {
        continue;
      }
      if (valid_output_index >= output_offsets.size()) {
        break;
      }
      int64_t inner_offset = 0;
      (void)ge::AttrUtils::GetInt(output_tensor_desc, ATTR_NAME_INNER_OFFSET, inner_offset);
      GELOGD("Node[%s] output[%u] has inner_offset[%ld]", op_desc->GetName().c_str(), i, inner_offset);
      output_var.push_back(
          VarManager::Instance(session_id)->IsVarAddr(output_offsets[valid_output_index] - inner_offset));
      valid_output_index++;
    }
    GE_CHK_BOOL_EXEC(AttrUtils::SetListBool(op_desc, kIsOutputVar, output_var),
                     REPORT_INNER_ERR_MSG("E19999", "Set Attr:%s fail for op:%s(%s)", kIsOutputVar,
                                        op_desc->GetName().c_str(), op_desc->GetType().c_str());
                     GELOGE(FAILED, "[Set][Attr] %s fail for op:%s(%s)", kIsOutputVar, op_desc->GetName().c_str(),
                            op_desc->GetType().c_str());
                     return FAILED);
  }
  return SUCCESS;
}

Status TaskGenerator::GenTaskForPartiallySupportedNode(const NodePtr &node, RunContext &context,
                                                       std::vector<domi::TaskDef> &tasks) const {
  bool partially_supported = false;
  const auto &op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  (void)AttrUtils::GetBool(op_desc, kPartiallySupported, partially_supported);
  if (!partially_supported) {
    return SUCCESS;
  }
  const auto &instance_ptr = ge::GELib::GetInstance();
  if ((instance_ptr == nullptr) || (!instance_ptr->InitFlag())) {
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "[Get][GELib] GELib is not init before.");
    REPORT_INNER_ERR_MSG("E19999", "[Get][GELib] GELib is not init before, check invalid.");
    return GE_CLI_GE_NOT_INITIALIZED;
  }

  std::string op_kernel_lib_name;
  for (const auto &aicpu_kernel_name : kAicpuKernelLibs) {
    auto kernel_info = instance_ptr->OpsKernelManagerObj().GetOpsKernelInfoStore(aicpu_kernel_name);
    GE_ASSERT_NOTNULL(kernel_info);
    std::vector<NodePtr> nodes;
    nodes.emplace_back(node);
    std::string unsupported_reason;
    if (kernel_info->CheckSupported(op_desc, unsupported_reason)) {
      GE_CHK_STATUS_RET(kernel_info->CompileOp(nodes),
                        "Failed to invoke compile op, libName = %s, type = %s, node = %s.", aicpu_kernel_name.c_str(),
                        op_desc->GetType().c_str(), op_desc->GetName().c_str());
      op_kernel_lib_name = aicpu_kernel_name;
      break;
    } else {
      GELOGI("Aicpu engine not support, kernel_name is %s, op type is %s, op name is %s", aicpu_kernel_name.c_str(),
             op_desc->GetType().c_str(), op_desc->GetName().c_str());
    }
  }
  if (op_kernel_lib_name.empty()) {
    GELOGW("Partially supported task doesn't find aicpu ops kernel info.");
    return SUCCESS;
  }

  const auto &ops_kernel_builder = OpsKernelBuilderManager::Instance().GetOpsKernelBuilder(op_kernel_lib_name);
  GE_ASSERT_NOTNULL(ops_kernel_builder);
  GELOGD("To invoke GenerateTask, node = %s, lib name = %s", node->GetName().c_str(), op_kernel_lib_name.c_str());
  GE_CHK_STATUS_RET(ops_kernel_builder->GenerateTask(*node, context, tasks),
                    "Failed to invoke GenerateTask, libName = %s, node = %s", op_kernel_lib_name.c_str(),
                    node->GetName().c_str());

  (void)AttrUtils::SetStr(node->GetOpDesc(), kAICpuKernelLibName, op_kernel_lib_name);
  GELOGD("Done invoking GenerateTask successfully, kernel lib is %s.", op_kernel_lib_name.c_str());
  return SUCCESS;
}

Status TaskGenerator::GenTaskForNodeByAliasEngine(const NodePtr &node, RunContext &context,
                                                  std::vector<domi::TaskDef> &tasks) const {
  const auto &op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  std::string op_kernel_lib_name;
  if (!AttrUtils::GetStr(op_desc, ATTR_NAME_ALIAS_ENGINE_NAME, op_kernel_lib_name)) {
    return SUCCESS;
  }
  if (op_kernel_lib_name.empty()) {
    GELOGW("node %s find empty kernel lib name.", op_desc->GetName().c_str());
    return SUCCESS;
  }

  const auto &ops_kernel_builder = OpsKernelBuilderManager::Instance().GetOpsKernelBuilder(op_kernel_lib_name);
  GE_ASSERT_NOTNULL(ops_kernel_builder);
  GE_CHK_STATUS_RET(ops_kernel_builder->GenerateTask(*node, context, tasks),
                    "Failed to invoke GenerateTask, libName = %s, node = %s", op_kernel_lib_name.c_str(),
                    node->GetName().c_str());
  GELOGD("Done invoking GenerateTask successfully, node = %s, lib name = %s", node->GetName().c_str(),
         op_kernel_lib_name.c_str());
  return SUCCESS;
}

Status TaskGenerator::UpdateAnchorStatus(const NodePtr &node) const {
  GELOGD("Start UpdateAnchorStatus for %s.", node->GetName().c_str());
  GE_ASSERT_GRAPH_SUCCESS(NodeUtils::SetAllAnchorStatus(node), "SetAllAnchorStatus fail for op:%s(%s)",
                          node->GetName().c_str(), node->GetType().c_str());
  for (auto &anchor : node->GetAllInDataAnchors()) {
    auto peer_anchor = anchor->GetPeerOutAnchor();
    if (peer_anchor == nullptr) {
      GE_ASSERT_GRAPH_SUCCESS(AnchorUtils::SetStatus(anchor, ANCHOR_SUSPEND),
                              "Set in peer anchor status fail for op:%s(%s), anchor_index:%d", node->GetName().c_str(),
                              node->GetType().c_str(), anchor->GetIdx());
      continue;
    }

    std::string const_type;
    bool is_const = NodeUtils::GetConstOpType(peer_anchor->GetOwnerNode(), const_type);
    // if const is in unknown graph, it will not assign memory and status should not be const
    // GetParentInput will not return nullptr according to GetConstOpType result
    if (is_const && (const_type == CONSTANT) &&
        ((!ge::OpTypeUtils::IsDataNode(peer_anchor->GetOwnerNodeBarePtr()->GetType())) ||
         (!NodeUtils::GetParentInput(peer_anchor->GetOwnerNode())->GetOwnerComputeGraph()->GetGraphUnknownFlag()))) {
      GE_ASSERT_GRAPH_SUCCESS(AnchorUtils::SetStatus(anchor, ANCHOR_CONST),
                              "Set in anchor CONST status fail for op:%s(%s), anchor_index:%d", node->GetName().c_str(),
                              node->GetType().c_str(), anchor->GetIdx());
    } else {
      GE_ASSERT_GRAPH_SUCCESS(AnchorUtils::SetStatus(anchor, ANCHOR_DATA),
                              "Set in anchor DATA status fail for op:%s(%s), anchor_index:%d", node->GetName().c_str(),
                              node->GetType().c_str(), anchor->GetIdx());
    }
  }
  return SUCCESS;
}

Status TaskGenerator::MarkNodeAndSetIndex(const ComputeGraphPtr &graph) const {
  auto ge_lib = GELib::GetInstance();
  if ((ge_lib == nullptr) || !ge_lib->InitFlag()) {
    REPORT_INNER_ERR_MSG("E19999", "Check GELib instance not init before");
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "[Check][Param] GE is not initialized or is finalized.");
    return GE_CLI_GE_NOT_INITIALIZED;
  }
  for (const auto &node : graph->GetAllNodes()) {
    GE_CHK_STATUS_RET(UpdateAnchorStatus(node), "[Call][UpdateAnchorStatus] node:%s(%s) failed.",
                      node->GetName().c_str(), node->GetType().c_str());
    const auto &op_desc = node->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc);
    GE_CHK_STATUS_RET(UpdateOpIsVarAttr(op_desc, graph->GetSessionID()));
  }
  std::unordered_set<Node *> need_to_gen_task_nodes;
  GE_ASSERT_SUCCESS(MarkFirstAndLastOpsForGraph(graph, need_to_gen_task_nodes));
  return SUCCESS;
}

Status TaskGenerator::MarkFirstAndLastOpsForGraph(const ComputeGraphPtr &graph, std::unordered_set<Node *> &target_nodes) const {
  // get nodes by graph unknown status for stream mark
  auto ge_lib = GELib::GetInstance();
  GE_ASSERT_NOTNULL(ge_lib);
  const auto all_nodes = graph->GetNodes(graph->GetGraphUnknownFlag());
  GE_ASSERT(!all_nodes.empty(), "Check param all_nodes empty in graph:%s", graph->GetName().c_str());
  std::map<int64_t, std::vector<Node *>> all_stream_nodes;
  for (auto &node : all_nodes) {
    const auto &op_desc = node->GetOpDesc();
    // Reset op kernel lib name
    if (op_desc->GetOpKernelLibName().empty()) {
      (void)ge_lib->DNNEngineManagerObj().GetDNNEngineName(node);
    }

    (void)op_desc->DelAttr(kIsFirstNode);
    (void)op_desc->DelAttr(kIsLastNode);

    if (op_desc->GetStreamId() != kInvalidStream) {
      all_stream_nodes[op_desc->GetStreamId()].emplace_back(node.get());
    }
  }

  bool is_single_stream = all_stream_nodes.size() == 1;
  for (const auto &stream_nodes : all_stream_nodes) {
    Status status = MarkFirstAndLastOps(stream_nodes.second, is_single_stream, target_nodes);
    if (status != SUCCESS) {
      GELOGE(status, "[Mark][FirstAndLastOps] failed, graph:%s.", graph->GetName().c_str());
      return status;
    }
  }
  return SUCCESS;
}

Status TaskGenerator::MarkFirstAndLastOps(const std::vector<Node *> &nodes, bool is_single_stream,
                                          std::unordered_set<Node *> &target_nodes) const {
  std::vector<std::vector<Node *>> continuous_node_lists(1);
  static const std::set<std::string> separator_types({LABELSET, LABELGOTOEX, LABELSWITCHBYINDEX, STREAMSWITCH});
  for (auto &node : nodes) {
    auto op_desc = node->GetOpDescBarePtr();
    GE_ASSERT_NOTNULL(op_desc);
    bool attr_notask = false;
    if (ge::AttrUtils::GetBool(op_desc, ATTR_NAME_NOTASK, attr_notask) && attr_notask) {
      continue;
    }
    std::string op_type = op_desc->GetType();
    if ((!is_single_stream && !op_desc->GetSubgraphInstanceNames().empty()) || separator_types.count(op_type) != 0) {
      continuous_node_lists.emplace_back(std::vector<Node *>());
    } else {
      continuous_node_lists.back().emplace_back(node);
    }
  }
  GELOGD("Number of continuous node lists is %zu.", continuous_node_lists.size());

  for (const auto &continuous_nodes : continuous_node_lists) {
    std::map<std::string, std::pair<Node *, Node *>> first_and_last_nodes;
    for (auto &node : continuous_nodes) {
      auto op_desc = node->GetOpDescBarePtr();
      std::string op_kernel_lib_name = op_desc->GetOpKernelLibName();
      if (op_kernel_lib_name.empty()) {
        REPORT_INNER_ERR_MSG("E19999", "Get ops kernel info store failed for op:%s(%s), op_kernel_name:%s",
                           op_desc->GetName().c_str(), op_desc->GetType().c_str(), op_kernel_lib_name.c_str());
        GELOGE(INTERNAL_ERROR, "[Check][Param] node:%s(%s) get op kernel lib failed.", op_desc->GetName().c_str(),
               op_desc->GetType().c_str());
        return INTERNAL_ERROR;
      }

      auto it = first_and_last_nodes.find(op_kernel_lib_name);
      if (it == first_and_last_nodes.end()) {
        first_and_last_nodes.emplace(op_kernel_lib_name, std::make_pair(node, node));
      } else {
        it->second.second = node;
      }
    }

    for (auto &it : first_and_last_nodes) {
      auto &op_pair = it.second;
      GE_ASSERT_TRUE(ge::AttrUtils::SetBool(op_pair.first->GetOpDescBarePtr(), kIsFirstNode, true),
                     "[Set][Attr] %s fail for op:%s(%s)", kIsFirstNode, op_pair.first->GetName().c_str(),
                     op_pair.first->GetType().c_str());
      GE_ASSERT_TRUE(ge::AttrUtils::SetBool(op_pair.second->GetOpDescBarePtr(), kIsLastNode, true),
                     "[Set][Attr] %s fail for op:%s(%s)", kIsLastNode, op_pair.second->GetName().c_str(),
                     op_pair.second->GetType().c_str());
      target_nodes.insert(op_pair.first);
      target_nodes.insert(op_pair.second);
    }
  }
  return SUCCESS;
}

std::unordered_map<int64_t , std::vector<domi::TaskDef>> &TaskGenerator::MutableNodeId2TaskDefs() {
  return node_id_2_node_tasks_;
}

Status TaskGenerator::SetKernelInfo() {
  const auto &ge_lib = GELib::GetInstance();
  GE_ASSERT_NOTNULL(ge_lib);
  GE_ASSERT_TRUE(ge_lib->InitFlag());
  for (const auto node : nodes_) {
    GE_ASSERT_NOTNULL(node);
    const auto &op_desc = node->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc);
    const std::string &op_kernel_lib_name = op_desc->GetOpKernelLibName();
    // Set opsKernelInfoStorePtr for hccl which will be use in DistributeTask and InitTaskInfo
    if (kHcclDvppKernelInfoNames.count(op_kernel_lib_name) > 0) {
      auto kernel_info_store = ge_lib->OpsKernelManagerObj().GetOpsKernelInfoStore(op_kernel_lib_name);
      GE_ASSERT_NOTNULL(kernel_info_store);
      (void)op_desc->SetExtAttr("OpsKernelInfoStorePtr", kernel_info_store.get());
    }
  }
  return SUCCESS;
}

Status TaskGenerator::FilterCandidatesNodes(const ComputeGraphPtr &graph) {
  for (const auto &node : graph->GetNodes(graph->GetGraphUnknownFlag(), nullptr, ffts_filter)) {
    const auto &op_desc = node->GetOpDesc();
    if (NoNeedGenTask(op_desc)) {
      continue;
    }
    nodes_.emplace_back(node.get());
  }
  return SUCCESS;
}

Status TaskGenerator::PrepareForGenerateTask(const ComputeGraphPtr &graph) {
  GE_CHK_STATUS_RET(MarkNodeAndSetIndex(graph), "[Call][MarkNodeAndSetIndex] failed, graph:%s.",
                    graph->GetName().c_str());
  GE_CHK_STATUS_RET(FilterCandidatesNodes(graph), "[Call][FilterCandidatesNodes] failed, graph:%s.",
                    graph->GetName().c_str());
  GE_CHK_STATUS_RET(SetKernelInfo(), "[Set][KernelInfo] failed, graph:%s.", graph->GetName().c_str());
  GE_CHK_STATUS_RET(FindProfilingNodeIndex(graph, profiling_point_), "[Call][FindProfilingNodeIndex] failed, graph:%s.",
                    graph->GetName().c_str());
  return SUCCESS;
}

Status TaskGenerator::GenerateTaskForNormalNode(Node *const node, const std::string &tag,
                                                std::vector<domi::TaskDef> &task_def_list_per_node,
                                                const GEThreadLocalContext &ge_context,
                                                const error_message::ErrorManagerContext &error_context,
                                                int32_t device_id) {
  GetThreadLocalContext() = ge_context;
  error_message::SetErrMgrContext(error_context);
  if (device_id != kInvalidDeviceId) {
    GE_CHK_RT_RET(aclrtSetDevice(device_id));
  }
  GE_MAKE_GUARD(reset_device, [device_id]() {
    if (device_id != kInvalidDeviceId) {
      GE_CHK_RT(aclrtResetDevice(device_id));
    }
  });
  return GenTaskForNormalNode(node, tag, task_def_list_per_node);
}

Status TaskGenerator::GenTaskForNormalNode(Node *const node, const std::string &tag,
                                           std::vector<domi::TaskDef> &task_def_list_per_node) {
  const auto &op_desc = node->GetOpDesc();  // node and op desc must not be null
  const size_t before_size = task_def_list_per_node.size();
  GE_CHK_STATUS_RET(ProfilingTaskUtils::InsertProfilingTaskBefore(op_desc, profiling_point_, task_def_list_per_node),
                    "[Insert][profiling] task failed");
  GE_ASSERT_NOTNULL(run_context_);
  RunContext run_context = *run_context_;
  GELOGI("%s node %s %s, lib name %s, start to gen task, origin task size is %zu.", tag.c_str(),
         op_desc->GetName().c_str(), op_desc->GetType().c_str(), op_desc->GetOpKernelLibName().c_str(), before_size);
  // call generate node task by engine name
  GE_CHK_STATUS_RET(OpsKernelBuilderManager::Instance().GenerateTask(*node, run_context, task_def_list_per_node),
                    "[Generate][Task] fail for op:%s(%s)", node->GetName().c_str(), node->GetType().c_str());
  // try to call generate task again by others engine
  GE_CHK_STATUS_RET(GenTaskForPartiallySupportedNode(node->shared_from_this(), run_context, task_def_list_per_node),
                    "[Generate][PartiallySupportedTask] fail for op:%s(%s)", node->GetName().c_str(),
                    node->GetType().c_str());
  // try to call generate task again by other alias engine
  GE_CHK_STATUS_RET(GenTaskForNodeByAliasEngine(node->shared_from_this(), run_context, task_def_list_per_node),
                    "[Generate][AliasTask] fail for op:%s(%s)", node->GetName().c_str(), node->GetType().c_str());
  GE_CHK_STATUS_RET(ProfilingTaskUtils::InsertProfilingTaskAfter(op_desc, profiling_point_, task_def_list_per_node),
                    "[Insert][profiling] task failed");
  GE_ASSERT_TRUE((task_def_list_per_node.size() >= before_size));
  RefreshStreamId(op_desc, before_size, task_def_list_per_node);
  GELOGI("%s node %s %s, lib name %s, gen task finished, generate %zu task(s).", tag.c_str(),
         op_desc->GetName().c_str(), op_desc->GetType().c_str(), op_desc->GetOpKernelLibName().c_str(),
         task_def_list_per_node.size() - before_size);
  return SUCCESS;
}

Status TaskGenerator::GenerateTaskForFftsNode(Node *ffts_node, const std::string &tag,
                                              std::vector<domi::TaskDef> &task_def_list_per_node,
                                              const GEThreadLocalContext &ge_context,
                                              const error_message::ErrorManagerContext &error_context,
                                              int32_t device_id) {
  GetThreadLocalContext() = ge_context;
  error_message::SetErrMgrContext(error_context);
  if (device_id != kInvalidDeviceId) {
    GE_CHK_RT_RET(aclrtSetDevice(device_id));
  }
  GE_MAKE_GUARD(reset_device, [device_id]() {
    if (device_id != kInvalidDeviceId) {
      GE_CHK_RT(aclrtResetDevice(device_id));
    }
  });
  const auto &op_desc = ffts_node->GetOpDesc();  // node and op desc must not be null
  if (!op_desc->HasAttr(ATTR_NAME_FFTS_PLUS_SUB_GRAPH)) {
    return SUCCESS;
  }
  GE_CHK_STATUS_RET(ProfilingTaskUtils::InsertProfilingTaskBefore(op_desc, profiling_point_, task_def_list_per_node),
                    "[Insert][profiling] task failed");
  {
    const std::lock_guard<std::mutex> lock(ffts_mutex_);
    if (ffts_inner_thread_pool_ == nullptr) {
      ffts_inner_thread_pool_ = MakeUnique<ThreadPool>("ge_fftsgtsk", GetThreadNum(), false);
    }
  }
  GE_ASSERT_NOTNULL(ffts_inner_thread_pool_);
  std::vector<ComputeGraphPtr> subgraphs;
  GE_CHK_STATUS_RET(NodeUtils::GetDirectSubgraphs(ffts_node->shared_from_this(), subgraphs),
                    "[Check][Param] Get subgraphs of node %s failed", op_desc->GetName().c_str());
  std::map<int64_t, std::vector<domi::TaskDef>> node_id_2_node_tasks;
  std::vector<std::future<Status>> vector_future;
  GenTaskCall func = &TaskGenerator::GenerateTaskForNormalNode;
  for (const auto &subgraph : subgraphs) {
    for (const auto &node : subgraph->GetAllNodes()) {
      auto tmp_op_dec = node->GetOpDesc();
      GE_ASSERT_NOTNULL(tmp_op_dec);
      if (NoNeedGenTask(tmp_op_dec)) {
        continue;
      }
      std::vector<NodePtr> atomic_node_vec;
      atomic_node_vec.emplace_back(node);
      const auto &parent_graph = subgraph->GetParentGraph();
      if ((parent_graph != nullptr) && (parent_graph->GetGraphUnknownFlag()) &&
          (!subgraph->GetGraphUnknownFlag()) && (NodeUtils::IsLikeAtomicClean(node))) {
        GE_CHK_STATUS_RET(AtomicAddrCleanPass::CallCompileOp(atomic_node_vec),
                          "compile single op failed, parent_graph:%s, subgraph:%s, node:%s, tmp_type:%d.",
                          parent_graph->GetName().c_str(), subgraph->GetName().c_str(),
                          node->GetName().c_str(), node->GetType().c_str());
      }
      auto &task_defs = node_id_2_node_tasks[node->GetOpDesc()->GetId()];
      std::future<Status> f =
          ffts_inner_thread_pool_->commit(func, this, node.get(), "FFTS INNER", std::ref(task_defs),
                                          ge_context, error_context, device_id);
      if (f.valid()) {
        vector_future.emplace_back(std::move(f));
      }
    }
  }
  for (auto &i : vector_future) {
    GE_CHK_STATUS_RET(i.get(), "[GenTask] gen inner ffts task ctx failed.");
  }
  for (const auto &iter : node_id_2_node_tasks) {
    task_def_list_per_node.insert(task_def_list_per_node.end(), iter.second.begin(), iter.second.end());
  }
  // ffts inner task call generate firstly by multi thread, then we can call ffts task
  GE_CHK_STATUS_RET(
      OpsKernelBuilderManager::Instance().GenerateTask(*ffts_node, *run_context_, task_def_list_per_node, false),
      "[Generate][Task] fail for ffts op:%s(%s)", ffts_node->GetName().c_str(), ffts_node->GetType().c_str());
  GE_CHK_STATUS_RET(ProfilingTaskUtils::InsertProfilingTaskAfter(op_desc, profiling_point_, task_def_list_per_node),
                    "[Insert][profiling] task failed");
  RefreshStreamId(op_desc, 0U, task_def_list_per_node);
  GELOGI("%s node %s %s gen task finished, generate %zu task(s).", tag.c_str(), op_desc->GetName().c_str(),
         op_desc->GetType().c_str(), task_def_list_per_node.size());
  return SUCCESS;
}

Status TaskGenerator::SaveFusionNodes(std::map<int64_t, std::vector<NodePtr>> &fusion_nodes,
                                      const std::vector<Node *> nodes) const {
  std::map<NodePtr, int64_t> nodes_with_group_attr;
  for (auto node_bare_ptr : nodes) {
    GE_ASSERT_NOTNULL(node_bare_ptr);
    const auto &node = node_bare_ptr->shared_from_this();
    GE_ASSERT_NOTNULL(node);
    const auto &op_desc = node->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc);
    int64_t group_id = kInvalidGroupId;
    std::string name = node->GetName();
    std::string type = node->GetType();
    // For fusion ddb pass, task def must be continuous.
    // Part1: store
    // If op_desc have this tag, store it in the map firstly,
    // call the elements in the map GenerateTask at last
    // l1 and l2 is for now
    if (ge::AttrUtils::GetInt(op_desc, ATTR_NAME_L1_FUSION_GROUP_ID, group_id) ||
        ge::AttrUtils::GetInt(op_desc, ATTR_NAME_L2_FUSION_GROUP_ID, group_id)) {
      auto stream_id = op_desc->GetStreamId();
      auto group_key = group_id + stream_id * kHashFactor;
      (void)ge::AttrUtils::SetInt(op_desc, ATTR_NAME_FUSION_GROUP_KEY, group_key);
      GELOGD("Fusion: store node[name:%s(%s), group id:%ld, group key:%ld, stream_id:%ld] task.", name.c_str(),
             type.c_str(), group_id, group_key, op_desc->GetStreamId());
      fusion_nodes[group_key].push_back(node);
      nodes_with_group_attr.insert({node, group_id});
    }

    // if node's all in nodes both with same group attr
    // and it have no attr or group attr different
    // which means bad case, return error
    bool call_check = true;
    std::set<int64_t> input_group_ids;
    for (const auto &input_node : node->GetInNodes()) {
      std::map<NodePtr, int64_t>::const_iterator iter = nodes_with_group_attr.find(input_node);
      if (iter == nodes_with_group_attr.cend()) {
        call_check = false;
        break;
      } else {
        input_group_ids.insert(iter->second);
      }
    }
    call_check = (call_check && (input_group_ids.size() == 1));
    if (call_check) {
      auto input_group_id = *input_group_ids.cbegin();
      if (group_id != input_group_id) {
        GELOGW("Fusion: node[name:%s(%s) with group id:%ld and diff from it's input nodes's group id:%ld ",
               name.c_str(), type.c_str(), group_id, input_group_id);
      }
    }
  }
  GELOGD("Fusion: get fusion group numbers [%zu].", fusion_nodes.size());
  return SUCCESS;
}

Status TaskGenerator::GenerateTaskForFusionNode(Node *const node,
                                                const std::map<int64_t, std::vector<NodePtr>> &fusion_nodes,
                                                std::unordered_set<Node *> &fusion_nodes_seen) {
  int64_t group_key;
  const auto &fusion_op_desc = node->GetOpDesc();
  // without attr or has been seen, skip directly
  if (!(ge::AttrUtils::GetInt(fusion_op_desc, ATTR_NAME_FUSION_GROUP_KEY, group_key) &&
        (fusion_nodes_seen.count(node) == 0U))) {
    return SUCCESS;
  }
  GELOGI("Fusion: start fusion group index[%ld], nodes size[%zu].", group_key, fusion_nodes.at(group_key).size());
  // If op_desc have this attr, call nodes with same group key in a stream together
  for (const auto &fusion_node : fusion_nodes.at(group_key)) {
    auto &task_defs = node_id_2_node_tasks_[fusion_node->GetOpDesc()->GetId()];
    GE_ASSERT_SUCCESS(GenTaskForNormalNode(fusion_node.get(), "Fusion inner", task_defs));
    fusion_ordered_node_list_.emplace_back(fusion_node->GetOpDesc()->GetId());
    fusion_task_node_name_list_.emplace_back(fusion_op_desc->GetName());
    // record nodes which have call generate task successfully
    fusion_nodes_seen.insert(fusion_node.get());
  }
  return SUCCESS;
}

Status TaskGenerator::GenTaskForFusionNodes(const std::map<int64_t, std::vector<NodePtr>> &fusion_nodes) {
  fusion_nodes_seen_.clear();
  fusion_ordered_node_list_.clear();
  fusion_task_node_name_list_.clear();
  int64_t group_key;
  for (const auto node : nodes_) {
    const auto &op_desc = node->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc);

    const auto &name = op_desc->GetName();
    const auto &type = op_desc->GetType();
    GE_ASSERT_SUCCESS(GenerateTaskForFusionNode(node, fusion_nodes, fusion_nodes_seen_),
                      "[Call][GenerateTaskForFusionNode] node:%s(%s) failed", name.c_str(), type.c_str());
    // continue directly
    if (ge::AttrUtils::GetInt(op_desc, ATTR_NAME_FUSION_GROUP_KEY, group_key)) {
      GELOGI("Fusion node[name:%s, type:%s] do not need generate task again.", name.c_str(), type.c_str());
      continue;
    }
    const auto node_id = op_desc->GetId();
    auto &task_defs = node_id_2_node_tasks_[node_id];
    GE_ASSERT_SUCCESS(GenTaskForNormalNode(node, "Fusion outter", task_defs));
    fusion_ordered_node_list_.emplace_back(node_id);
    fusion_task_node_name_list_.emplace_back(name);
  }
  return SUCCESS;
}

Status TaskGenerator::GenerateTaskForNodes(const std::vector<Node *> nodes) {
  if (NeedDoFusionTask()) {
    // fusion node场景所有node都重新做gen task，因为fusion node不只是对自己GenTask，还会对同一个group key的node GenTask
    // todo 可以改成第二次GenTask时就只对部分node GenTaskForNormalNode
    std::map<int64_t, std::vector<NodePtr>> fusion_nodes;
    GE_RUN_PERF(TaskGenerator, SaveFusionNodes, fusion_nodes, nodes_);
    if (!fusion_nodes.empty()) {
      for(const auto &node : nodes_) {
        // 个别节点会二次GenTask，所以需要将第一次的结果清空
        node_id_2_node_tasks_[node->GetOpDesc()->GetId()].clear();
      }
      return GenTaskForFusionNodes(fusion_nodes);
    }
  }
  for(const auto &node : nodes) {
    // 个别节点会二次GenTask，所以需要将第一次的结果清空
    node_id_2_node_tasks_[node->GetOpDesc()->GetId()].clear();
  }

  static std::map<GenTaskCallKey, GenTaskCall> handles =
      {{GenTaskCallKey::kAtomicEngine, &TaskGenerator::GenerateTaskForNormalNode},
       {GenTaskCallKey::kFftsEngine, &TaskGenerator::GenerateTaskForFftsNode}};
  thread_pool_ = MakeUnique<ThreadPool>("ge_gentask", GetThreadNum(), false);
  GE_ASSERT_NOTNULL(thread_pool_);
  std::vector<std::future<Status>> vector_future;
  int32_t device_id = kInvalidDeviceId;
  // 离线场景不会SetDevice，所以离线场景GetDevice会报错，可以通过device id是否是-1判断是在线or离线。在线场景需要给子线程SetDevice
  (void)aclrtGetDevice(&device_id);
  GELOGI("Get device id %d", device_id);
  for (const auto node : nodes) {
    const auto key = GetKey(node);
    auto &task_defs = node_id_2_node_tasks_[node->GetOpDesc()->GetId()];
    // key must be valid
    const auto &func = handles.find(key)->second;
    std::future<Status> f = thread_pool_->commit(func, this, node, key_2_string.at(key), std::ref(task_defs),
                                                 GetThreadLocalContext(), error_message::GetErrMgrContext(), device_id);
    if (f.valid()) {
      vector_future.emplace_back(std::move(f));
    }
  }

  for (auto &i : vector_future) {
    GE_CHK_STATUS_RET(i.get(), "[GenTask] Fail!");
  }
  thread_pool_.reset(nullptr);
  ffts_inner_thread_pool_.reset(nullptr);
  return SUCCESS;
}

Status TaskGenerator::GenerateTask(const ComputeGraphPtr &graph, Model &model) {
  GE_RUN_PERF(TaskGenerator, PrepareForGenerateTask, graph);
  std::string soc_version;
  (void)GetThreadLocalContext().GetOption(ge::SOC_VERSION, soc_version);
  if (kNanoSocVersion.count(soc_version) > 0) {
    PreRuntimeParam runtime_param;
    auto model_ptr = ge::MakeShared<ge::Model>(model);
    InitRuntimeParams(model_ptr, runtime_param);
    GE_RUN_PERF(TaskGenerator, InitZeroCopyInfo, graph, runtime_param);
  }
  GE_RUN_PERF(TaskGenerator, GenerateTaskForNodes, nodes_);
  return SUCCESS;
}

Status TaskGenerator::InitRuntimeParams(const ModelPtr &model_ptr, PreRuntimeParam &runtime_param) {
  (void)AttrUtils::GetInt(model_ptr, ATTR_MODEL_MEMORY_SIZE, runtime_param.mem_size);
  (void)AttrUtils::GetInt(model_ptr, ATTR_MODEL_WEIGHT_SIZE, runtime_param.weight_size);
  return SUCCESS;
}

Status TaskGenerator::InitZeroCopyInfo(const ComputeGraphPtr &graph, const PreRuntimeParam &runtime_param) {
  uint32_t search_id = 0U;
  SymbolToAnchors symbol_to_anchors;
  AnchorToSymbol anchor_to_symbol;
  GE_ASSERT_SUCCESS(GraphUtils::GetRefMapping(graph, symbol_to_anchors, anchor_to_symbol),
                    "[Call][GetRefMapping] for graph:%s failed.", graph->GetName().c_str());
  for (const auto &node : graph->GetAllNodes()) {
    GE_CHECK_NOTNULL(node);
    const auto &op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    if (node->GetOwnerComputeGraph() != graph) {
      return SUCCESS;
    }

    if (kDataOpType.count(node->GetType()) > 0U) {
      GE_CHK_STATUS_RET(GenZeroCopyTable(node->GetOpDesc(), search_id, true),
                        "GenZeroCopyTable for node[%s] input failed", node->GetName().c_str());
      for (const auto &anchor : node->GetAllOutDataAnchors()) {
        for (const auto &in_anchors : anchor->GetPeerInDataAnchors()) {
          if (in_anchors == nullptr) {
            continue;
          }
          GE_IF_BOOL_EXEC(in_anchors == nullptr, continue);
          auto owner_node = in_anchors->GetOwnerNode();
          auto index = in_anchors->GetIdx();
          GE_CHK_STATUS_RET(SetAnchorsOffset(owner_node, true, index, runtime_param, node->GetOpDesc()));
        }
      }
    }
    if (node->GetType() == NETOUTPUT) {
      GE_CHK_STATUS_RET(GenZeroCopyTable(node->GetOpDesc(), search_id, false),
                        "GenZeroCopyTable for node[%s] input failed", node->GetName().c_str());
      for (const auto &anchor : node->GetAllInDataAnchors()) {
        GE_ASSERT_SUCCESS(
            SetNetOutputNodeInAnchorAndPeerOffset(anchor, runtime_param, symbol_to_anchors, anchor_to_symbol));
        auto in_anchor = anchor->GetPeerOutAnchor();
        if (in_anchor == nullptr) {
            continue;
          }
        auto owner_node = in_anchor->GetOwnerNode();
        auto index = in_anchor->GetIdx();
        GE_CHK_STATUS_RET(SetAnchorsOffset(owner_node, false, index, runtime_param, node->GetOpDesc()));
      }
    }
  }
  return SUCCESS;
}

Status TaskGenerator::SetNetOutputNodeInAnchorAndPeerOffset(const InDataAnchorPtr &in_anchor,
                                                            const PreRuntimeParam &runtime_param,
                                                            SymbolToAnchors &symbol_to_anchors,
                                                            AnchorToSymbol &anchor_to_symbol) {
  auto out_node = in_anchor->GetOwnerNode();
  auto out_node_inanchor_index = in_anchor->GetIdx();
  auto out_opdesc = out_node->GetOpDesc();
  auto ori_offset_list = out_opdesc->GetInputOffset();
  const int64_t ori_offset = ori_offset_list.at(out_node_inanchor_index);
  const auto zero_copy_offset_to_ids = PreModelPartitionUtils::GetInstance().GetZeroCopyTable();
  GE_ASSERT_TRUE(zero_copy_offset_to_ids.find(ori_offset) != zero_copy_offset_to_ids.end(),
                 "NetOutput input no[%d] offset[%ld] get zerocopy offset id faild", out_node_inanchor_index,
                 ori_offset);
  const uint32_t offset_to_id = zero_copy_offset_to_ids.find(ori_offset)->second;

  NodeIndexIO out_node_in_anchor_node(out_node, out_node_inanchor_index, kIn);
  const auto anchor_symbol_iter = anchor_to_symbol.find(out_node_in_anchor_node.ToString());
  GELOGD("anchor[%s] origin offset[%ld] set to idx[%u]", out_node_in_anchor_node.ToString().c_str(), ori_offset, offset_to_id);
  if (anchor_symbol_iter != anchor_to_symbol.end()) {
    const auto &node_indexes = symbol_to_anchors[anchor_symbol_iter->second];
    for (const auto &node_index : node_indexes) {
      const bool is_input = node_index.io_type_ == kIn ? true : false;
      const auto peer_node = node_index.node_;
      const uint32_t peer_index = node_index.index_;
      GE_IF_BOOL_EXEC(peer_node == nullptr, continue);
      GE_ASSERT_SUCCESS(
          SetNetOutputNodePeerNodeOffset(peer_node, is_input, peer_index, ori_offset, offset_to_id, runtime_param),
          "NetOutput set peer offset fail, anchor symbol:%s", node_index.ToString().c_str());
    }
  }
  return SUCCESS;
}

Status TaskGenerator::SetNetOutputNodePeerNodeOffset(const NodePtr &peer_node, const bool is_input, uint32_t index,
                                                     const int64_t ori_offset, const uint32_t offset_to_id,
                                                     const PreRuntimeParam &runtime_param) {
  auto peer_op_desc = peer_node->GetOpDesc();
  GE_ASSERT_NOTNULL(peer_op_desc);
  uint64_t args_ddr_type = invalidAddrType;
  std::vector<KernelArgsParam> args_param;
  std::vector<uint64_t> args_offset_vals;
  if (is_input) {
    std::vector<uint32_t> index_to_valid_idx;
    const auto input_data_addr_offset = PreModelUtils::GetInputDataAddrOffset(runtime_param, peer_op_desc, args_param,
                                                                              args_offset_vals, index_to_valid_idx);
    // 对于有可选输入的算子，存在未选输入时，输入index转化为实际的idx来访问相关数组
    index = index_to_valid_idx[index];
    args_ddr_type = args_param.empty() ? invalidAddrType : args_param.at(index).para;
  } else {
    const auto output_data_addr_offset =
        PreModelUtils::GetOutputDataAddrOffset(runtime_param, peer_op_desc, args_param, args_offset_vals);
    args_ddr_type = args_param.empty() ? invalidAddrType : args_param.at(index).para;
  }
  GELOGI("node[%s] %s no.%u, args_ddr_type[%lu]", peer_node->GetName().c_str(), is_input ? "input" : "output", index,
         args_ddr_type);
  if ((args_ddr_type != KERNEL_ARG_UPADTE_ADDR_TYPE_ARGS) && (args_ddr_type != KERNEL_ARG_UPADTE_ADDR_TYPE_WORKSPACE)) {
    return SUCCESS;
  }
  auto ori_offset_list = is_input ? peer_op_desc->GetInputOffset() : peer_op_desc->GetOutputOffset();
  GELOGD("get origin offset[%ld] set offset from[%ld] to idx[%d]", ori_offset_list.at(index), ori_offset, offset_to_id);
  if (ori_offset == ori_offset_list.at(index)) {
    ori_offset_list[index] = offset_to_id;
    if (is_input) {
      peer_op_desc->SetInputOffset(ori_offset_list);
    } else {
      peer_op_desc->SetOutputOffset(ori_offset_list);
    }
    return SUCCESS;
  }
  GELOGW("node[%s] %s no.%u offset[%ld] should equal to %ld", peer_node->GetName().c_str(),
         is_input ? "input" : "output", index, ori_offset_list.at(index), ori_offset);
  return SUCCESS;
}

Status TaskGenerator::SetAnchorsOffset(const NodePtr &owner_node, const bool is_input, const uint32_t index,
                                       const PreRuntimeParam &runtime_param, const OpDescPtr &op_desc) {
  auto owner_node_op_desc = owner_node->GetOpDesc();
  GE_ASSERT_NOTNULL(owner_node_op_desc);
  std::vector<KernelArgsParam> args_param;
  std::vector<uint64_t> args_offset_vals;
  uint64_t args_ddr_type = invalidAddrType;
  uint32_t valid_idx = index;
  if (is_input) {
    std::vector<uint32_t> index_to_valid_idx;
    const auto input_data_addr_offset =
      PreModelUtils::GetInputDataAddrOffset(runtime_param, owner_node_op_desc, args_param, args_offset_vals,
                                            index_to_valid_idx);
    // 对于有可选输入的算子，存在未选输入时，输入index转化为实际的idx来访问相关数组
    valid_idx = index_to_valid_idx[index];
    args_ddr_type = args_param.empty() ? invalidAddrType : args_param.at(valid_idx).para;
  } else {
    const auto output_data_addr_offset =
      PreModelUtils::GetOutputDataAddrOffset(runtime_param, owner_node_op_desc, args_param, args_offset_vals);
    args_ddr_type = args_param.empty() ? invalidAddrType : args_param.at(valid_idx).para;
  }
  GELOGI("node[%s] %s no.%u valid_idx:%u, args_ddr_type[%lu]", owner_node->GetName().c_str(),
         is_input ? "input": "output", index, valid_idx, args_ddr_type);
  if ((args_ddr_type != KERNEL_ARG_UPADTE_ADDR_TYPE_ARGS) &&
      (args_ddr_type != KERNEL_ARG_UPADTE_ADDR_TYPE_WORKSPACE)) {
    return SUCCESS;
  }
  auto base_offset_list = is_input ? owner_node_op_desc->GetInputOffset() : owner_node_op_desc->GetOutputOffset();
  const auto zero_copy_offset_to_ids = PreModelPartitionUtils::GetInstance().GetZeroCopyTable();
  auto base_offset = base_offset_list.at(valid_idx);
  if (zero_copy_offset_to_ids.find(base_offset) != zero_copy_offset_to_ids.end()) {
    const auto offset_to_id = zero_copy_offset_to_ids.find(base_offset)->second;
    base_offset_list[valid_idx] = offset_to_id;
    GELOGD("base offset[%ld] set to idx[%d]", base_offset, offset_to_id);
    GE_CHK_STATUS_RET(SetOpOffset(op_desc, is_input, base_offset, offset_to_id));
  }

  if (is_input) {
    owner_node_op_desc->SetInputOffset(base_offset_list);
  } else {
    owner_node_op_desc->SetOutputOffset(base_offset_list);
  }
  return SUCCESS;
}

Status TaskGenerator::SetOpOffset(const OpDescPtr &op_desc, const bool is_input, const int64_t offset,
                                  const uint32_t offset_to_id) {
  GE_ASSERT_NOTNULL(op_desc);
  auto base_offset_list = is_input ? op_desc->GetOutputOffset() : op_desc->GetInputOffset();
  for (uint32_t i = 0; i < base_offset_list.size(); i++) {
    if (base_offset_list.at(i) == offset) {
      base_offset_list[i] = offset_to_id;
    }
  }

  if (is_input) {
    op_desc->SetOutputOffset(base_offset_list);
  } else {
    op_desc->SetInputOffset(base_offset_list);
  }
  return SUCCESS;
}

Status TaskGenerator::GenZeroCopyTable(const OpDescPtr &op_desc, uint32_t &search_id, const bool is_input) {
  std::vector<int64_t> zero_copy_basic_offset;
  std::vector<int64_t> zero_copy_relative_offset;
  (void)ge::AttrUtils::GetListInt(op_desc, ATTR_ZERO_COPY_BASIC_OFFSET, zero_copy_basic_offset);
  (void)ge::AttrUtils::GetListInt(op_desc, ATTR_ZERO_COPY_RELATIVE_OFFSET, zero_copy_relative_offset);
  GE_ASSERT_TRUE((zero_copy_basic_offset.size() == zero_copy_relative_offset.size()),
                 "[Check][Param] basic_offset_size:%zu should be equal to relative_offset_size:%zu",
                 zero_copy_basic_offset.size(), zero_copy_relative_offset.size());
  auto base_offset_list = is_input ? op_desc->GetOutputOffset() : op_desc->GetInputOffset();
  constexpr uint32_t offset_size = 8U;
  for (const int64_t base_offset : base_offset_list) {
    PreModelPartitionUtils::GetInstance().SetZeroCopyTable(base_offset, search_id * offset_size);
    GELOGI("zero_copy_offset_to_ids_ node name is %s, offset[%ld], ids[%u].",
           op_desc->GetName().c_str(), base_offset, search_id * offset_size);
    search_id++;
    for (size_t position = 0U; position < zero_copy_basic_offset.size(); position++) {
      if ((base_offset == zero_copy_basic_offset[position]) && (zero_copy_relative_offset[position] != 0)) {
        PreModelPartitionUtils::GetInstance().SetZeroCopyTable(base_offset + zero_copy_relative_offset[position],
                                                               (search_id++) * offset_size);
        GELOGI("zero_copy_offset_to_ids_ offset[%ld], ids[%u].", base_offset + zero_copy_relative_offset[position],
               search_id * offset_size);
      }
    }
  }

  return SUCCESS;
}

Status TaskGenerator::GetTaskInfo(const ComputeGraphPtr &graph, uint64_t session_id, Model &model) {
  GE_ASSERT_NOTNULL(graph);
  session_id_ = session_id;
  GELOGD("Begin to gen task info with graph:%s session_id:%lu", graph->GetName().c_str(), session_id);
  GE_RUN_PERF(TaskGenerator, GenerateTask, graph, model);
  return SUCCESS;
}

Status TaskGenerator::ReGetTaskInfo(const ComputeGraphPtr &comp_graph) {
  std::unordered_set<Node *> first_last_nodes;
  // 这里除了标记首尾节点，还会Reset op kernel lib name，主要是是拆流时插入的send recv
  GE_ASSERT_SUCCESS(MarkFirstAndLastOpsForGraph(comp_graph, first_last_nodes));
  const auto &all_nodes = comp_graph->GetNodes(comp_graph->GetGraphUnknownFlag(), nullptr, ffts_filter);
  std::unordered_set<Node *> need_to_gen_task_nodes;
  for (const auto &node : all_nodes) {
    if (!NoNeedGenTask(node->GetOpDesc()) &&
        (std::find(nodes_.begin(), nodes_.end(), node.get()) == nodes_.end())) {
      need_to_gen_task_nodes.insert(node.get());
      nodes_.emplace_back(node.get());
    }
  }
  if (!need_to_gen_task_nodes.empty()) {
    need_to_gen_task_nodes.insert(first_last_nodes.begin(), first_last_nodes.end());
    std::vector<Node *> second_gen_task_nodes;
    for (auto &node : need_to_gen_task_nodes) {
      second_gen_task_nodes.emplace_back(node);
    }
    GE_ASSERT_SUCCESS(GenerateTaskForNodes(second_gen_task_nodes));
    for (const auto &node : need_to_gen_task_nodes) {
      auto stream_id = node->GetOpDesc()->GetStreamId();
      const auto &iter = node_id_2_node_tasks_.find(node->GetOpDesc()->GetId());
      GE_ASSERT_TRUE(iter != node_id_2_node_tasks_.end(), "node: %s doesn't have taskdef", node->GetNamePtr());
      bool has_attached_stream = node->GetOpDesc()->HasValidAttachedStreamId();
      if (!has_attached_stream) {
        RefreshTaskDefStreamId(has_attached_stream, stream_id, stream_id, iter->second);
      }
    }
    GE_ASSERT_SUCCESS(UpdateTaskDef());
  }
  return SUCCESS;
}

Status TaskGenerator::UpdateTaskDef() {
  for (const auto &node : nodes_) {
    const auto &iter = node_id_2_node_tasks_.find(node->GetOpDesc()->GetId());
    if ((iter != node_id_2_node_tasks_.end()) && (!iter->second.empty())) {
      auto &task_defs = iter->second;
      GE_ASSERT_SUCCESS(OpsKernelBuilderManager::Instance().UpdateTask(*node, task_defs),
                        "[Update][Task] fail for op:%s(%s)", node->GetName().c_str(), node->GetType().c_str());
    }
  }
  return SUCCESS;
}

Status TaskGenerator::GenModelTaskDef(const ComputeGraphPtr &graph, uint64_t session_id, Model &model) {
  // Init and serialize model_task_def
  ModelTaskDef model_task_def;
  model_task_def.set_memory_size(run_context_->dataMemSize);
  model_task_def.set_weight_size(run_context_->weightMemSize);
  std::vector<TaskDef> task_def_list;
  task_def_list.reserve(nodes_.size());
  if (!fusion_ordered_node_list_.empty()) {
    GE_ASSERT_TRUE(fusion_ordered_node_list_.size() == fusion_task_node_name_list_.size());
    for (size_t i = 0U; i < fusion_ordered_node_list_.size(); i++) {
      const auto &iter = node_id_2_node_tasks_.find(fusion_ordered_node_list_[i]);
      GE_ASSERT_TRUE(iter != node_id_2_node_tasks_.end());
      op_names_.insert(op_names_.end(), iter->second.size(), fusion_task_node_name_list_[i]);
      task_def_list.insert(task_def_list.end(), iter->second.begin(), iter->second.end());
    }
  } else {
    for (const auto &node : graph->GetNodes(graph->GetGraphUnknownFlag(), nullptr, ffts_filter)) {
      if(!NoNeedGenTask(node->GetOpDesc())) {
        const auto &iter = node_id_2_node_tasks_.find(node->GetOpDesc()->GetId());
        GE_ASSERT_TRUE(iter != node_id_2_node_tasks_.end(), "node %s does not gen task", node->GetNamePtr());
        op_names_.insert(op_names_.end(), iter->second.size(), node->GetOpDesc()->GetName());
        task_def_list.insert(task_def_list.end(), iter->second.begin(), iter->second.end());
      }
    }
  }
  GELOGI("Graph %s gen task success, task list:%zu, logic mem base:%p, logic weight base:%p, logic var base:%p",
         graph->GetName().c_str(), task_def_list.size(), run_context_->dataMemBase, run_context_->weightMemBase,
         var_mem_base_);
  for (const auto &task_def_temp : task_def_list) {
    auto *task_def = model_task_def.add_task();
    GE_ASSERT_NOTNULL(task_def);
    *task_def = task_def_temp;
  }
  GE_RUN_PERF(TaskGenerator, AddModelTaskToModel, model_task_def, session_id, model, *run_context_);
  return SUCCESS;
}

Status TaskGenerator::FindProfilingNodeIndex(const ComputeGraphPtr &graph, ProfilingPoint &profiling_point) {
  if (nodes_.empty()) {
    GE_CHK_STATUS_RET(FilterCandidatesNodes(graph), "[Call][FilterCandidatesNodes] failed, graph:%s.",
                      graph->GetName().c_str());
  }
  ProfilingTaskUtils profiling_task_utils(nodes_);
  return profiling_task_utils.FindProfilingTaskIndex(graph, profiling_point);
}
}  // namespace ge
