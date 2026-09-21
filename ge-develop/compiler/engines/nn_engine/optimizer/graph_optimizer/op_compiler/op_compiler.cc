/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/op_compiler/op_compiler.h"
#include <utility>
#include <vector>
#include "common/configuration.h"
#include "common/fe_context_utils.h"
#include "common/fe_inner_attr_define.h"
#include "common/fe_type_utils.h"
#include "common/ffts_plus_type.h"
#include "common/math_util.h"
#include "common/fe_op_info_common.h"
#include "common/util/op_info_util.h"
#include "common/scope_allocator.h"
#include "framework/common/types.h"
#include "ge/ge_api_types.h"
#include "graph/tuning_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/op_desc_utils_ex.h"
#include "graph/utils/op_type_utils.h"
#include "register/graph_optimizer/fusion_common/unknown_shape_utils.h"
#include "ops_store/ops_kernel_manager.h"
#include "platform/platform_info.h"
#include "param_calculate/tensor_compute_util.h"
#include "common/weight_compress_utils.h"
#include "common/sgt_slice_type.h"
#include "common/graph/fe_graph_utils.h"
#include "graph_optimizer/op_compiler/op_format_tune.h"

namespace fe {
namespace {
const size_t COMPRESS_PARAMETER_SIZE = 8;
const size_t MULTI_CORE_COMPRESS_PARAMETER_SIZE = 9;
const size_t COMPRESS_PARAMETER_INFOSIZE_INDEX = 1;
const int64_t WEIGHT_SIZE_THRESHOLD = 262144;  // 256K
const double WEIGHT_CHECK_THRESHOLD = 0.8;
const uint64_t CUBE_K_OF_INT8 = 32;
const uint32_t kFFTSNoCompileThreadNum = 3;
constexpr const char* kCannKbImplType = "impl_type";
constexpr const char* kCannKbDynamicCompileStatic = "dynamic_compile_static";
constexpr const char* kCannKbOpImplSwitch = "op_impl_switch";

using RelatedNodesMap = std::map<ge::NodePtr, std::shared_ptr<std::vector<ge::NodePtr>>>;

bool IsNodeQualified(const ge::NodePtr &node, ffts::ThreadSliceMapPtr &slice_info_ptr) {
  if (node->GetType() == OP_TYPE_END || node->GetType() == OP_TYPE_PLACE_HOLDER) {
    return false;
  }
  slice_info_ptr = node->GetOpDesc()->TryGetExtAttr(ffts::kAttrSgtStructInfo, slice_info_ptr);
  if (slice_info_ptr == nullptr) {
    FE_LOGD("Node %s does not have sgt slice info.", node->GetName().c_str());
    return false;
  }

  return true;
}

void FullOriginalNode(const ge::NodePtr &node, const ffts::ThreadSliceMapPtr &slice_info_ptr) {
  if (slice_info_ptr->original_node.empty()) {
    auto op_name = node->GetName();
    const string inner_str = "_lxslice";
    auto pos = op_name.find(inner_str);
    if (pos != string::npos && (pos + 1 < op_name.size())) {
      slice_info_ptr->original_node = op_name.substr(0, pos);
      auto pos_next_underline = op_name.find("_", pos + 1);
      if (pos_next_underline != string::npos && (pos_next_underline + 1 < op_name.size())) {
        slice_info_ptr->original_node += op_name.substr(pos_next_underline);
        FE_LOGI("op_name %s's original node name is %s.", op_name.c_str(),
                slice_info_ptr->original_node.c_str());
      }
    } else {
      FE_LOGW("op_name %s is not valid.", op_name.c_str());
    }
  }
}

/*
 *  @ingroup fe
 *  @brief   Count all thread0 nodes and all their related same layer nodes.
 *  In FFTS+ mode, all sliced nodes in the same layer are originated from
 *  a same original node.
 *  Their type, shape, format, attr are the same and we just need compile
 *  them once. Other related same layer nodes will copy all necessary attributes
 *  from the first compiled one(in tbe_json_parse.cc and AddRelatedThreadNode in
 *  fusion_graph_merge.cc).
 *  @param   [out]thread1_nodes: key: original node name, value: the pointer thread 1 node.
 *  @param   [out]slice0_node_to_same_layer_nodes: key: the pointer of thread 0 node,
 *  value: all related same layer nodes of this thread 0 node.
 *  @return  SUCCESS or FAILED
 */
Status CountThread1Node(const ge::ComputeGraph::Vistor<ge::NodePtr> &direct_nodes,
                        std::map<string, ge::NodePtr> &thread1_nodes,
                        RelatedNodesMap &thread1_node_to_other_threads_nodes) {
  for (auto &node : direct_nodes) {
    ffts::ThreadSliceMapPtr slice_info_ptr = nullptr;
    bool ret = IsNodeQualified(node, slice_info_ptr);
    if (!ret) {
      FE_LOGD("Node %s is not qualified as an FFTs node.", node->GetName().c_str());
      continue;
    }

    uint32_t thread_mode = 1;
    if (!ge::AttrUtils::GetInt(node->GetOpDesc(), kThreadMode, thread_mode) || thread_mode != 0) {
      FE_LOGD("node %s is not in manual mode.", node->GetName().c_str());
      return FAILED;
    }

    if (slice_info_ptr->slice_instance_num <= kFFTSNoCompileThreadNum) {
      return FAILED;
    }

    bool is_virtual_op = false;
    (void)ge::AttrUtils::GetBool(node->GetOpDesc(), ge::ATTR_NAME_NOTASK, is_virtual_op);
    if (is_virtual_op) {
      continue;
    }

    if (slice_info_ptr->thread_id != 1) {
      continue;
    }

    FullOriginalNode(node, slice_info_ptr);
    if (!slice_info_ptr->original_node.empty()) {
      FE_LOGD("Add node %s into origin node %s's %p map.", node->GetName().c_str(),
              slice_info_ptr->original_node.c_str(), node.get());
      thread1_nodes.emplace(slice_info_ptr->original_node, node);
      FE_MAKE_SHARED(thread1_node_to_other_threads_nodes[node] =
                     std::make_shared<std::vector<ge::NodePtr>>(),
                     return FAILED);
    }
  }
  return SUCCESS;
}

void CountOtherThreadsNode(const ge::ComputeGraph::Vistor<ge::NodePtr> &direct_nodes,
                           const std::map<string, ge::NodePtr> &thread1_nodes,
                           RelatedNodesMap &thread1_node_to_other_threads_nodes) {
  for (const auto &node : direct_nodes) {
    ffts::ThreadSliceMapPtr slice_info_ptr = nullptr;
    bool ret = IsNodeQualified(node, slice_info_ptr);
    if (!ret || slice_info_ptr->thread_id == 0 || slice_info_ptr->thread_id == 1 ||
        slice_info_ptr->thread_id == (slice_info_ptr->slice_instance_num - 1) ||
        slice_info_ptr->slice_instance_num < kMinThreadNum) {
      continue;
    }
    FullOriginalNode(node, slice_info_ptr);
    FE_LOGD("Node %s, thread_id: %u, slice instance number: %u, original node: [%s].", node->GetName().c_str(),
            slice_info_ptr->thread_id, slice_info_ptr->slice_instance_num,
            slice_info_ptr->original_node.c_str());
    if (slice_info_ptr->original_node.empty()) {
      continue;
    }
    auto iter = thread1_nodes.find(slice_info_ptr->original_node);
    if (iter == thread1_nodes.end()) {
      FE_LOGD("Cannot find the original node [%s] for node [%s].", slice_info_ptr->original_node.c_str(),
              node->GetName().c_str());
      continue;
    } else {
      node->GetOpDesc()->SetExtAttr(kAttrThread1Node, iter->second);
      (void)ge::AttrUtils::SetStr(node->GetOpDesc(), kOriginalNode, slice_info_ptr->original_node);
      (void)node->GetOpDesc()->DelAttr(IS_NEED_TUNEFORMAT);
      auto others_iter = thread1_node_to_other_threads_nodes.find(iter->second);
      if (others_iter != thread1_node_to_other_threads_nodes.end()) {
        others_iter->second->emplace_back(node);
      }
    }
  }
}

Status GetParam(const ge::OpDescPtr& op_desc, int64_t& fm_h, int64_t& fm_w, int64_t& weight_h, int64_t& weight_w,
                int64_t& cin, int64_t& cout, uint64_t& cube_k) {
  auto fm_in_tensor_desc = op_desc->GetInputDescPtr(0);
  FE_CHECK_NOTNULL(fm_in_tensor_desc);
  auto weight_tensor_desc = op_desc->GetInputDescPtr(1);
  FE_CHECK_NOTNULL(weight_tensor_desc);
  auto fm_out_tensor_desc = op_desc->GetOutputDescPtr(0);
  FE_CHECK_NOTNULL(fm_out_tensor_desc);
  if (!GetDimValueByFormatAndShape(fm_out_tensor_desc->GetOriginFormat(), fm_out_tensor_desc->GetOriginShape(), "H",
                                   fm_h)) {
    REPORT_FE_ERROR(
        "[SubGraphOpt][Compile][WeightCmprs][node %s] Failed to obtain the dimension H of the output feature map with format %d.",
        op_desc->GetName().c_str(), fm_out_tensor_desc->GetOriginFormat());
    return FAILED;
  }
  if (!GetDimValueByFormatAndShape(fm_out_tensor_desc->GetOriginFormat(), fm_out_tensor_desc->GetOriginShape(), "W",
                                   fm_w)) {
    REPORT_FE_ERROR(
        "[SubGraphOpt][Compile][WeightCmprs][node %s]Failed to get Dim W of output feature map with format %d",
        op_desc->GetName().c_str(), fm_out_tensor_desc->GetOriginFormat());
    return FAILED;
  }
  if (!GetDimValueByFormatAndShape(weight_tensor_desc->GetOriginFormat(), weight_tensor_desc->GetOriginShape(), "H",
                                   weight_h)) {
    REPORT_FE_ERROR("[SubGraphOpt][Compile][WeightCmprs][node %s] Failed to get Dim H of weight with format %d.",
                    op_desc->GetName().c_str(), weight_tensor_desc->GetOriginFormat());
    return FAILED;
  }
  if (!GetDimValueByFormatAndShape(weight_tensor_desc->GetOriginFormat(), weight_tensor_desc->GetOriginShape(), "W",
                                   weight_w)) {
    REPORT_FE_ERROR("[SubGraphOpt][Compile][WeightCmprs][node %s] Failed to retrieve Dim W of the weight with format %d",
                    op_desc->GetName().c_str(), weight_tensor_desc->GetOriginFormat());
    return FAILED;
  }
  if (!GetDimValueByFormatAndShape(fm_in_tensor_desc->GetOriginFormat(), fm_in_tensor_desc->GetOriginShape(), "C",
                                   cin)) {
    REPORT_FE_ERROR(
        "[SubGraphOpt][Compile][WeightCmprs][node %s] Failed to get Dim C of input feature map with format %d",
        op_desc->GetName().c_str(), fm_in_tensor_desc->GetOriginFormat());
    return FAILED;
  }
  if (!GetDimValueByFormatAndShape(fm_out_tensor_desc->GetOriginFormat(), fm_out_tensor_desc->GetOriginShape(), "C",
                                   cout)) {
    REPORT_FE_ERROR(
        "[SubGraphOpt][Compile][WeightCmprs] Failed to get Dim C of output feature map for node %s with format %d",
        op_desc->GetName().c_str(), fm_out_tensor_desc->GetOriginFormat());
    return FAILED;
  }
  // if data type is int8, cube_k should be 32
  cube_k = (fm_in_tensor_desc->GetDataType() == ge::DT_INT8) ? CUBE_K_OF_INT8 : cube_k;
  return SUCCESS;
}
}  // namespace

void ProcCompressOpMemType(const ge::NodePtr &node) {
  // input TENSOR_INDEX_COMPRESS_INDEX(2) add new anchor to host op FC, so sgt set ATTR_NAME_INPUT_MEM_TYPE_LIST size
  // not equal with node in size
  bool ffts_node = false;
  (void)ge::AttrUtils::GetBool(node->GetOpDesc(), kTypeFFTSPlus, ffts_node);
  if (!ffts_node) {
    return;
  }
  std::vector<int64_t> memory_type;
  (void)ge::AttrUtils::GetListInt(node->GetOpDesc(), ge::ATTR_NAME_INPUT_MEM_TYPE_LIST, memory_type);
  size_t in_size = node->GetOpDesc()->GetInputsSize();
  if (memory_type.size() == in_size) {
    return;
  }
  FE_LOGD("Resize in memory type from %zu to %zu.", memory_type.size(), in_size);
  memory_type.clear();
  memory_type.resize(in_size, RT_MEMORY_HBM);
  (void)ge::AttrUtils::SetListInt(node->GetOpDesc(), ge::ATTR_NAME_INPUT_MEM_TYPE_LIST, memory_type);
  std::vector<int64_t> input_offset = node->GetOpDesc()->GetInputOffset();
  input_offset.resize(in_size, 0);
  node->GetOpDesc()->SetInputOffset(input_offset);
  return;
}

OpCompiler::OpCompiler(const std::string &compiler_name, const std::string &engine_name,
                       const LxFusionOptimizerPtr &lx_fusion_optimizer)
    : init_flag_(false),
      compiler_name_(compiler_name),
      engine_name_(engine_name),
      need_post_process_(true),
      lx_fusion_optimizer_ptr_(lx_fusion_optimizer) {}

OpCompiler::OpCompiler(const std::string &compiler_name, const std::string &engine_name, const bool need_post_process,
                       const LxFusionOptimizerPtr &lx_fusion_optimizer)
        : init_flag_(false),
          compiler_name_(compiler_name),
          engine_name_(engine_name),
          need_post_process_(need_post_process),
          lx_fusion_optimizer_ptr_(lx_fusion_optimizer) {}

const string& OpCompiler::GetCompilerName() const {
  return compiler_name_;
}

bool OpCompiler::IsNeedPostProcess() const {
  return need_post_process_;
}

OpCompiler::~OpCompiler() {}

/*
 *  @ingroup fe
 *  @brief   initialize op compiler
 *  @return  SUCCESS or FAILED
 */
Status OpCompiler::Initialize() {
  // if graph optimizer has been initialized, return SUCCESS
  if (init_flag_) {
    FE_LOGW("OpCompiler has been initialized.");
    return SUCCESS;
  }

  init_flag_ = true;
  return SUCCESS;
}

/*
 *  @ingroup fe
 *  @brief   finalize op compiler
 *  @return  SUCCESS or FAILED
 */
Status OpCompiler::Finalize() {
  if (!init_flag_) {
    FE_LOGW("OpCompiler finalize is not allowed before initialize first.");
    return SUCCESS;
  }

  init_flag_ = false;
  return SUCCESS;
}

/*
 *  @ingroup fe
 *  @brief   add node to fusion_node_map according to scope id
 *  @param   [in]  node           node pointer
 *  @param   [in] scope_id         scope id
 *  @param   [out] fusion_node_map  scope id and node map
 *  @return  SUCCESS or FAILED
 */
Status OpCompiler::AddNodeToFusionMap(ge::Node &node, const int64_t scope_id, ScopeNodeIdMap &fusion_node_map) {
  ScopeNodeIdMap::iterator nodelist_it = fusion_node_map.find(scope_id);
  if (nodelist_it == fusion_node_map.end()) {
    std::vector<ge::Node *> node_list_new;
    node_list_new.push_back(&node);
    fusion_node_map.insert(std::pair<int64_t, std::vector<ge::Node *>>(scope_id, node_list_new));
  } else {
    nodelist_it->second.push_back(&node);
  }

  return SUCCESS;
}

/*
 *  @ingroup fe
 *  @brief   get scope node map
 *  @param   [in]  graph        node pointer
 *  @param   [out] fusion_map    scope id and node map
 *  @return  SUCCESS or FAILED
 */
Status OpCompiler::GetScopeNodeMap(const ge::ComputeGraph& graph, ScopeNodeIdMap& fusion_node_map) {
  std::vector<ge::NodePtr> all_nodes;
  for (auto const &node : graph.GetDirectNode()) {
    all_nodes.push_back(node);
  }
  if (GetScopeNodeMap(graph, all_nodes, fusion_node_map) != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][Compile][GetScopeNdMap] Failed to get scope_node_map.");
    return FAILED;
  }

  return SUCCESS;
}

Status OpCompiler::VerifyScopeIdAttr(const int64_t &scope_id, const bool &is_l1_fusion,
                                     std::map<int64_t, bool> &fusion_scope_type_map) {
  auto scope_iter = fusion_scope_type_map.find(scope_id);
  if (scope_iter != fusion_scope_type_map.end()) {
    if (scope_iter->second != is_l1_fusion) {
      if (is_l1_fusion) {
        FE_LOGE(
            "The scopeId[%ld] of this node is used for L1 fusion"
            "while the same scopeId of other node is used for UB fusion."
            "The scope id of L1 and UB fusion must not be the same.",
            scope_id);
      } else {
        FE_LOGE(
            "The scopeId[%ld] of this node is used for UB fusion,"
            "while the same scopeId of another node is used for L1 fusion."
            "The scope id for L1 and UB fusion must not be the same.",
            scope_id);
      }
      return FAILED;
    }
  } else {
    fusion_scope_type_map.emplace(scope_id, is_l1_fusion);
  }
  return SUCCESS;
}

Status OpCompiler::AddNormalTbeNodeIntoMap(ge::NodePtr node, ge::OpDescPtr op_desc_ptr,
                                           ScopeNodeIdMap &fusion_node_map,
                                           std::map<int64_t, bool> &fusion_scope_type_map) {
  int64_t scope_id = 0;
  bool is_l1_fusion = false;
  bool has_scope_id = GetFusionScopeAttr(op_desc_ptr, scope_id, is_l1_fusion);
  string op_type = op_desc_ptr->GetType();
  int64_t tmp_imply_type = 0;
  if (!ge::AttrUtils::GetInt(op_desc_ptr, FE_IMPLY_TYPE, tmp_imply_type)) {
    REPORT_FE_ERROR("[SubGraphOpt][Compile][AddNdIntoMap] get imply type failed, op[%s, optype[%s]].",
                    op_desc_ptr->GetName().c_str(), op_type.c_str());
    return OP_COMPILER_CHECK_FALSE_FAILED;
  }

  OpImplType impl_type = GetMainImplType<OpImplType>(tmp_imply_type);
  auto engine_name = node->GetOpDesc()->GetOpEngineName();
  if (IsInvalidImplType(engine_name, impl_type)) {
    return SUCCESS;
  }
  bool is_tbe = IsTbe(impl_type);
  /* In two scenario the scope id will be negative and duplicated.
   * 1. automatic ub fusion: some nodes are all -1 and they are expected
   *    to compile alone.
   * 2. ms-tuning process, some nodes will be duplicated and their negative scope id is also duplicated. */
  if (has_scope_id && is_tbe && scope_id >= 0) {
    if (VerifyScopeIdAttr(scope_id, is_l1_fusion, fusion_scope_type_map) != SUCCESS) {
      REPORT_FE_ERROR("[SubGraphOpt][Compile][CheckScopeId] The scope_id of node [%s, type: %s], is not valid.",
                      node->GetName().c_str(), op_type.c_str());
      return FAILED;
    }
    if (AddNodeToFusionMap(*node, scope_id, fusion_node_map) != SUCCESS) {
      REPORT_FE_ERROR("[SubGraphOpt][Compile][AddNdIntoMap] Failed to add node to Fusion Map, node [%s, type: %s].",
                      node->GetName().c_str(), op_type.c_str());
      return FAILED;
    }
  } else {
    FEOpsStoreInfo op_store_info;
    if (Configuration::Instance(node->GetOpDesc()->GetOpEngineName()).
            GetOpStoreInfoByImplType(static_cast<OpImplType>(tmp_imply_type), op_store_info) != SUCCESS) {
      REPORT_FE_ERROR("[SubGraphOpt][Compile][AddNdIntoMap] Failed to get op store info for node [%s] by impl_type [%ld].",
                      node->GetName().c_str(), tmp_imply_type);
      return OP_COMPILER_CHECK_FALSE_FAILED;
    }
    if (op_store_info.need_compile) {
      std::vector<ge::Node *> node_vec;
      node_vec.push_back(node.get());
      int64_t single_op_scope_id = ScopeAllocator::Instance().AllocateNegScopeId();
      fusion_node_map[single_op_scope_id] = node_vec;
      (void)ScopeAllocator::SetScopeAttr(op_desc_ptr, single_op_scope_id);
    }
  }
  return SUCCESS;
}

bool OpCompiler::CheckCompileCondition(const ge::NodePtr &node) const {
  auto op_desc_ptr = node->GetOpDesc();
  bool is_virtual_op = false;
  (void)ge::AttrUtils::GetBool(op_desc_ptr, ge::ATTR_NAME_NOTASK, is_virtual_op);
  if (is_virtual_op) {
    FE_LOGD("Op %s is virtual, type = %s; no compilation needed.", node->GetName().c_str(), node->GetType().c_str());
    return false;
  }

  bool need_single_op_compile = false;
  (void)ge::AttrUtils::GetBool(op_desc_ptr, ge::ATTR_NEED_COMPILE, need_single_op_compile);
  if (need_single_op_compile) {
    FE_LOGD("Op %s requires single-op compilation, type=%s, no compilation needed.", node->GetName().c_str(),
            node->GetType().c_str());
    return false;
  }

  if (IsThread1Node(node)) {
    FE_LOGD("Node [%s, %s] has been sliced and is not the first slice; hence, there is no need for compilation.",
            node->GetName().c_str(), node->GetType().c_str());
    return false;
  }

  if (ge::AttrUtils::HasAttr(op_desc_ptr, kOpShapeOrRangeUnsupport)) {
    FE_LOGD("op [name: %s] has an unknown rank shape, no compilation needed.", node->GetName().c_str());
    return false;
  }
  return true;
}

Status OpCompiler::GetScopeNodeMap(const ge::ComputeGraph& graph,
                                   const std::vector<ge::NodePtr>& scope_nodes,
                                   ScopeNodeIdMap& fusion_node_map) {
  vector<string> vec = {
      OP_TYPE_PLACE_HOLDER, OP_TYPE_END, CONSTANTOP, fe::CONSTANT, SWAPCO, BNHOST, RESHAPE, REFORMAT,
      SQUEEZE_V2, UNSQUEEZE_V2, COMPRESSOP, COMPRESSFCOP};
  std::map<int64_t, bool> fusion_scope_type_map;
  for (auto &node : scope_nodes) {
    FE_CHECK(node == nullptr, REPORT_FE_ERROR("[SubGraphOpt][Compile][GetScopeNdMap] Node is nullptr."), return FAILED);

    string session_graph_id;
    if (ge::AttrUtils::GetStr(graph, ge::ATTR_NAME_SESSION_GRAPH_ID, session_graph_id) && !session_graph_id.empty()) {
      FE_LOGD("SessionGraphId=%s, node is %s.", session_graph_id.c_str(), node->GetName().c_str());
      (void)ge::AttrUtils::SetStr(node->GetOpDesc(), ge::ATTR_NAME_SESSION_GRAPH_ID, session_graph_id);
    }
    string op_type = node->GetType();
    vector<string>::iterator ret = std::find(vec.begin(), vec.end(), op_type);
    if (ret != vec.end()) {
      continue;
    }

    auto op_desc_ptr = node->GetOpDesc();
    FE_CHECK(op_desc_ptr == nullptr, REPORT_FE_ERROR("[SubGraphOpt][Compile][GetScopeNdMap] opDescPtr is nullptr."),
             return FAILED);

    if (!CheckCompileCondition(node)) {
      continue;
    }

    // get op kernel info store name
    Status status = AddNormalTbeNodeIntoMap(node, op_desc_ptr, fusion_node_map, fusion_scope_type_map);
    if (status != SUCCESS) {
      return status;
    }
  }
  return SUCCESS;
}

bool OpCompiler::IsNeedPreCompile(ge::NodePtr &node, ge::OpDescPtr &op_desc_ptr, bool need_precompile_graph) const {
  if (!IsValidOp(node)) {
    return false;
  }
  const string &op_type = node->GetType();
  if (need_precompile_graph) {
    bool need_precompile_node = false;
    (void)ge::AttrUtils::GetBool(op_desc_ptr, NEED_RE_PRECOMPILE, need_precompile_node);
    if (!need_precompile_node) {
      return false;
    }
  }
  bool not_need_compile = (op_type == SWAPCO || op_type == BNHOST);
  if (not_need_compile) {
    (void)ge::AttrUtils::SetBool(op_desc_ptr, ge::ATTR_NEED_COMPILE, true);
    FE_LOGD("We skip the pre-compile of op %s.", node->GetName().c_str());
    return false;
  }

  if (CheckFallbackAclnn(op_desc_ptr)) {
    FE_LOGD("Op[%s, %s] requires a fallback to aclnn. Precompilation is not necessary.", node->GetName().c_str(),
            node->GetType().c_str());
    op_desc_ptr->DelAttr(ATTR_NAME_FALLBACK_ACLNN);
    (void)ge::AttrUtils::SetBool(op_desc_ptr, ge::ATTR_NAME_NOTASK, true);
    return false;
  }

  if (!CheckCompileCondition(node)) {
    FE_LOGD("Op[name:%s] failed to get CheckScopeNodeMap.", node->GetName().c_str());
    return false;
  }
  return true;
}

void OpCompiler::GetRePreCompileSwitch(ge::ComputeGraph &graph, string &session_graph_id,
                                       bool &need_precompile_graph) const {
  if (!ge::AttrUtils::GetStr(graph, ge::ATTR_NAME_SESSION_GRAPH_ID, session_graph_id)) {
    FE_LOGW("Attribute session_graph_id not found in graph.");
  }

  if (!ge::AttrUtils::GetBool(graph, NEED_RE_PRECOMPILE, need_precompile_graph)) {
    FE_LOGD("Attribute need_re_precompile is not found from graph.");
  }
  FE_LOGD("The need_re_precompile flag is %d.", need_precompile_graph);
}

Status OpCompiler::SetPreCompParameter(
    const ge::NodePtr &node, const FEOpsStoreInfo &op_store_info, const string &session_graph_id, OpImplType imply_type,
    std::unordered_map<OpStoreAdapterPtr, vector<PreCompileNodePara>> &pre_comp_map) {
  const std::string &imply_type_str = op_store_info.fe_ops_store_name;
  OpStoreAdapterPtr op_store_adapter = nullptr;
  Status status = OpStoreAdapterManager::Instance(engine_name_).GetOpStoreAdapter(imply_type, op_store_adapter);
  if (status != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][Compile][SetPreCompPara] Failed to get op store adapter, imply_type [%ld].",
                    imply_type);
    return FAILED;
  }
  OpKernelInfoPtr op_kernel_info_ptr = nullptr;
  std::string op_dsl_file_path = "";
  if (!ge::OpTypeUtils::IsAutofuseNode(node->GetOpDesc())) {
    op_kernel_info_ptr = OpsKernelManager::Instance(node->GetOpDesc()->GetOpEngineName()).
                                           GetOpKernelInfoByOpType(imply_type_str, node->GetType());
    if (op_kernel_info_ptr == nullptr) {
      REPORT_FE_ERROR(
          "[SubGraphOpt][Compile][SetPreCompPara] Node[%s]: failed to obtain op kernel information. The imply_type is %s.",
          node->GetType().c_str(), imply_type_str.c_str());
      return FAILED;
    }
    if (SetMemoryTypeForOutput(node, op_kernel_info_ptr) != SUCCESS) {
      return FAILED;
    }

    bool is_custom_op = false;
    (void)ge::AttrUtils::GetBool(node->GetOpDesc(), NON_PERSISTENT_CUSTOM_OP_FLAG, is_custom_op);

    bool ret_status = (is_custom_op && op_kernel_info_ptr != nullptr && !op_kernel_info_ptr->GetOpImpPath().empty());
    if (ret_status) {
      op_dsl_file_path = op_kernel_info_ptr->GetOpImpPath();
    } else {
      op_dsl_file_path = op_store_info.op_impl_file_path;
    }
  } else {
    FE_MAKE_SHARED(op_kernel_info_ptr = std::make_shared<OpKernelInfo>("AIcoreEngine", OpImplType::EN_IMPL_HW_TBE),
                  return FAILED);
    FE_CHECK_NOTNULL(op_kernel_info_ptr);
    op_dsl_file_path = "autofuse";
  }

  PreCompileNodePara pre_comp_node_para = {node.get(), op_kernel_info_ptr, imply_type_str, op_dsl_file_path,
                                           session_graph_id, nullptr};
  if (pre_comp_map.find(op_store_adapter) == pre_comp_map.end()) {
    vector<PreCompileNodePara> pre_comp_node_para_vec;
    pre_comp_node_para_vec.push_back(pre_comp_node_para);
    pre_comp_map.emplace(make_pair(op_store_adapter, pre_comp_node_para_vec));
  } else {
    pre_comp_map[op_store_adapter].push_back(pre_comp_node_para);
  }
  return SUCCESS;
}

bool OpCompiler::UpdateOpAttrByCannKbResult(const std::string &kb_result_str, const ge::OpDescPtr &op_desc) const {
  try {
    FE_LOGD("Try to parse kb result[%s] to json.", kb_result_str.c_str());
    nlohmann::json params_json = nlohmann::json::parse(kb_result_str);
    auto iter = params_json.find(kCannKbDynamicCompileStatic);
    if (iter != params_json.end()) {
      std::string dynamic_compile_static_str = iter.value();
      FE_LOGD("Dynamic compile static parameter from CANN KB repository: %s.", dynamic_compile_static_str.c_str());
      StringUtils::Trim(dynamic_compile_static_str);
      if (!dynamic_compile_static_str.empty()) {
        StringUtils::ToLowerString(dynamic_compile_static_str);
        bool dynamic_compile_static_flag = dynamic_compile_static_str == kStrTrue;
        (void)ge::AttrUtils::SetBool(op_desc, ATTR_NAME_IS_OP_DYNAMIC_IMPL, dynamic_compile_static_flag);
        FE_LOGD("Attribute [%s] of operator [%s, %s] has been set to [%d] due to repository.", ATTR_NAME_IS_OP_DYNAMIC_IMPL.c_str(),
                op_desc->GetName().c_str(), op_desc->GetType().c_str(), dynamic_compile_static_flag);
      }
    }
    iter = params_json.find(kCannKbOpImplSwitch);
    if (iter != params_json.end()) {
      std::string op_impl_str = iter.value();
      FE_LOGD("Op impl switch param from CANN KB repository is [%s].", op_impl_str.c_str());
      if (!op_impl_str.empty()) {
        (void)ge::AttrUtils::SetStr(op_desc, kAttrOpImplSwitchValue, op_impl_str);
        FE_LOGD("Attr[%s] of op[%s, %s] has been set to [%s] due to repository.", kAttrOpImplSwitchValue.c_str(),
                op_desc->GetName().c_str(), op_desc->GetType().c_str(), op_impl_str.c_str());
      }
    }
  } catch (nlohmann::json::parse_error& ex) {
    FE_LOGW("[%s] is not in JSON format.", kb_result_str.c_str());
    return false;
  }
  return true;
}

Status OpCompiler::UpdateNodeCompileParams(const ge::NodePtr &node, const OpKernelInfoPtr &op_kernel_info_ptr) const {
  if (node == nullptr || op_kernel_info_ptr == nullptr) {
    return FAILED;
  }
  ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
  FE_CHECK_NOTNULL(op_desc_ptr);
  bool is_tune_stage = op_desc_ptr->HasAttr(kAttrDynamicCompileStatic) || op_desc_ptr->HasAttr(kAttrOpImplSwitch);
  if (is_tune_stage) {
    FE_LOGD("Op[%s, %s] has at least one of [%s] and [%s] attr.",
            op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str(),
            kAttrDynamicCompileStatic.c_str(), kAttrOpImplSwitch.c_str());
    // if op has one of DynamicCompileStatic and OpImplSwitch attr, tune stage
    bool attr_dynamic_compile_static = false;
    if (ge::AttrUtils::GetBool(op_desc_ptr, kAttrDynamicCompileStatic, attr_dynamic_compile_static)) {
      FE_LOGD("Attr[%s] of op[%s, %s] is [%d].", kAttrDynamicCompileStatic.c_str(),
              op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str(), attr_dynamic_compile_static);
      (void)ge::AttrUtils::SetBool(op_desc_ptr, ATTR_NAME_IS_OP_DYNAMIC_IMPL, attr_dynamic_compile_static);
    }
    std::string op_impl_switch;
    if (ge::AttrUtils::GetStr(op_desc_ptr, kAttrOpImplSwitch, op_impl_switch) && !op_impl_switch.empty()) {
      FE_LOGD("Attr[%s] of op[%s, %s] is [%s].", kAttrOpImplSwitch.c_str(),
              op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str(), op_impl_switch.c_str());
      (void)ge::AttrUtils::SetStr(op_desc_ptr, kAttrOpImplSwitchValue, op_impl_switch);
    }
  } else {
    bool enable_op_compile = Configuration::Instance(AI_CORE_NAME).IsEnableOpImplStrategy();
    if (enable_op_compile && (op_kernel_info_ptr->GetDynamicCompileStatic() == DynamicCompileStatic::TUNE ||
                              !op_kernel_info_ptr->GetOpImplSwitch().empty())) {
      OpStoreAdapterPtr op_store_adapter =
              OpStoreAdapterManager::Instance(engine_name_).GetOpStoreAdapter(op_kernel_info_ptr->GetOpStoreImplType());
      FE_CHECK_NOTNULL(op_store_adapter);
      std::vector<std::string> op_unique_keys;
      (void)op_store_adapter->GetOpUniqueKeys(node, op_kernel_info_ptr, op_unique_keys);
      FE_LOGD("The Op_unique_key size of op [%s, %s] is [%zu].",
              node->GetName().c_str(), node->GetType().c_str(), op_unique_keys.size());
      std::map<std::string, std::string> config_map;
      config_map.emplace(EM_OP_TYPE, kCannKbImplType);
      std::string kb_result_str;
      if (!OpCompilerFormatTune::GetTuneKnowledgeResult(node, op_unique_keys, config_map, kb_result_str)) {
        REPORT_FE_ERROR("Failed to retrieve tuning knowledge for op [%s, %s].",
                        node->GetName().c_str(), node->GetType().c_str());
        return FAILED;
      }
      if (kb_result_str.empty()) {
        FE_LOGD("The CANN KB result is empty for node [%s, %s].",
                node->GetName().c_str(), node->GetType().c_str());
        return SUCCESS;
      }
      if (!UpdateOpAttrByCannKbResult(kb_result_str, op_desc_ptr)) {
        REPORT_FE_ERROR("Failed to parse kb result string for op [%s, %s].",
                        node->GetName().c_str(), node->GetType().c_str());
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

void OpCompiler::MarkDuplicatedNode(const ge::ComputeGraph &graph) const {
  std::map<string, ge::NodePtr> thread1_nodes;
  std::map<ge::NodePtr, std::shared_ptr<std::vector<ge::NodePtr>>> related_threads_nodes;
  auto direct_nodes = graph.GetDirectNode();
  if (CountThread1Node(direct_nodes, thread1_nodes, related_threads_nodes) != SUCCESS) {
    return;
  }

  CountOtherThreadsNode(direct_nodes, thread1_nodes, related_threads_nodes);

  for (const auto &thread1_iter : thread1_nodes) {
    auto thread1_node = thread1_iter.second;
    FE_LOGD("Setting other thread nodes for thread 1 node %s.",
            thread1_node->GetName().c_str());
    auto other_thread_nodes_iter = related_threads_nodes.find(thread1_node);
    if (other_thread_nodes_iter != related_threads_nodes.end()) {
      thread1_node->GetOpDesc()->SetExtAttr(kAttrRelatedThreadsNodes, other_thread_nodes_iter->second);
    }
  }
}

Status OpCompiler::UpdateCompileParams(const ge::ComputeGraph &graph) const {
  for (const ge::NodePtr &node : graph.GetDirectNode()) {
    FE_CHECK_NOTNULL(node);
    ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
    FE_CHECK_NOTNULL(op_desc_ptr);
    // only deal with static shape op
    if (UnknownShapeUtils::IsUnknownShapeOp(*op_desc_ptr)) {
      continue;
    }
    int64_t fe_imply_type = 0;
    if (!ge::AttrUtils::GetInt(op_desc_ptr, FE_IMPLY_TYPE, fe_imply_type)) {
      continue;
    }
    FE_LOGD("Attr fe imply type of op[%s, %s] is %ld.",
            op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str(), fe_imply_type);
    OpImplType imply_type = static_cast<OpImplType>(fe_imply_type);
    OpKernelInfoPtr op_kernel_info_ptr =
            OpsKernelManager::Instance(engine_name_).GetOpKernelInfoByOpType(imply_type, op_desc_ptr->GetType());
    if (op_kernel_info_ptr == nullptr) {
      FE_LOGD("Op Kernel Info of op[%s, %s] is nullptr",
              op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str());
      continue;
    }
    if (UpdateNodeCompileParams(node, op_kernel_info_ptr) != SUCCESS) {
      return FAILED;
    }
  }
  /* Mark the inner threads of ffts+ plus slices. Those inner threads does not
     need to compile. */
  MarkDuplicatedNode(graph);
  return SUCCESS;
}

/*
 *  @ingroup fe
 *  @brief   precompile tbe op
 *  @param   [in|out] graph  compute graph
 *  @return  SUCCESS or FAILED
 */
Status OpCompiler::PreCompileOp(ge::ComputeGraph &graph) {
  string session_graph_id;

  bool need_precompile_graph = false;
  GetRePreCompileSwitch(graph, session_graph_id, need_precompile_graph);
  std::unordered_map<OpStoreAdapterPtr, vector<PreCompileNodePara>> pre_comp_map;
  for (auto &node : graph.GetDirectNode()) {
    FE_CHECK(node == nullptr, REPORT_FE_ERROR("[SubGraphOpt][Compile][PreCompOp] Node is nullptr."), return FAILED);
    string op_type = node->GetType();
    ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
    FE_CHECK(op_desc_ptr == nullptr, REPORT_FE_ERROR("[SubGraphOpt][Compile][PreCompOp] opDescPtr is nullptr."),
             return FAILED);

    // LXfusion after slice, not precompile COMPIED_FUSION_OP
    bool no_need_compile = false;
    (void)ge::AttrUtils::GetBool(op_desc_ptr, ATTR_NAME_IS_COMPIED_FUSION_OP, no_need_compile);
    if (no_need_compile) {
      FE_LOGD("Op[name:%s, type:%s] does not require optimization of the fused graph.", node->GetName().c_str(), node->GetType().c_str());
      continue;
    }

    if (!IsNeedPreCompile(node, op_desc_ptr, need_precompile_graph)) {
      FE_LOGD("Op[name:%s, type:%s] not need precompile.", op_desc_ptr->GetName().c_str(), op_type.c_str());
      continue;
    }

    // get op imply type
    int64_t tmp_imply_type = 0;
    if (!ge::AttrUtils::GetInt(op_desc_ptr, FE_IMPLY_TYPE, tmp_imply_type)) {
      REPORT_FE_ERROR("[SubGraphOpt][Compile][PreCompOp] Failed to obtain imply type for op [%s], optype [%s], imply_type [%ld].",
                      op_desc_ptr->GetName().c_str(), op_type.c_str(), tmp_imply_type);
      return OP_COMPILER_CHECK_FALSE_FAILED;
    }

    // get op kernel info store name
    OpImplType imply_type = static_cast<OpImplType>(tmp_imply_type);
    FEOpsStoreInfo op_store_info;
    auto engine_name = node->GetOpDesc()->GetOpEngineName();
    if (Configuration::Instance(engine_name).
            GetOpStoreInfoByImplType(imply_type, op_store_info) != SUCCESS) {
      FE_LOGW("Engine[%s] failed to retrieve op store info for impl_type[%ld].", engine_name.c_str(), imply_type);
      return SUCCESS;
    }

    if (op_store_info.need_pre_compile) {
      // Prepare precompile parameter
      if (SetPreCompParameter(node, op_store_info, session_graph_id, imply_type, pre_comp_map) != SUCCESS) {
        return FAILED;
      }
    }
  }

  // Do Precompile
  for (auto &comp_para : pre_comp_map) {
    OpStoreAdapterPtr op_store_adapter = comp_para.first;
    if (op_store_adapter->PreCompileOp(comp_para.second) != SUCCESS) {
      REPORT_FE_ERROR("[SubGraphOpt][Pre-Comp]Failed to pre-compile graph [%s]", graph.GetName().c_str());
      return FAILED;
    }
  }
  return SUCCESS;
}

Status OpCompiler::PreCompileThreadOpHelper(const ge::NodePtr &node, const ge::OpDescPtr &op_desc_ptr,
                                            const ge::OpDescPtr &old_op_desc, const string &session_graph_id,
                                            const ge::ComputeGraph& graph) {
   // get op imply type
  int64_t tmp_imply_type = 0;
  if (!ge::AttrUtils::GetInt(op_desc_ptr, FE_IMPLY_TYPE, tmp_imply_type)) {
    REPORT_FE_ERROR("[SubGraphOpt][Compile][PreCompOp] Failed to obtain imply type for op [%s], optype [%s], imply_type [%ld].",
                    op_desc_ptr->GetName().c_str(), node->GetType().c_str(), tmp_imply_type);
    return OP_COMPILER_CHECK_FALSE_FAILED;
  }

  // get op kernel info store name
  OpImplType imply_type = static_cast<OpImplType>(tmp_imply_type);
  FEOpsStoreInfo op_store_info;
  auto engine_name = node->GetOpDesc()->GetOpEngineName();
  if (Configuration::Instance(engine_name).GetOpStoreInfoByImplType(imply_type, op_store_info) != SUCCESS) {
    FE_LOGW("Engine[%s] failed to retrieve op store info for impl_type[%ld].", engine_name.c_str(), imply_type);
    return SUCCESS;
  }

  std::unordered_map<OpStoreAdapterPtr, vector<PreCompileNodePara>> pre_comp_map;
  if (op_store_info.need_pre_compile) {
    // Prepare precompile parameter
    FE_LOGD("Op[name:%s, type:%s] start to set pre compile parameter.", op_desc_ptr->GetName().c_str(),
            node->GetType().c_str());
    if (SetPreCompParameter(node, op_store_info, session_graph_id, imply_type, pre_comp_map) != SUCCESS) {
      return FAILED;
    }
  }

  // Do Precompile
  for (auto &comp_para : pre_comp_map) {
    OpStoreAdapterPtr op_store_adapter = comp_para.first;
    if (op_store_adapter->PreCompileOp(comp_para.second) != SUCCESS) {
      REPORT_FE_ERROR("[SubGraphOpt][Pre-Comp]Failed to pre-compile graph [%s]", graph.GetName().c_str());
      return FAILED;
    }

    // set op_pattern
    for (auto &node_para : comp_para.second) {
      ge::Node *node_para_node = node_para.node;
      string op_pattern;
      ge::AttrUtils::GetStr(node_para_node->GetOpDesc(), kOpPattern, op_pattern);
      ge::AttrUtils::SetStr(old_op_desc, node_para_node->GetName() + kOpPattern, op_pattern);
      ge::AttrUtils::SetStr(old_op_desc, kOpPattern, op_pattern);
    }
  }
  return SUCCESS;
}

/*
 *  @ingroup fe
 *  @brief   precompile tbe thread op
 *  @param   [in|out] graph  compute graph
 *  @return  SUCCESS or FAILED
 */
Status OpCompiler::PreCompileThreadOp(ge::ComputeGraph &graph, bool &sgt_flag) {
  string session_graph_id;
  for (auto &node : graph.GetDirectNode()) {
    FE_CHECK(node == nullptr, REPORT_FE_ERROR("[SubGraphOpt][Compile][PreCompOp] Node is nullptr."), return FAILED);
    string op_type = node->GetType();
    ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
    FE_CHECK(op_desc_ptr == nullptr, REPORT_FE_ERROR("[SubGraphOpt][Compile][PreCompOp] opDescPtr is nullptr."),
             return FAILED);

    if (!IsNeedPreCompile(node, op_desc_ptr, false)) {
      FE_LOGD("Op[name:%s, type:%s] not need precompile.", op_desc_ptr->GetName().c_str(), op_type.c_str());
      continue;
    }
    uint32_t thread_scope_id = 0;
    uint32_t thread_mode = 1;
    ge::AttrUtils::GetInt(op_desc_ptr, kThreadScopeId, thread_scope_id);
    ge::AttrUtils::GetInt(op_desc_ptr, kThreadMode, thread_mode);
    bool flag = (thread_scope_id == 0 || thread_mode == 0);
    if (flag) {
      return SUCCESS;
    }
    sgt_flag = true;
    FE_LOGD("Start pre-compile thread op, sgt_flag: %d.", sgt_flag);
    ge::OpDescPtr old_op_desc = ge::AttrUtils::CopyOpDesc(op_desc_ptr);
    FE_CHECK_NOTNULL(old_op_desc);
    string op_name = op_desc_ptr->GetName();

    ffts::ThreadSliceMapPtr slice_info_ptr = nullptr;
    slice_info_ptr = op_desc_ptr->TryGetExtAttr(ffts::kAttrSgtStructInfo, slice_info_ptr);
    FE_CHECK(slice_info_ptr == nullptr,
             REPORT_FE_ERROR("[SubGraphOpt][Compile][PreCompThreadOp] slice_info_ptr is nullptr."), return FAILED);
    FE_CHECK((slice_info_ptr->input_tensor_slice.size() != slice_info_ptr->ori_input_tensor_shape.size()),
             REPORT_FE_ERROR("[Compile][PreCompThreadOp] Slice input size does not match original size."), return FAILED);
    FE_CHECK((slice_info_ptr->output_tensor_slice.size() != slice_info_ptr->ori_output_tensor_shape.size()),
             REPORT_FE_ERROR("[Compile][PreCompThreadOp] Slice output size does not match original size."), return FAILED);
    FE_CHECK((slice_info_ptr->input_tensor_slice.size() != slice_info_ptr->output_tensor_slice.size()),
             REPORT_FE_ERROR("[Compile][PreCompThreadOp] Slice input and output sizes are not equal."), return FAILED);
    auto slice_size = slice_info_ptr->input_tensor_slice.size();
    FE_LOGD("slice_size: %zu.", slice_size);
    for (size_t i = 0; i < slice_size; i++) {
      if (i != 0 && i != (slice_size - 1)) {
        continue;
      }
      op_desc_ptr->SetName(op_name + "_thread_" + to_string(i));
      auto input_tensors = op_desc_ptr->GetAllInputsDescPtr();
      FE_CHECK((slice_info_ptr->input_tensor_slice[i].size() != input_tensors.size()),
               REPORT_FE_ERROR("[SubGraphOpt][Compile][PreCompThreadOp] Slice input size is in error."), return FAILED);
      FE_CHECK((slice_info_ptr->ori_input_tensor_shape[i].size() != input_tensors.size()),
               REPORT_FE_ERROR("[SubGraphOpt][Compile][PreCompThreadOp] Original input size for Slice op is incorrect."), return FAILED);
      for (size_t j = 0; j < input_tensors.size(); j++) {
        auto tensor = input_tensors.at(j);
        ge::GeShape slice_shape;
        vector<int64_t> dims_shape;
        for (auto &range : slice_info_ptr->input_tensor_slice[i][j]) {
          dims_shape.emplace_back(range.higher - range.lower);
        }
        slice_shape = ge::GeShape(dims_shape);
        tensor->SetShape(slice_shape);

        ge::GeShape slice_origin_shape = ge::GeShape(slice_info_ptr->ori_input_tensor_shape[i][j]);
        tensor->SetOriginShape(slice_origin_shape);
        FE_LOGD("Node[%s, %s]: set thread %zu's configuration for input[%zu, %s]. The slice shape is %s. The original slice shape is %s.",
                node->GetType().c_str(), node->GetName().c_str(), i, j, tensor->GetName().c_str(),
                StringUtils::IntegerVecToString(slice_shape.GetDims()).c_str(),
                StringUtils::IntegerVecToString(slice_origin_shape.GetDims()).c_str());
      }
      auto output_tensors = op_desc_ptr->GetAllOutputsDescPtr();
      FE_CHECK((slice_info_ptr->output_tensor_slice[i].size() != output_tensors.size()),
               REPORT_FE_ERROR("[SubGraphOpt][Compile][PreCompThreadOp] Slice output size error."), return FAILED);
      FE_CHECK((slice_info_ptr->ori_output_tensor_shape[i].size() != output_tensors.size()),
               REPORT_FE_ERROR("[SubGraphOpt][Compile][PreCompThreadOp] Original slice output size is erroneous."), return FAILED);
      for (size_t j = 0; j < output_tensors.size(); j++) {
        auto tensor = output_tensors.at(j);
        ge::GeShape slice_shape;
        vector<int64_t> dims_shape;
        vector<vector<int64_t>> slice_range_vec;
        for (auto &range : slice_info_ptr->output_tensor_slice[i][j]) {
          dims_shape.emplace_back(range.higher - range.lower);
          vector<int64_t> slice_range;
          slice_range.emplace_back(range.lower);
          slice_range.emplace_back(range.higher - 1);
          slice_range_vec.emplace_back(slice_range);
        }
        slice_shape = ge::GeShape(dims_shape);
        tensor->SetShape(slice_shape);
        ge::AttrUtils::SetListListInt(tensor, ge::ATTR_NAME_DATA_SLICE, slice_range_vec);

        ge::GeShape slice_origin_shape;
        vector<int64_t> dims_origin_shape;
        for (auto &shape : slice_info_ptr->ori_output_tensor_shape[i][j]) {
          dims_origin_shape.emplace_back(shape);
        }
        slice_origin_shape = ge::GeShape(dims_origin_shape);
        tensor->SetOriginShape(slice_origin_shape);
        FE_LOGD("Node[%s, %s]: set thread %zu for output[%zu, %s]. Slice shape is %s. Slice original shape is %s.",
                node->GetType().c_str(), node->GetName().c_str(), i, j, tensor->GetName().c_str(),
                StringUtils::IntegerVecToString(slice_shape.GetDims()).c_str(),
                StringUtils::IntegerVecToString(slice_origin_shape.GetDims()).c_str());
      }

      ge::graphStatus ret = ge::OpDescUtilsEx::InferDataSlice(op_desc_ptr);
      vector<int64_t> input_size_;
      ge::AttrUtils::GetListInt(op_desc_ptr, "input_size", input_size_);
      FE_LOGD("After inferDataSlice, ret: %u, op[name:%s, type:%s] input_size_: %s.", ret,
              op_desc_ptr->GetName().c_str(), op_type.c_str(), StringUtils::IntegerVecToString(input_size_).c_str());

      Status status = PreCompileThreadOpHelper(node, op_desc_ptr, old_op_desc, session_graph_id, graph);
      if (status != SUCCESS) {
        return status;
      }
    }
    node->UpdateOpDesc(old_op_desc);
  }

  return SUCCESS;
}

Status OpCompiler::SetMemoryTypeForOutput(const ge::NodePtr &node, const OpKernelInfoPtr &op_kernel_info_ptr) const {
  auto op_desc = node->GetOpDesc();
  auto op_type = op_desc->GetType();
  auto op_name = op_desc->GetName();

  IndexNameMap output_index_map;
  Status res = GetOutputNameMap(*(op_desc.get()), op_kernel_info_ptr, output_index_map);
  if (res != SUCCESS) {
    return res;
  }

  size_t out_data_anchors_size = node->GetAllOutDataAnchors().size();
  for (size_t i = 0; i < out_data_anchors_size; ++i) {
    auto out_anchor = node->GetOutDataAnchor(i);
    FE_CHECK_NOTNULL(out_anchor);
    size_t peer_in_data_nodes = out_anchor->GetPeerInDataNodesSize();
    if (peer_in_data_nodes != 0) {
      continue;
    }

    std::map<uint32_t, std::string>::const_iterator tensor_iter = output_index_map.find(i);
    if (tensor_iter == output_index_map.end()) {
      FE_LOGW("Node[type=%s, name=%s]: The output %zu is not found in the ops store.", op_type.c_str(), op_name.c_str(),
              i);
      continue;
    }

    InputOrOutputInfoPtr out_info = nullptr;
    res = op_kernel_info_ptr->GetTensorInfoByName(false, tensor_iter->second, out_info);
    if (res != SUCCESS) {
      FE_LOGW("Node[type=%s,name=%s]: the output name %s is not found in the ops store.", op_type.c_str(),
              op_name.c_str(), tensor_iter->second.c_str());
      continue;
    }
    OpParamType param_type = out_info->GetParamType();
    if (param_type == OpParamType::OPTIONAL) {
      auto output_desc_ptr = op_desc->MutableOutputDesc(i);
      if (output_desc_ptr == nullptr) {
        continue;
      }
      (void)ge::AttrUtils::SetInt(output_desc_ptr, ge::ATTR_NAME_MEMORY_SIZE_CALC_TYPE,
                                  static_cast<int64_t>(ge::MemorySizeCalcType::ALWAYS_EMPTY));
      FE_LOGI("Node[type=%s,name=%s]: success to set the attribute %s to %d for the output %s.", op_type.c_str(),
              op_name.c_str(), ge::ATTR_NAME_MEMORY_SIZE_CALC_TYPE.c_str(),
              static_cast<int32_t>(ge::MemorySizeCalcType::ALWAYS_EMPTY),
              out_info->GetName().c_str());
    }
  }
  return SUCCESS;
}

Status OpCompiler::ParseTvmJsonToSetAttr(const ge::NodePtr& node, const ge::OpDescPtr op_desc_ptr,
                                         const CompileResultInfo &compile_result) const {
  // package tvm json info
  TbeJsonFileParsePtr parse_ptr = nullptr;
  FE_MAKE_SHARED(parse_ptr = std::make_shared<TbeJsonFileParse>(*node), return fe::OP_COMPILER_MAKE_SHARED_FAILED);
  FE_CHECK(parse_ptr == nullptr, FE_LOGE("parsePtr is nullptr."), return FAILED);
  if (parse_ptr->PackageTvmJsonInfo(compile_result) != SUCCESS) {
    FE_LOGE("PackageTvmJsonInfo failed, op[%s, optype [%s]].", op_desc_ptr->GetName().c_str(),
            op_desc_ptr->GetType().c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status OpCompiler::ParseJsonAndUpdateOp(const ge::NodePtr &node, const ge::OpDescPtr op_desc_ptr,
                                        const std::vector<CompileResultInfo> &compile_results) const {
  if (FEContextUtils::IsOpTuneMode()) {
    std::string json_file_path_tmp;
    if (!ge::AttrUtils::GetStr(node->GetOpDesc(), OP_JSON_FILE_PATH, json_file_path_tmp) ||
        json_file_path_tmp.empty()) {
      ge::AttrUtils::SetStr(node->GetOpDesc(), OP_JSON_FILE_PATH, compile_results[0].json_file_path);
    }
  }
  ffts::ThreadSliceMapPtr slice_info_ptr = nullptr;
  slice_info_ptr = node->GetOpDesc()->TryGetExtAttr(ffts::kAttrSgtStructInfo, slice_info_ptr);
  if ((slice_info_ptr == nullptr) || ((slice_info_ptr->thread_mode == kManualMode) &&
      (compile_results.size() == 1)) || !OpIsAutoThread(slice_info_ptr)) {
    if (ParseTvmJsonToSetAttr(node, op_desc_ptr, compile_results.at(0)) == FAILED) {
      return FAILED;
    }

    if (InsertCompressOp(node) != SUCCESS) {
      REPORT_FE_ERROR("[SubGraphOpt][Compile][ParseJsUpdOp] Failed to insert compress op for node[%s].",
                      node->GetName().c_str());
      return FAILED;
    }

    if (SetCompressWeightAttr(node) != SUCCESS) {
      FE_LOGW("Failed to set compress weight attribute on node[%s].", node->GetName().c_str());
    }
  }
  return SUCCESS;
}

Status OpCompiler::ParseJsonAndCompressOp(const ge::ComputeGraph& graph, const CompileResultMap& compile_ret_map,
                                          const std::vector<ge::NodePtr>& nodes_be_compiled) const {
  (void)graph;
  map<int64_t, bool> is_json_parsed;
  bool is_after_merge_step = FEContextUtils::GetBuildStep() == ge::BUILD_STEP_AFTER_MERGE;
  // get info from json file and set op descriptor
  auto node_size = nodes_be_compiled.size();
  for (size_t index = 0; index < node_size; ++index) {
    ge::NodePtr node = nodes_be_compiled.at(index);
    FE_CHECK(node == nullptr,
             REPORT_FE_ERROR("[SubGraphOpt][Compile][ParseJsCompsOp] node(index %zu) is nullptr.", index),
             return FAILED);

    auto op_desc_ptr = node->GetOpDesc();
    FE_CHECK(op_desc_ptr == nullptr, REPORT_FE_ERROR("[SubGraphOpt][Compile][ParseJsCompsOp] opDescPtr is nullptr."),
             return FAILED);

    int64_t scope_id = 0;
    if (!GetFusionScopeAttr(op_desc_ptr, scope_id)) {
      continue;
    }

    // find the json file path according to scope id
    CompileResultMap::const_iterator ret_iter = compile_ret_map.find(scope_id);
    if (ret_iter == compile_ret_map.end() || ret_iter->second.empty()) {
      bool need_precompile_node = false;
      bool buff_fusion_status =
          ((ge::AttrUtils::GetBool(op_desc_ptr, NEED_RE_PRECOMPILE, need_precompile_node) && need_precompile_node));
      if (buff_fusion_status) {
        FE_LOGD("scopeId [%ld], op[%s, optype [%s]].", scope_id, op_desc_ptr->GetName().c_str(),
                op_desc_ptr->GetType().c_str());
        continue;
      } else if (is_after_merge_step) {
        FE_LOGW("Node [%s, %s] has scope ID [%ld], but the compile result was not found after the merge step.",
                op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str(), scope_id);
        continue;
      } else {
        REPORT_FE_ERROR("[SubGraphOpt][Compile][ParseJson] scopeId [%ld], op[%s, %s], compile result not found.",
                        scope_id, op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str());
        return OP_COMPILER_CHECK_FALSE_FAILED;
      }
    }

    if (is_json_parsed.count(scope_id) && scope_id >= 0) {
      continue;
    }

    Status ret = ParseJsonAndUpdateOp(node, op_desc_ptr, ret_iter->second);
    if (ret != SUCCESS) {
      return ret;
    }

    is_json_parsed[scope_id] = true;
  }
  return SUCCESS;
}

bool OpCompiler::StopCompileOpInTuningAndAfterBuilderMode() const {
  std::string build_mode_value = FEContextUtils::GetBuildMode();
  std::string step_mode_value = FEContextUtils::GetBuildStep();
  bool no_need_to_wait_task_finish =
      (build_mode_value == ge::BUILD_MODE_TUNING &&
       (step_mode_value == ge::BUILD_STEP_AFTER_BUILDER || step_mode_value == ge::BUILD_STEP_AFTER_BUILDER_SUB ||
        step_mode_value == ge::BUILD_STEP_BEFORE_UB_MATCH));
  if (no_need_to_wait_task_finish) {
    FE_LOGD("No need to wait for the task to finish if build_mode is [%s] and step is [%s].", build_mode_value.c_str(),
            step_mode_value.c_str());
    return true;
  }
  return false;
}

Status OpCompiler::CompileOpOnly(CompileInfoParam &compile_info) const {
  if (compile_info.fusion_nodes_map.empty()) {
    FE_LOGI("No node in graph need to compile.");
    return SUCCESS;
  }

  OpStoreAdapterPtr op_store_adapter = nullptr;
  Status status = OpStoreAdapterManager::Instance(engine_name_).GetOpStoreAdapter(EN_IMPL_HW_TBE, op_store_adapter);

  if (status != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][Compile][CompOpOnly] Failed to get op store adapter by impl_type[%ld].",
                    EN_IMPL_HW_TBE);
    return OP_COMPILER_CHECK_FALSE_FAILED;
  }

  // get scope id and json file path map
  Status ret = op_store_adapter->CompileOp(compile_info);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][Compile][CompOpOnly] CompileOp failed.");
    return ret;
  }
  return SUCCESS;
}

Status OpCompiler::GetFusionScope(const ge::ComputeGraph& graph,
                                  const std::vector<ge::NodePtr>& buff_fus_rollback_nodes,
                                  ScopeNodeIdMap &fusion_nodes_map, std::vector<ge::NodePtr> &nodes_be_compiled) {
  std::string graph_name = graph.GetName();
  auto all_nodes = graph.GetDirectNode();
  for (auto &node : all_nodes) {
    (void)ge::AttrUtils::SetStr(node->GetOpDesc(), "_graph_name", graph_name);
  }

  if (buff_fus_rollback_nodes.empty()) {
    if (GetScopeNodeMap(graph, fusion_nodes_map) != SUCCESS) {
      REPORT_FE_ERROR("[SubGraphOpt][Compile][GetFusScope] GetScopeNodeMap failed, graph [%s].",
                      graph.GetName().c_str());
      return FAILED;
    }
    for (auto &node : all_nodes) {
      nodes_be_compiled.emplace_back(node);
    }
  } else {
    if (GetScopeNodeMap(graph, buff_fus_rollback_nodes, fusion_nodes_map) != SUCCESS) {
      REPORT_FE_ERROR("[SubGraphOpt][Compile][GetFusScope] GetRollbackScopeNodeMap failed, graph [%s].",
                      graph.GetName().c_str());
      return FAILED;
    }
    nodes_be_compiled = buff_fus_rollback_nodes;
  }
  return SUCCESS;
}

bool OpCompiler::CheckCompiledFusionGraph(const ge::ComputeGraph& graph) const {
  auto all_nodes = graph.GetDirectNode();
  for (const auto &node : all_nodes) {
    bool no_need_compile = false;
    (void)ge::AttrUtils::GetBool(node->GetOpDesc(), ATTR_NAME_IS_COMPIED_FUSION_OP, no_need_compile);
    if (no_need_compile) {
      FE_LOGD("Found compiled fusion op with name: %s and type: %s.", node->GetName().c_str(), node->GetType().c_str());
      return true;
    }
  }
  return false;
}

Status OpCompiler::GetMixComFusionScope(const ge::ComputeGraph& graph,
                                        const std::vector<ge::NodePtr>& buff_fus_rollback_nodes,
                                        ScopeNodeIdMap &fusion_nodes_map, std::vector<ge::NodePtr> &nodes_be_compiled) {
  (void)buff_fus_rollback_nodes;
  FE_LOGD("Graph has compiled fusion nodes; it filters the compiled nodes and calculates the scope ID when obtaining the nodes that need to be compiled.");
  std::string graph_name = graph.GetName();
  auto all_nodes = graph.GetDirectNode();
  for (const auto &node : all_nodes) {
    (void)ge::AttrUtils::SetStr(node->GetOpDesc(), "_graph_name", graph_name);
    bool no_need_compile = false;
    (void)ge::AttrUtils::GetBool(node->GetOpDesc(), ATTR_NAME_IS_COMPIED_FUSION_OP, no_need_compile);
    if (no_need_compile) {
      FE_LOGD("Op[name:%s, type:%s] does not require optimization of the fused graph.", node->GetName().c_str(), node->GetType().c_str());
      continue;
    }
    nodes_be_compiled.emplace_back(node);
  }

  FE_LOGD("The size of nodes_be_compiled is %zu.", nodes_be_compiled.size());

  if (GetScopeNodeMap(graph, nodes_be_compiled, fusion_nodes_map) != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][Compile][GetMixComFusionScope] GetScopeNodeMap failed, graph [%s].",
                    graph.GetName().c_str());
    return FAILED;
  }

  return SUCCESS;
}

Status OpCompiler::CompileOp(ge::ComputeGraph& graph) {
  std::vector<ge::NodePtr> buff_fus_rollback_nodes;
  std::vector<ge::NodePtr> buff_fus_to_del_nodes;
  std::vector<ge::NodePtr> buff_fus_compile_failed_nodes;
  return CompileOp(graph, buff_fus_compile_failed_nodes, buff_fus_rollback_nodes, buff_fus_to_del_nodes);
}

/*
 *  @ingroup fe
 *  @brief   compile tbe op
 *  @param   [in|out] graph  compute graph
 *  @return  SUCCESS or FAILED
 */
Status OpCompiler::CompileOp(ge::ComputeGraph& graph,
                             std::vector<ge::NodePtr>& buff_fus_compile_failed_nodes,
                             const std::vector<ge::NodePtr>& buff_fus_rollback_nodes,
                             const std::vector<ge::NodePtr>& buff_fus_to_del_nodes) {
  (void)buff_fus_to_del_nodes;
  CompileInfoParam compile_info(buff_fus_compile_failed_nodes);
  vector<ge::NodePtr> nodes_be_compiled;

  // LXfusion after slice, not compile COMPIED_FUSION_OP
  if (CheckCompiledFusionGraph(graph)) {
    (void)GetMixComFusionScope(graph, buff_fus_rollback_nodes, compile_info.fusion_nodes_map, nodes_be_compiled);
  } else {
    (void)GetFusionScope(graph, buff_fus_rollback_nodes, compile_info.fusion_nodes_map, nodes_be_compiled);
  }

  if (compile_info.fusion_nodes_map.empty()) {
    FE_LOGI("No node in graph need to compile.");
    return SUCCESS;
  }

  // get scope id and json file path map
  Status ret = CompileOpOnly(compile_info);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][Compile][CompOp] Failed to compile for graph[%s].", graph.GetName().c_str());
    return ret;
  }

  // rollback l1 fusion failed nodes
  ret = ReCompileL1FusionFailedNodes(graph, compile_info);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][Compile][CompOp] Failed to recompile L1 fusion failed nodes for graph [%s].",
                    graph.GetName().c_str());
    return ret;
  }

  ret = ParseJsonAndCompressOp(graph, compile_info.compile_ret_map, nodes_be_compiled);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][Compile][CompOp] ParseJsonAndCompressOp fail, graph[%s].", graph.GetName().c_str());
    return ret;
  }

  return SUCCESS;
}

Status OpCompiler::PostCompileOp(ge::ComputeGraph& graph, std::vector<ge::NodePtr>& buff_fus_compile_failed_nodes) {
  FE_CHECK_NOTNULL(lx_fusion_optimizer_ptr_);
  if (!buff_fus_compile_failed_nodes.empty()) {
    std::vector<ge::NodePtr> buff_fus_rollback_nodes;
    std::vector<ge::NodePtr> buff_fus_to_del_nodes;
    Status status = lx_fusion_optimizer_ptr_->LxFusionRecovery(graph, buff_fus_compile_failed_nodes,
                                                               buff_fus_rollback_nodes, buff_fus_to_del_nodes);
    if (status != SUCCESS) {
      REPORT_FE_ERROR("[PostCompileOp][CompileFailed] Failed to recover lxfusion for graph %s.",
                      graph.GetName().c_str());
      return status;
    }

    buff_fus_compile_failed_nodes.clear();
    status = CompileOp(graph, buff_fus_compile_failed_nodes, buff_fus_rollback_nodes, buff_fus_to_del_nodes);
    if (status != SUCCESS) {
      REPORT_FE_ERROR("[PostCompileOp][CompileFailed] Failed to compile rolled-back nodes for graph %s. Result is %u.",
                      graph.GetName().c_str(), status);
      return status;
    }
  }

  if (lx_fusion_optimizer_ptr_->LxFusionFinalize(graph) != SUCCESS) {
    REPORT_FE_ERROR("[PostCompileOp][CompileFailed] Failed to finalize lxfusion for graph: %s.",
                    graph.GetName().c_str());
    return FAILED;
  }

  return SUCCESS;
}

Status OpCompiler::SetCompressWeightAttr(ge::NodePtr node) const {
  // if the node is FC, just set the attr.
  // Because the weight of FC is always very large.
  std::string node_type = node->GetType();
  if (node_type == kFullyConnection) {
    (void)ge::AttrUtils::SetBool(node->GetOpDesc(), ATTR_NAME_FE_WEIGHT_COMPRESS, true);
    FE_LOGI("Node[%s, %s] has set the _fe_weight_compress attribute.", node->GetName().c_str(),
            node->GetType().c_str());
    return SUCCESS;
  }
  if (node_type != CONV2D && node_type != MATMULV2OP  && node_type != kConv2DTransposeD) {
    return SUCCESS;
  }

  // check whether is dynamic shape
  if (UnknownShapeUtils::IsUnknownShapeOp(*(node->GetOpDesc()))) {
    FE_LOGD("The shape of node[%s, %s] is dynamic, no need to calculate weight compress parameter.",
            node->GetName().c_str(), node->GetType().c_str());
    return SUCCESS;
  }

  FE_LOGD("SetCompressWeightAttr on node[%s, %s] begin.", node->GetName().c_str(), node->GetType().c_str());
  if (node->GetType() == MATMULV2OP) {
    ge::ConstGeTensorDescPtr weight_tensor_desc_ptr = node->GetOpDesc()->GetInputDescPtr(1);
    if (weight_tensor_desc_ptr == nullptr) {
      FE_LOGW("The weight description for node [%s] is empty.", node->GetName().c_str());
      return SUCCESS;
    }
    int64_t element_cnt = 1;
    ge::GeShape weight_origin_shape = weight_tensor_desc_ptr->GetOriginShape();
    ge::DataType weight_origin_dtype = weight_tensor_desc_ptr->GetOriginDataType();
    if (weight_origin_dtype == ge::DT_UNDEFINED) {
      FE_LOGW("The origin data type of the weight description for node [%s] is invalid.", node->GetName().c_str());
      return FAILED;
    }
    if (TensorComputeUtil::GetElementCountByMultiply(weight_origin_shape, element_cnt) != SUCCESS) {
      FE_LOGW("Failed to get element count for Node [%s].", node->GetName().c_str());
      return FAILED;
    }
    int64_t matmul_weight_size = -1;
    int32_t output_real_calc_flag = 1;

    if (TensorComputeUtil::GetTensorSizeByDataType(element_cnt, weight_origin_dtype, matmul_weight_size,
                                                   output_real_calc_flag) != SUCCESS) {
      FE_LOGW("Get tensor size failed! element_cnt[%ld], datatype[%d].", element_cnt, weight_origin_dtype);
      return FAILED;
    }
    FE_LOGI("The weight size of node [%s] is %ld bytes.", node->GetName().c_str(), matmul_weight_size);
    if (matmul_weight_size > WEIGHT_SIZE_THRESHOLD) {
      (void)ge::AttrUtils::SetBool(node->GetOpDesc(), ATTR_NAME_FE_WEIGHT_COMPRESS, true);
      FE_LOGI("Node[%s, %s] has been set _fe_weight_compress attr.", node->GetName().c_str(), node->GetType().c_str());
    }
    return SUCCESS;
  }

  // then begin to deal Conv2D
  PlatformInfo platform_info;
  OptionalInfo optional_info;
  if (PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, optional_info) != SUCCESS) {
    FE_LOGW("Failed to get platform info by current soc version.");
    return FAILED;
  }
  uint64_t cube_k = platform_info.ai_core_spec.cube_k_size;
  uint64_t cube_m = platform_info.ai_core_spec.cube_m_size;
  uint64_t cube_n = platform_info.ai_core_spec.cube_n_size;
  double cube_freq = platform_info.ai_core_spec.cube_freq;
  double ddr_band_width = cube_freq * platform_info.ai_core_memory_rates.ddr_rate;

  ge::OpDescPtr op_desc = node->GetOpDesc();

  int64_t fm_h = 0;
  int64_t fm_w = 0;
  int64_t weight_h = 0;
  int64_t weight_w = 0;
  int64_t cin = 0;
  int64_t cout = 0;
  if (GetParam(op_desc, fm_h, fm_w, weight_h, weight_w, cin, cout, cube_k) != SUCCESS) {
    return FAILED;
  }

  int64_t fm_hw = 0;
  FE_MUL_OVERFLOW(fm_h, fm_w, fm_hw)
  int64_t weight_hw_cin = 0;
  FE_MUL_OVERFLOW(weight_h, weight_w, weight_hw_cin)
  FE_MUL_OVERFLOW(weight_hw_cin, cin, weight_hw_cin)

  FE_DOUBLE_ZEROCHECK(cube_m)
  FE_DOUBLE_ZEROCHECK(cube_k)
  double tmp1 = std::ceil(static_cast<double>(fm_hw) / cube_m);
  double tmp2 = std::ceil(static_cast<double>(weight_hw_cin) / cube_k);
  FE_DOUBLE_MULCHECK(tmp1, tmp2)

  double cube_cycle = tmp1 * tmp2;
  FE_DOUBLE_ZEROCHECK(cube_n);
  double tmp3 = std::ceil(static_cast<double>(cout) / cube_n);
  FE_DOUBLE_MULCHECK(cube_cycle, tmp3);
  cube_cycle *= tmp3;

  int64_t weight_size = 0;
  FE_MUL_OVERFLOW(weight_hw_cin, cout, weight_size)

  FE_DOUBLE_ZEROCHECK(ddr_band_width);
  FE_DOUBLE_MULCHECK(cube_freq, weight_size);
  double weight_cycle = weight_size * cube_freq / ddr_band_width;

  if (weight_size > WEIGHT_SIZE_THRESHOLD) {
    FE_DOUBLE_ZEROCHECK(cube_cycle);
    double ret = weight_cycle / cube_cycle;
    FE_LOGD("The result for weight compress of [%s] is %f.", node->GetName().c_str(), ret);
    if (ret >= WEIGHT_CHECK_THRESHOLD) {
      (void)ge::AttrUtils::SetBool(op_desc, ATTR_NAME_FE_WEIGHT_COMPRESS, true);
      FE_LOGI("Node[%s, %s] has been set _fe_weight_compress attr. Ret = %f.", node->GetName().c_str(),
              node->GetType().c_str(), ret);
    }
  }
  return SUCCESS;
}

Status OpCompiler::InsertCompressOp(const ge::NodePtr &node) const {
  // 1. Only deal conv or FC node
  if (kCubeCompressOpList.count(node->GetType()) == 0) {
    return SUCCESS;
  }

  // 2. Check is there compress op before cube compress op
  if (HasInsertCompressOp(node)) {
    FE_LOGD("Compress op has already been insert before node[%s, %s].",
            node->GetName().c_str(), node->GetType().c_str());
    return SUCCESS;
  }

  WEIGHCOMPRESSINNERFLAG enable_sparsity = JudgeIsSparsityFlag();
  FE_LOGD("Begin to insert compression op for node [%s, %s], with sparsity enabled as [%d].",
          node->GetName().c_str(), node->GetType().c_str(), static_cast<int32_t>(enable_sparsity));

  ge::OpDescPtr conv_desc = node->GetOpDesc();
  FE_CHECK_NOTNULL(conv_desc);

  // 3. Get the compress_parameters from conv compress node and update compress index input's shape
  std::vector<int64_t> compress_param_vec;
  (void)ge::AttrUtils::GetListInt(conv_desc, ATTR_NAME_COMPRESS_PARAMETERS, compress_param_vec);
  FE_LOGD("The compression parameter of node [%s, %s] is [%s].", node->GetName().c_str(), node->GetType().c_str(),
          StringUtils::IntegerVecToString(compress_param_vec).c_str());

  // 4. Refresh conv2d compress index shape
  if (enable_sparsity != WEIGHCOMPRESSINNERFLAG::FOUR_TO_TWO_FLAG) {
    if (UpdateCompressIndexShape(conv_desc, compress_param_vec) != SUCCESS) {
      REPORT_FE_ERROR("[SubGraphOpt][Compile][InsrtCmprsOp] Failed to update compress index shape for node [%s].",
                      conv_desc->GetName().c_str());
      return FAILED;
    }
  }

  // 5. Create compress node
  ge::OpDescPtr compress_opdesc = CreateCompressOp(conv_desc, compress_param_vec);
  if (compress_opdesc == nullptr) {
    REPORT_FE_ERROR("[SubGraphOpt][Compile][InsrtCmprsOp] Failed to create compress op for node [%s].",
                    conv_desc->GetName().c_str());
    return FAILED;
  }

  // 6. Get the compress type from conv compress node and set attr to compress node
  if (enable_sparsity != WEIGHCOMPRESSINNERFLAG::FOUR_TO_TWO_FLAG) {
    int64_t weight_compress_type = -1;
    (void)ge::AttrUtils::GetInt(conv_desc, ATTR_NAME_WEIGHT_COMPRESS_TYPE, weight_compress_type);
    if (kWeightCompressTypes.count(static_cast<WeightCompressType>(weight_compress_type)) == 0) {
      REPORT_FE_ERROR("[SubGraphOpt][Compile][InsrtCmprsOp] Failed to retrieve the weight compression type from node [%s, %s].",
                      conv_desc->GetName().c_str(), conv_desc->GetType().c_str());
      return FAILED;
    }
    (void)ge::AttrUtils::SetInt(compress_opdesc, ATTR_NAME_WEIGHT_COMPRESS_TYPE, weight_compress_type);
    FE_LOGD("The weight compression type of node [%s, %s] is %ld.", node->GetName().c_str(), node->GetType().c_str(),
            weight_compress_type);
  }

  // 7. set attr and update the weight shape of compress op
  if (enable_sparsity == WEIGHCOMPRESSINNERFLAG::FOUR_TO_TWO_FLAG) {
    if (RefreshCompressShapeForSpasity(conv_desc, compress_opdesc) != SUCCESS) {
      REPORT_FE_ERROR("[SubGraphOpt][Compile][InsrtCmprsOp] Failed to refresh shape of compress op[%s].",
                      compress_opdesc->GetName().c_str());
      return FAILED;
    }
  }

  // 8. add compress node between conv compress and weight, then link anchor for conv and compress
  if (AddCompressNodeAndCopyWeight(node, compress_opdesc) != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][Compile][InsrtCmprsOp] Failed to insert compression node before node [%s, %s].",
                    conv_desc->GetName().c_str(), conv_desc->GetType().c_str());
    return FAILED;
  }
  FE_LOGI("The compress node[%s, %s] has been insert before node[%s, %s].",
          compress_opdesc->GetName().c_str(), compress_opdesc->GetType().c_str(),
          node->GetName().c_str(), node->GetType().c_str());
  ProcCompressOpMemType(node);
  return SUCCESS;
}

bool OpCompiler::HasInsertCompressOp(const ge::NodePtr &conv_node) const {
  ge::InDataAnchorPtr conv_weight_in_anchor = conv_node->GetInDataAnchor(TENSOR_INDEX_FILTER_COMPRESS);
  if (conv_weight_in_anchor == nullptr) {
    return false;
  }
  ge::OutDataAnchorPtr weight_peer_out_anchor = conv_weight_in_anchor->GetPeerOutAnchor();
  if (weight_peer_out_anchor == nullptr) {
    return false;
  }

  ge::NodePtr peer_node = weight_peer_out_anchor->GetOwnerNode();
  if (peer_node == nullptr) {
    return false;
  }
  FE_LOGD("The peer node of node[%s, %s]'s filter input is [%s, %s].", conv_node->GetName().c_str(),
          conv_node->GetType().c_str(), peer_node->GetName().c_str(), peer_node->GetType().c_str());
  return peer_node->GetType() == COMPRESSOP || peer_node->GetType() == COMPRESSFCOP;
}

Status OpCompiler::UpdateCompressIndexShape(const ge::OpDescPtr &conv_desc,
                                            const std::vector<int64_t> &compress_param_vec) const {
  if (compress_param_vec.size() <= COMPRESS_PARAMETER_INFOSIZE_INDEX) {
    FE_LOGE("The compress params size [%zu] of the conv node [%s, %s] is less than 2.", compress_param_vec.size(),
            conv_desc->GetName().c_str(), conv_desc->GetType().c_str());
    return FAILED;
  }
  std::vector<int64_t> compress_index_desc_dims = {compress_param_vec[COMPRESS_PARAMETER_INFOSIZE_INDEX]};
  ge::GeTensorDescPtr index_tensor_desc_ptr = conv_desc->MutableInputDesc(TENSOR_INDEX_COMPRESS_INDEX);
  if (index_tensor_desc_ptr == nullptr) {
    FE_LOGE("The compress index input for the conv node [%s, %s] is null.",
            conv_desc->GetName().c_str(), conv_desc->GetType().c_str());
    return FAILED;
  }
  if (index_tensor_desc_ptr->MutableShape().GetDims() == compress_index_desc_dims) {
    FE_LOGE("The shape of the compressed index for the convolution node [%s, %s] has already been updated.",
            conv_desc->GetName().c_str(), conv_desc->GetType().c_str());
    return FAILED;
  }
  ge::GeShape index_shape(compress_index_desc_dims);
  index_tensor_desc_ptr->SetShape(index_shape);
  index_tensor_desc_ptr->SetOriginShape(index_shape);
  FE_LOGI("The shape of compress index of node[%s, %s] has been update to [%s].",
          conv_desc->GetName().c_str(), conv_desc->GetType().c_str(),
          StringUtils::IntegerVecToString(index_shape.GetDims()).c_str());
  return SUCCESS;
}

ge::OpDescPtr OpCompiler::CreateCompressOp(const ge::OpDescPtr &conv_desc,
                                           const std::vector<int64_t> &compress_param_vec) const {
  std::string compress_type = (kConvCompressOpList.count(conv_desc->GetType()) > 0) ? COMPRESSOP : COMPRESSFCOP;
  std::string compress_op_name = conv_desc->GetName() + "_" + compress_type;
  ge::OpDescPtr compress_opdesc = nullptr;
  FE_MAKE_SHARED(compress_opdesc = std::make_shared<ge::OpDesc>(compress_op_name, compress_type), return nullptr);
  FE_CHECK(compress_opdesc == nullptr,
           REPORT_FE_ERROR("[CreateCompressOp] Create op desc failed."), return nullptr);
  // add input desc
  compress_opdesc->AddInputDesc("weight", conv_desc->GetInputDesc(TENSOR_INDEX_FILTER_COMPRESS));

  // add output desc
  compress_opdesc->AddOutputDesc("weight_compress", conv_desc->GetInputDesc(TENSOR_INDEX_FILTER_COMPRESS));
  compress_opdesc->AddOutputDesc("compress_index", conv_desc->GetInputDesc(TENSOR_INDEX_COMPRESS_INDEX));

  compress_opdesc->SetOpEngineName(AI_CORE_NAME);
  compress_opdesc->SetOpKernelLibName(AI_CORE_NAME);
  // set fe impl type to TBE, otherwise there would be exception
  // while calculating the tensor size
  (void)ge::AttrUtils::SetInt(compress_opdesc, FE_IMPLY_TYPE, static_cast<int64_t>(EN_IMPL_HW_TBE));
  (void)ge::AttrUtils::SetStr(compress_opdesc, ge::ATTR_NAME_ENGINE_NAME_FOR_LX, AI_CORE_NAME);
  (void)ge::AttrUtils::SetStr(compress_opdesc, ge::ATTR_NAME_KKERNEL_LIB_NAME_FOR_LX, AI_CORE_NAME);
  (void)ge::AttrUtils::SetListInt(compress_opdesc, ATTR_NAME_COMPRESS_PARAMETERS, compress_param_vec);
  return compress_opdesc;
}

Status OpCompiler::AddCompressNodeAndCopyWeight(const ge::NodePtr &conv_node,
                                                const ge::OpDescPtr &compress_opdesc) const {
  FE_CHECK_NOTNULL(conv_node->GetOwnerComputeGraph());
  ge::NodePtr compress_node = conv_node->GetOwnerComputeGraph()->AddNode(compress_opdesc);
  FE_CHECK_NOTNULL(compress_node);
  ge::InDataAnchorPtr conv_weight_in_anchor = conv_node->GetInDataAnchor(TENSOR_INDEX_FILTER_COMPRESS);
  FE_CHECK_NOTNULL(conv_weight_in_anchor);
  ge::InDataAnchorPtr conv_index_in_anchor = conv_node->GetInDataAnchor(TENSOR_INDEX_COMPRESS_INDEX);
  FE_CHECK_NOTNULL(conv_index_in_anchor);

  ge::OutDataAnchorPtr weight_peer_out_anchor = conv_weight_in_anchor->GetPeerOutAnchor();
  FE_CHECK_NOTNULL(weight_peer_out_anchor);

  conv_weight_in_anchor->UnlinkAll();
  conv_index_in_anchor->UnlinkAll();

  (void)ge::GraphUtils::AddEdge(compress_node->GetOutDataAnchor(COMPRESSOP_INDEX_WEIGHT_COMPRESS),
                                conv_weight_in_anchor);
  (void)ge::GraphUtils::AddEdge(compress_node->GetOutDataAnchor(COMPRESSOP_INDEX_COMPRESS_INDEX),
                                conv_index_in_anchor);
  (void)ge::GraphUtils::AddEdge(weight_peer_out_anchor, compress_node->GetInDataAnchor(0));

  // copy weight whilt tuning
  if (FEContextUtils::GetBuildMode() == ge::BUILD_MODE_TUNING &&
      FEContextUtils::GetBuildStep() == ge::BUILD_STEP_BEFORE_UB_MATCH) {
    ge::NodePtr peer_node = weight_peer_out_anchor->GetOwnerNode();
    CopyWeightAttrToPlaceHolder(peer_node);
  }
  return SUCCESS;
}

void OpCompiler::GetNodesNeedRePrcmpl(const Vistor<ge::NodePtr> &all_nodes,
                                      std::unordered_set<int64_t> &need_re_compile_scope_id,
                                      std::vector<ge::NodePtr> &nodes_be_compiled,
                                      std::vector<ge::NodePtr> &all_nodes_after_lx_fusion) {
  for (auto &node : all_nodes) {
    all_nodes_after_lx_fusion.emplace_back(node);
    bool need_re_compile = false;
    string cmp_strategy;
    (void)ge::AttrUtils::GetStr(node->GetOpDesc(), ge::ATTR_NAME_OP_COMPILE_STRATEGY, cmp_strategy);
    if ((ge::AttrUtils::GetBool(node->GetOpDesc(), NEED_RE_PRECOMPILE, need_re_compile) && need_re_compile) ||
        !cmp_strategy.empty()) {
      int64_t scope_id = 0;
      bool has_scope_id = GetFusionScopeAttr(node->GetOpDesc(), scope_id);
      if (has_scope_id && scope_id >= 0) {
        need_re_compile_scope_id.emplace(scope_id);
        /* fusion nodes will be emplaced into nodes_be_compiled in the following step.
         * Here only count all possible scope-ids. */
        FE_LOGD("Node [%s] needs re-precompile and compile. We need to find all nodes within its scope ID [%ld].",
                node->GetName().c_str(), scope_id);
      } else {
        nodes_be_compiled.emplace_back(node);
        FE_LOGD("Single Node %s requires re-precompilation and compilation.", node->GetName().c_str());
      }
    }
  }
}

Status OpCompiler::GetFusionScope(ge::ComputeGraph &graph, ScopeNodeIdMap &fusion_nodes_map,
                                  std::vector<ge::NodePtr> &nodes_be_compiled,
                                  std::vector<ge::NodePtr> &all_nodes_after_lx_fusion) {
  std::string graph_name = graph.GetName();
  auto all_nodes = graph.GetDirectNode();
  /* Find the minimum scope id in this graph. */
  for (auto &node : all_nodes) {
    (void)ge::AttrUtils::SetStr(node->GetOpDesc(), "_graph_name", graph_name);
  }

  /* Get all nodes which are need re-pre-compiled and store their scope id to
   * find all nodes with that scope id. */
  std::unordered_set<int64_t> need_re_compile_scope_id;
  GetNodesNeedRePrcmpl(all_nodes, need_re_compile_scope_id, nodes_be_compiled, all_nodes_after_lx_fusion);

  /* Find all nodes using special scope id. Every node uses that scope id
   * need to be re-compiled because they will be fused as one node. */
  for (auto &node : all_nodes) {
    int64_t scope_id = 0;
    bool has_scope_id = GetFusionScopeAttr(node->GetOpDesc(), scope_id);
    if (has_scope_id && scope_id >= 0) {
      if (need_re_compile_scope_id.count(scope_id) != 0) {
        FE_LOGD("node %s with scope id %ld needs to be re-compiled.", node->GetName().c_str(), scope_id);
        nodes_be_compiled.emplace_back(node);
      }
    }
  }

  if (GetScopeNodeMap(graph, nodes_be_compiled, fusion_nodes_map) != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][Compile][GetFusScope] GetScopeNodeMap failed, graph [%s].", graph.GetName().c_str());
    return FAILED;
  }

  return SUCCESS;
}

Status OpCompiler::ReCompileL1FusionFailedNodes(const ge::ComputeGraph &graph, CompileInfoParam &compile_info) {
  if (compile_info.l1_fusion_failed_nodes.empty()) {
    return SUCCESS;
  }
  FE_LOGI("L1 fusion node compilation unsuccessful, attempting UB fusion compile for l1 fusion failed nodes.");
  compile_info.fusion_nodes_map.clear();
  if (GetScopeNodeMap(graph, compile_info.l1_fusion_failed_nodes, compile_info.fusion_nodes_map) != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][Compile][Lx] Failed to get scope map for l1 fusion failed nodes of graph[%s].",
                    graph.GetName().c_str());
    return FAILED;
  }

  if (CompileOpOnly(compile_info) != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][Compile][Lx] Failed to compile L1 fusion nodes that previously failed in graph [%s].",
                    graph.GetName().c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status OpCompiler::ReCompileOpAfterLxFusion(ge::ComputeGraph &graph, CompileInfoParam &compile_info,
                                            const LxFusionOptimizeResult &opt_ret) {
  Status ret = PreCompileOp(graph);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][Compile][Lx] PreCompileOp failed after buffer fusion.");
    return ret;
  }

  /* scope_json_map is not need to be cleared. Put two rounds of compilation result together and
   * parse them once. */
  compile_info.fusion_nodes_map.clear();
  compile_info.buff_fus_to_del_nodes.clear();
  compile_info.l1_fusion_failed_nodes.clear();

  vector<ge::NodePtr> nodes_be_re_compiled;
  vector<ge::NodePtr> all_nodes;

  (void)GetFusionScope(graph, compile_info.fusion_nodes_map, nodes_be_re_compiled, all_nodes);

  /* Using the same scope json map and we will parse json for all nodes including
   * those which are compiled before lx-fusion(and are not changed in lx-fusion) and those
   * which are changed in lx-fusion and need re-compiling. */
  ret = CompileOpOnly(compile_info);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR(
        "[SubGraphOpt][Compile][Lx] Failed to compile after lx-fusion for graph %s with strategy %u.",
        graph.GetName().c_str(), static_cast<uint32_t>(compile_info.compile_strategy));
    return ret;
  }

  // rollback l1 fusion failed nodes
  if (opt_ret == LxFusionOptimizeResult::BOTH_FUSION_STRATEGY) {
    ret = ReCompileL1FusionFailedNodes(graph, compile_info);
    if (ret != SUCCESS) {
      return ret;
    }
  }

  ret = ParseJsonAndCompressOp(graph, compile_info.compile_ret_map, all_nodes);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][Compile][Lx] Failed to parse json for graph [%s] after lx-fusion.",
                    graph.GetName().c_str());
    return ret;
  }
  return SUCCESS;
}

Status OpCompiler::GenerateFormatTuneResult(ge::ComputeGraph& graph, LxFusionOptimizeResult& buffer_ret,
                                            const bool &need_re_compile) const {
  std::string aoe_type = "";
  bool need_re_precompile = false;
  (void)FeGraphUtils::GetAoeTypeFromRootGraph(graph, aoe_type);
  if (aoe_type != "op_format") {
    OpCompilerFormatTunePtr op_compiler_format_tune_ptr = nullptr;
    FE_MAKE_SHARED(op_compiler_format_tune_ptr = std::make_shared<OpCompilerFormatTune>(AI_CORE_NAME),
                   return fe::OP_COMPILER_MAKE_SHARED_FAILED);
    FE_CHECK_NOTNULL(op_compiler_format_tune_ptr);
    if (op_compiler_format_tune_ptr->UpdateTuneFormatByCannKbResult(graph, need_re_precompile) != SUCCESS) {
      return FAILED;
    }
  }
  if (buffer_ret != LxFusionOptimizeResult::NO_FUSION_STRATEGY) {
    (void) ge::AttrUtils::SetBool(graph, NEED_RE_PRECOMPILE, true);
    ge::GraphUtils::DumpGEGraph(graph.shared_from_this(), "AfterBufferFusion");
    ge::GraphUtils::DumpGEGraphToOnnx(graph, "AfterBufferFusion");
  } else if (buffer_ret == LxFusionOptimizeResult::NO_FUSION_STRATEGY && need_re_compile) {
    buffer_ret = LxFusionOptimizeResult::C_OPTIMIZE_STRATEGY;
    ge::GraphUtils::DumpGEGraph(graph.shared_from_this(), "AfterOptimizeConcat");
    ge::GraphUtils::DumpGEGraphToOnnx(graph, "AfterOptimizeConcat");
  } else if (buffer_ret == LxFusionOptimizeResult::NO_FUSION_STRATEGY && need_re_precompile) {
    buffer_ret = LxFusionOptimizeResult::FORMAT_TUNE_STRATEGY;
    ge::GraphUtils::DumpGEGraph(graph.shared_from_this(), "AfterFormatTune");
    ge::GraphUtils::DumpGEGraphToOnnx(graph, "AfterFormatTune");
  }
  return SUCCESS;
}
}  // namespace fe
