/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer_utils.h"

#include "graph/debug/ge_attr_define.h"
#include "graph/operator_factory_impl.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/op_desc_utils_ex.h"
#include "graph/utils/type_utils.h"
#include "util/log.h"
#include "util/util.h"
#include "util/constant.h"

using namespace ge;
using namespace std;

namespace {
const string kPlaceHolderOpType = "PlaceHolder";
const string kEndOpType = "End";
constexpr int64_t kRT_MEMORY_HOST_SVM = 0x90;
}  // namespace

namespace aicpu {
ge::Status GraphOptimizerUtils::VerifyPldAndEndNode(const ComputeGraph &graph) {
  for (const NodePtr &curr_node : graph.GetDirectNode()) {
    AICPU_CHECK_NOTNULL(curr_node)
    OpDescPtr curr_op_desc_ptr = curr_node->GetOpDesc();
    AICPU_CHECK_NOTNULL(curr_op_desc_ptr)
    string op_type = curr_op_desc_ptr->GetType();
    // if place hold
    if (op_type == kPlaceHolderOpType) {
      uint32_t only_one_input = 0;
      GeTensorDesc pld_in_tensor_desc = curr_op_desc_ptr->GetInputDesc(only_one_input);
      // our format
      ge::Format pld_format = pld_in_tensor_desc.GetFormat();
      // client format
      ge::Format pld_original_format = pld_in_tensor_desc.GetOriginFormat();
      // verify_succ = 1 if verify succ
      bool verify_succ =
          ((pld_format == pld_original_format) || (pld_format == ge::FORMAT_ND));
      CHECK_RES_BOOL(verify_succ, FAILED,
          AICPU_REPORT_INNER_ERR_MSG(
              "Invalied format[%s], should be [FORMAT_ND] or same as original"
              " format[%s]. op[%s], op type[%s]",
              ge::TypeUtils::FormatToSerialString(pld_format).c_str(),
              ge::TypeUtils::FormatToSerialString(pld_original_format).c_str(),
              curr_node->GetName().c_str(), curr_node->GetType().c_str()))
    } else if (op_type == kEndOpType) {
      uint32_t only_one_output = 0;
      GeTensorDesc end_out_tensor_desc =
          curr_op_desc_ptr->GetOutputDesc(only_one_output);
      ge::Format end_format = end_out_tensor_desc.GetFormat();
      ge::Format end_original_format = end_out_tensor_desc.GetOriginFormat();
      // verify_succ = 1 if verify succ
      bool verify_succ =
          ((end_format == end_original_format) || (end_format == ge::FORMAT_ND));
      CHECK_RES_BOOL(verify_succ, FAILED,
          AICPU_REPORT_INNER_ERR_MSG(
              "Invalied format[%s], should be [FORMAT_ND] or same as original"
              " format[%s]. op[%s], op type[%s]",
              ge::TypeUtils::FormatToSerialString(end_format).c_str(),
              ge::TypeUtils::FormatToSerialString(end_original_format).c_str(),
              curr_node->GetName().c_str(), curr_node->GetType().c_str()))
    }
  }
  return SUCCESS;
}

void GraphOptimizerUtils::DumpGraph(ComputeGraph &graph, const string &suffix) {
  ComputeGraphPtr graph_ptr = make_shared<ComputeGraph>(graph);
  GraphUtils::DumpGEGraph(graph_ptr, suffix);
  GraphUtils::DumpGEGraphToOnnx(graph, suffix);
}

ge::Status GraphOptimizerUtils::CheckIsFftsPlus(const ge::OpDescPtr &op_desc_ptr,
                                                ffts::ThreadSliceMapPtr &slice_info_ptr, bool &sgt_flag) {
  AICPU_CHECK_NOTNULL(op_desc_ptr)
  // get ffts+ support info
  uint32_t thread_scope_id = 0;
  (void)ge::AttrUtils::GetInt(op_desc_ptr, kAttrNameThreadScopeId, thread_scope_id);
  if (thread_scope_id == 0) {
    return ge::FAILED;
  }
  sgt_flag = true;
  AICPUE_LOGD("start to do ffts+, sgt_flag is [%d], scope_id[%u]", sgt_flag, thread_scope_id);

  // get sgt info
  slice_info_ptr = op_desc_ptr->TryGetExtAttr(kAttrNameSgtStruct, slice_info_ptr);
  if (slice_info_ptr == nullptr) {
    AICPU_REPORT_INNER_ERR_MSG("The Node[%s][%s] has no attr _sgt_struct_info.", op_desc_ptr->GetType().c_str(),
                            op_desc_ptr->GetName().c_str());
    return static_cast<uint32_t>(INVOKE_GRAPH_ITF_FAILED);
  }
  return ge::SUCCESS;
}

ge::Status CacheGraph::CreateAndInsertCacheUpdate(const OutDataAnchorPtr &src_anchor, const InDataAnchorPtr &dst_anchor,
                                                  ComputeGraph &graph, const ffts::ThreadSliceMapPtr &slice_info_ptr,
                                                  const bool &sgt_flag) {
  auto src_node = src_anchor->GetOwnerNodeBarePtr();
  AICPU_CHECK_NOTNULL(src_node)
  OpDescPtr src_op = src_node->GetOpDesc();
  AICPU_CHECK_NOTNULL(src_op)
  int32_t src_anchor_idx = AnchorUtils::GetIdx(src_anchor);
  CHECK_RES_BOOL(src_anchor_idx >= 0, FAILED,
      AICPU_REPORT_INNER_ERR_MSG("Invalid output anchor index[%d] of op[%s]", src_anchor_idx, src_op->GetName().c_str()))
  GeTensorDesc src_tensor_desc = src_op->GetOutputDesc(static_cast<uint32_t>(src_anchor_idx));

  auto dst_node = dst_anchor->GetOwnerNodeBarePtr();
  AICPU_CHECK_NOTNULL(dst_node)
  OpDescPtr dst_op = dst_node->GetOpDesc();
  AICPU_CHECK_NOTNULL(dst_op)
  int32_t dst_anchor_idx = AnchorUtils::GetIdx(dst_anchor);
  CHECK_RES_BOOL(
      dst_anchor_idx >= 0, FAILED,
      AICPU_REPORT_INNER_ERR_MSG("Invalid input anchor index[%d] of op[%s]", dst_anchor_idx, dst_op->GetName().c_str()))
  GeTensorDesc dst_tensor_desc = dst_op->GetInputDesc(static_cast<uint32_t>(dst_anchor_idx));
  // create CacheUpdate op
  static std::atomic<uint32_t> index{0};
  std::string cache_update_op_name = "CacheUpdate_Insert" + std::to_string(index++);
  const auto cache_update_op = ge::OperatorFactoryImpl::CreateOperator(cache_update_op_name.c_str(), "CacheUpdate");
  CHECK_RES_BOOL(
      !(cache_update_op.IsEmpty()), FAILED,
      AICPU_REPORT_INNER_ERR_MSG("[Verify][CheckParam] Get op from OperatorFactory failed, type: CacheUpdate"));
  AICPUE_LOGI("Get op from OperatorFactory succeeded. opType: CacheUpdate");
  auto cache_update_op_desc = ge::OpDescUtils::GetOpDescFromOperator(cache_update_op);
  AICPU_CHECK_NOTNULL(cache_update_op_desc)
  (void)AttrUtils::SetBool(cache_update_op_desc, ge::ATTR_NAME_REFERENCE, true);
  ge::OpDescUtilsEx::SetType(cache_update_op_desc, "CacheUpdate");
  std::string op_name = src_op->GetName() + "_" + cache_update_op_desc->GetName();
  cache_update_op_desc->SetName(op_name);
  AICPU_CHECK_RES_WITH_LOG(cache_update_op_desc->UpdateInputDesc(0, src_tensor_desc),
                           "Call UpdateInputDesc function failed to update input[0] desc, op[CacheUpdate].")
  AICPU_CHECK_RES_WITH_LOG(
      cache_update_op_desc->UpdateOutputDesc(0, dst_tensor_desc),
      "Call UpdateInputDesc function failed to update output[0] desc, op[CacheUpdate].")

  if ((sgt_flag) && (slice_info_ptr != nullptr)) {
    (void)cache_update_op_desc->SetExtAttr(kAttrNameSgtStruct, slice_info_ptr);
    (void)ge::AttrUtils::SetInt(cache_update_op_desc, kAttrNameThreadScopeId, slice_info_ptr->thread_scope_id);
  }
  NodePtr cache_update_node = graph.AddNode(cache_update_op_desc);
  AICPU_CHECK_NOTNULL(cache_update_node)

  // insert CacheUpdate op
  AICPU_CHECK_RES_WITH_LOG(GraphUtils::RemoveEdge(src_anchor, dst_anchor),
      "Call GraphUtils::RemoveEdge failed to remove edge between op[%s] and "
      "op[%s].", src_op->GetName().c_str(), dst_op->GetName().c_str())
  InDataAnchorPtr input_anchor = cache_update_node->GetInDataAnchor(0);
  AICPU_CHECK_NOTNULL(input_anchor)
  OutDataAnchorPtr output_anchor = cache_update_node->GetOutDataAnchor(0);
  AICPU_CHECK_NOTNULL(output_anchor)
  AICPU_CHECK_RES_WITH_LOG(GraphUtils::AddEdge(src_anchor, input_anchor),
      "Call GraphUtils::AddEdge failed to add edge between op[%s] and CacheUpdate.",
      src_op->GetName().c_str())
  AICPU_CHECK_RES_WITH_LOG(GraphUtils::AddEdge(output_anchor, dst_anchor),
      "Call GraphUtils::AddEdge failed to add edge CacheUpdate and op[%s].",
      dst_op->GetName().c_str())
  return ge::SUCCESS;
}

Status CacheGraph::GenerateNoCacheGraph(ComputeGraph &graph) {
  for (const NodePtr &curr_node : graph.GetDirectNode()) {
    AICPU_CHECK_NOTNULL(curr_node)
    OpDescPtr curr_op_desc_ptr = curr_node->GetOpDesc();
    AICPU_CHECK_NOTNULL(curr_op_desc_ptr)
    string op_type = curr_op_desc_ptr->GetType();
    // if op type is variable, replace it with assign and variable
    if (op_type == kPlaceHolderOpType) {
      const string *pld_front_node_engine = AttrUtils::GetStr(curr_op_desc_ptr, ge::ATTR_NAME_PLD_FRONT_NODE_ENGINE_NAME);
      CHECK_RES_BOOL(
          (pld_front_node_engine != nullptr),
          ErrorCode::GET_ATTR_FAILED,
          AICPU_REPORT_INNER_ERR_MSG("Call ge::AttrUtils::GetStr faield to get attr[%s], op[%s].",
                                  ge::ATTR_NAME_PLD_FRONT_NODE_ENGINE_NAME.c_str(),
                                  curr_op_desc_ptr->GetName().c_str()));

      // if front name engine is aicore enginge or vector engine, insert cache
      // update op
      if ((*pld_front_node_engine == "AIcoreEngine") ||
          (*pld_front_node_engine == "VectorEngine")) {
        // placeholder op just have one output edge
        OutDataAnchorPtr src_anchor = curr_node->GetOutDataAnchor(0);
        AICPU_CHECK_NOTNULL(src_anchor)
        auto nodes_and_anchors = curr_node->GetOutDataNodesAndAnchors();
        AICPU_IF_BOOL_EXEC(nodes_and_anchors.empty(),
            AICPUE_LOGI("no output data adge, op[%s]",
                curr_node->GetName().c_str());
            continue)
        InDataAnchorPtr dst_anchor = nodes_and_anchors.at(0).second;
        AICPU_CHECK_NOTNULL(dst_anchor)
        AICPU_CHECK_NOTNULL(nodes_and_anchors.at(0).first)

        bool sgt_flag = false;
        ffts::ThreadSliceMapPtr slice_info_ptr = nullptr;
        (void)GraphOptimizerUtils::CheckIsFftsPlus(curr_op_desc_ptr, slice_info_ptr, sgt_flag);

        AICPU_CHECK_RES_WITH_LOG(
            CreateAndInsertCacheUpdate(src_anchor, dst_anchor, graph, slice_info_ptr, sgt_flag),
            "Call CreateAndInsertCacheUpdatef failed to insert CacheUpdate op"
            " between op[%s] and op[%s].", curr_node->GetName().c_str(),
            nodes_and_anchors.at(0).first->GetName().c_str())
      }
    }
    if (op_type == kEndOpType) {
      const string *end_rear_node_engine = AttrUtils::GetStr(curr_op_desc_ptr, ge::ATTR_NAME_END_REAR_NODE_ENGINE_NAME);
      CHECK_RES_BOOL((end_rear_node_engine != nullptr),
                     ErrorCode::GET_ATTR_FAILED,
                     AICPU_REPORT_INNER_ERR_MSG("Call ge::AttrUtils::GetStr failed to get attr[%s], op[%s].",
                                             ge::ATTR_NAME_END_REAR_NODE_ENGINE_NAME.c_str(),
                                             curr_op_desc_ptr->GetName().c_str()));

      const string *parent_op_type = AttrUtils::GetStr(curr_op_desc_ptr, "parentOpType");
      // get attr parentOpType
      CHECK_RES_BOOL(
          (parent_op_type != nullptr),
          ErrorCode::GET_ATTR_FAILED,
          AICPU_REPORT_INNER_ERR_MSG(
              "Call ge::AttrUtils::GetStr failed to get attr[parentOpType], op[%s].",
              curr_op_desc_ptr->GetName().c_str()));

      // if rear name engine is aicore enginge or parent op is End, insert cache
      // update op
      if (*end_rear_node_engine == "AIcoreEngine" || *parent_op_type == "NetOutput") {
        auto nodes_and_anchors = curr_node->GetInDataNodesAndAnchors();
        AICPU_IF_BOOL_EXEC(nodes_and_anchors.empty(),
            AICPUE_LOGI("no in data adge, op[%s]",
                curr_node->GetName().c_str());
            continue)
        OutDataAnchorPtr src_anchor = nodes_and_anchors.at(0).second;
        AICPU_CHECK_NOTNULL(src_anchor)
        InDataAnchorPtr dst_anchor = curr_node->GetInDataAnchor(0);
        AICPU_CHECK_NOTNULL(dst_anchor)
        AICPU_CHECK_NOTNULL(nodes_and_anchors.at(0).first)
        AICPU_CHECK_RES_WITH_LOG(
            CreateAndInsertCacheUpdate(src_anchor, dst_anchor, graph, nullptr, false),
            "Call CreateAndInsertCacheUpdate failed to insert CacheUpdate op"
            " between op[%s] and op[%s].",
            nodes_and_anchors.at(0).first->GetName().c_str(),
            curr_node->GetName().c_str())
      }
    }
  }
  return SUCCESS;
}

ge::Status AutoCastGraph::GenerateAutoCastGraph(ge::ComputeGraph &graph,
                                                const std::map<std::string, OpFullInfo> &all_op_info) {
  for (ge::NodePtr &node : graph.GetDirectNode()) {
    AICPU_CHECK_NOTNULL(node)
    ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
    AICPU_CHECK_NOTNULL(op_desc_ptr)
    std::string op_type = op_desc_ptr->GetType();
    AICPU_IF_BOOL_EXEC(
        ((op_type == kPlaceHolderOpType) || (op_type == kEndOpType) || (op_type == kFunctionOp)),
        AICPUE_LOGI("Current op type is [%s]. Don't need to AutoCast.",
                    op_type.c_str());
        continue)
    // if op type is framework_op, get original op
    AICPU_IF_BOOL_EXEC(
        (op_type == kFrameworkOp),
        AICPU_CHECK_RES(GetFrameworkOpType(op_desc_ptr, op_type)))
    std::map<std::string, OpFullInfo>::const_iterator iter = all_op_info.find(op_type);
    if (iter == all_op_info.end()) {
      continue;
    }

    bool sgt_flag = false;
    ffts::ThreadSliceMapPtr slice_info_ptr = nullptr;
    (void)GraphOptimizerUtils::CheckIsFftsPlus(op_desc_ptr, slice_info_ptr, sgt_flag);

    OpFullInfo op_full_info = iter->second;
    auto cast_src_type = op_full_info.castSrcType;
    auto cast_dst_type = op_full_info.castDstType;
    // auto cast for input
    size_t input_num = op_desc_ptr->GetInputsSize();
    for (size_t i = 0; i < input_num; ++i) {
      // get src_to_dst_type map
      string input_real_name = "input" + to_string(i);
      vector<DataType> src_data_type;
      GetDataType(cast_src_type, input_real_name, src_data_type);
      vector<DataType> dst_data_type;
      GetDataType(cast_dst_type, input_real_name, dst_data_type);
      AICPU_IF_BOOL_EXEC(
        (src_data_type.size() != dst_data_type.size()),
        AICPU_REPORT_INNER_ERR_MSG("Src data type size[%lu] is not equal to dst data type size[%lu], please check!",
                                 src_data_type.size(), dst_data_type.size());
        return ErrorCode::SRC_DST_SIZE_ERROR)
      auto cast_size = src_data_type.size();
      map<DataType, DataType> src_to_dst_type;
      for (size_t j = 0; j < cast_size; ++j) {
        src_to_dst_type[src_data_type[j]] = dst_data_type[j];
      }
      // insert cast op if rules are set
      auto input_tensor_desc = op_desc_ptr->GetInputDesc(static_cast<uint32_t>(i));
      auto input_type = input_tensor_desc.GetDataType();
      map<DataType, DataType>::const_iterator iter1 = src_to_dst_type.find(input_type);
      if (iter1 != src_to_dst_type.end()) {
        DataType dst_type = iter1->second;
        InDataAnchorPtr dst_anchor = node->GetInDataAnchor(static_cast<int32_t>(i));
        AICPU_CHECK_NOTNULL(dst_anchor)
        AICPU_CHECK_RES_WITH_LOG(
            InsertCastForInput(dst_anchor, graph, dst_type, slice_info_ptr, sgt_flag),
            "Insert Cast op for input[%zu] of op[%s] failed.", i, node->GetName().c_str())
      }
    }
    // auto cast for output
    size_t output_num = op_desc_ptr->GetOutputsSize();
    for (size_t i = 0; i < output_num; ++i) {
      // get dst_to_src_type map
      string output_real_name = "output" + to_string(i);
      vector<DataType> src_data_type;
      GetDataType(cast_src_type, output_real_name, src_data_type);
      vector<DataType> dst_data_type;
      GetDataType(cast_dst_type, output_real_name, dst_data_type);
      auto cast_size = src_data_type.size() < dst_data_type.size() ? src_data_type.size() : dst_data_type.size();
      map<DataType, DataType> dst_to_src_type;
      for (size_t j = 0; j < cast_size; ++j) {
        dst_to_src_type[dst_data_type[j]] = src_data_type[j];
      }
      // insert cast op if rules are set
      auto output_tensor_desc = op_desc_ptr->GetOutputDesc(static_cast<uint32_t>(i));
      auto output_type = output_tensor_desc.GetDataType();
      map<DataType, DataType>::const_iterator iter2 = dst_to_src_type.find(output_type);
      if (iter2 != dst_to_src_type.end()) {
        DataType src_type = iter2->second;
        OutDataAnchorPtr src_anchor = node->GetOutDataAnchor(static_cast<int32_t>(i));
        AICPU_CHECK_NOTNULL(src_anchor)
        AICPU_CHECK_RES_WITH_LOG(
            InsertCastForOutput(src_anchor, graph, src_type, output_type),
            "Insert Cast op for output[%zu] of op[%s] failed.", i, node->GetName().c_str())
      }
    }
  }
  return ge::SUCCESS;
}

ge::Status AutoCastGraph::InsertCastForInput(const ge::InDataAnchorPtr &dst_anchor, ge::ComputeGraph &graph,
                                             ge::DataType dst_type,
                                             const ffts::ThreadSliceMapPtr &slice_info_ptr, const bool &sgt_flag) {
  // get src op desc
  OutDataAnchorPtr src_anchor = dst_anchor->GetPeerOutAnchor();
  AICPU_CHECK_NOTNULL(src_anchor)
  auto src_node = src_anchor->GetOwnerNodeBarePtr();
  AICPU_CHECK_NOTNULL(src_node)
  OpDescPtr src_op = src_node->GetOpDesc();
  AICPU_CHECK_NOTNULL(src_op)
  int32_t src_anchor_idx = AnchorUtils::GetIdx(src_anchor);
  CHECK_RES_BOOL(src_anchor_idx >= 0, FAILED,
      AICPU_REPORT_INNER_ERR_MSG("Invalid output anchor index[%d] of op[%s]", src_anchor_idx, src_op->GetName().c_str()))
  GeTensorDesc src_tensor_desc = src_op->GetOutputDesc(static_cast<uint32_t>(src_anchor_idx));
  // get dst op desc
  auto dst_node = dst_anchor->GetOwnerNodeBarePtr();
  AICPU_CHECK_NOTNULL(dst_node)
  OpDescPtr dst_op = dst_node->GetOpDesc();
  AICPU_CHECK_NOTNULL(dst_op)
  int32_t dst_anchor_idx = AnchorUtils::GetIdx(dst_anchor);
  CHECK_RES_BOOL(
      dst_anchor_idx >= 0, FAILED,
      AICPU_REPORT_INNER_ERR_MSG("Invalid input anchor index[%d] of op[%s]",
          dst_anchor_idx, dst_op->GetName().c_str()))
  GeTensorDesc dst_tensor_desc = dst_op->GetInputDesc(static_cast<uint32_t>(dst_anchor_idx));
  // update input desc of dst node
  dst_tensor_desc.SetDataType(dst_type);
  AICPU_CHECK_RES_WITH_LOG(dst_op->UpdateInputDesc(dst_anchor_idx, dst_tensor_desc),
                           "Dst op[%s] update input[%d] desc failed.", dst_op->GetName().c_str(), dst_anchor_idx)
  // create Cast op
  static std::atomic<uint32_t> index{0};
  std::string cast_op_name = "Cast_InsertForInput" + std::to_string(index++);
  const auto cast_op = ge::OperatorFactoryImpl::CreateOperator(cast_op_name.c_str(), "Cast");
  CHECK_RES_BOOL(
      !(cast_op.IsEmpty()), FAILED,
      AICPU_REPORT_INNER_ERR_MSG("[Verify][CheckParam] Get op from OperatorFactory failed, type: Cast"));
  AICPUE_LOGI("Get op from OperatorFactory succeeded. opType: Cast");
  auto cast_op_desc = ge::OpDescUtils::GetOpDescFromOperator(cast_op);
  AICPU_CHECK_NOTNULL(cast_op_desc)
  (void)ge::AttrUtils::SetInt(cast_op_desc, "dst_type", static_cast<int64_t>(dst_type));
  ge::OpDescUtilsEx::SetType(cast_op_desc, "Cast");
  std::string op_name = src_op->GetName() + "_" + cast_op_desc->GetName();
  cast_op_desc->SetName(op_name);
  cast_op_desc->SetOpEngineName(dst_op->GetOpEngineName());
  cast_op_desc->SetOpKernelLibName(dst_op->GetOpKernelLibName());
  AICPU_CHECK_RES_WITH_LOG(cast_op_desc->UpdateInputDesc(0, src_tensor_desc),
                           "Cast update input 0 desc failed.")
  AICPU_CHECK_RES_WITH_LOG(cast_op_desc->UpdateOutputDesc(0, dst_tensor_desc), "Cast update output 0 desc failed.")
  if ((sgt_flag) && (slice_info_ptr != nullptr)) {
    (void)cast_op_desc->SetExtAttr(kAttrNameSgtStruct, slice_info_ptr);
    (void)ge::AttrUtils::SetInt(cast_op_desc, kAttrNameThreadScopeId, slice_info_ptr->thread_scope_id);
  }
  // insert Cast op
  NodePtr cast_node = graph.AddNode(cast_op_desc);
  AICPU_CHECK_NOTNULL(cast_node)
  AICPU_CHECK_RES_WITH_LOG(GraphUtils::RemoveEdge(src_anchor, dst_anchor),
                           "Remove edge between op[%s] and op[%s] failed.",
                           src_op->GetName().c_str(), dst_op->GetName().c_str())
  InDataAnchorPtr input_anchor = cast_node->GetInDataAnchor(0);
  AICPU_CHECK_NOTNULL(input_anchor)
  OutDataAnchorPtr output_anchor = cast_node->GetOutDataAnchor(0);
  AICPU_CHECK_NOTNULL(output_anchor)
  AICPU_CHECK_RES_WITH_LOG(GraphUtils::AddEdge(src_anchor, input_anchor),
                           "Add edge between op[%s] and Cast failed.",
                           src_op->GetName().c_str())
  AICPU_CHECK_RES_WITH_LOG(GraphUtils::AddEdge(output_anchor, dst_anchor),
                           "Add edge between Cast and op[%s] failed.",
                           dst_op->GetName().c_str())
  AICPUE_LOGI("Insert Cast op between op[%s] and op[%s] success.", src_op->GetName().c_str(),
              dst_op->GetName().c_str());
  return ge::SUCCESS;
}

ge::Status AutoCastGraph::InsertCastForOutput(
    const ge::OutDataAnchorPtr &src_anchor, ge::ComputeGraph &graph, ge::DataType src_type, ge::DataType dst_type) {
  // get src op desc
  auto src_node = src_anchor->GetOwnerNodeBarePtr();
  AICPU_CHECK_NOTNULL(src_node)
  OpDescPtr src_op = src_node->GetOpDesc();
  AICPU_CHECK_NOTNULL(src_op)
  int32_t src_anchor_idx = AnchorUtils::GetIdx(src_anchor);
  CHECK_RES_BOOL(
      src_anchor_idx >= 0, FAILED,
      AICPU_REPORT_INNER_ERR_MSG("Invalid output anchor index[%d] of op[%s]",
          src_anchor_idx, src_op->GetName().c_str()))
  GeTensorDesc src_tensor_desc = src_op->GetOutputDesc(static_cast<uint32_t>(src_anchor_idx));
  // update output desc of src node
  src_tensor_desc.SetDataType(src_type);
  AICPU_CHECK_RES_WITH_LOG(
      src_op->UpdateOutputDesc(static_cast<uint32_t>(src_anchor_idx), src_tensor_desc),
      "Src op[%s] update output[%d] desc failed.", src_op->GetName().c_str(), src_anchor_idx)
  // create Cast op
  static std::atomic<uint32_t> index{0};
  std::string cast_op_name = "Cast_InsertForOutput" + std::to_string(index++);
  const auto cast_op = ge::OperatorFactoryImpl::CreateOperator(cast_op_name.c_str(), "Cast");
  CHECK_RES_BOOL(
      !(cast_op.IsEmpty()), FAILED,
      AICPU_REPORT_INNER_ERR_MSG("[Verify][CheckParam] Get op from OperatorFactory failed, type: Cast"));
  AICPUE_LOGI("Get op from OperatorFactory succeeded. opType: Cast");
  auto cast_op_desc = ge::OpDescUtils::GetOpDescFromOperator(cast_op);
  AICPU_CHECK_NOTNULL(cast_op_desc)
  (void)ge::AttrUtils::SetInt(cast_op_desc, "dst_type", static_cast<int64_t>(dst_type));
  ge::OpDescUtilsEx::SetType(cast_op_desc, "Cast");
  std::string op_name = src_op->GetName() + "_" + cast_op_desc->GetName();
  cast_op_desc->SetName(op_name);
  cast_op_desc->SetOpEngineName(src_op->GetOpEngineName());
  cast_op_desc->SetOpKernelLibName(src_op->GetOpKernelLibName());
  AICPU_CHECK_RES_WITH_LOG(cast_op_desc->UpdateInputDesc(0, src_tensor_desc),
                           "Cast update input 0 desc failed.")
  src_tensor_desc.SetDataType(dst_type);
  AICPU_CHECK_RES_WITH_LOG(
      cast_op_desc->UpdateOutputDesc(0, src_tensor_desc),
      "Cast update output 0 desc failed.")
  // insert Cast op
  NodePtr cast_node = graph.AddNode(cast_op_desc);
  AICPU_CHECK_NOTNULL(cast_node)
  InDataAnchorPtr input_anchor = cast_node->GetInDataAnchor(0);
  AICPU_CHECK_NOTNULL(input_anchor)
  OutDataAnchorPtr output_anchor = cast_node->GetOutDataAnchor(0);
  AICPU_CHECK_NOTNULL(output_anchor)
  for (auto dst_anchor: src_anchor->GetPeerInDataAnchors()) {
    AICPU_CHECK_NOTNULL(dst_anchor)
    AICPU_CHECK_NOTNULL(dst_anchor->GetOwnerNode())
    AICPU_CHECK_RES_WITH_LOG(GraphUtils::RemoveEdge(src_anchor, dst_anchor),
                             "Remove edge between op[%s] and op[%s] failed.",
                             src_op->GetName().c_str(), dst_anchor->GetOwnerNode()->GetName().c_str())
    AICPU_CHECK_RES_WITH_LOG(GraphUtils::AddEdge(output_anchor, dst_anchor),
                             "Add edge between op[%s] and op[%s] failed.",
                             cast_node->GetName().c_str(), dst_anchor->GetOwnerNode()->GetName().c_str())
  }
  AICPU_CHECK_RES_WITH_LOG(GraphUtils::AddEdge(src_anchor, input_anchor),
                           "Add edge between op[%s] and Cast failed.",
                           src_op->GetName().c_str())
  AICPUE_LOGI("Insert Cast op for output[%d] of op[%s] success.", src_anchor_idx, src_op->GetName().c_str());
  return ge::SUCCESS;
}

ge::Status AutoCastGraph::GetFrameworkOpType(const OpDescPtr &op_desc_ptr, string &op_type) {
  // op_desc_ptr already check not null
  const string *original_type = AttrUtils::GetStr(op_desc_ptr, kOriginalType);
  CHECK_RES_BOOL((original_type != nullptr),
      ErrorCode::GET_ATTR_FAILED,
      AICPU_REPORT_INNER_ERR_MSG(
          "Call ge::AttrUtils::GetStr failed to get attr[%s], op[%s].",
          kOriginalType.c_str(), op_desc_ptr->GetName().c_str()))
  if (original_type->empty()) {
    AICPU_REPORT_INNER_ERR_MSG("Attr[%s] is empty, op[%s].", kOriginalType.c_str(),
        op_desc_ptr->GetName().c_str());
    return ErrorCode::STR_IS_EMPTY;
  }
  ge::OpDescUtilsEx::SetType(const_cast<OpDescPtr &>(op_desc_ptr), *original_type);
  op_type = *original_type;
  return ge::SUCCESS;
}
}  // namespace aicpu
