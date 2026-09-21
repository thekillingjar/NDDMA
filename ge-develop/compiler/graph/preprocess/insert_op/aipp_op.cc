/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/preprocess/insert_op/aipp_op.h"

#include "base/err_msg.h"

#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "common/dynamic_aipp.h"
#include "common/plugin/ge_make_unique_util.h"
#include "framework/common/util.h"
#include "framework/common/op/ge_op_utils.h"
#include "framework/omg/omg_inner_types.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/op_desc_utils_ex.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"
#include "common/context/local_context.h"
#include "common/checker.h"

#define SAVE_AIPP_ATTR(AIPP_ATTRS, KEY, SAVE_TYPE)                                           \
  do {                                                                                       \
    (void)(AIPP_ATTRS).SetAttr(#KEY, GeAttrValue::CreateFrom<SAVE_TYPE>(aipp_params_->KEY())); \
  } while (0)

#define SAVE_AIPP_ATTR_LIST(AIPP_ATTRS, KEY, SAVE_TYPE)                                         \
  do {                                                                                          \
    if (aipp_params_->KEY##_size() > 0) {                                                       \
      (void)(AIPP_ATTRS).SetAttr(#KEY, GeAttrValue::CreateFrom<SAVE_TYPE>(aipp_params_->KEY(0))); \
    }                                                                                           \
  } while (0)

namespace {
const int32_t DEFAULT_MATRIX_R0C0_YUV2RGB = 298;
const int32_t DEFAULT_MATRIX_R0C1_YUV2RGB = 0;
const int32_t DEFAULT_MATRIX_R0C2_YUV2RGB = 409;
const int32_t DEFAULT_MATRIX_R1C0_YUV2RGB = 298;
const int32_t DEFAULT_MATRIX_R1C1_YUV2RGB = -100;
const int32_t DEFAULT_MATRIX_R1C2_YUV2RGB = -208;
const int32_t DEFAULT_MATRIX_R2C0_YUV2RGB = 298;
const int32_t DEFAULT_MATRIX_R2C1_YUV2RGB = 516;
const int32_t DEFAULT_MATRIX_R2C2_YUV2RGB = 0;
const int32_t DEFAULT_MATRIX_R0C0_RGB2YUV = 66;
const int32_t DEFAULT_MATRIX_R0C1_RGB2YUV = 129;
const int32_t DEFAULT_MATRIX_R0C2_RGB2YUV = 25;
const int32_t DEFAULT_MATRIX_R1C0_RGB2YUV = -38;
const int32_t DEFAULT_MATRIX_R1C1_RGB2YUV = -74;
const int32_t DEFAULT_MATRIX_R1C2_RGB2YUV = 112;
const int32_t DEFAULT_MATRIX_R2C0_RGB2YUV = 112;
const int32_t DEFAULT_MATRIX_R2C1_RGB2YUV = -94;
const int32_t DEFAULT_MATRIX_R2C2_RGB2YUV = -18;
const int32_t DEFAULT_OUTPUT_BIAS_0 = 16;
const int32_t DEFAULT_OUTPUT_BIAS_1 = 128;
const int32_t DEFAULT_OUTPUT_BIAS_2 = 128;
const int32_t DEFAULT_INPUT_BIAS_0 = 16;
const int32_t DEFAULT_INPUT_BIAS_1 = 128;
const int32_t DEFAULT_INPUT_BIAS_2 = 128;
const float DEFAULT_VAR_RECI_CHN = 1.0f;
const float EPSINON = 1e-6f;
const std::map<domi::AippOpParams::InputFormat, std::string> kInputFormatMap = {
  {domi::AippOpParams::YUV420SP_U8, "YUV420SP_U8"},
  {domi::AippOpParams::XRGB8888_U8, "XRGB8888_U8"},
  {domi::AippOpParams::RGB888_U8, "RGB888_U8"},
  {domi::AippOpParams::YUV400_U8, "YUV400_U8"},
  {domi::AippOpParams::NC1HWC0DI_FP16, "NC1HWC0DI_FP16"},
  {domi::AippOpParams::NC1HWC0DI_S8, "NC1HWC0DI_S8"},
  {domi::AippOpParams::ARGB8888_U8, "ARGB8888_U8"},
  {domi::AippOpParams::YUYV_U8, "YUYV_U8"},
  {domi::AippOpParams::YUV422SP_U8, "YUV422SP_U8"},
  {domi::AippOpParams::AYUV444_U8, "AYUV444_U8"},
  {domi::AippOpParams::RAW10, "RAW10"},
  {domi::AippOpParams::RAW12, "RAW12"},
  {domi::AippOpParams::RAW16, "RAW16"},
  {domi::AippOpParams::RAW24, "RAW24"},
  {domi::AippOpParams::RGB16, "RGB16"},
  {domi::AippOpParams::RGB20, "RGB20"},
  {domi::AippOpParams::RGB24, "RGB24"},
  {domi::AippOpParams::RGB8_IR, "RGB8_IR"},
  {domi::AippOpParams::RGB16_IR, "RGB16_IR"},
  {domi::AippOpParams::RGB24_IR, "RGB24_IR"}
};

const std::map<domi::AippOpParams::AippMode, std::string> kAippModeMap = {
  {domi::AippOpParams::static_, "static"},
  {domi::AippOpParams::dynamic, "dynamic"}
};
}  // namespace

namespace ge {
namespace {
const char *const kAippConfigPath = "aipp_config_path";
const char *const kCurrentAippIndex = "current_aipp_index";
const char *const kDynamicAippData = "ascend_dynamic_aipp_data";
const uint64_t kMinTransferShape = 3;
const int32_t kAippImageInputIndex = 0;
const int32_t kAippImageOuputIndex = 0;
const int32_t kAippParamsInputIndex = 1;
const int32_t kAippDataOutputIndex = 0;
const int64_t kDynamicDim = -1;
constexpr uint32_t kRecursionDepth = 10U;

// the `format` must one NCHW or NHWC
Status GetDataDimN(const ge::NodePtr &data_node, ge::Format format, int64_t &batch) {
  auto output_desc = NodeUtils::GetOutputDesc(*data_node, 0);
  auto shape = output_desc.GetShape().GetDims();
  if (shape.size() == kMinTransferShape) {
    batch = 1;
    return SUCCESS;
  }
  if (shape.size() == DIM_DEFAULT_SIZE) {
    switch (format) {
      case FORMAT_NCHW:
        batch = shape[NCHW_DIM_N];
        return SUCCESS;
      case FORMAT_NHWC:
        batch = shape[NHWC_DIM_N];
        return SUCCESS;
      default:
        REPORT_PREDEFINED_ERR_MSG(
            "E10001", std::vector<const char_t *>({"parameter", "value", "reason"}),
            std::vector<const char_t *>(
                {((data_node->GetName() + " format").c_str(), TypeUtils::FormatToSerialString(format)).c_str(),
                 ("Only format " + TypeUtils::FormatToSerialString(FORMAT_NCHW) + " and " +
                  TypeUtils::FormatToSerialString(FORMAT_NHWC) + " are supported when dynamic AIPP is linked.")
                     .c_str()}));
        GELOGE(PARAM_INVALID, "[Check][Param] Not support data format:%s, node:%s",
               TypeUtils::FormatToSerialString(format).c_str(), data_node->GetName().c_str());
        return PARAM_INVALID;
    }
  }
  std::string reason =
      "Its shape size must be in the range[3,4] when dynamic AIPP is linked. "
      "The input may not be suitable for dynamic AIPP.";
  REPORT_PREDEFINED_ERR_MSG("E10001", std::vector<const char *>({"parameter", "value", "reason"}),
                            std::vector<const char *>({(data_node->GetName() + " shape size").c_str(),
                                                       to_string(shape.size()).c_str(), reason.c_str()}));
  GELOGE(PARAM_INVALID, "[Check][Param] The shape size of this node [%s] "
         "which linked dynamic aipp must be in range[3, 4], but is %zu",
         data_node->GetName().c_str(), shape.size());
  return PARAM_INVALID;
}

// the batch_count must be more than 0
int64_t CalcMaxSize(int64_t batch_count) {
  batch_count--;
  if (batch_count > 0) {
    if (INT64_MAX / batch_count < static_cast<int64_t>(sizeof(kAippDynamicBatchPara))) {
      return -1;
    }
  }

  int64_t size = batch_count * sizeof(kAippDynamicBatchPara);
  if (size > (INT64_MAX - static_cast<int64_t>(sizeof(kAippDynamicPara)))) {
    return -1;
  }
  return size + sizeof(kAippDynamicPara);
}

Format GetAndCheckFormat() {
  switch (GetLocalOmgContext().format) {
    case domi::DOMI_TENSOR_NCHW:
      return FORMAT_NCHW;
    case domi::DOMI_TENSOR_NHWC:
      return FORMAT_NHWC;
    default:
      GELOGE(PARAM_INVALID, "[Check][Param] Unexpected format found %d",
             static_cast<int32_t>(GetLocalOmgContext().format));
      return FORMAT_ND;
  }
}
}  // namespace

Status AippOp::Init(domi::AippOpParams *aipp_params) {
  aipp_params_ = new (std::nothrow) domi::AippOpParams();
  if (aipp_params_ == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "New AippOpParams failed");
    GELOGE(FAILED, "[New][AippOpParams] failed");
    return FAILED;
  }
  aipp_params_->CopyFrom(*aipp_params);
  cfg_json_ = ConvertParamToJson();
  return SUCCESS;
}

AippOp::~AippOp() {
  if (aipp_params_ != nullptr) {
    delete aipp_params_;
    aipp_params_ = nullptr;
  }
}

Status AippOp::InsertAippToGraph(ComputeGraphPtr &graph, std::string &aippConfigPath, const uint32_t index) {
  (void)aippConfigPath;
  GE_CHECK_NOTNULL(graph);
  NodePtr target_input = nullptr;
  std::vector<std::pair<OutDataAnchorPtr, InDataAnchorPtr>> target_edges;

  if (this->ConvertRelatedInputNameToRank() != SUCCESS) {
    GELOGE(FAILED, "[Call][ConvertRelatedInputNameToRank] failed.");
    return FAILED;
  }
  GE_CHK_STATUS_RET(this->GetTargetPosition(graph, target_input, target_edges), "[Get][TargetPosition] failed");

  std::map<OutDataAnchorPtr, NodePtr> out_anchors_to_aipp;
  for (auto &out_in_anchors : target_edges) {
    auto iter = out_anchors_to_aipp.find(out_in_anchors.first);
    if (iter == out_anchors_to_aipp.end()) {
      auto aipp = CreateAipp(out_in_anchors.first, index, out_in_anchors.second);
      GE_CHECK_NOTNULL(aipp);
      out_anchors_to_aipp[out_in_anchors.first] = aipp;

      auto ret = GraphUtils::InsertNodeBetweenDataAnchors(out_in_anchors.first, out_in_anchors.second, aipp);
      if (ret != GRAPH_SUCCESS) {
        REPORT_INNER_ERR_MSG("E19999", "Insert aipp:%s(%s) node between op:%s(%s) and op:%s:%s failed",
                          aipp->GetName().c_str(), aipp->GetType().c_str(),
                          out_in_anchors.first->GetOwnerNode()->GetName().c_str(),
                          out_in_anchors.first->GetOwnerNode()->GetType().c_str(),
                          out_in_anchors.second->GetOwnerNode()->GetName().c_str(),
                          out_in_anchors.second->GetOwnerNode()->GetType().c_str());
        GELOGE(INTERNAL_ERROR, "[Insert][Node] %s(%s) between op:%s(%s) and op:%s:%s failed",
               aipp->GetName().c_str(), aipp->GetType().c_str(),
               out_in_anchors.first->GetOwnerNode()->GetName().c_str(),
               out_in_anchors.first->GetOwnerNode()->GetType().c_str(),
               out_in_anchors.second->GetOwnerNode()->GetName().c_str(),
               out_in_anchors.second->GetOwnerNode()->GetType().c_str());
        return INTERNAL_ERROR;
      }

      // add aipp data if needed
      if (GetAippMode() == domi::AippOpParams::dynamic) {
        ret = CreateAippData(aipp);
        if (ret != SUCCESS) {
          GELOGE(INTERNAL_ERROR, "[Create][AippData] for aipp %s data %s failed", aipp->GetName().c_str(),
                 out_in_anchors.first->GetOwnerNode()->GetName().c_str());
          return INTERNAL_ERROR;
        }
      }
      GELOGI("Create aipp %s and insert it to the graph", aipp->GetName().c_str());
    } else {
      out_in_anchors.second->UnlinkAll();
      auto &aipp = iter->second;
      auto ret = out_in_anchors.second->LinkFrom(aipp->GetOutDataAnchor(0));
      if (ret != GRAPH_SUCCESS) {
        REPORT_INNER_ERR_MSG("E19999", "link aipp:%s(%s) to peer op:%s(%s) failed",
                          aipp->GetName().c_str(), aipp->GetType().c_str(),
                          out_in_anchors.second->GetOwnerNode()->GetName().c_str(),
                          out_in_anchors.second->GetOwnerNode()->GetType().c_str());
        GELOGE(INTERNAL_ERROR, "[Call][LinkFrom] Failed to link aipp %s to the peer node %s", aipp->GetName().c_str(),
               out_in_anchors.second->GetOwnerNode()->GetName().c_str());
        return INTERNAL_ERROR;
      }
    }
  }

  return SUCCESS;
}

NodePtr AippOp::CreateAipp(const OutDataAnchorPtr &out_anchor,
                           const uint32_t index, const InDataAnchorPtr &in_anchor) {
  const auto &node = out_anchor->GetOwnerNode();
  std::string current_name = node->GetName() + "_" + std::to_string(out_anchor->GetIdx()) + "_aipp";
  auto aipp_opdesc_ptr = MakeShared<OpDesc>(current_name, AIPP);
  if (aipp_opdesc_ptr == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "New OpDesc failed");
    GELOGE(OUT_OF_MEMORY, "[New][OpDesc] failed, name %s", current_name.c_str());
    return nullptr;
  }

  // Update attributes
  GE_ASSERT_SUCCESS(AddAippAttrbutes(aipp_opdesc_ptr, index));
  // Update input desc, the output desc will not be flushed when InferShape
  // Just keep the same with the next node's input
  const auto &node_in_desc = out_anchor->GetOwnerNode()->GetOpDesc();
  GE_ASSERT_NOTNULL(node_in_desc, "[Call][CreateAipp] node_in_desc == nullptr.");
  auto opdesc_in_src_data = node_in_desc->GetOutputDesc(out_anchor->GetIdx());
  auto node_out_desc_dtype = opdesc_in_src_data.GetDataType();
  if (opdesc_in_src_data.GetDataType() != DT_FLOAT) {
    GELOGW("The datatype of data node %s is not FP32", node_in_desc->GetName().c_str());
    opdesc_in_src_data.SetDataType(DT_FLOAT);
  }

  // We must get the TensorDesc from the output anchor on the Data node,
  // and update the TensorDesc to the input anchor on the Aipp node.
  // Because the InferShape function for the Aipp node needs the input tensor format,
  // but the InferFormat process before InferShape can not infer the format
  // if the tensor on the Aipp has an unknown shape
  if (aipp_opdesc_ptr->UpdateInputDesc(kAippImageInputIndex, opdesc_in_src_data) != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Update the output desc from node:%s(%s) to aipp:%s(%s) failed",
                      node_in_desc->GetName().c_str(), node_in_desc->GetType().c_str(),
                      aipp_opdesc_ptr->GetName().c_str(), aipp_opdesc_ptr->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Call][UpdateInputDesc] Failed to update the output desc from node %s to aipp %s",
           node_in_desc->GetName().c_str(), aipp_opdesc_ptr->GetName().c_str());
    return nullptr;
  }

  // Update output desc, the output desc  will be flushed when InferShape
  const auto &node_out_desc = in_anchor->GetOwnerNode()->GetOpDesc();
  GE_ASSERT_NOTNULL(node_out_desc, "[Call][CreateAipp] node_out_desc == nullptr.");
  auto opdesc_out_src_data = node_out_desc->GetInputDesc(in_anchor->GetIdx());
  opdesc_out_src_data.SetDataType(node_out_desc_dtype);
  if (aipp_opdesc_ptr->UpdateOutputDesc(kAippImageOuputIndex, opdesc_out_src_data) != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Update the input desc from node:%s(%s) to aipp:%s(%s) failed",
                      node_out_desc->GetName().c_str(), node_out_desc->GetType().c_str(),
                      aipp_opdesc_ptr->GetName().c_str(), aipp_opdesc_ptr->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Call][UpdateOutputDesc] Failed to update the input desc from node %s to aipp %s",
           node_out_desc->GetName().c_str(), aipp_opdesc_ptr->GetName().c_str());
    return nullptr;
  }

  return node->GetOwnerComputeGraph()->AddNode(aipp_opdesc_ptr);
}

Status AippOp::AddAippAttrbutes(const OpDescPtr &op_desc, const uint32_t &index) {
  NamedAttrs aipp_attr;
  ConvertParamToAttr(aipp_attr);
  GE_CHK_BOOL_RET_STATUS(AttrUtils::SetNamedAttrs(op_desc, ATTR_NAME_AIPP, aipp_attr),
                         INTERNAL_ERROR, "[Set][NamedAttrs] %s for aipp node:%s failed", ATTR_NAME_AIPP.c_str(),
                         op_desc->GetName().c_str());
  if (cfg_json_.empty()) {
    REPORT_INNER_ERR_MSG("E19999", "Aipp config data of op: %s is empty when change it to json",
        op_desc->GetName().c_str());
    GELOGE(INTERNAL_ERROR, "Aipp config data of op: %s is empty when change it to json",
        op_desc->GetName().c_str());
    return INTERNAL_ERROR;
  }
  GE_CHK_BOOL_RET_STATUS(AttrUtils::SetStr(op_desc, kAippConfigPath, cfg_json_),
                         INTERNAL_ERROR, "[Set][Attr] config file path for aipp node:%s failed",
                         op_desc->GetName().c_str());
  std::vector<std::string> empty_names;
  GE_CHK_BOOL_RET_STATUS(AttrUtils::SetListStr(op_desc, ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, empty_names),
                         INTERNAL_ERROR, "[Set][Attr] %s for aipp node:%s failed",
                         ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES.c_str(), op_desc->GetName().c_str());

  GE_CHK_BOOL_RET_STATUS(AttrUtils::SetInt(op_desc, kCurrentAippIndex, index),
                         INTERNAL_ERROR, "[Set][Attr] %s for aipp node:%s failed", kCurrentAippIndex,
                         op_desc->GetName().c_str());
  // add input/output desc
  GeTensorDesc tensor;
  GE_CHK_GRAPH_STATUS_RET(op_desc->AddInputDesc("images", tensor),
                          "[Add][InputDesc] images for aipp node:%s failed", op_desc->GetName().c_str());

  if (GetAippMode() == domi::AippOpParams::dynamic) {
    GE_CHK_GRAPH_STATUS_RET(op_desc->AddOptionalInputDesc("params", tensor),
                            "[Call][AddOptionalInputDesc] Failed to add params for aipp node:%s",
                            op_desc->GetName().c_str());
  }
  GE_CHK_GRAPH_STATUS_RET(op_desc->AddOutputDesc("features", tensor),
                          "[Add][OutputDesc] features for aipp node:%s failed", op_desc->GetName().c_str());

  return SUCCESS;
}

domi::AippOpParams::AippMode AippOp::GetAippMode() { return aipp_params_->aipp_mode(); }

bool AippOp::InDataAnchorHasSubGraphs(const NodePtr &node) const {
  for (const auto &next_node_and_anchor : node->GetOutDataNodesAndAnchors()) {
    const auto &next_node = next_node_and_anchor.first;
    if (next_node->GetType() != PARTITIONEDCALL) {
      continue;
    }
    const auto &src_root_compute_graph = GraphUtils::FindRootGraph(next_node->GetOwnerComputeGraph());
    for (const auto &cur_sub_graph : src_root_compute_graph->GetAllSubgraphs()) {
      if (cur_sub_graph->GetParentNode()->GetName() != next_node->GetName()) {
        continue;
      }
      return true;
    }
  }
  return false;
}

Status AippOp::FindDataNodeInSubgraph(const ComputeGraphPtr &graph, uint32_t in_tensor_idx,
                                      std::vector<std::shared_ptr<Anchor>> &target_in_data_anchors,
                                      uint32_t recursion_depth) const {
  GELOGD("FindDataNodeInSubgraph begin, graph: %s, recursion_depth: %u.", graph->GetName().c_str(), recursion_depth);
  if (recursion_depth >= kRecursionDepth) {
    GELOGI("FindDataNodeInSubgraph recursion_depth=%u > %u, auto exit", recursion_depth, kRecursionDepth);
    return SUCCESS;
  }
  // find the data node inside PARTITIONEDCALL
  // in_tensor_idx describe the idx of PARTITIONEDCALL input node
  for (const auto &node : graph->GetDirectNode()) {
    if (node->GetType() != DATA) {
      continue;
    }
    GE_CHECK_NOTNULL(node->GetOpDesc());
    uint32_t parent_node_index = UINT32_MAX;  // get parent_node_index
    if (!AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, parent_node_index)) {
      GELOGE(INTERNAL_ERROR, "During FindDataNodeInSubgraph node:%s, [Get][Attr]:%s failed parent_node_index:%u",
             node->GetName().c_str(), ATTR_NAME_PARENT_NODE_INDEX.c_str(), parent_node_index);
      return INTERNAL_ERROR;
    }
    if (parent_node_index != in_tensor_idx) {
      continue;
    }
    auto is_has_subgraph = InDataAnchorHasSubGraphs(node);
    if (!is_has_subgraph) {  // the size of subgraph of this data node is zero
      GELOGD("get target node: %s, recursion_depth: %u.", node->GetName().c_str(),
             recursion_depth);
      GE_ASSERT_NOTNULL(node->GetOutAnchor(0U));
      GE_ASSERT_NOTNULL(node->GetOutAnchor(0U)->GetPeerAnchorsPtr().at(0L));
      target_in_data_anchors.push_back(node->GetOutAnchor(0U)->GetPeerAnchors().at(0L));
      return SUCCESS;
    }
    // Recursive Processing
    for (const auto &next_node_and_anchor : node->GetOutDataNodesAndAnchors()) {
      const auto &next_node = next_node_and_anchor.first;
      const auto &in_data_anchor = next_node_and_anchor.second;
      if (next_node->GetType() != PARTITIONEDCALL) {
        continue;
      }
      GELOGD("partitioned_call: %s find in the function of FindDataNodeInSubgraph.", next_node->GetName().c_str());
      const auto &src_root_compute_graph = GraphUtils::FindRootGraph(next_node->GetOwnerComputeGraph());
      for (const auto &cur_sub_graph : src_root_compute_graph->GetAllSubgraphs()) {
        if (cur_sub_graph->GetParentNode()->GetName() != next_node->GetName()) {
          continue;
        }
        GE_ASSERT_SUCCESS(
            FindDataNodeInSubgraph(cur_sub_graph, in_data_anchor->GetIdx(), target_in_data_anchors, recursion_depth));
        GELOGD("Recursive Processing cur_sub_graph: %s, recursion_depth: %u.", cur_sub_graph->GetName().c_str(),
               recursion_depth);
      }
    }
  }
  return SUCCESS;
}

Status AippOp::FindInSubGraph(const std::shared_ptr<Node> &node,
                              std::vector<std::shared_ptr<Anchor>> &target_in_data_anchors) const {
  for (const auto &next_node_and_anchor : node->GetOutDataNodesAndAnchors()) {
    const auto &next_node = next_node_and_anchor.first;
    const auto &in_data_anchor = next_node_and_anchor.second;
    if (next_node->GetType() != PARTITIONEDCALL) {
      continue;
    }
    GELOGD("partitioned_call: %s find in FindInSubGraph.", next_node->GetName().c_str());
    const auto &src_root_compute_graph = GraphUtils::FindRootGraph(next_node->GetOwnerComputeGraph());
    for (const auto &cur_sub_graph : src_root_compute_graph->GetAllSubgraphs()) {
      uint32_t recursion_depth = 0U;
      if (cur_sub_graph->GetParentNode()->GetName() != next_node->GetName()) {
        continue;
      }
      GE_ASSERT_SUCCESS(
          FindDataNodeInSubgraph(cur_sub_graph, in_data_anchor->GetIdx(), target_in_data_anchors, recursion_depth));
      if (!target_in_data_anchors.empty()) {
        GE_ASSERT_NOTNULL(target_in_data_anchors.back()->GetFirstPeerAnchor());
        auto target_anchors =
            target_in_data_anchors.back()->GetFirstPeerAnchor()->GetOwnerNode();
        GELOGD("target_anchors size is %zu. return the first in_data_anchor: %s", target_in_data_anchors.size(),
               target_anchors->GetName().c_str());
        return SUCCESS;
      } else {
        return PARAM_INVALID;
      }
    }
  }
  return SUCCESS;
}

NodePtr AippOp::FindDataByIndex(const ComputeGraphPtr &graph, int32_t rank) const {
  int64_t data_index = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() != DATA) {
      continue;
    }
    // For functional multi batch, Skip Data for index.
    if (node->GetOpDesc()->HasAttr(ATTR_INSERT_BY_MBATCH)) {
      continue;
    }

    GE_ASSERT_TRUE(AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_INDEX, data_index),
                   "find data %s has no index attr", node->GetName().c_str());
    GELOGD("find data node:[%s] data_index:[%ld], rank is %u.", node->GetName().c_str(), data_index, rank);
    if (data_index != rank) {
      continue;
    }
    auto is_has_subgraph = InDataAnchorHasSubGraphs(node);
    if (is_has_subgraph) {
      std::vector<std::shared_ptr<Anchor>> target_in_data_anchors;
      auto ret = FindInSubGraph(node, target_in_data_anchors);
      if (ret == SUCCESS && !target_in_data_anchors.empty()) {
        GE_ASSERT_NOTNULL(target_in_data_anchors.front()->GetFirstPeerAnchor());
        return target_in_data_anchors.front()->GetFirstPeerAnchor()->GetOwnerNode();
      }
    }
    return node;
  }
  std::string error_msg = "Can not find the data node by aipp parameter related_input_rank " + to_string(rank);
  REPORT_PREDEFINED_ERR_MSG("E10052", std::vector<const char *>({"reason"}), std::vector<const char *>({error_msg.c_str()}));
  GELOGE(PARAM_INVALID, "[Check][aipp]%s", error_msg.c_str());
  return nullptr;
}

Status AippOp::GetAndCheckTarget(const ComputeGraphPtr &graph, int32_t rank, NodePtr &target,
                                 std::set<uint32_t> &edge_indexes) {
  (void)edge_indexes;
  auto data_node = FindDataByIndex(graph, rank);
  if (data_node == nullptr) {
    GELOGE(PARAM_INVALID, "[Call][FindDataByIndex] Get target input node for rank %d failed", rank);
    return PARAM_INVALID;
  }
  data_node_linked_aipp = data_node;
  auto data_opdesc = data_node->GetOpDesc();
  GE_CHECK_NOTNULL(data_opdesc);
  std::string set_dt_str;
  if (ge::AttrUtils::GetStr(data_opdesc, ATTR_ATC_USER_DEFINE_DATATYPE, set_dt_str)) {
    REPORT_PREDEFINED_ERR_MSG("E10034", std::vector<const char *>({"opname"}),
                              std::vector<const char *>({data_opdesc->GetName().c_str()}));
    GELOGE(INTERNAL_ERROR,
           "[Get][Attr] This input op [%s] is linked to aipp, can not be set to fp16, "
           "please check your atc parameter --insert_op_conf, --input_fp16_nodes.",
           data_opdesc->GetName().c_str());
    return PARAM_INVALID;
  }

  // add dynamic or static attr memsage to data
  if (GetAippMode() == domi::AippOpParams::static_) {
    (void)AttrUtils::SetStr(data_opdesc, ATTR_DATA_RELATED_AIPP_MODE, "static_aipp");
  } else if (GetAippMode() == domi::AippOpParams::dynamic) {
    (void)AttrUtils::SetStr(data_opdesc, ATTR_DATA_RELATED_AIPP_MODE, "dynamic_aipp");
  }

  // In scenario AIPP+CONV2D+POOLING, keep the aipp info to Data, since AIPP disappear after subgraph optimize
  NamedAttrs aipp_attr;
  ConvertParamToAttr(aipp_attr);
  if (!AttrUtils::SetNamedAttrs(data_opdesc, ATTR_NAME_AIPP, aipp_attr)) {
    REPORT_INNER_ERR_MSG("E19999", "Set Attr:%s for op:%s(%s) failed", ATTR_NAME_AIPP.c_str(),
                       data_opdesc->GetName().c_str(), data_opdesc->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Set][Attr] %s for op:%s(%s) failed", ATTR_NAME_AIPP.c_str(),
           data_opdesc->GetName().c_str(), data_opdesc->GetType().c_str());
    return INTERNAL_ERROR;
  }
  target = data_node;

  return GetStaticTargetNode(data_node, target);
}

Status AippOp::GetStaticTargetNode(NodePtr &data_node, NodePtr &target) {
  if (GetAippMode() != domi::AippOpParams::static_) {
    return SUCCESS;
  }

  const auto out_anchor = data_node->GetOutDataAnchor(0);
  GE_ASSERT_NOTNULL(out_anchor);
  for (const auto &in_anchor : out_anchor->GetPeerInDataAnchors()) {
    if (in_anchor == nullptr) {
      continue;
    }

    const auto &case_node = in_anchor->GetOwnerNode();
    if (case_node->GetType() == CASE) {
      target = case_node;
      return SUCCESS;
    }
  }

  return SUCCESS;
}
Status AippOp::ConvertRelatedInputNameToRank() {
  GE_CHECK_NOTNULL(aipp_params_);

  std::string related_input_name = aipp_params_->related_input_name();
  if (related_input_name.empty()) {
    return SUCCESS;
  }

  std::vector<std::string> data_tensor_names = domi::GetContext().data_tensor_names;
  GELOGI("Convert name to rank start: data size[%zu]", data_tensor_names.size());
  uint32_t index = 0;
  bool convert_flag = false;
  for (const auto &data_tensor_name : data_tensor_names) {
    if (related_input_name == data_tensor_name) {
      aipp_params_->set_related_input_rank(index);
      convert_flag = true;
      GELOGI("AippOp: rank: %u, top name: %s.", index, data_tensor_name.c_str());
      break;
    }
    index++;
  }
  if (!convert_flag) {
    std::string error_msg = "related_input_name " + related_input_name + "convert rank failed, Please ensure it is"
                            " caffe scene or related_input_name in aipp config is the top name of data node";
    GELOGE(PARAM_INVALID, "[Check][InputParam]%s", error_msg.c_str());
    REPORT_PREDEFINED_ERR_MSG("E10052", std::vector<const char_t *>({"reason"}),
                              std::vector<const char_t *>({error_msg.c_str()}));
    return PARAM_INVALID;
  }

  return SUCCESS;
}


Status AippOp::GetTargetPosition(ComputeGraphPtr graph, NodePtr &target_input,
                                 std::vector<std::pair<OutDataAnchorPtr, InDataAnchorPtr>> &target_edges) {
  GE_CHECK_NOTNULL(graph);
  GE_CHECK_NOTNULL(aipp_params_);

  std::set<uint32_t> edge_indexes;
  const uint32_t related_input_rank = aipp_params_->related_input_rank();
  auto ret = GetAndCheckTarget(graph, related_input_rank, target_input, edge_indexes);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Get][TargetInputNode] for rank %u failed", related_input_rank);
    return ret;
  }

  target_edges.clear();
  if (target_input->GetType() != CASE) {
    for (OutDataAnchorPtr &src_out : target_input->GetAllOutDataAnchors()) {
      auto dst_ins = src_out->GetPeerInDataAnchors();
      for (uint32_t i = 0; i < dst_ins.size(); ++i) {
        auto dst_in = dst_ins.at(i);
        if (edge_indexes.empty() || edge_indexes.count(i) > 0) {
          target_edges.emplace_back(src_out, dst_in);
        }
      }
    }
  } else {
    const auto &func_desc = target_input->GetOpDesc();
    for (const auto &name : func_desc->GetSubgraphInstanceNames()) {
      const auto &subgraph = graph->GetSubgraph(name);
      if (subgraph == nullptr) {
        REPORT_INNER_ERR_MSG("E19999", "Subgraph:%s of op:%s(%s) not find in graph:%s, check invalid",
                           name.c_str(), func_desc->GetName().c_str(), func_desc->GetType().c_str(),
                           graph->GetName().c_str());
        GELOGE(GE_GRAPH_EMPTY_SUBGRAPH, "[Get][Subgraph] failed, Subgraph:%s of op:%s(%s) not find in graph:%s",
               name.c_str(), func_desc->GetName().c_str(), func_desc->GetType().c_str(), graph->GetName().c_str());
        return GE_GRAPH_EMPTY_SUBGRAPH;
      }

      auto data_node = FindDataByIndex(subgraph, related_input_rank);
      if (data_node == nullptr) {
        GELOGE(PARAM_INVALID, "[Get][TargetInputNode] for rank %d failed", related_input_rank);
        return PARAM_INVALID;
      }

      for (OutDataAnchorPtr &src_out : data_node->GetAllOutDataAnchors()) {
        auto dst_ins = src_out->GetPeerInDataAnchors();
        for (uint32_t i = 0; i < dst_ins.size(); ++i) {
          auto dst_in = dst_ins.at(i);
          if (edge_indexes.empty() || edge_indexes.count(i) > 0) {
            target_edges.emplace_back(src_out, dst_in);
          }
        }
      }
    }
  }

  return SUCCESS;
}

Status AippOp::SetDefaultParams() {
  GE_CHECK_NOTNULL(aipp_params_);
  const domi::AippOpParams::AippMode aipp_mode = aipp_params_->aipp_mode();
  if (aipp_mode == domi::AippOpParams::static_) {
    if (aipp_params_->csc_switch()) {
      SetCscDefaultValue();
    }

    SetDtcDefaultValue();

    GELOGI("parse aipp params:input_format:%s, csc_switch:%d.",
           domi::AippOpParams::InputFormat_Name(aipp_params_->input_format()).c_str(), aipp_params_->csc_switch());

    GELOGI("parse aipp params:mean_chn_0:%d, mean_chn_1:%d, mean_chn_2:%d, mean_chn_3:%d.", aipp_params_->mean_chn_0(),
           aipp_params_->mean_chn_1(), aipp_params_->mean_chn_2(), aipp_params_->mean_chn_3());

    GELOGI("parse aipp params:min_chn_0:%f, min_chn_1:%f, min_chn_2:%f.", aipp_params_->min_chn_0(),
           aipp_params_->min_chn_1(), aipp_params_->min_chn_2());

    GE_IF_BOOL_EXEC(!aipp_params_->crop(),
                    aipp_params_->set_load_start_pos_h(0);
                    aipp_params_->set_load_start_pos_w(0);
                    aipp_params_->set_crop_size_h(0);
                    aipp_params_->set_crop_size_w(0);
                    );

    GE_IF_BOOL_EXEC(!aipp_params_->resize(),
                    aipp_params_->set_resize_output_h(0);
                    aipp_params_->set_resize_output_w(0);
                    );

    GE_IF_BOOL_EXEC(!aipp_params_->padding(),
                    aipp_params_->set_left_padding_size(0);
                    aipp_params_->set_right_padding_size(0);
                    aipp_params_->set_top_padding_size(0);
                    aipp_params_->set_bottom_padding_size(0);
                    );
  }

  return SUCCESS;
}

Status AippOp::ValidateParams() {
  GE_CHECK_NOTNULL(aipp_params_);
  GE_CHK_LOG_AND_ERRORMSG(((aipp_params_->aipp_mode() == domi::AippOpParams::static_) ||
                          (aipp_params_->aipp_mode() == domi::AippOpParams::dynamic)), PARAM_INVALID,
                          "When insert AIPP op, aipp_mode must be configured as static or dynamic");

  GE_CHK_LOG_AND_ERRORMSG(aipp_params_->var_reci_chn_0_size() <= 1, PARAM_INVALID,
                          "The parameter var_reci_chn_0 can not be configured repeatedly");
  GE_CHK_LOG_AND_ERRORMSG(aipp_params_->var_reci_chn_1_size() <= 1, PARAM_INVALID,
                          "The parameter var_reci_chn_1 can not be configured repeatedly");
  GE_CHK_LOG_AND_ERRORMSG(aipp_params_->var_reci_chn_2_size() <= 1, PARAM_INVALID,
                          "The parameter var_reci_chn_2 can not be configured repeatedly");
  GE_CHK_LOG_AND_ERRORMSG(aipp_params_->var_reci_chn_3_size() <= 1, PARAM_INVALID,
                          "The parameter var_reci_chn_3 can not be configured repeatedly");

  GE_CHK_LOG_AND_ERRORMSG(aipp_params_->matrix_r0c0_size() <= 1, PARAM_INVALID,
                          "The parameter matrix_r0c0 can not be configured repeatedly");
  GE_CHK_LOG_AND_ERRORMSG(aipp_params_->matrix_r0c1_size() <= 1, PARAM_INVALID,
                          "The parameter matrix_r0c1 can not be configured repeatedly");
  GE_CHK_LOG_AND_ERRORMSG(aipp_params_->matrix_r0c2_size() <= 1, PARAM_INVALID,
                          "The parameter matrix_r0c2 can not be configured repeatedly");

  GE_CHK_LOG_AND_ERRORMSG(aipp_params_->matrix_r1c0_size() <= 1, PARAM_INVALID,
                          "The parameter matrix_r1c0 can not be configured repeatedly");
  GE_CHK_LOG_AND_ERRORMSG(aipp_params_->matrix_r1c1_size() <= 1, PARAM_INVALID,
                          "The parameter matrix_r1c1 can not be configured repeatedly");
  GE_CHK_LOG_AND_ERRORMSG(aipp_params_->matrix_r1c2_size() <= 1, PARAM_INVALID,
                          "The parameter matrix_r1c2 can not be configured repeatedly");

  GE_CHK_LOG_AND_ERRORMSG(aipp_params_->matrix_r2c0_size() <= 1, PARAM_INVALID,
                          "The parameter matrix_r2c0 can not be configured repeatedly");
  GE_CHK_LOG_AND_ERRORMSG(aipp_params_->matrix_r2c1_size() <= 1, PARAM_INVALID,
                          "The parameter matrix_r2c1 can not be configured repeatedly");
  GE_CHK_LOG_AND_ERRORMSG(aipp_params_->matrix_r2c2_size() <= 1, PARAM_INVALID,
                          "The parameter matrix_r2c2 can not be configured repeatedly");

  GE_CHK_LOG_AND_ERRORMSG(aipp_params_->output_bias_0_size() <= 1, PARAM_INVALID,
                          "The parameter output_bias_0 can not be configured repeatedly");
  GE_CHK_LOG_AND_ERRORMSG(aipp_params_->output_bias_1_size() <= 1, PARAM_INVALID,
                          "The parameter output_bias_1 can not be configured repeatedly");
  GE_CHK_LOG_AND_ERRORMSG(aipp_params_->output_bias_2_size() <= 1, PARAM_INVALID,
                          "The parameter output_bias_2 can not be configured repeatedly");

  GE_CHK_LOG_AND_ERRORMSG(aipp_params_->input_bias_0_size() <= 1, PARAM_INVALID,
                          "The parameter input_bias_0 can not be configured repeatedly");
  GE_CHK_LOG_AND_ERRORMSG(aipp_params_->input_bias_1_size() <= 1, PARAM_INVALID,
                          "The parameter input_bias_1 can not be configured repeatedly");
  GE_CHK_LOG_AND_ERRORMSG(aipp_params_->input_bias_2_size() <= 1, PARAM_INVALID,
                          "The parameter input_bias_2 can not be configured repeatedly");

  const domi::AippOpParams::AippMode aipp_mode = aipp_params_->aipp_mode();
  if (aipp_mode == domi::AippOpParams::dynamic) {
    GE_CHK_LOG_AND_ERRORMSG(
        aipp_params_->max_src_image_size() > 0, PARAM_INVALID,
        "For dynamic AIPP params, max_src_image_size must be set which number should be greater than 0");
  } else {
    GE_CHK_LOG_AND_ERRORMSG(aipp_params_->input_format() != domi::AippOpParams::UNDEFINED, PARAM_INVALID,
                            "Input format of AIPP conf must be set in static aipp");

    GE_CHK_LOG_AND_ERRORMSG(aipp_params_->src_image_size_w() >= 0, PARAM_INVALID,
                            "Src_image_size_w must not be configured smaller than 0");
    GE_CHK_LOG_AND_ERRORMSG(aipp_params_->src_image_size_h() >= 0, PARAM_INVALID,
                            "Src_image_size_h must not be configured smaller than 0");
    GE_CHK_LOG_AND_ERRORMSG(aipp_params_->load_start_pos_w() >= 0, PARAM_INVALID,
                            "Load_start_pos_w must not be configured smaller than 0");
    GE_CHK_LOG_AND_ERRORMSG(aipp_params_->load_start_pos_h() >= 0, PARAM_INVALID,
                            "Load_start_pos_h must not be configured smaller than 0");
    GE_CHK_LOG_AND_ERRORMSG(aipp_params_->crop_size_w() >= 0, PARAM_INVALID,
                            "Crop_size_w must not be configured smaller than 0");
    GE_CHK_LOG_AND_ERRORMSG(aipp_params_->resize_output_w() >= 0, PARAM_INVALID,
                            "Resize_output_w must not be configured smaller than 0");
    GE_CHK_LOG_AND_ERRORMSG(aipp_params_->resize_output_h() >= 0, PARAM_INVALID,
                            "Resize_output_h must not be configured smaller than 0");
    GE_CHK_LOG_AND_ERRORMSG(aipp_params_->left_padding_size() >= 0, PARAM_INVALID,
                            "Left_padding_size must not be configured smaller than 0");
    GE_CHK_LOG_AND_ERRORMSG(aipp_params_->right_padding_size() >= 0, PARAM_INVALID,
                            "Right_padding_size must not be configured smaller than 0");
    GE_CHK_LOG_AND_ERRORMSG(aipp_params_->top_padding_size() >= 0, PARAM_INVALID,
                            "Top_padding_size must not be configured smaller than 0");
    GE_CHK_LOG_AND_ERRORMSG(aipp_params_->bottom_padding_size() >= 0, PARAM_INVALID,
                            "Bottom_padding_size must not be configured smaller than 0");
  }

  return SUCCESS;
}

void AippOp::SetCscDefaultValue() {
  GE_CHECK_NOTNULL_JUST_RETURN(aipp_params_);
  if (aipp_params_->input_format() == domi::AippOpParams::YUV420SP_U8) {
    CHECK_FALSE_EXEC(aipp_params_->matrix_r0c0_size() > 0, aipp_params_->add_matrix_r0c0(DEFAULT_MATRIX_R2C0_YUV2RGB));
    CHECK_FALSE_EXEC(aipp_params_->matrix_r0c1_size() > 0, aipp_params_->add_matrix_r0c1(DEFAULT_MATRIX_R2C1_YUV2RGB));
    CHECK_FALSE_EXEC(aipp_params_->matrix_r0c2_size() > 0, aipp_params_->add_matrix_r0c2(DEFAULT_MATRIX_R2C2_YUV2RGB));
    CHECK_FALSE_EXEC(aipp_params_->matrix_r1c0_size() > 0, aipp_params_->add_matrix_r1c0(DEFAULT_MATRIX_R1C0_YUV2RGB));
    CHECK_FALSE_EXEC(aipp_params_->matrix_r1c1_size() > 0, aipp_params_->add_matrix_r1c1(DEFAULT_MATRIX_R1C1_YUV2RGB));
    CHECK_FALSE_EXEC(aipp_params_->matrix_r1c2_size() > 0, aipp_params_->add_matrix_r1c2(DEFAULT_MATRIX_R1C2_YUV2RGB));
    CHECK_FALSE_EXEC(aipp_params_->matrix_r2c0_size() > 0, aipp_params_->add_matrix_r2c0(DEFAULT_MATRIX_R0C0_YUV2RGB));
    CHECK_FALSE_EXEC(aipp_params_->matrix_r2c1_size() > 0, aipp_params_->add_matrix_r2c1(DEFAULT_MATRIX_R0C1_YUV2RGB));
    CHECK_FALSE_EXEC(aipp_params_->matrix_r2c2_size() > 0, aipp_params_->add_matrix_r2c2(DEFAULT_MATRIX_R0C2_YUV2RGB));
  } else {
    CHECK_FALSE_EXEC(aipp_params_->matrix_r0c0_size() > 0, aipp_params_->add_matrix_r0c0(DEFAULT_MATRIX_R0C0_RGB2YUV));
    CHECK_FALSE_EXEC(aipp_params_->matrix_r0c1_size() > 0, aipp_params_->add_matrix_r0c1(DEFAULT_MATRIX_R0C1_RGB2YUV));
    CHECK_FALSE_EXEC(aipp_params_->matrix_r0c2_size() > 0, aipp_params_->add_matrix_r0c2(DEFAULT_MATRIX_R0C2_RGB2YUV));
    CHECK_FALSE_EXEC(aipp_params_->matrix_r1c0_size() > 0, aipp_params_->add_matrix_r1c0(DEFAULT_MATRIX_R1C0_RGB2YUV));
    CHECK_FALSE_EXEC(aipp_params_->matrix_r1c1_size() > 0, aipp_params_->add_matrix_r1c1(DEFAULT_MATRIX_R1C1_RGB2YUV));
    CHECK_FALSE_EXEC(aipp_params_->matrix_r1c2_size() > 0, aipp_params_->add_matrix_r1c2(DEFAULT_MATRIX_R1C2_RGB2YUV));
    CHECK_FALSE_EXEC(aipp_params_->matrix_r2c0_size() > 0, aipp_params_->add_matrix_r2c0(DEFAULT_MATRIX_R2C0_RGB2YUV));
    CHECK_FALSE_EXEC(aipp_params_->matrix_r2c1_size() > 0, aipp_params_->add_matrix_r2c1(DEFAULT_MATRIX_R2C1_RGB2YUV));
    CHECK_FALSE_EXEC(aipp_params_->matrix_r2c2_size() > 0, aipp_params_->add_matrix_r2c2(DEFAULT_MATRIX_R2C2_RGB2YUV));
  }
  CHECK_FALSE_EXEC(aipp_params_->input_bias_0_size() > 0, aipp_params_->add_input_bias_0(DEFAULT_INPUT_BIAS_0));
  CHECK_FALSE_EXEC(aipp_params_->input_bias_1_size() > 0, aipp_params_->add_input_bias_1(DEFAULT_INPUT_BIAS_1));
  CHECK_FALSE_EXEC(aipp_params_->input_bias_2_size() > 0, aipp_params_->add_input_bias_2(DEFAULT_INPUT_BIAS_2));
  CHECK_FALSE_EXEC(aipp_params_->output_bias_0_size() > 0, aipp_params_->add_output_bias_0(DEFAULT_OUTPUT_BIAS_0));
  CHECK_FALSE_EXEC(aipp_params_->output_bias_1_size() > 0, aipp_params_->add_output_bias_1(DEFAULT_OUTPUT_BIAS_1));
  CHECK_FALSE_EXEC(aipp_params_->output_bias_2_size() > 0, aipp_params_->add_output_bias_2(DEFAULT_OUTPUT_BIAS_2));
}

void AippOp::SetDtcDefaultValue() {
  GE_CHECK_NOTNULL_JUST_RETURN(aipp_params_);
  CHECK_FALSE_EXEC(aipp_params_->var_reci_chn_0_size() > 0, aipp_params_->add_var_reci_chn_0(DEFAULT_VAR_RECI_CHN));
  GELOGD("var_reci_chn_0 is %f, size is %u.", DEFAULT_VAR_RECI_CHN, aipp_params_->var_reci_chn_0_size());
  CHECK_FALSE_EXEC(aipp_params_->var_reci_chn_1_size() > 0, aipp_params_->add_var_reci_chn_1(DEFAULT_VAR_RECI_CHN));
  GELOGD("var_reci_chn_1 is %f, size is %u.", DEFAULT_VAR_RECI_CHN, aipp_params_->var_reci_chn_1_size());
  CHECK_FALSE_EXEC(aipp_params_->var_reci_chn_2_size() > 0, aipp_params_->add_var_reci_chn_2(DEFAULT_VAR_RECI_CHN));
  GELOGD("var_reci_chn_2 is %f, size is %u.", DEFAULT_VAR_RECI_CHN, aipp_params_->var_reci_chn_2_size());
  CHECK_FALSE_EXEC(aipp_params_->var_reci_chn_3_size() > 0, aipp_params_->add_var_reci_chn_3(DEFAULT_VAR_RECI_CHN));
  GELOGD("var_reci_chn_3 is %f, size is %u.", DEFAULT_VAR_RECI_CHN, aipp_params_->var_reci_chn_3_size());
}

Status AippOp::GenerateOpDesc(OpDescPtr op_desc) {
  GE_CHECK_NOTNULL(op_desc);

  static std::atomic_long atomic_op_idx(0);
  auto op_idx = atomic_op_idx.fetch_add(1);
  op_desc->SetName(std::string("aipp_node").append(std::to_string(op_idx)));
  ge::OpDescUtilsEx::SetType(op_desc, AIPP);

  // Add two InputDesc, add the second after the first one is added successfully.
  if ((op_desc->AddInputDesc(GeTensorDesc()) != GRAPH_SUCCESS) ||
      (op_desc->AddInputDesc(GeTensorDesc()) != GRAPH_SUCCESS)) {
    REPORT_INNER_ERR_MSG("E19999", "Add input desc into op:%s(%s) failed",
                      op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(FAILED, "[Add][InputDesc] into op:%s(%s) failed",
           op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return FAILED;
  }

  if (op_desc->AddOutputDesc(GeTensorDesc()) != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Add output desc into op:%s(%s) failed",
                      op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(FAILED, "[Add][OutputDesc] into op:%s(%s) failed",
           op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return FAILED;
  }
  NamedAttrs aipp_attrs;
  ConvertParamToAttr(aipp_attrs);

  GE_IF_BOOL_EXEC(!AttrUtils::SetNamedAttrs(op_desc, ATTR_NAME_AIPP, aipp_attrs),
                  REPORT_INNER_ERR_MSG("E19999", "Set Attr:%s to op:%s(%s) failed", ATTR_NAME_AIPP.c_str(),
                                     op_desc->GetName().c_str(), op_desc->GetType().c_str());
                  GELOGE(FAILED, "[Set][Attr] %s to op:%s(%s) failed", ATTR_NAME_AIPP.c_str(),
                         op_desc->GetName().c_str(), op_desc->GetType().c_str());
                  return FAILED);

  return SUCCESS;
}

void AippOp::SetInputFormat(nlohmann::json &cfg_json) const {
  auto input_format = aipp_params_->input_format();
  const auto iter = kInputFormatMap.find(input_format);
  if (iter != kInputFormatMap.cend()) {
    cfg_json["input_format"] = iter->second;
  } else {
    GELOGW("Input format of config param in aipp node is invalid.");
  }
}

void AippOp::SetAippMode(nlohmann::json &cfg_json) const {
  auto aipp_mode = aipp_params_->aipp_mode();
  const auto iter = kAippModeMap.find(aipp_mode);
  if (iter != kAippModeMap.cend()) {
    cfg_json["aipp_mode"] = iter->second;
  } else {
    GELOGW("Aipp mode of config param in aipp node is invalid.");
  }
}

void AippOp::SetMatrix(nlohmann::json &cfg_json) const {
  // matrix_r0c0 repeated int32
  if (aipp_params_->matrix_r0c0_size() > 0) {
    cfg_json["matrix_r0c0"] = aipp_params_->matrix_r0c0(0);
  }
  // matrix_r0c1 repeated int32
  if (aipp_params_->matrix_r0c1_size() > 0) {
    cfg_json["matrix_r0c1"] = aipp_params_->matrix_r0c1(0);
  }
  // matrix_r0c2 repeated int32
  if (aipp_params_->matrix_r0c2_size() > 0) {
    cfg_json["matrix_r0c2"] = aipp_params_->matrix_r0c2(0);
  }
  // matrix_r1c0 repeated int32
  if (aipp_params_->matrix_r1c0_size() > 0) {
    cfg_json["matrix_r1c0"] = aipp_params_->matrix_r1c0(0);
  }
  // matrix_r1c1 repeated int32
  if (aipp_params_->matrix_r1c1_size() > 0) {
    cfg_json["matrix_r1c1"] = aipp_params_->matrix_r1c1(0);
  }
  // matrix_r1c2 repeated int32
  if (aipp_params_->matrix_r1c2_size() > 0) {
    cfg_json["matrix_r1c2"] = aipp_params_->matrix_r1c2(0);
  }
  // matrix_r2c0 repeated int32
  if (aipp_params_->matrix_r2c0_size() > 0) {
    cfg_json["matrix_r2c0"] = aipp_params_->matrix_r2c0(0);
  }
  // matrix_r2c1 repeated int32
  if (aipp_params_->matrix_r2c1_size() > 0) {
    cfg_json["matrix_r2c1"] = aipp_params_->matrix_r2c1(0);
  }
  // matrix_r2c2 repeated int32
  if (aipp_params_->matrix_r2c2_size() > 0) {
    cfg_json["matrix_r2c2"] = aipp_params_->matrix_r2c2(0);
  }
}

void AippOp::SetInputBias(nlohmann::json& cfg_json) const {
  // input_bias_0 repeated int32
  if (aipp_params_->input_bias_0_size() > 0) {
    cfg_json["input_bias_0"] = aipp_params_->input_bias_0(0);
  }
  // input_bias_1 repeated int32
  if (aipp_params_->input_bias_1_size() > 0) {
    cfg_json["input_bias_1"] = aipp_params_->input_bias_1(0);
  }
  // input_bias_2 repeated int32
  if (aipp_params_->input_bias_2_size() > 0) {
    cfg_json["input_bias_2"] = aipp_params_->input_bias_2(0);
  }
}

void AippOp::SetOutputBias(nlohmann::json& cfg_json) const {
  // output_bias_0 repeated int32
  if (aipp_params_->output_bias_0_size() > 0) {
    cfg_json["output_bias_0"] = aipp_params_->output_bias_0(0);
  }
  // output_bias_1 repeated int32
  if (aipp_params_->output_bias_1_size() > 0) {
    cfg_json["output_bias_1"] = aipp_params_->output_bias_1(0);
  }
  // output_bias_2 repeated int32
  if (aipp_params_->output_bias_2_size() > 0) {
    cfg_json["output_bias_2"] = aipp_params_->output_bias_2(0);
  }
}

void AippOp::SetVarReciChn(nlohmann::json& cfg_json) const {
  if (aipp_params_->var_reci_chn_0_size() > 0) {
    cfg_json["var_reci_chn_0"] = aipp_params_->var_reci_chn_0(0);
  }
  if (aipp_params_->var_reci_chn_1_size() > 0) {
    cfg_json["var_reci_chn_1"] = aipp_params_->var_reci_chn_1(0);
  }
    if (aipp_params_->var_reci_chn_2_size() > 0) {
    cfg_json["var_reci_chn_2"] = aipp_params_->var_reci_chn_2(0);
  }
    if (aipp_params_->var_reci_chn_3_size() > 0) {
    cfg_json["var_reci_chn_3"] = aipp_params_->var_reci_chn_3(0);
  }
}

void AippOp::SetCropData(nlohmann::json& cfg_json) const {
  if (aipp_params_->crop()) {
    cfg_json["crop"] = aipp_params_->crop();
  }
  if (aipp_params_->load_start_pos_w() != 0) {
    cfg_json["load_start_pos_w"] = aipp_params_->load_start_pos_w();
  }
  if (aipp_params_->load_start_pos_h() != 0) {
    cfg_json["load_start_pos_h"] = aipp_params_->load_start_pos_h();
  }
  if (aipp_params_->crop_size_w() != 0) {
    cfg_json["crop_size_w"] = aipp_params_->crop_size_w();
  }
  if (aipp_params_->crop_size_h() != 0) {
    cfg_json["crop_size_h"] = aipp_params_->crop_size_h();
  }
}

void AippOp::SetOtherStaticData(nlohmann::json& cfg_json) const {
  if (aipp_params_->csc_switch()) {
    cfg_json["csc_switch"] = aipp_params_->csc_switch();
  }
  if (aipp_params_->rbuv_swap_switch()) {
    cfg_json["rbuv_swap_switch"] = aipp_params_->rbuv_swap_switch();
  }
  if (aipp_params_->ax_swap_switch()) {
    cfg_json["ax_swap_switch"] = aipp_params_->ax_swap_switch();
  }
  if (aipp_params_->src_image_size_w() != 0) {
    cfg_json["src_image_size_w"] = aipp_params_->src_image_size_w();
  }
  if (aipp_params_->src_image_size_h() != 0) {
    cfg_json["src_image_size_h"] = aipp_params_->src_image_size_h();
  }
}

void AippOp::SetPaddingData(nlohmann::json& cfg_json) const {
  if (aipp_params_->padding()) {
    cfg_json["padding"] = aipp_params_->padding();
    if (abs(aipp_params_->padding_value()) >= EPSINON) {
      cfg_json["padding_value"] = aipp_params_->padding_value();
    }
  }
  if (aipp_params_->left_padding_size() != 0) {
    cfg_json["left_padding_size"] = aipp_params_->left_padding_size();
  }
  if (aipp_params_->right_padding_size() != 0) {
    cfg_json["right_padding_size"] = aipp_params_->right_padding_size();
  }
  if (aipp_params_->top_padding_size() != 0) {
    cfg_json["top_padding_size"] = aipp_params_->top_padding_size();
  }
  if (aipp_params_->bottom_padding_size() != 0) {
    cfg_json["bottom_padding_size"] = aipp_params_->bottom_padding_size();
  }
}

void AippOp::SetChnData(nlohmann::json& cfg_json) const {
  if (aipp_params_->mean_chn_0() != 0) {
    cfg_json["mean_chn_0"] = aipp_params_->mean_chn_0();
  }
  if (aipp_params_->mean_chn_1() != 0) {
    cfg_json["mean_chn_1"] = aipp_params_->mean_chn_1();
  }
  if (aipp_params_->mean_chn_2() != 0) {
    cfg_json["mean_chn_2"] = aipp_params_->mean_chn_2();
  }
  if (aipp_params_->mean_chn_3() != 0) {
    cfg_json["mean_chn_3"] = aipp_params_->mean_chn_3();
  }
  if (abs(aipp_params_->min_chn_0()) >= EPSINON) {
    cfg_json["min_chn_0"] = aipp_params_->min_chn_0();
  }
  if (abs(aipp_params_->min_chn_1()) >= EPSINON) {
    cfg_json["min_chn_1"] = aipp_params_->min_chn_1();
  }
  if (abs(aipp_params_->min_chn_2()) >= EPSINON) {
    cfg_json["min_chn_2"] = aipp_params_->min_chn_2();
  }
  if (abs(aipp_params_->min_chn_3()) >= EPSINON) {
    cfg_json["min_chn_3"] = aipp_params_->min_chn_3();
  }
}

void AippOp::SetResizeData(nlohmann::json& cfg_json) const {
  if (aipp_params_->resize()) {
    cfg_json["resize"] = aipp_params_->resize();
  }
  if (aipp_params_->resize_output_w() != 0) {
    cfg_json["resize_output_w"] = aipp_params_->resize_output_w();
  }
  if (aipp_params_->resize_output_h() != 0) {
    cfg_json["resize_output_h"] = aipp_params_->resize_output_h();
  }
}

std::string AippOp::ConvertParamToJson() const {
  nlohmann::json cfg_json;
  SetAippMode(cfg_json);
  if (aipp_params_->related_input_rank() != 0U) {
    cfg_json["related_input_rank"] = aipp_params_->related_input_rank();
  }
  if (aipp_params_->aipp_mode() == domi::AippOpParams::static_) {
    SetInputFormat(cfg_json);
    SetCropData(cfg_json);
    SetResizeData(cfg_json);
    SetPaddingData(cfg_json);
    SetChnData(cfg_json);
    SetVarReciChn(cfg_json);
    SetMatrix(cfg_json);
    SetInputBias(cfg_json);
    SetOutputBias(cfg_json);
    SetOtherStaticData(cfg_json);
  }
  if (aipp_params_->aipp_mode() == domi::AippOpParams::dynamic) {
    if (aipp_params_->max_src_image_size() != 0U) {
      cfg_json["max_src_image_size"] = aipp_params_->max_src_image_size();
    }
  }

  if (aipp_params_->raw_rgbir_to_f16_n() != 0U) {
    cfg_json["raw_rgbir_to_f16_n"] = aipp_params_->raw_rgbir_to_f16_n();
  }
  return cfg_json.dump();
}

void AippOp::ConvertParamToAttr(NamedAttrs &aipp_attrs) {
  GE_CHECK_NOTNULL_JUST_RETURN(aipp_params_);
  SAVE_AIPP_ATTR(aipp_attrs, aipp_mode, int64_t);
  SAVE_AIPP_ATTR(aipp_attrs, related_input_rank, int64_t);

  if (aipp_params_->aipp_mode() == domi::AippOpParams::static_) {
    SAVE_AIPP_ATTR(aipp_attrs, input_format, int64_t);
    SAVE_AIPP_ATTR(aipp_attrs, csc_switch, bool);
    SAVE_AIPP_ATTR(aipp_attrs, crop, bool);
    SAVE_AIPP_ATTR(aipp_attrs, resize, bool);
    SAVE_AIPP_ATTR(aipp_attrs, load_start_pos_w, int64_t);
    SAVE_AIPP_ATTR(aipp_attrs, load_start_pos_h, int64_t);
    SAVE_AIPP_ATTR(aipp_attrs, crop_size_w, int64_t);
    SAVE_AIPP_ATTR(aipp_attrs, crop_size_h, int64_t);
    SAVE_AIPP_ATTR(aipp_attrs, resize, bool);
    SAVE_AIPP_ATTR(aipp_attrs, resize_output_w, int64_t);
    SAVE_AIPP_ATTR(aipp_attrs, resize_output_h, int64_t);
    SAVE_AIPP_ATTR(aipp_attrs, padding, bool);
    SAVE_AIPP_ATTR(aipp_attrs, left_padding_size, int64_t);
    SAVE_AIPP_ATTR(aipp_attrs, right_padding_size, int64_t);
    SAVE_AIPP_ATTR(aipp_attrs, top_padding_size, int64_t);
    SAVE_AIPP_ATTR(aipp_attrs, bottom_padding_size, int64_t);
    SAVE_AIPP_ATTR(aipp_attrs, src_image_size_w, int64_t);
    SAVE_AIPP_ATTR(aipp_attrs, src_image_size_h, int64_t);
    SAVE_AIPP_ATTR(aipp_attrs, cpadding_value, float);
    SAVE_AIPP_ATTR(aipp_attrs, rbuv_swap_switch, bool);
    SAVE_AIPP_ATTR(aipp_attrs, ax_swap_switch, bool);
    SAVE_AIPP_ATTR(aipp_attrs, single_line_mode, bool);
    SAVE_AIPP_ATTR(aipp_attrs, mean_chn_0, int64_t);
    SAVE_AIPP_ATTR(aipp_attrs, mean_chn_1, int64_t);
    SAVE_AIPP_ATTR(aipp_attrs, mean_chn_2, int64_t);
    SAVE_AIPP_ATTR(aipp_attrs, mean_chn_3, int64_t);
    SAVE_AIPP_ATTR(aipp_attrs, min_chn_0, float);
    SAVE_AIPP_ATTR(aipp_attrs, min_chn_1, float);
    SAVE_AIPP_ATTR(aipp_attrs, min_chn_2, float);
    SAVE_AIPP_ATTR(aipp_attrs, min_chn_3, float);
    SAVE_AIPP_ATTR_LIST(aipp_attrs, var_reci_chn_0, float);
    SAVE_AIPP_ATTR_LIST(aipp_attrs, var_reci_chn_1, float);
    SAVE_AIPP_ATTR_LIST(aipp_attrs, var_reci_chn_2, float);
    SAVE_AIPP_ATTR_LIST(aipp_attrs, var_reci_chn_3, float);
    SAVE_AIPP_ATTR_LIST(aipp_attrs, matrix_r0c0, int64_t);
    SAVE_AIPP_ATTR_LIST(aipp_attrs, matrix_r0c1, int64_t);
    SAVE_AIPP_ATTR_LIST(aipp_attrs, matrix_r0c2, int64_t);
    SAVE_AIPP_ATTR_LIST(aipp_attrs, matrix_r1c0, int64_t);
    SAVE_AIPP_ATTR_LIST(aipp_attrs, matrix_r1c1, int64_t);
    SAVE_AIPP_ATTR_LIST(aipp_attrs, matrix_r1c2, int64_t);
    SAVE_AIPP_ATTR_LIST(aipp_attrs, matrix_r2c0, int64_t);
    SAVE_AIPP_ATTR_LIST(aipp_attrs, matrix_r2c1, int64_t);
    SAVE_AIPP_ATTR_LIST(aipp_attrs, matrix_r2c2, int64_t);
    SAVE_AIPP_ATTR_LIST(aipp_attrs, output_bias_0, int64_t);
    SAVE_AIPP_ATTR_LIST(aipp_attrs, output_bias_1, int64_t);
    SAVE_AIPP_ATTR_LIST(aipp_attrs, output_bias_2, int64_t);
    SAVE_AIPP_ATTR_LIST(aipp_attrs, input_bias_0, int64_t);
    SAVE_AIPP_ATTR_LIST(aipp_attrs, input_bias_1, int64_t);
    SAVE_AIPP_ATTR_LIST(aipp_attrs, input_bias_2, int64_t);
  } else {
    SAVE_AIPP_ATTR(aipp_attrs, max_src_image_size, int64_t);
    SAVE_AIPP_ATTR(aipp_attrs, support_rotation, bool);
  }
}
Status AippOp::CreateAippData(const NodePtr &aipp_node) {
  GELOGD("Enter add aipp data node process.");
  // get previous node, it should be DATA
  auto data_node = aipp_node->GetInDataNodes().at(kAippImageInputIndex);
  auto data_op_desc = data_node->GetOpDesc();
  GE_CHECK_NOTNULL(data_op_desc);

  auto ori_data_format = GetAndCheckFormat();
  if (ori_data_format != FORMAT_NCHW && ori_data_format != FORMAT_NHWC) {
    std::string format_str = TypeUtils::FormatToSerialString(ori_data_format);
    GELOGE(PARAM_INVALID, "[Check][Param] when dynamic aipp, input_format must be NCHW or NHWC, but [%s] format is %s",
           data_node->GetName().c_str(), format_str.c_str());
    std::string reason = "The format must be NCHW or NHWC in the dynamic AIPP scenario";
    REPORT_PREDEFINED_ERR_MSG(
        "E13014", std::vector<const char *>({"opname", "parameter", "value", "reason"}),
        std::vector<const char *>({data_node->GetName().c_str(), "format", format_str.c_str(), reason.c_str()}));
    return PARAM_INVALID;
  }

  // dynamic aipp shape HWC is not fixed, need to be set -1
  int64_t data_shape_n = 0;
  // dynamic batch or HW, need acquire N from ATTR_MBATCH_ORIGIN_INPUT_DIMS
  if (data_op_desc->HasAttr(ATTR_MBATCH_ORIGIN_INPUT_DIMS)) {
    std::vector<int64_t> origin_input_dims;
    (void)AttrUtils::GetListInt(data_op_desc, ATTR_MBATCH_ORIGIN_INPUT_DIMS, origin_input_dims);
    if (!origin_input_dims.empty()) {
      data_shape_n = origin_input_dims[0];
    }
  } else {
    const auto &input_desc = data_op_desc->MutableInputDesc(0);
    GE_ASSERT_NOTNULL(input_desc);
    data_shape_n = input_desc->GetShape().GetDim(0);
  }
  std::vector<int64_t> dynamic_aipp_linked_data_shape{data_shape_n, kDynamicDim, kDynamicDim, kDynamicDim};
  (void)AttrUtils::SetListInt(data_op_desc, ATTR_DYNAMIC_AIPP_INPUT_DIMS, dynamic_aipp_linked_data_shape);

  int64_t batch_count = -1;
  if (GetDataDimN(data_node, ori_data_format, batch_count) != ge::SUCCESS) {
    std::string error_msg = "Get data_node dims and transfer to nchw_dims failed!";
    GE_ERRORLOG_AND_ERRORMSG(PARAM_INVALID, error_msg.c_str());
    return PARAM_INVALID;
  }
  int64_t max_dynamic_aipp_size = -1;
  if (batch_count > 0) {
    max_dynamic_aipp_size = CalcMaxSize(batch_count);
  }

  GELOGI("Add aipp input data, batch count is %ld, max_dynamic_aipp_size is %ld", batch_count, max_dynamic_aipp_size);
  return AddNodeToGraph(aipp_node, max_dynamic_aipp_size);
}

Status AippOp::AddAttrToAippData(const OpDescPtr &aipp_data_op_desc) {
  // Add dynamic aipp config to aipp_data
  NamedAttrs aipp_attr;
  ConvertParamToAttr(aipp_attr);
  (void)AttrUtils::SetNamedAttrs(aipp_data_op_desc, ATTR_NAME_AIPP, aipp_attr);
  (void)AttrUtils::SetStr(aipp_data_op_desc, ATTR_DATA_RELATED_AIPP_MODE, "dynamic_aipp_conf");

  // add node name attr to data linked aipp_data, it can be queried by acl.
  GE_CHECK_NOTNULL(data_node_linked_aipp);
  auto data_op_desc = data_node_linked_aipp->GetOpDesc();
  GE_CHECK_NOTNULL(data_op_desc);
  (void)AttrUtils::SetStr(data_op_desc, ATTR_DATA_AIPP_DATA_NAME_MAP, aipp_data_op_desc->GetName());
  (void)AttrUtils::SetStr(aipp_data_op_desc, ATTR_DATA_AIPP_DATA_NAME_MAP, data_op_desc->GetName());
  return SUCCESS;
}

Status AippOp::AddNodeToGraph(const NodePtr &aipp_node, int64_t max_dynamic_aipp_size) {
  GE_CHECK_NOTNULL(aipp_node);
  const ComputeGraphPtr &graph = aipp_node->GetOwnerComputeGraph();
  std::string node_name;
  // First aippdata name should be definite.
  if (graph->FindFirstNodeMatchType(AIPPDATA) == nullptr) {
    GELOGI("Current graph has no aippdata node, so the name of it must be definite.");
    node_name = kDynamicAippData;
  } else {
    node_name = std::string(kDynamicAippData) + "_" + aipp_node->GetName();
  }
  GELOGI("Current add aippdata node name is %s", node_name.c_str());

  // new add aipp_data ops for dynamic aipp param input
  OpDescPtr op_desc_ptr_data = MakeShared<OpDesc>(node_name, AIPPDATA);
  GE_CHECK_NOTNULL(op_desc_ptr_data);

  if (AddAttrToAippData(op_desc_ptr_data) != SUCCESS) {
    return INTERNAL_ERROR;
  }

  std::vector<int64_t> input_shape_dim(1, max_dynamic_aipp_size);
  GeShape input_shape(input_shape_dim);
  // construct input tensor
  GeTensorDesc input_tensor(input_shape, FORMAT_ND, DT_UINT8);
  input_tensor.SetOriginShape(input_shape);
  TensorUtils::SetReuseInput(input_tensor, false);

  GeShape output_shape(input_shape_dim);
  // construct output tensor
  GeTensorDesc output_tensor(output_shape, FORMAT_ND, DT_UINT8);
  TensorUtils::SetReuseInput(output_tensor, false);
  if (max_dynamic_aipp_size > 0) {
    TensorUtils::SetSize(input_tensor, max_dynamic_aipp_size);
    TensorUtils::SetSize(output_tensor, max_dynamic_aipp_size);
  }
  auto res_add_in = op_desc_ptr_data->AddInputDesc(input_tensor);
  auto res_add_out = op_desc_ptr_data->AddOutputDesc(output_tensor);

  NodePtr aipp_data_node_ptr = graph->AddNode(op_desc_ptr_data);
  GE_CHECK_NOTNULL(aipp_data_node_ptr);

  // add node desc for aipp node
  auto res_update_in = aipp_node->GetOpDesc()->UpdateInputDesc(kAippParamsInputIndex, output_tensor);
  if (res_add_in != GRAPH_SUCCESS || res_add_out != GRAPH_SUCCESS || res_update_in != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Add and Update InputDesc to op:%s(%s) failed, index:%d",
                      aipp_node->GetName().c_str(), aipp_node->GetType().c_str(), kAippParamsInputIndex);
    GELOGE(INTERNAL_ERROR, "[Update][InputDesc] to op:%s(%s) failed, index:%d",
           aipp_node->GetName().c_str(), aipp_node->GetType().c_str(), kAippParamsInputIndex);
    return INTERNAL_ERROR;
  }
  // aipp_node should have two input data but now tbe only one input
  if (GraphUtils::AddEdge(aipp_data_node_ptr->GetOutDataAnchor(kAippDataOutputIndex),
                          aipp_node->GetInDataAnchor(kAippParamsInputIndex)) != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Add edge between op:%s(%s)(out_index:%d) and op:%s(%s)(in_index:%d) failed",
                       aipp_data_node_ptr->GetName().c_str(), aipp_data_node_ptr->GetType().c_str(),
                       kAippDataOutputIndex, aipp_node->GetName().c_str(), aipp_node->GetType().c_str(),
                       kAippParamsInputIndex);
    GELOGE(INTERNAL_ERROR, "[Add][Edge] between op:%s(%s)(out_index:%d) and op:%s(%s)(in_index:%d) failed",
           aipp_data_node_ptr->GetName().c_str(), aipp_data_node_ptr->GetType().c_str(),
           kAippDataOutputIndex, aipp_node->GetName().c_str(), aipp_node->GetType().c_str(),
           kAippParamsInputIndex);
    return INTERNAL_ERROR;
  }
  return SUCCESS;
}
}  // namespace ge
