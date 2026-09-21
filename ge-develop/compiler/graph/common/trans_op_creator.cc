/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "trans_op_creator.h"

#include "framework/common/ge_format_util.h"
#include "framework/common/types.h"
#include "framework/common/debug/ge_log.h"
#include "common/checker.h"
#include "formats/formats.h"
#include "formats/format_transfers/format_transfer_transpose.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/hash_utils.h"
#include "common/plugin/ge_make_unique_util.h"
#include "api/gelib/gelib.h"

namespace ge {
// For some historical reasons, only these two formats need groups
const static std::set<Format> kFormatWithGroups = {FORMAT_FRACTAL_Z, FORMAT_FRACTAL_Z_3D};
namespace {
void SetTransDataSrcFormat(const OpDescPtr &op_desc, Format src_format) {
  if (HasSubFormat(src_format)) {
    const int32_t src_subformat = GetSubFormat(src_format);
    if (!AttrUtils::SetInt(op_desc, FORMAT_TRANSFER_SRC_SUBFORMAT, src_subformat)) {
      GELOGW("Set attr [src_subformat] for node [%s] failed.", op_desc->GetName().c_str());
    }
    src_format = static_cast<ge::Format>(GetPrimaryFormat(src_format));
    if (kFormatWithGroups.count(src_format) > 0UL && !AttrUtils::SetInt(op_desc, "groups", src_subformat)) {
      GELOGW("Set attr [groups] for node [%s] failed.", op_desc->GetName().c_str());
    }
  }

  if (!AttrUtils::SetStr(op_desc, FORMAT_TRANSFER_SRC_FORMAT, TypeUtils::FormatToSerialString(src_format))) {
    GELOGW("Set attr [src_format] for node [%s] failed.", op_desc->GetName().c_str());
  }
}

void SetTransDataDstFormat(const OpDescPtr &op_desc, Format dst_format) {
  if (HasSubFormat(dst_format)) {
    const int32_t dst_subformat = GetSubFormat(dst_format);
    if (!AttrUtils::SetInt(op_desc, FORMAT_TRANSFER_DST_SUBFORMAT, dst_subformat)) {
      GELOGW("Set attr [dst_subformat] for node [%s] failed.", op_desc->GetName().c_str());
    }
    dst_format = static_cast<ge::Format>(GetPrimaryFormat(dst_format));
    if (kFormatWithGroups.count(dst_format) > 0UL && !AttrUtils::SetInt(op_desc, "groups", dst_subformat)) {
      GELOGW("Set attr [groups] for node [%s] failed.", op_desc->GetName().c_str());
    }
  }

  if (!AttrUtils::SetStr(op_desc, FORMAT_TRANSFER_DST_FORMAT, TypeUtils::FormatToSerialString(dst_format))) {
    GELOGW("Set attr [dst_format] for node [%s] failed.", op_desc->GetName().c_str());
  }
}

OpDescPtr CreateTensorShape(const GeTensorDesc &data_tensor) {
  GeTensorPtr tensor = MakeShared<GeTensor>();
  GE_ASSERT_NOTNULL(tensor, "New GeTensor failed");

  tensor->MutableTensorDesc().SetDataType(DT_INT32);
  tensor->MutableTensorDesc().SetFormat(FORMAT_ND);
  auto dst_ge_shape = data_tensor.GetOriginShape();
  auto dim_cnt = static_cast<int64_t>(dst_ge_shape.GetDimNum());
  if (dim_cnt == 0) {  // if the dim_cnt is 0, the tensor is a scalar
    tensor->MutableTensorDesc().SetShape(GeShape({0}));
  } else {
    tensor->MutableTensorDesc().SetShape(GeShape(std::vector<int64_t>({dim_cnt})));
    auto dst_shape = MakeUnique<int32_t[]>(dim_cnt);
    GE_ASSERT_NOTNULL(dst_shape, "Malloc buffer failed, size:%zu", dim_cnt);
    for (int64_t i = 0; i < dim_cnt; ++i) {
      dst_shape[i] = dst_ge_shape.GetDim(static_cast<size_t>(i));
    }
    GE_ASSERT_GRAPH_SUCCESS(
        tensor->SetData(reinterpret_cast<const uint8_t *>(dst_shape.get()), dim_cnt * sizeof(int32_t)),
        "Set data to tensor failed");
  }
  tensor->MutableTensorDesc().SetOriginShape(tensor->MutableTensorDesc().GetShape());
  GELOGD("Create shape input dim [%s]", dst_ge_shape.ToString().c_str());
  return OpDescUtils::CreateConstOpZeroCopy(tensor);
}

NodePtr CreateShapeConst(const ComputeGraphPtr &compute_graph, const GeTensorDesc &tensor_desc) {
  auto shape_op_desc = CreateTensorShape(tensor_desc);
  GE_ASSERT_NOTNULL(shape_op_desc, "[Create][TensorShape] Failed to add shape for reshape");

  auto shape_node = compute_graph->AddNode(shape_op_desc);
  GE_ASSERT_NOTNULL(shape_node, "Add node:%s(%s) to graph:%s failed", shape_op_desc->GetName().c_str(),
                    shape_op_desc->GetType().c_str(), compute_graph->GetName().c_str());
  return shape_node;
}
NodePtr GetOrCreateShapeConst(const ComputeGraphPtr &compute_graph, const GeTensorDesc &tensor_desc,
                              std::unordered_map<GeShape, NodePtr, GeShapeHasher> &reshape_target_shape_2_const_nodes) {
  const auto &reshape_input_shape = tensor_desc.GetShape();
  if (!reshape_target_shape_2_const_nodes.empty()) {
    const auto &iter = reshape_target_shape_2_const_nodes.find(reshape_input_shape);
    if (iter != reshape_target_shape_2_const_nodes.cend()) {
      GELOGD("Get shape const node[%s] from map, shape[%s].", iter->second->GetName().c_str(),
             reshape_input_shape.ToString().c_str());
      return iter->second;
    }
  }
  const auto &const_node = CreateShapeConst(compute_graph, tensor_desc);
  GE_ASSERT_NOTNULL(const_node);
  reshape_target_shape_2_const_nodes.emplace(std::pair<GeShape, NodePtr>(reshape_input_shape, const_node));
  GELOGD("Create shape const[%s], shape[%s]", const_node->GetName().c_str(), reshape_input_shape.ToString().c_str());
  return const_node;
}

NodePtr CreateReshapeNodeToGraphWithConstInput(const ComputeGraphPtr &compute_graph, const NodePtr &shape_const_node,
                                               const std::string &op_name, const GeTensorDesc &input_desc_x,
                                               const GeTensorDesc &output_desc) {
  auto op_desc = MakeShared<OpDesc>(op_name, RESHAPE);
  GE_ASSERT_NOTNULL(op_desc);

  auto ret = op_desc->AddInputDesc("x", input_desc_x);
  GE_ASSERT_GRAPH_SUCCESS(ret, "Add input desc to op:%s(%s) failed, name:x", op_name.c_str(), RESHAPE);

  ret = op_desc->AddInputDesc("shape", shape_const_node->GetOpDesc()->GetOutputDesc(0));
  GE_ASSERT_GRAPH_SUCCESS(ret, "Add input desc to op:%s(%s) failed, name:shape", op_name.c_str(), RESHAPE);

  ret = op_desc->AddOutputDesc("y", output_desc);
  GE_ASSERT_GRAPH_SUCCESS(ret, "Add output desc to op:%s(%s) failed, name:y", op_name.c_str(), RESHAPE);

  auto reshape_node = compute_graph->AddNode(op_desc);
  GE_ASSERT_NOTNULL(reshape_node, "Add node:%s(%s) to graph:%s failed", op_desc->GetName().c_str(),
                    op_desc->GetType().c_str(), compute_graph->GetName().c_str());

  ret = GraphUtils::AddEdge(shape_const_node->GetOutDataAnchor(0), reshape_node->GetInDataAnchor(1));
  GE_ASSERT_GRAPH_SUCCESS(ret, "Add edge between op:%s(%s)(out_index:0) and op:%s(%s)(in_index:1) failed",
                          shape_const_node->GetName().c_str(), shape_const_node->GetType().c_str(), op_name.c_str(),
                          RESHAPE);
  GELOGD("[Reshape][Const]Add edge between op:%s(%s)(out_index:0) and op:%s(%s)(in_index:1) success.",
         shape_const_node->GetName().c_str(), shape_const_node->GetType().c_str(), op_name.c_str(), RESHAPE);
  return reshape_node;
}
} // namespace

uint64_t GeShapeHasher::operator()(const GeShape &shape) const {
  uint64_t seed = HashUtils::MultiHash();
  for (size_t idx = 0U; idx < shape.GetDimNum(); ++idx) {
    seed = HashUtils::HashCombine(seed, shape.GetDim(idx));
  }
  return seed;
}

OpDescPtr TransOpCreator::CreateTransDataOp(const std::string &op_name, const GeTensorDesc &input_desc,
                                            const GeTensorDesc &output_desc, bool enable_check_acc_support) {
  auto op_desc = MakeShared<OpDesc>(op_name, TRANSDATA);
  if (op_desc == nullptr) {
    GELOGE(FAILED, "Failed to new transdata opdesc.");
    return nullptr;
  }
  // set src/dst format
  SetTransDataSrcFormat(op_desc, input_desc.GetFormat());
  SetTransDataDstFormat(op_desc, output_desc.GetFormat());

  if (AddInputOutputDesc(op_desc, {input_desc}, output_desc) != GRAPH_SUCCESS) {
    return nullptr;
  }

  // check accuarcy support
  if (enable_check_acc_support) {
    bool is_supported = false;
    if ((CheckAccuracySupported(op_desc, is_supported) != GRAPH_SUCCESS) || (!is_supported)) {
      GELOGW("[Check][AccuracySupported] %s(TransData) failed.", op_name.c_str());
      return nullptr;
    }
  }
  return op_desc;
}

OpDescPtr TransOpCreator::CreateTransPoseDOp(const std::string &op_name, const GeTensorDesc &input_desc,
                                             const GeTensorDesc &output_desc) {
  auto op_desc = MakeShared<OpDesc>(op_name, TRANSPOSED);
  if (op_desc == nullptr) {
    GELOGE(FAILED, "Failed to new transopsed opdesc.");
    return nullptr;
  }

  if (AddInputOutputDesc(op_desc, {input_desc}, output_desc) != GRAPH_SUCCESS) {
    return nullptr;
  }

  auto src_format = input_desc.GetFormat();
  auto dst_format = output_desc.GetFormat();
  std::vector<int64_t> perm_arg;
  if (formats::GetPermByForamt(src_format, dst_format, perm_arg) != SUCCESS) {
    GELOGW("Get perm by foramt failed.");
    return op_desc;
  }

  if (!AttrUtils::SetListInt(op_desc, PERMUTE_ATTR_PERM, perm_arg)) {
    GELOGW("SetStr PERMUTE_ATTR_PERM failed.");
  }
  return op_desc;
}

OpDescPtr TransOpCreator::CreateCastOp(const std::string &op_name, const GeTensorDesc &input_desc,
                                       const GeTensorDesc &output_desc, bool enable_check_acc_support) {
  auto op_desc = MakeShared<OpDesc>(op_name, CAST);
  if (op_desc == nullptr) {
    GELOGE(FAILED, "Failed to new cast opdesc.");
    return nullptr;
  }
  auto input_dtype = input_desc.GetDataType();
  if (!AttrUtils::SetInt(op_desc, CAST_ATTR_SRCT, static_cast<int64_t>(input_dtype))) {
    GELOGW("SetInt CAST_ATTR_SRCT failed");
  }
  auto output_dtype = output_desc.GetDataType();
  if (!AttrUtils::SetInt(op_desc, CAST_ATTR_DSTT, static_cast<int64_t>(output_dtype))) {
    GELOGW("SetInt CAST_ATTR_DSTT failed");
  }
  if (!AttrUtils::SetInt(op_desc, CAST_ATTR_DST_TYPE, static_cast<int64_t>(output_dtype))) {
    GELOGW("SetInt CAST_ATTR_DST_TYPE failed");
  }
  if (!AttrUtils::SetBool(op_desc, CAST_ATTR_TRUNCATE, false)) {
    GELOGW("SetBool CAST_ATTR_TRUNCATE failed");
  }

  if (AddInputOutputDesc(op_desc, {input_desc}, output_desc) != GRAPH_SUCCESS) {
    return nullptr;
  }

  // check accuarcy support
  if (enable_check_acc_support) {
    bool is_supported = false;
    if ((CheckAccuracySupported(op_desc, is_supported) != GRAPH_SUCCESS) || (!is_supported)) {
      GELOGW("[Check][AccuracySupported] %s(Cast) failed.", op_name.c_str());
      return nullptr;
    }
  }
  return op_desc;
}

NodePtr TransOpCreator::CreateReshapeNodeToGraph(const ComputeGraphPtr &compute_graph, const std::string &op_name,
                                                 const GeTensorDesc &input_desc_x, const GeTensorDesc &output_desc) {
  // build shape node
  auto shape_node = CreateShapeConst(compute_graph, output_desc);
  GE_ASSERT_NOTNULL(shape_node);
  return CreateReshapeNodeToGraphWithConstInput(compute_graph, shape_node, op_name, input_desc_x, output_desc);
}

NodePtr TransOpCreator::CreateReshapeNodeToGraph(
    const ComputeGraphPtr &compute_graph, const std::string &op_name,
    const GeTensorDesc &input_desc_x, const GeTensorDesc &output_desc,
    std::unordered_map<GeShape, NodePtr, GeShapeHasher> &reshape_target_shape_2_const_nodes) {
  // build shape node
  auto shape_node = GetOrCreateShapeConst(compute_graph, output_desc, reshape_target_shape_2_const_nodes);
  GE_ASSERT_NOTNULL(shape_node);
  return CreateReshapeNodeToGraphWithConstInput(compute_graph, shape_node, op_name, input_desc_x, output_desc);
}

OpDescPtr TransOpCreator::CreateOtherTransOp(const std::string &op_name, const std::string &op_type,
                                             const GeTensorDesc &input_desc, const GeTensorDesc &output_desc) {
  auto op_desc = MakeShared<OpDesc>(op_name, op_type);
  if (op_desc == nullptr) {
    GELOGE(FAILED, "Failed to new opdesc.");
    return nullptr;
  }
  if (AddInputOutputDesc(op_desc, {input_desc}, output_desc) != GRAPH_SUCCESS) {
    return nullptr;
  }
  return op_desc;
}

graphStatus TransOpCreator::CheckAccuracySupported(const OpDescPtr &op_desc, bool &is_supported) {
  std::string unsupported_reason;
  return CheckAccuracySupported(op_desc, "", is_supported, unsupported_reason);
}

graphStatus TransOpCreator::CheckAccuracySupported(const OpDescPtr &op_desc, const std::string &engine_name,
                                                   bool &is_supported, std::string &unsupported_reason) {
  // is_supported is set to be false as default value.
  auto instance = GELib::GetInstance();
  if ((instance == nullptr) || (!instance->InitFlag())) {
    REPORT_INNER_ERR_MSG("E19999", "GELib is not initialized!");
    GELOGE(GRAPH_FAILED, "GELib is not initialized!");
    return GRAPH_FAILED;
  }

  OpsKernelManager &ops_kernel_manager = instance->OpsKernelManagerObj();
  std::vector<OpInfo> op_infos = ops_kernel_manager.GetOpsKernelInfo(op_desc->GetType());
  if (op_infos.empty()) {
    unsupported_reason = "Can not get op info by op type " + op_desc->GetType();
    GELOGI("Can not get op info by op type:%s", op_desc->GetType().c_str());
    return GRAPH_FAILED;
  }

  bool real_query = true;
  for (const auto &it : op_infos) {
    auto kernel_map = ops_kernel_manager.GetAllOpsKernelInfoStores();
    auto &kernel_name = it.opKernelLib;
    if (!engine_name.empty() && kernel_name != engine_name) {
      continue;
    }
    auto kernel_info_store = kernel_map.find(kernel_name);
    if (kernel_info_store != kernel_map.end()) {
      if (kernel_info_store->second != nullptr &&
          kernel_info_store->second->CheckAccuracySupported(op_desc, unsupported_reason, real_query)) {
        GELOGD("OpKernelLibName %s and engine name %s into op_desc %s", kernel_name.c_str(), it.engine.c_str(),
               op_desc->GetName().c_str());
        is_supported = true;
        return GRAPH_SUCCESS;
      }
    }
  }
  GELOGI("op:%s CheckAccuracySupported result: %s", op_desc->GetName().c_str(), unsupported_reason.c_str());
  return GRAPH_SUCCESS;
}
}  // namespace ge
