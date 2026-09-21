/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "decompose_large_const_pass.h"
#include "common/checker.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph/type/tensor_type_impl.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "register/shape_inference.h"
#include "utils/cg_utils.h"
#include "utils/auto_fuse_config.h"
#include "lowering/asc_lowerer/loop_common.h"
#include "base/err_msg.h"
#include "debug/ge_attr_define.h"
#include "lowering/lowerings.h"
#include "util/mem_utils.h"

namespace ge {
namespace {
constexpr const char_t *kOpTypeFill = "Fill";
constexpr const char_t *kOpTypeTile = "Tile";
constexpr const char_t *kFuseTypePointwise = "pointwise";
constexpr const char_t *kFuseTypeExtern = "extern";

bool CheckDstFuseTypes(const NodePtr &node) {
  std::vector<std::string> dst_fuse_types;
  for (const auto &dst_node : node->GetOutDataNodes()) {
    (void) LoweringManager::Lowering(dst_node);
    const auto node_out_anchor = dst_node->GetOutDataAnchor(0);
    const auto node_kernel_box = loop::GetKernelBox(node_out_anchor);
    const auto &fuse_type = FuseTypeToString(node_kernel_box.Type());
    dst_fuse_types.emplace_back(fuse_type);
    GELOGD("dst node = %s, fuse type = %s", dst_node->GetNamePtr(), fuse_type.c_str());
  }
  // 如果存在多引用，则校验fuse type都是pointwise，减少劣化可能
  if (dst_fuse_types.size() > 1) {
    for (const auto &fuse_type : dst_fuse_types) {
      if (fuse_type != kFuseTypePointwise) {
        GELOGD("node: %s, output has multi-ref, and peer node contains non-pointwise fuse type: %s, do not decompose ",
               node->GetNamePtr(), fuse_type.c_str());
        return false;
      }
    }
  }
  GE_CHK_BOOL_RET_SPECIAL_STATUS(dst_fuse_types.empty(), false, "no peer node, do not decompose");
  GE_CHK_BOOL_RET_SPECIAL_STATUS(dst_fuse_types.front() == kFuseTypeExtern, false,
                                 "dst_fuse_types is extern, do not decompose");
  return true;
}

template <typename T>
bool CheckIsBrc(const T *buffer, int64_t block_num, int64_t block_size, int64_t dim_size, int64_t element_num) {
  for (int64_t k = 0; k < block_num; ++k) {
    auto buffer_base = buffer + k * block_size;
    std::vector<T> lhs(buffer_base, buffer_base + element_num);
    for (int64_t j = 1; j < dim_size; ++j) {
      std::vector<T> rhs(buffer_base + j * element_num, buffer_base + j * element_num + element_num);
      if (lhs != rhs) {
        return false;
      }
    }
  }
  return true;
}

template <typename T>
void ResolveTileMultiples(const T *buffer, const GeShape &shape, std::vector<int64_t> &multiples) {
  const auto &dims = shape.GetDims();
  int64_t elt_num = 1;
  int64_t block_num = shape.GetShapeSize();
  int64_t block_size = 1;
  for (auto i = static_cast<int32_t>(dims.size()); i > 0; --i) {
    auto dim_size = dims[i - 1];
    if (dim_size == 1) {
      multiples.insert(multiples.begin(), 1);
      continue;
    }
    block_size = block_size * dim_size;
    block_num /= dim_size;
    auto is_brc = CheckIsBrc(buffer, block_num, block_size, dim_size, elt_num);
    auto multiple = is_brc ? dim_size : 1;
    multiples.insert(multiples.begin(), multiple);
    elt_num *= dim_size;
  }
}

NodePtr AddValueNode(const ComputeGraphPtr &graph, const NodePtr &node, const GeShape &new_const_shape) {
  ConstGeTensorPtr value;
  GE_ASSERT_TRUE(AttrUtils::GetTensor(node->GetOpDesc(), ATTR_NAME_WEIGHTS, value));
  const auto new_const_tensor = MakeShared<GeTensor>();
  GE_ASSERT_NOTNULL(new_const_tensor);
  new_const_tensor->MutableTensorDesc().Update(new_const_shape, FORMAT_ND, value->GetTensorDesc().GetDataType());
  new_const_tensor->MutableTensorDesc().SetOriginShape(new_const_shape);
  const auto buffer = value->GetData().GetData();
  const auto dtype_size = ge::GetSizeByDataType(value->GetTensorDesc().GetDataType());
  const auto num_elements = new_const_shape.IsScalar() ? 1 : new_const_shape.GetShapeSize();
  new_const_tensor->SetData(buffer, dtype_size * num_elements);
  const auto op_desc = OpDescUtils::CreateConstOpZeroCopy(new_const_tensor);
  GE_ASSERT_NOTNULL(op_desc);
  const auto new_const_node = graph->AddNode(op_desc);
  return new_const_node;
}

NodePtr AddShapeNode(const ComputeGraphPtr &graph, const std::vector<int64_t> &multiples) {
  const auto multiple_tensor = MakeShared<GeTensor>();
  GE_ASSERT_NOTNULL(multiple_tensor);
  auto &tensor_desc = multiple_tensor->MutableTensorDesc();
  const auto multiple_tensor_shape = GeShape(std::vector<int64_t>{static_cast<int64_t>(multiples.size())});
  tensor_desc.Update(multiple_tensor_shape, FORMAT_ND, DT_INT64);
  tensor_desc.SetOriginShape(multiple_tensor_shape);
  multiple_tensor->SetData(PtrToPtr<int64_t, uint8_t>(multiples.data()), multiples.size() * sizeof(int64_t));
  const auto op_desc = OpDescUtils::CreateConstOpZeroCopy(multiple_tensor);
  GE_ASSERT_NOTNULL(op_desc);
  auto multiple_node = graph->AddNode(op_desc);
  return multiple_node;
}

Status UpdateSymShape(const NodePtr &node, const GeShape &shape) {
  auto src = node->GetOutDataAnchor(0);
  GE_ASSERT_NOTNULL(src);
  auto desc = src->GetOwnerNode()->GetOpDesc()->MutableOutputDesc(src->GetIdx());
  GE_ASSERT_NOTNULL(desc);
  auto sym_attr = desc->GetOrCreateAttrsGroup<SymbolicDescAttr>();
  GE_ASSERT_TRUE(sym_attr != nullptr, "Skip lowering node %s, as: No symbolic desc attr.", node->GetName().c_str());
  std::vector<ge::Expression> new_shape;
  for (const auto &dim : shape.GetDims()) {
    new_shape.emplace_back(ge::Symbol(dim));
  }
  sym_attr->symbolic_tensor.MutableOriginSymbolShape().MutableDims() = new_shape;
  return SUCCESS;
}

Status DecomposeConst(NodePtr &node, const GeShape &new_const_shape, const std::vector<int64_t> &tile_multiples) {
  const ComputeGraphPtr &graph = node->GetOwnerComputeGraph();
  GE_ASSERT_NOTNULL(graph);
  auto new_const_node = AddValueNode(graph, node, new_const_shape);
  GE_ASSERT_NOTNULL(new_const_node);
  GE_ASSERT_SUCCESS(UpdateSymShape(new_const_node, new_const_shape));
  GE_CHK_GRAPH_STATUS_RET(GraphUtils::MoveInCtrlEdges(node, new_const_node));
  GE_CHK_GRAPH_STATUS_RET(GraphUtils::MoveOutCtrlEdges(node, new_const_node));
  GELOGD("value node %s added, shape = [%s]", new_const_node->GetNamePtr(),
         new_const_node->GetOpDesc()->GetOutputDesc(0).GetShape().ToString().c_str());
  bool is_from_constant_folding = false;
  const std::string kAttrName = "_is_from_constant_folding";
  (void)AttrUtils::GetBool(node->GetOpDesc(), kAttrName, is_from_constant_folding);
  (void)AttrUtils::SetBool(new_const_node->GetOpDesc(), kAttrName, is_from_constant_folding);
  GELOGD("%s set attr %s = %d", new_const_node->GetNamePtr(), kAttrName.c_str(), is_from_constant_folding);
  const auto multiple_node = AddShapeNode(graph, tile_multiples);
  GE_ASSERT_NOTNULL(multiple_node);
  GELOGD("multiple node %s added, shape = [%s], value = %s", multiple_node->GetNamePtr(),
         multiple_node->GetOpDesc()->GetOutputDesc(0).GetShape().ToString().c_str(), ToString(tile_multiples).c_str());
  const auto is_fill = new_const_shape.IsScalar();
  const auto op_type = is_fill ? kOpTypeFill : kOpTypeTile;
  const int32_t value_index = is_fill ? 1 : 0;
  const int32_t shape_index = is_fill ? 0 : 1;
  auto tile_op = ge::OperatorFactory::CreateOperator((node->GetName() + "_by_autofuse_pass").c_str(), op_type);
  tile_op.BreakConnect();
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(tile_op);
  GE_CHECK_NOTNULL(op_desc);
  *op_desc->MutableInputDesc(shape_index) = multiple_node->GetOpDesc()->GetOutputDesc(0);
  *op_desc->MutableInputDesc(value_index) = new_const_node->GetOpDesc()->GetOutputDesc(0);
  *op_desc->MutableOutputDesc(0) = node->GetOpDesc()->GetOutputDesc(0);
  const auto tile_node = graph->AddNode(op_desc);
  GE_ASSERT_NOTNULL(tile_node);
  GE_ASSERT_GRAPH_SUCCESS(
      GraphUtils::AddEdge(multiple_node->GetOutDataAnchor(0), tile_node->GetInDataAnchor(shape_index)));
  GE_ASSERT_GRAPH_SUCCESS(
      GraphUtils::AddEdge(new_const_node->GetOutDataAnchor(0), tile_node->GetInDataAnchor(value_index)));
  GELOGD("calc node %s(%s) added", tile_node->GetNamePtr(), tile_node->GetTypePtr());
  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::ReplaceNodeAnchors(tile_node, node, {}, {0}));
  GE_ASSERT_GRAPH_SUCCESS(graph->RemoveNode(node));
  GELOGD("origin node %s removed", node->GetNamePtr());
  return SUCCESS;
}

Status ResolveNewConstShapeAndTileMultiples(const NodePtr &node, GeShape &new_shape, std::vector<int64_t> &multiples) {
  auto op_desc = node->GetOpDesc();
  ConstGeTensorPtr value;
  GE_ASSERT_TRUE(AttrUtils::GetTensor(op_desc, ATTR_NAME_WEIGHTS, value));
  const auto &shape = value->GetTensorDesc().GetShape();
  const auto buffer = value->GetData().GetData();
  const auto dtype_size = ge::GetSizeByDataType(value->GetTensorDesc().GetDataType());
  if (dtype_size == sizeof(uint8_t)) {
    ResolveTileMultiples(buffer, shape, multiples);
  } else if (dtype_size == sizeof(uint16_t)) {
    ResolveTileMultiples(PtrToPtr<uint8_t, uint16_t>(buffer), shape, multiples);
  } else if (dtype_size == sizeof(uint32_t)) {
    ResolveTileMultiples(PtrToPtr<uint8_t, uint32_t>(buffer), shape, multiples);
  } else if (dtype_size == sizeof(uint64_t)) {
    ResolveTileMultiples(PtrToPtr<uint8_t, uint64_t>(buffer), shape, multiples);
  } else {
  }
  GELOGD("multiples = %s", ToString(multiples).c_str());
  GE_ASSERT_EQ(shape.GetDimNum(), multiples.size());
  std::vector<int64_t> new_const_dims = shape.GetDims();
  bool can_brc = true;
  // 先支持高维连续brc场景
  for (size_t dim_index = 0; dim_index < multiples.size(); ++dim_index) {
    const auto dim_size = shape.GetDim(dim_index);
    const auto multiple = multiples[dim_index];
    can_brc = can_brc && ((dim_size == 1) || (multiple > 1));
    if (can_brc) {
      new_const_dims[dim_index] = 1;
    } else {
      multiples[dim_index] = 1;
    }
  }
  // 全1的shape转为标量
  new_shape = (new_const_dims == std::vector<int64_t>(shape.GetDimNum(), 1)) ? GeShape() : GeShape(new_const_dims);
  return SUCCESS;
}
}  // namespace

graphStatus DecomposeLargeConstPass::TryDecompose(NodePtr &node, bool &decomposed) {
  constexpr int64_t kLargeConstThreshold = 1024 * 1024 * 128;
  const auto &op_desc = node->GetOpDesc();
  const auto &shape = op_desc->GetOutputDesc(0).GetShape();
  const auto dtype_size = ge::GetSizeByDataType(op_desc->GetOutputDesc(0).GetDataType());
  GE_CHK_BOOL_RET_SPECIAL_STATUS(dtype_size <= 0, SUCCESS, "skip process node: %s, dtype %s is not supported",
                                 node->GetName().c_str(),
                                 TypeUtils::DataTypeToSerialString(op_desc->GetOutputDesc(0).GetDataType()).c_str());
  auto need_decompose = ((shape.GetShapeSize() * dtype_size) >= kLargeConstThreshold);
  GELOGD("const node: %s, element_num = %ld, dtype_size = %d, need decompose = %d", node->GetNamePtr(),
         shape.GetShapeSize(), dtype_size, need_decompose);
  if (!need_decompose) {
    return GRAPH_SUCCESS;
  }
  need_decompose = CheckDstFuseTypes(node);
  if (!need_decompose) {
    return GRAPH_SUCCESS;
  }
  GeShape new_shape;
  std::vector<int64_t> multiples;
  GE_ASSERT_SUCCESS(ResolveNewConstShapeAndTileMultiples(node, new_shape, multiples));
  const auto can_decompose = (new_shape.GetShapeSize() < shape.GetShapeSize());
  GELOGD("const node: %s, shape = [%s], new_shape = [%s], multiples = %s, can_decompose = %d", node->GetNamePtr(),
         shape.ToString().c_str(), new_shape.ToString().c_str(), ToString(multiples).c_str(),
         static_cast<int32_t>(can_decompose));
  if (can_decompose) {
    GE_ASSERT_SUCCESS(DecomposeConst(node, new_shape, multiples));
    decomposed = true;
    GELOGD("const node: %s, decomposed successfully, prev shape = [%s], new_shape = [%s], multiples = %s",
           node->GetNamePtr(), shape.ToString().c_str(), new_shape.ToString().c_str(), ToString(multiples).c_str());
  }
  return GRAPH_SUCCESS;
}

graphStatus DecomposeLargeConstPass::Run(const ComputeGraphPtr &graph) {
  bool need_topo_sort = false;
  for (auto &node : graph->GetDirectNode()) {
    GE_ASSERT_NOTNULL(node);
    if (NodeUtils::IsConst(*node)) {
      bool decomposed = false;
      GE_ASSERT_SUCCESS(TryDecompose(node, decomposed));
      need_topo_sort = need_topo_sort || decomposed;
    }
  }
  if (need_topo_sort) {
    GE_ASSERT_GRAPH_SUCCESS(graph->TopologicalSorting());
  }
  return GRAPH_SUCCESS;
}
}  // namespace ge