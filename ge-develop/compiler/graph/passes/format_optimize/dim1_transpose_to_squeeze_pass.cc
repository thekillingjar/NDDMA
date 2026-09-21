/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <memory>
#include <string>
#include <vector>
#include "graph/utils/node_utils.h"
#include "host_kernels/kernel.h"
#include "host_kernels/kernel_factory.h"
#include "graph/common/trans_op_creator.h"
#include "attribute_group/attr_group_symbolic_desc.h"
#include "checker.h"
#include "graph_utils.h"
#include "dim1_transpose_to_squeeze_pass.h"
#include "graph/operator_factory.h"
#include "graph/passes/pass_utils.h"
#include "platform/platform_info.h"
#include "platform/platform_info_def.h"
#include "graph/utils/type_utils.h"

namespace ge {
namespace {
constexpr uint32_t kTransposeInputX = 0U;
constexpr uint32_t kTransposeInputPerm = 1U;
constexpr uint32_t kTransposeOutputY = 0U;
constexpr uint32_t kSqueezeOutputY = 0U;
constexpr uint32_t kUnsqueezeInputX = 0U;
constexpr size_t kCreatedOpNum = 2;
const int64_t nanoBlockSize = 16; // nona芯片的ubblock_size为16, 用于隔离nano芯片
}  // namespace

Status Dim1TransposeToSqueezePass::Run(NodePtr &node) {
  GE_CHECK_NOTNULL(node);
  if (ShouldIgnoreOp(node)) {
    return SUCCESS;
  }

  const auto op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);

  std::vector<int64_t> perm;
  if (!PassUtils::GetPerm(node, kTransposeInputPerm, perm)) {
    return SUCCESS;
  }
  std::vector<int64_t> squeeze_axis, unsqueeze_axis, squeeze_output_shape;
  const auto input_x_desc = op_desc->MutableInputDesc(kTransposeInputX);
  GE_CHECK_NOTNULL(input_x_desc);
  bool is_useless_trans =
      IsUselessTransposeByShape(input_x_desc, perm, squeeze_axis, unsqueeze_axis, squeeze_output_shape);
  if (!is_useless_trans && !input_x_desc->GetShape().IsUnknownShape()) {
    return SUCCESS;
  }

  // 当非符号化场景判断为false，尝试用符号化再次判断是否成立
  bool is_useless_trans_symbolic = false;
  std::vector<int64_t> shape_index;
  if (!is_useless_trans) {
    squeeze_axis.clear();
    unsqueeze_axis.clear();
    is_useless_trans_symbolic =
        IsUselessTransposeBySymbolicShape(input_x_desc, perm, squeeze_axis, unsqueeze_axis, shape_index);
  }

  if (!is_useless_trans && !is_useless_trans_symbolic) {
    return SUCCESS;
  }

  // 非符号化或符号化判断成立，
  auto input_tensor_desc = op_desc->GetInputDesc(kTransposeInputX);
  const auto squeeze_out_tensor = ge::ComGraphMakeShared<GeTensorDesc>(input_tensor_desc);
  GE_CHECK_NOTNULL(squeeze_out_tensor);
  if (is_useless_trans_symbolic) {
    GE_CHK_STATUS_RET(SetShapeForSymbolic(input_x_desc, shape_index, squeeze_out_tensor, squeeze_output_shape));
  }
  squeeze_out_tensor->SetShape(GeShape(squeeze_output_shape));
  GE_CHK_STATUS_RET(ReplaceTransposeToSqueeze(squeeze_out_tensor, squeeze_axis, unsqueeze_axis, node));
  return SUCCESS;
}

OpDescPtr Dim1TransposeToSqueezePass::CreateOpDescPtr(const NodePtr &node, const string &op_type,
                                                      const GeTensorDesc &input_desc_x, const GeTensorDesc &output_desc,
                                                      const std::vector<int64_t> &axis_value_vec) const {
  const auto compute_graph = node->GetOwnerComputeGraph();
  const string origin_node_name = node->GetName();
  GE_ASSERT_NOTNULL(compute_graph);

  auto const insert_node_name = origin_node_name + "_replaced_" + op_type;
  auto created_op = OperatorFactory::CreateOperator(insert_node_name.c_str(), op_type.c_str());
  auto op_desc = OpDescUtils::GetOpDescFromOperator(created_op);
  GE_ASSERT_NOTNULL(op_desc);

  auto ret = op_desc->UpdateInputDesc("x", input_desc_x);
  GE_ASSERT_GRAPH_SUCCESS(ret, "Add input desc to op:%s(%s) failed, name:x", insert_node_name.c_str(), op_type.c_str());

  ret = op_desc->UpdateOutputDesc("y", output_desc);
  GE_ASSERT_GRAPH_SUCCESS(ret, "Add output desc to op:%s(%s) failed, name:y", insert_node_name.c_str(),
                          op_type.c_str());

  if (op_type == SQUEEZE) {
    ret = op_desc->SetAttr("axis", GeAttrValue::CreateFrom<std::vector<int64_t>>(axis_value_vec));
  } else if (op_type == UNSQUEEZE) {
    ret = op_desc->SetAttr("axes", GeAttrValue::CreateFrom<std::vector<int64_t>>(axis_value_vec));
  }
  GE_ASSERT_GRAPH_SUCCESS(ret, "Add attr axis op:%s(%s) failed, name:y", insert_node_name.c_str(), op_type.c_str());

  return op_desc;
}

Status Dim1TransposeToSqueezePass::ReplaceTransposeToSqueeze(const GeTensorDescPtr &tensor,
                                                             const std::vector<int64_t> &squeeze_axis,
                                                             const std::vector<int64_t> &unsqueeze_axis,
                                                             NodePtr &transpose_node) {
  GE_CHECK_NOTNULL(transpose_node);
  const auto op_desc = transpose_node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  auto input_tensor_desc = op_desc->MutableInputDesc(kTransposeInputX);
  auto output_tensor_desc = op_desc->MutableOutputDesc(kTransposeOutputY);
  GE_CHECK_NOTNULL(tensor);
  auto graph = transpose_node->GetOwnerComputeGraph();

  // 创建squeeze和unsqueeze,并继承transpose节点的边
  GE_CHECK_NOTNULL(input_tensor_desc);
  auto squeeze_op_desc = CreateOpDescPtr(transpose_node, SQUEEZE, *input_tensor_desc, *tensor, squeeze_axis);
  GE_CHECK_NOTNULL(output_tensor_desc);
  auto unsqueeze_op_desc = CreateOpDescPtr(transpose_node, UNSQUEEZE, *tensor, *output_tensor_desc, unsqueeze_axis);
  const auto created_ops = graph->FuseNodeKeepTopo({transpose_node}, {squeeze_op_desc, unsqueeze_op_desc});
  // created_ops中应该正好包含squeeze和unsqueeze两个算子
  if (created_ops.size() != kCreatedOpNum || created_ops.at(0) == nullptr || created_ops.at(1) == nullptr) {
    GELOGE(ge::FAILED, "Dim1TransposeToSqueezePass failed to create squeeze and unsqueeze op for %s",
           transpose_node->GetName().c_str());
    return FAILED;
  }
  auto squeeze = created_ops.at(0);
  auto unsqueeze = created_ops.at(1);

  GE_CHK_STATUS(GraphUtils::ReplaceNodesDataAnchors({squeeze, unsqueeze}, {transpose_node}, {0}, {-1, 0}));
  GE_CHK_STATUS(GraphUtils::InheritExecutionOrder({squeeze, unsqueeze}, {transpose_node}, graph));

  // 连接squeeze与unsqueeze
  GE_ASSERT_GRAPH_SUCCESS(
      GraphUtils::AddEdge(squeeze->GetOutDataAnchor(kSqueezeOutputY), unsqueeze->GetInDataAnchor(kUnsqueezeInputX)));

  // 删除无效节点
  GE_CHK_STATUS_RET(DeleteTranspose(transpose_node), "Dim1TransposeToSqueezePass failed");

  AddRePassNode(squeeze);
  AddRePassNode(unsqueeze);
  return SUCCESS;
}

Status Dim1TransposeToSqueezePass::DeleteTranspose(NodePtr &node) {
  GE_ASSERT_NOTNULL(node);
  auto node_name = node->GetName();
  auto node_type = node->GetType();
  if (node_type == TRANSPOSE) {
    auto perm_node = NodeUtils::GetInDataNodeByIndex(*node, kTransposeInputPerm);
    GE_ASSERT_NOTNULL(perm_node);
    auto perm_node_name = perm_node->GetName();
    GE_CHK_STATUS_RET(DeleteUselessConstAxisNode(perm_node), "Failed to remove const input node[%s] of node[%s]",
                      perm_node_name.c_str(), node_name.c_str());
  }
  std::vector<int32_t> data_relink_io_map = {kTransposeInputX};
  GE_CHK_STATUS_RET(IsolateAndDeleteNode(node, data_relink_io_map), "Failed to delete node:%s", node_name.c_str());
  GELOGI("The output of Node[%s][%s] is useless, success to remove useless transpose node", node_name.c_str(),
         node_type.c_str());
  return SUCCESS;
}

bool Dim1TransposeToSqueezePass::IsUselessTransposeByShape(const GeTensorDescPtr &input_x_desc,
                                                           const std::vector<int64_t> &perm,
                                                           std::vector<int64_t> &squeeze_axis,
                                                           std::vector<int64_t> &unsqueeze_axis,
                                                           std::vector<int64_t> &shape) {
  // 非符号化场景
  const auto &data_shape = input_x_desc->GetShape();
  int64_t current_largest_index = 0;
  for (size_t i = 0; i < perm.size(); ++i) {
    if (data_shape.GetDim(perm[i]) == 1) {
      squeeze_axis.emplace_back(perm[i]);
      unsqueeze_axis.emplace_back(i);
      continue;
    }
    if (perm[i] < current_largest_index) {
      return false;
    };
    current_largest_index = perm[i];
    shape.emplace_back(data_shape.GetDim(perm[i]));
  }
  return true;
}

bool Dim1TransposeToSqueezePass::IsUselessTransposeBySymbolicShape(const GeTensorDescPtr &input_x_desc,
                                                                   const std::vector<int64_t> &perm,
                                                                   std::vector<int64_t> &squeeze_axis,
                                                                   std::vector<int64_t> &unsqueeze_axis,
                                                                   std::vector<int64_t> &shape_index) {
  // 符号化场景
  int64_t current_largest_index = 0;
  const auto attr = input_x_desc->GetAttrsGroup<SymbolicDescAttr>();
  if (attr == nullptr) {
    return false;
  }
  const auto data_symbol_shape = attr->symbolic_tensor.GetOriginSymbolShape();
  for (size_t i = 0; i < perm.size(); ++i) {
    if (data_symbol_shape.GetDim(perm[i]) == Symbol(1)) {
      squeeze_axis.emplace_back(perm[i]);
      unsqueeze_axis.emplace_back(i);
      continue;
    }
    if (perm[i] < current_largest_index) {
      return false;
    };
    current_largest_index = perm[i];
    shape_index.emplace_back(perm[i]);
  }
  return true;
}

Status Dim1TransposeToSqueezePass::SetShapeForSymbolic(const GeTensorDescPtr &input_x_desc,
                                                       const std::vector<int64_t> &shape_index,
                                                       const GeTensorDescPtr &tensor,
                                                       std::vector<int64_t> &squeeze_output_shape) const {
  const auto attr = input_x_desc->GetAttrsGroup<SymbolicDescAttr>();
  GE_ASSERT_NOTNULL(attr);
  const auto data_symbol_shape = attr->symbolic_tensor.GetOriginSymbolShape();

  // 非符号化shape判断失效，需要根据符号化shape信息反推非符号化shape,同时为符号化shape赋值
  squeeze_output_shape.clear();
  gert::SymbolShape squeeze_output_shape_symbolic;
  for (const int64_t i : shape_index) {
    squeeze_output_shape.emplace_back(input_x_desc->GetShape().GetDim(i));
    squeeze_output_shape_symbolic.AppendDim(data_symbol_shape.GetDim(i));
  }
  tensor->GetOrCreateAttrsGroup<SymbolicDescAttr>()->symbolic_tensor.SetSymbolShape(squeeze_output_shape_symbolic);
  return SUCCESS;
}

bool IsAllMinusOne(const std::vector<int64_t> &dims) {
  return std::all_of(dims.begin(), dims.end(),[](int value) {
    return value == -1;
  });
}

bool Dim1TransposeToSqueezePass::ShouldIgnoreOp(const NodePtr &node) const {
  // 非transpose/transposeD算子，不作处理 或
  // nano芯片上会拦截消除transpose后输入直接送给输出（无task）的场景，因此对于nano芯片暂时不做该PASS
  int64_t block_size = GetBlockSize();
  if (((node->GetType() != TRANSPOSE) && (node->GetType() != TRANSPOSED)) || ((block_size == nanoBlockSize))) {
    return true;
  }

  // FE插入的transpose会存在output format和output origin format不一致情况，如NHWC和NCHW
  // 或者input format和input origin format不一致情况
  // 这两种情况下transpose不直接依赖perm进行转置，而是将输入origin shape透传至输出origin shape，在运行时根据format推导出预期的shape
  // 此时若将此类transpose替换为squeeze/unsqueeze并对shape为1的维度进行增删修改会破坏此类tranpose的运行逻辑导致最终shape不符预期，
  // 存在transpose输出全-1的场景，该场景下替换为squeeze/unsqueeze会导致shape不符合预期
  // 这些场景不作处理
  const auto op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  auto output_tensor_desc = op_desc->GetOutputDescPtr(kTransposeOutputY);
  GE_CHECK_NOTNULL(output_tensor_desc);
  GELOGD("Node %s[%s] has output origin format %s and output format %s",
        node->GetNamePtr(), node->GetTypePtr(), ge::TypeUtils::FormatToSerialString(output_tensor_desc->GetOriginFormat()).c_str(),
        ge::TypeUtils::FormatToSerialString(output_tensor_desc->GetFormat()).c_str());
  const auto dims = output_tensor_desc->GetShape().GetDims();
  int64_t inserted_by_fe = 0;
  // FE插入的transpose是非标transpose，其语义不能用transpose IR解释，此类transpose跳过处理
  if (AttrUtils::GetInt(op_desc, "_inserted_by_fe", inserted_by_fe) && (inserted_by_fe == 1)) {
    return true;
  }

  if (IsAllMinusOne(dims)) {
    return true;
  }

  return false;
}

int64_t Dim1TransposeToSqueezePass::GetBlockSize() const {
  // get ub size for the current device
  fe::PlatformInfo platform_info;
  fe::OptionalInfo opti_compilation_info;
  GE_CHK_STATUS_RET(fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(
                    platform_info, opti_compilation_info), "Get platform_info unsuccessful");
  GELOGI("UB block size is %ld.", platform_info.ai_core_spec.ubblock_size);
  return platform_info.ai_core_spec.ubblock_size;
}

REG_PASS_OPTION("Dim1TransposeToSqueezePass").LEVELS(OoLevel::kO2);
}  // namespace ge
