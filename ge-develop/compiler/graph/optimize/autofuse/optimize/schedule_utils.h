/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPTIMIZE_SCHEDULE_UTILS_H_
#define OPTIMIZE_SCHEDULE_UTILS_H_

#include "ascendc_ir/ascendc_ir_core/ascendc_ir_def.h"
#include "graph/symbolizer/symbolic_utils.h"
#include "asc_graph_utils.h"
#include "ascir_ops_utils.h"
#include "ascgen_log.h"
#include "ascir.h"
#include "ascir_ops.h"
#include "common/platform_context.h"

namespace optimize {
class ScheduleUtils {
 public:
  static ge::ComputeType GetComputeType(const ge::AscNodePtr &node) {
    return node->attr.api.compute_type;
  }

  // 后端默认采用逆dfs的拓扑排序方式
  static Status TopologicalSorting(ge::AscGraph &graph);

  static bool IsElewise(const ge::AscNodePtr &node) {
    return node->attr.api.compute_type == ge::ComputeType::kComputeElewise;
  }

  static bool IsBroadcast(const ge::AscNodePtr &node) {
    return node->attr.api.compute_type == ge::ComputeType::kComputeBroadcast;
  }

  static bool IsReduce(const ge::AscNodePtr &node) {
    return node->attr.api.compute_type == ge::ComputeType::kComputeReduce;
  }

  static bool IsTranspose(const ge::AscNodePtr &node) {
    return node->attr.api.compute_type == ge::ComputeType::kComputeTranspose;
  }

  static bool IsLoad(const ge::AscNodePtr &node) {
    return node->attr.api.compute_type == ge::ComputeType::kComputeLoad;
  }

  static bool IsStore(const ge::AscNodePtr &node) {
    return node->attr.api.compute_type == ge::ComputeType::kComputeStore;
  }

  static bool IsConcat(const ge::AscNodePtr &node) {
    return node->attr.api.compute_type == ge::ComputeType::kComputeConcat;
  }

  static bool IsSplit(const ge::AscNodePtr &node) {
    return node->attr.api.compute_type == ge::ComputeType::kComputeSplit;
  }

  static bool IsGather(const ge::AscNodePtr &node) {
    return node->attr.api.compute_type == ge::ComputeType::kComputeGather;
  }

  static bool IsBuffer(const ge::AscNodePtr &node) {
    return node->attr.api.type == ge::ApiType::kAPITypeBuffer;
  }

  static bool IsCompute(const ge::AscNodePtr &node) {
    return node->attr.api.type == ge::ApiType::kAPITypeCompute;
  }

  static bool IsCube(const ge::AscNodePtr &node) {
    return node->attr.api.compute_type == ge::ComputeType::kComputeCube;
  }

  static bool HasComputeType(const ascir::ImplGraph &impl_graph, const ge::ComputeType compute_type);

  static bool IsIOBuffer(const ge::NodePtr &node) {
    return ge::ops::IsOps<ge::ascir_op::Scalar>(node) || ge::ops::IsOps<ge::ascir_op::Data>(node) ||
           ge::ops::IsOps<ge::ascir_op::Output>(node);
  }

  static bool IsConstantScalar(const ge::Node *const node) {
    return ge::ops::IsOps<ge::ascir_op::Scalar>(node) || ge::ops::IsOps<ge::ascir_op::IndexExpr>(node);
  }

  static bool IsRemovePad(const ge::NodePtr &node) {
    return ge::ops::IsOps<ge::ascir_op::RemovePad>(node);
  }

  template <typename T>
  static bool IsNodeSupportDataType(const ge::DataType data_type) {
    std::string npu_arch;
    GE_ASSERT_SUCCESS(ge::PlatformContext::GetInstance().GetCurrentPlatformString(npu_arch));
    std::vector exp_dtypes{data_type};
    if (T::InferDataType({data_type}, exp_dtypes, npu_arch) != ge::SUCCESS) {
      GELOGD("%s not support dtype=%s", T::Type, ge::TypeUtils::DataTypeToSerialString(data_type).c_str());
      return false;
    }
    return true;
  }

  // 获取 npu_arch 并调用 InferDataType
  template <typename OpType>
  static Status CallAscirInferDataType(const std::vector<ge::DataType> &input_dtypes,
                                       std::vector<ge::DataType> &expect_output_dtypes) {
    std::string npu_arch;
    GE_ASSERT_SUCCESS(ge::PlatformContext::GetInstance().GetCurrentPlatformString(npu_arch));
    return OpType::InferDataType(input_dtypes, expect_output_dtypes, npu_arch);
  }

  static bool GetGatherParams(ge::AscGraph &graph, int64_t &attr_axis, int64_t &params_size) {
    bool has_gather = false;
    for (const auto &node : graph.GetAllNodes()) {
      if (!IsGather(node)) {
        continue;
      }
      has_gather = true;
      auto input_axis = node->inputs[1].attr.axis;
      GE_ASSERT_TRUE(!input_axis.empty(), "input attr is invalid.");
      auto output_axis = node->outputs[0].attr.axis;
      for (auto iter = output_axis.begin(); iter != output_axis.end(); ++iter) {
        auto axis = graph.FindAxis(*iter);
        GE_ASSERT_NOTNULL(axis);
        if (axis->id == input_axis[0] ||
            std::find(axis->from.begin(), axis->from.end(), input_axis[0]) != axis->from.end()) {
          attr_axis = std::distance(output_axis.begin(), iter);
          GELOGD("Axis index:[%ld], axis_id:[%ld].", attr_axis, axis->id);
        }
      }
      params_size = node->inputs[0].attr.repeats.size();
      break;
    }
    return has_gather;
  }

  /**
   * 从ComputeGraph上获取AscGraphAttr属性，如果graph或者获取到的ascGraphAttr是空，则抛出异常。
   */
  static ge::AscGraphAttr *GetOrCreateGraphAttrsGroup(const ge::ComputeGraphPtr &graph) {
    GE_CHECK_NOTNULL_EXEC(graph, return nullptr;);
    auto attr = graph->GetOrCreateAttrsGroup<ge::AscGraphAttr>();
    GE_CHECK_NOTNULL_EXEC(attr, return nullptr;);
    return attr;
  }

  /**
   * 给定node，根据其所属Graph的类型，获取循环轴。
   * 若是HintGraph，则获取图属性中的axis信息；
   * 否则ImplGraph，则直接从node的属性中获取axis信息
   */
  static Status GetLoopAxis(const ge::AscNode &node, std::vector<int64_t> &axes) {
    auto attr = GetOrCreateGraphAttrsGroup(node.GetOwnerComputeGraph());
    GE_CHECK_NOTNULL(attr, "Get ascgraph type failed, attr is null.");
    if (attr->type == ge::AscGraphType::kImplGraph) {
      GELOGD("GetLoopAxis Impl graph, node.name=%s", node.GetNamePtr());
      axes.insert(axes.end(), node.attr.sched.axis.begin(), node.attr.sched.axis.end());
      return ge::SUCCESS;
    }
    axes.clear();
    axes.reserve(attr->axis.size());
    for (const auto &graph_axis : attr->axis) {
      GELOGD("[GetLoopAxis] node %s[%s] axis.id=%ld, axis.size=%s", node.GetTypePtr(), node.GetNamePtr(),
             graph_axis->id, graph_axis->size.Str().get());
      axes.push_back(graph_axis->id);
    }
    GELOGD("[GetLoopAxis] attr.axis.size=%zu, ids.size=%zu", attr->axis.size(), axes.size());
    return ge::SUCCESS;
  }

  /**
   * 从HintGraph的图属性中，获取repeats，直接获取每个轴的size即可。
   */
  static Status GetLoopRepeats(const ge::AscNode &node, std::vector<ascir::SizeExpr> &repeats) {
    auto attr = GetOrCreateGraphAttrsGroup(node.GetOwnerComputeGraph());
    GE_CHECK_NOTNULL(attr, "Get ascgraph type failed, attr is null.");
    repeats.clear();
    repeats.reserve(attr->axis.size());
    for (const auto &axis : attr->axis) {
      GELOGD("[GetLoopRepeats] node %s[%s] axis.id=%ld, axis.size=%s", node.GetTypePtr(), node.GetNamePtr(), axis->id,
             axis->size.Str().get());
      repeats.push_back(axis->size);
    }
    GELOGD("[GetLoopRepeats] attr.axis.size=%zu, repeats.size=%zu", attr->axis.size(), repeats.size());
    return ge::SUCCESS;
  }

  /**
   * 从HintGraph的图属性中，获取strides信息。需要根据repeat从后往前计算出来，尾轴的stride=1，然后依次累乘repeat
   */
  static Status GetReduceInputStrides(ge::AscNode &node, std::vector<ascir::SizeExpr> &strides) {
    auto attr = GetOrCreateGraphAttrsGroup(node.GetOwnerComputeGraph());
    GE_CHECK_NOTNULL(attr, "Get ascgraph type failed, attr is null.");
    if (attr->type == ge::AscGraphType::kImplGraph) {
      strides = node.inputs[0].attr.strides;
      return ge::SUCCESS;
    }
    strides.resize(attr->axis.size());
    ascir::SizeExpr basic_stride = ge::Symbol(1);
    for (int64_t i = static_cast<int64_t>(attr->axis.size()) - 1; i >= 0; --i) {
      if (ge::SymbolicUtils::StaticCheckEq(attr->axis[i]->size, ge::sym::kSymbolOne) == ge::TriBool::kTrue) {
        strides[i] = ge::Symbol(0);
      } else {
        strides[i] = basic_stride;
        basic_stride = basic_stride * attr->axis[i]->size;
      }
      GELOGD("[GetLoopStrides] node %s[%s] axis.id=%ld, axis.strides=%s", node.GetTypePtr(), node.GetNamePtr(),
             attr->axis[i]->id, strides[i].Str().get());
    }
    GELOGD("[GetLoopStrides] attr.axis.size=%zu", attr->axis.size());
    return ge::SUCCESS;
  }

  template <typename T>
  static Status GetNodeIrAttrValue(const ge::NodePtr &node, const string &attr_name, T &attr_value) {
    auto asc_node = std::dynamic_pointer_cast<ge::AscNode>(node);
    GE_ASSERT_NOTNULL(asc_node);
    GE_ASSERT_NOTNULL(asc_node->attr.ir_attr);
    asc_node->attr.ir_attr->GetAttrValue(attr_name, attr_value);  // 允许获取失败
    return ge::SUCCESS;
  }

  static Status GetNodeIrAttrIndex(const ge::NodePtr &node, int64_t &index) {
    return GetNodeIrAttrValue(node, "index", index);
  }

  static Status GetNodeIrAttrOffset(const ge::NodePtr &node, ge::Expression &offset) {
    auto asc_node = std::dynamic_pointer_cast<ge::AscNode>(node);
    GE_ASSERT_NOTNULL(asc_node);
    GE_ASSERT_NOTNULL(asc_node->attr.ir_attr);
    return asc_node->attr.ir_attr->GetAttrValue("offset", offset);
  }

  static bool IsAxisStrideAllZero(const std::vector<ascir::AxisId> &origin_ids,
                                  const std::vector<ascir::SizeExpr> &axis_strides,
                                  const std::vector<ascir::AxisId> &axis_ids) {
    for (uint64_t i = 0; i < axis_ids.size(); i++) {
      for (uint64_t j = 0; j < origin_ids.size(); j++) {
        if (axis_ids[i] != origin_ids[j]) {
          continue;
        }
        if (ge::SymbolicUtils::StaticCheckEq(axis_strides[j], ge::ops::Zero) == ge::TriBool::kTrue) {
          return true;
        }
      }
    }
    return false;
  }

  static bool IsBroadcastNeedMemUnique(const ascir::NodeView &node, const std::vector<ascir::AxisId> &axis_ids) {
    if (IsBuffer(node) || node->GetInDataNodesSize() == 0) {
      return false;
    }
    if (IsLoad(node)) {
      auto output = node->outputs[0];
      bool is_all_zero = IsAxisStrideAllZero(output.attr.axis, output.attr.strides, axis_ids);
      return is_all_zero;
    }

    for (const auto &in : node->inputs()) {
      GE_ASSERT_NOTNULL(in);
      auto prev_node = std::dynamic_pointer_cast<ge::AscNode>(in->anchor.GetOwnerNode());
      if (IsBroadcastNeedMemUnique(prev_node, axis_ids)) {
        return true;
      }
    }
    return false;
  }

  static bool IsNextNodeRemovePad(const ascir::NodeView &node);

  static bool IsContinuesBroadcast(const std::vector<ge::Expression> &in_repeats,
                                   const std::vector<ge::Expression> &out_repeats);

  static bool IsContinuesStrides(const std::vector<ge::Expression> &repeats,
                                 const std::vector<ge::Expression> &strides);

  static bool IsContinuesVecStrides(const ascir::NodeView &node);

  static bool IsVectorizedAxisContinuousInGM(const ge::AscTensorAttr &output_tensor);

  static bool IsLastAxisSliceLoad(const ge::AscNodePtr &node);

  static bool NotNeedAlignVectorStride(const ge::AscGraph &graph);

  static bool IsIntervalBroadcast(const std::vector<ge::Expression> &in_repeats,
                                  const std::vector<ge::Expression> &out_repeats);

  static bool GetTailAxisDataSize(const ge::AscNodePtr &node, uint32_t &size);
  static bool IsTailAxisLessThan(const ge::AscNodePtr &node, const uint32_t value);
  static bool IsTailAxisAlignedBy(const ge::AscNodePtr &node, const uint32_t align_bytes=32);

  static bool IsStaticShape(const ascir::NodeView &node);

  static bool IsStaticGraph(const ge::AscGraph &graph);

  static Status GetNonBrcInputTensor(const ascir::NodeView &node, const size_t index,
                                     std::unique_ptr<ge::AscTensor> &tensor);

  static bool IsComputeNodes(const ascir::NodeView &node) {
    for (auto &out_node : node->GetOutDataNodes()) {
      auto cur_node = std::dynamic_pointer_cast<ge::AscNode>(out_node);
      if (cur_node->attr.api.type == ge::ApiType::kAPITypeCompute && !IsBroadcast(cur_node)) {
        return true;
      }
    }
    return false;
  }

  static Status RemoveUnusedAxes(ge::AscGraph &graph);

  static void NormalizeAxisIds(const ge::AscGraph &graph);

  static Status GetVectorRepeats(const std::vector<ge::Expression> &repeats, const std::vector<int64_t> &axis,
                                 const std::vector<int64_t> &vector_axis, std::vector<ge::Expression> &vector_repeats);
  static Status GetNodeInputVectorRepeats(const ascir::NodeView &node, std::vector<ge::Expression> &vector_repeats);
  static Status GetNodeOutVectorRepeats(const ascir::NodeView &node, std::vector<ge::Expression> &vec_repeats);
  static Status GetConcatDim(const ge::AscNodePtr &node, size_t &concat_dim);

  static std::string AxesToString(const std::vector<ge::AxisPtr> &axes);
  static bool IsNormStruct(const ascir::ImplGraph &implGraph);
  static bool IsReduceArFullLoad(const ascir::ImplGraph &implGraph);
  static bool HasSameInput(const ge::AscNodePtr &node);
  static bool IsLastAxisReduce(const ascir::ImplGraph &impl_graph);
  static bool IsScalarBroadcastNode(const ascir::NodeView &node);
  static bool IsScalarBrc(const ge::AscNodePtr &node);
  static Status SwapInputIndex(const ascir::NodeView &node, const int32_t idx1, const int32_t idx2);
  static Status GetInputForTranspose(ge::AscNode &node, std::vector<ascir::AxisId> &input_axis);
  template<typename T>
  static ge::AscNodePtr FindFirstNodeOfType(const ge::AscGraph &graph) {
    for (const auto &node : graph.GetAllNodes()) {
      if (ge::ops::IsOps<T>(node)) {
        return node;
      }
    }
    return nullptr;
  }
  static Status RemoveNode(const ascir::ImplGraph &impl_graph, const ge::AscNodePtr &node,
    const ge::OutDataAnchorPtr &pre_out_anchor);
  static Status RemoveNodeDst(const ascir::ImplGraph &impl_graph, const ge::AscNodePtr &node,
    const ge::InDataAnchorPtr &next_in_anchor);
  static bool FindContinuesBroadcastNode(const ascir::NodeView &node, vector<ge::AscNodePtr> &continues_brc_nodes);

  static Status AddRemovePadAfter(ge::AscGraph &graph, const ge::AscNodePtr &node, ge::AscNodePtr &remove_pad_node,
    const int32_t idx = 0);
  static bool IsOutNodeWithMultiInputs(const ge::AscNodePtr &node);
  static Status ResolveDiffDim(const ge::AscNodePtr &node, size_t &diff_dim, bool &is_first_dim);
  static Status RecalculateStridesFromRepeats(const std::vector<ge::Expression> &repeats,
                                       std::vector<ge::Expression> &strides);
  static bool IsNeedDiscontinuousAligned(const ge::AscTensorAttr &attr);
  static Status ClearAllSizeVar(const ge::AscGraph &graph);
};
}  // namespace optimize

#endif  // OPTIMIZE_SCHEDULE_UTILS_H_
