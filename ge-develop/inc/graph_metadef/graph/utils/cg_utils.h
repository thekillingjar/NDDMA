/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_CG_UTILS_H
#define AUTOFUSE_CG_UTILS_H
#include <memory>
#include "ascendc_ir/ascend_reg_ops.h"
#include "ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "utils/axis_utils.h"
#include "utils/node_utils_ex.h"
#include "utils/dtype_transform_utils.h"
#include "graph/expression/const_values.h"
#include "graph/utils/op_desc_utils.h"

namespace ge {
namespace ascir {
using ge::Axis;
using ge::AxisId;
namespace cg {
constexpr char RELATED_OP[] = "RelatedOp";
#define THROW(condition) if (!(condition)) throw std::runtime_error("Check Failed: " #condition)
struct LoopOption {
  bool pad_tensor_axes_to_loop;
};
class CgContext {
 public:
  static CgContext *GetThreadLocalContext();
  static std::shared_ptr<CgContext> GetSharedThreadLocalContext();
  static void SetThreadLocalContext(const std::shared_ptr<CgContext> &context);

  void SetOption(const LoopOption &option);
  const LoopOption &GetOption() const;

  const std::vector<Axis> &GetLoopAxes() const;
  const std::vector<AxisId> &GetLoopAxisIds() const;
  void SetLoopAxes(std::vector<Axis> axes);
  void PushLoopAxis(const Axis &axis);
  void PopBackLoopAxis(const Axis &axis);

  void SetBlockLoopEnd(AxisId id);
  AxisId GetBlockLoopEnd() const;

  void SetVectorizedLoopEnd(AxisId id);
  AxisId GetVectorizedLoopEnd() const;

  void SetLoopEnd(AxisId id);
  AxisId GetLoopEnd() const;

 private:
  LoopOption option_;
  std::vector<Axis> loop_axes_;
  std::vector<AxisId> loop_axis_ids_cache_;  // 与 loop_axes_ 同源，避免反复创建

  AxisId block_loop_end_{0};
  AxisId vectorized_loop_end_{0};
  AxisId loop_end_{0};
};

class LoopGuard {
 public:
  explicit LoopGuard(const Axis &axis);
  ~LoopGuard();

  static std::unique_ptr<LoopGuard> Create(const Axis &axis) {
    return Create(axis, {});
  }

  static std::unique_ptr<LoopGuard> Create(const Axis &axis, const LoopOption &option);

 private:
  Axis axis_;
  std::shared_ptr<CgContext> context_;
};
using Axes = std::vector<Axis>;

class BlockLoopGuard {
  explicit BlockLoopGuard(std::vector<Axis> axes);
  ~BlockLoopGuard();
};

class VectorizedLoopGuard {
  explicit VectorizedLoopGuard(std::vector<Axis> axes);
  ~VectorizedLoopGuard();
};

#define INNER_LOOP_COUNTER_1(counter, axis)                                                       \
  for (auto guarder_##counter = ascir::cg::LoopGuard::Create(axis); guarder_##counter != nullptr; \
       guarder_##counter = nullptr)
#define INNER_LOOP_COUNTER(counter, axis) INNER_LOOP_COUNTER_1(counter, axis)
#define LOOP(axis) INNER_LOOP_COUNTER(__COUNTER__, axis)

#define OPTION_LOOP_COUNTER_1(counter, axis, option)                                                      \
  for (auto guarder_##counter = ascir::cg::LoopGuard::Create(axis, option); guarder_##counter != nullptr; \
       guarder_##counter = nullptr)
#define OPTION_LOOP_COUNTER(counter, axis, option) OPTION_LOOP_COUNTER_1(counter, axis, option)
#define OPTION_LOOP(axis, option) OPTION_LOOP_COUNTER(__COUNTER__, axis, option)

#define SET_SCHED_AXIS_IF_IN_CONTEXT(op)                           \
  do {                                                             \
    auto context = ascir::cg::CgContext::GetThreadLocalContext();  \
    if (context != nullptr) {                                      \
      (op).attr.sched.axis = (context)->GetLoopAxisIds();          \
      if (!((op).attr.sched.axis.empty()))   {                     \
        (op).attr.sched.loop_axis = ((op).attr.sched.axis.back()); \
      }                                                            \
    }                                                              \
  } while (0)

class CodeGenUtils {
 public:
  static int64_t GenNextExecId(const ge::Operator &op);
  static int64_t GenNextExecId(const ge::AscGraph &graph);
  static int64_t GenNextTensorId(const ge::Operator &op);
  static int64_t GenNextContainerId(const ge::Operator &op);
  static int64_t GenNextReuseId(const ge::Operator &op);
  static AscGraphAttr *GetOwnerGraphAscAttr(const Operator &op) {
    const auto &node = NodeUtilsEx::GetNodeFromOperator(op);
    GE_ASSERT_NOTNULL(node, "Node is null.");
    const auto &compute_graph = node->GetOwnerComputeGraph();
    GE_ASSERT_NOTNULL(compute_graph, "Compute graph is null.");

    auto attr = compute_graph->GetOrCreateAttrsGroup<AscGraphAttr>();
    GE_ASSERT_NOTNULL(attr, "AscGraphAttr is null.");
    return attr;
  }

  static AscNodeAttr *GetOwnerOpAscAttr(const Operator &op) {
    const auto &op_desc = OpDescUtils::GetOpDescFromOperator(op);
    GE_ASSERT_NOTNULL(op_desc, "op_desc is null.");
    return op_desc->GetOrCreateAttrsGroup<AscNodeAttr>();
  }
 private:
  static int64_t GenNextExecId(const ge::ComputeGraphPtr &graph);
};

inline bool PadOutputViewToSched(ge::AscOpOutput &output) {
  auto context = ascir::cg::CgContext::GetThreadLocalContext();
  if (context == nullptr || !context->GetOption().pad_tensor_axes_to_loop) {
    return true;
  }

  // check if need pad
  auto &sched_ids = context->GetLoopAxisIds();
  const auto &origin_axis_ids = output.axis;
  if (origin_axis_ids->size() == sched_ids.size()) {
    return *origin_axis_ids == sched_ids;
  }

  // calc pad indexes, if op_i not iter to the end, means the axis order in tensor is different from sched
  // max: pad, positive: index of origin_axis_ids
  std::vector<size_t> indexes;
  size_t op_i = 0U;
  for (auto sched_axis_id : sched_ids) {
    if (op_i < origin_axis_ids->size() && sched_axis_id == (*origin_axis_ids).at(op_i)) {
      indexes.push_back(op_i++);
    } else {
      indexes.push_back(std::numeric_limits<size_t>::max());
    }
  }
  if (op_i != origin_axis_ids->size()) {
    return false;
  }

  // do pad
  const auto &origin_repeats = output.repeats;
  const auto &origin_strides = output.strides;
  std::vector<AxisId> padded_axis_ids;
  std::vector<ge::Expression> padded_repeats;
  std::vector<ge::Expression> padded_strides;
  for (size_t i = 0U; i < indexes.size(); ++i) {
    op_i = indexes[i];
    if (op_i == std::numeric_limits<size_t>::max()) {
      padded_axis_ids.push_back(sched_ids.at(i));
      padded_repeats.push_back(sym::kSymbolOne);
      padded_strides.push_back(sym::kSymbolZero);
    } else {
      padded_axis_ids.push_back((*origin_axis_ids).at(op_i));
      padded_repeats.push_back((*origin_repeats).at(op_i));
      padded_strides.push_back((*origin_strides).at(op_i));
    }
  }

  *output.axis = padded_axis_ids;
  *output.repeats = padded_repeats;
  *output.strides = padded_strides;

  return true;
}
}  // namespace cg
}  // namespace ascir
}

#endif  // AUTOFUSE_CG_UTILS_H
