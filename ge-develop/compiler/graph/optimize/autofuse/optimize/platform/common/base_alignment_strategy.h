/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPTIMIZE_PLATFORM_COMMON_BASE_ALIGNMENT_STRATEGY_H
#define OPTIMIZE_PLATFORM_COMMON_BASE_ALIGNMENT_STRATEGY_H

#include <cstdint>
#include <queue>
#include "ascir.h"
#include "ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "optimize/schedule_utils.h"
#include "graph/utils/graph_utils.h"
#include "symbolizer/symbolic_utils.h"

namespace optimize {
constexpr uint32_t kAlignWidth = 32U;

enum class AlignmentType : uint32_t {
  kNotAligned = 0U,  // 不需要对齐
  kAligned,          // 尾轴对齐
  kDiscontinuous,    // 尾轴非连续
  kFixedNotAligned,  // 固定非对齐
  kInvalid,          // 无效值
};

struct TensorAlignState {
  AlignmentType align_type = AlignmentType::kNotAligned;
  bool conflict_with_output = false;
};

using AlignInferFunc = std::function<ge::Status(const ge::AscNodePtr &)>;

class BaseAlignmentStrategy {
 public:
  virtual ~BaseAlignmentStrategy() = default;
  explicit BaseAlignmentStrategy() = default;
  BaseAlignmentStrategy(const BaseAlignmentStrategy &) = delete;
  BaseAlignmentStrategy &operator=(const BaseAlignmentStrategy &) = delete;
  BaseAlignmentStrategy(BaseAlignmentStrategy &&) = delete;
  BaseAlignmentStrategy &operator=(BaseAlignmentStrategy &&) = delete;

  // 只允许load出现尾轴非连续
  ge::Status AlignVectorizedStrides(ascir::ImplGraph &impl_graph);
  static ge::Status SetVectorizedStridesForTensor(const ge::NodePtr &node, ge::AscTensorAttr &output_attr, const AlignmentType align_type);

 protected:
  virtual AlignmentType GetDefaultAlignmentType() = 0;

  virtual void InitAlignmentInferFunc();
  virtual ge::Status DefaultAlignmentInferFunc(const ge::AscNodePtr &node);
  virtual ge::Status BroadcastAlignmentInferFunc(const ge::AscNodePtr &node);
  virtual ge::Status ConcatAlignmentInferFunc(const ge::AscNodePtr &node);
  virtual ge::Status EleWiseAlignmentInferFunc(const ge::AscNodePtr &node);
  virtual ge::Status LoadAlignmentInferFunc(const ge::AscNodePtr &node);
  virtual ge::Status StoreAlignmentInferFunc(const ge::AscNodePtr &node);
  virtual ge::Status ReduceAlignmentInferFunc(const ge::AscNodePtr &node);
  virtual ge::Status SplitAlignmentInferFunc(const ge::AscNodePtr &node);

  static ge::Status SetAlignWidth(const ascir::ImplGraph &impl_graph);
  ge::Status InferAlignmentForOneNode(const ge::AscNodePtr &node);
  // 当前tensor的对齐行为只会出现在尾轴,如果没有新的对齐行为或者类型,该函数不应该修改
  ge::Status SetVectorizedStridesForOneNode(const ge::AscNodePtr &node);
  // 反向推导对齐逻辑
  virtual ge::Status BackPropagateAlignment(const ge::AscNodePtr &node,
                                            AlignmentType aligned_type = AlignmentType::kAligned);
  void SetAlignInfoForNodeInputs(AlignmentType aligned_type, ge::AscNode *node, std::set<ge::Node *> &visited_nodes,
                                 std::queue<ge::Node *> &node_queue);
  bool SetAlignInfoForNodeOutputs(AlignmentType aligned_type, ge::AscNode *node, std::set<ge::Node *> &visited_nodes,
                                  std::queue<ge::Node *> &node_queue);

  static ge::Status AddRemovePadForTailAxisDiscontinuousLoad(ascir::ImplGraph &impl_graph);
  ge::Status CheckIsNoNeedPad(const ge::AscNodePtr &node, ge::AscTensorAttr &out_attr, bool &is_no_need_pad) const;
  ge::Status AddPadForAlignmentConflictNode(ascir::ImplGraph &impl_graph);
  // 多输入elewise,有一个fix,需要向上传递fix状态,防止输入链路上被后续节点刷成align
  ge::Status BackPropagateFixUnAlignType(const ge::AscNodePtr &node);

  std::unordered_map<const ge::AscTensorAttr *, TensorAlignState> tensor_to_align_type_;
  std::map<ge::ComputeType, AlignInferFunc> compute_type_to_infer_func_;
  inline static uint32_t align_width_ = 32U;
};

bool IsLoadNeedAlignForReduce(const ge::AscNodePtr &node);
bool IsLoadNeedAlign(const ge::AscNodePtr &node_load);
}  // namespace optimize
#endif  // OPTIMIZE_PLATFORM_COMMON_BASE_ALIGNMENT_STRATEGY_H
