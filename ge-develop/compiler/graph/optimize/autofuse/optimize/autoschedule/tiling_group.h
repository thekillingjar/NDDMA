/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __OPTIMIZE_AUTOSCHEDULE_TILING_GROUP_H__
#define __OPTIMIZE_AUTOSCHEDULE_TILING_GROUP_H__

#include "ascendc_ir.h"
#include "ascir.h"
#include "ascgen_log.h"
#include "autoschedule/axis_group.h"

namespace optimize::autoschedule {
enum GroupType : int32_t {
  GROUP_INVALID = 0x0,
  GROUP_X = 0x1,
  GROUP_Y = 0x2,
  GROUP_R = 0x4,
  GROUP_XY = GROUP_X | GROUP_Y,
  GROUP_YR = GROUP_Y | GROUP_R,
  GROUP_XYR = GROUP_X | GROUP_Y | GROUP_R
};

class TilingGroup {
  using AxisGroupGenFunc = std::function<int32_t(ge::AscNode &node, AxisGroup &axes_group)>;
  using AxisGroupMergeFunc =
      std::function<bool(AxisGroup &lhs_group, AxisGroup &rhs_group, const bool is_canfuse_call, const bool is_ge_call)>;

 public:
  TilingGroup() = default;

  /**
   * @brief 算子级别的axes group生成
   * 1. 对于每个节点，根据其compute type生成不同的axes group
   * 2. 将生成的axes group与当前的axes group进行合并生成一个新的axes group
   * 由于Ngroup会前后传播，因此采用的策略是先提取公共的Ngroup作为axis_group的Ngroup，
   * Merge前先每个TilingGroup中XYR中归属Ngroup的轴移除再重新按照XYR三个Group进行merge。
   */
  static Status GenTilingGroup(const ascir::ImplGraph &impl_graph, AxisGroup &tiling_group, bool is_reduce_fullload = false);
  /**
   * 由于MergeAxesGroup同时要对接canfuse和schedule，而canfuse时轴序还不确定，
   * 因此和前端约定，canfuse时先不考虑轴序，canfuse之后由前端保证轴序
   * @return
   */
  static bool MergeAxesGroup(AxisGroup &target, AxisGroup &src, const bool is_canfuse_call = false, const bool is_ge_call = false);
  static void NormGroup(AxisGroup &group);

 private:
  static Status GenAxisGroupForSingleNode(ge::AscNode &node, AxisGroup &axes_group, bool is_reduce_ar_fullLoad = false);

  static Status GenElewiseTilingGroup(ge::AscNode &node, AxisGroup &axes_group);
  static Status GenReduceTilingGroup(ge::AscNode &node, AxisGroup &axes_group);
  static Status GenReduceTilingGroupFullLoad(ge::AscNode &node, AxisGroup &axes_group);
  static Status GenConcatTilingGroup(ge::AscNode &node, AxisGroup &axes_group);
  static Status GenTransposeTilingGroup(ge::AscNode &node, AxisGroup &axes_group);
  static Status GenSplitTilingGroup(ge::AscNode &node, AxisGroup &axes_group);
  static GroupType GetGroupType(const AxisGroup &axes_group);
};
}  // namespace optimize::autoschedule

#endif  // __OPTIMIZE_AUTOSCHEDULE_TILING_GROUP_H__
