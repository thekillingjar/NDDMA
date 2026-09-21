/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_CAN_NOT_FUSE_REASON_CODE_H
#define AUTOFUSE_CAN_NOT_FUSE_REASON_CODE_H

#include <cstdint>

namespace ge {
enum class NotFuseReason : uint32_t {
  kNoSharedData = 1,
  kWillCreateCycle,
  kIncreasePeakMemory,
  kGetSubgraphFailed,
  kUpdateSubgraphOutputAttrFailed,
  kUnifySubgraphAxisFailed,
  kInputNumsExceedThreshold,
  kGetAscgraphAxisGroupFailed,
  kMergeAxisGroupFailed,
  kMaxFusionNodesSizeExceedThreshold,
  kConcatCanNotFuseBackward,
  kConcatCanNotFuseWithReduce,
  kConcatCanNotFuseWithConcat,
  kConcatCanNotBackwardFuseNonSimplestPointwise,
  kFusedAscBackendCanNotBackwardFuseFromNonConcat,
  kFusedAscBackendCanNotBackwardFuseBroadcast,
  kConcatNodeSchedAxisNotEqual,
  kConcatNodeSchedAxisSizeNotEqual,
  kGatherCanNotFuseForward,
  kGatherCanNotFuseWithReduce,
  kGatherCanNotFuseWithConcat,
  kGatherNodeAxisSizeNotEqual,
  kGatherNextNodeIsPointWiseAndInputHasViewOp,
  kGatherNextNodeIsReduceAndInputHasViewOp,
  kGatherNextNodeIsConcatAndInputHasViewOp,
  kReduceCanNotFuseHorizontal,
  kTransposeCanNotFuseWithNotPointWise,
  kSliceHasOnlyHorizontalLink,
  kCanNotMergeAscGraph,
  kVectorCoreNumNotEqual,
  kNodeInputHasSplit,
  kGatherCanOnlyVerticallyFuseWithElementwise,
  kSplitCanNotFuseForward,
  kSplitCanNotFuseConcatBackward,
  kReduceCanFuseWithin3SubNodes,
  kReduceCanOnlyBackwardFuse3Elementwise,
  kElementwiseHasMoreThanOneInput,
  kElementwiseOnlyOneInDataAnchorButOtherInputIsScalar,
  kSplitCanNotFuseSplitHorizontal,
  kSplitCanNotFuseReduction,
  kSplitCanNotFuseHorizontal,
  kCubeCanNotFuseForward,
  kCubeCanNotFuseWithViewElementwise,
  kCubeCanNotFuseWithUnsupportedType,
  kCubeCanNotFuseWithNotComputeNode,
  kCubeCanNotFuseInThisChip,
  kFusedSliceHasViewOp,
  kCubeCanNotFuseWithNotElementwise,
  kCubeCanNotFuseHoriZontal,
  kSplitLowFuseRatio
};

inline const char* NotFuseReasonCode(NotFuseReason reason) noexcept {
  switch (reason) {
    case NotFuseReason::kNoSharedData:
      return "NF1001";
    case NotFuseReason::kWillCreateCycle:
      return "NF1002";
    case NotFuseReason::kIncreasePeakMemory:
      return "NF1003";
    case NotFuseReason::kGetSubgraphFailed:
      return "NF1004";
    case NotFuseReason::kUpdateSubgraphOutputAttrFailed:
      return "NF1005";
    case NotFuseReason::kUnifySubgraphAxisFailed:
      return "NF1006";
    case NotFuseReason::kInputNumsExceedThreshold:
      return "NF1007";
    case NotFuseReason::kGetAscgraphAxisGroupFailed:
      return "NF1008";
    case NotFuseReason::kMergeAxisGroupFailed:
      return "NF1009";
    case NotFuseReason::kMaxFusionNodesSizeExceedThreshold:
      return "NF1010";
    case NotFuseReason::kConcatCanNotFuseBackward:
      return "NF1011";
    case NotFuseReason::kConcatCanNotFuseWithReduce:
      return "NF1012";
    case NotFuseReason::kConcatCanNotFuseWithConcat:
      return "NF1013";
    case NotFuseReason::kConcatNodeSchedAxisNotEqual:
      return "NF1014";
    case NotFuseReason::kConcatNodeSchedAxisSizeNotEqual:
      return "NF1015";
    case NotFuseReason::kGatherCanNotFuseForward:
      return "NF1016";
    case NotFuseReason::kGatherCanNotFuseWithReduce:
      return "NF1017";
    case NotFuseReason::kGatherCanNotFuseWithConcat:
      return "NF1018";
    case NotFuseReason::kGatherNodeAxisSizeNotEqual:
      return "NF1019";
    case NotFuseReason::kGatherNextNodeIsPointWiseAndInputHasViewOp:
      return "NF1020";
    case NotFuseReason::kReduceCanNotFuseHorizontal:
      return "NF1021";
    case NotFuseReason::kTransposeCanNotFuseWithNotPointWise:
      return "NF1022";
    case NotFuseReason::kSliceHasOnlyHorizontalLink:
      return "NF1023";
    case NotFuseReason::kCanNotMergeAscGraph:
      return "NF1024";
    case NotFuseReason::kVectorCoreNumNotEqual:
      return "NF1025";
    case NotFuseReason::kNodeInputHasSplit:
      return "NF1026";
    case NotFuseReason::kGatherCanOnlyVerticallyFuseWithElementwise:
      return "NF1027";
    case NotFuseReason::kSplitCanNotFuseForward:
      return "NF1028";
    case NotFuseReason::kSplitCanNotFuseConcatBackward:
      return "NF1029";
    case NotFuseReason::kReduceCanFuseWithin3SubNodes:
      return "NF1030";
    case NotFuseReason::kReduceCanOnlyBackwardFuse3Elementwise:
      return "NF1031";
    case NotFuseReason::kElementwiseHasMoreThanOneInput:
      return "NF1032";
    case NotFuseReason::kConcatCanNotBackwardFuseNonSimplestPointwise:
      return "NF1033";
    case NotFuseReason::kFusedAscBackendCanNotBackwardFuseFromNonConcat:
      return "NF1034";
    case NotFuseReason::kFusedAscBackendCanNotBackwardFuseBroadcast:
      return "NF1035";
    case NotFuseReason::kElementwiseOnlyOneInDataAnchorButOtherInputIsScalar:
      return "NF1036";
    case NotFuseReason::kSplitCanNotFuseSplitHorizontal:
      return "NF1037";
    case NotFuseReason::kSplitCanNotFuseReduction:
      return "NF1038";
    case NotFuseReason::kSplitCanNotFuseHorizontal:
      return "NF1039";
    case NotFuseReason::kGatherNextNodeIsReduceAndInputHasViewOp:
      return "NF1040";
    case NotFuseReason::kGatherNextNodeIsConcatAndInputHasViewOp:
      return "NF1041";
    case NotFuseReason::kCubeCanNotFuseForward:
      return "NF1042";
    case NotFuseReason::kCubeCanNotFuseWithViewElementwise:
      return "NF1043";
    case NotFuseReason::kCubeCanNotFuseWithUnsupportedType:
      return "NF1044";
    case NotFuseReason::kCubeCanNotFuseWithNotComputeNode:
      return "NF1045";
    case NotFuseReason::kCubeCanNotFuseInThisChip:
      return "NF1046";
    case NotFuseReason::kFusedSliceHasViewOp:
      return "NF1047";
    case NotFuseReason::kCubeCanNotFuseWithNotElementwise:
      return "NF1048";
    case NotFuseReason::kCubeCanNotFuseHoriZontal:
      return "NF1049";
    case NotFuseReason::kSplitLowFuseRatio:
      return "NF1050";
  }
  return "NF9999";  // unknown
}
}  // namespace ge

#endif  // AUTOFUSE_CAN_NOT_FUSE_REASON_CODE_H
