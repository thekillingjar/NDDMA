/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_REUSE_GROUP_UTILS_H
#define AUTOFUSE_REUSE_GROUP_UTILS_H

#include "base/model_info.h"
namespace att {
class ReuseGroupUtils {
 public:
  ReuseGroupUtils() = default;
  ~ReuseGroupUtils() = default;
  static bool IsGroupGraphsEquivalent(const std::vector<ge::AscGraph> &graphs_to,
                                      const std::vector<ge::AscGraph> &graphs_from,
                                      ReuseScheduleGroupInfo &group_info_to,
                                      ReuseScheduleGroupInfo &group_info_from);
  static ge::Status InitReuseScheduleGroup(const ScheduleGroupIdent &group_ident,
                                           TilingModelInfo &group_tiling_model_info);
  static ge::Status MergeAllReusableGroups(
      const std::vector<std::vector<std::vector<std::vector<ge::AscGraph>>>> &all_graphs_lists,
      FusedParsedScheduleResult &out_fused_schedule_result);

 private:
  static ge::Status MergeEqualReusableGroups(
    const ReuseScheduleGroupPtr &group_to, const ReuseScheduleGroupPtr &group_from,
    const std::vector<std::vector<std::vector<std::vector<ge::AscGraph>>>> &all_graphs_lists,
    FusedParsedScheduleResult &out_fused_schedule_result);
};
}
#endif  // AUTOFUSE_REUSE_GROUP_UTILS_H
