/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SLICE_DATASLICE_DATA_SLICE_ADAPTER_H_
#define SLICE_DATASLICE_DATA_SLICE_ADAPTER_H_

#include <vector>
#include "ge/ge_api_types.h"
#include "framework/common/debug/ge_log.h"
#include "graph/axis_type_info.h"
#include "graph/utils/op_desc_utils.h"
#include "slice/data_slice_infer_base.h"

namespace ge {
enum class TransType {
  CUR_TO_ORI = 0,
  ORI_TO_CUR,
};

class DataSliceAdapter {
 public:
 // cut tensor: axis index: shape range
 using DataSliceType = std::vector<std::vector<std::vector<int64_t>>>;
 static void PrintOp(const OpDescPtr &op);
 static void PrintAxis(const OpDescPtr &op, const std::vector<AxisTypeInfo> &axis_type_vec,
   const std::string &type, bool print_ori);
 static void PrintSlice(const OpDescPtr &op, const DataSliceType &slice_info, const std::string &tensor_type,
  const std::string &tag);
 static void TransAxisInfo(const OpDescPtr &op, std::vector<AxisTypeInfo> &axis_type_vec);
 static Status GetOriOutputSlice(const OpDescPtr &op, const AxisTypeInfo &slice_info, DataSliceType &ori_output_slice);
 static Status GetCurInputSlice(const OpDescPtr &op, const AxisTypeInfo &slice_info,
    const DataSliceType &ori_input_slice, DataSliceType &cur_input_slice);
 static AxisTypeInfo GetTmpAxisTypeInfo(const AxisTypeInfo &slice_info);
 static bool CheckOriInfo(const OpDescPtr &op);
 static void SetOriOpInfo(OpDescPtr &op,
   std::vector<std::pair<Format, GeShape>> &cache_input_info,
   std::vector<std::pair<Format, GeShape>> &cache_output_info);
 static void SetCurOpInfo(OpDescPtr &op,
   const std::vector<std::pair<Format, GeShape>> &cache_input_info,
   const std::vector<std::pair<Format, GeShape>> &cache_output_info);

 private:
  static inline bool IsLogEnable(const int32_t module_name, const int32_t log_level) {
    const int32_t enable = CheckLogLevel(module_name, log_level);
    // 1:enable, 0:disable
    return (enable == 1);
  }
  static void PrintAxisItem(const AxisTypeInfo &axis_type, bool print_ori, std::stringstream &ss);
  static std::string GetTensorStr(const OpDesc::Vistor<ge::GeTensorDescPtr> all_tensor_desc);
  static std::vector<int64_t> TransAxisToNZ(const GeTensorDescPtr &tensor, int64_t axis);
  static bool CheckReshape(const GeTensorDescPtr &tensor, const std::string &reshape_type, int64_t axis,
    int64_t &format_match_axis);
  static bool CheckRank(size_t rank, size_t dim_num, const std::string &reshape_type);
  static std::vector<int64_t> TransAxisForSplit(const GeTensorDescPtr &tensor, const int64_t axis,
    size_t dim_num);
  static std::vector<int64_t> TransAxisForNoSplit(const GeTensorDescPtr &tensor, const int64_t axis,
    size_t dim_num);
  static bool IsFormatInSet(const Format format, const std::set<Format> &format_set);
  static std::vector<int64_t> TransAxis(const GeTensorDescPtr &tensor, int64_t ori_axis);
  static Status FixAxisTypeInfoToOne(AxisTypeInfo &axis_type_info);
  static bool ValidateRelateInputOutput(const AxisTypeInfo &axis_type_info);
  static Status TransAxisForInputTensor(const OpDescPtr &op, const std::string &axis_type_str,
    AxisTypeInfo &axis_type_info);
  static Status TransAxisForOutputTensor(const OpDescPtr &op, const std::string &axis_type_str,
    AxisTypeInfo &axis_type_info);
  static Status TransByAxisTypeStr(const OpDescPtr &op, const std::string &axis_type_str,
    AxisTypeInfo &axis_type_info);
  static Status TransReduceMean(const OpDescPtr &op, AxisTypeInfo &axis_type_info);
  static Status TransOther(const OpDescPtr &op, AxisTypeInfo &axis_type_info);
  static void BackupOriAxisTypeInfo(AxisTypeInfo &axis_type_info);
  static void ResetOriAxisTypeInfo(AxisTypeInfo &axis_type_info);
  static Status TransAxisByType(const AxisType axis_type, const OpDescPtr &op, AxisTypeInfo &axis_type_info);
  static int64_t SearchOriAxis(const std::vector<CutInfo> &ori_relate, int64_t tensor_idx, int64_t axis_idx);
  static bool ValidateAxisIndex(int64_t from_axis, const std::vector<std::vector<int64_t>> &slice_info,
    int64_t to_axis, const std::vector<std::vector<int64_t>> &cur_tensor_range);
  static Status TransSliceInfoToOriForElement(const OpDescPtr &op, const AxisTypeInfo &axis_type_info,
    const DataSliceType &slice_info_list, DataSliceType &ori_slice_info_list);
  static Status TransSliceInfoToCurForElement(const OpDescPtr &op, const AxisTypeInfo &axis_type_info,
    const DataSliceType &slice_info_list, DataSliceType &cur_slice_info_list);
  static AxisType GetAxisTypeForTransAxis(const AxisTypeInfo &axis_type_info);
  static AxisType GetAxisTypeForTransSlice(const AxisTypeInfo &axis_type_info);
  static Status TransSliceInfo(const OpDescPtr &op, const AxisTypeInfo &axis_type_info, TransType trans_type,
    const DataSliceType &slice_info_list, DataSliceType &out_slice_info_list);
};
}
#endif // SLICE_DATASLICE_DATA_SLICE_ADAPTER_H_
