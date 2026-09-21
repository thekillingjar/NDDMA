/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/format/axis_name_util.h"
#include "common/format/axis_util.h"

namespace fe {
const std::map<ge::Format, GetAxisNameByAxisValueInfoPtr> AxisNameUtil::get_axis_name_except_func_map = {
    {ge::FORMAT_NCHW, std::make_shared<GetAxisNameByAxisValueInfo>(GetNCHWAxisExceptName)},
    {ge::FORMAT_NHWC, std::make_shared<GetAxisNameByAxisValueInfo>(GetNHWCAxisExceptName)},
    {ge::FORMAT_HWCN, std::make_shared<GetAxisNameByAxisValueInfo>(GetHWCNAxisExceptName)},
    {ge::FORMAT_CHWN, std::make_shared<GetAxisNameByAxisValueInfo>(GetCHWNAxisExceptName)},
    {ge::FORMAT_NDHWC, std::make_shared<GetAxisNameByAxisValueInfo>(GetNDHWCAxisExceptName)},
    {ge::FORMAT_NCDHW, std::make_shared<GetAxisNameByAxisValueInfo>(GetNCDHWAxisExceptName)},
    {ge::FORMAT_DHWCN, std::make_shared<GetAxisNameByAxisValueInfo>(GetDHWCNAxisExceptName)},
    {ge::FORMAT_DHWNC, std::make_shared<GetAxisNameByAxisValueInfo>(GetDHWNCAxisExceptName)}};

std::string AxisNameUtil::AxisNameToStr(std::vector<std::string> &axis_name) {
  std::string str;
  if (axis_name.empty()) {
    return str;
  }

  for (size_t i = 0; i < axis_name.size(); i++) {
    str += axis_name[i];
  }
  return str;
}

/** get reshape type according to format and axis value of reduce op
 *  1. get axis name except for reduce axis value,
 *     format: NCHW, axis_values: [0,1],
 *     the axis name is HW
 *  2. get reshape type according to axis name.
 *  the axis_except is [0, 3] */
std::string AxisNameUtil::GetReshapeType(const ge::Format &format, std::vector<int64_t> &axis_values) {
  std::string reshape_type;
  if (axis_values.empty()) {
    FE_LOGD("Axis value is empty, return default reshape type.");
    return reshape_type;
  }
  vector<std::string> axis_names;
  // get axis name except for reduce axis
  auto iter_get_axis_func = get_axis_name_except_func_map.find(format);
  if (iter_get_axis_func == get_axis_name_except_func_map.end()) {
    FE_LOGW("Can not get axis name of old format %u!", format);
    return reshape_type;
  }
  GetAxisNameByAxisValueInfoPtr get_axis_func = nullptr;
  FE_MAKE_SHARED(get_axis_func = iter_get_axis_func->second, return reshape_type);
  if (get_axis_func == nullptr) {
    return reshape_type;
  }
  (void)(*get_axis_func)(axis_values, axis_names);
  if (axis_names.empty()) {
    FE_LOGD("Axis name is empty, return default reshape type.");
    return reshape_type;
  }
  return AxisNameToStr(axis_names);
}

/** get value except redcue axis
 *  for example, a reduce op, its format is NCHW, axis value is [1, 2]
 *  the axis_except is [0, 3] */
std::vector<int64_t> AxisNameUtil::GetExceptAxisValue(vector<int64_t> &axis_values, const size_t &axis_nums) {
  // the axis value may be negative number, such as -1, -2
  for (size_t i = 0; i < axis_values.size(); i++) {
    if (axis_values[i] < 0) {
      axis_values[i] += static_cast<int64_t>(axis_nums);
    }
  }
  std::vector<int64_t> axis_except;
  for (int64_t i = 0; i < static_cast<int64_t>(axis_nums); i++) {
    auto iter = std::find(axis_values.begin(), axis_values.end(), i);
    if (iter != axis_values.end()) {
      continue;
    }
    axis_except.emplace_back(i);
  }
  return axis_except;
}

Status AxisNameUtil::GetNCHWAxisExceptName(std::vector<int64_t> &axis_values, std::vector<std::string> &axis_name) {
  std::vector<int64_t> except_axis = GetExceptAxisValue(axis_values, DIM_DEFAULT_SIZE);
  for (size_t i = 0; i < except_axis.size(); i++) {
    int64_t axis_value_a = except_axis[i];
    if (axis_value_a == NCHW_DIM_N) {
      axis_name.emplace_back(N_AXIS_NAME);
    } else if (axis_value_a == NCHW_DIM_C) {
      axis_name.emplace_back(C_AXIS_NAME);
    } else if (axis_value_a == NCHW_DIM_H) {
      axis_name.emplace_back(H_AXIS_NAME);
    } else if (axis_value_a == NCHW_DIM_W) {
      axis_name.emplace_back(W_AXIS_NAME);
    }
  }
  return SUCCESS;
}

Status AxisNameUtil::GetNHWCAxisExceptName(std::vector<int64_t> &axis_values, std::vector<std::string> &axis_name) {
  std::vector<int64_t> except_axis = GetExceptAxisValue(axis_values, DIM_DEFAULT_SIZE);
  for (size_t i = 0; i < except_axis.size(); i++) {
    int64_t axis_value_b = except_axis[i];
    if (axis_value_b == NHWC_DIM_N) {
      axis_name.emplace_back(N_AXIS_NAME);
    } else if (axis_value_b == NHWC_DIM_H) {
      axis_name.emplace_back(H_AXIS_NAME);
    } else if (axis_value_b == NHWC_DIM_W) {
      axis_name.emplace_back(W_AXIS_NAME);
    } else if (axis_value_b == NHWC_DIM_C) {
      axis_name.emplace_back(C_AXIS_NAME);
    }
  }
  return SUCCESS;
}

Status AxisNameUtil::GetHWCNAxisExceptName(std::vector<int64_t> &axis_values, std::vector<std::string> &axis_name) {
  std::vector<int64_t> except_axis = GetExceptAxisValue(axis_values, DIM_DEFAULT_SIZE);
  for (size_t i = 0; i < except_axis.size(); i++) {
    int64_t axis_value_c = except_axis[i];
    if (axis_value_c == HWCN_DIM_H) {
      axis_name.emplace_back(H_AXIS_NAME);
    } else if (axis_value_c == HWCN_DIM_W) {
      axis_name.emplace_back(W_AXIS_NAME);
    } else if (axis_value_c == HWCN_DIM_C) {
      axis_name.emplace_back(C_AXIS_NAME);
    } else if (axis_value_c == HWCN_DIM_N) {
      axis_name.emplace_back(N_AXIS_NAME);
    }
  }
  return SUCCESS;
}

Status AxisNameUtil::GetCHWNAxisExceptName(std::vector<int64_t> &axis_values, std::vector<std::string> &axis_name) {
  std::vector<int64_t> except_axis = GetExceptAxisValue(axis_values, DIM_DEFAULT_SIZE);
  for (size_t i = 0; i < except_axis.size(); i++) {
    int64_t axis_value_d = except_axis[i];
    if (axis_value_d == CHWN_DIM_C) {
      axis_name.emplace_back(C_AXIS_NAME);
    } else if (axis_value_d == CHWN_DIM_H) {
      axis_name.emplace_back(H_AXIS_NAME);
    } else if (axis_value_d == CHWN_DIM_W) {
      axis_name.emplace_back(W_AXIS_NAME);
    } else if (axis_value_d == CHWN_DIM_N) {
      axis_name.emplace_back(N_AXIS_NAME);
    }
  }
  return SUCCESS;
}

Status AxisNameUtil::GetNDHWCAxisExceptName(std::vector<int64_t> &axis_values, std::vector<std::string> &axis_name) {
  std::vector<int64_t> except_axis = GetExceptAxisValue(axis_values, DIMENSION_NUM_FIVE);
  for (size_t i = 0; i < except_axis.size(); i++) {
    int64_t axis_value_d = except_axis[i];
    if (axis_value_d == NDHWC_DIM_N) {
      axis_name.emplace_back(N_AXIS_NAME);
    } else if (axis_value_d == NDHWC_DIM_D) {
      axis_name.emplace_back(D_AXIS_NAME);
    } else if (axis_value_d == NDHWC_DIM_H) {
      axis_name.emplace_back(H_AXIS_NAME);
    } else if (axis_value_d == NDHWC_DIM_W) {
      axis_name.emplace_back(W_AXIS_NAME);
    } else if (axis_value_d == NDHWC_DIM_C) {
      axis_name.emplace_back(C_AXIS_NAME);
    }
  }
  return SUCCESS;
}

Status AxisNameUtil::GetNCDHWAxisExceptName(std::vector<int64_t> &axis_values, std::vector<std::string> &axis_name) {
  std::vector<int64_t> except_axis = GetExceptAxisValue(axis_values, DIMENSION_NUM_FIVE);
  for (size_t i = 0; i < except_axis.size(); i++) {
    int64_t axis_value_d = except_axis[i];
    if (axis_value_d == NCDHW_DIM_N) {
      axis_name.emplace_back(N_AXIS_NAME);
    } else if (axis_value_d == NCDHW_DIM_C) {
      axis_name.emplace_back(C_AXIS_NAME);
    } else if (axis_value_d == NCDHW_DIM_D) {
      axis_name.emplace_back(D_AXIS_NAME);
    } else if (axis_value_d == NCDHW_DIM_H) {
      axis_name.emplace_back(H_AXIS_NAME);
    } else if (axis_value_d == NCDHW_DIM_W) {
      axis_name.emplace_back(W_AXIS_NAME);
    }
  }
  return SUCCESS;
}

Status AxisNameUtil::GetDHWCNAxisExceptName(std::vector<int64_t> &axis_values, std::vector<std::string> &axis_name) {
  std::vector<int64_t> except_axis = GetExceptAxisValue(axis_values, DIMENSION_NUM_FIVE);
  for (size_t i = 0; i < except_axis.size(); i++) {
    int64_t axis_value_d = except_axis[i];
    if (axis_value_d == DHWCN_DIM_D) {
      axis_name.emplace_back(D_AXIS_NAME);
    } else if (axis_value_d == DHWCN_DIM_H) {
      axis_name.emplace_back(H_AXIS_NAME);
    } else if (axis_value_d == DHWCN_DIM_W) {
      axis_name.emplace_back(W_AXIS_NAME);
    } else if (axis_value_d == DHWCN_DIM_C) {
      axis_name.emplace_back(C_AXIS_NAME);
    } else if (axis_value_d == DHWCN_DIM_N) {
      axis_name.emplace_back(N_AXIS_NAME);
    }
  }
  return SUCCESS;
}

Status AxisNameUtil::GetDHWNCAxisExceptName(std::vector<int64_t> &axis_values, std::vector<std::string> &axis_name) {
  std::vector<int64_t> except_axis = GetExceptAxisValue(axis_values, DIMENSION_NUM_FIVE);
  for (size_t i = 0; i < except_axis.size(); i++) {
    int64_t axis_value_d = except_axis[i];
    if (axis_value_d == DHWNC_DIM_D) {
      axis_name.emplace_back(D_AXIS_NAME);
    } else if (axis_value_d == DHWNC_DIM_H) {
      axis_name.emplace_back(H_AXIS_NAME);
    } else if (axis_value_d == DHWNC_DIM_W) {
      axis_name.emplace_back(W_AXIS_NAME);
    } else if (axis_value_d == DHWNC_DIM_N) {
      axis_name.emplace_back(N_AXIS_NAME);
    } else if (axis_value_d == DHWNC_DIM_C) {
      axis_name.emplace_back(C_AXIS_NAME);
    }
  }
  return SUCCESS;
}

Status AxisNameUtil::CheckFormatConveByExpect(const ge::Format &ori_format, const std::string &reshape_type,
                                              const vector<int64_t> axis_values) {
  auto iter_get_axis_func = get_axis_name_except_func_map.find(ori_format);
  if (iter_get_axis_func != get_axis_name_except_func_map.end() && reshape_type.empty() && !axis_values.empty()) {
    const auto axis_names = GetAxisNames(axis_values, iter_get_axis_func);
    if (!axis_names.empty()) {
      return FAILED;
    }
  }
  return SUCCESS;
}

std::vector<std::string> AxisNameUtil::GetAxisNames(
    vector<int64_t> axis_values,
    const std::map<ge::Format, GetAxisNameByAxisValueInfoPtr>::const_iterator &iter_get_axis_func) {
  vector<std::string> axis_names;
  const GetAxisNameByAxisValueInfoPtr get_axis_func = iter_get_axis_func->second;
  if (get_axis_func == nullptr) {
    return axis_names;
  }
  (void)(*get_axis_func)(axis_values, axis_names);
  return axis_names;
}

Status AxisNameUtil::GetNewAxisAttributeValue(const ge::OpDesc &op_desc, const ge::Format &origin_format,
                                              const ge::Format &current_format, const ge::GeShape &origin_shape,
                                              std::vector<int64_t> &axis_index_vec) {
  // get old axis name
  std::vector<std::string> axis_names;
  if (GetOriginalAxisName(op_desc, origin_format, origin_shape, axis_names) != SUCCESS) {
    REPORT_INNER_ERR_MSG(EM_INNER_ERROR.c_str(),
                       "[GraphOpt][SetAxis][GetAxisName][Op %s,type=%s]:Get axis name for ori format %d failed!",
                       op_desc.GetName().c_str(), op_desc.GetType().c_str(), origin_format);
    FE_LOGW("[GraphOpt][SetAxis][GetAxisName][Op name=%s,type=%s]:Get axis name for format %u failed!",
            op_desc.GetName().c_str(), op_desc.GetType().c_str(), origin_format);
    return FAILED;
  }
  // get new axis info
  if (GetNewAxisInfoByName(op_desc, current_format, axis_names, axis_index_vec) != SUCCESS) {
    REPORT_INNER_ERR_MSG(EM_INNER_ERROR.c_str(),
                       "[GraphOpt][SetAxis][GetAxisName][Op %s,type=%s]:Get axis name for current format %d failed!",
                       op_desc.GetName().c_str(), op_desc.GetType().c_str(), current_format);
    FE_LOGW("[GraphOpt][SetAxis][GetAxisName][Op name=%s,type=%s]:Get axis name for ori format %u failed!",
            op_desc.GetName().c_str(), op_desc.GetType().c_str(), current_format);
    return FAILED;
  }
  return SUCCESS;
}

Status AxisNameUtil::GetNewAxisInfoByName(const ge::OpDesc &op_desc, const ge::Format &format,
                                          std::vector<std::string> &axis_name, std::vector<int64_t> &axis_index_vec) {
  for (const auto &i : axis_name) {
    auto iter = FORMAT_AXIS_NAME_NUMBER_MAP.find(format);
    if (iter != FORMAT_AXIS_NAME_NUMBER_MAP.end()) {
      auto axis_name_number_map = iter->second;
      auto iter_axis_number = axis_name_number_map.find(i);
      if (iter_axis_number != axis_name_number_map.end()) {
        for (auto ele : iter_axis_number->second) {
          axis_index_vec.emplace_back(ele);
        }
      }
    }
  }

  for (const auto &axis_index:axis_index_vec) {
    FE_LOGD("Get reduce op [%s] axis new value is [%ld].", op_desc.GetName().c_str(), axis_index);
  }
  return SUCCESS;
}

Status AxisNameUtil::GetOriginalAxisName(const ge::OpDesc &op_desc, const ge::Format &format,
                                         const ge::GeShape &origin_shape, std::vector<std::string> &axis_name_vec) {
  Status ret = FAILED;
  std::vector<int64_t> axis_index_vec;

  if (AxisUtil::GetOriginAxisAttribute(op_desc, origin_shape, axis_index_vec) != SUCCESS) {
    FE_LOGW("Get reduce op [%s] new axis info failed!", op_desc.GetName().c_str());
    return FAILED;
  }

  if (format == ge::FORMAT_NCHW) {
    ret = GetNCHWAxisName(axis_index_vec, axis_name_vec);
  } else if (format == ge::FORMAT_NHWC) {
    ret = GetNHWCAxisName(axis_index_vec, axis_name_vec);
  } else if (format == ge::FORMAT_HWCN) {
    ret = GetHWCNAxisName(axis_index_vec, axis_name_vec);
  } else if (format == ge::FORMAT_CHWN) {
    ret = GetCHWNAxisName(axis_index_vec, axis_name_vec);
  } else if (format == ge::FORMAT_NDHWC) {
    ret = GetNDHWCAxisName(axis_index_vec, axis_name_vec);
  } else if (format == ge::FORMAT_NCDHW) {
    ret = GetNCDHWAxisName(axis_index_vec, axis_name_vec);
  } else if (format == ge::FORMAT_DHWCN) {
    ret = GetDHWCNAxisName(axis_index_vec, axis_name_vec);
  }

  for (const auto &axis_name: axis_name_vec) {
    FE_LOGD("Get reduce op [%s] axis name is [%s].", op_desc.GetName().c_str(), axis_name.c_str());
  }
  return ret;
}

Status AxisNameUtil::GetNCHWAxisName(std::vector<int64_t> &axis_values, std::vector<std::string> &axis_name) {
  for (size_t i = 0; i < axis_values.size(); i++) {
    int64_t axis_value_e = axis_values[i];
    if (axis_value_e == NCHW_DIM_N) {
      axis_name.emplace_back(N_AXIS_NAME);
    } else if (axis_value_e == NCHW_DIM_C) {
      axis_name.emplace_back(C_AXIS_NAME);
    } else if (axis_value_e == NCHW_DIM_H) {
      axis_name.emplace_back(H_AXIS_NAME);
    } else if (axis_value_e == NCHW_DIM_W) {
      axis_name.emplace_back(W_AXIS_NAME);
    }
  }
  return SUCCESS;
}

Status AxisNameUtil::GetNHWCAxisName(std::vector<int64_t> &axis_values, std::vector<std::string> &axis_name) {
  for (size_t i = 0; i < axis_values.size(); i++) {
    int64_t axis_value_f = axis_values[i];
    if (axis_value_f == NHWC_DIM_N) {
      axis_name.emplace_back(N_AXIS_NAME);
    } else if (axis_value_f == NHWC_DIM_H) {
      axis_name.emplace_back(H_AXIS_NAME);
    } else if (axis_value_f == NHWC_DIM_W) {
      axis_name.emplace_back(W_AXIS_NAME);
    } else if (axis_value_f == NHWC_DIM_C) {
      axis_name.emplace_back(C_AXIS_NAME);
    }
  }
  return SUCCESS;
}

Status AxisNameUtil::GetHWCNAxisName(std::vector<int64_t> &axis_values, std::vector<std::string> &axis_name) {
  for (size_t i = 0; i < axis_values.size(); i++) {
    int64_t axis_value_g = axis_values[i];
    if (axis_value_g == HWCN_DIM_H) {
      axis_name.emplace_back(H_AXIS_NAME);
    } else if (axis_value_g == HWCN_DIM_W) {
      axis_name.emplace_back(W_AXIS_NAME);
    } else if (axis_value_g == HWCN_DIM_C) {
      axis_name.emplace_back(C_AXIS_NAME);
    } else if (axis_value_g == HWCN_DIM_N) {
      axis_name.emplace_back(N_AXIS_NAME);
    }
  }
  return SUCCESS;
}

Status AxisNameUtil::GetCHWNAxisName(std::vector<int64_t> &axis_values, std::vector<std::string> &axis_name) {
  for (size_t i = 0; i < axis_values.size(); i++) {
    int64_t axis_value_h = axis_values[i];
    if (axis_value_h == CHWN_DIM_C) {
      axis_name.emplace_back(C_AXIS_NAME);
    } else if (axis_value_h == CHWN_DIM_H) {
      axis_name.emplace_back(H_AXIS_NAME);
    } else if (axis_value_h == CHWN_DIM_W) {
      axis_name.emplace_back(W_AXIS_NAME);
    } else if (axis_value_h == CHWN_DIM_N) {
      axis_name.emplace_back(N_AXIS_NAME);
    }
  }
  return SUCCESS;
}

Status AxisNameUtil::GetNDHWCAxisName(std::vector<int64_t> &axis_values, std::vector<std::string> &axis_name) {
  for (size_t i = 0; i < axis_values.size(); i++) {
    int64_t axis_value_i = axis_values[i];
    if (axis_value_i == NDHWC_DIM_C) {
      axis_name.emplace_back(C_AXIS_NAME);
    } else if (axis_value_i == NDHWC_DIM_H) {
      axis_name.emplace_back(H_AXIS_NAME);
    } else if (axis_value_i == NDHWC_DIM_W) {
      axis_name.emplace_back(W_AXIS_NAME);
    } else if (axis_value_i == NDHWC_DIM_N) {
      axis_name.emplace_back(N_AXIS_NAME);
    } else if (axis_value_i == NDHWC_DIM_D) {
      axis_name.emplace_back(D_AXIS_NAME);
    }
  }
  return SUCCESS;
}

Status AxisNameUtil::GetNCDHWAxisName(std::vector<int64_t> &axis_values, std::vector<std::string> &axis_name) {
  for (size_t i = 0; i < axis_values.size(); i++) {
    int64_t axis_value_j = axis_values[i];
    if (axis_value_j == NCDHW_DIM_C) {
      axis_name.emplace_back(C_AXIS_NAME);
    } else if (axis_value_j == NCDHW_DIM_H) {
      axis_name.emplace_back(H_AXIS_NAME);
    } else if (axis_value_j == NCDHW_DIM_W) {
      axis_name.emplace_back(W_AXIS_NAME);
    } else if (axis_value_j == NCDHW_DIM_N) {
      axis_name.emplace_back(N_AXIS_NAME);
    } else if (axis_value_j == NCDHW_DIM_D) {
      axis_name.emplace_back(D_AXIS_NAME);
    }
  }
  return SUCCESS;
}

Status AxisNameUtil::GetDHWCNAxisName(std::vector<int64_t> &axis_values, std::vector<std::string> &axis_name) {
  for (size_t i = 0; i < axis_values.size(); i++) {
    int64_t axis_value_k = axis_values[i];
    if (axis_value_k == DHWCN_DIM_C) {
      axis_name.emplace_back(C_AXIS_NAME);
    } else if (axis_value_k == DHWCN_DIM_H) {
      axis_name.emplace_back(H_AXIS_NAME);
    } else if (axis_value_k == DHWCN_DIM_W) {
      axis_name.emplace_back(W_AXIS_NAME);
    } else if (axis_value_k == DHWCN_DIM_N) {
      axis_name.emplace_back(N_AXIS_NAME);
    } else if (axis_value_k == DHWCN_DIM_D) {
      axis_name.emplace_back(D_AXIS_NAME);
    }
  }
  return SUCCESS;
}
};  // namespace fe
