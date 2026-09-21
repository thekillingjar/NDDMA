/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATT_INPUT_OUTPUT_SETTERS_H_
#define ATT_INPUT_OUTPUT_SETTERS_H_
#include <string>

namespace att {

/**
 * @class InputOutputSetters
 * @brief Composition class managing input/output settings for solver generation
 *
 * This class encapsulates the data and methods for managing input/output definitions,
 * calls, tiling data subgroup names, and group uniqueness flags. It is designed to be
 * used as a member variable (composition) rather than through inheritance.
 */
class InputOutputSetters {
 public:
  InputOutputSetters() = default;
  ~InputOutputSetters() = default;

  // Setters
  void SetInputOutputDef(const std::string &value) {
    input_output_def_ = value;
  }
  void SetInputOutputCall(const std::string &value) {
    input_output_call_ = value;
  }
  void SetTilingDataSubGroupItemName(const std::string &value) {
    tiling_data_sub_group_item_name_ = value;
  }
  void SetIsUniGroup(bool value) {
    is_uniq_group_ = value;
  }

  // Getters
  const std::string &GetInputOutputDef() const {
    return input_output_def_;
  }
  const std::string &GetInputOutputCall() const {
    return input_output_call_;
  }
  const std::string &GetTilingDataSubGroupItemName() const {
    return tiling_data_sub_group_item_name_;
  }
  bool GetIsUniGroup() const {
    return is_uniq_group_;
  }

 private:
  std::string input_output_def_;
  std::string input_output_call_;
  std::string tiling_data_sub_group_item_name_;
  bool is_uniq_group_{true};
};

}  // namespace att

#endif  // ATT_INPUT_OUTPUT_SETTERS_H_
