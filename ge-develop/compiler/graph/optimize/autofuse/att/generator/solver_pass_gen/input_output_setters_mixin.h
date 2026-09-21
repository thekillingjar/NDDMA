/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software; you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef ATT_INPUT_OUTPUT_SETTERS_MIXIN_H_
#define ATT_INPUT_OUTPUT_SETTERS_MIXIN_H_

#include "input_output_setters.h"

namespace att {

/**
 * @class InputOutputSettersMixin
 * @brief CRTP (Curiously Recurring Template Pattern) mixin for InputOutputSetters delegation
 *
 * This template class provides delegation methods to the InputOutputSetters composition member.
 * It eliminates code duplication by providing these methods through inheritance using CRTP.
 * Derived classes should inherit as: class Derived : public InputOutputSettersMixin<Derived>
 *
 * @tparam Derived The derived class type (CRTP pattern)
 */
template <typename Derived>
class InputOutputSettersMixin {
 public:
  InputOutputSettersMixin() = default;
  ~InputOutputSettersMixin() = default;

  // Setters - delegate to composition member
  void SetInputOutputDef(const std::string& value) {
    input_output_setters_.SetInputOutputDef(value);
  }
  void SetInputOutputCall(const std::string& value) {
    input_output_setters_.SetInputOutputCall(value);
  }
  void SetTilingDataSubGroupItemName(const std::string& value) {
    input_output_setters_.SetTilingDataSubGroupItemName(value);
  }
  void SetIsUniGroup(bool value) {
    input_output_setters_.SetIsUniGroup(value);
  }

  // Getters - delegate to composition member
  const std::string& GetInputOutputDef() const {
    return input_output_setters_.GetInputOutputDef();
  }
  const std::string& GetInputOutputCall() const {
    return input_output_setters_.GetInputOutputCall();
  }
  const std::string& GetTilingDataSubGroupItemName() const {
    return input_output_setters_.GetTilingDataSubGroupItemName();
  }
  bool GetIsUniGroup() const {
    return input_output_setters_.GetIsUniGroup();
  }

 protected:
  // Composition member - accessible to derived classes
  InputOutputSetters input_output_setters_;
};

}  // namespace att

#endif  // ATT_INPUT_OUTPUT_SETTERS_MIXIN_H_
