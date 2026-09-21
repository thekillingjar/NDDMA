/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLOW_FUNC_ATTR_VALUE_H
#define FLOW_FUNC_ATTR_VALUE_H

#include <vector>
#include <memory>
#include <string>
#include "flow_func_defines.h"
#include "tensor_data_type.h"
#include "ascend_string.h"

namespace FlowFunc {
class FLOW_FUNC_VISIBILITY AttrValue {
 public:
  AttrValue() = default;

  ~AttrValue() = default;

  AttrValue(const AttrValue &) = delete;

  AttrValue(AttrValue &&) = delete;

  AttrValue &operator=(const AttrValue &) = delete;

  AttrValue &operator=(AttrValue &&) = delete;

  /*
   * get string value of attr.
   * @param value: string value of attr
   * @return 0:SUCCESS, other:failed
   */
  int32_t GetVal(AscendString &value) const {
    std::string name = "stub";
    value = name.c_str();
    return 0;
  }

  /*
   * get string list value of attr.
   * @param value: string list value of attr
   * @return 0:SUCCESS, other:failed
   */
  int32_t GetVal(std::vector<AscendString> &value) const {
    std::string name = "stub";
    value.emplace_back(name.c_str());
    return 0;
  }

  /*
   * get int value of attr.
   * @param value: int value of attr
   * @return 0:SUCCESS, other:failed
   */
  int32_t GetVal(int64_t &value) const {
    value = 0L;
    return 0;
  }

  /*
   * get int list value of attr.
   * @param value: int list value of attr
   * @return 0:SUCCESS, other:failed
   */
  int32_t GetVal(std::vector<int64_t> &value) const {
    value.emplace_back(0L);
    return 0;
  }

  /*
   * get int list list value of attr.
   * @param value: int list list value of attr
   * @return 0:SUCCESS, other:failed
   */
  int32_t GetVal(std::vector<std::vector<int64_t>> &value) const {
    return 0;
  }

  /*
   * get float value of attr.
   * @param value: float value of attr
   * @return 0:SUCCESS, other:failed
   */
  int32_t GetVal(float &value) const {
    value = 0.1;
    return 0;
  }

  /*
   * get float list value of attr.
   * @param value: float list value of attr
   * @return 0:SUCCESS, other:failed
   */
  int32_t GetVal(std::vector<float> &value) const {
    value.emplace_back(0.1);
    return 0;
  }

  /*
   * get bool value of attr.
   * @param value: bool value of attr
   * @return 0:SUCCESS, other:failed
   */
  int32_t GetVal(bool &value) const {
    value = true;
    return 0;
  }

  /*
   * get bool list value of attr.
   * @param value: bool list value of attr
   * @return 0:SUCCESS, other:failed
   */
  int32_t GetVal(std::vector<bool> &value) const {
    value.emplace_back(true);
    return 0;
  }

  /*
   * get data type value of attr.
   * @param value: data type value of attr
   * @return 0:SUCCESS, other:failed
   */
  int32_t GetVal(TensorDataType &value) const {
    value = TensorDataType::DT_INT16;
    return 0;
  }

  /*
   * get data type list value of attr.
   * @param value: data type list value of attr
   * @return 0:SUCCESS, other:failed
   */
  int32_t GetVal(std::vector<TensorDataType> &value) const {
    value.emplace_back(TensorDataType::DT_INT16);
    return 0;
  }
};
}  // namespace FlowFunc

#endif  // FLOW_FUNC_ATTR_VALUE_H
