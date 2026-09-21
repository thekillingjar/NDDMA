/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SOURCE_STUB_H
#define SOURCE_STUB_H

#include "graph/symbolizer/symbolic.h"
#include "common/checker.h"

#include <graph/utils/type_utils.h>

namespace ge {
static std::map<ge::DataType, std::string> kGeDType2CppDtype = {
  {ge::DT_INT32, "int32_t"},
  {ge::DT_INT64, "int64_t"},
  {ge::DT_UINT32, "uint32_t"},
  {ge::DT_UINT64, "uint64_t"},
};
class GraphInputShapeSourceStub : public Source {
public:
  GraphInputShapeSourceStub(int32_t input_data_idx, size_t dim_idx)
      : input_data_idx_(input_data_idx), dim_idx_(dim_idx) {}

  std::string GetSourceStr() const override {
    return R"([&]() -> int64_t {
      const auto *tensor = context->GetGraphInputTensor()" + std::to_string(input_data_idx_) + R"();
      if (tensor == nullptr) {
        return -1;
      }
      return tensor->GetOriginShape().GetDim()" + std::to_string(dim_idx_) + R"();
    }())";
  }
  ~GraphInputShapeSourceStub() override = default;
private:
  int32_t input_data_idx_;  // Data的index，描述symbol来自于graph输入中第几个输入data
  size_t dim_idx_;          // 描述symbol来自于data中对应shape的第几个dim
};

class InputValueSumSourceStub : public ge::Source {
public:
  InputValueSumSourceStub(int32_t input_data_idx, ge::DataType dtype)
      : input_data_idx_(input_data_idx), dtype_(dtype) {}

  [[nodiscard]] std::string GetSourceStr() const override {
    if (kGeDType2CppDtype.find(dtype_) == kGeDType2CppDtype.end()) {
      GELOGE(FAILED, "Unsupported data type: %s", TypeUtils::DataTypeToSerialString(dtype_).c_str());
      return "";
    }
    return R"([&]() -> int64_t {
              const auto* tensor = context->GetGraphInputTensor()" + std::to_string(input_data_idx_) + R"();
                if (tensor == nullptr) {
                  return -1;
                }
                const auto* data = tensor->GetData<)" + kGeDType2CppDtype[dtype_] + R"(>();
                int64_t sum = 0;
                for (size_t i = 0; i < tensor->GetSize() / sizeof()" + kGeDType2CppDtype[dtype_] + R"(); ++i) {
                  sum += data[i];
                }
                return sum;
            }())";
  }
  ~InputValueSumSourceStub() override = default;

private:
  int32_t input_data_idx_;  // Data的index，描述symbol来自于graph输入中第几个输入data
  ge::DataType dtype_;      // 描述value的数据类型，用于后续执行时取值
};
}

#endif  // SOURCE_STUB_H