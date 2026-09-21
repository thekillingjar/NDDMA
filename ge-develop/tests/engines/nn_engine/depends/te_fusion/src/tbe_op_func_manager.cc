/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tbe_op_func_manager.h"
#include <nlohmann/json.hpp>
#include "graph/utils/type_utils.h"

TbeOpFuncManager& TbeOpFuncManager::Instance() {
  static TbeOpFuncManager tbe_op_func_manager;
  return tbe_op_func_manager;
}

void TbeOpFuncManager::AddSelectFormatFunction(const std::string &op_type, const SelectFormatFunc &func) {
  select_format_func_map_.emplace(op_type, func);
  std::cout << "Register SelectFormatFunc of " << op_type << endl;
}
void TbeOpFuncManager::AddCheckSupportFunction(const std::string &op_type, const CheckSupportFunc &func) {
  check_support_func_map_.emplace(op_type, func);
  std::cout << "Register CheckSupportFunc of " << op_type << endl;
}

bool TbeOpFuncManager::GetCheckSupportFunction(const std::string &op_type, CheckSupportFunc &func) {
  auto iter = check_support_func_map_.find(op_type);
  if (iter == check_support_func_map_.end()) {
    return false;
  }
  func = iter->second;
  return true;
}

bool TbeOpFuncManager::GetSelectFormatFunction(const std::string &op_type, SelectFormatFunc &func) {
  auto iter = select_format_func_map_.find(op_type);
  if (iter == select_format_func_map_.end()) {
    return false;
  }
  func = iter->second;
  return true;
}

struct TensorDescInfo {
  string name;
  vector<ge::DataType> dtypes;
  vector<ge::Format> formats;
  vector<ge::Format> unknown_formats;
};

class FormatBuider {
 public:
  FormatBuider& AddInput(const string &name, const vector<ge::DataType> &dtype_vec,
                         const vector<ge::Format> &format_vec) {
    return AddInput(name, dtype_vec, format_vec, {});
  }
  FormatBuider& AddInput(const string &name, const vector<ge::DataType> &dtype_vec,
                         const vector<ge::Format> &format_vec, const vector<ge::Format> &unknown_format_vec) {
    return AddTensorInfo(true, name, dtype_vec, format_vec, unknown_format_vec);
  }
  FormatBuider& AddOutput(const string &name, const vector<ge::DataType> &dtype_vec,
                          const vector<ge::Format> &format_vec) {
    return AddOutput(name, dtype_vec, format_vec, {});
  }
  FormatBuider& AddOutput(const string &name, const vector<ge::DataType> &dtype_vec,
                          const vector<ge::Format> &format_vec, const vector<ge::Format> &unknown_format_vec) {
    return AddTensorInfo(false, name, dtype_vec, format_vec, unknown_format_vec);
  }
  string Build() {
    nlohmann::json format_json;
    for (size_t i = 0; i < inputs_.size(); i++) {
      nlohmann::json input_json;
      ConvertToJson(inputs_[i], input_json);
      format_json["input"+to_string(i)] = input_json;
    }

    for (size_t i = 0; i < outputs_.size(); i++) {
      nlohmann::json output_json;
      ConvertToJson(outputs_[i], output_json);
      format_json["output"+to_string(i)] = output_json;
    }
    return format_json.dump();
  }
 private:
  FormatBuider& AddTensorInfo(const bool is_input, const string &name, const vector<ge::DataType> &dtype_vec,
                              const vector<ge::Format> &format_vec, const vector<ge::Format> &unknown_format_vec) {
    TensorDescInfo tensor_info = {name, dtype_vec, format_vec, unknown_format_vec};
    if (is_input) {
      inputs_.push_back(tensor_info);
    } else {
      outputs_.push_back(tensor_info);
    }
    return *this;
  }
  void ConvertToJson(const TensorDescInfo &tensor_info, nlohmann::json &tensor_json) {
    tensor_json["name"] = tensor_info.name;
    if (!tensor_info.dtypes.empty()) {
      string dtype_str;
      for (size_t i = 0; i < tensor_info.dtypes.size(); i++) {
        if (i != 0) {
          dtype_str += ",";
        }
        dtype_str += ge::TypeUtils::DataTypeToSerialString(tensor_info.dtypes[i]).substr(3);
      }
      std::transform(dtype_str.begin(), dtype_str.end(), dtype_str.begin(), ::tolower);
      tensor_json["dtype"] = dtype_str;
    }
    if (!tensor_info.formats.empty()) {
      string format_str;
      for (size_t i = 0; i < tensor_info.formats.size(); i++) {
        if (i != 0) {
          format_str += ",";
        }
        format_str += ge::TypeUtils::FormatToSerialString(tensor_info.formats[i]);
      }
      tensor_json["format"] = format_str;
    }
    if (!tensor_info.unknown_formats.empty()) {
      string unknown_format_str;
      for (size_t i = 0; i < tensor_info.unknown_formats.size(); i++) {
        if (i != 0) {
          unknown_format_str += ",";
        }
        unknown_format_str += ge::TypeUtils::FormatToSerialString(tensor_info.unknown_formats[i]);
      }
      tensor_json["unknownshape_format"] = unknown_format_str;
    }
  }
  vector<TensorDescInfo> inputs_;
  vector<TensorDescInfo> outputs_;
};

IMPL_CHECK_SUPPORT_FUNC(Conv2D) {
  is_support = te::FULLY_SUPPORTED;
  return true;
}
REG_CHECK_SUPPORT_FUNC(Conv2D)

IMPL_SELECT_FORMAT_FUNC(Conv2D) {
  FormatBuider builder;
  builder.AddInput("x", {ge::DT_FLOAT16, ge::DT_INT16, ge::DT_INT8}, {ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0})
         .AddInput("filter", {ge::DT_FLOAT16, ge::DT_INT8, ge::DT_INT8}, {ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_Z})
         .AddInput("bias", {ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32}, {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
         .AddInput("offset_w", {ge::DT_INT8, ge::DT_INT8, ge::DT_INT8}, {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
         .AddOutput("y", {ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32}, {ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0});
  format_str = builder.Build();
  return true;
}
REG_SELECT_FORMAT_FUNC(Conv2D)

IMPL_SELECT_FORMAT_FUNC(Add) {
  FormatBuider builder;
  builder.AddInput("x1", {ge::DT_FLOAT16, ge::DT_INT32}, {ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0})
         .AddInput("x2", {ge::DT_FLOAT16, ge::DT_INT32}, {ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0})
         .AddOutput("y", {ge::DT_FLOAT16, ge::DT_INT32}, {ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0});
  format_str = builder.Build();
  return true;
}
REG_SELECT_FORMAT_FUNC(Add)

IMPL_SELECT_FORMAT_FUNC(SoftmaxV2) {
  FormatBuider builder;
  builder.AddInput("x", {ge::DT_FLOAT16, ge::DT_INT32}, {ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0})
         .AddOutput("y", {ge::DT_FLOAT16, ge::DT_INT32}, {ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0});
  format_str = builder.Build();
  return true;
}
REG_SELECT_FORMAT_FUNC(SoftmaxV2)

IMPL_SELECT_FORMAT_FUNC(StridedSliceD) {
  FormatBuider builder;
  builder.AddInput("x", {ge::DT_FLOAT16, ge::DT_INT32}, {ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0})
         .AddOutput("y", {ge::DT_FLOAT16, ge::DT_INT32}, {ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0});
  format_str = builder.Build();
  return true;
}
REG_SELECT_FORMAT_FUNC(StridedSliceD)

IMPL_SELECT_FORMAT_FUNC(ConcatV2D) {
  FormatBuider builder;
  builder.AddInput("input_values", {ge::DT_FLOAT16, ge::DT_INT32}, {ge::FORMAT_ND, ge::FORMAT_ND})
          .AddOutput("output_data", {ge::DT_FLOAT16, ge::DT_INT32}, {ge::FORMAT_ND, ge::FORMAT_ND});
  format_str = builder.Build();
  return true;
}
REG_SELECT_FORMAT_FUNC(ConcatV2D)

IMPL_SELECT_FORMAT_FUNC(SplitD) {
  FormatBuider builder;
  builder.AddInput("x", {ge::DT_FLOAT16, ge::DT_FLOAT}, {ge::FORMAT_NCHW, ge::FORMAT_NCHW}, {ge::FORMAT_NCHW, ge::FORMAT_NCHW})
         .AddOutput("y", {ge::DT_FLOAT16, ge::DT_FLOAT}, {ge::FORMAT_NCHW, ge::FORMAT_NCHW}, {ge::FORMAT_NCHW, ge::FORMAT_NCHW});
  format_str = builder.Build();
  return true;
}
REG_SELECT_FORMAT_FUNC(SplitD)

IMPL_SELECT_FORMAT_FUNC(AvgPoolUpdate) {
  FormatBuider builder;
  builder.AddInput("x1", {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT16},
                   {ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0, ge::FORMAT_ND, ge::FORMAT_NC1HWC0},
                   {ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0, ge::FORMAT_ND, ge::FORMAT_NC1HWC0})
      .AddInput("x2", {ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT, ge::DT_INT4},
                {ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0, ge::FORMAT_ND, ge::FORMAT_NC1HWC0},
                {ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0, ge::FORMAT_ND, ge::FORMAT_NC1HWC0})
      .AddOutput("y", {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT16},
                 {ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0, ge::FORMAT_ND, ge::FORMAT_NC1HWC0},
                 {ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0, ge::FORMAT_ND, ge::FORMAT_NC1HWC0});
  format_str = builder.Build();
  return true;
}
REG_SELECT_FORMAT_FUNC(AvgPoolUpdate)

IMPL_SELECT_FORMAT_FUNC(Pooling) {
  FormatBuider builder;
  builder.AddInput("x", {ge::DT_FLOAT16, ge::DT_FLOAT}, {ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0}, {ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0})
         .AddInput("matrix", {ge::DT_FLOAT16, ge::DT_FLOAT}, {ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0}, {ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0})
         .AddInput("bias", {ge::DT_FLOAT16, ge::DT_FLOAT}, {ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0}, {ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0})
         .AddOutput("y", {ge::DT_FLOAT16, ge::DT_FLOAT}, {ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0}, {ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0});
  format_str = builder.Build();
  return true;
}
REG_SELECT_FORMAT_FUNC(Pooling)
