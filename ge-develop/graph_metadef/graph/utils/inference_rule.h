/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef REGISTER_INFERENCE_RULE_H
#define REGISTER_INFERENCE_RULE_H

#include <string>
#include <sstream>

#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/infer_datatype_context.h"
#include "graph/op_desc.h"

namespace ge {
/**
 * @brief 推导规则基类
 *
 * 为了引导原始错误记录在对象上，不分散的打印日志，有助于向用户展示明确报错。
 */
class InferenceRule {
 public:
  template<typename T>
  InferenceRule &operator<<(const T &msg) {
    err_ << msg;
    return *this;
  }

  std::string Error() const {
    return err_.str();
  }

  bool IsValid() const {
    return err_.str().empty();
  }
  static std::string GetInferenceRule(const ge::OpDescPtr &op);

 protected:
  std::stringstream err_;
};

/**
 * @brief Shape推导实现类
 *
 * 负责从不同类型的输入编译并加载得到Shape推导可执行函数，并与GE数据结构配合工作。
 */
class ShapeInferenceRule : public InferenceRule {
 public:
  // Ctx接口定义，供推导函数调用，不依赖任何头文件。实现与用户环境完全隔离。
  class Ctx {
   public:
    virtual ~Ctx() = default;

    virtual bool GetInputDim(int64_t input, int64_t dim_index, int64_t &dim) = 0;

    virtual bool GetInputValue(int64_t input, int64_t offset, int64_t &value) = 0;

    virtual bool SetOutputDimNum(int64_t output, int64_t dim_num) = 0;

    virtual bool SetOutputDim(int64_t output, int64_t dim_index, int64_t dim) = 0;

    virtual void SetError(const char *) = 0;
  };

  using InferShapeFunc = bool (*)(Ctx *);

  ShapeInferenceRule() : handle_(nullptr), infer_shape_(nullptr), infer_shape_on_compile_(nullptr) {}
  ~ShapeInferenceRule();
  ShapeInferenceRule(const ShapeInferenceRule &) = delete;
  ShapeInferenceRule &operator=(const ShapeInferenceRule &) = delete;
  ShapeInferenceRule &operator=(ShapeInferenceRule &&other) = delete;
  ShapeInferenceRule(ShapeInferenceRule &&other) noexcept {
    handle_ = other.handle_;
    infer_shape_ = other.infer_shape_;
    infer_shape_on_compile_ = other.infer_shape_on_compile_;
    err_ << other.err_.str();
    other.handle_ = nullptr;
    other.infer_shape_ = nullptr;
    other.infer_shape_on_compile_ = nullptr;
  }

  static std::shared_ptr<ShapeInferenceRule> FromOpDesc(const ge::OpDescPtr &op);
  static std::shared_ptr<ShapeInferenceRule> FromJsonString(const std::string &json_str);

  // 编译后的二进制以属性的方式保存在节点上，用于RT2执行时加载
  static ge::graphStatus CompileJsonString(const std::string &json_str, std::vector<uint8_t> &binary);
  static ShapeInferenceRule FromCompiledBinary(const std::vector<uint8_t> &binary);
  static ShapeInferenceRule FromCompiledBinary(const uint8_t *binary, const size_t size);

  ge::graphStatus InferOnRuntime(gert::InferShapeContext *infer_shape_ctx) const;
  ge::graphStatus InferOnCompile(gert::InferShapeContext *infer_shape_ctx) const;

  ge::graphStatus InferOnRuntime(Ctx *ctx) const;
  ge::graphStatus InferOnCompile(Ctx *ctx) const;

 private:
  void *handle_;
  InferShapeFunc infer_shape_;
  InferShapeFunc infer_shape_on_compile_;
};

/**
 * @brief Dtype推导实现类
 *
 * 负责从不同类型的解析得到Shape推导可执行函数，并与GE图结构配合工作。Dtype推导实现无需编译。
 */
class DtypeInferenceRule : public InferenceRule {
 public:
  static std::shared_ptr<DtypeInferenceRule> FromOpDesc(const ge::OpDescPtr &op);
  static std::shared_ptr<DtypeInferenceRule> FromJsonString(const std::string &json_str);

  ge::graphStatus InferDtype(gert::InferDataTypeContext *infer_dtype_ctx) const;

 private:
  std::vector<ge::DataType> dtypes_;
};
}  // namespace ge
#endif
