/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_EAGER_STYLE_GRAPH_BUILDER_GENERATOR_INTERFACE_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_EAGER_STYLE_GRAPH_BUILDER_GENERATOR_INTERFACE_H_

#include <string>
#include <vector>
#include <sstream>
#include <unordered_map>
#include "graph/op_desc.h"
#include "utils.h"

namespace ge {
namespace es {

/**
 * 代码生成器接口
 * 定义所有生成器必须实现的基本操作
 */
class ICodeGenerator {
 public:
  virtual ~ICodeGenerator() = default;

  // 生成单个Op的代码
  virtual void GenOp(const OpDescPtr &op) = 0;

  // 当生成失败时重置状态(删除当前Op的生成信息)
  virtual void ResetWhenGenFailed(const OpDescPtr &op) = 0;

  // 生成聚合文件头部
  virtual void GenAggregateHeader() = 0;

  // 生成聚合文件包含语句
  virtual void GenAggregateIncludes(const std::vector<std::string> &sorted_op_types) = 0;

  // 获取聚合文件名
  virtual std::string GetAggregateFileName() const = 0;

  // 获取聚合的内容流
  virtual std::stringstream &GetAggregateContentStream() = 0;

  // 生成单个op的文件
  virtual void GenPerOpFiles(const std::string &output_dir, const std::vector<std::string> &sorted_op_types) = 0;

  // 获取单个Op文件名
  virtual std::string GetPerOpFileName(const std::string &op_type) const = 0;

  // 获取单个Op的内容映射
  virtual const std::unordered_map<std::string, std::stringstream> &GetPerOpContents() const = 0;

  // 获取生成器名称（用于日志和错误信息）
  virtual std::string GetGeneratorName() const = 0;

  // 获取批注释开始标记
  virtual std::string GetCommentStart() const = 0;

  // 获取批注释结束标记
  virtual std::string GetCommentEnd() const = 0;

  // 获取注释单行前缀
  virtual std::string GetCommentLinePrefix() const = 0;
};

}  // namespace es
}  // namespace ge

#endif  // AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_EAGER_STYLE_GRAPH_BUILDER_GENERATOR_INTERFACE_H_