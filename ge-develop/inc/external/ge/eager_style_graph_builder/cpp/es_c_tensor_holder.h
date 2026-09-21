/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_EAGER_STYLE_GRAPH_BUILDER_ESC_TENSOR_HOLDER_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_EAGER_STYLE_GRAPH_BUILDER_ESC_TENSOR_HOLDER_H_
#include "graph/gnode.h"
#include "ge/ge_api_types.h"
#include "ge/ge_api_error_codes.h"
struct EsCGraphBuilder;

struct EsCTensorHolder {
 public:
  /**
   * @brief 构造函数
   * 需要保证入参有效, 指向合法内存，否则会抛出异常。
   * 内部构建失败时不捕获异常，直接抛出给用户侧。
   * @param owner 所属的图构建器
   * @param producer 生产者节点
   * @param index 输出索引
   *
   * 注意：调用者需要保证传入的`producer`不为空，`index`合法
   */
  EsCTensorHolder(EsCGraphBuilder &owner, const ge::GNode &producer, int32_t index);

  ~EsCTensorHolder();
  EsCTensorHolder(EsCTensorHolder &&) noexcept;
  EsCTensorHolder &operator=(EsCTensorHolder &&) noexcept;

  EsCTensorHolder(const EsCTensorHolder &) = delete;
  EsCTensorHolder &operator=(const EsCTensorHolder &) = delete;
  /**
   * @brief 设置数据类型
   * @param data_type 数据类型
   * @return 操作状态
   */
  ge::Status SetDataType(const ge::DataType data_type);

  /**
   * @brief 设置数据格式
   * @param format 数据格式
   * @return 操作状态
   */
  ge::Status SetFormat(const ge::Format format);

  /**
   * @brief 设置原始数据格式
   * @param format 原始数据格式
   * @return 操作状态
   */
  ge::Status SetOriginFormat(const ge::Format format);

  /**
   * @brief 设置存储数据格式
   * @param format 存储数据格式
   * @return 操作状态
   */
  ge::Status SetStorageFormat(const ge::Format format);

  /**
   * @brief 设置原始形状
   * @param shape 原始形状
   * @return 操作状态
   */
  ge::Status SetOriginShape(const ge::Shape &shape);

  /**
   * @brief 设置存储形状
   * @param shape 存储形状
   * @return 操作状态
   */
  ge::Status SetStorageShape(const ge::Shape &shape);

  /**
   * @brief 设置形状（同时设置原始形状和存储形状）
   * @param shape 形状
   * @return 操作状态
   */
  ge::Status SetShape(const ge::Shape &shape);

  /**
   * @brief 设置原始符号形状
   * @param shape_str 形状字符串数组
   * @param shape_str_num 形状字符串数量
   * @return 操作状态
   */
  ge::Status SetOriginSymbolShape(const char *const *shape_str, const int64_t shape_str_num);

  /**
   * @brief 获取生产者节点
   * @return 生产者节点引用
   */
  ge::GNode &GetProducer();

  /**
   * @brief 获取输出索引
   * @return 输出索引
   */
  int32_t GetOutIndex() const;

  /**
   * @brief 获取所属的图构建器
   * @return 图构建器引用
   */
  EsCGraphBuilder &GetOwnerBuilder();

 private:
  struct EsCTensorHolderImpl;
  std::unique_ptr<EsCTensorHolderImpl> impl_;
};

#endif  // AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_EAGER_STYLE_GRAPH_BUILDER_ESC_TENSOR_HOLDER_H_
