/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_ESC_GRAPH_BUILDER_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_ESC_GRAPH_BUILDER_H_
#include <list>
#include <string>
#include "graph/graph.h"
#include "graph/c_types.h"
#include "es_c_tensor_holder.h"

struct EsCGraphBuilder {
 public:
  /**
   * @brief 默认构造函数，创建一个名为"graph"的图构建器
   */
  EsCGraphBuilder();

  /**
   * @brief 构造函数
   * 需要保证入参有效, 指向合法内存，否则会抛出异常。
   * 内部构建失败时不捕获异常，直接抛出给用户侧。
   * @param name 图名称
   */
  explicit EsCGraphBuilder(const char *name);

  ~EsCGraphBuilder();
  EsCGraphBuilder(EsCGraphBuilder &&) noexcept;
  EsCGraphBuilder &operator=(EsCGraphBuilder &&) noexcept;

  EsCGraphBuilder(const EsCGraphBuilder &) = delete;
  EsCGraphBuilder &operator=(const EsCGraphBuilder &) = delete;

  /**
   * @brief 从图节点获取张量持有者
   * @param node 图节点
   * @param output_index 输出索引
   * @return 张量持有者指针
   */
  EsCTensorHolder *GetTensorHolderFromNode(const ge::GNode &node, int32_t output_index);

  /**
   * @brief 在图的末尾添加图输入
   * @param name 输入名称，可选
   * @param type 输入类型，可选
   * @return 张量持有者指针
   */
  EsCTensorHolder *AppendGraphInput(const ge::char_t *name = nullptr, const ge::char_t *type = nullptr);

  /**
   * @brief 在指定位置添加图输入
   * @param index 输入索引位置
   * @param name 输入名称，可选
   * @param type 输入类型，可选
   * @param data_type 数据类型，默认为C_DT_FLOAT
   * @param format 数据格式，默认为C_FORMAT_ND
   * @param dims 维度数组，可选
   * @param dim_num 维度数量，默认为0
   * @return 张量持有者指针
   */
  EsCTensorHolder *AddGraphInput(int32_t index, const ge::char_t *name = nullptr, const ge::char_t *type = nullptr,
                                 C_DataType data_type = C_DT_FLOAT, C_Format format = C_FORMAT_ND,
                                 const int64_t *dims = nullptr, int64_t dim_num = 0);

  /**
   * @brief 设置图输出
   * @param tensor 张量持有者
   * @param output_index 输出索引
   * @return 操作状态
   */
  ge::Status SetGraphOutput(EsCTensorHolder *tensor, int64_t output_index);

  /**
   * @brief 获取内部的图对象
   * @return 图对象指针
   */
  ge::Graph *GetGraph();

  /**
   * @brief 构建并返回图对象
   * @return 构建完成的图对象
   */
  std::unique_ptr<ge::Graph> BuildGraphAndReset();

  /**
   * @brief 生成节点名称
   * @param node_type 节点类型
   * @return 生成的节点名称字符串
   */
  ge::AscendString GenerateNodeName(const ge::char_t *node_type);

  /**
   * @brief 从图节点获取动态输出的张量
   * @param node 图节点
   * @param start_idx 输出索引起始点
   * @param output_num 总动态输出数量
   * @return 动态输出容器持有者指针
   */
  std::vector<EsCTensorHolder *> *CreateDynamicTensorHolderFromNode(const ge::GNode &node, int32_t start_idx,
                                                                    int32_t output_num);

  /**
   * @brief 将传入的资源转移给EsCGraphBuilder进行管理
   * @tparam T 将被转移的资源类型
   * @param value 将被转移的资源
   * @param deleter 资源析构函数
   * @return 资源持有者指针
   */
  template <typename T>
  T *AddResource(std::unique_ptr<T> value, std::function<void(void *)> deleter = DefaultDeleter<T>) {
    return static_cast<T *>(AddResource(value.release(), deleter));
  }

 private:
  /**
   * @brief 创建资源并存入EsCGraphBuilder进行生命周期管理
   * @param resource_ptr 资源指针
   * @param deleter 资源析构函数
   */
  void *AddResource(void *resource_ptr, std::function<void(void *)> deleter);

  template <typename T>
  static void DefaultDeleter(void *ptr) {
    delete static_cast<T *>(ptr);
  }

  /**
   * @brief 从图节点获取张量持有者的内部实现
   * @param node 图节点
   * @param output_index 输出索引
   * @return 张量持有者指针
   */
  EsCTensorHolder *GetTensorHolderFromNodeInner(const ge::GNode &node, int32_t output_index);
  struct EsCGraphBuilderImpl;
  std::unique_ptr<EsCGraphBuilderImpl> impl_;
};

#endif  // AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_ESC_GRAPH_BUILDER_H_
