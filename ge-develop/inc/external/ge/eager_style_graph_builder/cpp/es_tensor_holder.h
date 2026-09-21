/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_ES_TENSOR_HOLDER_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_ES_TENSOR_HOLDER_H_
#include "graph/types.h"
#include "graph/gnode.h"
#include "ge/ge_api_types.h"
#include "esb_funcs.h"

namespace ge {
namespace es {

/**
 * @brief 张量持有者类，封装了C语言的张量操作接口
 *
 * 该类提供了对张量的各种操作，包括设置数据类型、格式、形状等属性，
 * 以及设置张量和节点的属性。支持算术运算操作。
 */
class EsTensorHolder {
 public:
  /**
   * @brief 构造函数，使用C语言张量持有者指针初始化
   * @param tensor C语言张量持有者指针
   */
  EsTensorHolder(EsCTensorHolder *tensor) : tensor_holder_(tensor) {}

  /**
   * @brief 默认构造函数
   */
  EsTensorHolder() = default;

  /**
   * @brief 张量加法运算
   * @param other 另一个张量持有者
   * @return 返回运算结果张量持有者
   */
  EsTensorHolder operator+(const EsTensorHolder &other) const;

  /**
   * @brief 张量减法运算
   * @param other 另一个张量持有者
   * @return 返回运算结果张量持有者
   */
  EsTensorHolder operator-(const EsTensorHolder &other) const;

  /**
   * @brief 张量乘法运算
   * @param other 另一个张量持有者
   * @return 返回运算结果张量持有者
   */
  EsTensorHolder operator*(const EsTensorHolder &other) const;

  /**
   * @brief 张量除法运算
   * @param other 另一个张量持有者
   * @return 返回运算结果张量持有者
   */
  EsTensorHolder operator/(const EsTensorHolder &other) const;

  /**
   * @brief 设置张量的数据类型
   * @param data_type 数据类型
   * @return 返回当前对象的引用，支持链式调用
   */
  EsTensorHolder &SetDataType(const ge::DataType data_type) {
    EsSetDataType(tensor_holder_, static_cast<C_DataType>(data_type));
    return *this;
  }

  /**
   * @brief 设置张量的格式
   * @param format 张量格式
   * @return 返回当前对象的引用，支持链式调用
   */
  EsTensorHolder &SetFormat(const ge::Format format) {
    EsSetFormat(tensor_holder_, static_cast<C_Format>(format));
    return *this;
  }

  /**
   * @brief 设置张量的形状
   * @param dims 张量维度向量
   * @return 返回当前对象的引用，支持链式调用
   */
  EsTensorHolder &SetShape(const std::vector<int64_t> &dims) {
    EsSetShape(tensor_holder_, dims.data(), static_cast<int64_t>(dims.size()));
    return *this;
  }

  /**
   * @brief 设置张量的原始符号形状
   * @param dims 符号维度名称向量
   * @return 返回当前对象的引用，支持链式调用
   */
  EsTensorHolder &SetOriginSymbolShape(const std::vector<const char *> &dims) {
    EsSetOriginSymbolShape(tensor_holder_, dims.data(), dims.size());
    return *this;
  }

  /**
   * @brief 设置张量的int64类型属性
   * @param attr_name 属性名称
   * @param value 属性值
   * @return 返回操作状态，成功为SUCCESS，其他失败
   */
  Status SetAttr(const char *attr_name, int64_t value) {
    return static_cast<Status>(EsSetInt64AttrForTensor(tensor_holder_, attr_name, value));
  }

  /**
   * @brief 设置张量的字符串类型属性
   * @param attr_name 属性名称
   * @param value 属性值
   * @return 返回操作状态，成功为SUCCESS，其他失败
   */
  Status SetAttr(const char *attr_name, const char *value) {
    return static_cast<Status>(EsSetStringAttrForTensor(tensor_holder_, attr_name, value));
  }

  /**
   * @brief 设置张量的布尔类型属性
   * @param attr_name 属性名称
   * @param value 属性值
   * @return 返回操作状态，成功为SUCCESS，其他失败
   */
  Status SetAttr(const char *attr_name, bool value) {
    return static_cast<Status>(EsSetBoolAttrForTensor(tensor_holder_, attr_name, value));
  }

  /**
   * @brief 设置张量对应节点的int64类型属性
   * @param attr_name 属性名称
   * @param value 属性值
   * @return 返回操作状态，成功为SUCCESS，其他失败
   */
  Status SetAttrForNode(const char *attr_name, int64_t value) {
    return static_cast<Status>(EsSetInt64AttrForNode(tensor_holder_, attr_name, value));
  }

  /**
   * @brief 设置张量对应节点的字符串类型属性
   * @param attr_name 属性名称
   * @param value 属性值
   * @return 返回操作状态，成功为SUCCESS，其他失败
   */
  Status SetAttrForNode(const char *attr_name, const char *value) {
    return static_cast<Status>(EsSetStringAttrForNode(tensor_holder_, attr_name, value));
  }

  /**
   * @brief 设置张量对应节点的布尔类型属性
   * @param attr_name 属性名称
   * @param value 属性值
   * @return 返回操作状态，成功为SUCCESS，其他失败
   */
  Status SetAttrForNode(const char *attr_name, bool value) {
    return static_cast<Status>(EsSetBoolAttrForNode(tensor_holder_, attr_name, value));
  }

  /**
   * @brief 获取底层的C语言张量持有者指针, 返回值的生命周期受EsGraphBuilder管理，使用方不能释放
   * @return 返回C语言张量持有者指针
   */
  EsCTensorHolder *GetCTensorHolder() const {
    return tensor_holder_;
  }

  /**
   * @brief 获取输出索引
   * @return 输出索引
   */
  int32_t GetProducerOutIndex() const;

  /**
   * @brief 获取产生该张量的图节点
   * @return 返回图节点指针
   */
  GNode *GetProducer() const;

  /**
   * @brief 控制边连接函数
   * @param ctrl_ins 控制边输入，支持多个
   * @return 返回操作状态，成功为SUCCESS，其他失败
   */
  Status AddControlEdge(const std::vector<EsTensorHolder> &ctrl_ins) const;
 private:
  friend class EsGraphBuilder;
  EsCTensorHolder *tensor_holder_{nullptr};
};
}  // namespace es
}  // namespace ge
#endif  // AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_ES_TENSOR_HOLDER_H_
