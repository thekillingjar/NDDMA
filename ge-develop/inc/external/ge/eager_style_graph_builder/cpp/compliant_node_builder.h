/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_COMPLIANT_NODE_BUILDER_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_COMPLIANT_NODE_BUILDER_H_
#include <vector>
#include <string>
#include <utility>
#include <type_traits>
#include <unordered_set>
#include <unordered_map>
#include <cmath>
#include <cstring>
#include <algorithm>
#include "graph/attr_value.h"
#include "graph/gnode.h"
#include "graph/operator.h"
#include "ge/ge_api_types.h"

namespace ge {
namespace es {

/**
 * @brief 从任意类型创建属性值的辅助函数
 * @tparam T 输入类型
 * @param t 输入值
 * @return 创建的属性值对象
 */
template <typename T>
AttrValue CreateFrom(T &&t) {
  AttrValue av;
  av.SetAttrValue(std::forward<T>(t));
  return av;
}

/**
 * @brief 比较两个值是否相等（考虑浮点数精度）
 * @tparam T 值类型
 * @param a 第一个值
 * @param b 第二个值
 * @return 如果相等返回 true，否则返回 false
 */
template <typename T>
bool ValuesEqual(const T &a, const T &b) {
  return a == b;
}

/**
 * @brief float 类型的特殊比较（使用 epsilon）
 */
template <>
inline bool ValuesEqual<float>(const float &a, const float &b) {
  return std::fabs(a - b) < 1e-6f;
}

/**
 * @brief 比较两个 vector 是否相等（考虑浮点数精度）
 * @tparam T 元素类型
 * @param a 第一个 vector
 * @param b 第二个 vector
 * @return 如果相等返回 true，否则返回 false
 */
template <typename T>
bool ValuesEqual(const std::vector<T> &a, const std::vector<T> &b) {
  if (a.size() != b.size()) {
    return false;
  }

  return std::equal(a.begin(), a.end(), b.begin(), [](const T &x, const T &y) {
    return ValuesEqual(x, y);
  });
}

/**
 * @brief 如果值不等于默认值，则创建属性值对象
 * @tparam T 值类型
 * @param value 实际值
 * @param default_value 默认值
 * @return 如果值不等于默认值，返回包含该值的 AttrValue；否则返回空的 AttrValue
 */
template <typename T>
AttrValue CreateFromIfNotEqual(T &&value, typename std::decay<T>::type default_value) {
  AttrValue av;
  if (!ValuesEqual(value, default_value)) {
    av.SetAttrValue(std::forward<T>(value));
  }
  return av;
}

/**
 * @brief 添加边并使用源节点的输出tensor更新peer tensor的描述信息
 * 当前仅处理可选输入. 连边同时会把输入的TensorDesc变成有效信息
 * @param graph src_node和dst_node所属的图对象
 * @param src_node 源节点
 * @param src_port_index 源端口索引
 * @param dst_node 目标节点
 * @param dst_port_index 目标端口索引
 * @return 成功返回GRAPH_SUCCESS，失败返回其他
 */
ge::graphStatus AddEdgeAndUpdatePeerDesc(Graph &graph, GNode &src_node, int32_t src_port_index, GNode &dst_node,
                                         int32_t dst_port_index);

/**
 * @brief 合规节点构建器类，用于构建符合IR规范的图节点
 *
 * 该类提供了流式API来定义节点的IR输入、输出、属性和子图，
 * 确保生成的节点符合图引擎的IR规范要求。
 */
class CompliantNodeBuilder {
 public:
  /**
   * @brief 任意类型操作符类，继承自Operator
   *
   * 提供了对Operator类中IR相关方法的访问，包括动态输入/输出注册、
   * 输入/输出注册、属性注册和子图注册等功能。
   */
  class AnyTypeOperator : public Operator {
   public:
    AnyTypeOperator(const char_t *name, const char_t *type) : Operator(name, type) {}
    using Operator::DynamicInputRegister;
    using Operator::InputRegister;
    using Operator::OptionalInputRegister;

    using Operator::DynamicOutputRegister;
    using Operator::OutputRegister;

    using Operator::AttrRegister;
    using Operator::RequiredAttrWithTypeRegister;

    using Operator::SubgraphRegister;
  };

  /**
   * @brief IR属性类型枚举
   */
  enum IrAttrType {
    kEsAttrRequired,  ///< 必需属性
    kEsAttrOptional   ///< 可选属性
  };

  /**
   * @brief IR输入类型枚举
   */
  enum IrInputType {
    kEsIrInputRequired,  ///< 必需输入
    kEsIrInputOptional,  ///< 可选输入
    kEsIrInputDynamic,   ///< 动态输入
    kEsIrInputTypeEnd
  };

  /**
   * @brief IR输出类型枚举
   */
  enum IrOutputType {
    kEsIrOutputRequired,  ///< 必需输出
    kEsIrOutputDynamic,   ///< 动态输出
    kEsIrOutputTypeEnd
  };

  /**
   * @brief IR属性定义结构体
   */
  struct IrAttrDef {
    std::string attr_name;
    IrAttrType ir_attr_type;
    std::string attr_data_type;  // see `kIrAttrTypesMap` in `operator.cc`
    AttrValue attr_default_value;
  };

  /**
   * @brief IR输入定义结构体
   */
  struct IrInputDef {
    std::string name;
    IrInputType ir_input_type;
    std::string symbol_id;
  };

  /**
   * @brief IR输出定义结构体
   */
  struct IrOutputDef {
    std::string name;
    IrOutputType ir_output_type;
    std::string symbol_id;
  };

 public:
  /**
   * @brief 构造函数
   * 需要保证入参有效, 指向合法内存，否则会抛出异常。
   * 内部构建失败时不捕获异常，直接抛出给用户侧。
   * @param graph 所属的图对象
   */
  explicit CompliantNodeBuilder(ge::Graph *graph);

  ~CompliantNodeBuilder();
  CompliantNodeBuilder(CompliantNodeBuilder &&) noexcept;
  CompliantNodeBuilder &operator=(CompliantNodeBuilder &&) noexcept;

  CompliantNodeBuilder(const CompliantNodeBuilder &) = delete;
  CompliantNodeBuilder &operator=(const CompliantNodeBuilder &) = delete;
  /**
   * @brief 设置操作符类型
   * @param type 操作符类型字符串
   * @return 当前构建器对象的引用，支持链式调用
   */
  CompliantNodeBuilder &OpType(const char_t *type);

  /**
   * @brief 定义IR输入规范
   * @param input_ir_def 输入IR定义向量
   * @return 当前构建器对象的引用，支持链式调用
   */
  CompliantNodeBuilder &IrDefInputs(std::vector<IrInputDef> input_ir_def);

  /**
   * @brief 定义IR输出规范
   * @param output_ir_def 输出IR定义向量
   * @return 当前构建器对象的引用，支持链式调用
   */
  CompliantNodeBuilder &IrDefOutputs(std::vector<IrOutputDef> output_ir_def);

  /**
   * @brief 定义IR属性规范
   * @param attr_ir_def 属性IR定义向量
   * @return 当前构建器对象的引用，支持链式调用
   */
  CompliantNodeBuilder &IrDefAttrs(std::vector<IrAttrDef> attr_ir_def);

  /**
   * @brief 设置节点名称
   * @param name 节点名称
   * @return 当前构建器对象的引用，支持链式调用
   */
  CompliantNodeBuilder &Name(const char_t *name);

  /**
   * @brief 设置动态输入实例数量
   * @param ir_name IR输入名称
   * @param num 实例数量
   * @return 当前构建器对象的引用，支持链式调用
   */
  CompliantNodeBuilder &InstanceDynamicInputNum(const char_t *ir_name, int32_t num);

  /**
   * @brief 设置动态输出实例数量
   * @param ir_name IR输出名称
   * @param num 实例数量
   * @return 当前构建器对象的引用，支持链式调用
   */
  CompliantNodeBuilder &InstanceDynamicOutputNum(const char_t *ir_name, int32_t num);

  /**
   * @brief 设置输出数据类型
   * @param name 输出名称
   * @param data_type 数据类型
   * @return 当前构建器对象的引用，支持链式调用
   */
  CompliantNodeBuilder &InstanceOutputDataType(const char_t *name, ge::DataType data_type);

  /**
   * @brief 设置输出形状
   * @param name 输出名称
   * @param shape 形状向量
   * @return 当前构建器对象的引用，支持链式调用
   */
  CompliantNodeBuilder &InstanceOutputShape(const char_t *name, const std::vector<int64_t> &shape);

  /**
   * @brief 设置输出原始形状
   * @param name 输出名称
   * @param shape 原始形状向量
   * @return 当前构建器对象的引用，支持链式调用
   */
  CompliantNodeBuilder &InstanceOutputOriginShape(const char_t *name, const std::vector<int64_t> &shape);

  /**
   * @brief 设置输出存储形状
   * @param name 输出名称
   * @param shape 存储形状向量
   * @return 当前构建器对象的引用，支持链式调用
   */
  CompliantNodeBuilder &InstanceOutputStorageShape(const char_t *name, const std::vector<int64_t> &shape);

  /**
   * @brief 设置输出格式
   * @param name 输出名称
   * @param format 格式
   * @return 当前构建器对象的引用，支持链式调用
   */
  CompliantNodeBuilder &InstanceOutputFormat(const char_t *name, ge::Format format);

  /**
   * @brief 设置输出原始格式
   * @param name 输出名称
   * @param format 原始格式
   * @return 当前构建器对象的引用，支持链式调用
   */
  CompliantNodeBuilder &InstanceOutputOriginFormat(const char_t *name, ge::Format format);

  /**
   * @brief 设置输出存储格式
   * @param name 输出名称
   * @param format 存储格式
   * @return 当前构建器对象的引用，支持链式调用
   */
  CompliantNodeBuilder &InstanceOutputStorageFormat(const char_t *name, ge::Format format);

  /**
   * @brief 构建并返回图节点
   * @return 构建完成的图节点对象
   */
  ge::GNode Build() const;

 private:
  class CompliantNodeBuilderImpl;
  std::unique_ptr<CompliantNodeBuilderImpl> impl_;
};
}  // namespace es
}  // namespace ge

#endif  // AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_COMPLIANT_NODE_BUILDER_H_
