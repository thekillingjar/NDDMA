/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_ES_TENSOR_LIKE_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_ES_TENSOR_LIKE_H_

#include <memory>
#include <vector>
#include "es_tensor_holder.h"
#include "es_c_tensor_holder.h"
#include "es_graph_builder.h"

namespace ge {
namespace es {

/**
 * @brief 类 Tensor 类型，支持 EsTensorHolder、标量和向量
 *
 * 该类用于统一处理 Tensor 输入和标量/向量输入，支持：
 * - EsTensorHolder
 * - int64_t、float
 * - std::vector<int64_t>、std::vector<float>
 */
class EsTensorLike {
 public:
  /**
   * @brief 构造函数，使用张量持有者、标量和多维向量初始化
   * @param tensor 张量持有者、标量和多维向量
   */
  EsTensorLike(const EsTensorHolder &tensor);
  EsTensorLike(const std::nullptr_t);
  EsTensorLike(const int64_t value);
  EsTensorLike(const float value);
  EsTensorLike(const std::vector<int64_t> &values);
  EsTensorLike(const std::vector<float> &values);

  /**
   * @brief 析构函数
   */
  ~EsTensorLike();

  EsTensorLike(const EsTensorLike &) = delete;
  EsTensorLike(EsTensorLike &&) = delete;
  EsTensorLike &operator=(const EsTensorLike &) = delete;
  EsTensorLike &operator=(EsTensorLike &&) = delete;

  /**
   * @brief 获取当前 TensorLike 对应 Tensor 的 owner builder。
   *
   * @return EsCGraphBuilder指针
   */
  EsCGraphBuilder *GetOwnerBuilder() const;

  /**
   * @brief 将当前 EsTensorLike 转换为 EsTensorHolder
   *
   * - 若本身是 Tensor，则直接返回内部 TensorHolder；
   * - 若是标量/向量，则创建对应的常量 Tensor。
   *
   * @param graph 所属的图构建器
   * @return 转换后的EsTensorHolder
   */
  EsTensorHolder ToTensorHolder(EsCGraphBuilder *graph) const;

 private:
  class EsTensorLikeImpl;
  std::unique_ptr<EsTensorLikeImpl> impl_;
};

/**
 * @brief 从EsTensorLike对象中解析所属的 EsCGraphBuilder
 * @param tensor_like EsTensorLike对象（可能是EsTensorHolder、标量、张量等包装类型）
 * @return 如果 tensor_like 关联了 GraphBuilder，则返回对应 EsCGraphBuilder*；否则返回 nullptr
 */
inline EsCGraphBuilder *ResolveBuilderImpl(const EsTensorLike &tensor_like) {
  return tensor_like.GetOwnerBuilder();
}

/**
 * @brief 从EsTensorHolder向量中解析所属的 EsCGraphBuilder
 * @param tensors EsTensorHolder向量
 * @return 第一个解析成功的 C 层 builder 指针；若都失败则返回 nullptr
 */
inline EsCGraphBuilder *ResolveBuilderImpl(const std::vector<EsTensorHolder> &tensors) {
  for (const auto &tensor : tensors) {
    EsCGraphBuilder *builder = ResolveBuilderImpl(tensor);
    if (builder != nullptr) {
      return builder;
    }
  }
  return nullptr;
}

/**
 * @brief 从 EsCTensorHolder* 中解析所属的 EsCGraphBuilder
 * @param esb_tensor EsCTensorHolder 指针（可为空）
 * @return 若 esb_tensor 非空且已绑定 owner builder，则返回对应的 C 层 builder 指针；否则返回 nullptr
 */
inline EsCGraphBuilder *ResolveBuilderImpl(EsCTensorHolder *esb_tensor) {
  if (esb_tensor != nullptr) {
    return &esb_tensor->GetOwnerBuilder();
  }
  return nullptr;
}

/**
 * @brief 从 EsCTensorHolder* 向量中解析所属的 EsCGraphBuilder
 * @param esb_tensors EsCTensorHolder 指针向量
 * @return 第一个解析成功的 C 层 builder 指针；若都失败则返回 nullptr
 */
inline EsCGraphBuilder *ResolveBuilderImpl(const std::vector<EsCTensorHolder *> &esb_tensors) {
  for (auto *esb_tensor : esb_tensors) {
    if (esb_tensor != nullptr) {
      EsCGraphBuilder *builder = ResolveBuilderImpl(esb_tensor);
      if (builder != nullptr) {
        return builder;
      }
    }
  }
  return nullptr;
}

/**
 * @brief 从 EsGraphBuilder 指针中获取底层的 EsCGraphBuilder
 * @param graph_builder 图构建指针
 * @return C层builder指针
 */
inline EsCGraphBuilder *ResolveBuilderImpl(const EsGraphBuilder *graph_builder) {
  if (graph_builder != nullptr) {
    return graph_builder->GetCGraphBuilder();
  }
  return nullptr;
}

/**
 * @brief 直接返回已是 EsCGraphBuilder* 的参数
 * @param graph_builder C层builder指针（可为空）
 * @return 返回 graph_builder；若为 nullptr，则返回 nullptr
 */
inline EsCGraphBuilder *ResolveBuilderImpl(EsCGraphBuilder *graph_builder) {
  return graph_builder;
}

/**
 * @brief 处理 nullptr_t 参数的实现
 */
inline EsCGraphBuilder *ResolveBuilderImpl(const std::nullptr_t) {
  return nullptr;
}

/**
 * @brief 无参数时返回nullptr
 */
inline EsCGraphBuilder *ResolveBuilder() {
  return nullptr;
}

/**
 * @brief 从一组参数中解析所属的 EsCGraphBuilder
 *
 * 支持的参数类型包括：
 * - EsTensorLike
 * - std::vector<EsTensorHolder>
 * - EsCTensorHolder*
 * - std::vector<EsCTensorHolder*>
 * - EsGraphBuilder
 * - EsCGraphBuilder*
 * - std::nullptr_t
 *
 * @tparam T   第一个参数的类型。
 * @tparam Ts  其余参数的类型。
 * @param first 第一个参与解析的参数。
 * @param rest  剩余参与解析的参数。
 * @return 第一个成功解析出来的 C 层 builder 指针；若全部失败则返回 nullptr。
 */
template <typename T, typename... Ts>
inline EsCGraphBuilder *ResolveBuilder(const T& first, const Ts&... rest) {
  EsCGraphBuilder *builder = ResolveBuilderImpl(first);
  if (builder != nullptr) {
    return builder;
  }
  return ResolveBuilder(rest...);
}
}  // namespace es
}  // namespace ge

#endif  // AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_ES_TENSOR_LIKE_H_
