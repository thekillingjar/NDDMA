/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_ES_GRAPH_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_ES_GRAPH_H_

#include <memory>
#include <vector>
#include "es_tensor_holder.h"
#include "graph/graph.h"
#include "graph/attr_value.h"
#include "compliant_node_builder.h"
#include "es_c_graph_builder.h"

namespace ge {
namespace es {

/**
 * @brief 创建指定数据类型和张量形状的张量
 * @tparam T 张量数据类型
 * @param value 张量数据指针
 * @param dims 张量维度数组指针
 * @param dim_num 张量维度数量
 * @param dt 张量的数据类型
 * @param format 张量格式，默认为FORMAT_ND
 * @return 返回创建的张量智能指针，失败时返回nullptr
 *
 * 该函数用于创建具有指定数据类型、形状和格式的张量。
 * 如果dims为空，则创建标量张量。
 */
template <typename T>
std::unique_ptr<Tensor> CreateTensor(const T *value, const int64_t *dims, int64_t dim_num, DataType dt,
                                     Format format = FORMAT_ND) {
  int64_t shape_size = 1;  // dims为空代表标量，默认值使用1
  std::vector<int64_t> dims_vec;
  Shape shape;
  if (dims != nullptr) {
    dims_vec.assign(dims, dims + dim_num);
    shape = Shape{dims_vec};
    shape_size = shape.GetShapeSize();
  }
  TensorDesc td{shape, format, dt};
  td.SetOriginShape(shape);
  auto tensor = std::unique_ptr<Tensor>(new (std::nothrow) Tensor(td));
  if (tensor == nullptr) {
    return nullptr;
  }
  tensor->SetData(static_cast<const uint8_t *>(static_cast<const void *>(value)), sizeof(T) * shape_size);
  return tensor;
}

/**
 * @brief 创建指定数据类型和张量形状的张量
 * @tparam T 张量数据类型
 * @param data_file_path binary文件数据地址
 * @param dims 张量维度数组指针
 * @param dt 张量的数据类型
 * @param format 张量格式，默认为FORMAT_ND
 * @return 返回创建的张量智能指针，失败时返回nullptr
 */
template <typename T>
std::unique_ptr<Tensor> CreateTensorFromFile(const char *data_file_path, const std::vector<int64_t> &dims, DataType dt,
                                             Format format = FORMAT_ND) {
  auto es_c_tensor = EsCreateEsCTensorFromFile(data_file_path, dims.data(), dims.size(), static_cast<C_DataType>(dt),
                                               static_cast<C_Format>(format));

  return std::unique_ptr<Tensor>(static_cast<Tensor *>(static_cast<void *>(es_c_tensor)));
}

/**
 * @brief 创建指定类型的Const算子（原始指针版本）
 * @tparam T 张量数据类型
 * @param graph 图构建器指针
 * @param value 张量数据指针
 * @param dims 张量维度数组指针
 * @param dim_num 维度数量
 * @param dt 张量的数据类型
 * @param format 张量格式，默认为FORMAT_ND
 * @return 返回创建的Const的张量持有者算子，失败时返回nullptr
 *
 * 该函数用于创建具有指定数据类型和形状的Const算子。
 * 支持原始指针输入，适用于C风格数组。
 */
template <typename T>
EsCTensorHolder *EsCreateConst(EsCGraphBuilder *graph, const T *value, const int64_t *dims, int64_t dim_num,
                                ge::DataType dt, ge::Format format = FORMAT_ND) {
  if (graph == nullptr || value == nullptr) {
    return nullptr;
  }
  auto tensor = ge::es::CreateTensor<T>(value, dims, dim_num, dt, format);
  if (tensor == nullptr) {
    return nullptr;
  }
  auto av = ge::AttrValue();
  if (av.SetAttrValue(*tensor) != ge::GRAPH_SUCCESS) {
    return nullptr;
  }
  if (graph->GetGraph() == nullptr) {
    return nullptr;
  }
  auto node = ge::es::CompliantNodeBuilder(graph->GetGraph())
                  .OpType("Const")
                  .Name(graph->GenerateNodeName("Const").GetString())
                  .IrDefOutputs({{"y", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                  .IrDefAttrs({{"value", ge::es::CompliantNodeBuilder::kEsAttrOptional, "VT_TENSOR", av}})
                  .InstanceOutputShape("y", tensor->GetTensorDesc().GetShape().GetDims())
                  .InstanceOutputDataType("y", dt)
                  .InstanceOutputFormat("y", format)
                  .Build();
  return graph->GetTensorHolderFromNode(node, 0);
}

/**
 * @brief 图构建器类，用于构建计算图
 *
 * 该类提供了创建输入、变量、标量、向量等图元素的方法，
 * 以及设置图属性和构建最终图的功能。
 */
class EsGraphBuilder {
 public:
  /**
   * @brief 构造函数，创建指定名称的图构建器
   * @param name 构建的图名称
   */
  explicit EsGraphBuilder(const char *name) : graph_builder_(EsCreateGraphBuilder(name), EsDestroyGraphBuilder) {}

  /**
   * @brief 创建指定数据类型和张量形状的张量
   * @tparam T 张量数据类型
   * @param value 张量数据指针
   * @param dims 张量维度数组指针
   * @param dt 张量的数据类型
   * @param format 张量格式，默认为FORMAT_ND
   * @return 返回创建的张量智能指针，失败时返回nullptr
   */
  template <typename T>
  std::unique_ptr<Tensor> CreateTensor(const std::vector<T> &value, const std::vector<int64_t> &dims, DataType dt,
                                       Format format = FORMAT_ND) {
    EsCTensor *ret_esc_tensor = EsCreateEsCTensor(value.data(), dims.data(), static_cast<int64_t>(dims.size()),
                                                  static_cast<C_DataType>(dt), static_cast<C_Format>(format));

    return std::unique_ptr<Tensor>(static_cast<Tensor *>(static_cast<void *>(ret_esc_tensor)));
  }

  /**
   * @brief 创建bool数据类型和张量形状的张量
   * @param value 张量数据指针
   * @param dims 张量维度数组指针
   * @param format 张量格式，默认为FORMAT_ND
   * @return 返回创建的张量智能指针，失败时返回nullptr
   */
  std::unique_ptr<Tensor> CreateBoolTensor(const std::vector<bool> &value, const std::vector<int64_t> &dims,
                                       Format format = FORMAT_ND) {
    std::vector<uint8_t> converted_value{};
    converted_value.reserve(value.size());
    std::transform(value.begin(), value.end(), std::back_inserter(converted_value),
                   [](const bool &ele) { return ele ? 1 : 0; });
    EsCTensor *ret_esc_tensor =
        EsCreateEsCTensor(converted_value.data(), dims.data(), static_cast<int64_t>(dims.size()),
                          static_cast<C_DataType>(ge::DT_BOOL), static_cast<C_Format>(format));

    return std::unique_ptr<Tensor>(static_cast<Tensor *>(static_cast<void *>(ret_esc_tensor)));
  }

  /**
   * @brief 创建图输入节点
   * @param index 输入节点的索引
   * @param name 输入节点的名称
   * @param type 输入节点的类型字符串
   * @return 返回创建的输入张量持有者
   *
   * 创建具有指定名称和类型的图输入节点，使用默认的float数据类型和ND格式。
   */
  EsTensorHolder CreateInput(int64_t index, const char *name, const char *type) {
    return EsCreateGraphInputWithDetails(graph_builder_.get(), index, name, type, C_DT_FLOAT, C_FORMAT_ND, nullptr, 0);
  }

  /**
   * @brief 创建指定数量的输入节点，返回数组，适用于编译时的结构化绑定
   * @tparam inputs_num 输入节点数量
   * @param start_index 输入节点起始索引, 如果不为0表示前面已经创建了其他输入节点, 整体输入节点的索引应该从0开始连续递增
   * @return 返回创建的输入张量持有者数组
   *
   * 创建默认数据类型是float，格式为ND的标量节点。
   */

  template <size_t inputs_num>
  std::array<EsTensorHolder, inputs_num> CreateInputs(size_t start_index = 0) {
    std::array<EsTensorHolder, inputs_num> rets;
    for (size_t index = 0; index < inputs_num; index++) {
      rets[index] = CreateInput(start_index + index);
    }
    return rets;
  }

  /**
   * @brief 创建指定数量的输入节点，返回容器，适用于运行时动态绑定
   * @param inputs_num 输入节点数量
   * @param start_index 输入节点起始索引, 如果不为0表示前面已经创建了其他输入节点, 整体输入节点的索引应该从0开始连续递增
   * @return 返回创建的输入张量持有者容器
   *
   * 创建默认数据类型是float，格式为ND的标量节点。
   */
  std::vector<EsTensorHolder> CreateInputs(size_t inputs_num, size_t start_index = 0) {
    std::vector<EsTensorHolder> inputs;
    inputs.reserve(inputs_num);
    for (size_t i = 0; i < inputs_num; ++i) {
      inputs.emplace_back(CreateInput(static_cast<int64_t>(start_index + i)));
    }
    return inputs;
  }

  /**
   * @brief 创建默认输入节点，从0开始计数, 节点命名为input_{index}
   * @param index 输入节点的索引
   * @return 返回创建的输入张量持有者
   *
   * 创建默认数据类型是float，格式为ND的标量节点。
   */
  EsTensorHolder CreateInput(int64_t index) {
    return EsCreateGraphInput(graph_builder_.get(), index);
  }

  /**
   * @brief 创建指定名称的输入节点
   * @param index 输入节点的索引
   * @param name 输入节点的名称
   * @return 返回创建的输入张量持有者
   *
   * 创建具有指定名称的输入节点，使用默认的float数据类型和ND格式。
   */
  EsTensorHolder CreateInput(int64_t index, const char *name) {
    return EsCreateGraphInputWithDetails(graph_builder_.get(), index, name, nullptr, C_DT_FLOAT, C_FORMAT_ND, nullptr,
                                         0);
  }

  /**
   * @brief 创建具有完整信息的输入节点
   * @param index 输入节点的索引
   * @param name 输入节点的名称
   * @param data_type 数据类型
   * @param format 张量格式
   * @param shape 张量形状
   * @return 返回创建的输入张量持有者
   *
   * 创建具有指定数据类型、格式和形状的输入节点。
   */
  EsTensorHolder CreateInput(int64_t index, const char *name, ge::DataType data_type, ge::Format format,
                             const std::vector<int64_t> &shape) {
    return EsCreateGraphInputWithDetails(graph_builder_.get(), index, name, nullptr, static_cast<C_DataType>(data_type),
                                         static_cast<C_Format>(format), shape.data(), shape.size());
  }

  /**
   * @brief 创建int64类型的Const算子
   * @param value 张量数据指针
   * @param dims 张量维度数组指针
   * @return 返回创建的Const的张量持有者算子
   */
  EsTensorHolder CreateConst(const std::vector<int64_t> &value, const std::vector<int64_t> dims) {
    return EsCreateConstInt64(graph_builder_.get(), value.data(), dims.data(), dims.size());
  }

  /**
   * @brief 创建int32类型的Const算子
   * @param value 张量数据指针
   * @param dims 张量维度数组指针
   * @return 返回创建的Const的张量持有者算子
   */
  EsTensorHolder CreateConst(const std::vector<int32_t> &value, const std::vector<int64_t> dims) {
    return EsCreateConstInt32(graph_builder_.get(), value.data(), dims.data(), dims.size());
  }

  /**
   * @brief 创建uint64类型的Const算子
   * @param value 张量数据指针
   * @param dims 张量维度数组指针
   * @return 返回创建的Const的张量持有者算子
   */
  EsTensorHolder CreateConst(const std::vector<uint64_t> &value, const std::vector<int64_t> dims) {
    return EsCreateConstUInt64(graph_builder_.get(), value.data(), dims.data(), dims.size());
  }

  /**
   * @brief 创建uint32类型的Const算子
   * @param value 张量数据指针
   * @param dims 张量维度数组指针
   * @return 返回创建的Const的张量持有者算子
   */
  EsTensorHolder CreateConst(const std::vector<uint32_t> &value, const std::vector<int64_t> dims) {
    return EsCreateConstUInt32(graph_builder_.get(), value.data(), dims.data(), dims.size());
  }

  /**
   * @brief 创建float类型的Const算子
   * @param value 张量数据指针
   * @param dims 张量维度数组指针
   * @return 返回创建的Const的张量持有者算子
   */
  EsTensorHolder CreateConst(const std::vector<float> &value, const std::vector<int64_t> dims) {
    return EsCreateConstFloat(graph_builder_.get(), value.data(), dims.data(), dims.size());
  }

  /**
   * @brief 创建指定类型、格式和维度的Const算子（通用模板方法）
   * @tparam T 张量数据类型
   * @param value 张量数据向量
   * @param dims 张量维度向量
   * @param dt 张量的数据类型
   * @param format 张量格式，默认为FORMAT_ND
   * @return 返回创建的Const的张量持有者算子
   *
   * 创建具有指定数据类型、格式和形状的Const算子。
   */
  template <typename T>
  EsTensorHolder CreateConst(const std::vector<T> &value, const std::vector<int64_t> &dims, ge::DataType dt,
                             ge::Format format = FORMAT_ND) {
    return EsTensorHolder(
        ge::es::EsCreateConst<T>(graph_builder_.get(), value.data(), dims.data(), static_cast<int64_t>(dims.size()),
                                 dt, format));
  }

  /**
   * @brief 创建int64类型的向量常量
   * @param value 向量值
   * @return 返回创建的向量张量持有者
   */
  EsTensorHolder CreateVector(const std::vector<int64_t> &value) {
    return EsCreateVectorInt64(graph_builder_.get(), value.data(), static_cast<int64_t>(value.size()));
  }

  /**
   * @brief 创建int32类型的向量常量
   * @param value 向量值
   * @return 返回创建的向量张量持有者
   */
  EsTensorHolder CreateVector(const std::vector<int32_t> &value) {
    return EsCreateVectorInt32(graph_builder_.get(), value.data(), static_cast<int64_t>(value.size()));
  }

  /**
   * @brief 创建uint64类型的向量常量
   * @param value 向量值
   * @return 返回创建的向量张量持有者
   */
  EsTensorHolder CreateVector(const std::vector<uint64_t> &value) {
    return EsCreateVectorUInt64(graph_builder_.get(), value.data(), static_cast<int64_t>(value.size()));
  }

  /**
   * @brief 创建uint32类型的向量常量
   * @param value 向量值
   * @return 返回创建的向量张量持有者
   */
  EsTensorHolder CreateVector(const std::vector<uint32_t> &value) {
    return EsCreateVectorUInt32(graph_builder_.get(), value.data(), static_cast<int64_t>(value.size()));
  }

  /**
   * @brief 创建float类型的向量常量
   * @param value 向量值
   * @return 返回创建的向量张量持有者
   */
  EsTensorHolder CreateVector(const std::vector<float> &value) {
    return EsCreateVectorFloat(graph_builder_.get(), value.data(), static_cast<int64_t>(value.size()));
  }

  /**
   * @brief 创建int64类型的标量常量
   * @param value 标量值
   * @return 返回创建的标量张量持有者
   */
  EsTensorHolder CreateScalar(int64_t value) {
    return EsCreateScalarInt64(graph_builder_.get(), value);
  }

  /**
   * @brief 创建int32类型的标量常量
   * @param value 标量值
   * @return 返回创建的标量张量持有者
   */
  EsTensorHolder CreateScalar(int32_t value) {
    return EsCreateScalarInt32(graph_builder_.get(), value);
  }

  /**
   * @brief 创建uint64类型的标量常量
   * @param value 标量值
   * @return 返回创建的标量张量持有者
   */
  EsTensorHolder CreateScalar(uint64_t value) {
    return EsCreateScalarUInt64(graph_builder_.get(), value);
  }

  /**
   * @brief 创建uint32类型的标量常量
   * @param value 标量值
   * @return 返回创建的标量张量持有者
   */
  EsTensorHolder CreateScalar(uint32_t value) {
    return EsCreateScalarUInt32(graph_builder_.get(), value);
  }

  /**
   * @brief 创建float类型的标量常量
   * @param value 标量值
   * @return 返回创建的标量张量持有者
   */
  EsTensorHolder CreateScalar(float value) {
    return EsCreateScalarFloat(graph_builder_.get(), value);
  }

  /**
   * @brief 创建变量
   * @param index
   * @param name var的名称相同代表共享，因此我们要求显示指定
   * @return
   */
  EsTensorHolder CreateVariable(int32_t index, const char *name) {
    return EsCreateVariable(graph_builder_.get(), index, name);
  }

  /**
   * @brief 设置图的int64类型属性
   * @param attr_name 属性名称
   * @param value 属性值
   * @return 返回操作状态, 成功为SUCCESS，其他失败
   */
  Status SetAttr(const char *attr_name, int64_t value) {
    return static_cast<Status>(EsSetInt64AttrForGraph(graph_builder_.get(), attr_name, value));
  }

  /**
   * @brief 设置图的字符串类型属性
   * @param attr_name 属性名称
   * @param value 属性值
   * @return 返回操作状态, 成功为SUCCESS，其他失败
   */
  Status SetAttr(const char *attr_name, const char *value) {
    return static_cast<Status>(EsSetStringAttrForGraph(graph_builder_.get(), attr_name, value));
  }

  /**
   * @brief 设置图的布尔类型属性
   * @param attr_name 属性名称
   * @param value 属性值
   * @return 返回操作状态, 成功为SUCCESS，其他失败
   */
  Status SetAttr(const char *attr_name, bool value) {
    return static_cast<Status>(EsSetBoolAttrForGraph(graph_builder_.get(), attr_name, value));
  }

  /**
   * @brief 设置图的输出节点
   * @param tensor 输出张量持有者
   * @param index 输出索引
   * @return 返回操作结果，成功为0，其他失败
   */
  static int64_t SetOutput(const EsTensorHolder &tensor, int64_t index) {
    return EsSetGraphOutput(tensor.tensor_holder_, index);
  }

  /**
   * @brief 构建计算图
   * @return 返回构建的图智能指针
   *
   * 根据已设置的输入、操作和输出构建最终的计算图。
   */
  std::unique_ptr<Graph> BuildAndReset() const {
    return std::unique_ptr<Graph>(
        static_cast<Graph *>(static_cast<void *>(EsBuildGraphAndReset(graph_builder_.get()))));
  }

  /**
   * @brief 根据输出张量列表构建计算图
   * @param outputs 输出张量持有者列表
   * @return 返回构建的图智能指针
   *
   * 自动设置所有输出张量，然后构建计算图。
   */
  std::unique_ptr<Graph> BuildAndReset(const std::vector<EsTensorHolder> &outputs) {
    for (size_t i = 0U; i < outputs.size(); ++i) {
      SetOutput(outputs[i], static_cast<int64_t>(i));
    }
    return BuildAndReset();
  }

  /**
   * @brief 获取底层的C语言图构建器
   * @return 返回C语言图构建器指针
   */
  EsCGraphBuilder *GetCGraphBuilder() const {
    return graph_builder_.get();
  }

 private:
  std::unique_ptr<EsCGraphBuilder, decltype(&EsDestroyGraphBuilder)> graph_builder_;
};

/**
 * @brief 将EsTensorHolder向量转换为EsCTensorHolder指针向量
 * @param tensors EsTensorHolder向量
 * @return 返回EsCTensorHolder指针向量
 *
 * 该函数用于在C++接口和C接口之间进行转换。
 */
inline std::vector<EsCTensorHolder *> TensorsToEsCTensorHolders(const std::vector<EsTensorHolder> &tensors) {
  std::vector<EsCTensorHolder *> esb_tensors;
  esb_tensors.reserve(tensors.size());
  for (const auto &tensor : tensors) {
    esb_tensors.push_back(tensor.GetCTensorHolder());
  }
  return esb_tensors;
}

/**
 * @brief ge::DataType集合转换为C_DataType集合
 * @param data_types ge::DataType类型容器
 * @return C_DataType类型容器
 *
 * 该函数用于在C++接口和C接口之间进行转换。
 */
inline std::vector<C_DataType> DataTypesToEsCDataTypes(const std::vector<ge::DataType> &data_types) {
  std::vector<C_DataType> es_c_data_types;
  es_c_data_types.reserve(data_types.size());
  for (const auto &date_type : data_types) {
    es_c_data_types.push_back(static_cast<C_DataType>(date_type));
  }
  return es_c_data_types;
}

/**
 * @brief 将ListListType属性入参转换为包含List of Ptrs和List inner size的pair
 * @tparam T List of List的类型
 * @param list_list_type ListListType属性入参
 * @return 包含List of Ptrs和List inner size的pair
 */
template <typename T>
std::pair<std::vector<const T *>, std::vector<int64_t>> ListListTypeToPtrAndCounts(
    const std::vector<std::vector<T>> &list_list_type) {
  std::vector<const T *> list_int_ptr;
  list_int_ptr.reserve(list_list_type.size());
  std::vector<int64_t> inner_sizes;
  inner_sizes.reserve(list_list_type.size());
  for (const auto &list_type : list_list_type) {
    list_int_ptr.push_back(list_type.data());
    inner_sizes.emplace_back(list_type.size());
  }

  return {list_int_ptr, inner_sizes};
}

/**
 * @brief 将动态子图转换为匿名指针类型子图容器
 * @param graphs 控制算子的动态子图入参容器
 * @return 匿名指针类型的子图容器
 *
 * 该函数用于在C++接口和C接口之间进行转换。
 */
inline std::vector<EsCGraph *> GeGraphsToEsCGraphs(std::vector<std::unique_ptr<ge::Graph>> graphs) {
  std::vector<EsCGraph *> esb_subgraphs;
  esb_subgraphs.reserve(graphs.size());
  for (auto &subgraph : graphs) {
    esb_subgraphs.emplace_back(static_cast<EsCGraph *>(static_cast<void *>(subgraph.release())));
  }

  return esb_subgraphs;
}
}  // namespace es
}  // namespace ge

#endif  // AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_ES_GRAPH_H_
