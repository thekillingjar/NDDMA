/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_ESB_FUNCS_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_ESB_FUNCS_H_
#include <cstdint>
#include "graph/c_types.h"

#ifdef __cplusplus
extern "C" {
#endif

struct EsCTensorHolder;
struct EsCGraphBuilder;
struct EsCTensor;
struct EsCGraph;
struct EsCNode;
/**
 * 创建一个图构建器
 * @param name 可选，图的名字，如果为空则默认为`"graph"`
 * @return
 */
EsCGraphBuilder *EsCreateGraphBuilder(const char *name);
/**
 * 释放图构建器
 * @param graph
 */
void EsDestroyGraphBuilder(EsCGraphBuilder *graph);

/**
 * 创建一个图上的输入
 * @param graph
 * @param index 第几个图的输入，从0开始计数
 * @param name 输入的名字，如果为空则默认为input_{index}
 * @param type 输入的类型，如果为空则默认为Data
 * @param data_type 数据类型
 * @param format 格式类型
 * @param shape && dim_num shape信息, shape如果为空则为标量
 * @return 返回的tensor资源被图构建器管理
 */
EsCTensorHolder *EsCreateGraphInputWithDetails(EsCGraphBuilder *graph, int64_t index, const char *name,
                                               const char *type, C_DataType data_type, C_Format format,
                                               const int64_t *shape, int64_t dim_num);
/**
 * 创建一个图上的Data输入
 * @param graph
 * @param index 第几个图的输入，从0开始计数, 节点命名为input_{index}
 * @return
 */
EsCTensorHolder *EsCreateGraphInput(EsCGraphBuilder *graph, int64_t index);
/**
 * 设置tensor的数据类型，不设置默认为float
 * @param tensor
 * @param data_type
 * @return 成功为0，其他失败
 */
uint32_t EsSetDataType(EsCTensorHolder *tensor, C_DataType data_type);
/**
 * 设置tensor的内存格式，不设置默认为ND
 * @param tensor
 * @param format
 * @return 成功为0，其他失败
 */
uint32_t EsSetFormat(EsCTensorHolder *tensor, C_Format format);
/**
 * 设置tensor的shape，不设置默认为[],也就是标量
 * @param tensor
 * @param shape
 * @param dim_num
 * @return 成功为0，其他失败
 */
uint32_t EsSetShape(EsCTensorHolder *tensor, const int64_t *shape, int64_t dim_num);
/**
 * 设置tensor的shape的符号化表达，不设置默认无符号化信息
 * @param tensor
 * @param shape
 * @param dim_num
 * @return 成功为0，其他失败
 */
uint32_t EsSetOriginSymbolShape(EsCTensorHolder *tensor, const char *const *shape, int64_t dim_num);

// 创建const节点的一系列函数, 如要外部保证value内存的合法性
EsCTensorHolder *EsCreateConstInt64(EsCGraphBuilder *graph, const int64_t *value, const int64_t *dims, int64_t dim_num);
EsCTensorHolder *EsCreateConstInt32(EsCGraphBuilder *graph, const int32_t *value, const int64_t *dims, int64_t dim_num);
EsCTensorHolder *EsCreateConstUInt64(EsCGraphBuilder *graph, const uint64_t *value, const int64_t *dims,
                                     int64_t dim_num);
EsCTensorHolder *EsCreateConstUInt32(EsCGraphBuilder *graph, const uint32_t *value, const int64_t *dims,
                                     int64_t dim_num);
EsCTensorHolder *EsCreateConstFloat(EsCGraphBuilder *graph, const float *value, const int64_t *dims, int64_t dim_num);

EsCTensorHolder *EsCreateVectorInt64(EsCGraphBuilder *graph, const int64_t *value, int64_t dim);
EsCTensorHolder *EsCreateVectorInt32(EsCGraphBuilder *graph, const int32_t *value, int64_t dim);
EsCTensorHolder *EsCreateVectorUInt64(EsCGraphBuilder *graph, const uint64_t *value, int64_t dim);
EsCTensorHolder *EsCreateVectorUInt32(EsCGraphBuilder *graph, const uint32_t *value, int64_t dim);
EsCTensorHolder *EsCreateVectorFloat(EsCGraphBuilder *graph, const float *value, int64_t dim);

EsCTensorHolder *EsCreateScalarInt64(EsCGraphBuilder *graph, int64_t value);
EsCTensorHolder *EsCreateScalarInt32(EsCGraphBuilder *graph, int32_t value);
EsCTensorHolder *EsCreateScalarUInt64(EsCGraphBuilder *graph, uint64_t value);
EsCTensorHolder *EsCreateScalarUInt32(EsCGraphBuilder *graph, uint32_t value);
EsCTensorHolder *EsCreateScalarFloat(EsCGraphBuilder *graph, float value);

/**
 * @param graph
 * @param index 节点索引
 * @param name 节点名称
 * @return
 */
EsCTensorHolder *EsCreateVariable(EsCGraphBuilder *graph, int32_t index, const char *name);

/**
 * 设置图的输出
 * @param tensor 被输出的Tensor
 * @param output_index 图的输出索引
 * @return 成功为0，其他失败
 */
uint32_t EsSetGraphOutput(EsCTensorHolder *tensor, int64_t output_index);

/**
 * 本接口返回`EsCGraphBuilder`实例的快照, 所有权不发生转移
 * @param tensor
 * @return 成功非空，失败为空
 */

EsCGraphBuilder *EsGetOwnerBuilder(EsCTensorHolder *tensor);

/**
 * 本接口返回`ge::GNode`实例的快照, 所有权不发生转移
 * @param tensor
 * @return 成功非空，失败为空
 */
EsCNode *EsGetProducer(EsCTensorHolder *tensor);

/**
 * 本接口返回`ge::Graph`实例的指针, 所有权转移给调用方
 * @param graph
 * @return 成功非空，失败为空
 */
EsCGraph *EsBuildGraphAndReset(EsCGraphBuilder *graph);

/**
 * @brief 控制边连接函数
 * @param dest_ctrl_tensor 控制边连边对象
 * @param src_ctrl_tensors 控制边输入
 * @param ctrl_tensors_num 控制边数量
 * @return 成功为0，其他失败
 */
uint32_t EsAddControlEdge(EsCTensorHolder *dest_ctrl_tensor, EsCTensorHolder **src_ctrl_tensors,
                          int64_t ctrl_tensors_num);

/**
 * @brief 本接口用于C用户创建EsCTensor
 * @param data 张量数据指针
 * @param dim 张量维度数组指针
 * @param dim_num 张量维度数量
 * @param data_type 张量的DataType枚举值
 * @param format 张量格式
 * @return 张量的匿名指针，所有权交给调用方控制, 失败时返回nullptr
 */
EsCTensor *EsCreateEsCTensor(const void *data, const int64_t *dim, int64_t dim_num, C_DataType data_type,
                             C_Format format);

/**
 * @brief 本接口用于C用户通过binary文件创建EsCTensor
 * @param data_file_path 张量binary数据文件路径
 * @param dim 张量维度数组指针
 * @param dim_num 张量维度数量
 * @param data_type 张量的DataType枚举值
 * @param format 张量格式
 * @return 张量的匿名指针，所有权交给调用方控制, 失败时返回nullptr
 */
EsCTensor *EsCreateEsCTensorFromFile(const char *data_file_path, const int64_t *dim, int64_t dim_num,
                                     C_DataType data_type, C_Format format);

// 下面是支持自定义属性的接口
uint32_t EsSetInt64AttrForGraph(EsCGraphBuilder *graph, const char *attr_name, int64_t value);
uint32_t EsSetStringAttrForGraph(EsCGraphBuilder *graph, const char *attr_name, const char *value);
uint32_t EsSetBoolAttrForGraph(EsCGraphBuilder *graph, const char *attr_name, bool value);

uint32_t EsSetInt64AttrForTensor(EsCTensorHolder *tensor, const char *attr_name, int64_t value);
uint32_t EsSetStringAttrForTensor(EsCTensorHolder *tensor, const char *attr_name, const char *value);
uint32_t EsSetBoolAttrForTensor(EsCTensorHolder *tensor, const char *attr_name, bool value);

uint32_t EsSetInt64AttrForNode(EsCTensorHolder *tensor, const char *attr_name, int64_t value);
uint32_t EsSetStringAttrForNode(EsCTensorHolder *tensor, const char *attr_name, const char *value);
uint32_t EsSetBoolAttrForNode(EsCTensorHolder *tensor, const char *attr_name, bool value);

#ifdef __cplusplus
}
#endif
#endif  // AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_ESB_FUNCS_H_
