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
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

struct EsbTensor;
struct EsbGraph;

EsbGraph *EsCreateGraph(const char *name);
void EsDestroyGraph(EsbGraph *graph);

/**
 * 创建一个图上的输入
 * @param graph
 * @param index 第几个图的输入，从0开始计数
 * @param name 可选，输入的名字，如果为空则默认为input_{index}
 * @param type 可选，输入的类型，如果为空则默认为Data
 * @return
 */
EsbTensor *EsCreateGraphInputWithDetails(EsbGraph *graph, int index, const char *name, const char *type);
#define EsCreateGraphInput(graph, index) EsCreateGraphInputWithDetails(graph, index, nullptr, nullptr)

int EsSetShape(EsbTensor *tensor, const int64_t *shape, int64_t dim_num);
int EsSetSymbolShape(EsbTensor *tensor, const char *const *shape, int64_t dim_num);
int EsSetInputSymbolShape(EsbTensor *tensor, const char *const *shape, int64_t dim_num);

EsbTensor *EsCreateConstInt64(EsbGraph *graph, const int64_t *value, const int64_t *dims, int64_t dim_num);
EsbTensor *EsCreateVectorInt64(EsbGraph *graph, const int64_t *value, int64_t dim);
EsbTensor *EsCreateScalarInt64(EsbGraph *graph, int64_t value);

EsbTensor *EsCreateConstInt32(EsbGraph *graph, const int32_t *value, const int64_t *dims, int64_t dim_num);
EsbTensor *EsCreateScalarInt32(EsbGraph *graph, int32_t value);

EsbTensor *EsCreateScalarFloat(EsbGraph *graph, float value);
EsbTensor *EsCreateScalarDouble(EsbGraph *graph, double value);

EsbTensor *EsCreateVariableInt32(EsbGraph *graph, int32_t index, const int32_t *value, const int64_t *dims,
                                 int64_t dim_num, const char *container, const char *shared_name);

EsbTensor *EsCreateVariableInt64(EsbGraph *graph, int32_t index, const int64_t *value, const int64_t *dims,
                                 int64_t dim_num, const char *container, const char *shared_name);

EsbTensor *EsCreateVariableFloat(EsbGraph *graph, int32_t index, const float *value, const int64_t *dims,
                                 int64_t dim_num, const char *container, const char *shared_name);

/**
 * 设置图的输出
 * @param tensor 被输出的Tensor
 * @param output_index 图的输出索引
 */
int EsSetGraphOutput(EsbTensor *tensor, int output_index);

EsbGraph *EsGetOwnerGraph(EsbTensor *tensor);

/**
 * 本接口返回`ge::Graph`实例的指针
 * @param graph
 * @return
 */
void *EsBuildGraph(EsbGraph *graph);

#ifdef __cplusplus
}
#endif


#endif  // AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_ESB_FUNCS_H_
