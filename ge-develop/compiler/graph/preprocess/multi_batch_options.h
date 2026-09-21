/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PREPROCESS_MULTI_BATCH_OPTIONS_H_
#define GE_GRAPH_PREPROCESS_MULTI_BATCH_OPTIONS_H_

#include <vector>

#include "ge/ge_api_error_codes.h"
#include "ge/ge_api_types.h"
#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "graph/node.h"

namespace ge {
namespace multibatch {
using DimsVector = std::vector<std::vector<std::string>>;
/// @ingroup ge
/// @brief Update Dynamic Param from Options.
/// @param [in] ComputeGraphPtr &graph: the train graph
/// @return SUCCESS: valid / PARAM_INVALID: invalid.
Status CheckSequenceOfOptions(const ComputeGraphPtr &graph, std::vector<NodePtr> &data_nodes,
                              std::vector<NodePtr> &getnext_nosink_nodes,
                              std::vector<NodePtr> &getnext_sink_nodes,
                              bool &need_multi_batch);

Status UpdateNameOfData(const ComputeGraphPtr &graph, const std::vector<NodePtr> &data_nodes);

Status UpdateNameOfInputShape(const ComputeGraphPtr &graph, const std::vector<NodePtr> &data_nodes,
                              const std::vector<NodePtr> &getnext_nosink_nodes,
                              const std::vector<NodePtr> &getnext_sink_nodes);

Status DeleteIdentityInsertByAdapter(const ComputeGraphPtr &graph);

Status CheckNegativeCountOfOptions(const std::vector<std::vector<int64_t>> &shapes);

/// @ingroup ge
/// @brief Init Dynamic Param from Options.
/// @param [out] std::vector<std::vector<int64_t>> &shapes: Result for Params.
/// @return Status: Init Status.
Status InitDynamicParams(std::vector<std::vector<int64_t>> &shapes);

/// @ingroup ge
/// @brief Update shape for data by user input.
/// @param [out] std::vector<std::vector<int64_t>> &shapes: Result for Params.
/// @return true: Configed for Multi batch / false: Not configed for Multi batch.
Status UpdateDataShapeByUserInput();

void SortDataNodesByIndex(std::vector<NodePtr> &data_nodes);
void SortDataNodesByName(std::vector<NodePtr> &data_nodes);

Status UpdateDataShape(const std::vector<NodePtr> &data_nodes);

Status ChangeStrToNum(const std::string &str, int64_t &num);

/// @ingroup ge
/// @brief Check Dynamic Param is invalid.
/// @param [in] const std::vector<vector<int64_t>> &shapes: Params for check.
/// @return SUCCESS: valid / PARAM_INVALID: invalid.
Status CheckDynamicParams(const std::vector<std::vector<int64_t>> &shapes);

/// @ingroup ge
/// @brief Get GeShape from configed shape.
/// @param [in] const std::vector<int64_t> &batch_shape: Configed shape.
/// @param [out] GeShape &data_shape: GeShape for configed shape.
/// @return SUCCESS / PARAM_INVALID
Status CalcShape(const std::vector<int64_t> &batch_shape, GeShape &data_shape);

/// @ingroup ge
/// @brief parse each data's own dynamic dims.
/// @param [in] std::vector<vector<int64_t>> &shapes: dynamic batch gears info.
/// @param [in] std::vector<std::pair<std::string, std::vector<int64_t>>> data_name_and_shape: eg:{{data:{1,1,-1,2}}}.
/// @param [out] std::map<std::string, std::vector<vector<int64_t>>> &data_to_dynamic_info: key:data_name. value:dynamic
/// dims.
/// @return SUCCESS / PARAM_INVALID
Status ParserDataToDynamicInfo(const std::vector<vector<int64_t>> &shapes,
                               std::vector<std::pair<std::string, std::vector<int64_t>>> &data_name_and_shape,
                               std::map<std::string, std::vector<vector<int64_t>>> &data_to_dynamic_info);

/// @ingroup ge
/// @brief Set mbatch_dynamic_type on node.
/// @param [in] const OpDescPtr &op_desc: Node for set attribute.
/// @return 0: SUCCESS / others: INTERNAL_ERROR
Status StampDynamicType(const OpDescPtr &op_desc);

/// @ingroup ge
/// @brief Check dynamic batch Shape.
/// @param [in] const std::vector<int64_t> &shape: data_shape to be checked.
/// @param [in] const std::string &data_name: cur data name.
/// @return 0: true/false
GE_FUNC_VISIBILITY bool CheckDynamicBatchShape(const std::vector<int64_t> &shape, const std::string &data_name);

/// @ingroup ge
/// @brief Check Dynamic image size shape.
/// @param [in] unordered_map<std::string, std::vector<int64_t>> &shape_map: map of data_name and data_shape.
/// @param [in] const std::string &data_name: cur data name.
/// @param [in] const std::string &input_format: cur data format.
/// @param [in]  const std::string &input_format: format of input.
/// @return 0: true/false
GE_FUNC_VISIBILITY bool CheckDynamicImageSizeShape(const std::vector<int64_t> &shape, const std::string &input_format);

/// @ingroup ge
/// @brief Parse mbatch_dynamic_shape
/// @param [in] const std::string &input_shapes: User input input_shape.
/// @param [out] std::vector<std::pair<std::string, std::vector<int64_t>>> &user_shape_map.
/// @return 0: SUCCESS / others: INTERNAL_ERROR
Status ParseInputShapes(const std::string &input_shapes,
                        std::vector<std::pair<std::string, std::vector<int64_t>>> &user_shape_map);

Status BuildSubgraphMuliDimsInput(const std::vector<std::pair<std::string, std::vector<int64_t>>> &user_shape_map,
                                  const DimsVector &dynamic_dims_vec,
                                  std::vector<std::string> &subgraph_multi_dims_input_shape,
                                  std::vector<std::string> &subgraph_multi_dims_input_dims);

Status ParseDynamicShapesAndDims(const std::string &input_shapes, const std::string &dynamic_dims,
                                 std::vector<std::pair<std::string, std::vector<int64_t>>> &user_shape_map,
                                 DimsVector &dynamic_dims_vec,
                                 std::vector<std::pair<std::string, std::vector<int64_t>>> &max_shape_range_map);

Status ParseDynamicDims(const std::string &dynamic_dims, DimsVector &dynamic_dims_vec,
                        std::vector<std::vector<int64_t>> &dynamic_dims_digit_vec,
                        const std::vector<std::pair<std::string, std::vector<int64_t>>> &user_shape_map);

Status ParseDynamicShapes(const std::string &input_shapes,
                          std::vector<std::pair<std::string, std::vector<int64_t>>> &user_shape_map);

Status ParseMaxShapeRange(const std::vector<std::pair<std::string, std::vector<int64_t>>> &user_shape_map,
                          const std::vector<std::vector<int64_t>> &dynamic_dims_digit_vec,
                          std::vector<std::pair<std::string, std::vector<int64_t>>> &max_shape_range_map);
std::vector<int32_t> GetDataNodesUnknownDimIndex(const ge::NodePtr &data_nodes);
}  // namespace multibatch
}  // namespace ge
#endif // GE_GRAPH_PREPROCESS_MULTI_BATCH_OPTIONS_H_
