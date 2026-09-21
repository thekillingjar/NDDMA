/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_MODEL_DATA_INFO_H_
#define INC_MODEL_DATA_INFO_H_

#include <string>
#include <map>
#include <memory>
#include "graph/graph.h"
#include "graph/ge_error_codes.h"
#include "graph/detail/attributes_holder.h"

namespace ge {
class ModelDataInfo : public AttrHolder {
 public:
  ModelDataInfo() = default;
  ~ModelDataInfo() override = default;
  ModelDataInfo(const ModelDataInfo &) = delete;
  ModelDataInfo &operator=(const ModelDataInfo &) = delete;
  size_t GetGraphMemorySize() const { return graph_memory_size_; }
  size_t GetVarMemorySize() const { return var_memory_size_; }
  void SetGraphMemorySize(const size_t graph_memory_size) { graph_memory_size_ = graph_memory_size; }
  void SetVarMemorySize(const size_t var_memory_size) { var_memory_size_ = var_memory_size; }

 protected:
  ProtoAttrMap &MutableAttrMap() override { return attrs_; }
  ConstProtoAttrMap &GetAttrMap() const override { return attrs_; }

 private:
  size_t graph_memory_size_;
  size_t var_memory_size_;
  AttrStore attrs_;
};

/**
 * @ingroup ge
 * @brief Simplified graph resource evaluation(No graph fusion optimization)
 *
 * @param options[IN] options used for build
 * @param compute_graph[IN/OUT] the graph ready to build
 * @param model_info[OUT] model info(memory size)
 * @retval GRAPH_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
graphStatus EvaluateGraphResource(const std::map<std::string, std::string> &options,
    ge::ComputeGraphPtr &compute_graph, ModelDataInfo &model);
};      // namespace ge
#endif  // INC_MODEL_DATA_INFO_H_
