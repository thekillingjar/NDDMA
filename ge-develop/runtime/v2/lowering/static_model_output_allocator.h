/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_LOWERING_STATIC_MODEL_OUTPUT_ALLOCATOR_H_
#define AIR_CXX_RUNTIME_V2_LOWERING_STATIC_MODEL_OUTPUT_ALLOCATOR_H_

#include "common/ge_inner_attrs.h"
#include "common/model/ge_model.h"
#include "graph_builder/converter_checker.h"
#include "graph/load/model_manager/davinci_model.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_types.h"
#include "kernel/known_subgraph/davinci_model_kernel.h"

namespace gert {
enum class OutputReuseType { kReuseInput, kReuseOutput, kRefOutput, kRefVariable, kNoReuse, kEnd };

struct OutputReuseInfo {
  bool is_reuse;
  OutputReuseType reuse_type;
  int32_t reuse_index;
  kernel::MemoryBaseTypeOffset mem_base_type_offset;
  ge::ConstGeTensorDescPtr ge_tensor_desc_ptr;
  std::string var_name;
};

struct ParseParam {
  ParseParam(const ge::OpDesc *op_desc, std::vector<OutputReuseInfo> &output_reuse_infos,
             std::map<int64_t, int32_t> &offset_to_index_map) : op_desc(op_desc),
        output_reuse_infos(output_reuse_infos),
        offset_to_index_map(offset_to_index_map) {}
  const ge::OpDesc *op_desc;
  std::map<int64_t, int32_t> data_address_2_index{};
  std::map<int64_t, ge::NodePtr> var_address_2_nodes{};

  ge::NodePtr src_node{nullptr};
  int32_t input_index{0};
  int64_t input_address{0};
  ge::ConstGeTensorDescPtr ge_tensor_desc_ptr{nullptr};
  std::vector<OutputReuseInfo> &output_reuse_infos;
  std::map<int64_t, int32_t> &offset_to_index_map;
};

using ParseFunc = std::function<ge::Status(ParseParam &)>;

class StaticModelOutputAllocator {
 public:
  StaticModelOutputAllocator(const bg::ValueHolderPtr &davinci_model_holder,
                             const std::vector<bg::DevMemValueHolderPtr> &input_addrs,
                             const bg::ValueHolderPtr &update_workspaces_holder);
  StaticModelOutputAllocator(const bg::ValueHolderPtr &davinci_model_holder,
                             const std::vector<bg::DevMemValueHolderPtr> &input_addrs);
  ~StaticModelOutputAllocator();

  LowerResult AllocAllOutputs(const std::vector<OutputReuseInfo> &output_reuse_infos,
                              LoweringGlobalData &global_data) const;
  static ge::Status GenerateOutputsReuseInfos(const ge::ComputeGraphPtr &graph,
                                              std::vector<OutputReuseInfo> &output_reuse_infos);

 private:
  std::vector<bg::DevMemValueHolderPtr> GetRefOutputsAddress(
      const std::vector<kernel::MemoryBaseTypeOffset> &mem_base_types_offsets) const;

  std::vector<bg::DevMemValueHolderPtr> AllocAllOutputsForRefOutputType(
      const std::vector<OutputReuseInfo> &output_reuse_infos) const;

  std::vector<bg::DevMemValueHolderPtr> AllocAllOutputsForRefVariableType(
      const std::vector<OutputReuseInfo> &output_reuse_infos, LoweringGlobalData &global_data) const;

  static std::vector<bg::DevMemValueHolderPtr> AllocAllOutputsForNoReuse(
      const std::vector<OutputReuseInfo> &output_reuse_infos, LoweringGlobalData &global_data);

  static std::vector<bg::ValueHolderPtr> GetNoReuseOutputsSize(const std::vector<OutputReuseInfo> &output_reuse_infos);

  static ge::Status ParseModelOutputReuseInfo(ParseParam &param);

  static ge::Status ParseReuseInputs(ParseParam &param);

  static ge::Status ParseRefOutputs(ParseParam &param);

  static ge::Status ParseRefVariable(ParseParam &param);

  static ge::graphStatus ParseReuseOutputs(ParseParam &param);
 private:
  const bg::ValueHolderPtr davinci_model_holder_;
  std::vector<bg::DevMemValueHolderPtr> input_addrs_;
  const bg::ValueHolderPtr update_workspaces_holder_;
};
}  // namespace gert

#endif  // AIR_CXX_RUNTIME_V2_LOWERING_STATIC_MODEL_OUTPUT_ALLOCATOR_H_
