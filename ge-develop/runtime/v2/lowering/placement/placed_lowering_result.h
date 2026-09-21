/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_LOWERING_PLACEMENT_PLACED_LOWERING_RESULT_H_
#define AIR_CXX_RUNTIME_V2_LOWERING_PLACEMENT_PLACED_LOWERING_RESULT_H_
#include <utility>
#include <vector>
#include "graph/node.h"
#include "exe_graph/lowering/value_holder.h"
#include "exe_graph/lowering/frame_selector.h"
#include "exe_graph/runtime/tensor.h"
#include "register/node_converter_registry.h"
#include "common/table_driven.h"
namespace gert {
struct OutputLowerResult {
  OutputLowerResult() : has_init(false), order_holders(), shape(nullptr), address(nullptr) {}
  OutputLowerResult(const vector<bg::ValueHolderPtr> &in_order_holders, bg::ValueHolderPtr in_shape,
                    bg::DevMemValueHolderPtr in_address)
      : has_init(true), order_holders(in_order_holders),
        shape(std::move(in_shape)),
        address(std::move(in_address)) {}
  bool has_init;
  std::vector<bg::ValueHolderPtr> order_holders;
  bg::ValueHolderPtr shape;
  bg::DevMemValueHolderPtr address;
};

struct TargetAddrDesc {
  int32_t address_placement;
  int64_t logic_stream_id;
};

class PlacedLoweringResult {
 public:
  PlacedLoweringResult(const ge::NodePtr &node, LowerResult &&lower_result);
  const LowerResult *GetResult() const;
  const OutputLowerResult *GetOutputResult(LoweringGlobalData &global_data, int32_t output_index,
                                           TargetAddrDesc target_addr_desc, bool is_data_dependent);
  const OutputLowerResult *GetOutputTensorResult(LoweringGlobalData &global_data,
                                                 int32_t output_index,
                                                 TargetAddrDesc target_addr_desc);

 private:
  const OutputLowerResult *GetOrCreateNoDataDependentResult(LoweringGlobalData &global_data, int32_t output_index,
                                                            TargetAddrDesc &target_addr_desc);
  const OutputLowerResult *CreateNoDataDependentResult(LoweringGlobalData &global_data, int32_t output_index,
                                                       TargetAddrDesc &target_addr_desc);
  const OutputLowerResult *CreateDataDependentResult(int32_t output_index, TargetAddrDesc &target_addr_desc,
                                                     const OutputLowerResult &result,
                                                     const OutputLowerResult &host_result);
  const OutputLowerResult *CreateRawTensorResult(const OutputLowerResult &result,
                                                 int32_t output_index,
                                                 int32_t address_placement);
 private:
  ge::Node *node_;
  LowerResult result_;
  // 0 represents data dependency, 1 represents data independency,
  // 2 represents raw tensor.
  std::vector<TableDriven2<static_cast<size_t>(kTensorPlacementEnd) + 1UL, 3UL, OutputLowerResult>> outputs_results_;
};
}  // namespace gert

#endif  // AIR_CXX_RUNTIME_V2_LOWERING_PLACEMENT_PLACED_LOWERING_RESULT_H_
