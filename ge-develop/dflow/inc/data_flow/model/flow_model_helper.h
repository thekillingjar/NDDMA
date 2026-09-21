/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_FRAMEWORK_PNE_FLOW_MODEL_HELPER_H_
#define INC_FRAMEWORK_PNE_FLOW_MODEL_HELPER_H_
#include "dflow/inc/data_flow/model/flow_model.h"
#include "common/ge_common/ge_types.h"

namespace ge {
class GE_FUNC_VISIBILITY FlowModelHelper {
 public:
  static Status LoadToFlowModel(const std::string &model_path, FlowModelPtr &flow_model,
                                const std::string &split_om_data_base_dir = "");
  static Status UpdateSessionGraphId(const FlowModelPtr &flow_model, const std::string &session_graph_id);
  static Status LoadModelDataToFlowModel(const ModelData &model_data, FlowModelPtr &flow_model);
  static Status SaveToOmModel(const FlowModelPtr &flow_model, const std::string &output_file);
  static Status LoadFlowModelFromBuffData(const ModelBufferData &model_buffer_data,
                                          ge::FlowModelPtr &flow_model);
  static Status LoadFlowModelFromOmFile(const char_t *const model_path, ge::FlowModelPtr &flow_model);

  static PneModelPtr ToPneModel(const ModelData &model_data, const ComputeGraphPtr &compute_graph, const std::string &model_type = "");
  static Status EnsureWithModelRelation(const FlowModelPtr &flow_model);
 private:
  static Status TransModelDataToFlowModel(const ge::ModelData &model_data, ge::FlowModelPtr &flow_model);
};

}  // namespace ge

#endif  // INC_FRAMEWORK_PNE_FLOW_MODEL_HELPER_H_
