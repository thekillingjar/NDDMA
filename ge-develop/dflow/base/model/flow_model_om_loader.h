/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BASE_PNE_MODEL_FLOW_MODEL_OM_LOADER_H_
#define BASE_PNE_MODEL_FLOW_MODEL_OM_LOADER_H_

#include "dflow/inc/data_flow/model/flow_model.h"
#include "framework/common/types.h"
#include "framework/common/helper/om_file_helper.h"

namespace ge {
class FlowModelOmLoader {
 public:
  static Status TransModelDataToComputeGraph(const ge::ModelData &model_data, ge::ComputeGraphPtr &root_graph);
  static Status LoadToFlowModel(const ge::ModelData &model_data, FlowModelPtr &flow_model,
                                const std::string &split_om_data_path = "");
  static Status LoadToFlowModelDesc(const ge::ModelData &model_data, const FlowModelPtr &flow_model);
  static Status RefreshModel(const FlowModelPtr &flow_model,
                                     const std::string &model_path,
                                     const uint64_t session_id,
                                     const uint32_t graph_id);
  static bool CheckFilePathValid(const std::string &base_dir, const std::string &check_dir);
 private:
  static Status CheckModelPartitions(const std::vector<ModelPartition> &model_partitions);
  static ComputeGraphPtr LoadRootGraph(const ModelPartition &model_def_partition);
  static Status LoadFlowModelPartition(const ModelPartition &flow_model_partition, const FlowModelPtr &flow_model,
                                       std::vector<string> &submodel_names);
  static Status LoadFlowSubmodelPartition(const std::vector<ModelPartition> &model_partitions,
                                          const std::string &split_om_data_base_dir,
                                          std::map<std::string, PneModelPtr> &flow_submodels);
};
}  // namespace ge
#endif  // BASE_PNE_MODEL_FLOW_MODEL_OM_LOADER_H_
