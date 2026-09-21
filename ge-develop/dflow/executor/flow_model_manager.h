/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DFLOW_EXECUTOR_FLOW_MODEL_MANAGER_H
#define DFLOW_EXECUTOR_FLOW_MODEL_MANAGER_H

#include "ge_common/ge_api_types.h"
#include "graph/manager/graph_manager_utils.h"
#include "dflow/inc/data_flow/model/flow_model.h"
#include "dflow/executor/heterogeneous_model_executor.h"

namespace ge {
class FlowModelManager {
 public:
  static FlowModelManager &GetInstance();

  Status LoadFlowModel(uint32_t &model_id, const FlowModelPtr &flow_model);

  Status ExecuteFlowModel(uint32_t model_id, const std::vector<GeTensor> &inputs, std::vector<GeTensor> &outputs);
  FlowModelPtr GetFlowModelByModelId(uint32_t model_id);

  bool IsLoadedByFlowModel(uint32_t model_id);

  /// @ingroup domi_ome
  /// @brief unload model and free resources
  /// @param [in] model_id model id
  /// @return Status run result
  /// @author
  Status Unload(uint32_t model_id);

  std::shared_ptr<HeterogeneousModelExecutor> GetHeterogeneousModelExecutor(uint32_t model_id);

 private:
  /// @ingroup domi_ome
  /// @brief constructor
  FlowModelManager() = default;

  /// @ingroup domi_ome
  /// @brief destructor
  ~FlowModelManager() = default;

  void GenModelId(uint32_t &id);

  Status DoLoadFlowModel(uint32_t model_id, const FlowModelPtr &flow_model);
  Status StopAndUnloadModel(const std::shared_ptr<HeterogeneousModelExecutor> &executor,
                            uint32_t deployed_model_id) const;

  std::mutex map_mutex_;
  std::map<uint32_t, std::shared_ptr<HeterogeneousModelExecutor>> heterogeneous_model_map_;
  // start from 100001 to avoid same with ge
  std::atomic<uint32_t> max_model_id_ = 100001U;
};
}  // namespace ge
#endif  // DFLOW_EXECUTOR_FLOW_MODEL_MANAGER_H
