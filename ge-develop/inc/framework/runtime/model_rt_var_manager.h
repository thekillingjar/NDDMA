/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_MODEL_RT_VAR_MANAGER_H
#define AIR_CXX_MODEL_RT_VAR_MANAGER_H
#include <map>

#include "rt_var_manager.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tensor_data.h"
#include "ge/ge_api_types.h"
#include "graph/compute_graph.h"
#include "graph/manager/graph_var_manager.h"
#include "common/model/ge_model.h"

namespace gert {
class ModelRtVarManager : public RtVarManager {
 public:
  static std::shared_ptr<ModelRtVarManager> Instance(const uint64_t session_id);
  explicit ModelRtVarManager(uint64_t session_id) : session_id_(session_id) {}
  ge::Status Init(const uint64_t device_id, const uint64_t logic_var_base, const int64_t total_var_size,
                  void* external_var_addr, uint64_t external_var_size);
  bool IsInited() const {
    return inited;
  }
  ge::Status RestoreDeviceVariables(const std::vector<ge::NodePtr> &variables, const uint32_t graph_id,
                                    const uint32_t device_id, const bool need_collect = true);
  ge::Status GetVarShapeAndMemory(const std::string &id, StorageShape &shape, TensorData &memory) const override;
  void Destroy() { name_to_var_info_.clear();};
 private:
  struct VarInfo {
    void *var_addr{nullptr};
    size_t var_size{0UL};
    StorageShape shape_info;
    gert::TensorPlacement placement{kOnDeviceHbm};
  };
  bool inited = false;
  uint64_t session_id_{0UL};
  std::map<std::string, VarInfo> name_to_var_info_;
};

class RtVarManagerPool {
 public:
  static RtVarManagerPool &Instance();

  std::shared_ptr<ModelRtVarManager> GetVarManager(const uint64_t session_id);

  void RemoveRtVarManager(const uint64_t session_id);

 private:
  RtVarManagerPool() = default;
  std::mutex var_manager_mutex_;
  std::map<uint64_t, std::shared_ptr<ModelRtVarManager>> session_id_to_var_manager_;
};

}  // namespace gert
#endif  // AIR_CXX_MODEL_RT_VAR_MANAGER_H
