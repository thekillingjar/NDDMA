/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_INIT_GELIB_H_
#define GE_INIT_GELIB_H_
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "engines/manager/engine_manager/dnnengine_manager.h"
#include "engines/manager/opskernel_manager/ops_kernel_manager.h"
#include "graph/tuning_utils.h"
#include "graph/operator_factory.h"
#include "graph/ge_local_context.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/anchor_utils.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/ge_types.h"

namespace ge {
class GE_FUNC_VISIBILITY GELib {
 public:
  GELib() = default;
  ~GELib() = default;

  // get GELib singleton instance
  static std::shared_ptr<GELib> GetInstance();

  // GE Environment Initialize, return Status: SUCCESS,FAILED
  static Status Initialize(const std::map<std::string, std::string> &options);

  static std::string GetPath();

  // GE Environment Finalize, return Status: SUCCESS,FAILED
  Status Finalize();

  // get DNNEngineManager object
  DNNEngineManager &DNNEngineManagerObj() const {
    return DNNEngineManager::GetInstance();
  }

  // get OpsKernelManager object
  OpsKernelManager &OpsKernelManagerObj() const {
    return OpsKernelManager::GetInstance();
  }

  // get Initial flag
  bool InitFlag() const {
    return init_flag_;
  }

 private:
  GELib(const GELib &);
  const GELib &operator=(const GELib &);
  Status InnerInitialize(std::map<std::string, std::string> &options);
  Status SystemInitialize(const std::map<std::string, std::string> &options);
  Status SystemFinalize();
  Status SetRTSocVersion(std::map<std::string, std::string> &new_options) const;
  void RollbackInit();
  void SetDumpModelOptions(const std::map<std::string, std::string> &options);
  void SetOpDebugOptions(const std::map<std::string, std::string> &options);

  std::mutex status_mutex_;
  bool init_flag_ = false;
  bool is_train_mode_ = false;
  bool is_system_inited = false;
  int32_t device_id_;
};
}  // namespace ge

#endif  // GE_INIT_GELIB_H_
