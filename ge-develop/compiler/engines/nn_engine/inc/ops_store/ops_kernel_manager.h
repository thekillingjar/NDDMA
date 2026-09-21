/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_INC_OPS_STORE_OPS_KERNEL_MANAGER_H_
#define FUSION_ENGINE_INC_OPS_STORE_OPS_KERNEL_MANAGER_H_

#include <string>
#include <map>
#include <mutex>

#include "register/graph_optimizer/graph_optimize_register_error_codes.h"
#include "ops_store/sub_op_info_store.h"

namespace fe {
class OpsKernelManager {
public:
  static OpsKernelManager &Instance(const std::string &engine_name);
  /*
   * initialize the ops kernel manager
   * param[in] the options of init
   * param[in] engineName
   * param[in] socVersion soc version from ge
   * return Status(SUCCESS/FAILED)
   */
  Status Initialize();

  /*
   * to release the source of fusion manager
   * return Status(SUCCESS/FAILED)
   */
  Status Finalize();

  void GetAllOpsKernelInfo(map<string, ge::OpInfo> &infos) const;

  SubOpInfoStorePtr GetSubOpsKernelByStoreName(const std::string &store_name);

  SubOpInfoStorePtr GetSubOpsKernelByImplType(const OpImplType &op_impl_type);

  OpKernelInfoPtr GetOpKernelInfoByOpType(const std::string &store_name, const std::string &op_type);

  OpKernelInfoPtr GetOpKernelInfoByOpType(const OpImplType &op_impl_type, const std::string &op_type);

  OpKernelInfoPtr GetOpKernelInfoByOpDesc(const ge::OpDescPtr &op_desc_ptr);

  OpKernelInfoPtr GetHighPrioOpKernelInfo(const std::string &op_type);

  Status AddSubOpsKernel(SubOpInfoStorePtr sub_op_info_store_ptr);

  bool FindOpKernelByType(const std::string &op_type, OpKernelInfoPtr &op_kernel_ptr);

  void UpdatePatternForAllKernel(const std::vector<std::pair<std::string, std::string>> &op_prebuild_patterns);

  OpKernelInfoPtr GetHighPrioOpKernelInfoPtr(const string &op_type);

private:
  explicit OpsKernelManager(const std::string &engine_name);
  ~OpsKernelManager();
  bool is_init_;
  std::string engine_name_;
  std::map<std::string, SubOpInfoStorePtr> sub_ops_kernel_map_;
  std::map<OpImplType, SubOpInfoStorePtr> sub_ops_store_map_;
  std::mutex ops_kernel_manager_lock_;
};
}  // namespace fe
#endif  // FUSION_ENGINE_INC_OPS_STORE_OPS_KERNEL_MANAGER_H_
