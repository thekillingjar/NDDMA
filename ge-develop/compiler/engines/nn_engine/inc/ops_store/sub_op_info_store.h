/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_INC_OPS_STORE_SUB_OP_INFO_STORE_H_
#define FUSION_ENGINE_INC_OPS_STORE_SUB_OP_INFO_STORE_H_

#include <string>
#include <vector>
#include <map>
#include "register/graph_optimizer/graph_optimize_register_error_codes.h"
#include "ops_store/op_kernel_info.h"
#include "ops_store/op_kernel_info_constructor.h"

namespace fe {
class SubOpInfoStore {
 public:
  explicit SubOpInfoStore(const FEOpsStoreInfo &ops_store_info);
  virtual ~SubOpInfoStore();

  /*
   * @ingroup fe
   * @brief : Initialize the SubOpInfoStore, load the op info from the specific.json file
   * @param[in] options: none
   * @return : SUCCESS/FAILED
   */
  Status Initialize(const std::string &engine_name);

  /*
   * @ingroup fe
   * @brief : finalize the SubOpInfoStore, clear all op info and op kernel info;
   * @param[in] None
   * @return : SUCCESS/FAILED
   */
  Status Finalize();

  const std::map<std::string, OpKernelInfoPtr>& GetAllOpKernels();

  OpKernelInfoPtr GetOpKernelByOpType(const std::string &op_type);

  Status GetOpContentByOpType(const std::string &op_type, OpContent &op_content) const;

  Status SetOpContent(const OpContent &op_content);

  void SplitOpStoreJson(const std::string& json_file_path, std::string& ops_path_name_prefix);

  Status LoadOpJsonFile(const std::string &json_file_path);

  Status ConstructOpKernelInfo(const std::string &engine_name);

  const std::string& GetOpsStoreName() const;

  const OpImplType& GetOpImplType() const;

  void UpdateStrToOpContent(OpContent &op_content, const std::string key1, const std::string key2,
                            const std::string &value) const;

 private:
  bool init_flag_;
  FEOpsStoreInfo ops_store_info_;
  /* std::string represents op_type */
  std::map<std::string, OpKernelInfoPtr> op_kernel_info_map_;
  std::map<std::string, OpContent> op_content_map_;

  /*
   * @brief : read the op information json file, and load the op
   * information to this sub store;
   */
  Status LoadOpInfo(const std::string &real_path);
};
using SubOpInfoStorePtr = std::shared_ptr<SubOpInfoStore>;
}  // namespace fe
#endif  // FUSION_ENGINE_INC_OPS_STORE_SUB_OP_INFO_STORE_H_
