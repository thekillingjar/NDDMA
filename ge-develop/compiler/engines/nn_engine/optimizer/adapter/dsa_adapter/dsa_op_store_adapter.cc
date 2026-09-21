/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "adapter/dsa_adapter/dsa_op_store_adapter.h"
#include "common/configuration.h"
#include "common/fe_error_code.h"
#include "common/fe_log.h"
#include "common/fe_type_utils.h"
#include "common/fe_op_info_common.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/tuning_utils.h"
#include "graph/utils/tensor_utils.h"

namespace fe {
/*
 *  @ingroup fe
 *  @brief   initial resources needed by DSAOpStoreAdapter, such as dlopen so
 *  files and load function symbols etc.
 *  @return  SUCCESS or FAILED
 */
Status DSAOpStoreAdapter::Initialize(const std::map<std::string, std::string> &options) {
  (void)options;
  // return SUCCESS if graph optimizer has been initialized.
  if (init_flag) {
    FE_LOGW("DSAOpStoreAdapter has been initialized");
    return SUCCESS;
  }

  init_flag = true;
  FE_LOGI("Initialize dsa op store adapter success.");
  return SUCCESS;
}

/*
 *  @ingroup fe
 *  @brief   finalize resources initialized in Initialize function,
 *           such as dclose so files etc.
 *  @return  SUCCESS or FAILED
 */
Status DSAOpStoreAdapter::Finalize() {
  // return SUCCESS if graph optimizer has been initialized.
  if (!init_flag) {
    REPORT_FE_ERROR("[GraphOpt][Finalize] DSAOpStoreAdapter not allowed to finalize before initialized");
    return FAILED;
  }

  init_flag = false;
  FE_LOGI("Finalize dsa op store adapter success.");
  return SUCCESS;
}
}  // namespace fe
