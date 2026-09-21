/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_ADAPTER_DSA_ADAPTER_TBE_OP_STORE_ADAPTER_H_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_ADAPTER_DSA_ADAPTER_TBE_OP_STORE_ADAPTER_H_

#include <map>
#include <memory>
#include <string>
#include "adapter/adapter_itf/op_store_adapter.h"
#include "graph_optimizer/graph_optimize_register_error_codes.h"

namespace fe {
class DSAOpStoreAdapter : public OpStoreAdapter {
 public:
  /*
   *  @ingroup fe
   *  @brief   initial resources needed by TbeCompilerAdapter, such as dlopen so
   *  files and load function symbols etc.
   *  @return  SUCCESS or FAILED
   */
  Status Initialize(const std::map<std::string, std::string> &options) override;

  /*
   *  @ingroup fe
   *  @brief   finalize resources initialized in Initialize function,
   *           such as dclose so files etc.
   *  @return  SUCCESS or FAILED
   */
  Status Finalize() override;

 private:
  bool init_flag{false};
};
using DSAOpStoreAdapterPtr = std::shared_ptr<DSAOpStoreAdapter>;
}  // namespace fe

#endif  // AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_ADAPTER_DSA_ADAPTER_TBE_OP_STORE_ADAPTER_H_
