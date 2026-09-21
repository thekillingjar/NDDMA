/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gen_autofuse_api_tiling.h"
#include "api_tiling_gen_register.h"
namespace att {
ge::Status AutofuseApiTilingGenerator::Generate() {
  auto gen_func_impl = ApiTilingGenRegistry::Instance().GetApiTilingGenImplFunc(node_->GetType());
  auto gen_func_invoke = ApiTilingGenRegistry::Instance().GetApiTilingGenInvokeFunc(node_->GetType());
  auto gen_head_files = ApiTilingGenRegistry::Instance().GetApiTilingGenHeadFiles(node_->GetType());
  GE_ASSERT_NOTNULL(gen_func_invoke, "Get gen function invoke of %s failed.", node_->GetNamePtr());
  GE_ASSERT_NOTNULL(gen_func_impl, "Get gen function impl of %s failed.", node_->GetNamePtr());
  GE_ASSERT_NOTNULL(gen_head_files, "Get gen head files of %s failed.", node_->GetNamePtr());

  GE_ASSERT_SUCCESS(gen_func_impl(tiling_data_type_, graph_, node_, function_impl_, tiling_case_id_),
                    "Generate function impl code failed, graph[%s], node[%s] tiling data type[%s]",
                    graph_.GetName().c_str(), node_->GetName().c_str(), tiling_data_type_.c_str());
  GE_ASSERT_SUCCESS(gen_func_invoke(tiling_data_type_, graph_, node_, function_invoke_, tiling_case_id_),
                    "Generate function invoke code failed, graph[%s], node[%s] tiling data type[%s]",
                    graph_.GetName().c_str(), node_->GetName().c_str(), tiling_data_type_.c_str());
  GE_ASSERT_SUCCESS(gen_head_files(tiling_data_type_, graph_, node_, head_files_, tiling_case_id_),
                    "Generate head files code failed, graph[%s], node[%s] tiling data type[%s]",
                    graph_.GetName().c_str(), node_->GetName().c_str(), tiling_data_type_.c_str());
  return ge::SUCCESS;
}
}