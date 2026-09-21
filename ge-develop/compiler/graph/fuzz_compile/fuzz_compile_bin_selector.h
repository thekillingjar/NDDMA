/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_FUZZ_COMPILE_BIN_SELECTOR_H_
#define GE_GRAPH_FUZZ_COMPILE_BIN_SELECTOR_H_

#include <mutex>
#include <unordered_map>

#include "common/tbe_handle_store/bin_register_utils.h"
#include "framework/common/ge_inner_error_codes.h"
#include "graph/bin_cache/node_bin_selector.h"
#include "graph/bin_cache/node_bin_selector_factory.h"
#include "graph/bin_cache/node_compile_cache_module.h"
#include "graph/bin_cache/op_binary_cache.h"
#include "engines/manager/opskernel_manager/ops_kernel_manager.h"

namespace ge {

namespace hybrid {
using NodeInOutShape = struct {
  std::unordered_map<size_t, ge::GeShape> in_ori_shapes;
  std::unordered_map<size_t, ge::GeShape> in_shapes;
  std::unordered_map<size_t, ge::GeShape> out_ori_shapes;
  std::unordered_map<size_t, ge::GeShape> out_shapes;
};

class FuzzCompileBinSelector : public NodeBinSelector {
 public:
  FuzzCompileBinSelector() = default;

  explicit FuzzCompileBinSelector(NodeCompileCacheModule *nccm) : nccm_(nccm){};

  Status Initialize() override;

  NodeCompileCacheItem *SelectBin(const NodePtr &node, const GEThreadLocalContext *ge_context,
                                  std::vector<domi::TaskDef> &task_defs) override;

 private:
  Status DoCompileOp(const NodePtr &node_ptr) const;
  Status DoRegisterBin(const OpDesc &op_desc, const domi::TaskDef &task_def, KernelLaunchBinType &bin_type,
                       void *&handle) const;

  Status DoGenerateTask(const NodePtr &node_ptr, std::vector<domi::TaskDef> &task_defs);

  NodeCompileCacheModule *nccm_ = nullptr;
  std::mutex gen_task_mutex_;
  OpsKernelInfoStorePtr aicore_kernel_store_;
};

REGISTER_BIN_SELECTOR(fuzz_compile::kOneNodeMultipleBinsMode, FuzzCompileBinSelector,
                      OpBinaryCache::GetInstance().GetNodeCcm());
}  // namespace hybrid
}  // namespace ge

#endif