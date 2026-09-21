/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_UTILS_COMMON_OP_STORE_ADAPTER_BASE_H_
#define FUSION_ENGINE_UTILS_COMMON_OP_STORE_ADAPTER_BASE_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "nlohmann/json.hpp"
#include "graph/node.h"
#include "graph/op_desc.h"
#include "graph/op_kernel_bin.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/node_utils.h"
#include "common/op_info_common.h"
#include "common/util/op_info_util.h"
#include "common/aicore_util_types.h"

namespace fe {
using ScopeNodeIdMap = std::map<int64_t, std::vector<ge::Node *>>;
using CompileResultMap = std::map<int64_t, std::vector<CompileResultInfo>>;

enum class CompileStrategy {
  COMPILE_STRATEGY_OP_SPEC,
  COMPILE_STRATEGY_KEEP_OPTUNE,
  COMPILE_STRATEGY_NO_TUNE,
  COMPILE_STRATEGY_ONLINE_FUZZ
};

struct CompileInfoParam {
  std::vector<ge::NodePtr> &buff_fus_compile_failed_nodes;
  std::vector<ge::NodePtr> buff_fus_to_del_nodes;
  std::vector<ge::NodePtr> l1_fusion_failed_nodes;
  ScopeNodeIdMap fusion_nodes_map;
  CompileResultMap compile_ret_map;
  CompileStrategy compile_strategy;
  bool is_fusion_check;
  explicit CompileInfoParam(std::vector<ge::NodePtr> &buff_fus_compile_failed_nodes_param)
      : buff_fus_compile_failed_nodes(buff_fus_compile_failed_nodes_param),
        compile_strategy(CompileStrategy::COMPILE_STRATEGY_OP_SPEC), is_fusion_check(false) {
  }
};

class OpStoreAdapterBase {
 public:
  virtual ~OpStoreAdapterBase() {}

  /*
   *  @ingroup fe
   *  @brief   initialize op adapter
   *  @return  SUCCESS or FAILED
   */
  virtual Status Initialize(const std::map<std::string, std::string> &options) = 0;

  /*
   *  @ingroup fe
   *  @brief   finalize op adapter
   *  @return  SUCCESS or FAILED
   */
  virtual Status Finalize() = 0;

  virtual Status CompileOp(ScopeNodeIdMap &fusion_nodes_map, map<int64_t, std::string> &json_path_map,
                            std::vector<ge::NodePtr> &buff_fus_compile_failed_nodes,
                            const std::vector<ge::NodePtr> &buff_fus_to_del_nodes) {
    (void)fusion_nodes_map;
    (void)json_path_map;
    (void)buff_fus_compile_failed_nodes;
    (void)buff_fus_to_del_nodes;
    return SUCCESS;
  }

  virtual Status CompileOp(CompileInfoParam &compile_info) {
    (void)compile_info;
    return SUCCESS;
  }
};
using OpStoreAdapterBasePtr = std::shared_ptr<OpStoreAdapterBase>;
}  // namespace fe
#endif  // FUSION_ENGINE_UTILS_COMMON_OP_STORE_ADAPTER_H_
