/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_FUSION_COMMON_FUSION_OP_COMM_H_
#define FUSION_ENGINE_FUSION_COMMON_FUSION_OP_COMM_H_

#include <string>
#include <vector>
#include "common/fe_inner_attr_define.h"
#include "common/fe_log.h"
#include "graph/compute_graph.h"

namespace fe {
class FusionOpComm {
 public:
  FusionOpComm() {};
  virtual ~FusionOpComm() {};
  FusionOpComm(const FusionOpComm &in) = delete;
  FusionOpComm &operator=(const FusionOpComm &in) = delete;

  static ge::OpDescPtr CreateFusionOp(std::vector<ge::NodePtr> &fus_nodelist, const string &engine_name);

  static bool GetOutputInplaceAbilityAttrs(const vector<ge::NodePtr> &fus_nodelist,
                                           vector<vector<ge::ConstGeTensorDescPtr>> &output_place);
 private:
  static Status CopyFusionScopeAttr(const ge::OpDescPtr &src_op_desc, ge::OpDescPtr &dst_op_desc);

  static ge::OpDescPtr SetMultiKernelTBEFusionOp(const vector<ge::NodePtr> &fus_nodelist, ge::OpDescPtr &fus_opdef,
                                                 const string &engine_name, const string &pass_name);
  static ge::OpDescPtr SetTBEFusionOp(const vector<ge::NodePtr> &fus_nodelist, ge::OpDescPtr &fus_opdef,
                                      const string &engine_name, const string &pass_name);

  static Status AddTvmNode(ge::OpDescPtr op_desc);

  static void SetDataDumpAttrToFusionOp(const vector<ge::NodePtr> &fus_nodelist, ge::OpDescPtr &fus_opdef,
                                        const string& pass_name);

  static void SetOriginalNodesOpDescToFusionOp(const vector<ge::NodePtr> &fus_nodelist, const ge::OpDescPtr &fus_opdef);

  static void RecordOriginalNames(std::vector<ge::NodePtr> original_nodes, ge::OpDescPtr &op_desc);
};
}  // namespace fe
#endif  // FUSION_ENGINE_FUSION_COMMON_FUSION_OP_COMM_H_
