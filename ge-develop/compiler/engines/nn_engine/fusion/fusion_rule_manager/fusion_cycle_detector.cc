/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fusion_rule_manager/fusion_cycle_detector.h"
#include "common/fe_log.h"

namespace fe {
FusionCycleDetector::FusionCycleDetector() {}

FusionCycleDetector::~FusionCycleDetector() {}

std::vector<FusionPattern *> FusionCycleDetector::DefinePatterns() {
  std::vector<FusionPattern *> ret;
  return ret;
}

Status FusionCycleDetector::Fusion(ge::ComputeGraph &graph, Mapping &mapping, std::vector<ge::NodePtr> &new_nodes) {
  (void)graph;
  (void)mapping;
  (void)new_nodes;
  return SUCCESS;
}

Status FusionCycleDetector::Initialize(const ge::ComputeGraph &graph) {
  std::unique_ptr<ConnectionMatrix> connection_matrix;
  GetConnectionMatrix(connection_matrix);
  if (connection_matrix == nullptr) {
    FE_MAKE_SHARED(connection_matrix =
        std::unique_ptr<ConnectionMatrix>(new(std::nothrow) ConnectionMatrix(true)),
        return FAILED);
  }
  connection_matrix->Generate(graph);
  SetConnectionMatrix(connection_matrix);
  return SUCCESS;
}

void FusionCycleDetector::BackupConnectionMatrix() {
  std::unique_ptr<ConnectionMatrix> connection_matrix;
  GetConnectionMatrix(connection_matrix);
  if (connection_matrix == nullptr) {
    return;
  }
  connection_matrix->BackupBitMap();
  SetConnectionMatrix(connection_matrix);
}

void FusionCycleDetector::RestoreConnectionMatrix() {
  std::unique_ptr<ConnectionMatrix> connection_matrix;
  GetConnectionMatrix(connection_matrix);
  if (connection_matrix == nullptr) {
    return;
  }
  connection_matrix->RestoreBitMap();
  SetConnectionMatrix(connection_matrix);
}

Status FusionCycleDetector::UpdateConnectionMatrix(const ge::ComputeGraph &graph,
                                                   const vector<ge::NodePtr> &fusion_nodes) {
  if (fusion_nodes.size() <= 1) {
    FE_LOGD("Only one fusion node, do not need to update.");
    return SUCCESS;
  }
  std::unique_ptr<fe::ConnectionMatrix> connection_matrix;
  GetConnectionMatrix(connection_matrix);
  if (connection_matrix == nullptr) {
    return FAILED;
  }

  connection_matrix->Update(graph, fusion_nodes);
  SetConnectionMatrix(connection_matrix);
  return SUCCESS;
}

bool FusionCycleDetector::IsConnected(const ge::NodePtr &a, const ge::NodePtr &b) {
  std::unique_ptr<ConnectionMatrix> connection_matrix;

  GetConnectionMatrix(connection_matrix);
  if (connection_matrix == nullptr) {
    return false;
  }
  bool result = connection_matrix->IsConnected(a, b);
  SetConnectionMatrix(connection_matrix);
  return result;
}

bool FusionCycleDetector::IsDataFlowConnected(const ge::NodePtr &a, const ge::NodePtr &b) {
  std::unique_ptr<ConnectionMatrix> connection_matrix;

  GetConnectionMatrix(connection_matrix);
  if (connection_matrix == nullptr) {
    return false;
  }
  bool result = connection_matrix->IsDataConnected(a, b);
  SetConnectionMatrix(connection_matrix);
  return result;
}
}
