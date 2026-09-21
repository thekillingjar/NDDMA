/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_FUSION_BASE_PUB_H
#define OP_FUSION_BASE_PUB_H

#include "hccl/base.h"
#include "hccl/hccl_types.h"
#include "common/optimizer/graph_optimizer.h"
#include "common/optimizer/graph_optimizer_types.h"
#include "graph/compute_graph.h"
#include "hcom_log.h"
#include "op_hcom_comm.h"

namespace hccl {
// bcast 和 reduce使用
using FusionSection = std::vector<ge::NodePtr>;
using FusionInfos = std::map<std::string, FusionSection>;
const std::string HCOM_ATTR_NAME_FUSION = "fusion";
const std::string HCOM_ATTR_NAME_FUSION_ID = "fusion_id";
constexpr int64_t HCOM_ATTR_FUSION_MIN = 0;
constexpr int64_t HCOM_ATTR_FUSION_MAX = 2;
constexpr int64_t HCOM_ATTR_FUSION_NO_FUSION = 0;
constexpr int64_t HCOM_ATTR_FUSION_BY_SPLIT_STRATEGY = 1;
constexpr int64_t HCOM_ATTR_FUSION_BY_FUSION_ID = 2;
constexpr int64_t HCOM_ATTR_FUSION_ID_DEFAULT = -1;
constexpr int64_t HCOM_ATTR_FUSION_ID_MIN = 0;
constexpr int64_t HCOM_ATTR_FUSION_ID_MAX = 0x7FFFFFFF;
constexpr int64_t HCOM_ATTR_FUSION_ROOT_DEFAULT = 0;
constexpr int64_t HCOM_ATTR_FUSION_COMM_DEFAULT = 0;

using FusionOption = struct FusionOptionDef {
  std::string optype;
  std::string group;
  std::string dtype;
  std::string reduction;
  int64_t root;
  int64_t fusionComm;  // lint !e148
  int64_t fusionAttr;
  int64_t fusionId;
  FusionOptionDef()
      : optype(""),
        group(""),
        dtype(""),
        reduction(""),
        root(HCOM_ATTR_FUSION_ROOT_DEFAULT),
        fusionComm(HCOM_ATTR_FUSION_COMM_DEFAULT),
        fusionAttr(HCOM_ATTR_FUSION_NO_FUSION),
        fusionId(HCOM_ATTR_FUSION_ID_DEFAULT) {}
  FusionOptionDef(std::string group, int64_t id)
      : optype(""),
        group(group),
        dtype(""),
        reduction(""),
        root(HCOM_ATTR_FUSION_ROOT_DEFAULT),
        fusionComm(HCOM_ATTR_FUSION_COMM_DEFAULT),
        fusionAttr(HCOM_ATTR_FUSION_NO_FUSION),
        fusionId(id) {}
};

using FusionAnchorInfo_t = struct fusionAnchorInfo {
  std::vector<ge::OutDataAnchorPtr> peerOutDataAnchor;
  std::vector<ge::OutDataAnchorPtr> peerOutDataToInControl;
  std::vector<ge::OutControlAnchorPtr> peerOutControlAnchor;
  std::vector<std::pair<u32, ge::InDataAnchorPtr>> peerInDataAnchor;
  std::vector<std::pair<u32, ge::InControlAnchorPtr>> peerInControlFromOutData;
  std::vector<ge::InControlAnchorPtr> peerInControlAnchor;

  fusionAnchorInfo()
      : peerOutDataAnchor(0),
        peerOutDataToInControl(0),
        peerOutControlAnchor(0),
        peerInDataAnchor(0),
        peerInControlFromOutData(0),
        peerInControlAnchor(0) {}
};

class OpFusionBase {
 public:
  explicit OpFusionBase();
  virtual ~OpFusionBase();
  virtual HcclResult Run(ge::ComputeGraph &graph);

 protected:
  HcclResult RunFusionOps(ge::ComputeGraph &graph, std::vector<ge::NodePtr> &fusionOps, u32 segmentNum,
                          std::vector<u32> &segmentIndex);
  HcclResult RunFusionOpsReduce(ge::ComputeGraph &graph, std::vector<ge::NodePtr> &fusionOps);
  HcclResult AddNodesDependence(const std::vector<ge::NodePtr> &fusedNodes);
  virtual HcclResult RestoreOpsEdges(std::vector<ge::NodePtr> &fusedNodes,
                                     std::vector<FusionAnchorInfo_t> &anchorInfos);
  virtual HcclResult AddFusionNode(ge::ComputeGraph &graph, std::vector<ge::OpDescPtr> fusedOps,
                                   std::vector<ge::NodePtr> &fusedNodes);
  virtual HcclResult RemoveOpsEdges(ge::ComputeGraph &graph, std::vector<ge::NodePtr> &fusionOps,
                                    std::vector<FusionAnchorInfo_t> &anchorInfos, std::vector<ge::OpDescPtr> &fusedOps);
  virtual HcclResult RemoveOpsEdges(ge::ComputeGraph &graph, std::vector<ge::NodePtr> &fusionOps, u32 segmentNum,
                                    const std::vector<u32> &segmentIndex, std::vector<FusionAnchorInfo_t> &anchorInfos,
                                    std::vector<ge::OpDescPtr> &fusedOps);
  virtual HcclResult GetPeerOutDataToInData(std::unordered_set<uintptr_t> &anchorSet,
                                            std::vector<ge::OutDataAnchorPtr> &peerOutDataAnchorVec,
                                            ge::NodePtr &srcNodePtr, ge::OpDescPtr &dstOpDescPtr,
                                            std::vector<std::pair<int, int>> &duplication);
  virtual HcclResult GetPeerOutDataToInControl(std::unordered_set<uintptr_t> &anchorSet,
                                               vector<ge::OutDataAnchorPtr> &peerOutDataToInControlVec,
                                               ge::NodePtr &srcNodePtr);
  virtual HcclResult GetPeerOutControlToInControl(std::unordered_set<uintptr_t> &anchorSet,
                                                  vector<ge::OutControlAnchorPtr> &peerOutControlToInControlVec,
                                                  ge::NodePtr &srcNodePtr);
  virtual HcclResult GetPeerAnchorFromOutData(
      std::unordered_set<uintptr_t> &anchorSet,
      std::vector<std::pair<u32, ge::InDataAnchorPtr>> &peerInDataFromOutDataVec,
      std::vector<std::pair<u32, ge::InControlAnchorPtr>> &peerInControlFromOutDataVec, ge::NodePtr &srcNodePtr,
      ge::OpDescPtr &dstOpDescPtr, int &index, std::vector<std::pair<int, int>> &duplication);
  virtual HcclResult GetPeerInDataAnchorFromOutData(
      std::unordered_set<uintptr_t> &anchorSet,
      std::vector<std::pair<u32, ge::InDataAnchorPtr>> &peerInDataFromOutDataVec, ge::OutDataAnchorPtr outDataAnchor,
      ge::NodePtr &srcNodePtr, int index);
  virtual HcclResult GetPeerInControlAnchorFromOutData(
      std::unordered_set<uintptr_t> &anchorSet,
      std::vector<std::pair<u32, ge::InControlAnchorPtr>> &peerInControlFromOutDataVec,
      ge::OutDataAnchorPtr outDataAnchor, ge::NodePtr &srcNodePtr, int index);
  virtual HcclResult GetPeerInControlFromOutControl(std::unordered_set<uintptr_t> &anchorSet,
                                                    vector<ge::InControlAnchorPtr> &peerInControlFromOutControlVec,
                                                    ge::NodePtr &srcNodePtr);
  HcclResult GetCommFromOpDescPtr(const ge::NodePtr &nodePtr, FusionOption &fusionOption);
  HcclResult GetGroupFromOpDescPtr(const ge::NodePtr &nodePtr, std::string &group);
  HcclResult GetFusionhashFromGraph(ge::ComputeGraph &graph, std::string &fusionHash);

 private:
};
}  // namespace hccl

#endif