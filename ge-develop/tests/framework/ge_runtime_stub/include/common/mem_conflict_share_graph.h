/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_MEM_CONFLICT_SHARE_GRAPH_H
#define AIR_CXX_MEM_CONFLICT_SHARE_GRAPH_H
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/utils/graph_utils.h"
#include "graph/compute_graph.h"
#include "ge_context.h"
#include "ge_local_context.h"
#include "ge_common/ge_api_types.h"
#include "ge_common/debug/ge_log.h"

namespace ge {
struct MemConflictShareGraph {
  static ComputeGraphPtr BuildUserInAndUserInGraph();
  static ComputeGraphPtr BuildUserInAndUserOutGraph();
  static ComputeGraphPtr BuildUserInAndUserOutGraphInSubGraph();
  static ComputeGraphPtr BuildUserInAndNormalInWithSubGraph();
  static ComputeGraphPtr BuildUserInAndNormalOutWithSubGraph();
  static ComputeGraphPtr BuildUserInAndNotSupportRefreshInGraph();
  static ComputeGraphPtr BuildUserInRefDataAndNotSupportRefreshInGraph();
  static ComputeGraphPtr BuildUserInRefDataAndNotSupportRefreshOutGraph();
  static ComputeGraphPtr BuildUserInAndNotSupportRefreshOutByRefGraph();
  static ComputeGraphPtr BuildUserInAndNotSupportRefreshOutByRefWithHcomGraph();
  static ComputeGraphPtr BuildUserInAndNotSupportRefreshOutByRefGraph2();
  static ComputeGraphPtr BuildUserInAndNotSupportRefreshOutByContinuousInGraph();
  static ComputeGraphPtr BuildUserInAndRtsSpecialInGraph();
  static ComputeGraphPtr BuildUserInRefDataAndRtsSpecialInGraph();
  static ComputeGraphPtr BuildUserInAndRtsSpecialOutGraph();
  static ComputeGraphPtr BuildUserInRefDataAndRtsSpecialOutGraph();
  static ComputeGraphPtr BuildUserInAndOtherConflictTypeGraph();
  static ComputeGraphPtr BuildUserInAndUserIO();
  static ComputeGraphPtr BuildUserInAndUnRefreshableInputAndConstInput();
  static ComputeGraphPtr BuildUserInConnectNetoutputGraph();
  static ComputeGraphPtr BuildUserInConnectContinuousInputGraph();
  static ComputeGraphPtr BuildUserInConnectContinuousOutputGraph();
  static ComputeGraphPtr BuildUserInConnectNoPaddingContinuousInputGraph();
  static ComputeGraphPtr BuildUserInConnectNoPaddingContinuousOutputGraph();
  static ComputeGraphPtr BuildUserInOutConnectContinuousInAndOutGraph();
  static ComputeGraphPtr BuildUserInOutConnectNoPaddingContinuousInAndOutGraph();
  static ComputeGraphPtr BuildUserInAndNoPaddingContinuousInByAssignOutput();

  static ComputeGraphPtr BuildUserOutAndUserOutGraph();
  static ComputeGraphPtr BuildUserOutAndUserOutGraphWithSubGraph();
  static ComputeGraphPtr BuildUserOutAndUserOutGraphInSubGraph();
  static ComputeGraphPtr BuildUserOutAndConstOutGraph();
  static ComputeGraphPtr BuildUserOutAndVariableGraph();
  static ComputeGraphPtr BuildUserOutAndRefdataGraph();
  static ComputeGraphPtr BuildUserOutAndConstOutGraphWithSubGraph();
  static ComputeGraphPtr BuildUserOutAndConstOutGraphInSubgraph();
  static ComputeGraphPtr BuildUserOutAndNotSupportRefreshOutWithSubgraphDataGraph();
  static ComputeGraphPtr BuildUserOutAndNotSupportRefreshInGraph();
  static ComputeGraphPtr BuildUserOutAndNotSupportRefreshInByRefGraph();
  static ComputeGraphPtr BuildUserOutAndNotSupportRefreshOutOneOutMultiRefGraph();
  static ComputeGraphPtr BuildUserOutAndNotSupportRefreshOutByRefGraph();
  static ComputeGraphPtr BuildUserOutAndRtsSpecialInGraph();
  static ComputeGraphPtr BuildUserOutAndRtsSpecialOutGraph();
  static ComputeGraphPtr BuildUserOutAndOtherConflictTypeGraph();
  static ComputeGraphPtr BuildUserOutAndContinuousInputGraph();
  static ComputeGraphPtr BuildUserOutAndNoPaddingContinuousInputGraph();
  static ComputeGraphPtr BuildUserOutAndNoPaddingAndContinuousInputGraph();
  static ComputeGraphPtr BuildUserOutAndContinuousOutputGraph();
  static ComputeGraphPtr BuildUserOutAndALotOfContinuousOutputGraph();
  static ComputeGraphPtr BuildUserOutAndNoPaddingContinuousOutputGraph();
  static ComputeGraphPtr BuildUserOutAndNoPaddingContinuousInputByReferenceGraph();
  static ComputeGraphPtr BuildUserOutAndContinuousInputByReferenceGraph();

  static ComputeGraphPtr BuildImmutableOutAndNotSupportedRefreshInGraph();
  static ComputeGraphPtr BuildImmutableOutAndNotSupportedRefreshOutByRefGraph();
  static ComputeGraphPtr BuildImmutableOutAndNotSupportedRefreshOutByContinuousInGraph();
  static ComputeGraphPtr BuildImmutableOutAndRtsSpecailInGraph();
  static ComputeGraphPtr BuildImmutableOutAndRtsSpecailInByAssignGraph();
  static ComputeGraphPtr BuildImmutableOutAndRtsSpecailOutGraph();
  static ComputeGraphPtr BuildImmutableOutAnRtsSpecailOutByAssignOutput();
  static ComputeGraphPtr BuildImmutableOutAnRtsSpecailOutContinuousInOutByAssignGraph();
  static ComputeGraphPtr BuildConstantConnectNetoutputGraph();
  static ComputeGraphPtr BuildConstantConnectNetoutputSubGraph();
  static ComputeGraphPtr BuildImmutableOutAndOtherTypeGraph();
  static ComputeGraphPtr BuildRefDataGraph();
  static ComputeGraphPtr BuildImmutableOutAndNoPaddingContinuousOutput();
  static ComputeGraphPtr BuildRefDataAndNoPaddingContinuousOutput();
  static ComputeGraphPtr BuildVarAndNoPaddingContinuousOutputWithMultiReference();
  static ComputeGraphPtr BuildImmutableOutAndContinuousInput();
  static ComputeGraphPtr BuildImmutableOutAndNoPaddingContinuousInput();
  static ComputeGraphPtr BuildImmutableOutAndContinuousOutput();
  static ComputeGraphPtr BuildImmutableOutAndContinuousByAssignOutput();
  static ComputeGraphPtr BuildImmutableOutAndNoPaddingContinuousInByAssignOutput();
  static ComputeGraphPtr BuildImmutableOutAndNoPaddingAndContinuousByRefOutput();

  static ComputeGraphPtr BuildNotSupportRefreshInAndRtsSpecialInGraph();
  static ComputeGraphPtr BuildNotSupportRefreshInAndRtsSpecialOutGraph();
  static ComputeGraphPtr BuildNotSupportRefreshInAndNormalOutGraph();
  static ComputeGraphPtr BuildPhysicalRefreshableInAndNormalOutGraph();
  static ComputeGraphPtr BuildNotSupportRefreshInAndNormalOutInKnowSubgraph();
  static ComputeGraphPtr BuildNotSupportRefreshInputAndOtherConflictTypeGraph();
  static ComputeGraphPtr BuildNotSupportRefreshInputAndContinuousInputGraph();
  static ComputeGraphPtr BuildNotSupportRefreshInputAndNoPaddingContinuousInputGraph();
  static ComputeGraphPtr BuildNotSupportRefreshInputAndContinuousOutputGraph();
  static ComputeGraphPtr BuildNotSupportRefreshInputAndNoPaddingContinuousOutputGraph();
  static ComputeGraphPtr BuildNotSupportRefreshInputAndRefDataGraph();

  static ComputeGraphPtr BuildNotSupportRefreshOutAndRtsSpecialInGraph();
  static ComputeGraphPtr BuildNotSupportRefreshOutAnchorIndex1Graph();
  static ComputeGraphPtr BuildNotSupportRefreshOutAndRtsSpecailOutGraph();
  static ComputeGraphPtr BuildNotSupportRefreshOutAndNormalOutGraph();
  static ComputeGraphPtr BuildNotSupportRefreshOutMultiReferenceAndNormalOutGraph();
  static ComputeGraphPtr BuildNotSupportRefreshOutAndRtsSpecailIOAndNormalOutGraph();
  static ComputeGraphPtr BuildNotSupportRefreshOutAndContinuousInputGraph();
  static ComputeGraphPtr BuildNotSupportRefreshOutAndNoPaddingContinuousInputGraph();
  static ComputeGraphPtr BuildNotSupportRefreshOutAndContinuousOutputGraph();
  static ComputeGraphPtr BuildNotSupportRefreshOutAndNoPaddingContinuousOutputGraph();
  static ComputeGraphPtr BuildNotSupportRefreshOutAndNotSupportRefreshOutGraph();
  static ComputeGraphPtr BuildNotSupportRefreshOutAndNotSupportRefreshOutByRefGraph();
  static ComputeGraphPtr BuildNotSupportRefreshOutAndNotSupportRefreshInGraph();

  static ComputeGraphPtr BuildContinuousInAndContinuousInGraph();
  static ComputeGraphPtr BuildContinuousInAndContinuousInByRefGraph();
  static ComputeGraphPtr BuildContinuousInAndContinuousInMixGraph();
  static ComputeGraphPtr BuildContinuousInAndNoPaddingContinuousInMixGraph();
  static ComputeGraphPtr BuildContinuousInAndContinuousOutGraph();
  static ComputeGraphPtr BuildContinuousInHeadSameWithContinuousOutTailGraph();
  static ComputeGraphPtr BuildContinuousInTailSameWithContinuousOutHeadGraph();
  static ComputeGraphPtr BuildContinuousInIsSubSetOfContinuousOutGraph();
  static ComputeGraphPtr BuildContinuousOutIsSubSetOfContinuousInGraph();
  static ComputeGraphPtr BuildContinuousInIsTheSameAsContinuousOutGraph();
  static ComputeGraphPtr BuildContinuousInReverseWithContinuousOutGraph();
  static ComputeGraphPtr BuildContinuousInReverseWithContinuousOut2Graph();
  static ComputeGraphPtr BuildContinuousInAndContinuousOutPartialSame1Graph();
  static ComputeGraphPtr BuildContinuousInAndContinuousOutPartialSame2Graph();
  static ComputeGraphPtr BuildContinuousInAndContinuousOutPartialSame3Graph();
  static ComputeGraphPtr BuildContinuousInAndContinuousOutPartialSame4Graph();
  static ComputeGraphPtr BuildContinuousInAndNoPaddingContinuousInGraph();
  static ComputeGraphPtr BuildContinuousInAndNoPaddingContinuousInByRefGraph();
  static ComputeGraphPtr BuildContinuousInAndNoPaddingContinuousOutGraph();
  static ComputeGraphPtr BuildContinuousInAndRtsSpecailInGraph();
  static ComputeGraphPtr BuildContinuousInAndRtsSpecailOutGraph();
  static ComputeGraphPtr BuildContinuousInAndRtsSpecailInByRefGraph();
  static ComputeGraphPtr BuildContinuousInAndRtsSpecailInOutGraph();
  static ComputeGraphPtr BuildContinuousInAndRtsSpecailInOutSameMemTypeGraph();

  static ComputeGraphPtr BuildContinuousOutAndContinuousOutByRefGraph();
  static ComputeGraphPtr BuildContinuousOutAndContinuousOutHcomByRefGraph();
  static ComputeGraphPtr BuildContinuousOutAndNoPaddingContinuousInGraph();
  static ComputeGraphPtr BuildContinuousOutAndNoPaddingContinuousOutByRefGraph();

  static ComputeGraphPtr BuildNoPaddingContinuousInAndNoPaddingContinuousInGraph();
  static ComputeGraphPtr BuildNoPaddingContinuousInCascadedGraph();
  static ComputeGraphPtr BuildNoPaddingContinuousInAndNoPaddingContinuousInPartialSameInputsGraph();
  static ComputeGraphPtr BuildNoPaddingContinuousInAndNoPaddingContinuousInPartialSameInputsConflictGraph();
  static ComputeGraphPtr BuildNoPaddingContinuousInAndNoPaddingContinuousInTailPartialSameInputsGraph();
  static ComputeGraphPtr BuildNoPaddingContinuousInAndNoPaddingContinuousInTailIntersectionGraph();
  static ComputeGraphPtr BuildNoPaddingContinuousInAndNoPaddingContinuousInCrossGraph();
  static ComputeGraphPtr BuildNoPaddingContinuousInAndNoPaddingContinuousInByRefGraph();
  static ComputeGraphPtr BuildNoPaddingContinuousInAndNoPaddingContinuousInMixGraph();
  static ComputeGraphPtr BuildNoPaddingContinuousInAndNoPaddingContinuousOutGraph();
  static ComputeGraphPtr BuildNoPaddingContinuousInAndNoPaddingContinuousOutByRefGraph();
  static ComputeGraphPtr BuildNoPaddingContinuousInAndNoPaddingContinuousOutConnectGraph();
  static ComputeGraphPtr BuildNoPaddingContinuousInAndRtsSpecailInByRefAndOutAnchorSuspendedGraph();
  static ComputeGraphPtr BuildNoPaddingContinuousInWithSameAnchorData();
  static ComputeGraphPtr BuildNoPaddingContinuousInWithSameAnchorVariable();

  static ComputeGraphPtr BuildNoPaddingContinuousOutAndNoPaddingContinuousOutGraph();
  static ComputeGraphPtr BuildNoPaddingContinuousOutAndNormalOutGraph();

  static ComputeGraphPtr BuildRtsSpecialInAndRtsSpecialInGraph();
  static ComputeGraphPtr BuildRtsSpecialInAndRtsSpecialInByRefGraph();
  static ComputeGraphPtr BuildRtsSpecialInAndRtsSpecialInSameTypeGraph();
  static ComputeGraphPtr BuildRtsSpecialInAndRtsSpecialOutGraph();
  static ComputeGraphPtr BuildRtsSpecialInAndRtsSpecialOutByRefGraph();
  static ComputeGraphPtr BuildRtsSpecialInAndRtsSpecialOutSameTypeGraph();
  static ComputeGraphPtr BuildRtsSpecialInAndNormalOutGraph();
  static ComputeGraphPtr BuildRtsSpecialInAndOtherTypeGraph();

  static ComputeGraphPtr BuildRtsSpecialOutAndRtsSpecialOutGraph();
  static ComputeGraphPtr BuildRtsSpecialOutAndRtsSpecialOutByRefGraph();

  static ComputeGraphPtr BuildIfGraph();
  static ComputeGraphPtr BuildIfSingleOutMultiRefToNetoutputSubGraph();
  static ComputeGraphPtr BuildIfSingleOutMultiRefToNetoutputSubGraphWithPartitionedCall();
  static ComputeGraphPtr BuildIfInKnowSubGraph();
  static ComputeGraphPtr BuildWhileDataToNetoutputGraph();
  static ComputeGraphPtr BuildWhileDataMulNetoutputGraph();
  static ComputeGraphPtr BuildWhileDataMulMulNetoutputGraph();
  static ComputeGraphPtr BuildWhileTwoDataToNetoutputExchangeGraph();
  static ComputeGraphPtr BuildWhileTwoDataToNetoutputNoExchangeGraph();
  static ComputeGraphPtr BuildWhileDataRefNetoutputGraph();
  static ComputeGraphPtr BuildWhileDataPartitionedCallNetoutputGraph();
  static ComputeGraphPtr BuildWhileInNodeConnectToMultiNodesGraph();

  static ComputeGraphPtr BuildContinuousOutGraph();
  static ComputeGraphPtr BuildNotContinuousOutGraph();
  static ComputeGraphPtr BuildWhileSplitGraph();

  static ComputeGraphPtr BuildUnknowShapeGraph();
  static ComputeGraphPtr BuildConnectToNoPaddingContinuousInputThroughtRefNodeGraph();
  static ComputeGraphPtr BuildConnectToContinuousInputThroughtRefNodeGraph();
  static ComputeGraphPtr BuildContinousOutConnectRefNodeGraph();
  static ComputeGraphPtr BuildNopaddingContinousOutConnectRefNodeGraph();

  static void SetContinuousOutput(ComputeGraphPtr &root_graph, const std::string &node_name);
  static void SetContinuousInput(ComputeGraphPtr &root_graph, const std::string &node_name);
  static void SetNoPaddingContinuousOutput(ComputeGraphPtr &root_graph, const std::string &node_name);
  static void SetNoPaddingContinuousInput(ComputeGraphPtr &root_graph, const std::string &node_name);
  static void SetOutReuseInput(ComputeGraphPtr &root_graph, const std::string &node_name);
  static void SetSizeForAllNodes(ComputeGraphPtr &graph);
  static void SetShapeForAllNodes(ComputeGraphPtr &graph, const std::vector<int64_t> &shape);
  static void SetShapeForNodesInputs(ComputeGraphPtr &graph, const std::vector<int64_t> &shape,
                                     const std::vector<std::string> &names);
  static void SetShapeForNodesOutputs(ComputeGraphPtr &graph, const std::vector<int64_t> &shape,
                                     const std::vector<std::string> &names);
  static void SetStreamForNodes(ComputeGraphPtr &graph, int64_t stream_id, const std::vector<std::string> &names);

  /*
   * 按照node_names中的顺序进行拓扑排序，得完全指定
   */
  static Status TopologicalSortingMock(const ComputeGraphPtr &graph, const std::vector<std::string> &node_names);
};
struct DebugLog {
  DebugLog () {
    dlog_setlevel(GE_MODULE_NAME, 0, 0);
  }
  ~DebugLog() {
    dlog_setlevel(GE_MODULE_NAME, 3, 0);
  }
};
struct InfoLog {
  InfoLog () {
    dlog_setlevel(GE_MODULE_NAME, 1, 0);
  }
  ~InfoLog() {
    dlog_setlevel(GE_MODULE_NAME, 3, 0);
  }
};

struct OptionSetter {
  explicit OptionSetter(const std::map<std::string, std::string> &options) {
    origin_options = ge::GetThreadLocalContext().GetAllGlobalOptions();
    auto new_options = origin_options;
    for (const auto &option : options) {
      new_options[option.first] = option.second;
    }
    ge::GetThreadLocalContext().SetGlobalOption(new_options);
  }
  explicit OptionSetter(std::initializer_list<std::pair<const std::string, std::string>> options)
    : OptionSetter(std::map<std::string, std::string>(options)) {
  }
  ~OptionSetter() {
    GetThreadLocalContext().SetGlobalOption(origin_options);
  }

  map<std::string, std::string> origin_options;
};
class FeatureMapRefreshOptionGuarder {
 public:
  FeatureMapRefreshOptionGuarder() {
    old_graph_options_ = GetThreadLocalContext().GetAllGraphOptions();
    std::map<std::string, std::string> new_graph_options = old_graph_options_;
    new_graph_options[OPTION_FEATURE_BASE_REFRESHABLE] = "1";
    GetThreadLocalContext().SetGraphOption(new_graph_options);
  }
  ~FeatureMapRefreshOptionGuarder() {
    GetThreadLocalContext().SetGraphOption(old_graph_options_);
  }

 private:
  std::map<std::string, std::string> old_graph_options_;
};

class StaticMemoryPolicy4Guarder {
 public:
  StaticMemoryPolicy4Guarder() {
    old_graph_options_ = GetThreadLocalContext().GetAllGraphOptions();
    std::map<std::string, std::string> new_graph_options = old_graph_options_;
    new_graph_options[STATIC_MEMORY_POLICY] = "4";
    GetThreadLocalContext().SetGraphOption(new_graph_options);
  }
  ~StaticMemoryPolicy4Guarder() {
    GetThreadLocalContext().SetGraphOption(old_graph_options_);
  }

 private:
  std::map<std::string, std::string> old_graph_options_;
};

}  // namespace ge
#endif  // AIR_CXX_MEM_CONFLICT_SHARE_GRAPH_H
