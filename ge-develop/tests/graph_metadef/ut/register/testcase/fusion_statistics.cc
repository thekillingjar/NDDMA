/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "graph/graph.h"
#include "graph/compute_graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/operator_factory_impl.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_builder_utils.h"
#include "register/graph_optimizer/fusion_common/pattern_fusion_base_pass.h"
#include "register/graph_optimizer/graph_fusion/fusion_pattern.h"
#include "register/graph_optimizer/fusion_common/fusion_statistic_recorder.h"

using namespace ge;
class UtestFusionStatistics : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};



TEST_F(UtestFusionStatistics, test_01) {
  auto &fs_instance = fe::FusionStatisticRecorder::Instance();
  fe::FusionInfo fusion_info(0, "", "test_pass");

  fs_instance.UpdateGraphFusionMatchTimes(fusion_info);

  fs_instance.UpdateGraphFusionEffectTimes(fusion_info);

  fs_instance.UpdateBufferFusionMatchTimes(fusion_info);

  string session_graph_id = "0_1";
  std::map<std::string, fe::FusionInfo> graph_fusion_info_map;
  std::map<std::string, fe::FusionInfo> buffer_fusion_info_map;
  fs_instance.GetAndClearFusionInfo(session_graph_id, graph_fusion_info_map,
                                    buffer_fusion_info_map);

  fs_instance.GetFusionInfo(session_graph_id, graph_fusion_info_map,
                            buffer_fusion_info_map);

  fs_instance.ClearFusionInfo(session_graph_id);

  std::vector<string> session_graph_id_vec = {session_graph_id};
  EXPECT_NO_THROW(fs_instance.GetAllSessionAndGraphIdList(session_graph_id_vec));
}

TEST_F(UtestFusionStatistics, test_02) {
  auto &fs_instance = fe::FusionStatisticRecorder::Instance();
  fe::FusionInfo fusion_info(0, "", "test_pass");
  fusion_info.SetEffectTimes(2);
  fusion_info.SetMatchTimes(2);
  fusion_info.AddEffectTimes(1);
  fusion_info.AddMatchTimes(1);
  fusion_info.GetEffectTimes();
  fusion_info.GetMatchTimes();
  fusion_info.GetGraphId();
  fusion_info.GetPassName();
  fusion_info.GetSessionId();
  fusion_info.SetRepoHitTimes(5);
  fusion_info.GetRepoHitTimes();

  fs_instance.UpdateGraphFusionMatchTimes(fusion_info);

  fs_instance.UpdateGraphFusionEffectTimes(fusion_info);

  fs_instance.UpdateBufferFusionMatchTimes(fusion_info);

  fs_instance.UpdateBufferFusionEffectTimes(fusion_info);
  string session_graph_id = "0_1";
  std::map<std::string, fe::FusionInfo> graph_fusion_info_map;
  std::map<std::string, fe::FusionInfo> buffer_fusion_info_map;
  fs_instance.GetAndClearFusionInfo(session_graph_id, graph_fusion_info_map,
                                    buffer_fusion_info_map);

  fs_instance.GetFusionInfo(session_graph_id, graph_fusion_info_map,
                            buffer_fusion_info_map);

  fs_instance.ClearFusionInfo(session_graph_id);

  std::vector<string> session_graph_id_vec = {session_graph_id};
  EXPECT_NO_THROW(fs_instance.GetAllSessionAndGraphIdList(session_graph_id_vec));
}




