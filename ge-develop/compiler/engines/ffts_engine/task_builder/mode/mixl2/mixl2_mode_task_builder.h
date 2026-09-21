/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_UTILS_FFTS_PLUS_TASK_BUILDER_FFTS_PLUS_MIX_L2_MODE_H_
#define FUSION_ENGINE_UTILS_FFTS_PLUS_TASK_BUILDER_FFTS_PLUS_MIX_L2_MODE_H_
#include "task_builder/mode/thread_task_builder.h"

namespace ffts {
class Mixl2ModeTaskBuilder : public TheadTaskBuilder {
 public:
  Mixl2ModeTaskBuilder();
  ~Mixl2ModeTaskBuilder() override;

  Status Initialize() override;

  Status GenFftsPlusContextId(ge::ComputeGraph &sgt_graph, std::vector<ge::NodePtr> &sub_graph_nodes,
                              uint64_t &ready_context_num, uint64_t &total_context_number,
                              std::vector<ge::NodePtr> &memset_nodes) override;
  Status GenSubGraphTaskDef(std::vector<ge::NodePtr> &memset_nodes, std::vector<ge::NodePtr> &sub_graph_nodes,
                            domi::TaskDef &task_def) override;
};
}
#endif
