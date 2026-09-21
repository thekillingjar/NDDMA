/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __FUSION_STUB_HPP__
#define __FUSION_STUB_HPP__

#include "graph_optimizer/ub_fusion/buffer_fusion.h"
#include "common/scope_allocator.h"
#include "graph/compute_graph.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "common/configuration.h"
#include "graph_optimizer/fusion_common/fusion_pass_name.h"

using namespace std;
using namespace fe;
using namespace ge;


static const std::string OPDESC_SRC_NAME = "opdesc_src_name";
static const std::string OPDESC_DST_NAME = "opdesc_dst_name";

typedef struct {
    string targetname;
    uint32_t index;
} desc_info;


void CreateModelGraph(ComputeGraphPtr model_graph, vector<OpDescPtr> &op_list, map<string, vector<desc_info>> &src_map,
                      map<string, vector<desc_info>> &dst_map);

OpDescPtr CreateOpDefUbFusion(string name, string type, vector<string> &srcname_list, vector<string> &dstname_list,
                              vector<GeTensorDesc> &inputdesc_list, vector<GeTensorDesc> &outputdesc_list);

void filltensordesc(GeTensorDesc &tensor_desc, uint32_t n, uint32_t c, uint32_t h, uint32_t w, uint32_t datatype,
                    uint32_t format);

std::vector<BufferFusionInfo> SortedBufferFusionFun();

#ifndef DAVINCI_LITE
void SetTvmType(OpDescPtr opdef);
#endif
//void AllocScopeId(OpDescPtr opdef, uint32_t scopeid);

void SetPattern(OpDescPtr opdef, string optype);

bool GetPattern(OpDescPtr opdef, string &optype);

#ifndef DAVINCI_LITE

void SetAICoreOp(OpDescPtr opdef);

#endif

void PrintGraph(ComputeGraphPtr graph);



#endif
