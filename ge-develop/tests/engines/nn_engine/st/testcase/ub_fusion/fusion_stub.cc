/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <stdio.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include "fusion_stub.hpp"
#include "graph/compute_graph.h"
#include "graph/op_desc.h"

using namespace std;
using namespace ge;
using namespace fe;

//using AttrMap = ::google::protobuf::Map<::std::string, ::fe::AttrDef>;
//using AttrPair = ::google::protobuf::MapPair<std::string, fe::AttrDef>;

static const std::string SCOPE_KEY = "fusion_scope";
static const std::string PATTERN_KEY = "_pattern";
static const int64_t INVALID_OFFSET = -1;

void GetSrcDstIndex(OpDescPtr src_op, OpDescPtr dst_op, map<string, vector<desc_info>> &src_map,
                    map<string, vector<desc_info>> &dst_map, uint32_t &src_index, uint32_t &dst_index) {
    map<string, vector<desc_info>>::iterator it;

    it = dst_map.find(src_op->GetName());
    vector<desc_info> &vec1 = it->second;
    for (uint32_t loop = 0; loop < vec1.size(); loop++) {
        if (vec1[loop].targetname == dst_op->GetName()) {
            src_index = vec1[loop].index;
        }
    }

    it = src_map.find(dst_op->GetName());
    vector<desc_info> &vec2 = it->second;
    for (uint32_t loop = 0; loop < vec2.size(); loop++) {
        if (vec2[loop].targetname == src_op->GetName()) {
            dst_index = vec2[loop].index;
        }
    }
}

void
CreateModelGraph(ge::ComputeGraphPtr model_graph, vector<ge::OpDescPtr> &op_list, map<string, vector<desc_info>> &src_map,
                 map<string, vector<desc_info>> &dst_map) {
    uint32_t src_index = 0;
    uint32_t dst_index = 0;

    for (auto opdef : op_list) {
        NodePtr node = model_graph->AddNode(opdef);
    }

    for (OpDescPtr opdef : op_list) {
        vector<string> dst_name_temp_list;
        ge::AttrUtils::GetListStr(opdef, OPDESC_DST_NAME, dst_name_temp_list);
        NodePtr node = model_graph->FindNode(opdef->GetName());

        for (OpDescPtr dst_opdef : op_list) {
            vector<string> src_name_temp_list;
            ge::AttrUtils::GetListStr(dst_opdef, OPDESC_SRC_NAME, src_name_temp_list);
            for (auto src_name_temp : src_name_temp_list) {
                if (src_name_temp == opdef->GetName()) {
                    //cout << "73 " <<  opdef->GetName() << endl;
                    src_index = 0;
                    dst_index = 0;
                    GetSrcDstIndex(opdef, dst_opdef, src_map, dst_map, src_index, dst_index);
                    NodePtr dst_node = model_graph->FindNode(dst_opdef->GetName());

                    ge::GraphUtils::AddEdge(node->GetOutDataAnchor(src_index), dst_node->GetInDataAnchor(dst_index));
                }
            }

        }

    }

    vector<string> dst_name_temp_list1;
    for (NodePtr node: model_graph->GetDirectNode()) {

        cout << "nodename = " << node->GetName() << endl;
        for (auto dstnode1: node->GetOutDataNodes()) {
            cout << "output node = " << dstnode1->GetName() << endl;
        }
        cout << "===========================================" << endl;
    }

    //5. fusion Graph����Topo����
    (void) model_graph->TopologicalSorting();

    //6. set input and output ddr addr
    uint32_t ddr_addr = 0;
    for (auto node : model_graph->GetDirectNode()) {
        OpDescPtr opdef = node->GetOpDesc();
        string node_type = opdef->GetType();
        int32_t input_size = opdef->GetInputsSize();
        int32_t output_size = opdef->GetOutputsSize();

        vector<int64_t> input_list;

        for (int32_t loop = 0; loop < input_size; loop++) {
            input_list.push_back(ddr_addr++);
        }

        opdef->SetInputOffset(input_list);

        vector<int64_t> output_list;

        for (int32_t loop = 0; loop < output_size; loop++) {
            output_list.push_back(ddr_addr++);
        }

        opdef->SetOutputOffset(input_list);
    }

    return;
}

//void *UBFusionTest(ComputeGraphPtr model_graph, ScopeAllocator *scope_allocator) {
//    std::shared_ptr<TEUBFusion> graph_builder(new TEUBFusion(model_graph, scope_allocator));
//    graph_builder->Fusion();
//}

ge::OpDescPtr CreateOpDefUbFusion(string name, string type, vector<string> &srcname_list, vector<string> &dstname_list,
                                  vector<ge::GeTensorDesc> &inputdesc_list, vector<ge::GeTensorDesc> &outputdesc_list) {
    ge::OpDescPtr opdef = std::make_shared<OpDesc>(name, type);

    uint32_t src_node_num = inputdesc_list.size();
    vector<bool> fusion_is_input_const_vector;
    for (uint32_t loop = 0; loop < src_node_num; loop++) {
        fusion_is_input_const_vector.push_back(false);
        opdef->AddInputDesc(inputdesc_list[loop]);
    }
    opdef->SetIsInputConst(fusion_is_input_const_vector);

    uint32_t dst_node_num = outputdesc_list.size();
    for (uint32_t loop = 0; loop < dst_node_num; loop++) {
        opdef->AddOutputDesc(outputdesc_list[loop]);
    }

    return opdef;
}


void filltensordesc(GeTensorDesc &tensor_desc, uint32_t n, uint32_t c, uint32_t h, uint32_t w, uint32_t datatype,
                    uint32_t format) {
    std::vector<int64_t> s_v;
    s_v.push_back(n);
    s_v.push_back(c);
    s_v.push_back(h);
    s_v.push_back(w);
    GeShape s(s_v);
    tensor_desc.SetShape(s);
    tensor_desc.SetFormat((Format) format);
    tensor_desc.SetDataType((DataType) datatype);
    return;
}

std::vector<BufferFusionInfo> SortedBufferFusionFun() {
  std::vector<BufferFusionInfo> sorted_buffer_fusion_vec;
  return sorted_buffer_fusion_vec;
}

#ifndef DAVINCI_LITE
void SetTvmType(ge::OpDescPtr opdef)
{
    ge::AttrUtils::SetInt(opdef, ge::ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM));
}
#endif

void SetPattern(ge::OpDescPtr opdef, string optype) {
    auto key_pattern = PATTERN_KEY;
    ge::AttrUtils::SetStr(opdef, key_pattern, optype);
}

bool GetPattern(ge::OpDescPtr opdef, string &optype) {
    auto key_pattern = PATTERN_KEY;
    if (ge::AttrUtils::GetStr(opdef, key_pattern, optype)) {
        return true;
    }
    return false;
}

void PrintGraph(ge::ComputeGraphPtr graph) {

    for (auto node : graph->GetDirectNode()) {
        for (auto out_anchor : node->GetAllOutDataAnchors()) {
            for (auto peer_in_anchor : out_anchor->GetPeerInDataAnchors()) {
                ge::NodePtr peer_node = peer_in_anchor->GetOwnerNode();
                cout << " src_node name = " << node->GetName() << " dst_node name = " << peer_node->GetName() << endl;
                cout << " src index = " << out_anchor->GetIdx() << " dst index = " << peer_in_anchor->GetIdx() << endl;
                cout << " src format = " << node->GetOpDesc()->GetOutputDesc(out_anchor->GetIdx()).GetFormat()
                     << " dst format = " << peer_node->GetOpDesc()->GetOutputDesc(peer_in_anchor->GetIdx()).GetFormat()
                     << endl;
            }
        }
    }
}

#ifndef DAVINCI_LITE

void SetAICoreOp(ge::OpDescPtr opdef) {
    ge::AttrUtils::SetStr(opdef, "tvm_magic", "RT_DEV_BINARY_MAGIC_ELF");
}

#endif







