/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <map>
#include <memory>
#include <stdio.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#define protected public
#define private public
#include "common/platform_utils.h"
#include "common/configuration.h"
#include "common/l2_stream_info.h"
#include "graph_optimizer/stream_graph_optimizer/l2_optimizer/l2_optimizer.h"
#include "graph_optimizer/fe_graph_optimizer.h"
#include "graph_optimizer/stream_graph_optimizer/l2_optimizer/l2_fusion_comm/l2_fusion_comm.h"
#include "graph_optimizer/stream_graph_optimizer/l2_optimizer/l2_fusion_parser/l2_fusion_parser.h"
#include "graph_optimizer/stream_graph_optimizer/l2_optimizer/l2_fusion_allocation/l2_fusion_allocation.h"
#undef private
#undef protected

#include "runtime/base.h"
#include "graph/ge_tensor.h"
#include "graph/compute_graph.h"
#include "graph/op_desc.h"
#include "graph/ge_attr_value.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/types.h"
#include "graph/op_kernel_bin.h"

using namespace fe;
using namespace ge;
static const std::string OPDESC_SRC_NAME = "opdesc_src_name";
static const std::string OPDESC_DST_NAME = "opdesc_dst_name";

typedef struct
{
    string targetname;
    uint32_t index;
} desc_info;

class L2FUSION_UT : public testing::Test {
protected:
    static void SetUpTestCase()
    {
    }
    static void TearDownTestCase()
    {
        std::cout << "L2 optimizer TearDown" << std::endl;
    }

    virtual void SetUp()
    {
    }
    virtual void TearDown()
    {

    }
};

void AllocScopeId(ge::OpDescPtr opdef, uint32_t scopeid)
{
    // ge::AttrUtils::SetInt(opdef, SCOPE_KEY, scopeid);
    // ref SCOPE_ID_ATTR in scope_allocator.cc
    ge::AttrUtils::SetInt(opdef, SCOPE_ID_ATTR, scopeid);
}

ge::OpDescPtr CreateOpDef(string name, string type, vector<string> &srcname_list, vector<string> &dstname_list,
                          vector<ge::GeTensorDesc> &inputdesc_list, vector<ge::GeTensorDesc> &outputdesc_list)
{
    ge::OpDescPtr opdef = std::make_shared<ge::OpDesc>(name, type);

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

// fill shape, datatype and format in tensor_desc
void FillTensorDesc(ge::GeTensorDesc &tensor_desc, int64_t n, int64_t c, int64_t h, int64_t w, ge::DataType d_type,
                    ge::Format d_format)
{
    vector<int64_t> dim;
    dim.push_back(n);
    dim.push_back(c);
    dim.push_back(h);
    dim.push_back(w);
    ge::GeShape shape(dim);
    tensor_desc.SetShape(shape);
    tensor_desc.SetDataType(d_type);
    tensor_desc.SetFormat(d_format);
    return;
}

void GetSrcDstIndexL2(ge::NodePtr &src_op, ge::NodePtr &dst_op, map<string, vector<desc_info>> &src_map,
                      map<string, vector<desc_info>> &dst_map, int &src_index, int &dst_index)
{
    map<string, vector<desc_info>>::iterator it;

    // find src_op's output op -> dst_op's output index
    it = dst_map.find(src_op->GetName());
    if (it != dst_map.end()) {
        vector<desc_info> &vec1 = it->second;
        for (int loop = 0; loop < vec1.size(); loop++) {
            if (vec1[loop].targetname == dst_op->GetName()) {
                src_index = loop;
            }
        }
    } else {
        std::cout << "not be found in dst_map!" << std::endl;
    }

    // find dst_op's input op -> src_op's input index
    it = src_map.find(dst_op->GetName());
    if (it != src_map.end()) {
        vector<desc_info> &vec2 = it->second;
        for (int loop = 0; loop < vec2.size(); loop++) {
            if (vec2[loop].targetname == src_op->GetName()) {
                dst_index = loop;
            }
        }
    } else {
        std::cout << "not be found in src_map!" << std::endl;
    }
}

void CreateL2Graph(ge::ComputeGraphPtr model_graph, vector<ge::OpDescPtr> &op_list,
                   map<string, vector<desc_info>> &src_map, map<string, vector<desc_info>> &dst_map)
{
    uint32_t src_index = 0;
    uint32_t dst_index = 0;
    bool flag = false;

    std::cout << "graph added nodes as below: " << std::endl;
    uint32_t node_num = 0;
    for (auto opdef : op_list) {
        ge::NodePtr node = model_graph->AddNode(opdef);
        cout << "To be added: node[" << node_num << "]name = " << node->GetName() << endl;
        node_num++;
    }

    for (ge::OpDescPtr opdef : op_list) {
        ge::NodePtr src_node = model_graph->FindNode(opdef->GetName());
        map<string, vector<desc_info>>::iterator it;
        it = dst_map.find(opdef->GetName());
        ge::NodePtr dst_node;
        if (it != dst_map.end()) {
            vector<desc_info> &vec = it->second;

            // TODO!
            for (int32_t i = 0; i < vec.size(); i++) {
                dst_node = model_graph->FindNode(vec[i].targetname);
                int src_index = 0;
                int dst_index = 0;
                GetSrcDstIndexL2(src_node, dst_node, src_map, dst_map, src_index, dst_index);
                ge::GraphUtils::AddEdge(src_node->GetOutDataAnchor(src_index), dst_node->GetInDataAnchor(dst_index));
            }

        } else {
            std::cout << "srcNode not found in the dst_map!" << std::endl;
        }
    }
    // Dump graph
    std::cout << "Created graph linking as below: " << std::endl;
    model_graph->Dump();

    /*
    vector<string> dst_name_temp_list1;
    node_num = 0;
    for (ge::NodePtr node: model_graph->GetDirectNode())
    {
        cout<<"creat graph: node[" << node_num << "]name = "<<node->GetName()<<endl;
        for(auto dstnode1: node->GetOutDataNodes())
        {
            cout<<"des node name = "<<dstnode1->GetName()<<endl;
        }
        node_num++;
    } */

    // 5. fusion graph sorting
    // (void)model_graph->TopologicalSorting();

    // 6. set input and output ddr addr
    uint32_t ddr_addr = 0;
    for (auto node : model_graph->GetDirectNode()) {
        ge::OpDescPtr opdef = node->GetOpDesc();
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

void CreateL2Graph2(ge::ComputeGraphPtr model_graph, vector<ge::OpDescPtr> &op_list,
                   map<string, vector<desc_info>> &src_map, map<string, vector<desc_info>> &dst_map)
{
    uint32_t src_index = 0;
    uint32_t dst_index = 0;
    bool flag = false;

    std::cout << "graph added nodes as below: " << std::endl;
    uint32_t node_num = 0;
    for (auto opdef : op_list) {
        ge::NodePtr node = model_graph->AddNode(opdef);
        cout << "To be added: node[" << node_num << "]name = " << node->GetName() << endl;
        node_num++;
    }

    for (ge::OpDescPtr opdef : op_list) {
        ge::NodePtr src_node = model_graph->FindNode(opdef->GetName());
        map<string, vector<desc_info>>::iterator it;
        it = dst_map.find(opdef->GetName());
        ge::NodePtr dst_node;
        if (it != dst_map.end()) {
            vector<desc_info> &vec = it->second;

            // TODO!
            for (int32_t i = 0; i < vec.size(); i++) {
                dst_node = model_graph->FindNode(vec[i].targetname);
                std::cout << "src_node " << opdef->GetName().c_str() << " dst_node " << vec[i].targetname << std::endl;
                // special process for one data and two quoting to one dst node
                if (i == 0) {
                    ge::GraphUtils::AddEdge(src_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(0));
                }
                else if (i == 1) {
                    ge::GraphUtils::AddEdge(src_node->GetOutDataAnchor(0), dst_node->GetInDataAnchor(1));
                }
            }

        } else {
            std::cout << "srcNode not found in the dst_map!" << std::endl;
        }
    }
    // Dump graph
    std::cout << "Created graph linking as below: " << std::endl;
    model_graph->Dump();

    /*
    vector<string> dst_name_temp_list1;
    node_num = 0;
    for (ge::NodePtr node: model_graph->GetDirectNode())
    {
        cout<<"creat graph: node[" << node_num << "]name = "<<node->GetName()<<endl;
        for(auto dstnode1: node->GetOutDataNodes())
        {
            cout<<"des node name = "<<dstnode1->GetName()<<endl;
        }
        node_num++;
    } */

    // 5. fusion graph sorting
    // (void)model_graph->TopologicalSorting();

    // 6. set input and output ddr addr
    uint32_t ddr_addr = 0;
    for (auto node : model_graph->GetDirectNode()) {
        ge::OpDescPtr opdef = node->GetOpDesc();
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

/************************
*conv1-->prelu1-->eltw-->prelu2-->conv3
*                   ^
*                   |
*                 conv2
*redu eltw relu fusion
*************************/
static uint32_t CreateGraph_conv1_conv2_redu_eltw_relu_conv3(ge::ComputeGraphPtr &model_graph)
{
    map<string, vector<desc_info>> src_map;
    map<string, vector<desc_info>> dst_map;
    map<string, vector<desc_info>>::iterator it;
    desc_info dscinfo;
    vector<desc_info> desctmp;

    ge::OpDescPtr opdef;
    vector<ge::OpDescPtr> op_list;
    vector<string> srcname_list;
    vector<string> dstname_list;
    ge::GeTensorDesc input_desc;
    ge::GeTensorDesc input_desc2;
    ge::GeTensorDesc output_desc;
    vector<ge::GeTensorDesc> inputdesc_list;
    vector<ge::GeTensorDesc> outputdesc_list;

    // 1.creat src map
    desctmp.clear();
    src_map.insert(pair<string, vector<desc_info >> ("conv1", desctmp));
    src_map.insert(pair<string, vector<desc_info >> ("conv2", desctmp));
    src_map.insert(pair<string, vector<desc_info >> ("conv3", desctmp));
    src_map.insert(pair<string, vector<desc_info >> ("prelu1", desctmp));
    src_map.insert(pair<string, vector<desc_info >> ("eltw", desctmp));
    src_map.insert(pair<string, vector<desc_info >> ("prelu2", desctmp));

    // 2.creat dst map
    dst_map.insert(pair<string, vector<desc_info >> ("conv1", desctmp));
    dst_map.insert(pair<string, vector<desc_info >> ("conv2", desctmp));
    dst_map.insert(pair<string, vector<desc_info >> ("conv3", desctmp));
    dst_map.insert(pair<string, vector<desc_info >> ("prelu1", desctmp));
    dst_map.insert(pair<string, vector<desc_info >> ("eltw", desctmp));
    dst_map.insert(pair<string, vector<desc_info >> ("prelu2", desctmp));

    // conv2
    FillTensorDesc(input_desc, 1, 16, 300, 300, ge::DT_FLOAT16, ge::FORMAT_NCHW);   // create input tensor info
    FillTensorDesc(output_desc, 1, 16, 150, 150, ge::DT_FLOAT16, ge::FORMAT_NCHW);  // creat output tensor info

    srcname_list.clear();
    dstname_list.clear();
    dstname_list.push_back("eltw");
    inputdesc_list.clear();
    inputdesc_list.push_back(input_desc);
    outputdesc_list.clear();
    outputdesc_list.push_back(output_desc);
    opdef = CreateOpDef("conv2", "Convolution", srcname_list, dstname_list, inputdesc_list, outputdesc_list);
    op_list.push_back(opdef);

    it = dst_map.find("conv2");
    vector<desc_info> &vec1 = it->second;
    dscinfo.targetname = "eltw";
    dscinfo.index = 0;
    vec1.push_back(dscinfo);

    // conv1
    FillTensorDesc(input_desc, 1, 16, 200, 200, ge::DT_FLOAT16, ge::FORMAT_NCHW);
    FillTensorDesc(output_desc, 1, 16, 160, 160, ge::DT_FLOAT16, ge::FORMAT_NCHW);

    srcname_list.clear();
    dstname_list.clear();
    dstname_list.push_back("prelu1");
    inputdesc_list.clear();
    inputdesc_list.push_back(input_desc);
    outputdesc_list.clear();
    outputdesc_list.push_back(output_desc);
    opdef = CreateOpDef("conv1", "Convolution", srcname_list, dstname_list, inputdesc_list, outputdesc_list);
    op_list.push_back(opdef);

    it = dst_map.find("conv1");
    vector<desc_info> &vec2 = it->second;
    dscinfo.targetname = "prelu1";
    dscinfo.index = 0;
    vec2.push_back(dscinfo);

    // redu
    FillTensorDesc(input_desc, 1, 16, 160, 160, ge::DT_FLOAT16, ge::FORMAT_NCHW);
    FillTensorDesc(output_desc, 1, 16, 100, 100, ge::DT_FLOAT16, ge::FORMAT_NCHW);

    srcname_list.clear();
    srcname_list.push_back("conv1");
    dstname_list.clear();
    dstname_list.push_back("eltw");
    inputdesc_list.clear();
    inputdesc_list.push_back(input_desc);
    outputdesc_list.clear();
    outputdesc_list.push_back(output_desc);
    opdef = CreateOpDef("prelu1", "PReLU", srcname_list, dstname_list, inputdesc_list, outputdesc_list);

    AllocScopeId(opdef, 100);
    op_list.push_back(opdef);

    it = dst_map.find("prelu1");
    vector<desc_info> &vec3 = it->second;
    dscinfo.targetname = "eltw";
    dscinfo.index = 0;
    vec3.push_back(dscinfo);

    it = src_map.find("prelu1");
    vector<desc_info> &vec4 = it->second;
    dscinfo.targetname = "conv1";
    dscinfo.index = 0;
    vec4.push_back(dscinfo);

    // eltw
    FillTensorDesc(input_desc, 1, 16, 150, 150, ge::DT_FLOAT16, ge::FORMAT_NCHW);
    FillTensorDesc(input_desc2, 1, 16, 100, 100, ge::DT_FLOAT16, ge::FORMAT_NCHW);
    FillTensorDesc(output_desc, 1, 16, 101, 101, ge::DT_FLOAT16, ge::FORMAT_NCHW);

    srcname_list.clear();
    srcname_list.push_back("conv2");
    srcname_list.push_back("prelu1");
    dstname_list.clear();
    dstname_list.push_back("prelu2");
    inputdesc_list.clear();
    inputdesc_list.push_back(input_desc);
    inputdesc_list.push_back(input_desc2);
    outputdesc_list.clear();
    outputdesc_list.push_back(output_desc);
    opdef = CreateOpDef("eltw", "Eltwise", srcname_list, dstname_list, inputdesc_list, outputdesc_list);

    AllocScopeId(opdef, 100);
    op_list.push_back(opdef);

    it = dst_map.find("eltw");
    vector<desc_info> &vec5 = it->second;
    dscinfo.targetname = "prelu2";
    dscinfo.index = 0;
    vec5.push_back(dscinfo);

    it = src_map.find("eltw");
    vector<desc_info> &vec6 = it->second;
    dscinfo.targetname = "conv2";
    dscinfo.index = 0;
    vec6.push_back(dscinfo);
    dscinfo.targetname = "prelu1";
    dscinfo.index = 1;
    vec6.push_back(dscinfo);

    // prelu
    FillTensorDesc(input_desc, 1, 16, 101, 101, ge::DT_FLOAT16, ge::FORMAT_NCHW);
    FillTensorDesc(output_desc, 1, 16, 55, 55, ge::DT_FLOAT16, ge::FORMAT_NCHW);

    srcname_list.clear();
    srcname_list.push_back("eltw");
    dstname_list.clear();
    dstname_list.push_back("conv3");
    inputdesc_list.clear();
    inputdesc_list.push_back(input_desc);
    outputdesc_list.clear();
    outputdesc_list.push_back(output_desc);
    opdef = CreateOpDef("prelu2", "PReLU", srcname_list, dstname_list, inputdesc_list, outputdesc_list);

    AllocScopeId(opdef, 100);
    op_list.push_back(opdef);

    it = dst_map.find("prelu2");
    vector<desc_info> &vec7 = it->second;
    dscinfo.targetname = "conv3";
    dscinfo.index = 0;
    vec7.push_back(dscinfo);

    it = src_map.find("prelu2");
    vector<desc_info> &vec8 = it->second;
    dscinfo.targetname = "eltw";
    dscinfo.index = 0;
    vec8.push_back(dscinfo);

    // conv3
    FillTensorDesc(input_desc, 1, 16, 55, 55, ge::DT_FLOAT16, ge::FORMAT_NCHW);
    FillTensorDesc(output_desc, 1, 16, 50, 50, ge::DT_FLOAT16, ge::FORMAT_NCHW);

    srcname_list.clear();
    srcname_list.push_back("prelu2");
    dstname_list.clear();
    inputdesc_list.clear();
    inputdesc_list.push_back(input_desc);
    outputdesc_list.clear();
    outputdesc_list.push_back(output_desc);
    opdef = CreateOpDef("conv3", "Convolution", srcname_list, dstname_list, inputdesc_list, outputdesc_list);
    op_list.push_back(opdef);

    it = src_map.find("conv3");
    vector<desc_info> &vec9 = it->second;
    dscinfo.targetname = "prelu2";
    dscinfo.index = 0;
    vec9.push_back(dscinfo);

    // invoke creat graph
    CreateL2Graph(model_graph, op_list, src_map, dst_map);
    return fe::SUCCESS;
}

TEST_F(L2FUSION_UT, fusion_test_tefusion_conv1_conv2_relu_eltw_relu_conv3)
{
    ge::ComputeGraphPtr modelgraph = std::make_shared<ge::ComputeGraph>("test");
    EXPECT_EQ(CreateGraph_conv1_conv2_redu_eltw_relu_conv3(modelgraph), fe::SUCCESS);

    // GetL2DynamincInfo(modelgraph.get());
    Configuration &config_inst = Configuration::Instance(fe::VECTOR_CORE_NAME);
    map<string, string> options;
    string soc_version = "Ascend310";
    PlatformUtils::Instance().soc_version_ = soc_version;
    config_inst.Initialize(options);

    L2Optimizer l2_opt(fe::VECTOR_CORE_NAME);
    auto nodes = modelgraph->GetDirectNode();
    auto op_desc = nodes.at(0)->GetOpDesc();
    EXPECT_NE(op_desc, nullptr);
    int64_t streamId = op_desc->GetStreamId();
    EXPECT_EQ(fe::SUCCESS, l2_opt.GetL2DataAlloc((*modelgraph), 0, streamId));
}

/************************
*conv1-->prelu1-->eltw-->prelu2-->conv3
*                   ^
*                   |
*                 conv2
*************************/
static uint32_t CreateGraph_conv1_conv2_redu_eltw_relu_conv3_small(ge::ComputeGraphPtr &model_graph)
{
    map<string, vector<desc_info>> src_map;
    map<string, vector<desc_info>> dst_map;
    map<string, vector<desc_info>>::iterator it;
    desc_info dscinfo;
    vector<desc_info> desctmp;

    ge::OpDescPtr opdef;
    vector<ge::OpDescPtr> op_list;
    vector<string> srcname_list;
    vector<string> dstname_list;
    ge::GeTensorDesc input_desc;
    ge::GeTensorDesc input_desc2;
    ge::GeTensorDesc output_desc;
    vector<ge::GeTensorDesc> inputdesc_list;
    vector<ge::GeTensorDesc> outputdesc_list;

    desctmp.clear();
    src_map.insert(pair<string, vector<desc_info >> ("conv1", desctmp));
    src_map.insert(pair<string, vector<desc_info >> ("conv2", desctmp));
    src_map.insert(pair<string, vector<desc_info >> ("conv3", desctmp));
    src_map.insert(pair<string, vector<desc_info >> ("prelu1", desctmp));
    src_map.insert(pair<string, vector<desc_info >> ("eltw", desctmp));
    src_map.insert(pair<string, vector<desc_info >> ("prelu2", desctmp));

    dst_map.insert(pair<string, vector<desc_info >> ("conv1", desctmp));
    dst_map.insert(pair<string, vector<desc_info >> ("conv2", desctmp));
    dst_map.insert(pair<string, vector<desc_info >> ("conv3", desctmp));
    dst_map.insert(pair<string, vector<desc_info >> ("prelu1", desctmp));
    dst_map.insert(pair<string, vector<desc_info >> ("eltw", desctmp));
    dst_map.insert(pair<string, vector<desc_info >> ("prelu2", desctmp));

    // conv2
    FillTensorDesc(input_desc, 1, 16, 32, 32, ge::DT_FLOAT16, ge::FORMAT_NCHW);
    FillTensorDesc(output_desc, 1, 16, 32, 32, ge::DT_FLOAT16, ge::FORMAT_NCHW);

    srcname_list.clear();
    dstname_list.clear();
    dstname_list.push_back("eltw");
    inputdesc_list.clear();
    inputdesc_list.push_back(input_desc);
    outputdesc_list.clear();
    outputdesc_list.push_back(output_desc);
    opdef = CreateOpDef("conv2", "Convolution", srcname_list, dstname_list, inputdesc_list, outputdesc_list);
    op_list.push_back(opdef);

    it = dst_map.find("conv2");
    vector<desc_info> &vec1 = it->second;
    dscinfo.targetname = "eltw";
    dscinfo.index = 0;
    vec1.push_back(dscinfo);

    // conv1
    FillTensorDesc(input_desc, 1, 16, 32, 32, ge::DT_FLOAT16, ge::FORMAT_NCHW);
    FillTensorDesc(output_desc, 1, 16, 32, 32, ge::DT_FLOAT16, ge::FORMAT_NCHW);

    srcname_list.clear();
    dstname_list.clear();
    dstname_list.push_back("prelu1");
    inputdesc_list.clear();
    inputdesc_list.push_back(input_desc);
    outputdesc_list.clear();
    outputdesc_list.push_back(output_desc);
    opdef = CreateOpDef("conv1", "Convolution", srcname_list, dstname_list, inputdesc_list, outputdesc_list);
    op_list.push_back(opdef);

    it = dst_map.find("conv1");
    vector<desc_info> &vec2 = it->second;
    dscinfo.targetname = "prelu1";
    dscinfo.index = 0;
    vec2.push_back(dscinfo);

    // redu
    FillTensorDesc(input_desc, 1, 16, 32, 32, ge::DT_FLOAT16, ge::FORMAT_NCHW);
    FillTensorDesc(output_desc, 1, 16, 32, 32, ge::DT_FLOAT16, ge::FORMAT_NCHW);

    srcname_list.clear();
    srcname_list.push_back("conv1");
    dstname_list.clear();
    dstname_list.push_back("eltw");
    inputdesc_list.clear();
    inputdesc_list.push_back(input_desc);
    outputdesc_list.clear();
    outputdesc_list.push_back(output_desc);
    opdef = CreateOpDef("prelu1", "PReLU", srcname_list, dstname_list, inputdesc_list, outputdesc_list);

    AllocScopeId(opdef, 100);
    op_list.push_back(opdef);

    it = dst_map.find("prelu1");
    vector<desc_info> &vec3 = it->second;
    dscinfo.targetname = "eltw";
    dscinfo.index = 0;
    vec3.push_back(dscinfo);

    it = src_map.find("prelu1");
    vector<desc_info> &vec4 = it->second;
    dscinfo.targetname = "conv1";
    dscinfo.index = 0;
    vec4.push_back(dscinfo);

    // eltw
    FillTensorDesc(input_desc, 1, 16, 32, 32, ge::DT_FLOAT16, ge::FORMAT_NCHW);
    FillTensorDesc(input_desc2, 1, 16, 32, 32, ge::DT_FLOAT16, ge::FORMAT_NCHW);
    FillTensorDesc(output_desc, 1, 16, 32, 32, ge::DT_FLOAT16, ge::FORMAT_NCHW);

    srcname_list.clear();
    srcname_list.push_back("conv2");
    srcname_list.push_back("prelu1");
    dstname_list.clear();
    dstname_list.push_back("prelu2");
    inputdesc_list.clear();
    inputdesc_list.push_back(input_desc);
    inputdesc_list.push_back(input_desc2);
    outputdesc_list.clear();
    outputdesc_list.push_back(output_desc);
    opdef = CreateOpDef("eltw", "Eltwise", srcname_list, dstname_list, inputdesc_list, outputdesc_list);

    AllocScopeId(opdef, 100);
    op_list.push_back(opdef);

    it = dst_map.find("eltw");
    vector<desc_info> &vec5 = it->second;
    dscinfo.targetname = "prelu2";
    dscinfo.index = 0;
    vec5.push_back(dscinfo);

    it = src_map.find("eltw");
    vector<desc_info> &vec6 = it->second;
    dscinfo.targetname = "conv2";
    dscinfo.index = 0;
    vec6.push_back(dscinfo);
    dscinfo.targetname = "prelu1";
    dscinfo.index = 1;
    vec6.push_back(dscinfo);

    // relu
    FillTensorDesc(input_desc, 1, 16, 32, 32, ge::DT_FLOAT16, ge::FORMAT_NCHW);
    FillTensorDesc(output_desc, 1, 16, 32, 32, ge::DT_FLOAT16, ge::FORMAT_NCHW);

    srcname_list.clear();
    srcname_list.push_back("eltw");
    dstname_list.clear();
    dstname_list.push_back("conv3");
    inputdesc_list.clear();
    inputdesc_list.push_back(input_desc);
    outputdesc_list.clear();
    outputdesc_list.push_back(output_desc);
    opdef = CreateOpDef("prelu2", "PReLU", srcname_list, dstname_list, inputdesc_list, outputdesc_list);

    AllocScopeId(opdef, 100);
    op_list.push_back(opdef);

    it = dst_map.find("prelu2");
    vector<desc_info> &vec7 = it->second;
    dscinfo.targetname = "conv3";
    dscinfo.index = 0;
    vec7.push_back(dscinfo);

    it = src_map.find("prelu2");
    vector<desc_info> &vec8 = it->second;
    dscinfo.targetname = "eltw";
    dscinfo.index = 0;
    vec8.push_back(dscinfo);

    // conv3
    FillTensorDesc(input_desc, 1, 16, 32, 32, ge::DT_FLOAT16, ge::FORMAT_NCHW);
    FillTensorDesc(output_desc, 1, 16, 32, 32, ge::DT_FLOAT16, ge::FORMAT_NCHW);

    srcname_list.clear();
    srcname_list.push_back("prelu2");
    dstname_list.clear();
    inputdesc_list.clear();
    inputdesc_list.push_back(input_desc);
    outputdesc_list.clear();
    outputdesc_list.push_back(output_desc);
    opdef = CreateOpDef("conv3", "Convolution", srcname_list, dstname_list, inputdesc_list, outputdesc_list);
    op_list.push_back(opdef);

    it = src_map.find("conv3");
    vector<desc_info> &vec9 = it->second;
    dscinfo.targetname = "prelu2";
    dscinfo.index = 0;
    vec9.push_back(dscinfo);

    CreateL2Graph(model_graph, op_list, src_map, dst_map);
    return fe::SUCCESS;
}

TEST_F(L2FUSION_UT, fusion_test_tefusion_conv1_conv2_relu_eltw_relu_conv3_small)
{
    ge::ComputeGraphPtr modelgraph = std::make_shared<ge::ComputeGraph>("test");
    EXPECT_EQ(CreateGraph_conv1_conv2_redu_eltw_relu_conv3_small(modelgraph), fe::SUCCESS);

    Configuration &config_inst = Configuration::Instance(fe::VECTOR_CORE_NAME);
    map<string, string> options;
    string soc_version = "Ascend310";
    PlatformUtils::Instance().soc_version_ = soc_version;
    config_inst.Initialize(options);

    L2Optimizer l2_opt(fe::VECTOR_CORE_NAME);
    auto nodes = modelgraph->GetDirectNode();
    auto op_desc = nodes.at(0)->GetOpDesc();
    EXPECT_NE(op_desc, nullptr);
    int64_t streamId = op_desc->GetStreamId();
    EXPECT_EQ(fe::SUCCESS, l2_opt.GetL2DataAlloc((*modelgraph), 0, streamId));
}

/************************
*conv1-->prelu1-->eltw-->prelu2-->conv3
*                   ^
*                   |
*                 conv2
*redu eltw relu fusion
*************************/
static uint32_t CreateGraph_conv1_conv2_redu_eltw_relu_conv3_splitBatch(ge::ComputeGraphPtr &model_graph)
{
    uint32_t batch_size = 32;
    map<string, vector<desc_info>> src_map;
    map<string, vector<desc_info>> dst_map;
    map<string, vector<desc_info>>::iterator it;
    desc_info dscinfo;
    vector<desc_info> desctmp;

    ge::OpDescPtr opdef;
    vector<ge::OpDescPtr> op_list;
    vector<string> srcname_list;
    vector<string> dstname_list;
    ge::GeTensorDesc input_desc;
    ge::GeTensorDesc input_desc2;
    ge::GeTensorDesc output_desc;
    vector<ge::GeTensorDesc> inputdesc_list;
    vector<ge::GeTensorDesc> outputdesc_list;

    desctmp.clear();
    src_map.insert(pair<string, vector<desc_info >> ("conv1", desctmp));
    src_map.insert(pair<string, vector<desc_info >> ("conv2", desctmp));
    src_map.insert(pair<string, vector<desc_info >> ("conv3", desctmp));
    src_map.insert(pair<string, vector<desc_info >> ("prelu1", desctmp));
    src_map.insert(pair<string, vector<desc_info >> ("eltw", desctmp));
    src_map.insert(pair<string, vector<desc_info >> ("prelu2", desctmp));

    dst_map.insert(pair<string, vector<desc_info >> ("conv1", desctmp));
    dst_map.insert(pair<string, vector<desc_info >> ("conv2", desctmp));
    dst_map.insert(pair<string, vector<desc_info >> ("conv3", desctmp));
    dst_map.insert(pair<string, vector<desc_info >> ("prelu1", desctmp));
    dst_map.insert(pair<string, vector<desc_info >> ("eltw", desctmp));
    dst_map.insert(pair<string, vector<desc_info >> ("prelu2", desctmp));

    FillTensorDesc(input_desc, batch_size, 64, 56, 56, ge::DT_FLOAT16, ge::FORMAT_NCHW);
    FillTensorDesc(output_desc, batch_size, 64, 56, 56, ge::DT_FLOAT16, ge::FORMAT_NCHW);

    srcname_list.clear();
    dstname_list.clear();
    dstname_list.push_back("eltw");
    inputdesc_list.clear();
    inputdesc_list.push_back(input_desc);
    outputdesc_list.clear();
    outputdesc_list.push_back(output_desc);
    opdef = CreateOpDef("conv2", "Convolution", srcname_list, dstname_list, inputdesc_list, outputdesc_list);
    op_list.push_back(opdef);

    it = dst_map.find("conv2");
    vector<desc_info> &vec1 = it->second;
    dscinfo.targetname = "eltw";
    dscinfo.index = 0;
    vec1.push_back(dscinfo);

    FillTensorDesc(input_desc, batch_size, 64, 56, 56, ge::DT_FLOAT16, ge::FORMAT_NCHW);
    FillTensorDesc(output_desc, batch_size, 64, 56, 56, ge::DT_FLOAT16, ge::FORMAT_NCHW);

    srcname_list.clear();
    dstname_list.clear();
    dstname_list.push_back("prelu1");
    inputdesc_list.clear();
    inputdesc_list.push_back(input_desc);
    outputdesc_list.clear();
    outputdesc_list.push_back(output_desc);
    opdef = CreateOpDef("conv1", "Convolution", srcname_list, dstname_list, inputdesc_list, outputdesc_list);
    op_list.push_back(opdef);

    it = dst_map.find("conv1");
    vector<desc_info> &vec2 = it->second;
    dscinfo.targetname = "prelu1";
    dscinfo.index = 0;
    vec2.push_back(dscinfo);

    FillTensorDesc(input_desc, batch_size, 64, 56, 56, ge::DT_FLOAT16, ge::FORMAT_NCHW);
    FillTensorDesc(output_desc, batch_size, 64, 56, 56, ge::DT_FLOAT16, ge::FORMAT_NCHW);

    srcname_list.clear();
    srcname_list.push_back("conv1");
    dstname_list.clear();
    dstname_list.push_back("eltw");
    inputdesc_list.clear();
    inputdesc_list.push_back(input_desc);
    outputdesc_list.clear();
    outputdesc_list.push_back(output_desc);
    opdef = CreateOpDef("prelu1", "PReLU", srcname_list, dstname_list, inputdesc_list, outputdesc_list);

    AllocScopeId(opdef, 100);
    op_list.push_back(opdef);

    it = dst_map.find("prelu1");
    vector<desc_info> &vec3 = it->second;
    dscinfo.targetname = "eltw";
    dscinfo.index = 0;
    vec3.push_back(dscinfo);

    it = src_map.find("prelu1");
    vector<desc_info> &vec4 = it->second;
    dscinfo.targetname = "conv1";
    dscinfo.index = 0;
    vec4.push_back(dscinfo);

    FillTensorDesc(input_desc, batch_size, 64, 56, 56, ge::DT_FLOAT16, ge::FORMAT_NCHW);
    FillTensorDesc(input_desc2, batch_size, 64, 56, 56, ge::DT_FLOAT16, ge::FORMAT_NCHW);
    FillTensorDesc(output_desc, batch_size, 16, 28, 28, ge::DT_FLOAT16, ge::FORMAT_NCHW);

    srcname_list.clear();
    srcname_list.push_back("conv2");
    srcname_list.push_back("prelu1");
    dstname_list.clear();
    dstname_list.push_back("prelu2");
    inputdesc_list.clear();
    inputdesc_list.push_back(input_desc);
    inputdesc_list.push_back(input_desc2);
    outputdesc_list.clear();
    outputdesc_list.push_back(output_desc);
    opdef = CreateOpDef("eltw", "Eltwise", srcname_list, dstname_list, inputdesc_list, outputdesc_list);

    AllocScopeId(opdef, 100);
    op_list.push_back(opdef);

    it = dst_map.find("eltw");
    vector<desc_info> &vec5 = it->second;
    dscinfo.targetname = "prelu2";
    dscinfo.index = 0;
    vec5.push_back(dscinfo);

    it = src_map.find("eltw");
    vector<desc_info> &vec6 = it->second;
    dscinfo.targetname = "conv2";
    dscinfo.index = 0;
    vec6.push_back(dscinfo);
    dscinfo.targetname = "prelu1";
    dscinfo.index = 1;
    vec6.push_back(dscinfo);

    FillTensorDesc(input_desc, batch_size, 16, 28, 28, ge::DT_FLOAT16, ge::FORMAT_NCHW);
    FillTensorDesc(output_desc, batch_size, 16, 28, 28, ge::DT_FLOAT16, ge::FORMAT_NCHW);

    srcname_list.clear();
    srcname_list.push_back("eltw");
    dstname_list.clear();
    dstname_list.push_back("conv3");
    inputdesc_list.clear();
    inputdesc_list.push_back(input_desc);
    outputdesc_list.clear();
    outputdesc_list.push_back(output_desc);
    opdef = CreateOpDef("prelu2", "PReLU", srcname_list, dstname_list, inputdesc_list, outputdesc_list);

    AllocScopeId(opdef, 100);
    op_list.push_back(opdef);

    it = dst_map.find("prelu2");
    vector<desc_info> &vec7 = it->second;
    dscinfo.targetname = "conv3";
    dscinfo.index = 0;
    vec7.push_back(dscinfo);

    it = src_map.find("prelu2");
    vector<desc_info> &vec8 = it->second;
    dscinfo.targetname = "eltw";
    dscinfo.index = 0;
    vec8.push_back(dscinfo);

    FillTensorDesc(input_desc, batch_size, 16, 28, 28, ge::DT_FLOAT16, ge::FORMAT_NCHW);
    FillTensorDesc(output_desc, batch_size, 16, 28, 28, ge::DT_FLOAT16, ge::FORMAT_NCHW);

    srcname_list.clear();
    srcname_list.push_back("prelu2");
    dstname_list.clear();
    inputdesc_list.clear();
    inputdesc_list.push_back(input_desc);
    outputdesc_list.clear();
    outputdesc_list.push_back(output_desc);
    opdef = CreateOpDef("conv3", "Convolution", srcname_list, dstname_list, inputdesc_list, outputdesc_list);
    op_list.push_back(opdef);

    it = src_map.find("conv3");
    vector<desc_info> &vec9 = it->second;
    dscinfo.targetname = "prelu2";
    dscinfo.index = 0;
    vec9.push_back(dscinfo);

    CreateL2Graph(model_graph, op_list, src_map, dst_map);
    return fe::SUCCESS;
}

TEST_F(L2FUSION_UT, test_conv1_conv2_relu_eltw_relu_conv3_split_batch)
{
    ge::ComputeGraphPtr modelgraph = std::make_shared<ge::ComputeGraph>("test");
    EXPECT_EQ(CreateGraph_conv1_conv2_redu_eltw_relu_conv3_splitBatch(modelgraph), fe::SUCCESS);

    Configuration &config_inst = Configuration::Instance(fe::VECTOR_CORE_NAME);
    map<string, string> options;
    string soc_version = "Ascend310";
    PlatformUtils::Instance().soc_version_ = soc_version;
    config_inst.Initialize(options);
    //configure.SetNeedL2Fusion(true);
    L2Optimizer l2_opt(fe::VECTOR_CORE_NAME);
    auto nodes = modelgraph->GetDirectNode();
    auto op_desc = nodes.at(0)->GetOpDesc();
    EXPECT_NE(op_desc, nullptr);
    int64_t streamId = op_desc->GetStreamId();
    EXPECT_EQ(fe::SUCCESS, l2_opt.GetL2DataAlloc((*modelgraph), 0, streamId));
}

/************************
*conv1-->prelu1-->eltw-->prelu2-->conv3
*                   ^
*                   |
*                 conv2
*redu eltw relu fusion
*************************/
static uint32_t CreateGraph_conv1_conv2_redu_eltw_relu_conv3_splitBatch_Big(ge::ComputeGraphPtr &model_graph)
{
    uint32_t batch_size = 64;
    map<string, vector<desc_info>> src_map;
    map<string, vector<desc_info>> dst_map;
    map<string, vector<desc_info>>::iterator it;
    desc_info dscinfo;
    vector<desc_info> desctmp;

    ge::OpDescPtr opdef;
    vector<ge::OpDescPtr> op_list;
    vector<string> srcname_list;
    vector<string> dstname_list;
    ge::GeTensorDesc input_desc;
    ge::GeTensorDesc input_desc2;
    ge::GeTensorDesc output_desc;
    vector<ge::GeTensorDesc> inputdesc_list;
    vector<ge::GeTensorDesc> outputdesc_list;

    desctmp.clear();
    src_map.insert(pair<string, vector<desc_info >> ("conv1", desctmp));
    src_map.insert(pair<string, vector<desc_info >> ("conv2", desctmp));
    src_map.insert(pair<string, vector<desc_info >> ("conv3", desctmp));
    src_map.insert(pair<string, vector<desc_info >> ("prelu1", desctmp));
    src_map.insert(pair<string, vector<desc_info >> ("eltw", desctmp));
    src_map.insert(pair<string, vector<desc_info >> ("prelu2", desctmp));

    dst_map.insert(pair<string, vector<desc_info >> ("conv1", desctmp));
    dst_map.insert(pair<string, vector<desc_info >> ("conv2", desctmp));
    dst_map.insert(pair<string, vector<desc_info >> ("conv3", desctmp));
    dst_map.insert(pair<string, vector<desc_info >> ("prelu1", desctmp));
    dst_map.insert(pair<string, vector<desc_info >> ("eltw", desctmp));
    dst_map.insert(pair<string, vector<desc_info >> ("prelu2", desctmp));

    FillTensorDesc(input_desc, batch_size, 64, 128, 128, ge::DT_FLOAT16, ge::FORMAT_NCHW);
    FillTensorDesc(output_desc, batch_size, 64, 128, 128, ge::DT_FLOAT16, ge::FORMAT_NCHW);

    srcname_list.clear();
    dstname_list.clear();
    dstname_list.push_back("eltw");
    inputdesc_list.clear();
    inputdesc_list.push_back(input_desc);
    outputdesc_list.clear();
    outputdesc_list.push_back(output_desc);
    opdef = CreateOpDef("conv2", "Convolution", srcname_list, dstname_list, inputdesc_list, outputdesc_list);
    op_list.push_back(opdef);

    it = dst_map.find("conv2");
    vector<desc_info> &vec1 = it->second;
    dscinfo.targetname = "eltw";
    dscinfo.index = 0;
    vec1.push_back(dscinfo);

    FillTensorDesc(input_desc, batch_size, 64, 128, 128, ge::DT_FLOAT16, ge::FORMAT_NCHW);
    FillTensorDesc(output_desc, batch_size, 64, 128, 128, ge::DT_FLOAT16, ge::FORMAT_NCHW);

    srcname_list.clear();
    dstname_list.clear();
    dstname_list.push_back("prelu1");
    inputdesc_list.clear();
    inputdesc_list.push_back(input_desc);
    outputdesc_list.clear();
    outputdesc_list.push_back(output_desc);
    opdef = CreateOpDef("conv1", "PlaceHolder", srcname_list, dstname_list, inputdesc_list, outputdesc_list);
    ge::AttrUtils::SetStr(opdef, "parentOpType", "SwitchN");
    op_list.push_back(opdef);

    it = dst_map.find("conv1");
    vector<desc_info> &vec2 = it->second;
    dscinfo.targetname = "prelu1";
    dscinfo.index = 0;
    vec2.push_back(dscinfo);

    FillTensorDesc(input_desc, batch_size, 64, 128, 128, ge::DT_FLOAT16, ge::FORMAT_NCHW);
    FillTensorDesc(output_desc, batch_size, 64, 128, 128, ge::DT_FLOAT16, ge::FORMAT_NCHW);

    srcname_list.clear();
    srcname_list.push_back("conv1");
    dstname_list.clear();
    dstname_list.push_back("eltw");
    inputdesc_list.clear();
    inputdesc_list.push_back(input_desc);
    outputdesc_list.clear();
    outputdesc_list.push_back(output_desc);
    opdef = CreateOpDef("prelu1", "PReLU", srcname_list, dstname_list, inputdesc_list, outputdesc_list);

    AllocScopeId(opdef, 100);
    op_list.push_back(opdef);

    it = dst_map.find("prelu1");
    vector<desc_info> &vec3 = it->second;
    dscinfo.targetname = "eltw";
    dscinfo.index = 0;
    vec3.push_back(dscinfo);

    it = src_map.find("prelu1");
    vector<desc_info> &vec4 = it->second;
    dscinfo.targetname = "conv1";
    dscinfo.index = 0;
    vec4.push_back(dscinfo);

    FillTensorDesc(input_desc, batch_size, 64, 128, 128, ge::DT_FLOAT16, ge::FORMAT_NCHW);
    FillTensorDesc(input_desc2, batch_size, 64, 128, 128, ge::DT_FLOAT16, ge::FORMAT_NCHW);
    FillTensorDesc(output_desc, batch_size, 64, 128, 128, ge::DT_FLOAT16, ge::FORMAT_NCHW);

    srcname_list.clear();
    srcname_list.push_back("conv2");
    srcname_list.push_back("prelu1");
    dstname_list.clear();
    dstname_list.push_back("prelu2");
    inputdesc_list.clear();
    inputdesc_list.push_back(input_desc);
    inputdesc_list.push_back(input_desc2);
    outputdesc_list.clear();
    outputdesc_list.push_back(output_desc);
    opdef = CreateOpDef("eltw", "Eltwise", srcname_list, dstname_list, inputdesc_list, outputdesc_list);

    AllocScopeId(opdef, 100);
    op_list.push_back(opdef);

    it = dst_map.find("eltw");
    vector<desc_info> &vec5 = it->second;
    dscinfo.targetname = "prelu2";
    dscinfo.index = 0;
    vec5.push_back(dscinfo);

    it = src_map.find("eltw");
    vector<desc_info> &vec6 = it->second;
    dscinfo.targetname = "conv2";
    dscinfo.index = 0;
    vec6.push_back(dscinfo);
    dscinfo.targetname = "prelu1";
    dscinfo.index = 1;
    vec6.push_back(dscinfo);

    FillTensorDesc(input_desc, batch_size, 64, 128, 128, ge::DT_FLOAT16, ge::FORMAT_NCHW);
    FillTensorDesc(output_desc, batch_size, 64, 128, 128, ge::DT_FLOAT16, ge::FORMAT_NCHW);

    srcname_list.clear();
    srcname_list.push_back("eltw");
    dstname_list.clear();
    dstname_list.push_back("conv3");
    inputdesc_list.clear();
    inputdesc_list.push_back(input_desc);
    outputdesc_list.clear();
    outputdesc_list.push_back(output_desc);
    opdef = CreateOpDef("prelu2", "PReLU", srcname_list, dstname_list, inputdesc_list, outputdesc_list);

    AllocScopeId(opdef, 100);
    op_list.push_back(opdef);

    it = dst_map.find("prelu2");
    vector<desc_info> &vec7 = it->second;
    dscinfo.targetname = "conv3";
    dscinfo.index = 0;
    vec7.push_back(dscinfo);

    it = src_map.find("prelu2");
    vector<desc_info> &vec8 = it->second;
    dscinfo.targetname = "eltw";
    dscinfo.index = 0;
    vec8.push_back(dscinfo);

    FillTensorDesc(input_desc, batch_size, 64, 128, 128, ge::DT_FLOAT16, ge::FORMAT_NCHW);
    FillTensorDesc(output_desc, batch_size, 64, 128, 128, ge::DT_FLOAT16, ge::FORMAT_NCHW);

    srcname_list.clear();
    srcname_list.push_back("prelu2");
    dstname_list.clear();
    inputdesc_list.clear();
    inputdesc_list.push_back(input_desc);
    outputdesc_list.clear();
    outputdesc_list.push_back(output_desc);
    opdef = CreateOpDef("conv3", "Convolution", srcname_list, dstname_list, inputdesc_list, outputdesc_list);
    op_list.push_back(opdef);

    it = src_map.find("conv3");
    vector<desc_info> &vec9 = it->second;
    dscinfo.targetname = "prelu2";
    dscinfo.index = 0;
    vec9.push_back(dscinfo);

    CreateL2Graph(model_graph, op_list, src_map, dst_map);
    return fe::SUCCESS;
}

TEST_F(L2FUSION_UT, test_conv1_conv2_relu_eltw_relu_conv3_split_batch_Big)
{
    ge::ComputeGraphPtr modelgraph = std::make_shared<ge::ComputeGraph>("test");
    EXPECT_EQ(CreateGraph_conv1_conv2_redu_eltw_relu_conv3_splitBatch_Big(modelgraph), fe::SUCCESS);

    Configuration &config_inst = Configuration::Instance(fe::VECTOR_CORE_NAME);
    map<string, string> options;
    string soc_version = "Ascend310";
    PlatformUtils::Instance().soc_version_ = soc_version;
    config_inst.Initialize(options);
    //configure.SetNeedL2Fusion(true);

    L2Optimizer l2_opt(fe::VECTOR_CORE_NAME);
    auto nodes = modelgraph->GetDirectNode();
    auto op_desc = nodes.at(0)->GetOpDesc();
    EXPECT_NE(op_desc, nullptr);
    int64_t streamId = op_desc->GetStreamId();
    EXPECT_EQ(fe::SUCCESS, l2_opt.GetL2DataAlloc((*modelgraph), 0, streamId));
}


/******************************************
* multioutput---->add
*         \------>/
* one data and two quoting to dst op
******************************************/
static uint32_t CreateGraph_multioutput_add(ge::ComputeGraphPtr &model_graph)
{
    uint32_t batch_size = 1;
    map<string, vector<desc_info>> src_map;
    map<string, vector<desc_info>> dst_map;
    map<string, vector<desc_info>>::iterator it;
    desc_info dscinfo;
    vector<desc_info> desctmp;

    ge::OpDescPtr opdef;
    vector<ge::OpDescPtr> op_list;
    vector<string> srcname_list;
    vector<string> dstname_list;
    ge::GeTensorDesc input_desc;
    ge::GeTensorDesc input_desc2;
    ge::GeTensorDesc output_desc;
    vector<ge::GeTensorDesc> inputdesc_list;
    vector<ge::GeTensorDesc> outputdesc_list;

    desctmp.clear();
    src_map.insert(pair<string, vector<desc_info >> ("multiOutput", desctmp));
    src_map.insert(pair<string, vector<desc_info >> ("add", desctmp));

    dst_map.insert(pair<string, vector<desc_info >> ("multiOutput", desctmp));
    dst_map.insert(pair<string, vector<desc_info >> ("add", desctmp));

    // conv1
    FillTensorDesc(input_desc, batch_size, 16, 128, 128, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0);
    FillTensorDesc(output_desc, batch_size, 16, 128, 128, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0);

    srcname_list.clear();
    dstname_list.clear();
    dstname_list.push_back("add");
    dstname_list.push_back("add");
    inputdesc_list.clear();
    inputdesc_list.push_back(input_desc);
    outputdesc_list.clear();
    outputdesc_list.push_back(output_desc);
    opdef = CreateOpDef("multiOutput", "multiOutput", srcname_list, dstname_list, inputdesc_list, outputdesc_list);
    ge::AttrUtils::SetListStr(opdef, OPDESC_DST_NAME, dstname_list);
    ge::AttrUtils::SetListStr(opdef, OPDESC_SRC_NAME, srcname_list);
    AllocScopeId(opdef, 100);
    op_list.push_back(opdef);

    it = dst_map.find("multiOutput");
    vector<desc_info> &vec1 = it->second;
    dscinfo.targetname = "add";
    dscinfo.index = 0;
    vec1.push_back(dscinfo);
    dscinfo.targetname = "add";
    dscinfo.index = 0;
    vec1.push_back(dscinfo);

    // add
    FillTensorDesc(input_desc, batch_size, 16, 128, 128, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0);
    FillTensorDesc(input_desc2, batch_size, 16, 128, 128, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0);
    FillTensorDesc(output_desc, batch_size, 16, 128, 128, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0);

    srcname_list.clear();
    srcname_list.push_back("multiOutput");
    srcname_list.push_back("multiOutput");
    dstname_list.clear();
    inputdesc_list.clear();
    inputdesc_list.push_back(input_desc);
    inputdesc_list.push_back(input_desc2);
    outputdesc_list.clear();
    outputdesc_list.push_back(output_desc);
    opdef = CreateOpDef("add", "add", srcname_list, dstname_list, inputdesc_list, outputdesc_list);
    ge::AttrUtils::SetListStr(opdef, OPDESC_DST_NAME, dstname_list);
    ge::AttrUtils::SetListStr(opdef, OPDESC_SRC_NAME, srcname_list);

    AllocScopeId(opdef, 101);
    op_list.push_back(opdef);

    it = src_map.find("add");
    vector<desc_info> &vec2 = it->second;
    dscinfo.targetname = "multiOutput";
    dscinfo.index = 0;
    vec2.push_back(dscinfo);
    dscinfo.targetname = "multiOutput";
    dscinfo.index = 1;
    vec2.push_back(dscinfo);

    CreateL2Graph2(model_graph, op_list, src_map, dst_map);
    return fe::SUCCESS;
}

TEST_F(L2FUSION_UT, test_multioutput_add)
{
  ge::ComputeGraphPtr modelgraph = std::make_shared<ge::ComputeGraph>("test");
  EXPECT_EQ(CreateGraph_multioutput_add(modelgraph), fe::SUCCESS);

  Configuration &config_inst = Configuration::Instance(fe::AI_CORE_NAME);
  PlatformUtils::Instance().pm_item_vec_[static_cast<size_t>(PlatformUtils::PlatformInfoItem::L2Type)] = static_cast<int64_t>(L2Type::Buff);

  map<string, string> options;
  string soc_version = "Ascend310";
  PlatformUtils::Instance().soc_version_ = soc_version;
  config_inst.Initialize(options);

  L2Optimizer l2_opt(fe::AI_CORE_NAME);
  auto nodes = modelgraph->GetDirectNode();
  auto op_desc = nodes.at(0)->GetOpDesc();
  EXPECT_NE(op_desc, nullptr);
  int64_t streamId = op_desc->GetStreamId();
  EXPECT_EQ(fe::SUCCESS, l2_opt.GetL2DataAlloc((*modelgraph), 0, streamId));
}

/************************
*tile1-->prelu1-->eltw-->fc-->conv3
*                   ^
*                   |
*                 conv2
*************************/
static uint32_t CreateGraph_conv1_conv2_redu_eltw_relu_conv3_special_nodes(ge::ComputeGraphPtr &model_graph)
{
    uint32_t batch_size = 2;
    map<string, vector<desc_info>> src_map;
    map<string, vector<desc_info>> dst_map;
    map<string, vector<desc_info>>::iterator it;
    desc_info dscinfo;
    vector<desc_info> desctmp;

    ge::OpDescPtr opdef;
    vector<ge::OpDescPtr> op_list;
    vector<string> srcname_list;
    vector<string> dstname_list;
    ge::GeTensorDesc input_desc;
    ge::GeTensorDesc input_desc2;
    ge::GeTensorDesc output_desc;
    vector<ge::GeTensorDesc> inputdesc_list;
    vector<ge::GeTensorDesc> outputdesc_list;

    desctmp.clear();
    src_map.insert(pair<string, vector<desc_info >> ("tile1", desctmp));
    src_map.insert(pair<string, vector<desc_info >> ("conv2", desctmp));
    src_map.insert(pair<string, vector<desc_info >> ("conv3", desctmp));
    src_map.insert(pair<string, vector<desc_info >> ("prelu1", desctmp));
    src_map.insert(pair<string, vector<desc_info >> ("eltw", desctmp));
    src_map.insert(pair<string, vector<desc_info >> ("fc", desctmp));

    dst_map.insert(pair<string, vector<desc_info >> ("tile1", desctmp));
    dst_map.insert(pair<string, vector<desc_info >> ("conv2", desctmp));
    dst_map.insert(pair<string, vector<desc_info >> ("conv3", desctmp));
    dst_map.insert(pair<string, vector<desc_info >> ("prelu1", desctmp));
    dst_map.insert(pair<string, vector<desc_info >> ("eltw", desctmp));
    dst_map.insert(pair<string, vector<desc_info >> ("fc", desctmp));

    FillTensorDesc(input_desc, batch_size, 64, 56, 56, ge::DT_FLOAT16, ge::FORMAT_NCHW);
    FillTensorDesc(output_desc, batch_size, 64, 56, 56, ge::DT_FLOAT16, ge::FORMAT_NCHW);

    srcname_list.clear();
    dstname_list.clear();
    dstname_list.push_back("eltw");
    inputdesc_list.clear();
    inputdesc_list.push_back(input_desc);
    outputdesc_list.clear();
    outputdesc_list.push_back(output_desc);
    opdef = CreateOpDef("conv2", "Convolution", srcname_list, dstname_list, inputdesc_list, outputdesc_list);
    op_list.push_back(opdef);

    it = dst_map.find("conv2");
    vector<desc_info> &vec1 = it->second;
    dscinfo.targetname = "eltw";
    dscinfo.index = 0;
    vec1.push_back(dscinfo);

    FillTensorDesc(input_desc, batch_size, 64, 56, 56, ge::DT_FLOAT16, ge::FORMAT_NCHW);
    FillTensorDesc(output_desc, batch_size, 64, 56, 56, ge::DT_FLOAT16, ge::FORMAT_NCHW);

    srcname_list.clear();
    dstname_list.clear();
    dstname_list.push_back("prelu1");
    inputdesc_list.clear();
    inputdesc_list.push_back(input_desc);
    outputdesc_list.clear();
    outputdesc_list.push_back(output_desc);
    opdef = CreateOpDef("tile1", "Tile", srcname_list, dstname_list, inputdesc_list, outputdesc_list);
    op_list.push_back(opdef);

    it = dst_map.find("tile1");
    vector<desc_info> &vec2 = it->second;
    dscinfo.targetname = "prelu1";
    dscinfo.index = 0;
    vec2.push_back(dscinfo);

    FillTensorDesc(input_desc, batch_size, 64, 56, 56, ge::DT_FLOAT16, ge::FORMAT_NCHW);
    FillTensorDesc(output_desc, batch_size, 64, 56, 56, ge::DT_FLOAT16, ge::FORMAT_NCHW);

    srcname_list.clear();
    srcname_list.push_back("tile1");
    dstname_list.clear();
    dstname_list.push_back("eltw");
    inputdesc_list.clear();
    inputdesc_list.push_back(input_desc);
    outputdesc_list.clear();
    outputdesc_list.push_back(output_desc);
    opdef = CreateOpDef("prelu1", "PReLU", srcname_list, dstname_list, inputdesc_list, outputdesc_list);
    op_list.push_back(opdef);

    it = dst_map.find("prelu1");
    vector<desc_info> &vec3 = it->second;
    dscinfo.targetname = "eltw";
    dscinfo.index = 0;
    vec3.push_back(dscinfo);

    it = src_map.find("prelu1");
    vector<desc_info> &vec4 = it->second;
    dscinfo.targetname = "tile1";
    dscinfo.index = 0;
    vec4.push_back(dscinfo);

    FillTensorDesc(input_desc, batch_size, 64, 56, 56, ge::DT_FLOAT16, ge::FORMAT_NCHW);
    FillTensorDesc(input_desc2, batch_size, 64, 56, 56, ge::DT_FLOAT16, ge::FORMAT_NCHW);
    FillTensorDesc(output_desc, batch_size, 64, 56, 56, ge::DT_FLOAT16, ge::FORMAT_NCHW);

    srcname_list.clear();
    srcname_list.push_back("conv2");
    srcname_list.push_back("prelu1");
    dstname_list.clear();
    dstname_list.push_back("fc");
    inputdesc_list.clear();
    inputdesc_list.push_back(input_desc);
    inputdesc_list.push_back(input_desc2);
    outputdesc_list.clear();
    outputdesc_list.push_back(output_desc);
    opdef = CreateOpDef("eltw", "Eltwise", srcname_list, dstname_list, inputdesc_list, outputdesc_list);
    op_list.push_back(opdef);

    it = dst_map.find("eltw");
    vector<desc_info> &vec5 = it->second;
    dscinfo.targetname = "fc";
    dscinfo.index = 0;
    vec5.push_back(dscinfo);

    it = src_map.find("eltw");
    vector<desc_info> &vec6 = it->second;
    dscinfo.targetname = "conv2";
    dscinfo.index = 0;
    vec6.push_back(dscinfo);
    dscinfo.targetname = "prelu1";
    dscinfo.index = 1;
    vec6.push_back(dscinfo);

    FillTensorDesc(input_desc, batch_size, 64, 56, 56, ge::DT_FLOAT16, ge::FORMAT_NCHW);
    FillTensorDesc(output_desc, batch_size, 64, 56, 56, ge::DT_FLOAT16, ge::FORMAT_NCHW);

    srcname_list.clear();
    srcname_list.push_back("eltw");
    dstname_list.clear();
    dstname_list.push_back("conv3");
    inputdesc_list.clear();
    inputdesc_list.push_back(input_desc);
    outputdesc_list.clear();
    outputdesc_list.push_back(output_desc);
    opdef = CreateOpDef("fc", "FullConnection", srcname_list, dstname_list, inputdesc_list, outputdesc_list);

    AllocScopeId(opdef, 100);
    op_list.push_back(opdef);

    it = dst_map.find("fc");
    vector<desc_info> &vec7 = it->second;
    dscinfo.targetname = "conv3";
    dscinfo.index = 0;
    vec7.push_back(dscinfo);

    it = src_map.find("fc");
    vector<desc_info> &vec8 = it->second;
    dscinfo.targetname = "eltw";
    dscinfo.index = 0;
    vec8.push_back(dscinfo);

    FillTensorDesc(input_desc, batch_size, 64, 56, 56, ge::DT_FLOAT16, ge::FORMAT_NCHW);
    FillTensorDesc(output_desc, batch_size, 64, 56, 56, ge::DT_FLOAT16, ge::FORMAT_NCHW);

    srcname_list.clear();
    srcname_list.push_back("fc");
    dstname_list.clear();
    inputdesc_list.clear();
    inputdesc_list.push_back(input_desc);
    outputdesc_list.clear();
    outputdesc_list.push_back(output_desc);
    opdef = CreateOpDef("conv3", "Convolution", srcname_list, dstname_list, inputdesc_list, outputdesc_list);
    op_list.push_back(opdef);

    it = src_map.find("conv3");
    vector<desc_info> &vec9 = it->second;
    dscinfo.targetname = "fc";
    dscinfo.index = 0;
    vec9.push_back(dscinfo);

    CreateL2Graph(model_graph, op_list, src_map, dst_map);
    return fe::SUCCESS;
}

TEST_F(L2FUSION_UT, test_conv1_conv2_relu_eltw_relu_conv3_special_nodes)
{
    ge::ComputeGraphPtr modelgraph = std::make_shared<ge::ComputeGraph>("test");
    EXPECT_EQ(CreateGraph_conv1_conv2_redu_eltw_relu_conv3_special_nodes(modelgraph), fe::SUCCESS);


    Configuration &config_inst = Configuration::Instance(fe::VECTOR_CORE_NAME);
    map<string, string> options;
    string soc_version = "Ascend310";
    PlatformUtils::Instance().soc_version_ = soc_version;
    config_inst.Initialize(options);
    //configure.SetNeedL2Fusion(true);

    L2Optimizer l2_opt(fe::VECTOR_CORE_NAME);
    auto nodes = modelgraph->GetDirectNode();
    auto op_desc = nodes.at(0)->GetOpDesc();
    EXPECT_NE(op_desc, nullptr);
    int64_t streamId = op_desc->GetStreamId();
    //STREAM_L2_MAP.insert(std::pair<rtStream_t, TaskL2InfoMap>(streamId, l2_info));//if has been inserted
    TaskL2InfoMap l2_info_map;
    StreamL2Info::Instance().SetStreamL2Info(streamId, l2_info_map, "Batch_-1");
    EXPECT_EQ(fe::SUCCESS, l2_opt.GetL2DataAlloc((*modelgraph), 0, streamId));
}
namespace {

  static void CreateOneOpGraph(ge::ComputeGraphPtr graph) {
    ge::OpDescPtr scale_op = std::make_shared<ge::OpDesc>("scale", "Scale");
    // add descriptor
    vector<int64_t> dim(5, 1);
    ge::GeShape shape(dim);
    ge::GeTensorDesc out_desc1(shape);
    ge::GeTensorDesc out_desc2(shape);
    out_desc1.SetFormat(ge::FORMAT_NC1HWC0);
    out_desc1.SetDataType(ge::DT_FLOAT16);
    out_desc2.SetFormat(ge::FORMAT_NC1HWC0);
    out_desc2.SetDataType(ge::DT_FLOAT16);
    scale_op->AddInputDesc("x", out_desc1);
    scale_op->AddInputDesc("y", out_desc2);
    scale_op->AddOutputDesc("z", out_desc2);
    ge::NodePtr relu_node = graph->AddNode(scale_op);
  }
}

/************************
*common tests
*************************/
TEST_F(L2FUSION_UT, L2_cache_mode)
{
    ge::ComputeGraphPtr modelgraph = std::make_shared<ge::ComputeGraph>("test");
    EXPECT_EQ(CreateGraph_conv1_conv2_redu_eltw_relu_conv3_splitBatch_Big(modelgraph), fe::SUCCESS);

    Configuration &config_inst = Configuration::Instance(fe::VECTOR_CORE_NAME);
    map<string, string> options;
    string soc_version = "Ascend910A";
    PlatformUtils::Instance().soc_version_ = soc_version;
    config_inst.Initialize(options);

    L2Optimizer l2_opt(fe::VECTOR_CORE_NAME);
    auto nodes = modelgraph->GetDirectNode();
    auto op_desc = nodes.at(0)->GetOpDesc();
    EXPECT_NE(op_desc, nullptr);
    int64_t streamId = op_desc->GetStreamId();
    EXPECT_EQ(fe::SUCCESS, l2_opt.GetL2DataAlloc((*modelgraph), 0, streamId));
}

TEST_F(L2FUSION_UT, update_input_for_l2_fusion_failed)
{
    Configuration &configure = Configuration::Instance(fe::AI_CORE_NAME);
    ge::ComputeGraphPtr modelgraph = std::make_shared<ge::ComputeGraph>("test");
    EXPECT_EQ(CreateGraph_conv1_conv2_redu_eltw_relu_conv3_splitBatch_Big(modelgraph), fe::SUCCESS);

    L2Optimizer l2_opt(fe::AI_CORE_NAME);
    L2FusionInfoPtr L2Info = std::make_shared<TaskL2FusionInfo_t>();
    L2FusionData_t l2_data_ = {1, 2, 3};
    L2FusionDataMap_t output_;
    output_.emplace(0, l2_data_);
    L2Info->node_name = "testName";
    L2Info->output = output_;
    vector<int64_t> input_vector = {1};
    vector<int64_t> output_vector = {1};
    for (auto &node : (*modelgraph).GetDirectNode()) {
      node->GetOpDesc()->SetExtAttr(ATTR_NAME_TASK_L2_FUSION_INFO_EXTEND_PTR, L2Info);
      node->GetOpDesc()->SetInputOffset(input_vector);
      node->GetOpDesc()->SetOutputOffset(output_vector);
    }
    EXPECT_EQ(fe::FAILED, l2_opt.UpdateInputForL2Fusion(*modelgraph));
}

TEST_F(L2FUSION_UT, case1_test_l2_fusion)
{
    Configuration &configure = Configuration::Instance(fe::AI_CORE_NAME);
    ge::ComputeGraphPtr modelgraph = std::make_shared<ge::ComputeGraph>("test");
    EXPECT_EQ(CreateGraph_conv1_conv2_redu_eltw_relu_conv3_splitBatch_Big(modelgraph), fe::SUCCESS);

    L2Optimizer l2_opt(fe::AI_CORE_NAME);
    L2FusionInfoPtr L2Info = std::make_shared<TaskL2FusionInfo_t>();
    L2Info->node_name = "testName";
    vector<int64_t> input_vector = {1};
    vector<int64_t> output_vector = {1};
    for (auto &node : (*modelgraph).GetDirectNode()) {
      node->GetOpDesc()->SetExtAttr(ATTR_NAME_TASK_L2_FUSION_INFO_EXTEND_PTR, L2Info);
      node->GetOpDesc()->SetInputOffset(input_vector);
      node->GetOpDesc()->SetOutputOffset(output_vector);
    }

    EXPECT_EQ(fe::SUCCESS, l2_opt.UpdateDDRForL2Fusion((*modelgraph), 0));
}

TEST_F(L2FUSION_UT, case2_test_l2_fusion)
{
    Configuration &configure = Configuration::Instance(fe::AI_CORE_NAME);
    ge::ComputeGraphPtr modelgraph = std::make_shared<ge::ComputeGraph>("test");
    EXPECT_EQ(CreateGraph_conv1_conv2_redu_eltw_relu_conv3_splitBatch_Big(modelgraph), fe::SUCCESS);

    L2Optimizer l2_opt(fe::AI_CORE_NAME);
    L2FusionInfoPtr L2Info = std::make_shared<TaskL2FusionInfo_t>();
    L2Info->node_name = "testName";
    L2FusionData_t l2_data_ = {1, 2, 3};
    L2FusionDataMap_t input_;
    input_.emplace(0, l2_data_);
    L2Info->input = input_;
    L2FusionDataMap_t output_;
    output_.emplace(0, l2_data_);
    L2Info->output = output_;
    fe_sm_desc_t fe_sm_;
    fe_sm_.node_name[0] = "node_name0";
    L2Info->l2_info = fe_sm_;
    vector<int64_t> input_vector = {1};
    vector<int64_t> output_vector = {1};
    for (auto &node : (*modelgraph).GetDirectNode()) {
      node->GetOpDesc()->SetExtAttr(ATTR_NAME_TASK_L2_FUSION_INFO_EXTEND_PTR, L2Info);
      node->GetOpDesc()->SetInputOffset(input_vector);
      node->GetOpDesc()->SetOutputOffset(output_vector);
    }

    EXPECT_EQ(fe::SUCCESS, l2_opt.UpdateDDRForL2Fusion((*modelgraph), 0));
}

TEST_F(L2FUSION_UT, test_get_stream_no_op_index)
{
    L2Optimizer l2_opt(fe::AI_CORE_NAME);
    TaskL2InfoMap l2_info_map;
    StreamL2Info::Instance().SetStreamL2Info(0, l2_info_map, "Batch_-1");
    TaskL2Info * output;
    std::string node_name = "node1";
    EXPECT_EQ(fe::FAILED, StreamL2Info::Instance().GetStreamL2Info(0, node_name, output, "Batch_-1"));
}


TEST_F(L2FUSION_UT, test_get_stream_no_stream_id)
{
    L2Optimizer l2_opt(fe::AI_CORE_NAME);
    int64_t streamId = 0;
    TaskL2Info * output;
    std::string node_name = "node1";
    EXPECT_EQ(fe::FAILED, StreamL2Info::Instance().GetStreamL2Info(streamId, node_name, output, "Batch_-1"));
}

TEST_F(L2FUSION_UT, test_get_stream)
{
    L2Optimizer l2_opt(fe::AI_CORE_NAME);
    int64_t streamId = 0;
    TaskL2InfoMap l2_info_map;
    TaskL2Info l2_info;
    l2_info_map.insert(TaskL2InfoPair("node1", l2_info));
    StreamL2Info::Instance().SetStreamL2Info(streamId, l2_info_map, "Batch_-1");
    TaskL2Info * output;
    std::string node_name = "node1";
    EXPECT_EQ(fe::SUCCESS, StreamL2Info::Instance().GetStreamL2Info(streamId, node_name, output, "Batch_-1"));
}

static uint32_t CreateGraph_eltw(ge::ComputeGraphPtr &model_graph)
{
    uint32_t batch_size = 4;
    map<string, vector<desc_info>> src_map;
    map<string, vector<desc_info>> dst_map;
    map<string, vector<desc_info>>::iterator it;
    desc_info dscinfo;
    vector<desc_info> desctmp;

    ge::OpDescPtr opdef;
    vector<ge::OpDescPtr> op_list;
    vector<string> srcname_list;
    vector<string> dstname_list;
    ge::GeTensorDesc input_desc;
    ge::GeTensorDesc input_desc2;
    ge::GeTensorDesc output_desc;
    vector<ge::GeTensorDesc> inputdesc_list;
    vector<ge::GeTensorDesc> outputdesc_list;

    desctmp.clear();
    src_map.insert(pair<string, vector<desc_info >> ("tile1", desctmp));
    src_map.insert(pair<string, vector<desc_info >> ("conv2", desctmp));
    src_map.insert(pair<string, vector<desc_info >> ("conv3", desctmp));
    src_map.insert(pair<string, vector<desc_info >> ("prelu1", desctmp));
    src_map.insert(pair<string, vector<desc_info >> ("eltw", desctmp));
    src_map.insert(pair<string, vector<desc_info >> ("fc", desctmp));

    dst_map.insert(pair<string, vector<desc_info >> ("tile1", desctmp));
    dst_map.insert(pair<string, vector<desc_info >> ("conv2", desctmp));
    dst_map.insert(pair<string, vector<desc_info >> ("conv3", desctmp));
    dst_map.insert(pair<string, vector<desc_info >> ("prelu1", desctmp));
    dst_map.insert(pair<string, vector<desc_info >> ("eltw", desctmp));
    dst_map.insert(pair<string, vector<desc_info >> ("fc", desctmp));

    FillTensorDesc(input_desc, batch_size, 64, 56, 56, ge::DT_FLOAT16, ge::FORMAT_NCHW);
    FillTensorDesc(output_desc, batch_size, 64, 56, 56, ge::DT_FLOAT16, ge::FORMAT_NCHW);

    srcname_list.clear();
    dstname_list.clear();
    dstname_list.push_back("eltw");
    inputdesc_list.clear();
    inputdesc_list.push_back(input_desc);
    outputdesc_list.clear();
    outputdesc_list.push_back(output_desc);
    opdef = CreateOpDef("conv2", "Convolution", srcname_list, dstname_list, inputdesc_list, outputdesc_list);
    op_list.push_back(opdef);

    it = dst_map.find("conv2");
    vector<desc_info> &vec1 = it->second;
    dscinfo.targetname = "eltw";
    dscinfo.index = 0;
    vec1.push_back(dscinfo);

    FillTensorDesc(input_desc, batch_size, 64, 56, 56, ge::DT_FLOAT16, ge::FORMAT_NCHW);
    FillTensorDesc(input_desc2, batch_size, 64, 56, 56, ge::DT_FLOAT16, ge::FORMAT_NCHW);
    FillTensorDesc(output_desc, batch_size, 64, 56, 64, ge::DT_FLOAT16, ge::FORMAT_NCHW);

    srcname_list.clear();
    srcname_list.push_back("conv2");
    srcname_list.push_back("prelu1");
    dstname_list.clear();
    dstname_list.push_back("fc");
    inputdesc_list.clear();
    inputdesc_list.push_back(input_desc);
    inputdesc_list.push_back(input_desc2);
    outputdesc_list.clear();
    outputdesc_list.push_back(output_desc);
    opdef = CreateOpDef("eltw", "Eltwise", srcname_list, dstname_list, inputdesc_list, outputdesc_list);
    op_list.push_back(opdef);

    AllocScopeId(opdef, 100);

    CreateL2Graph(model_graph, op_list, src_map, dst_map);
    return fe::SUCCESS;
}
void BuildGraphForL2Fusion1(ge::ComputeGraphPtr graph, int32_t reluflag) {
  OpDescPtr data = std::make_shared<OpDesc>("DATA0", fe::DATA);
  OpDescPtr data1 = std::make_shared<OpDesc>("DATA1", fe::DATA);
  OpDescPtr data2 = std::make_shared<OpDesc>("DATA2", fe::DATA);
  OpDescPtr conv = std::make_shared<OpDesc>("conv", "Convolution");
  OpDescPtr elemwise = std::make_shared<OpDesc>("elem", "Eltwise");
  OpDescPtr elemwise1 = std::make_shared<OpDesc>("elem1", "Eltwise");
  OpDescPtr relu = std::make_shared<OpDesc>("relu", "ReLU");

  vector<int64_t> dim(4, 4);
  GeShape shape(dim);
  GeTensorDesc out_desc(shape);
  out_desc.SetDataType(DT_DUAL_SUB_INT8);

  data->AddOutputDesc(out_desc);
  data1->AddOutputDesc(out_desc);
  data2->AddOutputDesc(out_desc);
  conv->AddInputDesc(out_desc);
  conv->AddInputDesc(out_desc);
  conv->AddOutputDesc(out_desc);
  elemwise->AddInputDesc(out_desc);
  elemwise->AddInputDesc(out_desc);
  elemwise->AddOutputDesc(out_desc);
  elemwise1->AddInputDesc(out_desc);
  elemwise1->AddOutputDesc(out_desc);
  relu->AddInputDesc(out_desc);
  relu->AddOutputDesc(out_desc);

  AttrUtils::SetInt(conv, FE_IMPLY_TYPE, fe::EN_IMPL_HW_TBE);
  std::vector<int64_t> params = {0, 0, 0, 0, 0, 1, 0, 1};
  AttrUtils::SetListInt(conv, "ub_atomic_params", params);
  AttrUtils::SetBool(conv, "Aipp_Conv_Flag", true);
  conv->SetWorkspaceBytes({0});
  AttrUtils::SetInt(elemwise, FE_IMPLY_TYPE, fe::EN_IMPL_HW_TBE);
  AttrUtils::SetInt(elemwise1, FE_IMPLY_TYPE, fe::EN_IMPL_HW_TBE);
  AttrUtils::SetInt(relu, FE_IMPLY_TYPE, fe::EN_IMPL_HW_TBE);
  vector<int64_t> input_vector = {1};
  vector<int64_t> output_vector = {1};
  elemwise1->SetInputOffset(input_vector);
  elemwise1->SetOutputOffset(output_vector);

  NodePtr data_node = graph->AddNode(data);
  NodePtr data1_node = graph->AddNode(data1);
  NodePtr data2_node = graph->AddNode(data2);
  NodePtr conv_node = graph->AddNode(conv);
  NodePtr elemwise_node = graph->AddNode(elemwise);
  NodePtr elemwise1_node = graph->AddNode(elemwise1);
  NodePtr relu_node = graph->AddNode(relu);

  const char tbe_bin[] = "tbe_bin";
  vector<char> buffer(tbe_bin, tbe_bin + strlen(tbe_bin));
  ge::OpKernelBinPtr tbe_kernel_ptr = std::make_shared<ge::OpKernelBin>(
    conv_node->GetName(), std::move(buffer));
  conv_node->GetOpDesc()->SetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel_ptr);

  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0),
                      conv_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(data1_node->GetOutDataAnchor(0),
                      conv_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0),
                      elemwise_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(data2_node->GetOutDataAnchor(0),
                      elemwise_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(elemwise_node->GetOutDataAnchor(0),
                      elemwise1_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(elemwise1_node->GetOutDataAnchor(0),
                      relu_node->GetInDataAnchor(0));

  //elementwise l2 Info
  L2FusionInfoPtr elementwise_l2_info_ptr = std::make_shared<TaskL2FusionInfo_t>();
  //data
  uint64_t L2_mirror_addr=0;          // preload or swap source address
  uint32_t L2_data_section_size=123;    // every data size
  uint8_t L2_preload=0;               // 1 - preload from mirror_addr, 0 - no preload
  uint8_t modified=1;                 // 1 - data will be modified by kernel, 0 - no modified
  uint8_t priority=1;                 // data priority
  int8_t prev_L2_page_offset_base=-1;  // remap source section offset
  uint8_t L2_page_offset_base=0;      // remap destination section offset
  uint8_t L2_load_to_ddr=0;           // 1 - need load out, 0 - no need
  rtSmData_t tmp_data={L2_mirror_addr,L2_data_section_size,L2_preload,modified,priority,prev_L2_page_offset_base,L2_page_offset_base,L2_page_offset_base,L2_load_to_ddr};
  tmp_data.reserved[2]={0};

  elementwise_l2_info_ptr->l2_info.l2ctrl.data[0]=tmp_data;
  elementwise_l2_info_ptr->l2_info.node_name[0]={"elem1"};
  elementwise_l2_info_ptr->l2_info.node_name[1]={"elem1"};
  elementwise_l2_info_ptr->l2_info.node_name[2]={"elem1"};
  elementwise_l2_info_ptr->l2_info.node_name[3]={"elem1"};
  elementwise_l2_info_ptr->l2_info.output_index[0]={0};
  elementwise_l2_info_ptr->l2_info.output_index[1]={0};
  elementwise_l2_info_ptr->l2_info.output_index[2]={0};
  elementwise_l2_info_ptr->l2_info.output_index[3]={0};
  elementwise_l2_info_ptr->l2_info.l2ctrl.size=60;
  elementwise_l2_info_ptr->node_name="elem";
  L2FusionData_t elem_output={1,123,2};
  elementwise_l2_info_ptr->output[0]=elem_output;
  elementwise_l2_info_ptr->input[0]=elem_output;
  elementwise_l2_info_ptr->output[1]=elem_output;
  elementwise_l2_info_ptr->input[1]=elem_output;
  elementwise_l2_info_ptr->output[2]=elem_output;
  elementwise_l2_info_ptr->input[2]=elem_output;

  (void)ge::AttrUtils::SetBool(elemwise1_node->GetOpDesc(), NEED_RE_PRECOMPILE, true);
  elemwise1_node->GetOpDesc()->SetExtAttr(
    ATTR_NAME_TASK_L2_FUSION_INFO_EXTEND_PTR, elementwise_l2_info_ptr);

}
TEST_F(L2FUSION_UT, test_update_standing_datasize_withoutputset)
{
    ge::ComputeGraphPtr modelgraph = std::make_shared<ge::ComputeGraph>("test");
    EXPECT_EQ(CreateGraph_eltw(modelgraph), fe::SUCCESS);


    Configuration &config_inst = Configuration::Instance(fe::VECTOR_CORE_NAME);
    map<string, string> options;
    string soc_version = "Ascend310";
    PlatformUtils::Instance().soc_version_ = soc_version;
    config_inst.Initialize(options);
    L2Optimizer l2_opt(fe::VECTOR_CORE_NAME);
    auto nodes = modelgraph->GetDirectNode();
    auto op_desc = nodes.at(0)->GetOpDesc();
    EXPECT_NE(op_desc, nullptr);
    int64_t streamId = op_desc->GetStreamId();
    EXPECT_EQ(fe::SUCCESS, l2_opt.GetL2DataAlloc((*modelgraph), 0, streamId));
}

TEST_F(L2FUSION_UT, test_Get_l2_data_alloc_L2_FUSION)
{
  ge::ComputeGraphPtr modelgraph = std::make_shared<ge::ComputeGraph>("test");

  BuildGraphForL2Fusion1(modelgraph, 1);
  Configuration &config_inst = Configuration::Instance(fe::AI_CORE_NAME);
  map<string, string> options;
  string soc_version = "Ascend910B1";
  PlatformUtils::Instance().soc_version_ = soc_version;
  config_inst.Initialize(options);
  L2Optimizer l2_opt(fe::AI_CORE_NAME);
  auto nodes = modelgraph->GetDirectNode();
  auto op_desc = nodes.at(0)->GetOpDesc();
  EXPECT_NE(op_desc, nullptr);
  int64_t streamId = op_desc->GetStreamId();
  EXPECT_EQ(fe::SUCCESS, l2_opt.GetL2DataAlloc((*modelgraph), 10, streamId));
}

TEST_F(L2FUSION_UT, test_allocdata_standing_data_special)
{
    L2FusionAllocation l2fusionAlloc;
    L2BufferInfo l2;
    std::map<uint64_t, size_t> count_map;
    std::vector<OpL2DataInfo> datas_map;
    TensorL2AllocMap standing_alloc_data;
    TensorL2DataMap converge_data;
    OpL2AllocMap alloc_map;
    uint64_t max_page = 63;
    uint32_t data_in_l2_id = 1;
    int64_t page_size = 1;
    int32_t page_num_left = 0;
    Status status = l2fusionAlloc.AllocateStandingDataSpecial(page_size, count_map, standing_alloc_data,
                                                              data_in_l2_id, page_num_left);
    EXPECT_EQ(status, fe::SUCCESS);
}
