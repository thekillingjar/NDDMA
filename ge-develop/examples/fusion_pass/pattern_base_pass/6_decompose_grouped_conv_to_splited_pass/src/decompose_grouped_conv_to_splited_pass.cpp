/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include "es_all_ops.h"
#include "ge/fusion/pass/decompose_pass.h"
#include "ge/ge_utils.h"

using namespace ge;
using namespace fusion;

/*
|o>-----------------------------------
|o>                                a
|o>        a                       |
|o>        |                     Split
|o>        |                  /    |    \
|o>    Group Conv          Conv0  ...  Convn
|o>        |          ==>     \    |    /
|o>        |                    Concat
|o>    MaxPooling                  |
|o>                            MaxPooling
|o>-----------------------------------
说明：本例识别上图中左边的分组卷积结构并通过图修改接口用Split+卷积+Concat代替
*/

class DecomposeGroupedConvToSplitedPass : public DecomposePass {
public:
    DecomposeGroupedConvToSplitedPass(const std::vector<AscendString> &op_types) : DecomposePass(op_types) {}

protected:
    // 判断图中类型符合op_types（此为Conv2D）的节点matched_node，是否满足groups不为1且数据格式为NCHW，并保存相关属性
    bool MeetRequirements(const GNode &matched_node) override {
        int64_t groups;
        AscendString data_format;
        std::cout << "Define MeetRequirements for DecomposeGroupedConvToSplitedPass" << std::endl;
        if (matched_node.GetAttr("groups", groups) != GRAPH_SUCCESS ||
            matched_node.GetAttr("data_format", data_format) != GRAPH_SUCCESS) {
            std::cout << "Get attributes failed" << std::endl;
            return false;
        }
        if (groups == 1 || data_format != "NCHW") {
            return false;
        }
        return true;
    }

    /*
    |o>-----------------------------------
    |o>             Split
    |o>          /    |    \
    |o>      Conv0   ...   Convn
    |o>          \    |    /
    |o>             Concat
    |o>-----------------------------------
    上图为Replacement定义的replacement结构
    通过此处定义的replacement替换图中类型在op_types中且MeetRequirements为True的节点
    */
    GraphUniqPtr Replacement(const GNode &matched_node) override {
        std::cout << "Define Replacement for DecomposeGroupedConvToSplitedPass" << std::endl;
        int64_t groups;
        std::vector<int64_t> strides;
        std::vector<int64_t> pads;
        std::vector<int64_t> dilations;
        AscendString data_format;
        if (matched_node.GetAttr("groups", groups) != GRAPH_SUCCESS ||
            matched_node.GetAttr("strides", strides) != GRAPH_SUCCESS ||
            matched_node.GetAttr("pads", pads) != GRAPH_SUCCESS ||
            matched_node.GetAttr("dilations", dilations) != GRAPH_SUCCESS ||
            matched_node.GetAttr("data_format", data_format) != GRAPH_SUCCESS) {
            std::cout << "Get attributes failed" << std::endl;
            return nullptr;
        }

        auto replacement_graph_builder = es::EsGraphBuilder("replacement");
        auto [r_tensor, r_filter] = replacement_graph_builder.CreateInputs<2>();
        auto r_tensors = es::Split(replacement_graph_builder.CreateScalar(1), r_tensor, groups, groups);
        auto r_filters = es::Split(replacement_graph_builder.CreateScalar(0), r_filter, groups, groups);
        std::vector<es::EsTensorHolder> convs;
        for (int i = 0; i < groups; i++) {
            auto tensor = r_tensors[i];
            auto filter = r_filters[i];
            auto r_conv = es::Conv2D(tensor, filter, nullptr, nullptr, strides, pads, dilations, 1, "NCHW");

            TensorDesc conv2d_data_desc(Shape(), FORMAT_NCHW, DT_FLOAT);
            conv2d_data_desc.SetOriginFormat(FORMAT_NCHW);
            r_conv.GetProducer()->UpdateInputDesc(0, conv2d_data_desc);
            r_conv.GetProducer()->UpdateInputDesc(1, conv2d_data_desc);
            r_conv.GetProducer()->UpdateOutputDesc(0, conv2d_data_desc);
            convs.emplace_back(r_conv);
        }
        auto res = es::Concat(replacement_graph_builder.CreateScalar(1), convs, groups);
        auto replace_graph = replacement_graph_builder.BuildAndReset({res});

        // 当前pass注册在after infershape阶段，需要自行保证替换部分的shape连续
        bool is_support = InferShapeAndCheckSupport(matched_node, *replace_graph);
        if (!is_support) {
            return nullptr;
        }
        return replace_graph;
    }

private:
    // 因为pass会被重复执行，不建议使用私有成员
    // 如果使用，需要保证pass对象的可重入性
    bool InferShapeAndCheckSupport(const GNode &matched_node, const Graph &graph) {
        // Shape推导
        std::vector<Shape> input_shapes;
        auto input_size = matched_node.GetInputsSize();
        for (int i = 0; i < static_cast<int>(input_size); i++) {
            TensorDesc tensor_desc;
            matched_node.GetInputDesc(i, tensor_desc);
            input_shapes.emplace_back(tensor_desc.GetShape());
        }
        if (GeUtils::InferShape(graph, input_shapes) != SUCCESS) {
            std::cout << "InferShape failed" << std::endl;
            return false;
        }

        // 检查当前ai core是否支持新节点，在Shape推导之后调用CheckNodeSupportOnAicore
        auto nodes = graph.GetAllNodes();
        for (GNode node: nodes) {
            AscendString node_type;
            node.GetType(node_type);
            if (node_type == "Const" || node_type == "Data") {
                continue;
            }
            bool is_support;
            AscendString unsupport_reason;
            if (GeUtils::CheckNodeSupportOnAicore(node, is_support, unsupport_reason) != SUCCESS) {
                std::cout << "CheckNodeSupportOnAicore failed, reason: " << unsupport_reason.GetString() << std::endl;
                return false;
            }
            if (!is_support) {
                AscendString node_name;
                node.GetName(node_name);
                std::cout << "Node is not supported on current ai core! reason: " << unsupport_reason.GetString()
                        << " node name: " << node_name.GetString() << " node type: " << node_type.GetString() <<
                        std::endl;
                return false;
            }
        }
        return true;
    }
};

REG_DECOMPOSE_PASS(DecomposeGroupedConvToSplitedPass, {"Conv2D"}).Stage(CustomPassStage::kAfterInferShape);
