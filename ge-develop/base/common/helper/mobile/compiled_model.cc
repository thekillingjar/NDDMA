/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "compiled_model.h"

#include <memory>

#include "common/checker.h"
#include "compiled_target_saver.h"
#include "common/scope_guard.h"
#include "model_buffer_helper.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_type_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/model.h"
#include "graph/buffer.h"
#include "graph/op_desc.h"
#include "graph/node.h"
#include "proto/ge_ir.pb.h"
#include "model.h"

namespace {

template<typename T>
auto MemSizeCeil(T size) -> T
{
    constexpr uint32_t align_bytes = 512;
    return ((size + static_cast<T>(align_bytes - 1U)) / static_cast<T>(align_bytes)) * static_cast<T>(align_bytes);
}

const std::string GRAPH_OP_TYPE = "GraphOp";

ge::Status SetGraphOpAttrs(
    ge::OpDescPtr& graph_op_desc,
    const std::vector<ge::BaseBuffer>& weights_list,
    const std::vector<ge::CompiledTargetPtr>& compiled_targets,
    int64_t input_output_size,
    int64_t memory_size)
{
    // only support one graph op node dispatch to npucl
    // align attr to mobile
    const std::string attr_flush_output_cache_str = "attr_flush_output_cache";
    constexpr bool attr_flush_output_cache_value = false;
    (void)ge::AttrUtils::SetBool(graph_op_desc, attr_flush_output_cache_str, attr_flush_output_cache_value);

    const std::string attr_invalid_feature_map_cache_str = "attr_invalid_feature_map_cache";
    constexpr bool attr_invalid_feature_map_cache_value = false;
    (void)ge::AttrUtils::SetBool(graph_op_desc, attr_invalid_feature_map_cache_str, attr_invalid_feature_map_cache_value);

    const std::string attr_invalid_input_cache_str = "attr_invalid_input_cache";
    constexpr bool attr_invalid_input_cache_value = false;
    (void)ge::AttrUtils::SetBool(graph_op_desc, attr_invalid_input_cache_str, attr_invalid_input_cache_value);

    const std::string cl_name_str = "cl_name";
    const std::string cl_name_value = "NPUCL";
    (void)ge::AttrUtils::SetStr(graph_op_desc, cl_name_str, cl_name_value);

    // task info
    const std::string graphop_task_offset_str = "graphop_task_offset";
    constexpr int64_t graphop_task_offset_value = 0;
    (void)ge::AttrUtils::SetInt(graph_op_desc, graphop_task_offset_str, graphop_task_offset_value);

    const std::string graphop_task_size_str = "graphop_task_size";
    GE_ASSERT_NOTNULL(compiled_targets[0], "[Mobile] compiled_targets[0] is nullptr.");
    const int64_t graphop_task_size_value = static_cast<int64_t>(compiled_targets[0]->GetSize());
    (void)ge::AttrUtils::SetInt(graph_op_desc, graphop_task_size_str, graphop_task_size_value);

    // weight info
    const std::string graphop_weight_offset_str = "graphop_weight_offset";
    constexpr int64_t graphop_weight_offset_value = 0;
    (void)ge::AttrUtils::SetInt(graph_op_desc, graphop_weight_offset_str, graphop_weight_offset_value);

    const std::string graphop_weight_size_str = "graphop_weight_size";
    const int64_t graphop_weight_size_value = static_cast<int64_t>(weights_list[0].GetSize());
    (void)ge::AttrUtils::SetInt(graph_op_desc, graphop_weight_size_str, graphop_weight_size_value);

    // in and out and fm info
    const std::string input_output_size_str = "input_output_size";
    (void)ge::AttrUtils::SetInt(graph_op_desc, input_output_size_str, input_output_size);

    const std::string memory_size_str = "memory_size";
    (void)ge::AttrUtils::SetInt(graph_op_desc, memory_size_str, memory_size);

    // sc info
    const std::string model_cache_offset_str = "model_cache_offset";
    constexpr int64_t model_cache_offset_value = 0;
    (void)ge::AttrUtils::SetInt(graph_op_desc, model_cache_offset_str, model_cache_offset_value);

    const std::string subgraph_name_str = "subgraph_name";
    const std::string subgraph_name_value = "SubGraph_0";
    (void)ge::AttrUtils::SetStr(graph_op_desc, subgraph_name_str, subgraph_name_value);
    return ge::SUCCESS;
}

ge::ComputeGraphPtr ConvertComputeGraphToMobile(
    const ge::ComputeGraphPtr& ori_compute_graph,
    const std::vector<ge::BaseBuffer>& weights_list,
    const std::vector<ge::CompiledTargetPtr>& compiled_targets)
{
    // copy subgraph from ori graph and set name
    ge::ComputeGraphPtr mobile_subgraph = std::make_shared<ge::ComputeGraph>(ori_compute_graph->GetName());
    if (mobile_subgraph == nullptr) {
        GELOGE(ge::FAILED, "[Mobile] mobile compute graph is nullptr.");
        return nullptr;
    }
    if (ge::GraphUtils::CopyComputeGraph(ori_compute_graph, mobile_subgraph) != ge::SUCCESS) {
        GELOGE(ge::FAILED, "[Mobile] copy compute graph failed.");
        return nullptr;
    }
    std::string main_graph_name = "ge_default";
    std::string subgraph_name = "SubGraph_0";
    mobile_subgraph->SetName(main_graph_name + "/" + subgraph_name);

    // acquire all data and netoutput op desc in subgraph
    std::vector<ge::OpDescPtr> data_op_descs;
    ge::OpDescPtr netoutput_op_desc;
    // also get input and output size and fm size(exclude data and netoutput op)
    int64_t input_output_size = 0;
    int64_t memory_size = 0;
    // also get weight info
    for (const ge::NodePtr &node : mobile_subgraph->GetNodes(mobile_subgraph->GetGraphUnknownFlag())) {
        GE_ASSERT_NOTNULL(node, "[Mobile] node is nullptr.");
        // const is weight not process
        if (ge::OpTypeUtils::IsConstNode(node->GetType())) {
            GELOGI("[Mobile] skip const node name: %s", node->GetName().c_str());
            continue;
        }
        GE_ASSERT_NOTNULL(node->GetOpDesc(), "[Mobile] node->GetOpDesc() is nullptr.");
        // acquire op input meminfo
        std::vector<int64_t> input_offset = node->GetOpDesc()->GetInputOffset();
        std::vector<int64_t> input_data_size;
        for (size_t i = 0; i < input_offset.size(); i++) {
            int64_t size_of_byte = 0;
            (void)ge::TensorUtils::GetSize(node->GetOpDesc()->GetInputDesc(static_cast<uint32_t>(i)), size_of_byte);
            (void)input_data_size.emplace_back(size_of_byte);
        }
        // acquire op output meminfo
        std::vector<int64_t> output_offset = node->GetOpDesc()->GetOutputOffset();
        std::vector<int64_t> output_data_size;
        for (size_t i = 0; i < output_offset.size(); i++) {
            int64_t size_of_byte = 0;
            (void)ge::TensorUtils::GetSize(node->GetOpDesc()->GetOutputDesc(static_cast<uint32_t>(i)), size_of_byte);
            (void)output_data_size.emplace_back(size_of_byte);
        }
        // debug info
        GELOGI("[Mobile] op name: %s, mem info: ", node->GetName().c_str());
        for (size_t i = 0; i < input_offset.size(); i++) {
            GELOGI("[Mobile]    input[%d] data size: %d , offset: %d",
                i, input_data_size[i], input_offset[i]);
        }
        for (size_t i = 0; i < output_offset.size(); i++) {
            GELOGI("[Mobile]    output[%d] data size: %d , offset: %d",
                i, output_data_size[i], output_offset[i]);
        }
        // acquire data and netoutput op desc and input output mem size
        if (ge::OpTypeUtils::IsDataNode(node->GetType())) {
            data_op_descs.push_back(node->GetOpDesc());
            input_output_size += MemSizeCeil(output_data_size[0]);
            GELOGI("[Mobile] output_data_size[0]: %d", output_data_size[0]);
        }
        if (ge::OpTypeUtils::IsGraphOutputNode(node->GetType())) {
            netoutput_op_desc = node->GetOpDesc();
            for (const auto& size_of_byte: input_data_size) {
                input_output_size += MemSizeCeil(size_of_byte);
                GELOGI("[Mobile] input_data_size: %d", size_of_byte);
            }
        }
        GELOGI("[Mobile] input_output_size: %d", input_output_size);
        // acquire fm size
        if (ge::OpTypeUtils::IsGraphOutputNode(node->GetType())) {
            for (size_t i = 0; i < input_offset.size(); i++) {
                const int64_t size_of_byte = input_offset[i] + MemSizeCeil(input_data_size[i]);
                memory_size = std::max(memory_size, size_of_byte);
            }
        } else {
            for (size_t i = 0; i < output_offset.size(); i++) {
                const int64_t size_of_byte = output_offset[i] + MemSizeCeil(output_data_size[i]);
                memory_size = std::max(memory_size, size_of_byte);
            }
        }
    }
    memory_size -= input_output_size;
    GELOGI("[Mobile] memory_size: %d", memory_size);

    // add main graph
    ge::ComputeGraphPtr mobile_main_graph = std::make_shared<ge::ComputeGraph>(main_graph_name);
    if (mobile_main_graph == nullptr) {
        GELOGE(ge::FAILED, "[Mobile] mobile main subgraph is nullptr.");
        return nullptr;
    }

    // add graph op to main graph
    ge::OpDescPtr graph_op_desc = std::make_shared<ge::OpDesc>(subgraph_name, GRAPH_OP_TYPE);
    if (graph_op_desc == nullptr) {
        GELOGE(ge::FAILED, "[Mobile] graph op des is nullptr.");
        return nullptr;
    }
    for (const auto& data_op_desc: data_op_descs) {
        (void)graph_op_desc->AddInputDesc(data_op_desc->GetOutputDesc(0));
    }
    GE_ASSERT_NOTNULL(netoutput_op_desc, "netoutput op des is nullptr");
    for (uint32_t i = 0; i < netoutput_op_desc->GetInputsSize(); i++) {
        (void)graph_op_desc->AddOutputDesc(netoutput_op_desc->GetInputDesc(i));
        if (netoutput_op_desc->GetOutputsSize() < netoutput_op_desc->GetInputsSize()) {
            GELOGI("[Mobile] netoutput op output size is smaller than input size, use input desc as output desc");
            (void)netoutput_op_desc->AddOutputDesc(netoutput_op_desc->GetInputDesc(i));
        }
    }
    GELOGI("[Mobile] netoutput_op_desc input size: %d", netoutput_op_desc->GetInputsSize());
    GELOGI("[Mobile] netoutput_op_desc output size: %d", netoutput_op_desc->GetOutputsSize());
    const auto ret = SetGraphOpAttrs(graph_op_desc, weights_list, compiled_targets,
        input_output_size, memory_size);
    GE_ASSERT_TRUE(ret == ge::SUCCESS, "[Mobile] set graph op attr failed.");
    auto graph_op_node = mobile_main_graph->AddNode(graph_op_desc);
    (void)ge::NodeUtils::AddSubgraph(*graph_op_node, subgraph_name, mobile_subgraph);

    // add data and netoutput to main graph and link
    int32_t graph_op_input_idx = 0;
    for (const auto& data_op_desc: data_op_descs) {
        auto data_node = mobile_main_graph->AddNode(data_op_desc);
        if (data_node == nullptr) {
            GELOGE(ge::FAILED, "[Mobile] data node is nullptr.");
            return nullptr;
        }
        (void)ge::GraphUtils::AddEdge(
            data_node->GetOutDataAnchor(0),
            graph_op_node->GetInDataAnchor(graph_op_input_idx++));
    }
    auto netoutput_node = mobile_main_graph->AddNode(netoutput_op_desc);
    if (netoutput_node == nullptr) {
        GELOGE(ge::FAILED, "[Mobile] data node is nullptr.");
        return nullptr;
    }
    for (size_t i = 0; i < netoutput_op_desc->GetInputsSize(); i++) {
        (void)ge::GraphUtils::AddEdge(
            graph_op_node->GetOutDataAnchor(static_cast<int32_t>(i)),
            netoutput_node->GetInDataAnchor(static_cast<int32_t>(i)));
    }
    return mobile_main_graph;
}
} // namespace

namespace ge {

Status CompiledModel::GetCompiledTargetsBuffer(std::vector<ge::BaseBuffer>& all_targets_buffer)
{
    compiled_targets_buffer_.clear();
    CompiledTargetSaver saver;
    for (size_t i = 0; i < compiled_targets_.size(); i++) {
        GE_ASSERT_NOTNULL(compiled_targets_[i], "[Mobile] compiled_targets_[%d] is nullptr.", i);
        GE_ASSERT_TRUE(compiled_targets_[i]->UpdateModelModelTaskDef(mobile_model_def_) == ge::SUCCESS,
            "[Mobile] update model task def failed.");
        const size_t task_size = compiled_targets_[i]->GetSize();
        GELOGI("[Mobile] task size: %u", task_size);
        compiled_targets_buffer_.emplace_back();
        compiled_targets_buffer_.back().resize(task_size);
        ge::BaseBuffer buffer(compiled_targets_buffer_.back().data(), task_size);
        const auto ret = saver.SaveToBuffer(compiled_targets_[i], buffer);
        GE_ASSERT_TRUE((ret == SUCCESS) && (buffer.GetData() != nullptr) && (buffer.GetSize() == task_size),
            "[Mobile] compiled target[%d] save to buffer failed.", i);
        (void)all_targets_buffer.emplace_back(std::move(buffer));
    }
    return SUCCESS;
}

Status CompiledModel::SaveToBuffer(
    ge::BaseBuffer& buffer,
    bool save_weights_as_external_data,
    std::map<std::string, ge::BaseBuffer>* weights_list_external)
{
    // convert computer graph and set attr
    ge_model_->SetGraph(ConvertComputeGraphToMobile(
        ge_model_->GetGraph(), weights_list_, compiled_targets_));

    // convert to mobile model def
    const auto model = std::make_unique<ge::Model>(
        ge_model_->GetName(),
        ge_model_->GetPlatformVersion());
    GE_ASSERT_NOTNULL(model, "[Mobile] model is nullptr");
    model->SetGraph(ge_model_->GetGraph());
    model->SetVersion(ge_model_->GetVersion());
    model->SetAttr(ge_model_->MutableAttrMap());
    ge::proto::ModelDef model_def;
    GE_ASSERT_TRUE((model->Save(model_def) == ge::SUCCESS),
        "[Mobile] save to model def failed.");
    GE_ASSERT_TRUE(
        ge::MobileModel::ConvertToMobileModelDef(model_def, mobile_model_def_) == ge::SUCCESS,
        "[Mobile] convert to mobile model def failed.");

    // compiled target buffer
    std::vector<ge::BaseBuffer> all_targets_buffer;
    auto ret = GetCompiledTargetsBuffer(all_targets_buffer);
    GE_ASSERT_TRUE(ret == SUCCESS, "[Mobile] get all targets buffer failed.");

    // save weight external
    if (save_weights_as_external_data && (weights_list_external != nullptr)) {
        // only support graph op num == 1
        GE_ASSERT_TRUE(weights_list_.size() == 1,
            "[Mobile] weight list num %d is not support(only support 1).", weights_list_.size());
        // SubGraph_0.weight
        std::string weight_file_name = std::string("SubGraph_0") + std::string(".weight");
        (void)weights_list_external->insert(std::make_pair(weight_file_name, weights_list_[0]));
        weights_list_.clear();
    }

    // not support save weight info
    ge::BaseBuffer weights_info_buffer;
    (void)weights_info_buffer;

    // save to tlv buffer
    ret = saver_.SaveCompiledModelToBuffer(
        ge_model_,
        mobile_model_def_,
        weights_list_,
        all_targets_buffer,
        weights_info_buffer,
        buffer);
    GE_ASSERT_TRUE(ret == SUCCESS, "[Mobile] save compiled model to buffer failed.");
    return SUCCESS;
}

} // namespace ge
