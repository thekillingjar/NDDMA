/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <utility>

#ifndef LLT_FUSION_ENGINE_ST_TESTCASE_FE_TEST_UTILS_H_
#define LLT_FUSION_ENGINE_ST_TESTCASE_FE_TEST_UTILS_H_

#include <memory>
#include <assert.h>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#define protected public
#define private public
#include "graph/debug/ge_attr_define.h"
#include "graph/op_desc.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/op_desc_utils_ex.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/tensor_utils.h"
#undef private
#undef protected
#include "graph/utils/graph_utils.h"
#define private public
#define protected public
#include "graph/utils/node_utils.h"
#include "graph/compute_graph.h"
#include "graph/node.h"
#include "graph/ge_attr_value.h"
#include "graph/buffer.h"
#undef protected
#undef private

//using namespace domi;

class FeTestOpUtils{
public:
    static ge::NodePtr GenNodeFromOpDesc(ge::OpDescPtr op_desc){
        if(!op_desc){
            return nullptr;
        }

        //return std::make_shared<ge::Node>(op_desc, nullptr);
        auto g=std::make_shared<ge::ComputeGraph>("g");
        return g->AddNode(std::move(op_desc));
    }

    static  void InitDefaultTensorDesc(ge::GeTensorDesc& tensor_desc){
//        ge::TensorUtils::SetSize(tensor_desc, 0);
//        ge::TensorUtils::SetInputTensor(tensor_desc, false);
//        ge::TensorUtils::SetOutputTensor(tensor_desc, false);
//        ge::TensorUtils::SetReuseInput(tensor_desc, false);
//        ge::TensorUtils::SetDeviceType(tensor_desc, ge::NPU);
//        ge::TensorUtils::SetRealDimCnt(tensor_desc, 0);
//        ge::TensorUtils::SetReuseInputIndex(tensor_desc, 0);
    }
    static void AddInputDesc(ge::OpDescPtr op_desc, vector<int64_t> shape, ge::Format format, ge::DataType data_type,
                             int64_t data_size = 0){
        ge::GeTensorDesc tensor_desc(ge::GeShape(shape), format, data_type);
        InitDefaultTensorDesc(tensor_desc);
        ge::TensorUtils::SetSize(tensor_desc, data_size);
        op_desc->AddInputDesc(tensor_desc);
    }
    static void AddOutputDesc(ge::OpDescPtr op_desc, vector<int64_t> shape, ge::Format format, ge::DataType data_type,
                              int64_t data_size = 0){
        ge::GeTensorDesc tensor_desc(ge::GeShape(shape), format, data_type);
        InitDefaultTensorDesc(tensor_desc);
        ge::TensorUtils::SetSize(tensor_desc, data_size);
        op_desc->AddOutputDesc(tensor_desc);
    }
    static void AddWeight(ge::NodePtr node_ptr,
            uint8_t* data, size_t data_len, vector<int64_t> shape = {},
            ge::Format format = ge::FORMAT_NCHW, ge::DataType data_type = ge::DT_FLOAT){

        ge::GeTensorDesc tensor_desc(ge::GeShape(shape), format, data_type);
//        ge::TensorUtils::SetCmpsSize(tensor_desc, 0);
//        ge::TensorUtils::SetDataOffset(tensor_desc, 0);
//        ge::TensorUtils::SetCmpsTab(tensor_desc, nullptr, 0);
//        ge::TensorUtils::SetCmpsTabOffset(tensor_desc, 0);

        vector<ge::GeTensorPtr> weigths = ge::OpDescUtils::MutableWeights(node_ptr);
        weigths.push_back(std::make_shared<ge::GeTensor>(tensor_desc, data, data_len));
        ge::OpDescUtils::SetWeights(node_ptr, weigths);
    }
    static ge::OpDescPtr CreateOpDesc(){
        auto op_desc = std::make_shared<ge::OpDesc>();
        return op_desc;
    }
};

class FeTestOpDescBuilder{
public:
    FeTestOpDescBuilder(ge::OpDescPtr org_op_desc = nullptr):org_op_desc_(org_op_desc){
        if(org_op_desc_){
            stream_id_ = org_op_desc_->GetStreamId();
        }
    }

    FeTestOpDescBuilder& SetStreamId(int64_t streamId) {
        stream_id_ = streamId;
        return *this;
    }
    FeTestOpDescBuilder& SetWorkspace(vector<int64_t> workspace) {
        workspace_ = workspace;
        return *this;
    }
    FeTestOpDescBuilder& SetWorkspaceBytes(vector<int64_t> workspace_bytes) {
        workspace_bytes_ = workspace_bytes;
        return *this;
    }
    FeTestOpDescBuilder& SetType(const string& type) {
        type_ = type;
        return *this;
    }
    FeTestOpDescBuilder& SetName(const string& name) {
        name_ = name;
        return *this;
    }
    FeTestOpDescBuilder& SetInputs(vector<int64_t> inputs) {
        inputs_data_offeset_ = inputs;
        return *this;
    }
    FeTestOpDescBuilder& AddInput(int64_t input) {
        inputs_data_offeset_.push_back(input);
        return *this;
    }
    FeTestOpDescBuilder& SetOutputs(vector<int64_t> outputs) {
        outputs_data_offeset_ = outputs;
        return *this;
    }
    FeTestOpDescBuilder& AddOutput(int64_t output) {
        outputs_data_offeset_.push_back(output);
        return *this;
    }

    ge::GeTensorDesc& AddInputDesc(vector<int64_t> shape, ge::Format format, ge::DataType data_type, int64_t data_size = 0,
                                   std::string name = "") {
        ge::GeTensorDesc tensor_desc(ge::GeShape(shape), format, data_type);
        FeTestOpUtils::InitDefaultTensorDesc(tensor_desc);
        ge::TensorUtils::SetSize(tensor_desc, data_size);
        if (!name.empty()) {
          input_names.emplace_back(name);
        }
        input_tensor_descs.push_back(tensor_desc);
        return input_tensor_descs.back();
    }
    ge::GeTensorDesc& AddOutputDesc(vector<int64_t> shape, ge::Format format, ge::DataType data_type, int64_t data_size = 0,
                                    std::string name = "") {
        ge::GeTensorDesc tensor_desc(ge::GeShape(shape), format, data_type);
        FeTestOpUtils::InitDefaultTensorDesc(tensor_desc);
        ge::TensorUtils::SetSize(tensor_desc, data_size);
        if (!name.empty()) {
          output_names.emplace_back(name);
        }
        output_tensor_descs.push_back(tensor_desc);
        return output_tensor_descs.back();
    }
    ge::GeTensorPtr AddWeight(uint8_t* data, size_t data_len, vector<int64_t> shape = {},
            ge::Format format = ge::FORMAT_NCHW, ge::DataType data_type = ge::DT_FLOAT){

        ge::GeTensorDesc tensor_desc(ge::GeShape(shape), format, data_type);
//        ge::TensorUtils::SetCmpsSize(tensor_desc, 0);
//        ge::TensorUtils::SetDataOffset(tensor_desc, 0);
//        ge::TensorUtils::SetCmpsTab(tensor_desc, nullptr, 0);
//        ge::TensorUtils::SetCmpsTabOffset(tensor_desc, 0);

        weights_.emplace_back(std::make_shared<ge::GeTensor>(tensor_desc, data, data_len));
        return weights_.back();
    }
    ge::NodePtr Finish(){
        ge::OpDescPtr op_desc;
        if(org_op_desc_){
            op_desc = org_op_desc_;
        }
        else{
            op_desc = FeTestOpUtils::CreateOpDesc();//std::make_shared<ge::OpDesc>(name_, type_);
        }
        if(!type_.empty()){
            ge::OpDescUtilsEx::SetType(op_desc, type_);
        }
        if(!name_.empty()){
            op_desc->SetName(name_);
        }

        op_desc->SetStreamId(stream_id_);
        ge::AttrUtils::SetInt(op_desc, "id", 1);
        //ge::AttrUtils::SetInt(op_desc, ATTR_NAME_STREAM_ID, stream_id_);
        //if(!inputs_data_offeset_.empty())
        {
            vector<int64_t> inputs;
            inputs = op_desc->GetInputOffset();
            inputs.insert(inputs.end(), inputs_data_offeset_.begin(), inputs_data_offeset_.end());

            op_desc->SetInputOffset(inputs);
        }
        //if(!outputs_data_offeset_.empty())
        {
            vector<int64_t> outputs;
            outputs = op_desc->GetOutputOffset();
            outputs.insert(outputs.end(), outputs_data_offeset_.begin(), outputs_data_offeset_.end());

            op_desc->SetOutputOffset(outputs);
        }
        //if(!workspace_.empty())
        {
            vector<int64_t> workspace = op_desc->GetWorkspace();
            workspace.insert(workspace.end(), workspace_.begin(), workspace_.end());

            op_desc->SetWorkspace(workspace);
        }
        //if(!workspace_bytes_.empty())
        {
            vector<int64_t> workspace_bytes;
            workspace_bytes = op_desc->GetWorkspaceBytes();
            workspace_bytes.insert(workspace_bytes.end(), workspace_bytes_.begin(), workspace_bytes_.end());

            op_desc->SetWorkspaceBytes(workspace_bytes);
        }
        if (input_names.size() == input_tensor_descs.size()) {
          for(size_t i = 0; i < input_names.size(); ++i){
            op_desc->AddInputDesc(input_names[i], input_tensor_descs[i]);
          }
        } else {
          for(auto& tensor_desc : input_tensor_descs){
            op_desc->AddInputDesc(tensor_desc);
          }
        }
        if (output_names.size() == output_tensor_descs.size()) {
          for(size_t i = 0; i < output_names.size(); ++i){
            op_desc->AddOutputDesc(output_names[i], output_tensor_descs[i]);
          }
        } else {
          for(auto& tensor_desc : output_tensor_descs){
            op_desc->AddOutputDesc(tensor_desc);
          }
        }

        static std::shared_ptr<ge::ComputeGraph> graph;
        // clear graph
        graph = std::make_shared<ge::ComputeGraph>("g");

        ge::NodePtr node_op = graph->AddNode(op_desc);
        //for(int i=0; i < input_tensor_descs.size(); i++)
        for(int i=0; i < op_desc->GetInputsSize(); i++)
        {
            ge::OpDescPtr src_op_desc = std::make_shared<ge::OpDesc>();

            ge::GeTensorDesc src_out_desc;
            src_op_desc->AddOutputDesc(src_out_desc);

            ge::NodePtr src_node = graph->AddNode(src_op_desc);
            assert (src_node != nullptr);

            ge::graphStatus res = ge::GraphUtils::AddEdge(src_node->GetOutDataAnchor(0), node_op->GetInDataAnchor(i));
            assert (res == ge::GRAPH_SUCCESS);

            //ge::NodePtr src_node = node->GetOwnerComputeGraph()->AddNodeFront(src_op_desc);
            //node->AddLinkFrom(src_node);
        }

        {
            vector<ge::GeTensorPtr> weights;
            weights = ge::OpDescUtils::MutableWeights(node_op);
            weights.insert(weights.end(), weights_.begin(), weights_.end());

            ge::OpDescUtils::SetWeights(node_op, weights);
        }

        UpdateAnchorStatus(node_op);

        *this = FeTestOpDescBuilder(op_desc); // clear up

        return node_op;
    }

  ge::NodePtr Finish2(){
    ge::OpDescPtr op_desc;
    if(org_op_desc_){
      op_desc = org_op_desc_;
    }
    else{
      op_desc = FeTestOpUtils::CreateOpDesc();//std::make_shared<ge::OpDesc>(name_, type_);
    }
    if(!type_.empty()){
      ge::OpDescUtilsEx::SetType(op_desc, type_);
    }
    if(!name_.empty()){
      op_desc->SetName(name_);
    }

    op_desc->SetStreamId(stream_id_);
    ge::AttrUtils::SetInt(op_desc, "id", 1);
    //ge::AttrUtils::SetInt(op_desc, ATTR_NAME_STREAM_ID, stream_id_);
    //if(!inputs_data_offeset_.empty())
    {
      vector<int64_t> inputs;
      inputs = op_desc->GetInputOffset();
      inputs.insert(inputs.end(), inputs_data_offeset_.begin(), inputs_data_offeset_.end());

      op_desc->SetInputOffset(inputs);
    }
    //if(!outputs_data_offeset_.empty())
    {
      vector<int64_t> outputs;
      outputs = op_desc->GetOutputOffset();
      outputs.insert(outputs.end(), outputs_data_offeset_.begin(), outputs_data_offeset_.end());

      op_desc->SetOutputOffset(outputs);
    }
    //if(!workspace_.empty())
    {
      vector<int64_t> workspace = op_desc->GetWorkspace();
      workspace.insert(workspace.end(), workspace_.begin(), workspace_.end());

      op_desc->SetWorkspace(workspace);
    }
    //if(!workspace_bytes_.empty())
    {
      vector<int64_t> workspace_bytes;
      workspace_bytes = op_desc->GetWorkspaceBytes();
      workspace_bytes.insert(workspace_bytes.end(), workspace_bytes_.begin(), workspace_bytes_.end());

      op_desc->SetWorkspaceBytes(workspace_bytes);
    }
    for(auto& tensor_desc : input_tensor_descs){
      op_desc->AddInputDesc(tensor_desc);
    }
    for(auto& tensor_desc : output_tensor_descs){
      op_desc->AddOutputDesc(tensor_desc);
    }

    static std::shared_ptr<ge::ComputeGraph> graph;
    // clear graph
    graph = std::make_shared<ge::ComputeGraph>("g");

    ge::NodePtr node_op = graph->AddNode(op_desc);
    //for(int i=0; i < input_tensor_descs.size(); i++)
    for(int i=0; i < op_desc->GetInputsSize(); i++)
    {
      if(i==0) {
        ge::OpDescPtr src_op_desc = std::make_shared<ge::OpDesc>();

        ge::GeTensorDesc src_out_desc;
        src_op_desc->AddOutputDesc(src_out_desc);
        ge::OpDescUtilsEx::SetType(src_op_desc, "Const");

        ge::NodePtr src_node = graph->AddNode(src_op_desc);
        assert (src_node != nullptr);

        ge::graphStatus res = ge::GraphUtils::AddEdge(src_node->GetOutDataAnchor(0), node_op->GetInDataAnchor(i));
        assert (res == ge::GRAPH_SUCCESS);
      } else {
        ge::OpDescPtr src_op_desc = std::make_shared<ge::OpDesc>();

        ge::GeTensorDesc src_out_desc;
        src_op_desc->AddOutputDesc(src_out_desc);

        ge::NodePtr src_node = graph->AddNode(src_op_desc);
        assert (src_node != nullptr);

        ge::graphStatus res = ge::GraphUtils::AddEdge(src_node->GetOutDataAnchor(0), node_op->GetInDataAnchor(i));
        assert (res == ge::GRAPH_SUCCESS);
      }

      //ge::NodePtr src_node = node->GetOwnerComputeGraph()->AddNodeFront(src_op_desc);
      //node->AddLinkFrom(src_node);
    }

    {
      vector<ge::GeTensorPtr> weights;
      weights = ge::OpDescUtils::MutableWeights(node_op);
      weights.insert(weights.end(), weights_.begin(), weights_.end());

      ge::OpDescUtils::SetWeights(node_op, weights);
    }

    UpdateAnchorStatus(node_op);

    *this = FeTestOpDescBuilder(op_desc); // clear up

    return node_op;
  }
  
    void UpdateAnchorStatus(ge::NodePtr node)
    {
        ge::NodeUtils::SetAllAnchorStatus(node);

        for (auto anchor : node->GetAllInDataAnchors()) {
            auto peer_anchor = anchor->GetPeerOutAnchor();
            if (peer_anchor == nullptr) {
                assert(ge::AnchorUtils::SetStatus(anchor, ge::ANCHOR_SUSPEND) == ge::GRAPH_SUCCESS);
            } else if (peer_anchor->GetOwnerNode()->GetType() == "Const") {
                assert(ge::AnchorUtils::SetStatus(anchor, ge::ANCHOR_CONST) == ge::GRAPH_SUCCESS);
            } else {
                assert(ge::AnchorUtils::SetStatus(anchor, ge::ANCHOR_DATA) == ge::GRAPH_SUCCESS);
            }
        }
    }

private:
    ge::OpDescPtr org_op_desc_;
    int64_t stream_id_ = 0;
    string type_;
    string name_;
    vector<int64_t> inputs_data_offeset_;// input
    vector<int64_t> outputs_data_offeset_;// output
    vector<std::string> input_names;
    vector<ge::GeTensorDesc> input_tensor_descs;
    vector<std::string> output_names;
    vector<ge::GeTensorDesc> output_tensor_descs;
    vector<int64_t> workspace_;
    vector<int64_t> workspace_bytes_;
    vector<ge::GeTensorPtr> weights_;

    //std::shared_ptr<ge::ComputeGraph> graph_;
};

#endif  // LLT_FUSION_ENGINE_ST_TESTCASE_FE_TEST_UTILS_H_
