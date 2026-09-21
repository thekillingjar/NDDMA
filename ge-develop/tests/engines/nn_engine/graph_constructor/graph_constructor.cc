/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_constructor.h"
#include "ge/ge_api_types.h"
#include "graph/op_kernel_bin.h"
#include "graph_optimizer/op_judge/format_and_dtype/op_format_dtype_judge.h"
#include "graph/utils/op_desc_utils_ex.h"
using namespace std;

namespace fe {
using OpFormatDtypeJudgePtr = std::shared_ptr<fe::OpFormatDtypeJudge>;

static bool IsSignedInteger(string &str) {
  if (str.empty()) {
    return false;
  }

  for (size_t i = 0; i < str.size(); i++) {
    if (i == 0 && str.at(i) == '-') {
      if (str.length() == 1) {
        return false;
      } else {
        continue;
      }
    }
    if (!isdigit(str.at(i))) {
      return false;
    }
  }
  return true;
}

#define LOG_AND_RETURN(condition, ...)                                         \
  do {                                                                         \
    if (condition) {                                                           \
      FE_LOGE(__VA_ARGS__);                                                    \
      return *this;                                                            \
    }                                                                          \
  } while (0)

#define IS_INPUT_TO_STRING(is_input) (is_input ? "input" : "output")
void UpdateTensorDesc(const ge::GeTensorDesc &tensor,
                      const ge::OpDescPtr &op_desc_ptr, const uint32_t &index,
                      const bool &is_input) {
  if (is_input) {
    op_desc_ptr->UpdateInputDesc(index, tensor);
  } else {
    op_desc_ptr->UpdateOutputDesc(index, tensor);
  }
}

ge::GeTensorDesc GetTensorDesc(const ge::OpDescPtr &op_desc_ptr,
                               const uint32_t &index, bool is_input) {
  ge::GeTensorDesc tensor;
  if (is_input) {
    tensor = op_desc_ptr->GetInputDesc(index);
  } else {
    tensor = op_desc_ptr->GetOutputDesc(index);
  }
  return tensor;
}

void SetTensorDescIntAttr(const ge::OpDescPtr &op_desc_ptr,
                               const uint32_t &index, bool is_input,
                               const std::string &attr, const int64_t &attr_value) {
  ge::GeTensorDesc tensor;
  if (is_input) {
    tensor = op_desc_ptr->GetInputDesc(index);
    ge::AttrUtils::SetInt(tensor, attr, attr_value);
    op_desc_ptr->UpdateInputDesc(index, tensor);
  } else {
    tensor = op_desc_ptr->GetOutputDesc(index);
    ge::AttrUtils::SetInt(tensor, attr, attr_value);
    op_desc_ptr->UpdateOutputDesc(index, tensor);
  }
}

GraphConstructor::GraphConstructor(ComputeGraphPtr &graph, const string &name,
                                   const ge::Format &default_format,
                                   const ge::DataType &default_dtype,
                                   const ge::GeShape &default_shape)
    : graph_(graph), graph_name_(name) {
  for (auto node : graph->GetAllNodes()) {
    std::shared_ptr<OpDesc> temp = std::make_shared<OpDesc>();
    temp->node = node;
    temp->type = node->GetType();
    temp->op_name = node->GetName();
    op_map_[node->GetName()] = temp;
    ops_.push_back(temp);
  }
  DEFAULT_FORMAT = default_format;
  DEFAULT_DTYPE = default_dtype;
  DEFAULT_SHAPE = default_shape;
}

GraphConstructor::~GraphConstructor() {}

ge::NodePtr GraphConstructor::GetOp(const string &op_name) {
  auto it = op_map_.find(op_name);
  if (it != op_map_.end()) {
    return it->second->node;
  }
  return nullptr;
}

Status CheckOriginalFormatValid(ge::Format original_format) {
  for (auto &format : FE_ORIGIN_FORMAT_VECTOR) {
    if (original_format == format) {
      return SUCCESS;
    }
  }
  return FAILED;
}
GraphConstructor &GraphConstructor::AddOpDesc(const string &op_name,
                                              const string &op_type,
                                              const size_t &inputs_size,
                                              const size_t &outputs_size) {
  LOG_AND_RETURN(op_name.empty(), "Op name cannot be empty.");

  if (CheckOriginalFormatValid(DEFAULT_FORMAT) != SUCCESS) {
    FE_LOGE("original format %u is invalid", DEFAULT_FORMAT);
    return *this;
  }
  /* 1. Check the dst node is exist or not */
  auto iter = op_map_.find(op_name);
  /* 2. If the node is not in op_map_, we will create a new one */
  if (iter == op_map_.end()) {
    /*  TODO: Create new node with (dst_input_or_output_index+ size of input_names)
     *  inputs */
    ge::OpDescPtr ge_op_desc_ptr = nullptr;
    FE_MAKE_SHARED(ge_op_desc_ptr =
                       std::make_shared<ge::OpDesc>(op_name.c_str(), op_type),
                   ge_op_desc_ptr = nullptr;
                   return *this);

    for (size_t i = 0; i < inputs_size; i++) {
      ge::GeTensorDesc tensor =
          ge::GeTensorDesc(DEFAULT_SHAPE, DEFAULT_FORMAT, DEFAULT_DTYPE);
      SetTensorDescInfo(tensor, DEFAULT_FORMAT, DEFAULT_DTYPE, DEFAULT_SHAPE,
                        DEFAULT_FORMAT);
      ge_op_desc_ptr->AddInputDesc(tensor);
    }
    for (size_t i = 0; i < outputs_size; i++) {
      ge::GeTensorDesc tensor =
          ge::GeTensorDesc(DEFAULT_SHAPE, DEFAULT_FORMAT, DEFAULT_DTYPE);
      SetTensorDescInfo(tensor, DEFAULT_FORMAT, DEFAULT_DTYPE, DEFAULT_SHAPE,
                        DEFAULT_FORMAT);
      ge_op_desc_ptr->AddOutputDesc(tensor);
    }
    ge::AttrUtils::SetInt(ge_op_desc_ptr, "_fe_imply_type", EN_IMPL_HW_TBE);

    auto node = graph_->AddNode(ge_op_desc_ptr);

    std::shared_ptr<OpDesc> op(new (std::nothrow) OpDesc());
    LOG_AND_RETURN(op == nullptr, "new an object failed.");
    op->node = node;
    op->op_name = op_name;
    op->type = op_type;
    ops_.push_back(op);
    op_map_[op_name] = op;
    last_added_node_ = node;
    return *this;
  } else {
    /* The node is already in the graph */
    auto node = iter->second->node;
    ge::OpDescPtr existing_op_desc = node->GetOpDesc();

    if (existing_op_desc->GetInputsSize() < inputs_size) {
      for (size_t i = 0; i < inputs_size - existing_op_desc->GetInputsSize();
           i++) {
        existing_op_desc->AddInputDesc(
            ge::GeTensorDesc(DEFAULT_SHAPE, DEFAULT_FORMAT, DEFAULT_DTYPE));
      }
    }

    if (existing_op_desc->GetOutputsSize() < outputs_size) {
      for (size_t i = 0; i < outputs_size - existing_op_desc->GetOutputsSize();
           i++) {
        existing_op_desc->AddOutputDesc(
            ge::GeTensorDesc(DEFAULT_SHAPE, DEFAULT_FORMAT, DEFAULT_DTYPE));
      }
    }
    return *this;
    last_added_node_ = node;
  }
  return *this;
}

GraphConstructor &GraphConstructor::AddOpDesc(
    const OpImplType &impl_type, const string &pattern, const string &op_name,
    const string &op_type, const size_t &inputs_size, const size_t &outputs_size) {
  AddOpDesc(op_name, op_type, inputs_size, outputs_size);
  if (impl_type == EN_IMPL_CUSTOM_TBE || impl_type == EN_IMPL_HW_TBE) {
    SetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, 0);
  }

  return SetFeImPlType(impl_type).SetPattern(pattern);
}

Status GraphConstructor::SetTensorDescInfo(ge::GeTensorDesc &tensor,
                                           const ge::Format &original_format,
                                           const ge::DataType &data_type,
                                           const ge::GeShape &original_shape,
                                           const ge::Format &currentformat) {
  tensor.SetOriginFormat(original_format);
  tensor.SetOriginShape(original_shape);
  tensor.SetOriginDataType(data_type);
  tensor.SetDataType(data_type);
  tensor.SetFormat(currentformat);
  /* If the shape size if not equal to the standard size of original format,
   * For NDHWC, we will just set the current shape = original shape and return.
   * */

  auto iter = FORMAT_NAME_MAP.find(original_format);
  if (iter != FORMAT_NAME_MAP.end()) {
    if (iter->second != original_shape.GetDimNum()) {
      tensor.SetShape(original_shape);
      /* For NDHWC, if the original shape is less than 5, we will just use
       * the original shape as curent shape instead of transferring*/
      return SUCCESS;
    }
  }

  GC_LOGD("Current and original format is %u and %u, original shape is %s",
          currentformat, original_format,
          StringUtils::IntegerVecToString(original_shape.GetDims()).c_str());

  if (currentformat == original_format) {
    /* We do not need to set the current shape because the current shape is
     * as same as the original shape. */
    tensor.SetShape(original_shape);
    return SUCCESS;
  } else {
    /* For 4D format, we will padding the original shape and get the new shape
     * of current format. */
    ge::GeShape new_shape;
    string reshape_type;
    ge::GeShape origin_shape_afer_pad;
    ExpandDimension(original_format, currentformat, reshape_type, original_shape, origin_shape_afer_pad);

    ShapeAndFormat shape_and_format_info = {
        origin_shape_afer_pad, new_shape, original_format, currentformat, data_type};
    (void)GetShapeAccordingToFormat(shape_and_format_info);
    tensor.SetShape(new_shape);
    tensor.SetFormat(currentformat);
  }
  return SUCCESS;
}

Status GraphConstructor::ReplaceNodeWithNewBode(ge::NodePtr &old_node,
                                                ge::NodePtr &new_node) {
  uint32_t in_anchor_index = 0;
  for (auto &input_anchor : old_node->GetAllInDataAnchors()) {
    if (input_anchor != nullptr && input_anchor->GetPeerOutAnchor() != nullptr) {
      auto peer_out_anchor = input_anchor->GetPeerOutAnchor();
      ge::GraphUtils::RemoveEdge(peer_out_anchor, input_anchor);
      if (new_node->GetAllInDataAnchors().size() > in_anchor_index) {
        ge::GraphUtils::AddEdge(peer_out_anchor,
                                new_node->GetInDataAnchor(in_anchor_index));
      } else {
        FE_LOGE("new node %s does not have enough input anchors. size is %u. "
                "inAnchorIndex is %u. ",
                new_node->GetName().c_str(),
                new_node->GetAllInDataAnchors().size(), in_anchor_index);
        return FAILED;
      }
      in_anchor_index++;
    }
  }

  uint32_t out_anchor_index = 0;
  for (auto &output_anchor : old_node->GetAllOutDataAnchors()) {
    if (output_anchor != nullptr) {
      for (auto &peer_in_data_anchor : output_anchor->GetPeerInDataAnchors()) {
        if (peer_in_data_anchor == nullptr) {
          continue;
        }
        ge::GraphUtils::RemoveEdge(output_anchor, peer_in_data_anchor);
        if (new_node->GetAllOutDataAnchors().size() > out_anchor_index) {
          ge::GraphUtils::AddEdge(new_node->GetOutDataAnchor(out_anchor_index),
                                  peer_in_data_anchor);
        } else {
          FE_LOGE("new node %s does not have enough output anchors.",
                  new_node->GetName().c_str());
          return FAILED;
        }
      }
    }
    out_anchor_index++;
  }
  graph_->RemoveNode(old_node);
  return SUCCESS;
}

int32_t GetMinAvailableInputOrOutputIndex(ge::NodePtr &node, bool is_dst_node) {
  if (is_dst_node) {
    for (auto &ele : node->GetAllInDataAnchors()) {
      if (ele->GetPeerOutAnchor() == nullptr) {
        return ele->GetIdx();
      }
    }
    return node->GetAllInDataAnchors().size();
  } else {
    /* For output, if the user did not designate the output index, */
    return 0;
  }
}

Status GraphConstructor::AddNewNodeIntoGraph(
    const string &op_type, const string &op_real_name,
    const size_t &size_of_new_tensors, const DetailedTensor &tensor_info,
    const bool &is_dst_node, int32_t &tensor_index, ge::NodePtr &new_node) {
  /*  Create new node with (dst_input_or_output_index+ size of new tensors)
   *  inputs */
  ge::OpDescPtr ge_op_desc_ptr = nullptr;
  FE_MAKE_SHARED(ge_op_desc_ptr =
                     std::make_shared<ge::OpDesc>(op_real_name.c_str(), op_type),
                 ge_op_desc_ptr = nullptr;
                 return FAILED);
  GC_LOGI("Create new node %s", op_real_name.c_str());
  std::shared_ptr<OpDesc> st_op_desc(new (std::nothrow) OpDesc());
  if (tensor_index == -1) {
    GC_LOGI("no need to add control tensor desc", op_real_name.c_str());
  } else {
    tensor_index = tensor_index == 0xFFFF ? 0 : tensor_index;
    for (size_t i = 0; i < (size_of_new_tensors + tensor_index); i++) {

      ge::GeTensorDesc tensor = ge::GeTensorDesc(
          tensor_info.original_shape_, tensor_info.format_, tensor_info.data_type_);
      SetTensorDescInfo(tensor, tensor_info.original_format_, tensor_info.data_type_,
                        tensor_info.original_shape_, tensor_info.format_);
      if (is_dst_node) {
        ge_op_desc_ptr->AddInputDesc(tensor);
      } else {
        ge_op_desc_ptr->AddOutputDesc(tensor);
      }
    }
  }

  new_node = graph_->AddNode(ge_op_desc_ptr);
  st_op_desc->op_name = op_real_name;
  st_op_desc->node = new_node;
  st_op_desc->type = op_type;
  op_map_.emplace(op_real_name, st_op_desc);
  ops_.emplace_back(st_op_desc);
  return SUCCESS;
}

Status GraphConstructor::AddTensorIntoExistingNodes(
    const size_t &size_of_new_tensors, const DetailedTensor &tensor_info,
    const bool &is_dst_node, map<string, std::shared_ptr<OpDesc>>::iterator &iter,
    int32_t &input_index) {
  /* The node is already in the graph */
  ge::NodePtr node = iter->second->node;
  ge::OpDescPtr existing_op_desc = node->GetOpDesc();
  GC_LOGI("Node %s already exists.", existing_op_desc->GetName().c_str());
  size_t current_tensor_size;
  if (is_dst_node) {
    current_tensor_size = existing_op_desc->GetInputsSize();
  } else {
    current_tensor_size = existing_op_desc->GetOutputsSize();
  }
  if (size_of_new_tensors == 0) {
    FE_LOGE("Size of new tensors is zero.");
    return FAILED;
  }

  input_index = input_index == 0xFFFF
                   ? GetMinAvailableInputOrOutputIndex(node, is_dst_node)
                   : input_index;

  GC_LOGD("The first avalable %s index of %s is %d",
          IS_INPUT_TO_STRING(is_dst_node), existing_op_desc->GetName().c_str(),
          input_index);
  if (input_index == -1) {
    GC_LOGD("Do not update node when adding control edges for node %s.",
            existing_op_desc->GetName().c_str());
    return SUCCESS;
  }
  size_t max_tensor_index = (size_t)input_index + size_of_new_tensors - 1;
  ge::OpDescPtr new_opdesc_ptr = existing_op_desc;

  /* That means we need to create more tensor desc for this op. */
  if (max_tensor_index >= current_tensor_size) {
    for (size_t i = 0; i <= max_tensor_index; i++) {
      /* Input index is the first available index of inputs of new_opdesc_ptr.
       * And current_tensor_size is the max available index of current inputs.*/
      size_t mini_index_using_current_tensor_desc =
          (current_tensor_size) < input_index ? (current_tensor_size) : input_index;
      ge::GeTensorDesc new_tensor_desc;
      new_tensor_desc =
          ge::GeTensorDesc(tensor_info.original_shape_, tensor_info.format_,
                           tensor_info.data_type_);
      SetTensorDescInfo(new_tensor_desc, tensor_info.original_format_,
                        tensor_info.data_type_, tensor_info.original_shape_,
                        tensor_info.format_);
      /* We need to add new tensors from either the minimum of input_index and
       * current_tensor max index.
       * If i < minimum of input_index and current_tensor max index, we just
       * use the old tensor and if it's larger or equal than it,
       * we add new tensors. */
      if (i >= mini_index_using_current_tensor_desc) {
        if (i < current_tensor_size) {
          /* Update existing tensor */
          new_opdesc_ptr->UpdateInputDesc(i, new_tensor_desc);
        } else {
          /* Add new tensor */
          if (is_dst_node) {
            new_opdesc_ptr->AddInputDesc(new_tensor_desc);
          } else {
            new_opdesc_ptr->AddOutputDesc(new_tensor_desc);
          }
        }
      }
    }
    /* Create a new node and use it to replace the old one.
     * Reason: We can not add anchor into old node. */
    auto new_node = graph_->AddNode(new_opdesc_ptr);
    if (ReplaceNodeWithNewBode(node, new_node) != SUCCESS) {
      return FAILED;
    }
    GC_LOGI("Replace node %s with a new one, new %s anchor size is %zu",
            node->GetName().c_str(), IS_INPUT_TO_STRING(is_dst_node),
            max_tensor_index + 1);
    iter->second->node = new_node;
  } else {
    /* If the size_of_new_tensors is 1, we will update the existing tensor desc.
     * Otherwise, keep the original tensor desc.
     * TODO: Support updating multiple inputs or outputs. */
    if (size_of_new_tensors == 1) {
      ge::GeTensorDesc existing_tensor;
      existing_tensor = GetTensorDesc(existing_op_desc, input_index, is_dst_node);

      GC_LOGD("Set existing %s tensor %u's info for node %s, size is %u",
              IS_INPUT_TO_STRING(is_dst_node), input_index,
              existing_op_desc->GetName().c_str(), current_tensor_size);
      SetTensorDescInfo(existing_tensor, tensor_info.original_format_,
                        tensor_info.data_type_, tensor_info.original_shape_,
                        tensor_info.format_);
      UpdateTensorDesc(existing_tensor, existing_op_desc, input_index, is_dst_node);
    }
  }
  return SUCCESS;
}

Status GraphConstructor::ParseNodeNameAndAddNodeIntoGraph(
    const DetailedTensor &tensor_info, const size_t &size_of_new_tensors,
    bool is_dst_node, vector<ConnectionInfo> &connection_info_of_all_nodes) {
  string op_type;
  string op_real_name;
  int32_t input_index = 0;

  Status ret = NodeNameParser(tensor_info.name_, op_type, op_real_name, input_index);
  if (ret != SUCCESS) {
    FE_LOGE("Failed to parse node %s.", tensor_info.name_.c_str());
    return FAILED;
  }
  GC_LOGD("For name [%s], optype is %s, op real name is %s, index is %d",
          tensor_info.name_.c_str(), op_type.c_str(), op_real_name.c_str(),
          input_index);
  /* 1. Check the dst node is exist or not */
  auto iter = op_map_.find(op_real_name);
  /* 2. If the node is not in op_map_, we will create a new one */
  ge::NodePtr new_node;
  if (iter == op_map_.end()) {
    ret = AddNewNodeIntoGraph(op_type, op_real_name, size_of_new_tensors,
                              tensor_info, is_dst_node, input_index, new_node);
    if (ret != SUCCESS) {
      return ret;
    }
  } else {
    ret = AddTensorIntoExistingNodes(size_of_new_tensors, tensor_info,
                                     is_dst_node, iter, input_index);
    new_node = iter->second->node;
    if (ret != SUCCESS) {
      return ret;
    }
  }
  /* 3. Fill the connection_info_of_all_nodes */
  /* node is substituted with new node in function ReplaceNodeWithNewBode */
  struct ConnectionInfo connect_temp = {op_real_name, op_type, new_node, input_index};
  connection_info_of_all_nodes.emplace_back(connect_temp);
  return SUCCESS;
}

Status GraphConstructor::AddEdges(
    const vector<ConnectionInfo> &src_connection_info_of_all_nodes,
    const vector<ConnectionInfo> &dst_connection_info_of_all_nodes) {
  for (auto &dst_node : dst_connection_info_of_all_nodes) {
    FE_LOGD("dst op name %s type %s, tensor index %d", dst_node.op_name.c_str(), dst_node.type.c_str(),
            dst_node.starting_tensor_index);
    int32_t input_count = 0;
    for (auto &src_node : src_connection_info_of_all_nodes) {
      FE_LOGD("src op name %s type %s, tensor index %d", src_node.op_name.c_str(), src_node.type.c_str(),
              src_node.starting_tensor_index);
      if (dst_node.op_name == src_node.op_name) {
        FE_LOGE("Cannot create self loop edge for node %s %d->%d",
                src_node.op_name.c_str(), src_node.starting_tensor_index,
                dst_node.starting_tensor_index);
        continue;
      }

      if ((src_node.starting_tensor_index == -1 && dst_node.starting_tensor_index != -1) ||
          (src_node.starting_tensor_index != -1 && dst_node.starting_tensor_index == -1)) {
        FE_LOGE("Cannot create control edge for node %s %d->%d",
                src_node.op_name.c_str(), src_node.starting_tensor_index,
                dst_node.starting_tensor_index);
        continue;
      } else if (src_node.starting_tensor_index == -1 && dst_node.starting_tensor_index == -1) {
        auto out_ctrl_anchor = src_node.node->GetOutControlAnchor();
        auto in_ctrl_anchor = dst_node.node->GetInControlAnchor();
        ge::GraphUtils::AddEdge(out_ctrl_anchor, in_ctrl_anchor);
        continue;
      }
      size_t tensor_index = (size_t)(dst_node.starting_tensor_index + input_count);
      auto ge_dst_op_desc_ptr = dst_node.node->GetOpDesc();
      if (ge_dst_op_desc_ptr->GetInputsSize() <= tensor_index) {
        FE_LOGE("The tensor index %d of %s is larger than its inputs size %u",
                tensor_index, ge_dst_op_desc_ptr->GetName().c_str(),
                ge_dst_op_desc_ptr->GetInputsSize());
        continue;
      }

      auto in_anchor = dst_node.node->GetInDataAnchor(tensor_index);
      auto output_anchor = in_anchor->GetPeerOutAnchor();
      if (output_anchor != nullptr) {
        /* The input anchor is not occupied and we remove the existing
         * edges first. */
        if (ge::GraphUtils::RemoveEdge(output_anchor, in_anchor) !=
            ge::GRAPH_SUCCESS) {
          FE_LOGE("[1]:Failed to remove edge from [%s]: %d to [%s] : %d.",
                  output_anchor->GetOwnerNode()->GetName().c_str(),
                  output_anchor->GetIdx(), dst_node.node->GetName().c_str(),
                  tensor_index);
          return FAILED;
        }
        GC_LOGI("[1]:SuccessFully remove edge from [%s]: %d to [%s] : %d.",
                output_anchor->GetOwnerNode()->GetName().c_str(),
                output_anchor->GetIdx(), dst_node.node->GetName().c_str(),
                tensor_index);
      }

      /* The input anchor is not used yet or the existing edge is removed,
       * we just add an edge between it and the src_node's out anchor */
      output_anchor =
          src_node.node->GetOutDataAnchor(src_node.starting_tensor_index);
      if (output_anchor == nullptr) {
        FE_LOGE("Out put anchor %d of src node %s is nullptr",
                src_node.starting_tensor_index, src_node.op_name.c_str());
        return FAILED;
      }

      if (ge::GraphUtils::AddEdge(output_anchor, in_anchor) !=
          ge::GRAPH_SUCCESS) {
        FE_LOGE("[2]:Failed to Add edge from [%s]: %d to [%s] : %d.",
                src_node.node->GetName().c_str(), output_anchor->GetIdx(),
                dst_node.node->GetName().c_str(), tensor_index);
        return FAILED;
      }
      input_count++;
    }
  }
  return SUCCESS;
}

string GraphConstructor::GetInputString(const ge::NodePtr &node) {
  string input_node_name_string = "{";
  for (const auto &ele : node->GetAllInDataAnchors()) {
    if (ele->GetPeerOutAnchor() == nullptr ||
        ele->GetPeerOutAnchor()->GetOwnerNode() == nullptr) {
      input_node_name_string += "[], ";
    } else {
      input_node_name_string +=
          ("[" + ele->GetPeerOutAnchor()->GetOwnerNode()->GetName() + "], ");
    }
  }
  for (const auto &node : node->GetInControlNodes()) {
    input_node_name_string +=
        ("[" + node->GetName() + ":-1], ");
  }
  input_node_name_string += "}";
  return input_node_name_string;
}

Status GraphConstructor::DumpGraph(const ge::ComputeGraphPtr &graph) {
  if (graph == nullptr) {
    FE_LOGE("graph %s is nullptr.", graph->GetName().c_str());
    return FAILED;
  }

  if (ge::GRAPH_SUCCESS != graph->TopologicalSorting()) {
    FE_LOGE("TopologicalSorting failed!");
    return FAILED;
  }

  for (const auto &node : graph->GetDirectNode()) {
    string input_node_name_list = GetInputString(node);
    GC_LOGI("Node named: [%s], input List is %s", node->GetName().c_str(),
            input_node_name_list.c_str());
  }
  return SUCCESS;
}

/******************************************************************************/
/*******************The following is the SetInputs function********************/
GraphConstructor &
GraphConstructor::SetInputs(const DetailedTensor &dst_tensor,
                            const vector<DetailedTensor> &multiple_src_tensors) {
  if (CheckOriginalFormatValid(dst_tensor.original_format_) != SUCCESS) {
    FE_LOGE("original format %u of dst node %s is invalid",
            dst_tensor.original_format_, dst_tensor.name_.c_str());
    return *this;
  }
  for (const auto &src_tensor : multiple_src_tensors) {
    if (CheckOriginalFormatValid(src_tensor.original_format_) != SUCCESS) {
      FE_LOGE("original format %u of src node %s is invalid",
              src_tensor.original_format_, src_tensor.name_.c_str());
      return *this;
    }
  }

  vector<ConnectionInfo> dst_connection_info_of_all_nodes;
  vector<ConnectionInfo> src_connection_info_of_all_nodes;
  GC_LOGD("------------------------------------------------------------------");
  string src_tensor_name;
  for (auto &ele : multiple_src_tensors) {
    src_tensor_name += ele.name_;
    src_tensor_name += ", ";
  }
  GC_LOGD("Start to parse Set inputs of {%s, {%s}}", dst_tensor.name_.c_str(),
          src_tensor_name.c_str());
  auto size_of_source = multiple_src_tensors.size();
  GC_LOGD("Size of source tensors is %u", size_of_source);
  Status ret;
  if (!dst_tensor.name_.empty()) {
    /* 1. Parse the destination node */
    ret = ParseNodeNameAndAddNodeIntoGraph(dst_tensor, size_of_source,
                                           true, /* it's destination node */
                                           dst_connection_info_of_all_nodes);

    if (ret != SUCCESS) {
      return *this;
    }
  }

  /* 2. Parse the souce node */
  for (auto src_tensor : multiple_src_tensors) {
    if (src_tensor.name_.empty()) {
      /* if the src is empty, we will not add edge between empty node to
       * dst node. */
      continue;
    }
    /* Consider all src nodes as only one output and this output gives to
     * multiple users. */
    ret = ParseNodeNameAndAddNodeIntoGraph(src_tensor, 1,
                                           false, /* it's source node */
                                           src_connection_info_of_all_nodes);
    FE_CHECK(ret != SUCCESS, , return *this);
  }

  /* 3. Add Edges between all source nodes and all input nodes.
   * Input node will commonly be only 1 and but there will be multiple input
   * anchors for this input node. */
  AddEdges(src_connection_info_of_all_nodes, dst_connection_info_of_all_nodes);

  GC_LOGD("End of parsing SetInputs of {%s, {%s}}", dst_tensor.name_.c_str(),
          src_tensor_name.c_str());
  GC_LOGD("------------------------------------------------------------------");

  return *this;
}

GraphConstructor &GraphConstructor::SetInputs(const DetailedTensor &dst_tensor,
                                              const DetailedTensor &src_tensor) {
  vector<DetailedTensor> src_tensors = {src_tensor};
  return SetInputs(dst_tensor, src_tensors);
}

GraphConstructor &
GraphConstructor::SetInputs(const string &dst_name,
                            const vector<string> &multiple_src_names) {
  DetailedTensor dst_tensor(dst_name);
  vector<DetailedTensor> multiple_src_tensors;
  for (auto &ele : multiple_src_names) {
    multiple_src_tensors.emplace_back(DetailedTensor(ele));
  }
  return SetInputs(dst_tensor, multiple_src_tensors);
}

GraphConstructor &
GraphConstructor::SetInputs(const vector<string> &multiple_src_names) {
  DetailedTensor dst_tensor(last_added_node_->GetName());
  vector<DetailedTensor> multiple_src_tensors;
  for (auto &ele : multiple_src_names) {
    multiple_src_tensors.emplace_back(DetailedTensor(ele));
  }
  return SetInputs(dst_tensor, multiple_src_tensors);
}
/*******************End of the SetInputs function******************************/
/******************************************************************************/

/******************************************************************************/
/*******************The following is the SetInput function*********************/
GraphConstructor &GraphConstructor::SetInput(const string &dst_name,
                                             const string &src_name) {
  return SetInput(dst_name, src_name, DEFAULT_FORMAT);
}

GraphConstructor &GraphConstructor::SetInput(const string &dst_name,
                                             const string &src_name,
                                             const ge::Format &format) {
  return SetInput(dst_name, src_name, format, DEFAULT_FORMAT);
}

GraphConstructor &GraphConstructor::SetInput(const string &dst_name,
                                             const string &src_name,
                                             const ge::Format &format,
                                             const ge::Format &original_format) {
  return SetInput(dst_name, src_name, format, original_format,
                  DEFAULT_SHAPE.GetDims());
}

GraphConstructor &GraphConstructor::SetInput(
    const string &dst_name, const string &src_name, const ge::Format &format,
    const ge::Format &original_format, const vector<int64_t> &original_dims) {
  GC_LOGD("original dims is %s, format is %u for src %s and dst %s",
          StringUtils::IntegerVecToString(original_dims).c_str(), original_format,
          src_name.c_str(), dst_name.c_str());
  DetailedTensor src_tensor(src_name, format, original_format, DEFAULT_DTYPE,
                           ge::GeShape(original_dims));
  vector<DetailedTensor> src_tensors = {src_tensor};

  DetailedTensor dst_tensor(dst_name, format, original_format, DEFAULT_DTYPE,
                           ge::GeShape(original_dims));

  return SetInputs(dst_tensor, src_tensors);
}

GraphConstructor &GraphConstructor::SetInput(const string &dst_name,
                                             const ge::Format &dst_format,
                                             const string &src_name,
                                             const ge::Format &src_format) {
  return SetInput(dst_name, dst_format, src_name, src_format, DEFAULT_FORMAT,
                  DEFAULT_FORMAT, DEFAULT_SHAPE.GetDims(),
                  DEFAULT_SHAPE.GetDims());
}

GraphConstructor& GraphConstructor::SetInput(const string &dst_name,
                                             const ge::Format &dst_format,
                                             const ge::DataType &dst_dtype,
                                             const string &src_name,
                                             const ge::Format &src_format,
                                             const ge::DataType &src_dtype) {
  DetailedTensor src_tensor(src_name, src_format, DEFAULT_FORMAT,
                           src_dtype, DEFAULT_SHAPE);
  vector<DetailedTensor> src_tensors = {src_tensor};

  DetailedTensor dst_tensor(dst_name, dst_format, DEFAULT_FORMAT,
                           dst_dtype, DEFAULT_SHAPE);

  return SetInputs(dst_tensor, src_tensors);
}

GraphConstructor &
GraphConstructor::SetInput(const string &dst_name, const ge::Format &dst_format,
                           const string &src_name, const ge::Format &src_format,
                           const ge::Format &dst_original_format) {
  return SetInput(dst_name, dst_format, src_name, src_format, dst_original_format,
                  DEFAULT_FORMAT, DEFAULT_SHAPE.GetDims(),
                  DEFAULT_SHAPE.GetDims());
}

GraphConstructor &
GraphConstructor::SetInput(const string &dst_name, const ge::Format &dst_format,
                           const string &src_name, const ge::Format &src_format,
                           const ge::Format &dst_original_format,
                           const ge::Format &src_original_format) {
  return SetInput(dst_name, dst_format, src_name, src_format, dst_original_format,
                  src_original_format, DEFAULT_SHAPE.GetDims(),
                  DEFAULT_SHAPE.GetDims());
}

GraphConstructor &GraphConstructor::SetInput(
    const string &dst_name, const ge::Format &dst_format, const string &src_name,
    const ge::Format &src_format, const ge::Format &dst_original_format,
    const ge::Format &src_original_format, const vector<int64_t> &dst_original_dims,
    const vector<int64_t> &src_original_dims) {
  DetailedTensor src_tensor(src_name, src_format, src_original_format, DEFAULT_DTYPE,
                           ge::GeShape(src_original_dims));
  vector<DetailedTensor> src_tensors = {src_tensor};

  DetailedTensor dst_tensor(dst_name, dst_format, dst_original_format, DEFAULT_DTYPE,
                           ge::GeShape(dst_original_dims));

  return SetInputs(dst_tensor, src_tensors);
}

/** For specific cases, we want to set the input and output format and shape
 * of specific tensor. The following function provides an ability to set the
 * format and shape
 * Param dst_or_src is only for shape*/
GraphConstructor &GraphConstructor::SetInput(
    const string &dst_name, const string &src_name, const vector<int64_t> &dims,
    const uint32_t
        &dst_or_src /* default = SOURCE_AND_DESTINATION, only works for shape*/) {
  ge::GeShape src_shape = DEFAULT_SHAPE;
  ge::GeShape dst_shape = DEFAULT_SHAPE;
  if (dst_or_src == SOURCE || dst_or_src == SOURCE_AND_DESTINATION) {
    src_shape = ge::GeShape(dims);
  }
  if (dst_or_src == DESTINATION || dst_or_src == SOURCE_AND_DESTINATION) {
    dst_shape = ge::GeShape(dims);
  }

  DetailedTensor src_tensor(src_name, DEFAULT_FORMAT, DEFAULT_FORMAT,
                           DEFAULT_DTYPE, src_shape);

  vector<DetailedTensor> src_tensors = {src_tensor};

  DetailedTensor dst_tensor(dst_name, DEFAULT_FORMAT, DEFAULT_FORMAT,
                           DEFAULT_DTYPE, dst_shape);

  return SetInputs(dst_tensor, src_tensors);
}

GraphConstructor &GraphConstructor::SetInput(
    const string &dst_name, const string &src_name, const vector<int64_t> &dims,
    const ge::Format &format,
    const uint32_t
        &dst_or_src /* default = SOURCE_AND_DESTINATION, only works for shape*/) {
  ge::GeShape src_shape = DEFAULT_SHAPE;
  ge::GeShape dst_shape = DEFAULT_SHAPE;
  if (dst_or_src == SOURCE || dst_or_src == SOURCE_AND_DESTINATION) {
    src_shape = ge::GeShape(dims);
  }
  if (dst_or_src == DESTINATION || dst_or_src == SOURCE_AND_DESTINATION) {
    dst_shape = ge::GeShape(dims);
  }

  DetailedTensor src_tensor(src_name, format, DEFAULT_FORMAT, DEFAULT_DTYPE,
                           src_shape);

  vector<DetailedTensor> src_tensors = {src_tensor};

  DetailedTensor dst_tensor(dst_name, format, DEFAULT_FORMAT, DEFAULT_DTYPE,
                           dst_shape);

  return SetInputs(dst_tensor, src_tensors);
}
/*******************End of the SetInput function****************************/
/******************************************************************************/

Status ParseWhenBothUnderlineAndColonExists(
    const string &name, string &op_type, string &op_real_name,
    int32_t &input_or_output_index, const size_t &underline_position,
    const size_t &relative_colon_position, const string &str_behind_underline,
    const string &str_in_front_of_underline) {
  /* relative colon position is the last colon position in string
   * behind underline, so the absolute position should be the relative one
   * combined with the underlineposition */
  size_t absolute_colon_position = relative_colon_position + underline_position + 1;
  /* Type:_1 this case is Invalid */
  GC_LOGD("underline and colon exist for %s", name.c_str());
  auto length_of_name = name.length();
  if (absolute_colon_position < underline_position) {
    FE_LOGE("The absoulte colon_position %u should not be less than the "
            "underline position %u",
            absolute_colon_position, underline_position);
    return FAILED;
  }
  /* Both underline and colon exist */
  /* No thing is behind colon, we just get the number behind underline */
  string string_between_underline_and_colon =
      str_behind_underline.substr(0, relative_colon_position);
  GC_LOGD("stringBetweenUnderlineAndColon is %s",
          string_between_underline_and_colon.c_str());
  if (StringUtils::IsInteger(string_between_underline_and_colon) ||
      string_between_underline_and_colon.empty()) {
    /* In this case, the name is like "MatMul_5:xxx or MatMul_:xxx" */
    op_real_name =
        string_between_underline_and_colon.empty()
            ? str_in_front_of_underline
            : str_in_front_of_underline + "_" + string_between_underline_and_colon;
    op_type = str_in_front_of_underline;
    if (absolute_colon_position + 1 >= length_of_name) {
      /* In this branch the name is like "MatMul_5:".
       * Nothing is behind colon, we assume the input_or_output_index is the
       * first available anchor index. Available is defined as the first anchor
       * which does not have the peer out anchor. */
      input_or_output_index = 0xFFFF;
      return SUCCESS;
    }
    auto tensor_index = name.substr(absolute_colon_position + 1);
    if (IsSignedInteger(tensor_index)) {
      /* In this branch the name is quit meet the requirements.
       * It looks like "MatMul_5:0"*/
      input_or_output_index = std::stoi(tensor_index, nullptr);
      return SUCCESS;
    } else {
      FE_LOGE("Construct node %s failed."
              "The string behind last colon (%s) is not integer!",
              name.c_str(), tensor_index.c_str());
      return FAILED;
    }
  } else {
    /* In this branch, the name is like MatMul_a or MatMul_a5 */
    FE_LOGE("Construct node %s failed, because the string behind last"
            "underline(%s) cannot be converted to integer.",
            name.c_str(), str_behind_underline.c_str());
    return FAILED;
  }
}

Status ParseWhenUnderlineExists(const string &name, string &op_type,
                                string &op_real_name, int32_t &input_or_output_index,
                                const size_t &underline_position) {
  /* We have found underline. */
  auto length_of_name = name.length();
  string str_in_front_of_underline = name.substr(0, underline_position);
  // There must be no colon if nothing is left behind underline.
  if (str_in_front_of_underline.empty()) {
    FE_LOGE("String in front of underline is empty.");
    return FAILED;
  }
  string str_behind_underline;
  if (underline_position < length_of_name - 1) {
    str_behind_underline = name.substr(underline_position + 1);
  } else {
    /* Nothing exists at the end of underline. It looks like "MatMul_" */
    op_type = str_in_front_of_underline;
    op_real_name = str_in_front_of_underline;
    input_or_output_index = 0;
    return SUCCESS;
  }
  op_type = str_in_front_of_underline;
  auto relative_colon_position = str_behind_underline.rfind(':');
  if (relative_colon_position == string::npos) {
    /* In this case, the name is like MatMul_05 or MatMul_a or MatMul_a5.
     * It contains a underline and a char behind underline */
    if (str_behind_underline.empty()) {
      op_real_name = str_in_front_of_underline;
      input_or_output_index = 0xFFFF;
      return SUCCESS;
    }
    if (StringUtils::IsInteger(str_behind_underline)) {
      /* In this branch, the name is like MatMul_5 or MatMul_05 */
      op_real_name = name;
      /* Set the input or output index = 0xFFFF, that means if the node
       * exists, it's the first input or output of the all no-peer anchors.
       * no-peer anchors is the anchors which do not have peer anchor.
       * If the node does not exist, input_or_output_index is 0.*/
      input_or_output_index = 0xFFFF;
      GC_LOGD("':' is missing. Assume it's the 1st tensor of op %s_%zu",
              op_type.c_str(), std::stoi(str_behind_underline, nullptr));
    } else {
      /* In this branch, the name is like MatMul_a or MatMul_a5 */
      FE_LOGE("Construct node %s failed, because the string behind last"
              "underline(%s) cannot be converted to integer.",
              name.c_str(), str_behind_underline.c_str());
      return FAILED;
    }
  } else {
    return ParseWhenBothUnderlineAndColonExists(
        name, op_type, op_real_name, input_or_output_index, underline_position,
        relative_colon_position, str_behind_underline, str_in_front_of_underline);
  }
  return SUCCESS;
}

Status ParseWhenUnderlineMissing(const string &name, string &op_type,
                                 string &op_real_name,
                                 int32_t &input_or_output_index) {
  /* underline '_' is missing */
  auto length_of_name = name.length();
  auto colon_position = name.rfind(':');
  if (colon_position == string::npos) {
    GC_LOGD("'_' and ':' are missing. Assume it's the first available tensor "
            "of first node of op %s",
            name.c_str());
    op_type = name;
    op_real_name = name;
    /* Set the input or output index = 0xFFFF. That means if this node
     * exists, this input or output is the first of the all no-peer-anchors.
     *
     * no-peer-anchors are the anchors which do not have peer anchor.
     * In other words , they are unused anchors.
     * If the node does not exist, input_or_output_index is 0.*/
    input_or_output_index = 0xFFFF;
  } else {
    if (colon_position == 0) {
      FE_LOGE("Nothing is in front of colon for name %s", name.c_str());
      return FAILED;
    }
    auto tensor_index = name.substr(colon_position + 1);
    GC_LOGD("colon position is %zu", colon_position);
    /* If nothing is behind the colon, the tensor index is 0 */
    if (IsSignedInteger(tensor_index) || tensor_index.empty()) {
      op_type = name.substr(0, colon_position);
      /* MatMul1:1  ->  optype is MatMul1 and op_real_name is MatMul1 */
      op_real_name = op_type;
      if (tensor_index.empty()) {
        input_or_output_index = 0xFFFF;
      } else {
        input_or_output_index = std::stoi(tensor_index, nullptr);
      }
      GC_LOGD("'_' is missing. Assume it's the %dth tensor of op %s",
              input_or_output_index, op_real_name.c_str());
    } else {
      FE_LOGE("The string behind last colon (%s) is not integer!",
              tensor_index.c_str());
      return FAILED;
    }
  }
  return SUCCESS;
}
/** Parse the input or output name, the name is as the following format:
 *
 * OpType_Integer1:Interer2
 * Integer1 is the index of the node with same type.
 * Integer2 is the index of the inputs or outputs.
 *
 * @return Status
 */
Status GraphConstructor::NodeNameParser(const string &name, string &op_type,
                                        string &op_real_name,
                                        int32_t &input_or_output_index) {
  // 1. find positon of last underline '_'. The reason we get the last underline
  // is that the op type may contain the underline.
  size_t underline_position;
  if (name.find("sgt_graph") != name.npos) {
    underline_position = string::npos;
  } else {
    underline_position = name.rfind('_');
  }

  if (underline_position == string::npos) {
    GC_LOGD("underline is missing for %s", name.c_str());
    return ParseWhenUnderlineMissing(name, op_type, op_real_name,
                                     input_or_output_index);
  } else {
    GC_LOGD("underline exists for %s, underline position is %u", name.c_str(),
            underline_position);
    return ParseWhenUnderlineExists(name, op_type, op_real_name,
                                    input_or_output_index, underline_position);
  }
}

bool CheckPeerAnchor(const ge::OutDataAnchorPtr &in_anchor1,
                     const ge::OutDataAnchorPtr &in_anchor2) {
  auto peer_set1 = in_anchor1->GetPeerInDataAnchors();
  auto peer_set2 = in_anchor2->GetPeerInDataAnchors();
  if (peer_set1.size() != peer_set2.size()) {
    FE_LOGE("peer1 size %d and peer2 size %d are not the same.",
            peer_set1.size(), peer_set2.size());
    return false;
  }
  auto peer_size = peer_set1.size();
  for (size_t index = 0; index < peer_size; index++) {
    if (peer_set1.at(index)->GetOwnerNode()->GetName() !=
        peer_set2.at(index)->GetOwnerNode()->GetName()) {
        FE_LOGE("peer1 %s and peer2 %s are not the same, index %d.",
              peer_set1.at(index)->GetOwnerNode()->GetName().c_str(),
              peer_set2.at(index)->GetOwnerNode()->GetName().c_str(),
              index);
      return false;
    }
  }
  return true;
}

bool CheckPeerAnchor(const ge::InDataAnchorPtr &in_anchor1,
                     const ge::InDataAnchorPtr &in_anchor2) {
  auto peer1 = in_anchor1->GetPeerOutAnchor();
  auto peer2 = in_anchor2->GetPeerOutAnchor();
  if ((peer1 == nullptr && peer2 != nullptr) ||
      (peer1 != nullptr && peer2 == nullptr)) {
    return false;
  }
  if (peer1 == nullptr && peer2 == nullptr) {
    return true;
  } else {
    if (peer1->GetOwnerNode()->GetName() == peer2->GetOwnerNode()->GetName()) {
      return true;
    }
    FE_LOGE("peer1 %s and peer2 %s are not the same.",
              peer1->GetOwnerNode()->GetName().c_str(), peer2->GetOwnerNode()->GetName().c_str());
    return peer1->GetOwnerNode()->GetName() == peer2->GetOwnerNode()->GetName();
  }
}

bool CheckTensorDesc(const ge::GeTensorDesc &tensor1,
                     const ge::GeTensorDesc &tensor2) {
  bool ret =
      tensor1.GetFormat() == tensor2.GetFormat() &&
      tensor1.GetDataType() == tensor2.GetDataType() &&
      tensor1.GetShape().GetDims() == tensor2.GetShape().GetDims() &&
      tensor1.GetOriginFormat() == tensor2.GetOriginFormat() &&
      tensor1.GetOriginShape().GetDims() == tensor2.GetOriginShape().GetDims();
  if (ret == false) {
    FE_LOGE("Format         %u   :  %u", tensor1.GetFormat(),
            tensor2.GetFormat());
    FE_LOGE("DType          %u   :  %u", tensor1.GetDataType(),
            tensor2.GetDataType());

    FE_LOGE("OriginalFormat %u   :  %u", tensor1.GetOriginFormat(),
            tensor2.GetOriginFormat());
    FE_LOGE("OriginalDtype  %u   :  %u", tensor1.GetOriginDataType(),
            tensor2.GetOriginDataType());
    string shape1 = StringUtils::IntegerVecToString(tensor1.GetShape().GetDims());
    string shape2 = StringUtils::IntegerVecToString(tensor2.GetShape().GetDims());
    string original_shape1 =
            StringUtils::IntegerVecToString(tensor1.GetOriginShape().GetDims());
    string original_shape2 =
            StringUtils::IntegerVecToString(tensor2.GetOriginShape().GetDims());
    FE_LOGE("Shape          %s   :  %s", shape1.c_str(), shape2.c_str());
    FE_LOGE("OriginalShape  %s   :  %s", original_shape1.c_str(),
            original_shape2.c_str());
  }
  return ret;
}
bool GraphConstructor::CompareNode(const ge::NodePtr &node1,
                                   const ge::NodePtr &node2) {
  auto op_desc1 = node1->GetOpDesc();
  auto op_desc2 = node2->GetOpDesc();
  if (op_desc1->GetInputsSize() != op_desc2->GetInputsSize()) {
    FE_LOGE("Input size of node[%s] %d and [%s] %d are not the same.",
            node1->GetName().c_str(), op_desc1->GetInputsSize(),
            node2->GetName().c_str(), op_desc2->GetInputsSize());
    return false;
  }
  if (op_desc1->GetOutputsSize() != op_desc2->GetOutputsSize()) {
    FE_LOGE("Output size of node[%s] %d and [%s] %d are not the same.",
            node1->GetName().c_str(), op_desc1->GetOutputsSize(),
            node2->GetName().c_str(), op_desc2->GetOutputsSize());
    return false;
  }

  auto input_size = op_desc1->GetInputsSize();
  auto output_size = op_desc2->GetOutputsSize();

  for (size_t tensor_index = 0; tensor_index < input_size; tensor_index++) {
    if (!CheckTensorDesc(op_desc1->GetInputDesc(tensor_index),
                         op_desc2->GetInputDesc(tensor_index))) {
      
      FE_LOGE("Input tensor %u of node %s[%s] and %s[%s] are not the same.",
              tensor_index, node1->GetName().c_str(),
              node1->GetInDataAnchor(tensor_index)->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str(), node2->GetName().c_str(),
              node2->GetInDataAnchor(tensor_index)->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str());
      return false;
    }
    if (!CheckPeerAnchor(node1->GetInDataAnchor(tensor_index),
                         node2->GetInDataAnchor(tensor_index))) {
      FE_LOGE("Peer node of input %u of node %s and %s are not the same.",
              tensor_index, node1->GetName().c_str(), node2->GetName().c_str());
      return false;
    }
  }

  for (size_t tensor_index = 0; tensor_index < output_size; tensor_index++) {
    if (!CheckTensorDesc(op_desc1->GetOutputDesc(tensor_index),
                         op_desc2->GetOutputDesc(tensor_index))) {
      FE_LOGE("Output tensor %u of node %s and %s are not the same.",
              tensor_index, node1->GetName().c_str(), node2->GetName().c_str());
      return false;
    }
    if (!CheckPeerAnchor(node1->GetOutDataAnchor(tensor_index),
                         node2->GetOutDataAnchor(tensor_index))) {
      FE_LOGE("Peer node of output %u of node %s and %s are not the same.",
              tensor_index, node1->GetName().c_str(), node2->GetName().c_str());
      return false;
    }
  }
  return true;
}

bool GraphConstructor::CompareGraph(const ComputeGraphPtr &graph1,
                                    const ComputeGraphPtr &graph2) {
  auto node_set1 = graph1->GetDirectNode();
  auto node_set2 = graph2->GetDirectNode();
  if (node_set1.size() != node_set2.size()) {
    FE_LOGE("nodeSet1.size[%d] and node_set2.size[%d] are not the same",
            node_set1.size(), node_set2.size());
    return false;
  }
  map<string, ge::NodePtr> node_map;
  for (auto &node : node_set2) {
    node_map.emplace(std::make_pair(node->GetName(), node));
  }
  for (auto &node1 : node_set1) {
    auto iter = node_map.find(node1->GetName());
    if (iter == node_map.end()) {
      FE_LOGE("node[%s] of node_set1 not find in node_set2",
               node1->GetName().c_str());
      return false;
    }
    ge::NodePtr node2 = iter->second;
    if (!CompareNode(node1, node2)) {
      FE_LOGE("Node %s and %s are not the same", node1->GetName().c_str(),
              node2->GetName().c_str());
      return false;
    }
  }
  return true;
}

void GraphConstructor::GetNodeByName(const string &name, ge::NodePtr &node_out) {
  for (auto &node : graph_->GetDirectNode()) {
    if (node->GetName() == name) {
      node_out = node;
      return;
    }
  }
  node_out = nullptr;
}

ge::NodePtr GraphConstructor::GetNodeByName(const string& name, const ComputeGraphPtr& graph) {
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetName() == name) {
      return node;
    }
  }
  return nullptr;
}

template <class T>
GraphConstructor &GraphConstructor::SetExtAttr(string &&attr_name,
                                               const T &value) {
  auto opdesc = last_added_node_->GetOpDesc();
  if (attr_name == ge::OP_EXTATTR_NAME_TBE_KERNEL) {
    const char tbe_bin[] = "tbe_bin";
    vector<char> buffer(tbe_bin, tbe_bin + strlen(tbe_bin));
    ge::OpKernelBinPtr tbe_kernel_ptr = std::make_shared<ge::OpKernelBin>(
        last_added_node_->GetName(), std::move(buffer));
    opdesc->SetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel_ptr);
  } else {
    opdesc->SetExtAttr(attr_name, value);
  }
  return *this;
}

GraphConstructor &GraphConstructor::SetPattern(const string &optype) {
  auto opdesc = last_added_node_->GetOpDesc();
  auto key_pattern = "_pattern";
  ge::AttrUtils::SetStr(opdesc, key_pattern, optype);
  return *this;
}

GraphConstructor &GraphConstructor::SetTvmType() {
  auto opdesc = last_added_node_->GetOpDesc();
  int64_t tvm = (int64_t)domi::ImplyType::TVM;
  ge::AttrUtils::SetInt(opdesc, ge::ATTR_NAME_IMPLY_TYPE, tvm);
  return *this;
}

GraphConstructor &GraphConstructor::SetFeImPlType(OpImplType impl_type) {
  auto opdesc = last_added_node_->GetOpDesc();
  ge::AttrUtils::SetInt(opdesc, FE_IMPLY_TYPE, static_cast<int>(impl_type));
  auto iter = IMPL_TYPE_MAP.find(impl_type);
  if (iter == IMPL_TYPE_MAP.end()) {
    FE_LOGE(
        "Op[name=%s,type=%s]: the FE imply type and GE imply type map is not "
        "found, the FE impl_type is [%d].",
        opdesc->GetName().c_str(), opdesc->GetType().c_str(), impl_type);
    return *this;
  }
  ge::AttrUtils::SetInt(opdesc, ge::ATTR_NAME_IMPLY_TYPE,
                        static_cast<int>(iter->second));
  return *this;
}

void GraphConstructor::SetGraph(ComputeGraphPtr graph) {
  graph_ = graph;
}


Status GraphConstructor::AddFunctionNodeInputDesc(ge::OpDescPtr fus_op, vector<fe::FusionDataFlow> &fus_input_edge_list) {
  int32_t fusion_dst_index = -1;
  graph_comm_ptr_->ClearFusionSrc();
  vector<bool> fusion_is_input_const_vector;
  uint32_t tensor_desc_index = 0;

  for (fe::FusionDataFlow &dataflow : fus_input_edge_list) {
    auto inedge = dataflow.edge;
    if (inedge.first == nullptr) {
      continue;
    }
    std::pair<string, ge::AnchorPtr> &node_dstindex_pair = dataflow.node_dataindex_pair;
    int64_t src_op_id = inedge.first->GetOwnerNode()->GetOpDesc()->GetId();

    // only support data edge
    fusion_dst_index = fusion_dst_index + 1;
    vector<int64_t> input_vector;
    input_vector = fus_op->GetInputOffset();
    if (static_cast<uint32_t>(fusion_dst_index) < input_vector.size()) {
      return FAILED;
    }

    ge::DataAnchorPtr in_edge_dst_data_anchor_ptr = std::static_pointer_cast<ge::DataAnchor>(inedge.second);
    FE_CHECK(in_edge_dst_data_anchor_ptr == nullptr, FE_LOGE("inEdgeDstDataAnchorPtr is nullptr."), return FAILED);
    ge::OpDescPtr in_edge_dst_op_desc_ptr = in_edge_dst_data_anchor_ptr->GetOwnerNode()->GetOpDesc();
    vector<bool> is_input_const_vector = in_edge_dst_op_desc_ptr->GetIsInputConst();
    uint32_t dst_anchor_index = static_cast<uint32_t>(in_edge_dst_data_anchor_ptr->GetIdx());
    if (dst_anchor_index < in_edge_dst_op_desc_ptr->GetInputsSize()) {
      if (fus_op->AddInputDesc(*(in_edge_dst_op_desc_ptr->GetInputDescPtr(in_edge_dst_data_anchor_ptr->GetIdx()))) !=
          ge::GRAPH_SUCCESS) {
        FE_LOGE("Add input desc failed.");
        return FAILED;
      }
      FE_LOGD("fusOp name %s, %d", fus_op->GetName().c_str(), in_edge_dst_data_anchor_ptr->GetIdx());

      if (dst_anchor_index < is_input_const_vector.size()) {
        fusion_is_input_const_vector.push_back(is_input_const_vector[in_edge_dst_data_anchor_ptr->GetIdx()]);
      } else {
        fusion_is_input_const_vector.push_back(false);
      }
      tensor_desc_index++;
    } else {
      int32_t input_desc_size = in_edge_dst_op_desc_ptr->GetInputsSize();
      int32_t is_input_const_size = is_input_const_vector.size();
      int32_t DstIndex = in_edge_dst_data_anchor_ptr->GetIdx();
      FE_LOGE("AddFusionNodeInput input_desc_size %u, is_input_const_size %u, input DstIndex %u.",
                (uint32_t)input_desc_size, (uint32_t)is_input_const_size, (uint32_t)DstIndex);
      return FAILED;
    }
    fus_op->SetIsInputConst(fusion_is_input_const_vector);
    graph_comm_ptr_->AddFusionInputSrc(src_op_id, inedge.first, fusion_dst_index, node_dstindex_pair);
  }

  return SUCCESS;
}

Status GraphConstructor::AddFunctionNodeOutputDesc(ge::OpDescPtr fus_op, vector<fe::FusionDataFlow> &fus_output_edge_list) {
  graph_comm_ptr_->ClearFusionSrc();
  graph_comm_ptr_->ClearFusionDst();

  int32_t fusion_src_index = 0;
  for (fe::FusionDataFlow &dataflow : fus_output_edge_list) {
    auto outedge = dataflow.edge;
    std::pair<string, ge::AnchorPtr> &node_srcindex_pair = dataflow.node_dataindex_pair;
    if (outedge.second == nullptr) {
      continue;
    }
    int64_t dst_op_id = outedge.second->GetOwnerNode()->GetOpDesc()->GetId();
    ge::DataAnchorPtr out_edge_dst_data_anchor_ptr = std::static_pointer_cast<ge::DataAnchor>(outedge.second);
    FE_CHECK(out_edge_dst_data_anchor_ptr == nullptr, FE_LOGE("outEdgeDstDataAnchorPtr is nullptr."),
    return FAILED);
    if (graph_comm_ptr_->IsFusionDstExist(dst_op_id, outedge.second)) {
      FE_LOGI("MergeFusionNodeOutputEdgeList Dstid %u, DstIndex %u.", (uint32_t)dst_op_id,
                (uint32_t)out_edge_dst_data_anchor_ptr->GetIdx());
      continue;
    }
    graph_comm_ptr_->SaveFusionDst(dst_op_id, outedge.second);

    int32_t fusion_src_exist_index;
    int32_t fusion_dst_exist_index;

    ge::DataAnchorPtr out_edge_src_data_anchor_ptr = std::static_pointer_cast<ge::DataAnchor>(outedge.first);
    FE_CHECK(out_edge_src_data_anchor_ptr == nullptr, FE_LOGE("outEdgeSrcDataAnchorPtr is nullptr."),
    return FAILED);
    ge::OpDescPtr out_edge_src_op_desc_ptr = out_edge_src_data_anchor_ptr->GetOwnerNode()->GetOpDesc();

    auto src_op_id = outedge.first->GetOwnerNode()->GetOpDesc()->GetId();
    if (!graph_comm_ptr_->GetFusionSrc(src_op_id, outedge.first, fusion_src_exist_index, fusion_dst_exist_index)) {
      FE_CHECK(out_edge_src_op_desc_ptr == nullptr, FE_LOGE("outEdgeSrcOpDescPtr is nullptr."), return FAILED);
      if (static_cast<uint32_t>(out_edge_src_data_anchor_ptr->GetIdx()) < out_edge_src_op_desc_ptr->GetOutputsSize()) {
        FE_LOGI_IF(fus_op->AddOutputDesc(*(out_edge_src_op_desc_ptr->GetOutputDescPtr(
                out_edge_src_data_anchor_ptr->GetIdx()))) != ge::GRAPH_SUCCESS, "AddOutputDesc not success");
        graph_comm_ptr_->AddFusionOutputSrc(src_op_id, outedge.first, fusion_src_index, node_srcindex_pair);
      } else {
        int32_t output_desc_size = out_edge_src_op_desc_ptr->GetOutputsSize();
        int32_t SrcIndex = out_edge_src_data_anchor_ptr->GetIdx();
        FE_LOGE("MergeFusionNodeOutputEdgeList output_desc_size %u, SrcIndex %u.", (uint32_t)output_desc_size,
                  (uint32_t)SrcIndex);
        return FAILED;
      }
      fusion_src_index++;
    }
  }

  return SUCCESS;
}

ge::OpDescPtr GraphConstructor::CreateFunctionOp(vector<ge::NodePtr> &node_vec) const {
  if (node_vec.empty()) {
    return nullptr;
  }
  ge::NodePtr first_node = node_vec[0];
  FE_CHECK(first_node == nullptr, FE_LOGE("CreateFusionOp nullptr Pointer."), return nullptr);
  ge::OpDescPtr function_opdef = nullptr;
  FE_MAKE_SHARED(function_opdef = std::make_shared<ge::OpDesc>(ge::OpDesc()), return nullptr);
  std::string fusion_node_name;

  for (ge::NodePtr &node : node_vec) {
    FE_CHECK(node == nullptr, FE_LOGE("[FFTSSubGraphOpt][CrtFuncOp] node is nullptr."), return nullptr);
    fusion_node_name += node->GetOpDesc()->GetName();
    if (fusion_node_name.size() > kMaxOpNmaLen) {
      fusion_node_name = first_node->GetOpDesc()->GetName() + "_ffts_fusion";
      break;
    }
  }
  function_opdef->SetName(fusion_node_name);
  ge::OpDescUtilsEx::SetType(function_opdef, "PartitionedCall");

  // copy session graph id
  string session_graph_id;
  if (ge::AttrUtils::GetStr(first_node->GetOpDesc(), ge::ATTR_NAME_SESSION_GRAPH_ID, session_graph_id)) {
    if (!ge::AttrUtils::SetStr(function_opdef, ge::ATTR_NAME_SESSION_GRAPH_ID, session_graph_id)) {
      FE_LOGE("Op[%s]: fail to set the attribute %s.", fusion_node_name.c_str(),
              ge::ATTR_NAME_SESSION_GRAPH_ID.c_str());
      return nullptr;
    }
  }

  function_opdef->SetOpEngineName("ffts_plus");

  return function_opdef;
}

Status EstablishConnection(const ge::NodePtr &node,
                           const std::unordered_set<std::string> &node_name_set,
                           ge::CompleteGraphBuilder &builder) {
  const auto &src_node_name = node->GetName();
  for (const auto &out_data_anchor : node->GetAllOutDataAnchors()) {
    FE_CHECK_NOTNULL(out_data_anchor);
    auto src_idx = out_data_anchor->GetIdx();
    for (auto &in_data_anchor : out_data_anchor->GetPeerInDataAnchors()) {
      FE_CHECK_NOTNULL(in_data_anchor);
      auto dst_node = in_data_anchor->GetOwnerNode();
      FE_CHECK_NOTNULL(dst_node);
      if (!node_name_set.count(dst_node->GetName())) {
        FE_LOGD("Node[name:%s] is out op, skip.", dst_node->GetName().c_str());
        continue;
      }
      builder.AddDataLink(src_node_name, src_idx, dst_node->GetName(), in_data_anchor->GetIdx());
    }
  }
  auto out_ctrl_anchor =  node->GetOutControlAnchor();
  FE_CHECK_NOTNULL(out_ctrl_anchor);
  for (auto &in_ctrl_anchor : out_ctrl_anchor->GetPeerInControlAnchors()) {
    FE_CHECK_NOTNULL(in_ctrl_anchor);
    auto dst_node = in_ctrl_anchor->GetOwnerNode();
    FE_CHECK_NOTNULL(dst_node);
    if (!node_name_set.count(dst_node->GetName())) {
      FE_LOGD("Dst Node[name:%s] linked by control edge is out op, skip.", dst_node->GetName().c_str());
      continue;
    }
    builder.AddControlLink(src_node_name, dst_node->GetName());
  }
  return SUCCESS;
}


Status GraphConstructor::CreateFunctionOpSubGraph(ge::NodePtr &function_node,
                                                  std::vector<ge::NodePtr> &node_vec,
                                                  vector<fe::FusionDataFlow> &input_edge_list,
                                                  vector<fe::FusionDataFlow> &output_edge_list,
                                                  vector<fe::FusionDataFlow> &output_ctrl_edge_list) {
  if (node_vec.empty()) {
    return FAILED;
  }
  auto graph = node_vec[0]->GetOwnerComputeGraph();
  FE_LOGD("ai-core sub graph name is %s.", graph->GetName().c_str());
  auto func_graph = function_node->GetOwnerComputeGraph();
  FE_LOGD("func owner graph name is %s.", func_graph->GetName().c_str());
  ge::ComputeGraphPtr src_graph = graph->TryGetExtAttr("part_src_graph", ge::ComputeGraphPtr());
  FE_CHECK_NOTNULL(src_graph);
  FE_LOGD("src graph name is %s", src_graph->GetName().c_str());
  auto root_graph = ge::GraphUtils::FindRootGraph(src_graph);
  FE_CHECK_NOTNULL(root_graph);
  FE_LOGD("root graph name is %s", root_graph->GetName().c_str());
  std::unordered_set<std::string> node_name_set;
  for (auto &node : node_vec) {
    node_name_set.emplace(node->GetName());
  }

  string sgt_graph_name = func_graph->GetName() + "_sgt_graph_" + std::to_string(sgt_graph_index_++);
  FE_LOGD("function_node: %s, sgt graph name is %s", function_node->GetName().c_str(), sgt_graph_name.c_str());
  ge::CompleteGraphBuilder builder(sgt_graph_name, false);

  for (auto &node : node_vec) {
    /* Adds a node to the builder. Currently, only the input parameter op_desc
       is supported, and the connection information will be lost. Therefore,
       the AddDataLink interface needs to be invoked to re-establish the
       connection. */
    builder.AddNode(node->GetOpDesc());

    FE_LOGD("Begin re-establish connection of node[name:%s]. out data size is %d.",
              node->GetName().c_str(), node->GetAllOutDataAnchorsSize());
    (void)EstablishConnection(node, node_name_set, builder);
  }

  if (graph_comm_ptr_->AddOuterDataEdgeInputs(input_edge_list, builder) != SUCCESS) {
    return FAILED;
  }

  if (graph_comm_ptr_->AddOuterCtrlEdgeOutputs(output_ctrl_edge_list, builder) != SUCCESS ||
      graph_comm_ptr_->AddOuterDataEdgeOutputs(output_edge_list, builder) != SUCCESS) {
    return FAILED;
  }

  builder.SetParentNode(function_node);

  ge::graphStatus status = ge::GRAPH_SUCCESS;
  string err_msg;
  auto sgt_graph = builder.Build(status, err_msg);
  if (status != ge::GRAPH_SUCCESS) {
    FE_LOGE("Build graph not success. status is : %d, err_msg is : %s.", status, err_msg.c_str());
    return FAILED;
  }

  sgt_graph->SetParentGraph(graph);
  sgt_graph->SetParentNode(function_node);
  ge::OpDescPtr function_op_desc = function_node->GetOpDesc();
  function_op_desc->AddSubgraphName(sgt_graph->GetName());
  function_op_desc->SetSubgraphInstanceName(0, sgt_graph->GetName());
  (void)ge::AttrUtils::SetGraph(function_op_desc, "_sgt_sub_graph", sgt_graph);
  (void)ge::AttrUtils::SetStr(function_op_desc, "_ffts_plus_sub_graph", sgt_graph->GetName().c_str());
  (void)ge::AttrUtils::SetBool(function_op_desc, "_sgt_function_op", true);
  (void)ge::AttrUtils::SetStr(function_op_desc, "_composite_engine_name", "ffts_plus");
  (void)ge::AttrUtils::SetStr(function_op_desc, "_composite_engine_kernel_lib_name", "ffts_plus");
  if (root_graph->AddSubgraph(sgt_graph->GetName(), sgt_graph) != ge::GRAPH_SUCCESS) {
    FE_LOGE("Add sub graph not success. status is : %d, root graph: %s.", status, root_graph->GetName().c_str());
    return FAILED;
  }
  FE_LOGD("Add sub graph success. root graph: %s, subgraph: %s.",
            root_graph->GetName().c_str(), sgt_graph->GetName().c_str());

  for (auto &node : node_vec) {
    std::vector<int> io_map = {0};
    if (node->GetAllOutDataAnchors().size() == 0) {
      io_map = {};
    } else if (node->GetAllInDataAnchors().size() == 0) {
      io_map = {-1};
    }
    if (ge::GraphUtils::IsolateNode(node, io_map) != ge::GRAPH_SUCCESS) {
      FE_LOGE("Isolate Node %s not success.", node->GetName().c_str());
      return FAILED;
    }
    if (ge::GraphUtils::RemoveNodeWithoutRelink(graph, node) != ge::GRAPH_SUCCESS) {
      FE_LOGE("Remove node: %s without relink failed", node->GetName().c_str());
      return FAILED;
    }
  }
  return SUCCESS;
}

GraphConstructor &GraphConstructor::Attr(const string &node_name, const std::string &name, bool value) {
  auto iter = op_map_.find(node_name);
  if (iter == op_map_.end()) {
    FE_LOGE("Set int32_t: node %s does not exist.", node_name.c_str());
  } else {
    ge::AttrUtils::SetBool(iter->second->node->GetOpDesc(), name, static_cast<int64_t>(value));
  }
  return *this;
}

GraphConstructor &GraphConstructor::Attr(const string &node_name, const std::string &name, int32_t value) {
  auto iter = op_map_.find(node_name);
  if (iter == op_map_.end()) {
    FE_LOGE("Set int32_t: node %s does not exist.", node_name.c_str());
  } else {
    ge::AttrUtils::SetInt(iter->second->node->GetOpDesc(), name, static_cast<int64_t>(value));
  }
  return *this;
}

GraphConstructor &GraphConstructor::Attr(const string &node_name, const std::string &name, uint32_t value) {
  auto iter = op_map_.find(node_name);
  if (iter == op_map_.end()) {
    FE_LOGE("Set uint32_t: node %s does not exist.", node_name.c_str());
  } else {
    ge::AttrUtils::SetInt(iter->second->node->GetOpDesc(), name, static_cast<int64_t>(value));
  }
  return *this;
}

GraphConstructor &GraphConstructor::Attr(const string &node_name, const std::string &name, float value) {
  auto iter = op_map_.find(node_name);
  if (iter == op_map_.end()) {
    FE_LOGE("Set float: node %s does not exist.", node_name.c_str());
  } else {
    ge::AttrUtils::SetFloat(iter->second->node->GetOpDesc(), name, value);
  }
  return *this;
}

GraphConstructor &GraphConstructor::Attr(const string &node_name, const std::string &name,
                                         const std::vector<int32_t> &value) {
  auto iter = op_map_.find(node_name);
  if (iter == op_map_.end()) {
    FE_LOGE("Set list int32_t: node %s does not exist.", node_name.c_str());
  } else {
    ge::AttrUtils::SetListInt(iter->second->node->GetOpDesc(), name, value);
  }
  return *this;
}

GraphConstructor &GraphConstructor::Attr(const string &node_name, const std::string &name,
                                         const std::vector<uint32_t> &value) {
  auto iter = op_map_.find(node_name);
  if (iter == op_map_.end()) {
    FE_LOGE("Set list uint32_t: node %s does not exist.", node_name.c_str());
  } else {
    ge::AttrUtils::SetListInt(iter->second->node->GetOpDesc(), name, value);
  }
  return *this;
}

GraphConstructor &GraphConstructor::Attr(const string &node_name, const std::string &name,
                                         const std::vector<int64_t> &value) {
  auto iter = op_map_.find(node_name);
  if (iter == op_map_.end()) {
    FE_LOGE("Set list int64_t: node %s does not exist.", node_name.c_str());
  } else {
    ge::AttrUtils::SetListInt(iter->second->node->GetOpDesc(), name, value);
  }
  return *this;
}

GraphConstructor &GraphConstructor::Attr(const string &node_name, const std::string &name, const char *value) {
  auto iter = op_map_.find(node_name);
  if (iter == op_map_.end()) {
    FE_LOGE("Set list int32_t: node %s does not exist.", node_name.c_str());
  } else {
    ge::AttrUtils::SetStr(iter->second->node->GetOpDesc(), name, value);
  }
  return *this;
}

GraphConstructor &GraphConstructor::Attr(const std::string &name, bool value) {
  return Attr(last_added_node_->GetName(), name, value);
}

GraphConstructor &GraphConstructor::Attr(const std::string &name, int32_t value) {
  return Attr(last_added_node_->GetName(), name, value);
}

GraphConstructor &GraphConstructor::Attr(const std::string &name, uint32_t value) {
  return Attr(last_added_node_->GetName(), name, value);
}

GraphConstructor &GraphConstructor::Attr(const std::string &name, const std::vector<int32_t> &value) {
  return Attr(last_added_node_->GetName(), name, value);
}

GraphConstructor &GraphConstructor::Attr(const std::string &name, const std::vector<uint32_t> &value) {
  return Attr(last_added_node_->GetName(), name, value);
}

GraphConstructor &GraphConstructor::Attr(const std::string &name, const std::vector<int64_t> &value) {
  return Attr(last_added_node_->GetName(), name, value);
}

GraphConstructor &GraphConstructor::Attr(const std::string &name, const char *value) {
  return Attr(last_added_node_->GetName(), name, value);
}

Status GraphConstructor::TransSingleSubGraph(ge::ComputeGraph &graph, std::vector<ge::NodePtr> &node_vec) {
  vector<fe::FusionDataFlow> input_edge_list;
  vector<fe::FusionDataFlow> output_edge_list;
  vector<fe::FusionDataFlow> input_ctrl_edge_list;
  vector<fe::FusionDataFlow> output_ctrl_edge_list;
  std::unordered_set<ge::NodePtr> node_set;
  for (auto &node : node_vec) {
    node_set.emplace(node);
  }

  FE_LOGD("TransSingleSubGraph enter");
  if (graph_comm_ptr_->GetFusionNodeEdgeList(node_vec, node_set, input_edge_list, output_edge_list) != SUCCESS) {
    FE_LOGE("GetFusionNodeEdgeList failed.");
    return FAILED;
  }
  if (graph_comm_ptr_->GetFusionNodeCtrlEdgeList(node_vec, node_set, input_ctrl_edge_list,
      output_ctrl_edge_list) != SUCCESS) {
    FE_LOGE("GetFusionNodeCtrlEdgeList failed.");
    return FAILED;
  }
  ge::OpDescPtr function_op = CreateFunctionOp(node_vec);
  if (function_op == nullptr) {
    FE_LOGE("CreateFunctionOp failed.");
    return FAILED;
  }
  if (AddFunctionNodeInputDesc(function_op, input_edge_list) != SUCCESS) {
    FE_LOGE("Failed to AddFusionNodeInputDesc.");
    return FAILED;
  }
  if (AddFunctionNodeOutputDesc(function_op, output_edge_list) != SUCCESS) {
    FE_LOGE("Failed to AddFusionNodeOutputDesc.");
    return FAILED;
  }
  ge::NodePtr function_node = graph.AddNode(function_op);
  FE_CHECK_NOTNULL(function_node);

  // Merge Same scope node
  if (graph_comm_ptr_->MergeFusionNodeEdgeList(function_node, node_vec, input_edge_list, output_edge_list) !=
      SUCCESS) {
    FE_LOGE("MergeFusionNodeEdgeList failed!");
    return FAILED;
  }

  if (graph_comm_ptr_->MergeFusionNodeCtrlEdgeList(function_node, node_vec, input_ctrl_edge_list,
                                                  output_ctrl_edge_list) != SUCCESS) {
    FE_LOGE("MergeFusionNodeCtrlEdgeList failed!");
    return FAILED;
  }
  return CreateFunctionOpSubGraph(function_node, node_vec, input_edge_list, output_edge_list, output_ctrl_edge_list);
}
} // namespace fe
