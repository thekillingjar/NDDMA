/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_metadef/register/register.h"
#include <google/protobuf/message.h>
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph/debug/ge_op_types.h"
#include "framework/common/debug/ge_log.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/type_utils.h"
#include "proto/tensorflow/attr_value.pb.h"
#include "proto/tensorflow/node_def.pb.h"
#include "register/auto_mapping_util.h"
#include "register/op_registry.h"
#include "register/register_utils.h"
#include "graph/graph.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph/ascend_limits.h"
#include "graph/def_types.h"
#include "graph/utils/graph_utils_ex.h"

namespace domi {
/*lint -e1073*/
namespace {
const std::string kDefaultFormat = "ND";
const std::string kSrcFormat = "src_format";
const std::string kDstFormat = "dst_format";
const std::string kDataFormat = "data_format";
const std::string kTfInputDesc = "input_tensor_desc";
const std::string kTfOutputDesc = "output_tensor_desc";
const std::string kFuncNameKey = "name";

struct DynamicInfo {
  DynamicInfo(const DynamicType dynamic_type, const uint32_t index, const uint32_t num)
      : type(dynamic_type),
        inset_index(index),
        tensor_num(num) {}
  explicit DynamicInfo() : DynamicInfo(kInvalid, 0U, 0U) {}

  DynamicType GetType() const {return type;}
  uint32_t GetInsetIndex() const {return inset_index;}
  uint32_t GetTensorNum() const {return tensor_num;}
  void SetInsetIndex(const uint32_t insetIndex) {inset_index = insetIndex;}
private:
  DynamicType type;
  uint32_t inset_index;
  uint32_t tensor_num;
};

std::set<std::string> GetSubgraphAttrNames(const ge::Operator &op) {
  if (op.GetSubgraphNamesCount() == 0U) {
    return std::set<std::string>();
  }
  std::vector<ge::AscendString> subgraph_names;
  (void) op.GetSubgraphNames(subgraph_names);
  std::vector<std::string> subgraph_name_strings;
  for (const auto &subgraph_name : subgraph_names) {
    (void) subgraph_name_strings.emplace_back(subgraph_name.GetString());
  }
  return std::set<std::string>(subgraph_name_strings.begin(), subgraph_name_strings.end());
}

/// there are two forms to represent functions in TF:
/// case 1(subgraph of a `if` node) normal subgraph:
/// attr {
///   key: "else_branch"
///   value {
///     func {
///       name: "cond_false_9"
///     }
///   }
/// }

/// case 2(subgraph of a `case` node) dynamic subgraph:
/// attr {
///   key: "branches"
///   value {
///     list {
///       func {
///         name: "two_J6Sc96RZs5g"
///       }
///       func {
///         name: "three_3pYv7KFNs2M"
///       }
///       func {
///         name: "four_MdtG6T4LHxA"
///       }
///     }
///   }
/// }
/// \param func_attr
/// \param op_desc
/// \return
Status AutoMappingFunction(const std::pair<std::string, domi::tensorflow::AttrValue> &func_attr,
                           std::shared_ptr<ge::OpDesc> &op_desc) {
 if (func_attr.second.value_case() == domi::tensorflow::AttrValue::kFunc) {
    const auto &func_signature = func_attr.second.func().name();
    if (ge::OpDescUtils::SetSubgraphInstanceName(func_attr.first, func_signature, op_desc) != ge::GRAPH_SUCCESS) {
      GE_LOGE("Failed to set subgraph instance %s for node %s type %s, instance name %s",
              func_attr.first.c_str(), op_desc->GetName().c_str(),
              op_desc->GetType().c_str(), func_signature.c_str());
      return FAILED;
    }
  } else if (func_attr.second.value_case() == domi::tensorflow::AttrValue::kList) {
    uint32_t i = 0U;
    for (auto &dyn_func_attr : func_attr.second.list().func()) {
      const auto &func_signature = dyn_func_attr.name();
      const auto subgraph_name = func_attr.first + std::to_string(i++);
      auto ret = op_desc->AddSubgraphName(subgraph_name);
      if (ret != ge::GRAPH_SUCCESS) {
        GE_LOGE("Failed to add subgraph name %s to node %s type %s",
                subgraph_name.c_str(), op_desc->GetName().c_str(), op_desc->GetType().c_str());
        return FAILED;
      }
      ret = ge::OpDescUtils::SetSubgraphInstanceName(subgraph_name, func_signature, op_desc);
      if (ret != ge::GRAPH_SUCCESS) {
        GE_LOGE("Failed to set dynamic subgraph instance %s for node %s type %s, instance name %s",
                func_attr.first.c_str(), op_desc->GetName().c_str(),
                op_desc->GetType().c_str(), func_signature.c_str());
        return FAILED;
      }
    }
  } else {
    GE_LOGE("Unexpected attr value type %d for func", static_cast<int32_t>(func_attr.second.value_case()));
    return FAILED;
  }
  return SUCCESS;
}

Status CheckDynamicInfo(const vector<DynamicInputOutputInfo> &dynamic_name_attr_value) {
  for (const auto &dynamic_info : dynamic_name_attr_value) {
    if ((dynamic_info.port_name_len == 0) || (dynamic_info.port_name_len > kMaxNameLength) ||
        (dynamic_info.attr_name_len == 0) || (dynamic_info.attr_name_len > kMaxNameLength)) {
      GELOGE(PARAM_INVALID, "[Check][Param]port_name_len:%ld, attr_name_len:%ld",
             dynamic_info.port_name_len,  dynamic_info.attr_name_len);
      return PARAM_INVALID;
    }

    const int64_t port_name_len = static_cast<int64_t>(strnlen(dynamic_info.port_name, ge::kMaxNameLen));
    if ((dynamic_info.port_name == nullptr) || (port_name_len != dynamic_info.port_name_len)) {
      GELOGE(PARAM_INVALID, "[Check][Param]port_name:%s, port_name_len:%ld",
             dynamic_info.port_name, dynamic_info.port_name_len);
      return PARAM_INVALID;
    }

    const int64_t attr_name_len = static_cast<int64_t>(strnlen(dynamic_info.attr_name, ge::kMaxNameLen));
    if ((dynamic_info.attr_name == nullptr) || (attr_name_len != dynamic_info.attr_name_len)) {
      GELOGE(PARAM_INVALID, "[Check][Param]attr_name:%s, attr_name_len:%ld",
             dynamic_info.attr_name, dynamic_info.attr_name_len);
      return PARAM_INVALID;
    }
  }

  return SUCCESS;
}

Status GetDynamicTensorNum(const std::shared_ptr<ge::OpDesc> &op_desc, const string &attr_name, uint32_t &tensor_num) {
  GE_CHECK_NOTNULL(op_desc);

  ge::GeAttrValue attr_value;
  const ge::graphStatus ret = op_desc->GetAttr(attr_name, attr_value);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "[Get][Attr:%s]op name:%s", attr_name.c_str(), op_desc->GetName().c_str());
    return FAILED;
  }

  const ge::GeAttrValue::ValueType value_type = attr_value.GetValueType();
  switch (value_type) {
    case ge::GeAttrValue::VT_LIST_DATA_TYPE: {
      vector<ge::DataType> vec_d;
      (void)ge::AttrUtils::GetListDataType(op_desc, attr_name, vec_d);
      tensor_num = static_cast<uint32_t >(vec_d.size());
      break;
    }
    case ge::GeAttrValue::VT_INT: {
      (void)ge::AttrUtils::GetInt(op_desc, attr_name, tensor_num);
      break;
    }
    default:
      GELOGI("Default other value type: %d", static_cast<int32_t>(value_type));
      break;
  }

  return SUCCESS;
}

Status GetDynamicAttrNum(const std::shared_ptr<ge::OpDesc> &op_desc,
    const vector<DynamicInputOutputInfo> &dynamic_name_attrs, std::map<string, DynamicInfo> &port_dynamic_info) {
  GE_CHECK_NOTNULL(op_desc);
  for (const auto &dynamic_name_attr : dynamic_name_attrs) {
    const std::string attr_name = dynamic_name_attr.attr_name;
    uint32_t dynamic_tensor_num = 0U;
    if (op_desc->HasAttr(attr_name)) {
      if (GetDynamicTensorNum(op_desc, attr_name, dynamic_tensor_num) != SUCCESS) {
        GELOGE(FAILED, "[Get][DynamicTensorNum]op_name:%s, attr_name:%s",
               op_desc->GetName().c_str(), attr_name.c_str());
        return FAILED;
      }
    } else {
      GELOGW("[UpdateDynamic][GetAttr] Dynamic attr %s does not exist in op %s", attr_name.c_str(),
             op_desc->GetName().c_str());
      continue;
    }
    GELOGI("In Op %s dynamic attr [%s] is exist, tensor num: %u.", op_desc->GetName().c_str(), attr_name.c_str(),
           dynamic_tensor_num);
    port_dynamic_info[dynamic_name_attr.port_name] = DynamicInfo(dynamic_name_attr.type, 0U, dynamic_tensor_num);
  }
  return SUCCESS;
}

Status UpdateDynamicInputOutPutIndex(const std::shared_ptr<ge::OpDesc> &op_desc,
    const vector<DynamicInputOutputInfo> &dynamic_name_attrs, std::map<string, DynamicInfo> &port_dynamic_info) {
  GE_CHECK_NOTNULL(op_desc);
  if (GetDynamicAttrNum(op_desc, dynamic_name_attrs, port_dynamic_info) != SUCCESS) {
    GELOGE(FAILED, "[Get][DynamicAttrNum] fail, op_name:%s", op_desc->GetName().c_str());
    return FAILED;
  }

  const vector<string> register_input_names = op_desc->GetRegisterInputName();
  uint32_t input_index = 0U;
  uint32_t input_increment = 0U;
  for (const auto &input_name : register_input_names) {
    const auto input_iter = port_dynamic_info.find(input_name);
    if (input_iter != port_dynamic_info.end()) {
      port_dynamic_info[input_name].SetInsetIndex(input_index + input_increment);
      const uint32_t tensor_num = port_dynamic_info[input_name].GetTensorNum();
      if (tensor_num == 0U) {
        (void)port_dynamic_info.erase(input_iter);
        continue;
      }
      input_increment += (tensor_num > 0U) ? (tensor_num - 1U) : 0U;
      GELOGI("Dynamic input name[%s] insert index: %u, tensor num: %u, op proto index: %u", input_name.c_str(),
             port_dynamic_info[input_name].GetInsetIndex(), tensor_num, input_index);
    }
    input_index++;
  }
  const vector<string> register_output_names = op_desc->GetRegisterOutputName();
  uint32_t output_index = 0U;
  uint32_t out_increment = 0U;
  for (const auto &output_name : register_output_names) {
    const auto output_iter = port_dynamic_info.find(output_name);
    if (output_iter != port_dynamic_info.end()) {
      port_dynamic_info[output_name].SetInsetIndex(output_index + out_increment);
      const uint32_t tensor_num = port_dynamic_info[output_name].GetTensorNum();
      if (tensor_num == 0U) {
        (void)port_dynamic_info.erase(output_iter);
        continue;
      }
      out_increment += (tensor_num > 0U) ? (tensor_num - 1U) : 0U;
      GELOGI("Dynamic output name[%s] insert index: %u, tensor num: %u, op proto index: %u", output_name.c_str(),
             port_dynamic_info[output_name].GetInsetIndex(), tensor_num, output_index);
    }
    output_index++;
  }
  return SUCCESS;
}

Status SetOpdescInputOutputFormat(std::shared_ptr<ge::OpDesc> &op_desc) {
  GE_CHECK_NOTNULL(op_desc);

  const auto inputDescsPtr = op_desc->GetAllInputsDescPtr();
  const auto outputDescsPtr = op_desc->GetAllOutputsDescPtr();

  string src_data_format = kDefaultFormat;
  string dst_data_format = kDefaultFormat;
  if (op_desc->HasAttr(kSrcFormat)) {
    (void)ge::AttrUtils::GetStr(op_desc, kSrcFormat, src_data_format);
  }
  if (op_desc->HasAttr(kDstFormat)) {
    (void)ge::AttrUtils::GetStr(op_desc, kDstFormat, dst_data_format);
  }
  if (op_desc->HasAttr(kDataFormat)) {
    (void)ge::AttrUtils::GetStr(op_desc, kDataFormat, src_data_format);
    dst_data_format = src_data_format;
  }
  ge::Format format = ge::TypeUtils::DataFormatToFormat(src_data_format);
  for (const auto &inputDescPtr : inputDescsPtr) {
    inputDescPtr->SetOriginFormat(format);
    inputDescPtr->SetFormat(format);
  }
  format = ge::TypeUtils::DataFormatToFormat(dst_data_format);
  for (const auto &outputDescPtr : outputDescsPtr) {
    outputDescPtr->SetOriginFormat(format);
    outputDescPtr->SetFormat(format);
  }
  return SUCCESS;
}
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY Status AutoMappingFnDynamic(
    const google::protobuf::Message *op_src, ge::Operator &op,
    std::map<std::string, std::pair<std::string, std::string>> dynamic_name_attr_value,
    int32_t in_pos, int32_t out_pos) {
  // 1. automapping for parser
  const std::shared_ptr<ge::OpDesc> op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  GE_CHECK_NOTNULL(op_desc);
  GE_CHECK_NOTNULL(op_src);
  const Status ret = OperatorAutoMapping(op_src, op);
  if (ret != SUCCESS) {
    GE_LOGE("Op: %s call auto mapping function failed.", op_desc->GetName().c_str());
    return FAILED;
  }

  GELOGI("op[%s] call auto mapping function success.", op_desc->GetName().c_str());

  if (dynamic_name_attr_value.size() > 2U) {  // attr value size should be less than 2
    GE_LOGE("attr set size [%zu] should be less than 2.", dynamic_name_attr_value.size());
    return FAILED;
  }

  // add dynamic input and output
  const domi::tensorflow::NodeDef *const node = ge::PtrToPtr<const google::protobuf::Message,
                                                             const domi::tensorflow::NodeDef>(op_src);
  for (const auto &it : dynamic_name_attr_value) {
    const std::string flag = it.first;
    const std::pair<std::string, std::string> name_value = it.second;
    const std::string dynamic_name = name_value.first;
    const std::string attr_name = name_value.second;

    tensorflow::AttrValue attr_num;
    int32_t dynamic_tensor_num = 0;
    if (!(ge::AutoMappingUtil::FindAttrValue(node, attr_name, attr_num))) {
      GELOGW("[AutoMappingFn][GetAttr] Dynamic attr %s in node %s does not exist.", attr_name.c_str(), node->name().c_str());
    }

    dynamic_tensor_num = (attr_num.has_list()) ? attr_num.list().type_size() : static_cast<int32_t>(attr_num.i());
    if (dynamic_tensor_num <= 0) {
      GELOGW("[AutoMappingFn][Check] Dynamic num %d in node %s is less than 0.", dynamic_tensor_num,
             node->name().c_str());
      continue;
    }

    GELOGI("In NodeDef %s dynamic attr [%s] is exist: %d.", node->name().c_str(), attr_name.c_str(),
           dynamic_tensor_num);

    if (flag == "in") {
      const bool is_pushback = (in_pos == -1);
      (void)op_desc->AddDynamicInputDesc(dynamic_name, static_cast<uint32_t>(dynamic_tensor_num), is_pushback);
      (void)ge::AttrUtils::SetInt(op_desc, DYNAMIC_INPUT_TD_NUM(dynamic_name), dynamic_tensor_num);
      GELOGI("In NodeDef %s add dynamic input[%d]", node->name().c_str(), dynamic_tensor_num);
    } else if (flag == "out") {
      const bool is_pushback = (out_pos == -1);
      (void)op_desc->AddDynamicOutputDesc(dynamic_name, static_cast<uint32_t>(dynamic_tensor_num), is_pushback);
      (void)ge::AttrUtils::SetInt(op_desc, DYNAMIC_OUTPUT_TD_NUM(dynamic_name), dynamic_tensor_num);
      GELOGI("In NodeDef %s add dynamic output[%d]", node->name().c_str(), dynamic_tensor_num);
    } else {
      // no operation
    }
  }
  return SUCCESS;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY Status AutoMappingByOpFnDynamic(const ge::Operator &op_src,
    ge::Operator &op, const vector<DynamicInputOutputInfo> &dynamic_name_attr_value) {
  // 1. auto mapping for parser
  const std::shared_ptr<ge::OpDesc> op_desc_dst = ge::OpDescUtils::GetOpDescFromOperator(op);
  GE_CHECK_NOTNULL(op_desc_dst);

  const Status ret = AutoMappingByOpFn(op_src, op);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Mapping][Operator]op_name:%s", op_desc_dst->GetName().c_str());
    return FAILED;
  }

  GELOGI("Op[%s] call auto mapping function success.", op_desc_dst->GetName().c_str());
  // 2. check dynamic input output info;
  if (CheckDynamicInfo(dynamic_name_attr_value) != SUCCESS) {
    GELOGE(FAILED, "[Check][DynamicInfo]op_name:%s", op_desc_dst->GetName().c_str());
    return FAILED;
  }
  // 3. update dynamic input output index by tensor num;
  std::map<string, DynamicInfo> port_dynamic_info;
  if (UpdateDynamicInputOutPutIndex(op_desc_dst, dynamic_name_attr_value, port_dynamic_info) != SUCCESS) {
    GELOGE(FAILED, "[Update][DynamicIndex]op_name:%s", op_desc_dst->GetName().c_str());
    return FAILED;
  }
  // 4. sort map by port name insert index.
  vector<pair<string, DynamicInfo>> port_dynamic_info_vec(port_dynamic_info.cbegin(), port_dynamic_info.cend());
  std::sort(port_dynamic_info_vec.begin(), port_dynamic_info_vec.end(),
            [](const pair<string, DynamicInfo> &p1, const pair<string, DynamicInfo> &p2)
            { return p1.second.GetInsetIndex() < p2.second.GetInsetIndex(); });
  // 5. add dynamic input and output
  for (const auto &dynamic_info : port_dynamic_info_vec) {
    const string port_name = dynamic_info.first;
    const DynamicType dynamic_type = dynamic_info.second.GetType();
    const uint32_t insert_index = dynamic_info.second.GetInsetIndex();
    const uint32_t tensor_num = dynamic_info.second.GetTensorNum();

    if (dynamic_type == kInput) {
      (void)op_desc_dst->AddInputDescMiddle(port_name, tensor_num, static_cast<size_t>(insert_index));
      (void)ge::AttrUtils::SetInt(op_desc_dst, DYNAMIC_INPUT_TD_NUM(port_name), static_cast<int64_t>(tensor_num));
      GELOGI("Op[%s] add dynamic input[%u]", op_desc_dst->GetName().c_str(), tensor_num);
    } else if (dynamic_type == kOutput) {
      (void)op_desc_dst->AddOutputDescMiddle(port_name, tensor_num, static_cast<size_t>(insert_index));
      (void)ge::AttrUtils::SetInt(op_desc_dst, DYNAMIC_OUTPUT_TD_NUM(port_name),
                                  static_cast<int64_t>(tensor_num));
      GELOGI("Op[%s] add dynamic output[%u]", op_desc_dst->GetName().c_str(), tensor_num);
    } else {
      GELOGW("Do not add input or output desc with dynamic type :[%d].", static_cast<int32_t>(dynamic_type));
      continue;
    }
  }

  return SUCCESS;
}

// Convert tensorflow property to ge property
FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY Status OperatorAutoMapping(const Message *op_src, ge::Operator &op) {
  std::shared_ptr<ge::OpDesc> op_dst = ge::OpDescUtils::GetOpDescFromOperator(op);
  // Analysis of tensorflow operator parameters based on key value
  GE_CHECK_NOTNULL(op_src);
  GE_CHECK_NOTNULL(op_dst);

  const auto subgraph_attr_names = GetSubgraphAttrNames(op);
  const domi::tensorflow::NodeDef *const node_src = ge::PtrToPtr<const ascend_private::protobuf::Message,
                                                                 const domi::tensorflow::NodeDef>(op_src);
  GE_CHECK_NOTNULL(node_src);
  op_dst->SetName(node_src->name());
  for (const auto &attr_pair : node_src->attr()) {
    if ((attr_pair.first == kTfInputDesc) || (attr_pair.first == kTfOutputDesc)) {
      continue;
    }
    if (subgraph_attr_names.count(attr_pair.first) > 0U) {
      const Status ret = AutoMappingFunction(attr_pair, op_dst);
      if (ret != SUCCESS) {
        return ret;
      }
    } else {
      ge::AutoMappingUtil::ConvertValue(attr_pair.first, attr_pair.second, op_dst);
    }
  }

  const Status ret = SetOpdescInputOutputFormat(op_dst);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "[Set][Format]Set op[%s] desc input output format failed.", op_dst->GetName().c_str());
    return FAILED;
  }
  return SUCCESS;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY Status AutoMappingFn(const Message *op_src, ge::Operator &op) {
  return OperatorAutoMapping(op_src, op);
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY Status AutoMappingByOpFn(const ge::Operator &op_src,
                                                                          ge::Operator &op) {
  const std::shared_ptr<ge::OpDesc> op_desc_src = ge::OpDescUtils::GetOpDescFromOperator(op_src);
  std::shared_ptr<ge::OpDesc> op_desc_dst = ge::OpDescUtils::GetOpDescFromOperator(op);
  GE_CHECK_NOTNULL(op_desc_src);
  GE_CHECK_NOTNULL(op_desc_dst);

  op_desc_dst->SetName(op_desc_src->GetName());
  const auto subgraph_name_indexs = op_desc_src->GetSubgraphNameIndexes();
  for (const auto &subgraph_name_index : subgraph_name_indexs) {
    const auto ret = op_desc_dst->AddSubgraphName(subgraph_name_index.first);
    if (ret != ge::GRAPH_SUCCESS) {
      GELOGW("[AutoMappingFn][Check] %s subgraph of node %s, type %s already exist.",
             subgraph_name_index.first.c_str(), op_desc_dst->GetName().c_str(), op_desc_dst->GetType().c_str());
    }
  }

  const auto subgraph_instance_names = op_desc_src->GetSubgraphInstanceNames();
  uint32_t index = 0U;
  for (const auto &subgraph_instance_name : subgraph_instance_names) {
      const auto ret = op_desc_dst->SetSubgraphInstanceName(index, subgraph_instance_name);
      if (ret != ge::GRAPH_SUCCESS) {
        GELOGE(FAILED, "[Add][SubGraphInstance] subgraph_name: %s, index: %u, for node %s type %s.",
               subgraph_instance_name.c_str(), index, op_desc_dst->GetType().c_str(), op_desc_dst->GetName().c_str());
        return FAILED;
      }
      index++;
  }

  for (const auto &iter : op_desc_src->GetAllAttrs()) {
    (void) op_desc_dst->SetAttr(iter.first, iter.second);
  }

  const Status ret = SetOpdescInputOutputFormat(op_desc_dst);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "[Set][Format]op_name:%s", op_desc_dst->GetName().c_str());
    return FAILED;
  }
  return SUCCESS;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY
Status AutoMappingSubgraphIndex(const ge::Graph &graph,
                                const std::function<int32_t(int32_t data_index)> &input,
                                const std::function<int32_t(int32_t netoutput_index)> &output) {
  GE_CHECK_NOTNULL(input);
  GE_CHECK_NOTNULL(output);
  return AutoMappingSubgraphIndex(graph,
                                  [&input](const int32_t i, int32_t &o) -> Status {
                                    o = input(i);
                                    return SUCCESS;
                                  },
                                  [&output](const int32_t i, int32_t &o) -> Status {
                                    o = output(i);
                                    return SUCCESS;
                                  });
}

namespace {
  std::vector<std::shared_ptr<ge::Node>> FindNodesByType(const ge::ComputeGraphPtr &graph, const std::string &type) {
    std::vector<std::shared_ptr<ge::Node>> nodes;
    for (const auto &node : graph->GetDirectNode()) {
      GELOGI("Find node %s, node type is %s.", type.c_str(), node->GetOpDesc()->GetType().c_str());
      if (node->GetOpDesc()->GetType() == type) {
        nodes.push_back(node);
        continue;
      }
      if (node->GetOpDesc()->GetType() == "FrameworkOp") {
        std::string original_type;
        if (!ge::AttrUtils::GetStr(node->GetOpDesc(), ge::ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, original_type)) {
          // if there is no ref index on the TensorDesc, it means the output data will be ignored outer.
          continue;
        }
        if (original_type == type) {
          nodes.push_back(node);
        }
      }
    }
    return nodes;
  }
}

static Status AutoMappingSubgraphOutput(const ge::ComputeGraphPtr &graph,
                                        const std::function<Status(int32_t netoutput_index,
                                                                   int32_t &parent_output_index)> &output) {
  GE_CHECK_NOTNULL(graph);
  GE_CHECK_NOTNULL(output);
  const auto &output_node = graph->FindFirstNodeMatchType(ge::NETOUTPUT);
  if (output_node == nullptr) {   // Graph from parser no NetOutput.
    return SUCCESS;
  }

  const auto &op_desc = output_node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);

  for (size_t index = 0U; index < op_desc->GetInputsSize(); ++index) {
    int32_t parent_index = -1;
    const auto ret = output(static_cast<int32_t>(index), parent_index);
    if (ret != SUCCESS) {
      GELOGE(FAILED, "[Get][ParentIndex:output]net output index %ld, error code %u", index, ret);
      return FAILED;
    }

    GELOGI("Generate subgraph output map for subgraph %s, index %ld, parent node index %d",
           graph->GetName().c_str(), index, parent_index);
    if (parent_index == -1) {
      continue;
    }

    const ge::GeTensorDescPtr tensor = op_desc->MutableInputDesc(static_cast<uint32_t>(index));
    GE_CHECK_NOTNULL(tensor);
    if (!ge::AttrUtils::SetInt(tensor, ge::ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
      GELOGE(FAILED, "[Set][Attr:%s]Failed for graph %s, op_name:%s, parent_index:%d",
             ge::ATTR_NAME_PARENT_NODE_INDEX.c_str(), graph->GetName().c_str(),
             op_desc->GetName().c_str(), parent_index);
      return FAILED;
    }
  }

  return SUCCESS;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY
static Status AutoMappingSubgraphIndexByDataNode(const ge::ComputeGraphPtr &compute_graph,
    const std::function<Status(int32_t data_index, int32_t &parent_input_index)> &input) {
  const auto nodes = FindNodesByType(compute_graph, "Data");
  for (size_t i = 0U; i < nodes.size(); ++i) {
    int32_t parent_index = -1;
    int32_t index = -1;
    if (!ge::AttrUtils::GetInt(nodes[i]->GetOpDesc(), "index", index)) {
      GELOGE(FAILED, "[Get][Attr:index]data_index:%zu, op_name:%s", i, nodes[i]->GetOpDesc()->GetName().c_str());
      return FAILED;
    }
    GELOGI("Get index %d from data[%zu]", index, i);
    const auto ret = input(index, parent_index);
    if (ret != SUCCESS) {
      GELOGE(FAILED, "[Get][ParentIndex:input]data index %zu, error code %u", i, ret);
      return FAILED;
    }
    if (!ge::AttrUtils::SetInt(nodes[i]->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
      GELOGE(FAILED, "[Set][Attr:%s]data_index:%zu, op_name:%s, ",
             ge::ATTR_NAME_PARENT_NODE_INDEX.c_str(), i, nodes[i]->GetName().c_str());
      return FAILED;
    }
    GELOGI("Generate subgraph input map for subgraph %s, data index %zu, parent node index %d",
           compute_graph->GetName().c_str(), i, parent_index);
  }
  return SUCCESS;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY
Status AutoMappingSubgraphIndex(const ge::Graph &graph,
                                const std::function<Status(int32_t data_index, int32_t &parent_input_index)> &input,
                                const std::function<Status(int32_t netoutput_index,
                                                           int32_t &parent_output_index)> &output) {
  GE_CHECK_NOTNULL(input);
  GE_CHECK_NOTNULL(output);
  const auto compute_graph = ge::GraphUtilsEx::GetComputeGraph(graph);
  GE_CHECK_NOTNULL(compute_graph);
  ge::AscendString graph_name;
  (void) graph.GetName(graph_name);
  auto ret = AutoMappingSubgraphIndexByDataNode(compute_graph, input);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Mapping][Index] auto mapping graph:%s input index failed,", graph_name.GetString());
    return ret;
  }

  const auto nodes = FindNodesByType(compute_graph, "_Retval");
  for (auto &retval : nodes) {
    int64_t index = -1;
    if (!ge::AttrUtils::GetInt(retval->GetOpDesc(), "retval_index", index)) {
      GELOGE(FAILED, "[Get][Attr:retval_index]retval index %ld, op_name:%s",
             index, retval->GetOpDesc()->GetName().c_str());
      return FAILED;
    }
    int32_t parent_index = -1;
    ret = output(static_cast<int32_t>(index), parent_index);
    if (ret != SUCCESS) {
      GELOGE(FAILED, "[Get][ParentIndex:output]retval index %ld, error code %u", index, ret);
      return FAILED;
    }
    if (!ge::AttrUtils::SetInt(retval->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
      GELOGE(FAILED, "[Set][Attr:%s]op_name:%s, parent_index:%d",
             ge::ATTR_NAME_PARENT_NODE_INDEX.c_str(), retval->GetName().c_str(), parent_index);
      return FAILED;
    }
    GELOGI("Generate subgraph output map for subgraph %s, retval index %ld, parent node index %d",
           graph_name.GetString(), index, parent_index);
  }

  return nodes.empty() ? AutoMappingSubgraphOutput(compute_graph, output) : SUCCESS;
}

class FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY FrameworkRegistryImpl {
 public:
  void AddAutoMappingSubgraphIOIndexFunc(const domi::FrameworkType framework, AutoMappingSubgraphIOIndexFunc fun);
  AutoMappingSubgraphIOIndexFunc GetAutoMappingSubgraphIOIndexFunc(const domi::FrameworkType framework);
 private:
  std::map<domi::FrameworkType, AutoMappingSubgraphIOIndexFunc> fmk_type_to_auto_mapping_subgraph_index_fun_;
};

void FrameworkRegistryImpl::AddAutoMappingSubgraphIOIndexFunc(
    const domi::FrameworkType framework, AutoMappingSubgraphIOIndexFunc fun) {
  GELOGD("Regitser auto mapping function: framework type:%d.", framework);
  fmk_type_to_auto_mapping_subgraph_index_fun_[framework] = std::move(fun);
}

AutoMappingSubgraphIOIndexFunc FrameworkRegistryImpl::GetAutoMappingSubgraphIOIndexFunc(
    const domi::FrameworkType framework) {
  const auto itr = fmk_type_to_auto_mapping_subgraph_index_fun_.find(framework);
  if (itr != fmk_type_to_auto_mapping_subgraph_index_fun_.end()) {
    return itr->second;
  }
  return nullptr;
}

FrameworkRegistry::FrameworkRegistry() {
  impl_ = ge::ComGraphMakeUnique<FrameworkRegistryImpl>();
  if (impl_ == nullptr) {
    GELOGW("[Check][Param] make impl failed");
  }
}

FrameworkRegistry::~FrameworkRegistry() = default;

FrameworkRegistry& FrameworkRegistry::Instance() {
  static FrameworkRegistry instance;
  return instance;
}

void FrameworkRegistry::AddAutoMappingSubgraphIOIndexFunc(
    domi::FrameworkType framework, AutoMappingSubgraphIOIndexFunc fun) {
  if (impl_ != nullptr) {
    impl_->AddAutoMappingSubgraphIOIndexFunc(framework, fun);
  }
}

AutoMappingSubgraphIOIndexFunc FrameworkRegistry::GetAutoMappingSubgraphIOIndexFunc(
    domi::FrameworkType framework) {
  if (impl_ != nullptr) {
    return impl_->GetAutoMappingSubgraphIOIndexFunc(framework);
  }
  return nullptr;
}

AutoMappingSubgraphIOIndexFuncRegister::AutoMappingSubgraphIOIndexFuncRegister(
    domi::FrameworkType framework, AutoMappingSubgraphIOIndexFunc fun) {
  FrameworkRegistry::Instance().AddAutoMappingSubgraphIOIndexFunc(framework, fun);
}

OpReceiver::OpReceiver(OpRegistrationData &reg_data) { OpRegistry::Instance()->registrationDatas.push_back(reg_data); }

class OpRegistrationDataImpl {
 public:
  OpRegistrationDataImpl() = default;
  ~OpRegistrationDataImpl() = default;
  explicit OpRegistrationDataImpl(const std::string &om_optype);
private:
  friend class OpRegistrationData;
  friend class OpRegistry;
  domi::FrameworkType fmk_type_;
  std::set<std::string> ori_optype_set_;                   // OP type in the original model, there may be multiple
  std::string om_optype_;                                  // OP type in OM model
  domi::ImplyType imply_type_;                             // execution type
  ParseParamFunc parseParamFn_;                            // parseParam function
  ParseParamByOpFunc parse_param_by_op_fn_;                // parse param by op function
  FusionParseParamFunc fusionParseParamFn_;                // fusion parseParam function
  FusionParseParamByOpFunc fusion_parse_param_by_op_fn_;   // fusion parseParam by op function
  ParseSubgraphFunc parse_subgraph_post_fn_;               // a function called after the subgraph was generated
  ParseSubgraphFuncV2 parse_subgraph_post_fn_v2_;          // a function called after the subgraph was generated
  std::vector<RemoveInputConfigure> remove_input_configure_vec_;
  ParseOpToGraphFunc parse_op_to_graph_fn_;
};

OpRegistrationDataImpl::OpRegistrationDataImpl(const std::string &om_optype)
    : fmk_type_(FRAMEWORK_RESERVED),
      om_optype_(om_optype),
      imply_type_(domi::ImplyType::BUILDIN),
      parseParamFn_(nullptr),
      parse_param_by_op_fn_(nullptr),
      fusionParseParamFn_(nullptr),
      fusion_parse_param_by_op_fn_(nullptr),
      parse_subgraph_post_fn_(nullptr),
      parse_subgraph_post_fn_v2_(nullptr),
      parse_op_to_graph_fn_(nullptr) {}

OpRegistrationData::~OpRegistrationData() = default;

OpRegistrationData::OpRegistrationData(const std::string &om_optype) {
  impl_ = ge::ComGraphMakeShared<OpRegistrationDataImpl>(om_optype);
  if (impl_ == nullptr) {
    GELOGW("[Check][Param] make impl failed!");
  }
}

OpRegistrationData::OpRegistrationData(const char_t *om_optype) {
  std::string op_type;
  if (om_optype != nullptr) {
    op_type = om_optype;
  }
  impl_ = ge::ComGraphMakeShared<OpRegistrationDataImpl>(op_type);
  if (impl_ == nullptr) {
    GELOGW("[Check][Param] make impl failed!");
  }
}

std::string OpRegistrationData::GetOmOptype() const {
  if (impl_ != nullptr) {
    return impl_->om_optype_;
  }
  return "";
}

Status OpRegistrationData::GetOmOptype(ge::AscendString &om_op_type) const {
  if (impl_ != nullptr) {
    om_op_type = ge::AscendString(impl_->om_optype_.c_str());
  }
  return SUCCESS;
}

OpRegistrationData &OpRegistrationData::FrameworkType(const domi::FrameworkType &fmk_type) {
  if (impl_ != nullptr) {
    impl_->fmk_type_ = fmk_type;
  }
  return *this;
}

domi::FrameworkType OpRegistrationData::GetFrameworkType() const {
  if (impl_ != nullptr) {
    return impl_->fmk_type_;
  }
  return FRAMEWORK_RESERVED;
}

OpRegistrationData &OpRegistrationData::OriginOpType(const std::initializer_list<std::string> &ori_optype_list) {
  if (impl_ != nullptr) {
    for (const auto &ori_optype : ori_optype_list) {
      (void)impl_->ori_optype_set_.insert(ori_optype);
    }
  }
  return *this;
}

OpRegistrationData &OpRegistrationData::OriginOpType(const std::vector<ge::AscendString> &ori_op_type_list) {
  if (impl_ != nullptr) {
    for (auto &ori_op_type : ori_op_type_list) {
      std::string tmp_ori_op_type;
      if (ori_op_type.GetString() != nullptr) {
        tmp_ori_op_type = ori_op_type.GetString();
      }
      (void)impl_->ori_optype_set_.insert(tmp_ori_op_type);
    }
  }
  return *this;
}

OpRegistrationData &OpRegistrationData::OriginOpType(const std::string &ori_optype) {
  if (impl_ != nullptr) {
    (void)impl_->ori_optype_set_.insert(ori_optype);
  }
  return *this;
}

OpRegistrationData &OpRegistrationData::OriginOpType(const char_t *ori_op_type) {
  if (impl_ != nullptr) {
    std::string tmp_ori_op_type;
    if (ori_op_type != nullptr) {
      tmp_ori_op_type =  ori_op_type;
    }
    (void)impl_->ori_optype_set_.insert(tmp_ori_op_type);
  }
  return *this;
}

std::set<std::string> OpRegistrationData::GetOriginOpTypeSet() const {
  const std::set<std::string> ori_optype_set;
  if (impl_ != nullptr) {
    return impl_->ori_optype_set_;
  }
  return ori_optype_set;
}

Status OpRegistrationData::GetOriginOpTypeSet(std::set<ge::AscendString> &ori_op_type) const {
  std::set<std::string> ori_op_type_set;
  if (impl_ != nullptr) {
    ori_op_type_set =  impl_->ori_optype_set_;
  }
  for (auto &op_type : ori_op_type_set) {
    (void)ori_op_type.insert(ge::AscendString(op_type.c_str()));
  }
  return SUCCESS;
}

OpRegistrationData &OpRegistrationData::ParseParamsFn(const ParseParamFunc &parseParamFn) {
  if (impl_ != nullptr) {
    impl_->parseParamFn_ = parseParamFn;
  }
  return *this;
}

ParseParamFunc OpRegistrationData::GetParseParamFn() const {
  if (impl_ != nullptr) {
    return impl_->parseParamFn_;
  }
  return nullptr;
}

OpRegistrationData &OpRegistrationData::ParseParamsByOperatorFn(const ParseParamByOpFunc &parse_param_by_op_fn) {
  if (impl_ != nullptr) {
    impl_->parse_param_by_op_fn_ = parse_param_by_op_fn;
  }
  return *this;
}

ParseParamByOpFunc OpRegistrationData::GetParseParamByOperatorFn() const {
  if (impl_ != nullptr) {
    return impl_->parse_param_by_op_fn_;
  }
  return nullptr;
}

OpRegistrationData &OpRegistrationData::FusionParseParamsFn(const FusionParseParamFunc &fusionParseParamFn) {
  if (impl_ != nullptr) {
   impl_->fusionParseParamFn_ = fusionParseParamFn;
  }
  return *this;
}

FusionParseParamFunc OpRegistrationData::GetFusionParseParamFn() const {
  if (impl_ != nullptr) {
    return impl_->fusionParseParamFn_;
  }
  return nullptr;
}

OpRegistrationData &OpRegistrationData::FusionParseParamsFn(const FusionParseParamByOpFunc &fusion_parse_param_fn) {
  if (impl_ != nullptr) {
    impl_->fusion_parse_param_by_op_fn_ = fusion_parse_param_fn;
  }
  return *this;
}

FusionParseParamByOpFunc OpRegistrationData::GetFusionParseParamByOpFn() const {
  if (impl_ != nullptr) {
    return impl_->fusion_parse_param_by_op_fn_;
  }
  return nullptr;
}

OpRegistrationData &OpRegistrationData::ImplyType(const domi::ImplyType &imply_type) {
  if (impl_ != nullptr) {
    impl_->imply_type_ = imply_type;
  }
  return *this;
}

domi::ImplyType OpRegistrationData::GetImplyType() const {
  constexpr domi::ImplyType imply_type = domi::ImplyType::BUILDIN;
  if (impl_ != nullptr) {
    return impl_->imply_type_;
  }
  return imply_type;
}

OpRegistrationData &OpRegistrationData::DelInputWithCond(int32_t inputIdx, const std::string &attrName,
                                                         bool attrValue) {
  if (impl_ != nullptr) {
    struct RemoveInputConfigure registerStu;
    registerStu.inputIdx = inputIdx;
    registerStu.attrName = attrName;
    registerStu.moveType = RemoveInputType::OMG_REMOVE_TYPE_WITH_COND;
    registerStu.attrValue = attrValue;
    impl_->remove_input_configure_vec_.push_back(registerStu);
  }
  return *this;
}

OpRegistrationData &OpRegistrationData::DelInputWithCond(int32_t input_idx, const char_t *attr_name, bool attr_value) {
  std::string tmp_attr_name;
  if (attr_name != nullptr) {
    tmp_attr_name = attr_name;
  }
  if (impl_ != nullptr) {
    struct RemoveInputConfigure registerStu;
    registerStu.inputIdx = input_idx;
    registerStu.attrName = tmp_attr_name;
    registerStu.moveType = RemoveInputType::OMG_REMOVE_TYPE_WITH_COND;
    registerStu.attrValue = attr_value;
    impl_->remove_input_configure_vec_.push_back(registerStu);
  }
  return *this;
}

OpRegistrationData &OpRegistrationData::InputReorderVector(const vector<int32_t> &input_order) {
  if (impl_ != nullptr) {
    struct RemoveInputConfigure register_input;
    register_input.inputIdx = 0;
    register_input.input_order = input_order;
    register_input.moveType = RemoveInputType::OMG_INPUT_REORDER;
    impl_->remove_input_configure_vec_.push_back(register_input);
  }
  return *this;
}

OpRegistrationData &OpRegistrationData::DelInputWithOriginalType(int32_t input_idx, const std::string &ori_type) {
  if (impl_ != nullptr) {
    struct RemoveInputConfigure register_input;
    register_input.inputIdx = input_idx;
    register_input.originalType = ori_type;
    register_input.moveType = RemoveInputType::OMG_REMOVE_INPUT_WITH_ORIGINAL_TYPE;
    impl_->remove_input_configure_vec_.push_back(register_input);
  }
  return *this;
}

OpRegistrationData &OpRegistrationData::DelInputWithOriginalType(int32_t input_idx, const char_t *ori_type) {
  std::string tmp_ori_type;
  if (ori_type != nullptr) {
    tmp_ori_type = ori_type;
  }
  if (impl_ != nullptr) {
    struct RemoveInputConfigure register_input;
    register_input.inputIdx = input_idx;
    register_input.originalType = tmp_ori_type;
    register_input.moveType = RemoveInputType::OMG_REMOVE_INPUT_WITH_ORIGINAL_TYPE;
    impl_->remove_input_configure_vec_.push_back(register_input);
  }
  return *this;
}

OpRegistrationData &OpRegistrationData::ParseSubgraphPostFn(const ParseSubgraphFunc &subgraph_post_fn) {
  if (impl_ != nullptr) {
    impl_->parse_subgraph_post_fn_ = subgraph_post_fn;
  }
  return *this;
}

ParseSubgraphFunc OpRegistrationData::GetParseSubgraphPostFn() const {
  if (impl_ == nullptr) {
    return nullptr;
  }
  return impl_->parse_subgraph_post_fn_;
}

OpRegistrationData &OpRegistrationData::ParseOpToGraphFn(const ParseOpToGraphFunc &parse_op_to_graph_fn) {
  if (impl_ != nullptr) {
    impl_->parse_op_to_graph_fn_ = parse_op_to_graph_fn;
  }
  return *this;
}

OpRegistrationData &OpRegistrationData::ParseSubgraphPostFn(const ParseSubgraphFuncV2 &subgraph_post_fn) {
  if (impl_ != nullptr) {
    impl_->parse_subgraph_post_fn_v2_ = subgraph_post_fn;
  }
  return *this;
}

ParseOpToGraphFunc OpRegistrationData::GetParseOpToGraphFn() const {
  if (impl_ == nullptr) {
    return nullptr;
  }
  return impl_->parse_op_to_graph_fn_;
}

Status OpRegistrationData::GetParseSubgraphPostFn(ParseSubgraphFuncV2 &func) const {
  if (impl_ == nullptr) {
    return FAILED;
  }
  func = impl_->parse_subgraph_post_fn_v2_;
  return SUCCESS;
}

OpRegistry *OpRegistry::Instance() {
  static OpRegistry g_instance;
  return &g_instance;
}

namespace {
std::string GetParserKey(const std::string &om_type, const std::string &ori_type) {
  return om_type + "_" + ori_type;
}
} // namespace

bool OpRegistry::Register(const OpRegistrationData &reg_data, bool is_custom_op) {
  if (reg_data.impl_ == nullptr) {
    return false;
  }
  for (const auto &ori_type : reg_data.impl_->ori_optype_set_) {
    const std::string om_ori_type = GetParserKey(reg_data.impl_->om_optype_, ori_type);
    if ((op_parse_params_fn_map_.find(om_ori_type) != op_parse_params_fn_map_.end()) && !is_custom_op) {
      GELOGI("[Register][Check] Plugin of op type:%s, original type:%s already registered, skip",
             reg_data.impl_->om_optype_.c_str(), ori_type.c_str());
      continue;
    }

    GELOGD("The plugin of type:%s will be registered.", om_ori_type.c_str());
    op_parse_params_fn_map_[om_ori_type] = reg_data.impl_->parseParamFn_;
    fusion_op_parse_params_fn_map_[om_ori_type] = reg_data.impl_->fusionParseParamFn_;
    fusion_parse_params_by_op_fn_map_[om_ori_type] = reg_data.impl_->fusion_parse_param_by_op_fn_;
    parse_params_by_op_func_map_[om_ori_type] = reg_data.impl_->parse_param_by_op_fn_;
    remove_input_configure_map_[om_ori_type] = reg_data.impl_->remove_input_configure_vec_;
    parse_op_to_graph_fn_map_[om_ori_type] = reg_data.impl_->parse_op_to_graph_fn_;

    if (origin_type_to_om_type_.find(ori_type) == origin_type_to_om_type_.end()) {
      origin_type_to_om_type_[ori_type] = reg_data.impl_->om_optype_;
    }
  }

  if ((op_run_mode_map_.find(reg_data.impl_->om_optype_) != op_run_mode_map_.end()) && !is_custom_op) {
    GELOGI("[Register][Check] Plugin of %s already registered, skip", reg_data.impl_->om_optype_.c_str());
    return true;
  }
  op_run_mode_map_[reg_data.impl_->om_optype_] = reg_data.impl_->imply_type_;
  op_types_to_parse_subgraph_post_func_[reg_data.impl_->om_optype_] = reg_data.impl_->parse_subgraph_post_fn_;
  op_types_to_parse_subgraph_post_func_v2_[reg_data.impl_->om_optype_] = reg_data.impl_->parse_subgraph_post_fn_v2_;
  return true;
}

domi::ImplyType OpRegistry::GetImplyTypeByOriOpType(const std::string &ori_optype) {
  domi::ImplyType result = domi::ImplyType::BUILDIN;
  const auto iter = origin_type_to_om_type_.find(ori_optype);
  if (iter != origin_type_to_om_type_.end()) {
    result = GetImplyType(iter->second);
  }
  return result;
}

domi::ImplyType OpRegistry::GetImplyType(const std::string &op_type) {
  const auto it_find = op_run_mode_map_.find(op_type);
  if (it_find == op_run_mode_map_.end()) {
    return domi::ImplyType::BUILDIN;
  }
  return it_find->second;
}

domi::ParseParamByOpFunc OpRegistry::GetParseParamByOperatorFunc(const std::string &ori_type) {
  std::string om_type;
  const auto iter = origin_type_to_om_type_.find(ori_type);
  if (iter != origin_type_to_om_type_.end()) {
    om_type = iter->second;
  }
  const std::string type = GetParserKey(om_type, ori_type);
  const auto it_find = parse_params_by_op_func_map_.find(type);
  if (it_find == parse_params_by_op_func_map_.end()) {
    return nullptr;
  }
  return it_find->second;
}

domi::ParseParamFunc OpRegistry::GetParseParamFunc(const std::string &op_type, const std::string &ori_type) {
  const std::string type = GetParserKey(op_type, ori_type);
  const auto it_find = op_parse_params_fn_map_.find(type);
  if (it_find == op_parse_params_fn_map_.end()) {
    return nullptr;
  }
  return it_find->second;
}

domi::FusionParseParamFunc OpRegistry::GetFusionParseParamFunc(const std::string &op_type,
                                                               const std::string &ori_type) {
  const std::string type = GetParserKey(op_type, ori_type);
  const auto it_find = fusion_op_parse_params_fn_map_.find(type);
  if (it_find == fusion_op_parse_params_fn_map_.end()) {
    return nullptr;
  }
  return it_find->second;
}

domi::FusionParseParamByOpFunc OpRegistry::GetFusionParseParamByOpFunc(const std::string &op_type,
                                                                       const std::string &ori_type) {
  const std::string type = GetParserKey(op_type, ori_type);
  const auto it_find = fusion_parse_params_by_op_fn_map_.find(type);
  if (it_find == fusion_parse_params_by_op_fn_map_.end()) {
    return nullptr;
  }
  return it_find->second;
}

domi::ParseSubgraphFunc OpRegistry::GetParseSubgraphPostFunc(const std::string &op_type) {
  const auto it_find = op_types_to_parse_subgraph_post_func_.find(op_type);
  if (it_find == op_types_to_parse_subgraph_post_func_.end()) {
    return nullptr;
  }
  return it_find->second;
}

Status OpRegistry::GetParseSubgraphPostFunc(const std::string &op_type,
                                            domi::ParseSubgraphFuncV2 &parse_subgraph_func) {
  const auto it_find = op_types_to_parse_subgraph_post_func_v2_.find(op_type);
  if (it_find == op_types_to_parse_subgraph_post_func_v2_.end()) {
    return FAILED;
  }
  parse_subgraph_func = it_find->second;
  return SUCCESS;
}

void OpRegistry::GetOpTypeByImplyType(std::vector<std::string> &vec_op_type, const domi::ImplyType imply_type) const {
  for (const auto &iter : op_run_mode_map_) {
    if (iter.second == imply_type) {
      vec_op_type.push_back(iter.first);
    }
  }
  return;
}

const std::vector<RemoveInputConfigure> &OpRegistry::GetRemoveInputConfigure(const std::string &ori_optype) const {
  static const std::vector<RemoveInputConfigure> empty_ = {};
  const auto iter = origin_type_to_om_type_.find(ori_optype);
  if (iter != origin_type_to_om_type_.end()) {
    const std::string type = GetParserKey(iter->second, ori_optype);
    const auto it = remove_input_configure_map_.find(type);
    if (it != remove_input_configure_map_.end()) {
      return it->second;
    }
  }
  return empty_;
}

bool OpRegistry::GetOmTypeByOriOpType(const std::string &ori_optype, std::string &om_type) {
  const auto iter = origin_type_to_om_type_.find(ori_optype);
  if (iter != origin_type_to_om_type_.end()) {
    om_type = iter->second;
    return true;
  }
  return false;
}

ParseOpToGraphFunc OpRegistry::GetParseOpToGraphFunc(const std::string &op_type, const std::string &ori_type) {
  const std::string type = GetParserKey(op_type, ori_type);
  const auto iter = parse_op_to_graph_fn_map_.find(type);
  if (iter == parse_op_to_graph_fn_map_.end()) {
    return nullptr;
  }
  return iter->second;
}
/*lint +e1073*/
}  // namespace domi
