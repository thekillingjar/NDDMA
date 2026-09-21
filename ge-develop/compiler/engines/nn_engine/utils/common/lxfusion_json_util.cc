/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/lxfusion_json_util.h"
#include "common/fe_log.h"
#include "common/aicore_util_attr_define.h"
#include "graph/debug/ge_attr_define.h"
#include "runtime/kernel.h"

namespace {
const std::string RtSmData_L2MirrorAddr = "L2_mirror_addr";
const std::string RtSmData_L2DataSectionSize = "L2_data_section_size";
const std::string RtSmData_L2Preload = "L2_preload";
const std::string RtSmData_Modified = "modified";
const std::string RtSmData_Priority = "priority";
const std::string RtSmData_PrevL2PageOffsetBase = "prev_L2_page_offset_base";
const std::string RtSmData_L2PageOffsetBase = "L2_page_offset_base";
const std::string RtSmData_L2LoadToDDR = "L2_load_to_ddr";

const std::string RtSmDesc_Data = "data";
const std::string RtSmDesc_Size = "size";
const std::string RtSmDesc_Remap = "remap";
const std::string RtSmDesc_L2InMain = "l2_in_main";
}

void from_json(const nlohmann::json& json_value, rtSmData_t& rtSmData) {
  json_value.at(RtSmData_L2MirrorAddr).get_to(rtSmData.L2_mirror_addr);
  json_value.at(RtSmData_L2DataSectionSize).get_to(rtSmData.L2_data_section_size);
  json_value.at(RtSmData_L2Preload).get_to(rtSmData.L2_preload);
  json_value.at(RtSmData_Modified).get_to(rtSmData.modified);
  json_value.at(RtSmData_Priority).get_to(rtSmData.priority);
  json_value.at(RtSmData_PrevL2PageOffsetBase).get_to(rtSmData.prev_L2_page_offset_base);
  json_value.at(RtSmData_L2PageOffsetBase).get_to(rtSmData.L2_page_offset_base);
  json_value.at(RtSmData_L2LoadToDDR).get_to(rtSmData.L2_load_to_ddr);
}

void from_json(const nlohmann::json& json_value, rtSmDesc_t& rtSmDesc) {
  for (size_t i = 0; i < json_value[RtSmDesc_Data].size(); ++i) {
    json_value[RtSmDesc_Data][i].get_to(rtSmDesc.data[i]);
  }
  json_value.at(RtSmDesc_Size).get_to(rtSmDesc.size);
  for (size_t i = 0; i < json_value[RtSmDesc_Remap].size(); ++i) {
    json_value[RtSmDesc_Remap][i].get_to(rtSmDesc.remap[i]);
  }
  json_value.at(RtSmDesc_L2InMain).get_to(rtSmDesc.l2_in_main);
}

void to_json(nlohmann::json& json_value, const rtSmData_t& rtSmData) {
  json_value = nlohmann::json{{RtSmData_L2MirrorAddr, rtSmData.L2_mirror_addr},
                              {RtSmData_L2DataSectionSize, rtSmData.L2_data_section_size},
                              {RtSmData_L2Preload, rtSmData.L2_preload},
                              {RtSmData_Modified, rtSmData.modified},
                              {RtSmData_Priority, rtSmData.priority},
                              {RtSmData_PrevL2PageOffsetBase, rtSmData.prev_L2_page_offset_base},
                              {RtSmData_L2PageOffsetBase, rtSmData.L2_page_offset_base},
                              {RtSmData_L2LoadToDDR, rtSmData.L2_load_to_ddr}};
}

void to_json(nlohmann::json& json_value, const rtSmDesc_t& rtSmDesc) {
  json_value = nlohmann::json{{RtSmDesc_Data, rtSmDesc.data},
                              {RtSmDesc_Size, rtSmDesc.size},
                              {RtSmDesc_Remap, rtSmDesc.remap},
                              {RtSmDesc_L2InMain, rtSmDesc.l2_in_main}};
}

namespace fe {
const std::string L1_FUSION_TO_OP_STRUCT = "_l1fusion_ToOpStruct";
const std::string L2_FUSION_TO_OP_STRUCT = "_l2fusion_ToOpStruct";
namespace {
const std::string L1_FUSION_EXTEND_CONTENT = "_l1_fusion_extend_content";
const std::string L2_FUSION_EXTEND_CONTENT = "l2_fusion_extend_content";
const std::string TASK_L2_FUSION_INFO = "_task_L2FusionInfo";
const std::string TASK_L2_FUSION_INFO_EXTEND_CONTENT = "task_l2_fusion_info_extend_content";

const vector<std::string> SLICE_INFO_MODE = {fe::OP_SLICE_INFO, fe::FUSION_OP_SLICE_INFO};

const std::string ToOpStruct_OpL1Space = "opL1Space";
const std::string ToOpStruct_OpL1FusionType = "opL1FusionType";
const std::string ToOpStruct_OpL1WorkspaceFlag = "opL1WorkspaceFlag";
const std::string ToOpStruct_OpL1WorkspaceSize = "opL1WorkspaceSize";
const std::string ToOpStruct_ValidInputShape = "validInputShape";
const std::string ToOpStruct_ValidOutputShape = "validOutputShape";
const std::string ToOpStruct_SliceInputOffset = "sliceInputOffset";
const std::string ToOpStruct_SliceOutputOffset = "sliceOutputOffset";
const std::string ToOpStruct_TotalShape = "totalShape";
const std::string ToOpStruct_SplitIndex = "splitIndex";

const std::string L2FusionData_L2Index = "l2Index";
const std::string L2FusionData_L2Addr = "l2Addr";
const std::string L2FusionData_L2PageNum = "l2PageNum";

const std::string FeSmDesc_L2Ctrl = "l2ctrl";
const std::string FeSmDesc_NodeName = "nodeName";
const std::string FeSmDesc_OutputIndex = "outputIndex";

const std::string TaskL2FusionInfo_NodeName = "nodeName";
const std::string TaskL2FusionInfo_L2Info = "l2Info";
const std::string TaskL2FusionInfo_Input = "input";
const std::string TaskL2FusionInfo_Output = "output";
const std::string TaskL2FusionInfo_IsUsed = "isUsed";

const std::string InputSplitInfo_Idx = "idx";
const std::string InputSplitInfo_Axis = "axis";
const std::string InputSplitInfo_HeadOverLap = "headOverLap";
const std::string InputSplitInfo_TailOverLap = "tailOverLap";

const std::string OutputSplitInfo_Idx = "idx";
const std::string OutputSplitInfo_Axis = "axis";

const std::string AxisSplitMap_InputList = "inputList";
const std::string AxisSplitMap_OutputList = "outputList";

const std::string InputReduceInfo_Idx = "idx";
const std::string InputReduceInfo_Axis = "axis";

const std::string OutputReduceInfo_Idx = "idx";
const std::string OutputReduceInfo_ReduceType = "reduceType";
const std::string OutputReduceInfo_IsAtomic = "isAtomic";

const std::string AxisReduceMap_InputList = "inputList";
const std::string AxisReduceMap_OutputList = "outputList";

const std::string OpCalcInfo_SplitMaps = "splitMaps";
const std::string OpCalcInfo_ReduceMaps = "reduceMaps";
const std::string OpCalcInfo_L1FusionEnable = "l1FusionEnable";
const std::string OpCalcInfo_MinTbeL1Space = "minTbeL1Space";
}

void from_json(const nlohmann::json& json_value, ToOpStruct_t& op_struct) {
  json_value.at(ToOpStruct_OpL1Space).get_to(op_struct.op_l1_space);
  json_value.at(ToOpStruct_OpL1FusionType).get_to(op_struct.op_l1_fusion_type);
  json_value.at(ToOpStruct_OpL1WorkspaceFlag).get_to(op_struct.op_l1_workspace_flag);
  json_value.at(ToOpStruct_OpL1WorkspaceSize).get_to(op_struct.op_l1_workspace_size);
  json_value.at(ToOpStruct_ValidInputShape).get_to(op_struct.slice_input_shape);
  json_value.at(ToOpStruct_ValidOutputShape).get_to(op_struct.slice_output_shape);
  json_value.at(ToOpStruct_SliceInputOffset).get_to(op_struct.slice_input_offset);
  json_value.at(ToOpStruct_SliceOutputOffset).get_to(op_struct.slice_output_offset);
  json_value.at(ToOpStruct_TotalShape).get_to(op_struct.total_shape);
  json_value.at(ToOpStruct_SplitIndex).get_to(op_struct.split_index);
}

void to_json(nlohmann::json& json_value, const ToOpStruct_t& op_struct) {
  json_value = nlohmann::json{{ToOpStruct_OpL1Space, op_struct.op_l1_space},
                              {ToOpStruct_OpL1FusionType, op_struct.op_l1_fusion_type},
                              {ToOpStruct_OpL1WorkspaceFlag, op_struct.op_l1_workspace_flag},
                              {ToOpStruct_OpL1WorkspaceSize, op_struct.op_l1_workspace_size},
                              {ToOpStruct_ValidInputShape, op_struct.slice_input_shape},
                              {ToOpStruct_ValidOutputShape, op_struct.slice_output_shape},
                              {ToOpStruct_SliceInputOffset, op_struct.slice_input_offset},
                              {ToOpStruct_SliceOutputOffset, op_struct.slice_output_offset},
                              {ToOpStruct_TotalShape, op_struct.total_shape},
                              {ToOpStruct_SplitIndex, op_struct.split_index}};
}

void from_json(const nlohmann::json& json_value, L2FusionData_t& l2_fusion_data) {
  json_value.at(L2FusionData_L2Index).get_to(l2_fusion_data.l2Index);
  json_value.at(L2FusionData_L2Addr).get_to(l2_fusion_data.l2Addr);
  json_value.at(L2FusionData_L2PageNum).get_to(l2_fusion_data.l2PageNum);
}

void from_json(const nlohmann::json& json_value, fe_sm_desc_t& fe_sm_desc) {
  json_value.at(FeSmDesc_L2Ctrl).get_to(fe_sm_desc.l2ctrl);
  for (size_t i = 0; i < json_value[FeSmDesc_NodeName].size(); ++i) {
    json_value[FeSmDesc_NodeName][i].get_to(fe_sm_desc.node_name[i]);
  }
  for (size_t i = 0; i < json_value[FeSmDesc_OutputIndex].size(); ++i) {
    json_value[FeSmDesc_OutputIndex][i].get_to(fe_sm_desc.output_index[i]);
  }
}

void from_json(const nlohmann::json& json_value, TaskL2FusionInfo_t& task_l2_fusion_info) {
  json_value.at(TaskL2FusionInfo_NodeName).get_to(task_l2_fusion_info.node_name);
  json_value.at(TaskL2FusionInfo_L2Info).get_to(task_l2_fusion_info.l2_info);
  json_value.at(TaskL2FusionInfo_Input).get_to(task_l2_fusion_info.input);
  json_value.at(TaskL2FusionInfo_Output).get_to(task_l2_fusion_info.output);
  json_value.at(TaskL2FusionInfo_IsUsed).get_to(task_l2_fusion_info.is_used);
}

void to_json(nlohmann::json& json_value, const L2FusionData_t& l2_fusion_data) {
  json_value = nlohmann::json{{L2FusionData_L2Index, l2_fusion_data.l2Index},
                              {L2FusionData_L2Addr, l2_fusion_data.l2Addr},
                              {L2FusionData_L2PageNum, l2_fusion_data.l2PageNum}};
}

void to_json(nlohmann::json& json_value, const fe_sm_desc_t& fe_sm_desc) {
  json_value = nlohmann::json{{FeSmDesc_L2Ctrl, fe_sm_desc.l2ctrl},
                              {FeSmDesc_NodeName, fe_sm_desc.node_name},
                              {FeSmDesc_OutputIndex, fe_sm_desc.output_index}};
}

void to_json(nlohmann::json& json_value, const TaskL2FusionInfo_t& task_l2_fusion_info) {
  json_value = nlohmann::json{{TaskL2FusionInfo_NodeName, task_l2_fusion_info.node_name},
                              {TaskL2FusionInfo_L2Info, task_l2_fusion_info.l2_info},
                              {TaskL2FusionInfo_Input, task_l2_fusion_info.input},
                              {TaskL2FusionInfo_Output, task_l2_fusion_info.output},
                              {TaskL2FusionInfo_IsUsed, task_l2_fusion_info.is_used}};
}

void from_json(const nlohmann::json& json_value, InputSplitInfo& input_split_info) {
  if (input_split_info.IsPtrNull()) {
    if (!input_split_info.Initialize()) {
      return;
    }
  }
  auto idx = json_value.at(InputSplitInfo_Idx).get<size_t>();
  input_split_info.SetIndex(idx);
  auto axis = json_value.at(InputSplitInfo_Axis).get<std::vector<int64_t>>();
  input_split_info.SetAxis(axis);
  auto head_over_lap = json_value.at(InputSplitInfo_HeadOverLap).get<std::vector<int64_t>>();
  input_split_info.SetHeadOverLap(head_over_lap);
  auto tail_over_lap = json_value.at(InputSplitInfo_TailOverLap).get<std::vector<int64_t>>();
  input_split_info.SetTailOverLap(tail_over_lap);
}

void from_json(const nlohmann::json& json_value, OutputSplitInfo& output_split_info) {
  if (output_split_info.IsPtrNull()) {
    if (!output_split_info.Initialize()) {
      return;
    }
  }
  auto idx = json_value.at(OutputReduceInfo_Idx).get<size_t>();
  output_split_info.SetIndex(idx);
  auto axis = json_value.at(OutputSplitInfo_Axis).get<std::vector<int64_t>>();
  output_split_info.SetAxis(axis);
}

void from_json(const nlohmann::json& json_value, AxisSplitMap& axis_split_map) {
  if (axis_split_map.IsPtrNull()) {
    if (!axis_split_map.Initialize()) {
      return;
    }
  }
  auto input_list = json_value.at(AxisSplitMap_InputList).get<std::vector<InputSplitInfo>>();
  axis_split_map.SetInputSplitInfos(input_list);
  auto output_list = json_value.at(AxisSplitMap_OutputList).get<std::vector<OutputSplitInfo>>();
  axis_split_map.SetOutputSplitInfos(output_list);
}

void from_json(const nlohmann::json& json_value, InputReduceInfo& input_reduce_info) {
  if (input_reduce_info.IsPtrNull()) {
    if (!input_reduce_info.Initialize()) {
      return;
    }
  }
  auto idx = json_value.at(InputReduceInfo_Idx).get<size_t>();
  input_reduce_info.SetIndex(idx);
  auto axis = json_value.at(InputReduceInfo_Axis).get<std::vector<int64_t>>();
  input_reduce_info.SetAxis(axis);
}

void from_json(const nlohmann::json& json_value, OutputReduceInfo& output_reduce_info) {
  if (output_reduce_info.IsPtrNull()) {
    if (!output_reduce_info.Initialize()) {
      return;
    }
  }
  auto idx = json_value.at(OutputReduceInfo_Idx).get<size_t>();
  output_reduce_info.SetIndex(idx);
  auto reduce_type = json_value.at(OutputReduceInfo_ReduceType).get<OpReduceType>();
  output_reduce_info.SetReduceType(reduce_type);
  auto is_atomic = json_value.at(OutputReduceInfo_IsAtomic).get<bool>();
  output_reduce_info.SetIsAtomic(is_atomic);
}

void from_json(const nlohmann::json& json_value, AxisReduceMap& axis_reduce_map) {
  if (axis_reduce_map.IsPtrNull()) {
    if (!axis_reduce_map.Initialize()) {
      return;
    }
  }
  auto input_list = json_value.at(AxisReduceMap_InputList).get<std::vector<InputReduceInfo>>();
  axis_reduce_map.SetInputReduceInfos(input_list);
  auto output_list = json_value.at(AxisReduceMap_OutputList).get<std::vector<OutputReduceInfo>>();
  axis_reduce_map.SetOutputReduceInfos(output_list);
}

void from_json(const nlohmann::json& json_value, OpCalcInfo& op_calc_info) {
  if (op_calc_info.IsPtrNull()) {
    if (!op_calc_info.Initialize()) {
      return;
    }
  }
  auto split_maps = json_value.at(OpCalcInfo_SplitMaps).get<std::vector<AxisSplitMap>>();
  op_calc_info.SetAxisSplitMaps(split_maps);
  auto reduce_maps = json_value.at(OpCalcInfo_ReduceMaps).get<std::vector<AxisReduceMap>>();
  op_calc_info.SetAxisReduceMaps(reduce_maps);
  auto l1_fusion_enable = json_value.at(OpCalcInfo_L1FusionEnable).get<OpL1FusionType>();
  op_calc_info.SetL1FusionEnable(l1_fusion_enable);
  auto min_tbe_l1_space = json_value.at(OpCalcInfo_MinTbeL1Space).get<int64_t>();
  op_calc_info.SetMinTbeL1Space(min_tbe_l1_space);
}

void to_json(nlohmann::json& json_value, const InputSplitInfo& input_split_info) {
  json_value = nlohmann::json{{InputSplitInfo_Idx, input_split_info.GetIndex()},
                             {InputSplitInfo_Axis, input_split_info.GetAxis()},
                             {InputSplitInfo_HeadOverLap, input_split_info.GetHeadOverLap()},
                             {InputSplitInfo_TailOverLap, input_split_info.GetTailOverLap()}};
}

void to_json(nlohmann::json& json_value, const OutputSplitInfo& output_split_info) {
  json_value = nlohmann::json{{OutputSplitInfo_Idx, output_split_info.GetIndex()},
                             {OutputSplitInfo_Axis, output_split_info.GetAxis()}};
}

void to_json(nlohmann::json& json_value, const AxisSplitMap& axis_split_map) {
  json_value = nlohmann::json{{AxisSplitMap_InputList, axis_split_map.GetInputSplitInfoVec()},
                             {AxisSplitMap_OutputList, axis_split_map.GetOutputSplitInfoVec()}};
}

void to_json(nlohmann::json& json_value, const InputReduceInfo& input_reduce_info) {
  json_value = nlohmann::json{{InputReduceInfo_Idx, input_reduce_info.GetIndex()},
                             {InputReduceInfo_Axis, input_reduce_info.GetAxis()}};
}

void to_json(nlohmann::json& json_value, const OutputReduceInfo& output_reduce_info) {
  json_value = nlohmann::json{{OutputReduceInfo_Idx, output_reduce_info.GetIndex()},
                             {OutputReduceInfo_ReduceType, output_reduce_info.GetReduceType()},
                             {OutputReduceInfo_IsAtomic, output_reduce_info.GetIsAtomic()}};
}

void to_json(nlohmann::json& json_value, const AxisReduceMap& axis_reduce_map) {
  json_value = nlohmann::json{{AxisReduceMap_InputList, axis_reduce_map.GetInputReduceInfoVec()},
                             {AxisReduceMap_OutputList, axis_reduce_map.GetOutputReduceInfoVec()}};
}

void to_json(nlohmann::json& json_value, const OpCalcInfo& op_calc_info) {
  json_value = nlohmann::json{{OpCalcInfo_SplitMaps, op_calc_info.GetAxisSplitMapVec()},
                             {OpCalcInfo_ReduceMaps, op_calc_info.GetAxisReduceMapVec()},
                             {OpCalcInfo_L1FusionEnable, op_calc_info.GetL1FusionEnable()},
                             {OpCalcInfo_MinTbeL1Space, op_calc_info.GetMinTbeL1Space()}};
}

Status GetL1InfoFromJson(ge::OpDescPtr op_desc_ptr) {
  if (op_desc_ptr == nullptr) {
    FE_LOGW("The operators description for the node is null.");
    return FAILED;
  }
  // get l1 info
  string str_l1_info;
  (void)ge::AttrUtils::GetStr(op_desc_ptr, L1_FUSION_TO_OP_STRUCT, str_l1_info);
  if (str_l1_info.empty()) {
    FE_LOGW("L1 info is empty.");
    return FAILED;
  } else {
    try {
      nlohmann::json l1_info_json = nlohmann::json::parse(str_l1_info);
      if (l1_info_json.is_null()) {
        FE_LOGW("Json file: %s is empty.", L1_FUSION_TO_OP_STRUCT.c_str());
        return FAILED;
      } else {
        fe::ToOpStructPtr l1_info_ptr = nullptr;
        FE_MAKE_SHARED(l1_info_ptr = std::make_shared<fe::ToOpStruct_t>(), return FAILED);
        FE_CHECK_NOTNULL(l1_info_ptr);
        l1_info_json.at(L1_FUSION_TO_OP_STRUCT).get_to(*l1_info_ptr);
        (void)op_desc_ptr->SetExtAttr(L1_FUSION_EXTEND_CONTENT, l1_info_ptr);
      }
    } catch (...) {
      FE_LOGW("Failed to parse JSON string, please check the input string.");
      return FAILED;
    }
  }
  return SUCCESS;
}

Status GetL2InfoFromJson(ge::OpDescPtr op_desc_ptr) {
  if (op_desc_ptr == nullptr) {
    FE_LOGW("The operators description for the node is null.");
    return FAILED;
  }
  // get l2 info
  string str_l2_info;
  (void)ge::AttrUtils::GetStr(op_desc_ptr, L2_FUSION_TO_OP_STRUCT, str_l2_info);
  if (str_l2_info.empty()) {
    FE_LOGW("L2 info is empty.");
    return FAILED;
  } else {
    try {
      nlohmann::json l2_info_json = nlohmann::json::parse(str_l2_info);
      if (l2_info_json.is_null()) {
        FE_LOGW("Json file: %s is empty.", L2_FUSION_TO_OP_STRUCT.c_str());
        return FAILED;
      } else {
        fe::ToOpStructPtr l2_info_ptr = nullptr;
        FE_MAKE_SHARED(l2_info_ptr = std::make_shared<fe::ToOpStruct_t>(), return FAILED);
        FE_CHECK_NOTNULL(l2_info_ptr);
        l2_info_json.at(L2_FUSION_TO_OP_STRUCT).get_to(*l2_info_ptr);
        (void)op_desc_ptr->SetExtAttr(L2_FUSION_EXTEND_CONTENT, l2_info_ptr);
      }
    } catch (...) {
      FE_LOGW("Failed to parse JSON string, please check the input string.");
      return FAILED;
    }
  }
  return SUCCESS;
}

Status GetTaskL2FusionInfoFromJson(ge::OpDescPtr op_desc_ptr) {
  if (op_desc_ptr == nullptr) {
    FE_LOGW("The operators description for the node is null.");
    return FAILED;
  }
  // get l2 fusion task info
  string str_task_l2_fusion_info;
  (void)ge::AttrUtils::GetStr(op_desc_ptr, TASK_L2_FUSION_INFO, str_task_l2_fusion_info);
  if (str_task_l2_fusion_info.empty()) {
    FE_LOGW("Task l2 fusion info is empty.");
    return FAILED;
  } else {
    try {
      nlohmann::json task_l2_fusion_info_json = nlohmann::json::parse(str_task_l2_fusion_info);
      if (task_l2_fusion_info_json.is_null()) {
        FE_LOGW("Json file: %s is empty.", TASK_L2_FUSION_INFO.c_str());
        return FAILED;
      } else {
        fe::L2FusionInfoPtr task_l2_fusion_info_ptr = nullptr;
        FE_MAKE_SHARED(task_l2_fusion_info_ptr = std::make_shared<fe::TaskL2FusionInfo_t>(), return FAILED);
        FE_CHECK_NOTNULL(task_l2_fusion_info_ptr);
        task_l2_fusion_info_json.at(TASK_L2_FUSION_INFO).get_to(*task_l2_fusion_info_ptr);
        (void)op_desc_ptr->SetExtAttr(TASK_L2_FUSION_INFO_EXTEND_CONTENT, task_l2_fusion_info_ptr);
      }
    } catch (...) {
      FE_LOGW("Failed to parse JSON string, please check the input string.");
      return FAILED;
    }
  }
  return SUCCESS;
}

Status ReadGraphInfoFromJson(ge::ComputeGraph& graph) {
  for (auto& node : graph.GetAllNodes()) {
    if (node == nullptr) {
      FE_LOGW("Input node is null.");
      continue;
    }
    ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
    if (op_desc_ptr == nullptr) {
      FE_LOGW("The operators description for the node is null.");
      continue;
    }

    if (GetL1InfoFromJson(op_desc_ptr) != SUCCESS) {
      FE_LOGW("Failed to get l1 info");
    } else {
      FE_LOGD("Get l1 info of op[%s] successfully.", node->GetName().c_str());
    }

    if (GetL2InfoFromJson(op_desc_ptr) != SUCCESS) {
      FE_LOGW("Failed to get l2 info.");
    } else {
      FE_LOGD("Successfully obtained l2 info for op[%s].", node->GetName().c_str());
    }

    if (GetTaskL2FusionInfoFromJson(op_desc_ptr) != SUCCESS) {
      FE_LOGW("Failed to get l2 fusion task info.");
    } else {
      FE_LOGD("Get l2 fusion task info of op[%s] successfully.", node->GetName().c_str());
    }
  }
  return SUCCESS;
}

Status WriteGraphInfoToJson(ge::ComputeGraph& graph) {
  for (auto& node : graph.GetAllNodes()) {
    if (node == nullptr) {
      FE_LOGW("Input node is null.");
      continue;
    }
    ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
    if (op_desc_ptr == nullptr) {
      FE_LOGW("The operators description for the node is null.");
      continue;
    }

    // set l1 info
    fe::ToOpStructPtr l1_info_ptr;
    l1_info_ptr = op_desc_ptr->TryGetExtAttr(L1_FUSION_EXTEND_CONTENT, l1_info_ptr);
    if (l1_info_ptr != nullptr) {
      nlohmann::json l1_info_json = nlohmann::json{{L1_FUSION_TO_OP_STRUCT, *l1_info_ptr}};
      std::string str_l1_info = l1_info_json.dump();
      (void)ge::AttrUtils::SetStr(op_desc_ptr, L1_FUSION_TO_OP_STRUCT, str_l1_info);
      FE_LOGD("Set l1 info of op[%s] successfully.", node->GetName().c_str());
    } else {
      FE_LOGW("L1InfoPtr is null.");
    }

    // set l2 info
    fe::ToOpStructPtr l2_info_ptr;
    l2_info_ptr = op_desc_ptr->TryGetExtAttr(L2_FUSION_EXTEND_CONTENT, l2_info_ptr);
    if (l2_info_ptr != nullptr) {
      nlohmann::json l2_info_json = nlohmann::json{{L2_FUSION_TO_OP_STRUCT, *l2_info_ptr}};
      std::string str_l2_info = l2_info_json.dump();
      (void)ge::AttrUtils::SetStr(op_desc_ptr, L2_FUSION_TO_OP_STRUCT, str_l2_info);
      FE_LOGD("Set l2 info of op[%s] successfully.", node->GetName().c_str());
    } else {
      FE_LOGW("L2InfoPtr is null.");
    }

    // set l2 fusion task info
    fe::L2FusionInfoPtr task_l2_fusion_info_ptr;
    task_l2_fusion_info_ptr = op_desc_ptr->TryGetExtAttr(TASK_L2_FUSION_INFO_EXTEND_CONTENT, task_l2_fusion_info_ptr);
    if (task_l2_fusion_info_ptr != nullptr) {
      nlohmann::json task_l2_fusion_info_json = nlohmann::json{{TASK_L2_FUSION_INFO, *task_l2_fusion_info_ptr}};
      std::string str_task_l2_fusion_info = task_l2_fusion_info_json.dump();
      (void)ge::AttrUtils::SetStr(op_desc_ptr, TASK_L2_FUSION_INFO, str_task_l2_fusion_info);
      FE_LOGD("Set l2 fusion task info of op[%s] successfully.", node->GetName().c_str());
    } else {
      FE_LOGW("L2FusionTaskInfoPtr is NULL.");
    }
  }
  return SUCCESS;
}

Status ReadOpSliceInfoFromJson(ge::ComputeGraph& graph) {
  for (auto &node : graph.GetAllNodes()) {
    if (node == nullptr) {
      FE_LOGW("Input node is null.");
      continue;
    }
    ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
    if (op_desc_ptr == nullptr) {
      FE_LOGW("The operators description for the node is null.");
      continue;
    }
    for (auto &slice_mode : SLICE_INFO_MODE) {
      std::string str_op_slice_info;
      (void) ge::AttrUtils::GetStr(op_desc_ptr, slice_mode, str_op_slice_info);
      if (str_op_slice_info.empty()) {
        FE_LOGD("Slice info of op[%s] is empty.", node->GetName().c_str());
      } else {
        try {
          nlohmann::json op_slice_info_json = nlohmann::json::parse(str_op_slice_info);
          if (op_slice_info_json.is_null()) {
            FE_LOGW("Json file: %s is empty.", slice_mode.c_str());
          } else {
            fe::OpCalcInfoPtr op_calc_info_ptr = nullptr;
            FE_MAKE_SHARED(op_calc_info_ptr = std::make_shared<fe::OpCalcInfo>(), return FAILED);
            FE_CHECK_NOTNULL(op_calc_info_ptr);
            if (!op_calc_info_ptr->Initialize()) {
              FE_LOGW("op_calc_info initialize failed");
              return FAILED;
            }
            op_slice_info_json.at(slice_mode).get_to(*op_calc_info_ptr);
            (void) op_desc_ptr->SetExtAttr(slice_mode, op_calc_info_ptr);
            FE_LOGD("Get slice info of op[%s] successfully.", node->GetName().c_str());
          }
        } catch (...) {
          FE_LOGW("Failed to parse JSON string, please check the input string.");
          return FAILED;
        }
      }
    }
  }
  return SUCCESS;
}

Status WriteOpSliceInfoToJson(ge::ComputeGraph& graph) {
  for (auto &node : graph.GetAllNodes()) {
    if (node == nullptr) {
      FE_LOGW("Input node is null.");
      continue;
    }
    ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
    if (op_desc_ptr == nullptr) {
      FE_LOGW("The operators description for the node is null.");
      continue;
    }
    for (auto &slice_mode : SLICE_INFO_MODE) {
      fe::OpCalcInfoPtr op_calc_info_ptr;
      op_calc_info_ptr = op_desc_ptr->TryGetExtAttr(slice_mode, op_calc_info_ptr);
      if (op_calc_info_ptr != nullptr) {
        nlohmann::json op_calc_info_json = nlohmann::json{{slice_mode, *op_calc_info_ptr}};
        std::string str_op_slice_info = op_calc_info_json.dump();
        (void) ge::AttrUtils::SetStr(op_desc_ptr, slice_mode, str_op_slice_info);
        FE_LOGD("Set slice info of op[%s] successfully.", node->GetName().c_str());
        // save file
        FE_LOGD("Operation calculation information is: %s.", str_op_slice_info.c_str());
      } else {
        FE_LOGW("OpCalcInfoPtr is null.");
      }
    }
  }
  return SUCCESS;
}

void SetOpSliceInfoToJson(fe::OpCalcInfo &op_calc_info,
                          std::string &op_calc_info_str) {
  nlohmann::json l1_info_json = nlohmann::json{{fe::OP_SLICE_INFO, op_calc_info}};
  op_calc_info_str = l1_info_json.dump();
  FE_LOGD("Setting op_slice_info to %s.", op_calc_info_str.c_str());
}

void SetFusionOpSliceInfoToJson(fe::OpCalcInfo &op_calc_info,
                                std::string &op_calc_info_str) {
  nlohmann::json l1_info_json = nlohmann::json{{fe::FUSION_OP_SLICE_INFO, op_calc_info}};
  op_calc_info_str = l1_info_json.dump();
  FE_LOGI("Setting op_slice_info to %s.", op_calc_info_str.c_str());
}

void GetOpSliceInfoFromJson(fe::OpCalcInfo &op_calc_info,
                            std::string &op_calc_info_str) {
  try {
    nlohmann::json op_calc_info_json = nlohmann::json::parse(op_calc_info_str);
    op_calc_info_json.at(fe::OP_SLICE_INFO).get_to(op_calc_info);
    FE_LOGI("Get op_slice_info is %s", op_calc_info_str.c_str());
  } catch (...) {
    FE_LOGW("Failed to parse JSON string, please check the input string.");
    return;
  }
}

void GetFusionOpSliceInfoFromJson(fe::OpCalcInfo &op_calc_info,
                                  std::string &op_calc_info_str) {
  try {
    nlohmann::json op_calc_info_json = nlohmann::json::parse(op_calc_info_str);
    op_calc_info_json.at(fe::FUSION_OP_SLICE_INFO).get_to(op_calc_info);
    FE_LOGI("Get op_slice_info is %s", op_calc_info_str.c_str());
  } catch (...) {
    FE_LOGW("Failed to parse JSON string, please check the input string.");
    return;
  }
}

void GetL2ToOpStructFromJson(ge::OpDescPtr& op_desc_ptr, ToOpStructPtr& l2_info_ptr) {
  // set l2 info
  fe::ToOpStructPtr extra_l2_info_ptr = nullptr;
  std::string str_l2_info;
  extra_l2_info_ptr = op_desc_ptr->TryGetExtAttr(fe::ATTR_NAME_L2_FUSION_EXTEND_PTR, extra_l2_info_ptr);
  if (extra_l2_info_ptr == nullptr) {
    (void)ge::AttrUtils::GetStr(op_desc_ptr, L2_FUSION_TO_OP_STRUCT, str_l2_info);
    FE_LOGD("Successfully retrieved L2 info [%s] for operation [%s].", str_l2_info.c_str(), op_desc_ptr->GetName().c_str());
    if (str_l2_info.empty()) {
      FE_LOGD("L2 info is empty.");
      l2_info_ptr = extra_l2_info_ptr;
    } else {
      try {
        nlohmann::json l2_info_json = nlohmann::json::parse(str_l2_info);
        if (l2_info_json.is_null()) {
          FE_LOGW("Get l2 info: %s is empty.", str_l2_info.c_str());
          l2_info_ptr = extra_l2_info_ptr;
        } else {
          l2_info_json.at(L2_FUSION_TO_OP_STRUCT).get_to(*l2_info_ptr);
          (void)op_desc_ptr->SetExtAttr(fe::ATTR_NAME_L2_FUSION_EXTEND_PTR, l2_info_ptr);
        }
      } catch (...) {
      FE_LOGW("Failed to parse JSON string, please check the input string.");
      return;
      }
    }
  } else {
    l2_info_ptr = extra_l2_info_ptr;
    FE_LOGD("L2InfoPtr is not null. Directly read it from json.");
  }
}

void GetL1ToOpStructFromJson(ge::OpDescPtr& op_desc_ptr, ToOpStructPtr& l1_info_ptr) {
  // set l2 info
  std::string str_l1_info;
  fe::ToOpStructPtr extra_l1_info_ptr = nullptr;
  extra_l1_info_ptr = op_desc_ptr->TryGetExtAttr(ge::ATTR_NAME_L1_FUSION_EXTEND_PTR, extra_l1_info_ptr);
  if (extra_l1_info_ptr == nullptr) {
    (void)ge::AttrUtils::GetStr(op_desc_ptr, L1_FUSION_TO_OP_STRUCT, str_l1_info);
    FE_LOGD("Get l1 info %s of op[%s] successfully.", str_l1_info.c_str(), op_desc_ptr->GetName().c_str());
    if (str_l1_info.empty()) {
      FE_LOGD("L1 info is empty.");
      l1_info_ptr = extra_l1_info_ptr;
    } else {
      try {
        nlohmann::json l1_info_json = nlohmann::json::parse(str_l1_info);
        if (l1_info_json.is_null()) {
          FE_LOGW("Get l1 info: %s is empty.", str_l1_info.c_str());
          l1_info_ptr = extra_l1_info_ptr;
        } else {
          l1_info_json.at(L1_FUSION_TO_OP_STRUCT).get_to(*l1_info_ptr);
          (void)op_desc_ptr->SetExtAttr(ge::ATTR_NAME_L1_FUSION_EXTEND_PTR, l1_info_ptr);
        }
      } catch (...) {
      FE_LOGW("Failed to parse JSON string, please check the input string.");
      return;
      }
    }
  } else {
    l1_info_ptr = extra_l1_info_ptr;
    FE_LOGD("L1InfoPtr is not null. Reading it directly from JSON.");
  }
}

void GetStridedToOpStructFromJson(ge::OpDescPtr& op_desc_ptr, ToOpStructPtr& strided_info_ptr,
                                  const string& pointer_key, const string& json_key) {
  // set strided info
  std::string str_strided_info;
  fe::ToOpStructPtr extra_strided_info_ptr = nullptr;
  extra_strided_info_ptr = op_desc_ptr->TryGetExtAttr(pointer_key, extra_strided_info_ptr);
  if (extra_strided_info_ptr == nullptr) {
    (void)ge::AttrUtils::GetStr(op_desc_ptr, json_key, str_strided_info);
    FE_LOGD("Get strided info %s of op[%s] successfully.", str_strided_info.c_str(), op_desc_ptr->GetName().c_str());
    if (str_strided_info.empty()) {
      FE_LOGD("L1 info is empty.");
      strided_info_ptr = extra_strided_info_ptr;
    } else {
      try {
        nlohmann::json strided_info_json = nlohmann::json::parse(str_strided_info);
        if (strided_info_json.is_null()) {
          FE_LOGW("Get strided info: %s is empty.", str_strided_info.c_str());
          strided_info_ptr = extra_strided_info_ptr;
        } else {
          strided_info_json.at(json_key).get_to(*strided_info_ptr);
          (void)op_desc_ptr->SetExtAttr(pointer_key, strided_info_ptr);
        }
      } catch (...) {
      FE_LOGW("Failed to parse JSON string, please check the input string.");
      return;
      }
    }
  } else {
    strided_info_ptr = extra_strided_info_ptr;
    FE_LOGD("StridedInfoPtr is not null. Directly read it from json.");
  }
}

L2FusionInfoPtr GetL2FusionInfoFromJson(ge::OpDescPtr& op_desc_ptr) {
  // set l2 info
  std::string str_l2_info;
  L2FusionInfoPtr l2_fusion_info_ptr = nullptr;
  l2_fusion_info_ptr = op_desc_ptr->TryGetExtAttr(fe::ATTR_NAME_TASK_L2_FUSION_INFO_EXTEND_PTR, l2_fusion_info_ptr);
  if (l2_fusion_info_ptr == nullptr) {
    (void)ge::AttrUtils::GetStr(op_desc_ptr, TASK_L2_FUSION_INFO, str_l2_info);
    FE_LOGD("Successfully retrieved L2 fusion task info %s for operation [%s].", str_l2_info.c_str(), op_desc_ptr->GetName().c_str());
    if (str_l2_info.empty()) {
      FE_LOGD("L2 fusion task info is empty.");
    } else {
      try {
        nlohmann::json l2_info_json = nlohmann::json::parse(str_l2_info);
        if (l2_info_json.is_null()) {
          FE_LOGW("Get L2 fusion task info: %s is empty.", str_l2_info.c_str());
        } else {
          FE_MAKE_SHARED(l2_fusion_info_ptr = std::make_shared<fe::TaskL2FusionInfo_t>(), return nullptr);
          FE_CHECK(l2_fusion_info_ptr == nullptr, FE_LOGW("l2_fusion_info_ptr must not be null."), return nullptr);
          l2_info_json.at(TASK_L2_FUSION_INFO).get_to(*l2_fusion_info_ptr);
          (void)op_desc_ptr->SetExtAttr(fe::ATTR_NAME_TASK_L2_FUSION_INFO_EXTEND_PTR, l2_fusion_info_ptr);
        }
      } catch (...) {
        FE_LOGW("Failed to parse JSON string, please check the input string.");
        return nullptr;
      }
    }
  } else {
    FE_LOGD("L2FusionTaskInfoPtr is not null. Directly read it from json.");
    nlohmann::json task_l2_fusion_info_json = nlohmann::json{{TASK_L2_FUSION_INFO, *l2_fusion_info_ptr}};
    std::string str_l2_info_tmp = task_l2_fusion_info_json.dump();
    FE_LOGD("L2 fusion task info of node[%s] is [%s].", op_desc_ptr->GetName().c_str(), str_l2_info_tmp.c_str());
  }
  return l2_fusion_info_ptr;
}

void SetL2FusionInfoToNode(ge::OpDescPtr& op_desc_ptr, L2FusionInfoPtr& l2_fusion_info_ptr) {
  // set l2 info
  std::string str_l2_info;
  (void)op_desc_ptr->SetExtAttr(fe::ATTR_NAME_TASK_L2_FUSION_INFO_EXTEND_PTR, l2_fusion_info_ptr);
  if (l2_fusion_info_ptr != nullptr) {
    nlohmann::json task_l2_fusion_info_json = nlohmann::json{{TASK_L2_FUSION_INFO, *l2_fusion_info_ptr}};
    str_l2_info = task_l2_fusion_info_json.dump();
    (void)ge::AttrUtils::SetStr(op_desc_ptr, TASK_L2_FUSION_INFO, str_l2_info);
    FE_LOGD("Successfully set L2 fusion task info %s for operation [%s].", str_l2_info.c_str(), op_desc_ptr->GetName().c_str());
  } else {
    FE_LOGW("L2FusionTaskInfoPtr is NULL.");
  }
}

void SetStridedInfoToNode(ge::OpDescPtr& op_desc_ptr, ToOpStructPtr& strided_info_ptr,
                          const string& pointer_key, const string& json_key) {
  // set stride info
  std::string str_strided_info;
  (void)op_desc_ptr->SetExtAttr(pointer_key, strided_info_ptr);
  if (strided_info_ptr != nullptr) {
    nlohmann::json strided_info_json = nlohmann::json{{json_key, *strided_info_ptr}};
    str_strided_info = strided_info_json.dump();
    (void)ge::AttrUtils::SetStr(op_desc_ptr, json_key, str_strided_info);
    FE_LOGD("Set strided info %s of op[%s] successfully.", str_strided_info.c_str(), op_desc_ptr->GetName().c_str());
  } else {
    FE_LOGW("strided_info_ptr is nullptr.");
  }
}
}
