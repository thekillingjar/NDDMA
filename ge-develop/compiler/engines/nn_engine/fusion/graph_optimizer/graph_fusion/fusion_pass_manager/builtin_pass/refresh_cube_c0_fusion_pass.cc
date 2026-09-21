/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "refresh_cube_c0_fusion_pass.h"
#include "common/configuration.h"
#include "common/fe_log.h"
#include "common/util/op_info_util.h"
#include "common/op_info_common.h"
#include "common/format/axis_name_util.h"
#include "common/fe_inner_attr_define.h"
#include "common/unknown_shape_util.h"
#include "expand_dimension.h"
#include "common/platform_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "ops_store/ops_kernel_manager.h"
#include "ops_store/ops_kernel_common.h"

namespace {
const std::string kPatternCube = "cube";
const int64_t kFp32C0Val = 8;
}

namespace fe {
ge::GeShape GetNewShapeForPass(const ge::OpDescPtr& op_desc_ptr, const ge::GeShape& old_shape,
                              const ge::Format& old_format, const ge::Format& new_format,
                              const int64_t& op_imply_type, const ge::DataType& current_data_type,
                              const int64_t& group) {
  uint32_t old_dim_vec_size = static_cast<uint32_t>(old_shape.GetDimNum());
  ge::Format op_primary_format = static_cast<ge::Format>(ge::GetPrimaryFormat(new_format));
  /* other format to Nz or ND to (FRACTAL_Z / FRACTAL_ZN_RNN / ND_RNN_BIAS),
   * the size of original shape does not need to be 4 */
  if (old_dim_vec_size < NCHW_DIMENSION_NUM && op_primary_format != ge::FORMAT_FRACTAL_NZ &&
      ((op_primary_format != ge::FORMAT_FRACTAL_Z && op_primary_format != ge::FORMAT_FRACTAL_ZN_RNN &&
        op_primary_format != ge::FORMAT_ND_RNN_BIAS) || old_format != ge::FORMAT_ND)) {
    FE_LOGW("Get shape not successfully: old format is %s, new format is %s, old dimension is %u.",
            ge::TypeUtils::FormatToSerialString(old_format).c_str(),
            ge::TypeUtils::FormatToSerialString(op_primary_format).c_str(), old_dim_vec_size);
    return old_shape;
  }
  /* Assemble shape and format transmission information */
  ge::GeShape new_shape;
  int64_t hidden_size = 1;
  int64_t input_size = 1;
  int64_t state_size = -1;
  (void)ge::AttrUtils::GetInt(op_desc_ptr, "hidden_size", hidden_size);
  (void)ge::AttrUtils::GetInt(op_desc_ptr, "input_size", input_size);
  (void)ge::AttrUtils::GetInt(op_desc_ptr, "state_size", state_size);
  CalcShapeExtraAttr extra_attr = {hidden_size, input_size, state_size};
  ShapeAndFormat shape_and_format_info = {old_shape, new_shape, old_format, new_format,
                                          current_data_type, group, extra_attr};
  Status ret = GetShapeAccordingToFormat(shape_and_format_info);
  if (ret != SUCCESS) {
    FE_LOGW("Old format is %s, new format is %s, old dimension is %u and opImplyType is %ld.",
            ge::TypeUtils::FormatToSerialString(old_format).c_str(),
            ge::TypeUtils::FormatToSerialString(op_primary_format).c_str(), old_dim_vec_size, op_imply_type);
    return old_shape;
  }
  return new_shape;
}

void CountNdInputForPass(const UpdateInfo &update_info, ge::Format &ori_format,
                  ge::NodePtr &node, size_t &count_input_is_nd) {
  for (size_t i = 0; i < node->GetAllInDataAnchors().size(); i++) {
    auto input_desc_ptr = node->GetOpDesc()->GetInputDescPtr(static_cast<uint32_t>(i));
    if (input_desc_ptr == nullptr) {
      continue;
    }
    ge::Format node_input_ori_format = input_desc_ptr->GetOriginFormat();
    if (node_input_ori_format != ge::FORMAT_ND) {
      ori_format = node_input_ori_format;
      update_info.op_input_or_output_desc.SetOriginFormat(ori_format);
      (void)ge::AttrUtils::SetInt(update_info.node_ptr->GetOpDesc(), INPUT_ND_TO_OTHER_FORMAT, ori_format);
      break;
    }
    count_input_is_nd++;
  }
  return;
}

Status PadNDToOtherFormatForPass(const UpdateInfo &update_info, const ge::Format &op_kernel_format,
                                 ge::Format &ori_format) {
  FE_LOGD("Before padding ND to another format, node name: %s, op_kernel_format: %d, ori_format: %d.",
          update_info.node_ptr->GetName().c_str(), op_kernel_format, ori_format);
  if (op_kernel_format == ge::FORMAT_NC1HWC0 && update_info.is_input &&
      update_info.op_input_or_output_desc.GetOriginFormat() == ge::FORMAT_ND) {
    ge::NodePtr node = update_info.node_ptr;
    int64_t input_nd_other_format = -1;
    if (ge::AttrUtils::GetInt(node->GetOpDesc(), INPUT_ND_TO_OTHER_FORMAT, input_nd_other_format) &&
        input_nd_other_format != -1) {
      ge::Format other_format = static_cast<ge::Format>(input_nd_other_format);
      ori_format = other_format;
      update_info.op_input_or_output_desc.SetOriginFormat(ori_format);
    } else {
      size_t count_input_is_nd = 0;
      CountNdInputForPass(update_info, ori_format, node, count_input_is_nd);
      if (count_input_is_nd != 1 && count_input_is_nd == node->GetAllInDataAnchors().size()) {
        REPORT_FE_ERROR("[GraphOptJdgInst][UpdFmtAndDtype][PadNDToOtherFmt] Node[%s]: "
                        "original format of inputs is ND.", node->GetName().c_str());
        return FAILED;
      }
    }
  }
  FE_LOGD("After pad ND to other format, node name: %s, op_kernel_format: %d, ori_format: %d.",
          update_info.node_ptr->GetName().c_str(), op_kernel_format, ori_format);
  return SUCCESS;
}

int64_t GetReshapeTypeMaskForPass(const UpdateInfo &update_info, const ge::Format &ori_format,
                                  const ge::Format &primary_format) {
  std::string reshape_type = update_info.input_or_output_info_ptr->GetReshapeType();
  if (reshape_type.empty() && update_info.op_kernel_info_ptr != nullptr &&
      update_info.op_kernel_info_ptr->GetOpPattern() == OP_PATTERN_REDUCE) {
    vector<int64_t> axis_values;
    (void)ge::AttrUtils::GetListInt(update_info.node_ptr->GetOpDesc(), AXES_ATTR_NAME, axis_values);
    reshape_type = AxisNameUtil::GetReshapeType(ori_format, axis_values);
  }

  (void)ge::AttrUtils::SetStr(update_info.op_input_or_output_desc, ge::ATTR_NAME_RESHAPE_INFER_TYPE, reshape_type);
  int64_t reshape_type_mask = transformer::ExpandDimension::GenerateReshapeType(
      ori_format, primary_format, update_info.op_input_or_output_desc.GetShape().GetDimNum(), reshape_type);
  (void)ge::AttrUtils::SetInt(update_info.op_input_or_output_desc, ge::ATTR_NAME_RESHAPE_TYPE_MASK,
                              reshape_type_mask);
  return reshape_type_mask;
}

Status UpdateNewShapeAndFormatForPass(const UpdateInfo& update_info, ge::Format op_kernel_format,
                                      const int64_t& group, const ge::GeShape& original_shape,
                                      const ge::GeShape& new_shape, const std::string& op_type,
                                      const std::string& op_name) {
  auto old_format = update_info.op_input_or_output_desc.GetFormat();
  auto new_format = static_cast<ge::Format>(ge::GetPrimaryFormat(op_kernel_format));
  vector<uint32_t> support_sub_format;
  if (update_info.input_or_output_info_ptr != nullptr) {
    support_sub_format = update_info.input_or_output_info_ptr->GetSubformat();
  }
  if (std::find(FE_GROUP_RELA_FORMAT_VECTOR.begin(), FE_GROUP_RELA_FORMAT_VECTOR.end(), new_format) !=
      FE_GROUP_RELA_FORMAT_VECTOR.end() &&
      group > GROUPS_DEFAULT_VALUE) {
    FE_LOGD("Op[name=%s, type=%s]: the %s [%u], the group is more than 1, set sub_format to be %ld.", op_name.c_str(),
            op_type.c_str(), IS_INPUT_TO_STRING(update_info.is_input), update_info.index, group);
    if (group > UINT16_MAX) {
      REPORT_FE_ERROR("The group value [%ld] exceeds the maximum supported value [%d].", group, UINT16_MAX);
      return FAILED;
    }
    if (support_sub_format.size() > 0) {
      bool is_need_check_subformat = std::find(support_sub_format.begin(), support_sub_format.end(),
          SUPPORT_ALL_SUB_FORMAT) == support_sub_format.end();
      if (is_need_check_subformat && std::find(support_sub_format.begin(), support_sub_format.end(),
          static_cast<uint32_t>(group)) == support_sub_format.end()) {
        FE_LOGE("Op[name=%s,type=%s]: the %s [%u] group value [%ld] not supported by op_kernel lib.",
                op_name.c_str(), op_type.c_str(), IS_INPUT_TO_STRING(update_info.is_input), update_info.index, group);
        return FAILED;
      }
    } else {
      FE_LOGD("Op[name=%s,type=%s]: support sub_format is empty.", op_name.c_str(), op_type.c_str());
    }
    new_format = static_cast<ge::Format>(ge::GetFormatFromSub(op_kernel_format, group));
  }
  if (ge::HasC0Format(op_kernel_format)) {
    int32_t c0_bit = GetC0BitValFromC0(ge::GetC0Value(op_kernel_format));
    FE_LOGD("Op[name=%s,type=%s]: the %s [%u] format has c0_bit is %d.", op_name.c_str(),
            op_type.c_str(), IS_INPUT_TO_STRING(update_info.is_input), update_info.index, c0_bit);
    new_format = static_cast<ge::Format>(ge::GetFormatFromC0(new_format, c0_bit));
  }

  FE_LOGD(
      "Op[name=%s,type=%s]: update the %s [%u], new format is [%s], group is [%ld], old format is [%s], origin format "
      "is [%s].",
      op_name.c_str(), op_type.c_str(), IS_INPUT_TO_STRING(update_info.is_input), update_info.index,
      FormatToStr(static_cast<ge::Format>(ge::GetPrimaryFormat(new_format))).c_str(), group,
      FormatToStr(old_format).c_str(), FormatToStr(update_info.op_input_or_output_desc.GetOriginFormat()).c_str());

  update_info.op_input_or_output_desc.SetFormat(new_format);

  FE_LOGD("Op[name=%s,type=%s]: update the %s [%u], new shape is [%s], origin shape is [%s].",
          op_name.c_str(), op_type.c_str(), IS_INPUT_TO_STRING(update_info.is_input), update_info.index,
          GetShapeDims(new_shape).c_str(),
          GetShapeDims(original_shape).c_str());
  update_info.op_input_or_output_desc.SetShape(new_shape);
  return SUCCESS;
}

Status UpdateNewShapeForPass(const UpdateInfo& update_info, ge::Format op_kernel_format,
                            ge::DataType op_kernel_dtype, int64_t group, int64_t op_imply_type_input) {
  auto &original_shape = update_info.op_input_or_output_desc.GetShape();
  auto new_shape = update_info.op_input_or_output_desc.GetShape();
  auto ori_format = GetCurOpOriginFormat(update_info.op_input_or_output_desc);
  ge::Format op_primary_format = static_cast<ge::Format>(ge::GetPrimaryFormat(op_kernel_format));

  if (IsScalarInputOrOutput(original_shape, ori_format)) {
    FE_LOGD("%s %u of %s is scalar, we do not change its format and shape.",
            IS_INPUT_TO_STRING(update_info.is_input), update_info.index, update_info.node_ptr->GetName().c_str());
    return SUCCESS;
  }
  const string &op_name = update_info.node_ptr->GetName();
  const string &op_type = update_info.node_ptr->GetType();
  // 3.3 if the op_kernel_format is ND or ALL, use the ori_format
  if (IsNd(op_primary_format)) {
    FE_LOGD("The op_kernel_format is ND, get the origin format");
    op_kernel_format = ori_format;
  } else {
    // 3.4 pad the origin shape and get the new shape
    FE_LOGD("The %s op_kernel_format is [%s] and tensor original format is [%s].",
            IS_INPUT_TO_STRING(update_info.is_input), FormatToStr(op_primary_format).c_str(),
            FormatToStr(ori_format).c_str());
    if (!original_shape.IsUnknownDimNum() && op_primary_format != ori_format) {
      FE_LOGD("Format of op_kernel is not equal with origin format of input");
      // deal with Format_NC1HWC0, pad ND to other not ND format
      if (PadNDToOtherFormatForPass(update_info, op_primary_format, ori_format) != SUCCESS) {
        FE_LOGD("Pad ND format to other format unsuccessful.");
        return FAILED;
      }

      // get reshape type mask and expand dims
      int64_t reshape_type_mask = GetReshapeTypeMaskForPass(update_info, ori_format, op_primary_format);
      ge::GeShape origin_shape_afer_pad;
      if (reshape_type_mask == 0) {
        origin_shape_afer_pad = original_shape;
      } else {
        transformer::ExpandDimension::ExpandDims(reshape_type_mask, original_shape, origin_shape_afer_pad);
      }

      new_shape = GetNewShapeForPass(update_info.node_ptr->GetOpDesc(), origin_shape_afer_pad, ori_format,
                                    op_kernel_format, op_imply_type_input, op_kernel_dtype, group);

      /* update shape range for unknown op */
      if (CalcShapeRange(update_info.node_ptr->GetOpDesc(), op_kernel_format, origin_shape_afer_pad,
                         update_info.op_input_or_output_desc) != SUCCESS) {
        FE_LOGI("Set shape range of op[name:%s,type:%s] not successfully.",
                update_info.node_ptr->GetOpDesc()->GetName().c_str(),
                update_info.node_ptr->GetOpDesc()->GetType().c_str());
        return FAILED;
      }
      GetReshapeAxisValueByName(ori_format, origin_shape_afer_pad,
                                'C', update_info.op_input_or_output_desc);
    }
  }

  // 4. update the shape and format
  auto ret = UpdateNewShapeAndFormatForPass(update_info, op_kernel_format, group, original_shape,
                                           new_shape, op_type, op_name);
  FE_CHECK(ret != SUCCESS, FE_LOGE("Failed to update new shape and format"), return FAILED);
  return SUCCESS;
}

Status CalcNewShapeAndUpdateForPass(const UpdateInfo& update_info, ge::Format op_kernel_format,
                                    ge::DataType op_kernel_dtype) {
  /* 3.1 Get the origianl shape and original format */
  ge::Format op_input_or_output_desc_origin_format = GetCurOpOriginFormat(update_info.op_input_or_output_desc);

  const string &op_name = update_info.node_ptr->GetName();
  const string &op_type = update_info.node_ptr->GetType();
  ge::Format op_kernel_primary_format = static_cast<ge::Format>(ge::GetPrimaryFormat(op_kernel_format));
  /* 3.2 Get the op imply type */
  int64_t op_imply_type_input = -1;
  FE_LOGW_IF(!ge::AttrUtils::GetInt(update_info.node_ptr->GetOpDesc(), FE_IMPLY_TYPE, op_imply_type_input),
             "Op[name=%s,type=%s]: get imply type not success.", op_name.c_str(), op_type.c_str());
  int64_t imply_type = GetMainImplType<int64_t>(op_imply_type_input);
  FE_LOGD("Node[%s, %s]: get new format and shape for %s %u. The op kernel format is %s. Origin format is %s.",
          op_name.c_str(), op_type.c_str(), IS_INPUT_TO_STRING(update_info.is_input), update_info.index,
          FormatToStr(op_kernel_primary_format).c_str(), FormatToStr(op_input_or_output_desc_origin_format).c_str());

  // if the input or output tensor has chose format fraz_g,
  // then verify whether the opdesc has attr groups or _fe_group
  // and the group value should be greater than zero.
  int64_t group = GROUPS_DEFAULT_VALUE;
  if (std::find(FE_GROUP_RELA_FORMAT_VECTOR.begin(), FE_GROUP_RELA_FORMAT_VECTOR.end(), op_kernel_primary_format) !=
      FE_GROUP_RELA_FORMAT_VECTOR.end()) {
    if (GetGroupAttributeWithVerify(update_info.node_ptr->GetOpDesc(), group) != SUCCESS) {
      REPORT_FE_ERROR("[GraphOptJdgInst][UpdFmtAndDtype][CalcNewShpAndUpd] Node[%s, %s]: The groups value is invalid.",
                      update_info.node_ptr->GetOpDesc()->GetName().c_str(),
                      update_info.node_ptr->GetOpDesc()->GetType().c_str());
      return FAILED;
    }
  }

  Status ret = UpdateNewShapeForPass(update_info, op_kernel_format, op_kernel_dtype, group, imply_type);
  if (ret != SUCCESS) {
    return ret;
  }
  return SUCCESS;
}

std::vector<FusionPattern *> RefreshCubeC0FusionPass::DefinePatterns() {
  std::vector<FusionPattern *> patterns;
  FusionPattern *pattern = new (std::nothrow) FusionPattern("refresh_cube_c0_pattern");
  FE_CHECK(pattern == nullptr, REPORT_FE_ERROR("[GraphOpt][pattern][DefPtn] new an object failed."), return patterns);
  std::vector<std::string> kCubeOpList = {"AvgPoolUpdate", "AvgPool3DD", "AvgPool3DGradD", "Pooling",
                                          "AvgPoolGradD", "AvgPoolGrad", "AvgPoolV2GradD"};
  const auto &cube_op_list = Configuration::Instance(AI_CORE_NAME).GetFp16OpTypeList();
  kCubeOpList.insert(kCubeOpList.end(), cube_op_list.begin(), cube_op_list.end());
  pattern->AddOpDesc(kPatternCube, kCubeOpList)
          .SetOutput(kPatternCube);
  patterns.push_back(pattern);
  return patterns;
}

ge::GeTensorDescPtr RefreshCubeC0FusionPass::GetCubeInputDesc(const ge::OpDescPtr &op_desc) {
  if (op_desc == nullptr) {
    return nullptr;
  }
  auto input_x = op_desc->MutableInputDesc("x");
  if (input_x == nullptr) {
    input_x = op_desc->MutableInputDesc("x1");
  }
  if (input_x == nullptr) {
    input_x = op_desc->MutableInputDesc("input");
  }
  if (input_x == nullptr) {
    input_x = op_desc->MutableInputDesc("input_grad");
  }
  if (input_x == nullptr) {
    input_x = op_desc->MutableInputDesc("grads");
  }
  if (input_x == nullptr) {
    input_x = op_desc->MutableInputDesc("filter");
  }
  return input_x;
}

Status RefreshCubeC0FusionPass::UpdateTensorDesc(ge::NodePtr &node, ge::GeTensorDescPtr &tensor_desc,
                                                 const ge::DataType &data_type, const uint32_t &index,
                                                 const bool &is_input) const {
  FE_CHECK_NOTNULL(tensor_desc);
  OpKernelInfoPtr op_kernel_info_ptr =
          OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpDesc(node->GetOpDesc());
  FE_CHECK_NOTNULL(op_kernel_info_ptr);
  IndexNameMap index_map;
  if (is_input) {
    (void)GetInputIndexNameMap(*(node->GetOpDesc().get()), *op_kernel_info_ptr, index_map);
  } else {
    (void)GetOutputIndexNameMap(*(node->GetOpDesc().get()), *op_kernel_info_ptr, index_map);
  }
  auto tensor_iter = index_map.find(index);
  if (tensor_iter == index_map.end() || tensor_iter->second.empty()) {
    FE_LOGE("Node[name=%s, type=%s, %s %zu]: Failed to find tensor.", node->GetName().c_str(), node->GetType().c_str(),
            IS_INPUT_TO_STRING(is_input), index);
    return FAILED;
  }
  InputOrOutputInfoPtr tensor_info_ptr = nullptr;
  if (op_kernel_info_ptr->GetTensorInfoByName(is_input, tensor_iter->second, tensor_info_ptr) != SUCCESS) {
    FE_LOGE("Node[name=%s, type=%s, %s %zu]: Failed to get tensor information.",
            node->GetName().c_str(), node->GetType().c_str(), IS_INPUT_TO_STRING(is_input), index);
    return FAILED;
  }
  FE_CHECK_NOTNULL(tensor_info_ptr);

  ge::Format op_kernel_format = static_cast<ge::Format>(ge::GetPrimaryFormat(tensor_desc->GetFormat()));
  tensor_desc->SetFormat(tensor_desc->GetOriginFormat());
  tensor_desc->SetShape(tensor_desc->GetOriginShape());
  std::vector<std::pair<int64_t, int64_t>> ori_shape_range;
  tensor_desc->GetOriginShapeRange(ori_shape_range);
  tensor_desc->SetShapeRange(ori_shape_range);
  UpdateInfo update_info = {op_kernel_info_ptr, tensor_info_ptr, 0, node, index, *tensor_desc, is_input};
  if (FE_ORIGIN_FORMAT_SET.count(op_kernel_format) == 0) {
    op_kernel_format = static_cast<ge::Format>(ge::GetFormatFromC0(op_kernel_format, GetC0BitValFromC0(kFp32C0Val)));
  }
  /* update the tensor current format & shape according to the selected format and dtype */
  Status ret = CalcNewShapeAndUpdateForPass(update_info, op_kernel_format, data_type);
  if (ret != SUCCESS) {
    FE_LOGE("Node[name=%s, type=%s, %s %zu]: Failed to update format and shape.",
            node->GetName().c_str(), node->GetType().c_str(), IS_INPUT_TO_STRING(is_input), index);
    return FAILED;
  }
  return SUCCESS;
}

Status RefreshCubeC0FusionPass::Fusion(ge::ComputeGraph &graph, Mapping &mapping,
                                       std::vector<ge::NodePtr> &new_nodes) {
  (void)graph;
  (void)new_nodes;
  if (!PlatformUtils::Instance().IsEnableCubeHighPrecision()) {
    return NOT_CHANGED;
  }
  ge::NodePtr cube_node = GetNodeFromMapping(kPatternCube, mapping);
  FE_CHECK_NOTNULL(cube_node);
  FE_LOGD("Node[%s]: begin to refresh c0_format.", cube_node->GetName().c_str());
  ge::OpDescPtr op_desc = cube_node->GetOpDesc();
  // check aicore op
  int64_t fe_imply_type = -1;
  if (!ge::AttrUtils::GetInt(op_desc, FE_IMPLY_TYPE, fe_imply_type)) {
    FE_LOGD("Op[name=%s] is not aicore node.", op_desc->GetName().c_str());
    return NOT_CHANGED;
  }
  auto input_x = GetCubeInputDesc(op_desc);
  if (input_x == nullptr) {
    FE_LOGD("Node[%s]: can't find a suitable input desc.", op_desc->GetName().c_str());
    return NOT_CHANGED;
  }
  auto output_y = op_desc->MutableOutputDesc(0);
  if (output_y == nullptr) {
    FE_LOGD("Node [%s] output desc is nullptr.", op_desc->GetName().c_str());
    return NOT_CHANGED;
  }
  if (input_x->GetDataType() != ge::DT_FLOAT || output_y->GetDataType() != ge::DT_FLOAT) {
    return NOT_CHANGED;
  }
  // input&output all fp32
  auto all_input = op_desc->GetAllInputsDescPtr();
  for (size_t index = 0; index < all_input.size(); ++index) {
    auto input_desc = all_input.at(index);
    if (input_desc == nullptr) {
      continue;
    }
    if (input_desc->GetDataType() != ge::DT_FLOAT) {
      continue;
    }
    if (UpdateTensorDesc(cube_node, input_desc, input_desc->GetDataType(),
                         static_cast<uint32_t>(index), true) != SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][BeforeInsTrans][Fusion] Failed to update TensorDesc for node [%s %s].",
                      op_desc->GetName().c_str(), op_desc->GetType().c_str());
      return FAILED;
    }
  }
  auto all_output = op_desc->GetAllOutputsDescPtr();
  for (size_t index = 0; index < all_output.size(); ++index) {
    auto output_desc = all_output.at(index);
    if (output_desc == nullptr) {
      continue;
    }
    if (output_desc->GetDataType() != ge::DT_FLOAT) {
      continue;
    }
    if (UpdateTensorDesc(cube_node, output_desc, output_desc->GetDataType(),
                         static_cast<uint32_t>(index), false) != SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][BeforeInsTrans][Fusion] Failed to update TensorDesc for node [%s %s].",
                      op_desc->GetName().c_str(), op_desc->GetType().c_str());
      return FAILED;
    }
  }
  return SUCCESS;
}

REG_PASS("ARefreshCubeC0FusionPass", BUILT_IN_BEFORE_TRANSNODE_INSERTION_GRAPH_PASS,
         RefreshCubeC0FusionPass, SINGLE_SCENE_OPEN | FE_PASS | FORBIDDEN_CLOSE);
}  // namespace fe
