/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/model/model_introduction.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/tensor_utils.h"
#include "framework/common/types.h"
#include "base/err_msg.h"
#include "graph_metadef/common/ge_common/util.h"

namespace ge {
static constexpr uint32_t UINT32_SIZE = static_cast<uint32_t>(sizeof(uint32_t));
static constexpr uint32_t TYPE_SIZE = static_cast<uint32_t>(sizeof(ModelDescType));
static constexpr int32_t NOT_DYNAMIC_MODE = -1;
// Unknown shape mem size
static constexpr int64_t kMemSizeUnknownShape = -1;

Status ModelIntroduction::Init(const GeModelPtr &ge_model, const bool is_dynamic) {
  GELOGD("start to init model introduction, is_dynamic is %d", static_cast<int32_t>(is_dynamic));
  GE_CHECK_NOTNULL(ge_model->GetGraph());
  uint32_t data_index = 0U;
  is_dynamic_ = is_dynamic;
  for (const auto &node : ge_model->GetGraph()->GetDirectNode()) {
    static const std::set<std::string> kDataOpTypes{DATA, AIPPDATA, ANN_DATA};
    const auto &op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    if (kDataOpTypes.count(op_desc->GetType()) > 0U) {
      uint32_t tmp_index = data_index++;
      if (AttrUtils::GetInt(op_desc, ATTR_NAME_INDEX, tmp_index)) {
        GELOGD("Get new index %u, old %u", tmp_index, data_index - 1U);
      }
      input_op_list_[tmp_index] = op_desc;
      GELOGD("find input[%s], index[%u]", node->GetName().c_str(), tmp_index);
    }

    if (op_desc->GetType() == NETOUTPUT) {
      output_op_list_.push_back(op_desc);
      GELOGD("find output[%s].", node->GetName().c_str());
    }
    if (op_desc->GetType() == CASE) {
      case_desc_ = op_desc;
      GELOGD("find Case[%s].", node->GetName().c_str());
    }
  }
  (void)AttrUtils::GetListStr(ge_model, ATTR_MODEL_OUT_NODES_NAME, out_node_name_);

  Status ret = ConstructInputInfo();
  GE_CHK_STATUS_RET(ret, "construct input info failed.");
  ret = ConstructOutputInfo();
  GE_CHK_STATUS_RET(ret, "construct output info failed.");
  ret = ConstructDynamicInfo();
  GE_CHK_STATUS_RET(ret, "construct dynamic info failed.");
  ConstructNameOrder();
  ConstructDynamicOutShape();
  return SUCCESS;
}

Status ModelIntroduction::CreateInputDimsInfo(const OpDescPtr &op_desc, ModelTensorDesc &model_tensor_desc) {
  vector<int64_t> shape_info;
  vector<int64_t> dims_info;
  std::vector<std::pair<int64_t, int64_t>> shape_ranges;

  // judge if this data is linked dynamic aipp first, multiply batch has been considered
  if (op_desc->HasAttr(ATTR_DYNAMIC_AIPP_INPUT_DIMS)) {
    (void)AttrUtils::GetListInt(op_desc, ATTR_DYNAMIC_AIPP_INPUT_DIMS, shape_info);
  } else if (!op_desc->HasAttr(ATTR_MBATCH_ORIGIN_INPUT_DIMS)) {
    // judge if this data is multiply batch
    const auto input_desc_ptr = op_desc->GetInputDescPtr(0U);
    GE_CHK_BOOL_RET_SPECIAL_STATUS(input_desc_ptr == nullptr, FAILED);
    shape_info = input_desc_ptr->GetShape().GetDims();
  } else {
    (void)AttrUtils::GetListInt(op_desc, ATTR_MBATCH_ORIGIN_INPUT_DIMS, shape_info);
  }

  if (op_desc->HasAttr(ATTR_NAME_INPUT_DIMS)) {
    // When static aipp is set, need to get the model input dims which processed by aipp
    (void)AttrUtils::GetListInt(op_desc, ATTR_NAME_INPUT_DIMS, dims_info);
  } else {
    const auto input_desc_ptr = op_desc->GetInputDescPtr(0U);
    GE_CHK_BOOL_RET_SPECIAL_STATUS(input_desc_ptr == nullptr, FAILED);
    (void)input_desc_ptr->GetShapeRange(shape_ranges);
    GELOGD("op:%s(%s) shaperanges size=%zu", op_desc->GetName().c_str(), op_desc->GetType().c_str(),
           shape_ranges.size());
    dims_info = shape_info;
  }
  model_tensor_desc.dims = shape_info;
  model_tensor_desc.base_info.dims_len = static_cast<uint32_t>(sizeof(int64_t) * shape_info.size());

  model_tensor_desc.dimsV2 = dims_info;
  model_tensor_desc.base_info.dimsV2_len = static_cast<uint32_t>(sizeof(int64_t) * dims_info.size());

  model_tensor_desc.shape_range = shape_ranges;
  model_tensor_desc.base_info.shape_range_len =
      static_cast<uint32_t>(sizeof(std::pair<int64_t, int64_t>) * shape_ranges.size());
  return SUCCESS;
}

Status ModelIntroduction::ConstructInputInfo() {
  for (const auto &iterator : input_op_list_) {
    const auto op_desc = iterator.second;
    ModelTensorDesc model_tensor_desc;
    GE_CHK_STATUS_RET(CreateInputDimsInfo(op_desc, model_tensor_desc), "[Get][createInputDimsInfo] failed in op:%s",
                      op_desc->GetName().c_str());
    int64_t input_size = 0;
    GE_ASSERT_NOTNULL(op_desc->GetInputDescPtr(0U), "Get input desc ptr failed");
    if (!is_dynamic_) {
      GE_CHK_STATUS_RET(TensorUtils::GetSize(*op_desc->GetInputDescPtr(0U), input_size),
                        "[Get][InputSize] failed in op:%s.", op_desc->GetName().c_str());
    } else {
      int64_t shape_size = op_desc->GetInputDescPtr(0U)->GetShape().GetShapeSize();
      shape_size = (shape_size < 0) ? 0 : shape_size;
      input_size = GetSizeInBytes(shape_size, op_desc->GetInputDescPtr(0U)->GetDataType());
      input_size = (input_size < 0) ? 0 : input_size;
    }

    model_tensor_desc.base_info.size = static_cast<size_t>(input_size);
    model_tensor_desc.base_info.dt = op_desc->GetInputDescPtr(0U)->GetDataType();
    model_tensor_desc.base_info.format = op_desc->GetInputDescPtr(0U)->GetFormat();

    model_tensor_desc.name = op_desc->GetName();
    model_tensor_desc.base_info.name_len = static_cast<uint32_t>(model_tensor_desc.name.size());

    modelIntroduction_.inputDesc.tensor_desc_size++;
    modelIntroduction_.inputDesc.descs.push_back(std::move(model_tensor_desc));
  }
  return SUCCESS;
}

Status ModelIntroduction::CreateOutput(const uint32_t index, const OpDescPtr &op_desc, ModelTensorDesc &output) {
  // netoutput input tensor desc
  const auto &tensor_desc = op_desc->GetInputDescPtr(index);
  GE_CHK_BOOL_RET_STATUS(tensor_desc != nullptr, FAILED,
                         "[GET][InputDescPtr] intput_desc index:%u in op:%s(%s) does not exist", index,
                         op_desc->GetName().c_str(), op_desc->GetType().c_str());
  const Format format = tensor_desc->GetFormat();
  const GeShape shape = tensor_desc->GetShape();
  const DataType data_type = tensor_desc->GetDataType();
  std::vector<std::pair<int64_t, int64_t>> shape_ranges;
  (void)tensor_desc->GetShapeRange(shape_ranges);
  GELOGD("op:%s(%s) shape_ranges size=%zu", op_desc->GetName().c_str(), op_desc->GetType().c_str(),
         shape_ranges.size());
  if (tensor_desc->GetFormat() == FORMAT_FRACTAL_Z) {  // FraczToHWCK
    const int64_t k = shape.GetDim(0U);                // 0: first dim
    const int64_t c = shape.GetDim(1U);                // 1: second dim
    const int64_t h = shape.GetDim(2U);                // 2: third dim
    const int64_t w = shape.GetDim(3U);                // 3: fourth dim
    output.dims.push_back(h);
    output.dims.push_back(w);
    output.dims.push_back(c);
    output.dims.push_back(k);
    if (shape_ranges.size() == 4U) {                   // 4 dims
      output.shape_range.push_back(shape_ranges[2U]);  // h: 2
      output.shape_range.push_back(shape_ranges[3U]);  // w: 3
      output.shape_range.push_back(shape_ranges[1U]);  // c: 1
      output.shape_range.push_back(shape_ranges[0U]);  // k: 0
    }
  } else {
    for (size_t j = 0U; j < shape.GetDimNum(); ++j) {
      output.dims.push_back(shape.GetDim(j));
    }
    output.shape_range = shape_ranges;
  }
  output.base_info.dims_len = static_cast<uint32_t>(sizeof(int64_t) * output.dims.size());
  output.base_info.shape_range_len =
      static_cast<uint32_t>(sizeof(std::pair<int64_t, int64_t>) * output.shape_range.size());
  int64_t tensor_size = 0;
  if (AttrUtils::GetInt(tensor_desc, ATTR_NAME_SPECIAL_INPUT_SIZE, tensor_size) && (tensor_size > 0)) {
    GELOGD("netoutput[%s] [%d]th input special size [%" PRId64 "]", op_desc->GetName().c_str(), index, tensor_size);
  } else {
    (void)TensorUtils::CalcTensorMemSize(shape, format, data_type, tensor_size);
    tensor_size = (tensor_size == kMemSizeUnknownShape) ? 0 : tensor_size;
  }
  output.base_info.size = static_cast<uint64_t>(tensor_size);
  output.base_info.dt = data_type;
  output.base_info.format = format;
  output.base_info.dimsV2_len = output.base_info.dims_len;
  output.dimsV2 = output.dims;
  return SUCCESS;
}

Status ModelIntroduction::ConstructOutputInfo() {
  Status ret = SUCCESS;
  for (const auto &op_desc : output_op_list_) {
    const size_t out_size = op_desc->GetInputsSize();
    for (size_t i = 0U; i < out_size; ++i) {
      std::string output_name;
      ModelTensorDesc output;
      ret = CreateOutput(static_cast<uint32_t>(i), op_desc, output);
      GE_CHK_BOOL_RET_STATUS(ret == SUCCESS, FAILED, "output[%zu] is invalid", i);
      const auto src_name = op_desc->GetSrcName();
      const auto src_index = op_desc->GetSrcIndex();
      GE_CHK_BOOL_RET_STATUS((src_name.size() > i) && (src_index.size() > i), INTERNAL_ERROR,
                             "[Check][Param] construct output failed, as index:%zu >= src name size:%zu, "
                             "or index >= src index size:%zu, op:%s.",
                             i, src_name.size(), src_index.size(), op_desc->GetName().c_str());
      // forward compatbility, if old om has no out_node_name, need to return output follow origin way
      if (out_size == out_node_name_.size()) {
        // neweast plan, the index will add to name during generate model.
        const bool contains_colon = out_node_name_[i].find(":") != std::string::npos;
        output_name = contains_colon ? out_node_name_[i] : (out_node_name_[i] + ":" + std::to_string(src_index[i]));
      } else {
        output_name =
            std::string("output_") + std::to_string(i) + "_" + src_name[i] + "_" + std::to_string(src_index[i]);
      }
      output.name = output_name;
      output.base_info.name_len = static_cast<uint32_t>(output_name.length());
      modelIntroduction_.outputDesc.descs.push_back(std::move(output));
      modelIntroduction_.outputDesc.tensor_desc_size++;
    }
  }
  return SUCCESS;
}

Status ModelIntroduction::GetDynamicInfoFromCase(int32_t &dynamic_type, std::vector<std::vector<int64_t>> &batch_info) {
  uint32_t batch_num = 0U;
  GE_IF_BOOL_EXEC(!AttrUtils::GetInt(case_desc_, ATTR_NAME_BATCH_NUM, batch_num),
                  GELOGD("Not multi-batch Node: %s", case_desc_->GetName().c_str());
                  return SUCCESS);
  for (uint32_t i = 0U; i < batch_num; ++i) {
    std::vector<int64_t> batch_shape;
    const std::string attr_name = ATTR_NAME_PRED_VALUE + "_" + std::to_string(i);
    GE_IF_BOOL_EXEC(!AttrUtils::GetListInt(case_desc_, attr_name, batch_shape),
                    REPORT_INNER_ERR_MSG("E19999", "Get Attr:%s from op:%s(%s) fail", attr_name.c_str(),
                                       case_desc_->GetName().c_str(), case_desc_->GetType().c_str());
                    return FAILED);
    (void)batch_info.emplace_back(batch_shape);
  }
  (void)AttrUtils::GetInt(case_desc_, ATTR_DYNAMIC_TYPE, dynamic_type);
  return SUCCESS;
}

Status ModelIntroduction::ConstructDynamicInfo() {
  if (case_desc_ == nullptr) {
    GEEVENT("there is no case, no dynamic info");
    return SUCCESS;
  }

  std::vector<std::vector<int64_t>> batch_info;
  int32_t dynamic_type = NOT_DYNAMIC_MODE;
  const Status ret = GetDynamicInfoFromCase(dynamic_type, batch_info);
  GE_RETURN_WITH_LOG_IF_ERROR(ret, "Get dynamic info faild.");

  GELOGD("dynamic type:%u, bathc_info size is %zu", dynamic_type, batch_info.size());
  switch (dynamic_type) {
    case NOT_DYNAMIC_MODE:
      GELOGI("normal case node, no need to gather dynamic info.");
      break;
    case static_cast<int32_t>(ge::DYNAMIC_BATCH):
      modelIntroduction_.dynamicBatch.vec_size = static_cast<uint32_t>(batch_info.size());
      for (auto batch : batch_info) {
        modelIntroduction_.dynamicBatch.value.push_back(batch[0]);
      }
      break;
    case static_cast<int32_t>(ge::DYNAMIC_IMAGE):
      modelIntroduction_.dynamicHW.value = batch_info;
      for (auto batch : batch_info) {
        modelIntroduction_.dynamicHW.vec_part_size.push_back(static_cast<uint32_t>(batch.size()));
      }
      modelIntroduction_.dynamicHW.vec_size = static_cast<uint32_t>(batch_info.size());
      break;
    case static_cast<int32_t>(ge::DYNAMIC_DIMS):
      modelIntroduction_.dynamicDims.value = batch_info;
      for (auto batch : batch_info) {
        modelIntroduction_.dynamicDims.vec_part_size.push_back(static_cast<uint32_t>(batch.size()));
      }
      modelIntroduction_.dynamicDims.vec_size = static_cast<uint32_t>(batch_info.size());
      break;
    default:
      GELOGE(FAILED, "invalid dynamic type[%d]", dynamic_type);
      return FAILED;
  }
  return SUCCESS;
}

void ModelIntroduction::ConstructDynamicOutShape() {
  std::vector<std::string> shape_info;
  for (const auto &op_desc : output_op_list_) {
    if (AttrUtils::GetListStr(op_desc, ATTR_NAME_DYNAMIC_OUTPUT_DIMS, shape_info)) {
      (void)modelIntroduction_.dynamicOutputShape.value.insert(modelIntroduction_.dynamicOutputShape.value.cend(),
                                                               shape_info.cbegin(), shape_info.cend());
    }
  }
  modelIntroduction_.dynamicOutputShape.vec_size =
      static_cast<uint32_t>(modelIntroduction_.dynamicOutputShape.value.size());
  for (std::string &shape : modelIntroduction_.dynamicOutputShape.value) {
    modelIntroduction_.dynamicOutputShape.str_len.push_back(static_cast<uint32_t>(shape.length()));
  }
}

void ModelIntroduction::ConstructNameOrder() {
  if (case_desc_ == nullptr) {
    GEEVENT("there is no case, no data name order info.");
    return;
  }
  std::vector<std::string> user_designate_shape_order;
  if (!AttrUtils::GetListStr(case_desc_, ATTR_USER_DESIGNEATE_SHAPE_ORDER, user_designate_shape_order)) {
    return;
  }
  modelIntroduction_.dataNameOrder.value = user_designate_shape_order;
  modelIntroduction_.dataNameOrder.vec_size = static_cast<uint32_t>(user_designate_shape_order.size());
  for (auto &name : user_designate_shape_order) {
    modelIntroduction_.dataNameOrder.str_len.push_back(static_cast<uint32_t>(name.size()));
  }
}

void ModelIntroduction::TlvBlockSize(BaseTlvBlock &tlv_block) {
  if (!tlv_block.NeedSave()) {
    return;
  }
  total_size_ += UINT32_SIZE + UINT32_SIZE;
  total_size_ += static_cast<uint32_t>(tlv_block.Size());
}

uint32_t ModelIntroduction::DataSize() {
  if (total_size_ != 0) {
    GELOGD("Calculation is complete, total_size:%u", total_size_);
    return total_size_;
  }
  TlvBlockSize(modelIntroduction_.inputDesc);
  TlvBlockSize(modelIntroduction_.outputDesc);
  TlvBlockSize(modelIntroduction_.dynamicBatch);
  TlvBlockSize(modelIntroduction_.dynamicHW);
  TlvBlockSize(modelIntroduction_.dynamicDims);
  TlvBlockSize(modelIntroduction_.dynamicOutputShape);
  TlvBlockSize(modelIntroduction_.dataNameOrder);
  GELOGD("ModelIoInfo total_size:%u", total_size_);
  return total_size_;
}

Status ModelIntroduction::SaveTlvBlock(BaseTlvBlock &tlv_block, const ModelDescType type, uint8_t **const write_addr,
                                       size_t &left_size) {
  GE_IF_BOOL_EXEC(
      (write_addr == nullptr) || (*write_addr == nullptr),
      GELOGE(PARAM_INVALID, "input param is invalid, addr valid %d.", static_cast<int32_t>(write_addr != nullptr));
      return FAILED);
  // need to save
  if (!tlv_block.NeedSave()) {
    GELOGI("skip save block[type:%d]", static_cast<int32_t>(type));
    return SUCCESS;
  }
  const uint32_t begin_size = static_cast<uint32_t>(left_size);
  // copy type
  errno_t ret = memcpy_s(*write_addr, left_size, &type, TYPE_SIZE);
  GE_CHK_BOOL_RET_STATUS(ret == EOK, FAILED, "copy type failed");
  *write_addr = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(*write_addr) + TYPE_SIZE));
  left_size -= TYPE_SIZE;

  // copy length
  const uint32_t value_size = static_cast<uint32_t>(tlv_block.Size());
  ret = memcpy_s(*write_addr, left_size, &value_size, UINT32_SIZE);
  GE_CHK_BOOL_RET_STATUS(ret == EOK, FAILED, "copy length failed");
  *write_addr = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(*write_addr) + UINT32_SIZE));
  left_size -= UINT32_SIZE;

  // copy value
  const bool res = tlv_block.Serilize(write_addr, left_size);
  GE_CHK_BOOL_RET_STATUS(res, FAILED, "copy value failed");
  GELOGD("save data blob[%d], size[%u], left[%u].", static_cast<int32_t>(type), begin_size - left_size, left_size);
  return SUCCESS;
}

std::shared_ptr<uint8_t> ModelIntroduction::Data() {
  size_t left_size = static_cast<size_t>(DataSize());
  GEEVENT("model io_info size:%u", total_size_);
  buff_.reset(new (std::nothrow) uint8_t[total_size_], std::default_delete<uint8_t[]>());
  uint8_t *write_addr = buff_.get();
  GE_IF_BOOL_EXEC(write_addr == nullptr,
                  GELOGE(FAILED, " create continuous blob for modelIoInfo failed, size[%u]", total_size_);
                  return nullptr);
  Status ret = SaveTlvBlock(modelIntroduction_.inputDesc, ModelDescType::MODEL_INPUT_DESC, &write_addr, left_size);
  CHECK_FALSE_EXEC(ret == SUCCESS, GELOGE(FAILED, "save MODEL_INPUT_DESC failed.");
                   return nullptr);

  ret = SaveTlvBlock(modelIntroduction_.outputDesc, ModelDescType::MODEL_OUTPUT_DESC, &write_addr, left_size);
  CHECK_FALSE_EXEC(ret == SUCCESS, GELOGE(FAILED, "save MODEL_OUTPUT_DESC failed.");
                   return nullptr);

  ret = SaveTlvBlock(modelIntroduction_.dynamicBatch, ModelDescType::MODEL_DYNAMIC_BATCH, &write_addr, left_size);
  CHECK_FALSE_EXEC(ret == SUCCESS, GELOGE(FAILED, "save MODEL_DYNAMIC_BATCH failed.");
                   return nullptr);

  ret = SaveTlvBlock(modelIntroduction_.dynamicHW, ModelDescType::MODEL_DYNAMIC_HW, &write_addr, left_size);
  CHECK_FALSE_EXEC(ret == SUCCESS, GELOGE(FAILED, "save MODEL_DYNAMIC_HW failed.");
                   return nullptr);

  ret = SaveTlvBlock(modelIntroduction_.dynamicDims, ModelDescType::MODEL_DYNAMIC_DIMS, &write_addr, left_size);
  CHECK_FALSE_EXEC(ret == SUCCESS, GELOGE(FAILED, "save MODEL_DYNAMIC_DIMS failed.");
                   return nullptr);

  ret = SaveTlvBlock(modelIntroduction_.dynamicOutputShape, ModelDescType::MODEL_DYNAMIC_OUTPUT_SHAPE, &write_addr,
                     left_size);
  CHECK_FALSE_EXEC(ret == SUCCESS, GELOGE(FAILED, "save MODEL_DYNAMIC_OUTPUT_SHAPE failed.");
                   return nullptr);

  ret = SaveTlvBlock(modelIntroduction_.dataNameOrder, ModelDescType::MODEL_DESIGNATE_SHAPE_ORDER, &write_addr,
                     left_size);
  CHECK_FALSE_EXEC(ret == SUCCESS, GELOGE(FAILED, "save MODEL_DESIGNATE_SHAPE_ORDER failed.");
                   return nullptr);

  GE_IF_BOOL_EXEC(left_size != 0, GELOGE(FAILED, "left size must be 0, but current is %u", left_size);
                  return nullptr);
  return buff_;
}
}  // namespace ge
