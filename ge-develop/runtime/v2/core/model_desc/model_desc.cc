/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "runtime/model_desc.h"

#include <cinttypes>
#include "core/utils/tensor_utils.h"
#include "common/debug/ge_log.h"
#include "graph/load/model_manager/aipp_utils.h"

namespace gert {
const Shape &ShapeRange::GetMin() const {
  return min_;
}

const Shape &ShapeRange::GetMax() const {
  return max_;
}

Shape &ShapeRange::MutableMin() {
  return min_;
}

Shape &ShapeRange::MutableMax() {
  return max_;
}
ShapeRange::ShapeRange(const Shape &min_shape, const Shape &max_shape) : min_(min_shape), max_(max_shape) {}
bool ShapeRange::operator==(const ShapeRange &other) const {
  return min_ == other.min_ && max_ == other.max_;
}

const ge::char_t *ModelIoDesc::GetName() const {
  return name_;
}

int32_t ModelIoDesc::GetDataType() const {
  return data_type_;
}

ge::Format ModelIoDesc::GetStorageFormat() const {
  return format_.GetStorageFormat();
}

ge::Format ModelIoDesc::GetOriginFormat() const {
  return format_.GetOriginFormat();
}

int64_t ModelIoDesc::GetSize() const {
  for (size_t i = 0U; i < GetStorageShape().GetDimNum(); ++i) {
    if (GetStorageShape()[i] < 0) {
      return 0;
    }
  }
  auto shape_size = GetStorageShape().GetShapeSize();
  if (shape_size <= 0) {
    return 0;
  } else {
    int64_t tensor_size = GetSizeInBytes(shape_size, static_cast<ge::DataType>(data_type_));
    if (tensor_size > 0) {
      return tensor_size;
    } else {
      GELOGW("[Get][Size] failed, shape size %" PRId64 ".", shape_size);
      return 0;
    }
  }
}

const Shape &ModelIoDesc::GetStorageShape() const {
  return shape_.GetStorageShape();
}

const Shape &ModelIoDesc::GetOriginShape() const {
  return shape_.GetOriginShape();
}

const ShapeRange &ModelIoDesc::GetOriginShapeRange() const {
  return origin_shape_range_;
}

const ShapeRange &ModelIoDesc::GetStorageShapeRange() const {
  return storage_shape_range_;
}

std::vector<std::pair<int64_t, int64_t>> ModelIoDesc::GetOriginShapeRangeVector() const {
  std::vector<std::pair<int64_t, int64_t>> range;
  for (size_t i = 0U; i < origin_shape_range_.GetMax().GetDimNum(); ++i) {
    range.emplace_back(std::make_pair(origin_shape_range_.GetMin().GetDim(i), origin_shape_range_.GetMax().GetDim(i)));
  }
  return range;
}

std::vector<std::pair<int64_t, int64_t>> ModelIoDesc::GetStorageShapeRangeVector() const {
  std::vector<std::pair<int64_t, int64_t>> range;
  for (size_t i = 0U; i < storage_shape_range_.GetMax().GetDimNum(); ++i) {
    range.emplace_back(std::make_pair(storage_shape_range_.GetMin().GetDim(i),
                                      storage_shape_range_.GetMax().GetDim(i)));
  }
  return range;
}

bool ModelIoDesc::IsShapeInRange(const Shape &shape) const {
  return (TensorUtils::CheckShapeByShapeRange(shape, storage_shape_range_) == ge::SUCCESS);
}

bool ModelIoDesc::IsOriginShapeInRange(const Shape &shape) const {
  return (TensorUtils::CheckShapeByShapeRange(shape, origin_shape_range_) == ge::SUCCESS);
}

void ModelIoDesc::SetName(const ge::char_t *name) {
  name_ = name;
}

void ModelIoDesc::SetDataType(int32_t data_type) {
  data_type_ = data_type;
}

void ModelIoDesc::SetStorageFormat(ge::Format format) {
  format_.SetStorageFormat(format);
}

void ModelIoDesc::SetOriginFormat(ge::Format format) {
  format_.SetOriginFormat(format);
}

Shape &ModelIoDesc::MutableStorageShape() {
  return shape_.MutableStorageShape();
}

Shape &ModelIoDesc::MutableOriginShape() {
  return shape_.MutableOriginShape();
}

ShapeRange &ModelIoDesc::MutableOriginShapeRange() {
  return origin_shape_range_;
}

ShapeRange &ModelIoDesc::MutableStorageShapeRange() {
  return storage_shape_range_;
}

size_t ModelDesc::CalcSize(size_t input_num, size_t output_num) {
  return sizeof(ModelDesc) + (input_num + output_num) * sizeof(ModelIoDesc);
}

const ModelIoDesc *ModelDesc::GetInputDesc(size_t index) const {
  return (reinterpret_cast<const ModelIoDesc *>(model_io_descs_.GetData()) + index);
}

const ModelIoDesc *ModelDesc::GetAllInputsDesc(size_t &input_num) const {
  input_num = input_num_;
  return reinterpret_cast<const ModelIoDesc *>(model_io_descs_.GetData());
}

const ModelIoDesc *ModelDesc::GetOutputDesc(size_t index) const {
  return (reinterpret_cast<const ModelIoDesc *>(model_io_descs_.GetData()) + input_num_ + index);
}

const ModelIoDesc *ModelDesc::GetAllOutputsDesc(size_t &output_num) const {
  output_num = output_num_;
  return reinterpret_cast<const ModelIoDesc *>(model_io_descs_.GetData()) + input_num_;
}

ModelIoDesc *ModelDesc::MutableInputDesc(size_t index) {
  return (reinterpret_cast<ModelIoDesc *>(model_io_descs_.MutableData()) + index);
}

ModelIoDesc *ModelDesc::AllMutableIoDesc(size_t &input_num, size_t &output_num) {
  input_num = input_num_;
  output_num = output_num_;
  return reinterpret_cast<ModelIoDesc *>(model_io_descs_.MutableData());
}

void ModelDesc::SetInputNum(size_t input_num) {
  input_num_ = input_num;
}

void ModelDesc::SetOutputNum(size_t output_num) {
  output_num_ = output_num;
}

ModelIoDesc *ModelDesc::MutableOutputDesc(size_t index) {
  return (reinterpret_cast<ModelIoDesc *>(model_io_descs_.MutableData()) + input_num_ + index);
}

const Shape &ModelIoDesc::GetAippShape() const {
  return aipp_shape_;
}

Shape &ModelIoDesc::MutableAippShape() {
  return aipp_shape_;
}

ge::graphStatus ModelDesc::GetDynamicBatchInfo(std::vector<std::vector<int64_t>> &batch_info,
                                               int32_t &dynamic_type) const {
  (void)batch_info;
  dynamic_type = -1;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ModelDesc::GetUserDesignateShapeOrder(std::vector<std::string> &user_designate_shape_order) const {
  (void)user_designate_shape_order;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ModelDesc::GetModelAttrs(std::vector<std::string> &attrs) const {
  (void)attrs;
  return ge::GRAPH_SUCCESS;
}
size_t ModelDesc::GetInputNum() const {
  return input_num_;
}
size_t ModelDesc::GetOutputNum() const {
  return output_num_;
}
void ModelDesc::SetSpaceRegistries(OpImplSpaceRegistryV2Array *space_registries) {
  space_registries_ = space_registries;
}
OpImplSpaceRegistryV2Array *ModelDesc::GetSpaceRegistries() const {
  return space_registries_;
}
void ModelDesc::SetFileConstantWeightDir(const ge::char_t *file_constant_weight_dir) {
  file_constant_weight_dir_ = file_constant_weight_dir;
}
const ge::char_t *ModelDesc::GetFileConstantWeightDir() const {
  return file_constant_weight_dir_;
}

size_t ModelDesc::GetReusableStreamNum() const {
  return reusable_stream_num_;
}
void ModelDesc::SetReusableStreamNum(size_t stream_num) {
  reusable_stream_num_ = stream_num;
}
size_t ModelDesc::GetReusableEventNum() const {
  return reusable_event_num_;
}
void ModelDesc::SetReusableEventNum(size_t event_num) {
  reusable_event_num_ = event_num;
}

size_t ModelDesc::GetReusableNotifyNum() const {
  return reusable_notify_num_;
}

void ModelDesc::SetReusableNotifyNum(const size_t notify_num) {
  reusable_notify_num_ = notify_num;
}

size_t ModelDesc::GetAttachedStreamNum() const {
  return attached_stream_num_;
}
void ModelDesc::SetAttachedStreamNum(const size_t stream_num) {
  attached_stream_num_ = stream_num;
}
}  // namespace gert
