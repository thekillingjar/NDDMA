/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/graph_optimizer/fusion_common/op_slice_info.h"
#include <algorithm>
#include "framework/common/debug/ge_log.h"

namespace fe {
#define FE_MAKE_SHARED(exec_expr0, exec_expr1) \
  do {                                         \
    try {                                      \
      exec_expr0;                              \
    } catch (...) {                            \
      GELOGW("Make shared failed");            \
      exec_expr1;                              \
    }                                          \
  } while (0)

class InputSplitInfoImpl {
public:
  size_t GetIndex() const { return idx_; }
  std::vector<int64_t> GetAxis() { return axis_; }
  std::vector<int64_t> GetHeadOverLap() { return head_over_lap_; }
  std::vector<int64_t> GetTailOverLap() { return tail_over_lap_; }
  void SetIndex(const size_t& idx) { idx_ = idx; }
  void SetAxis(std::vector<int64_t>& axis) { axis_ = axis; }
  void SetHeadOverLap(std::vector<int64_t>& head_over_lap) { head_over_lap_ = head_over_lap; }
  void SetTailOverLap(std::vector<int64_t>& tail_over_lap) { tail_over_lap_ = tail_over_lap; }

private:
  size_t idx_ = 0;
  std::vector<int64_t> axis_;
  std::vector<int64_t> head_over_lap_;
  std::vector<int64_t> tail_over_lap_;
};

InputSplitInfo::InputSplitInfo() {}

InputSplitInfo::~InputSplitInfo() {}

InputSplitInfo::InputSplitInfo(const InputSplitInfo &input_split_info) {
  this->split_impl_ = input_split_info.split_impl_;
}

InputSplitInfo &InputSplitInfo::operator = (const InputSplitInfo &input_split_info) {
  this->split_impl_ = input_split_info.split_impl_;
  return *this;
}

bool InputSplitInfo::IsPtrNull() const {
  if (split_impl_ == nullptr) {
    return true;
  }

  return false;
}

bool InputSplitInfo::Initialize() {
  FE_MAKE_SHARED(split_impl_ = std::make_shared<InputSplitInfoImpl>(), return false);
  if (split_impl_== nullptr) {
    return false;
  }
  return true;
}

size_t InputSplitInfo::GetIndex() const {
  return split_impl_->GetIndex();
}

std::vector<int64_t> InputSplitInfo::GetAxis() const {
  return split_impl_->GetAxis();
}

std::vector<int64_t> InputSplitInfo::GetHeadOverLap() const {
  return split_impl_->GetHeadOverLap();
}

std::vector<int64_t> InputSplitInfo::GetTailOverLap() const {
  return split_impl_->GetTailOverLap();
}

void InputSplitInfo::SetIndex(const size_t& idx) {
  split_impl_->SetIndex(idx);
}

void InputSplitInfo::SetAxis(std::vector<int64_t>& axis) {
  split_impl_->SetAxis(axis);
}

void InputSplitInfo::SetHeadOverLap(std::vector<int64_t>& head_over_lap) {
  split_impl_->SetHeadOverLap(head_over_lap);
}

void InputSplitInfo::SetTailOverLap(std::vector<int64_t>& tail_over_lap) {
  split_impl_->SetTailOverLap(tail_over_lap);
}

class OutputSplitInfoImpl {
public:
  size_t GetIndex() const { return idx_; }
  std::vector<int64_t> GetAxis() { return axis_; }
  void SetIndex(const size_t& idx) { idx_ = idx; }
  void SetAxis(std::vector<int64_t>& axis) { axis_ = axis; }

private:
  size_t idx_ = 0;
  std::vector<int64_t> axis_;
};

OutputSplitInfo::OutputSplitInfo() {}

OutputSplitInfo::~OutputSplitInfo() {}

OutputSplitInfo::OutputSplitInfo(const OutputSplitInfo &output_split_info) {
  this->split_impl_ = output_split_info.split_impl_;
}

OutputSplitInfo &OutputSplitInfo::operator = (const OutputSplitInfo &output_split_info) {
  this->split_impl_ = output_split_info.split_impl_;
  return *this;
}

bool OutputSplitInfo::IsPtrNull() const {
  if (split_impl_ == nullptr) {
    return true;
  }
  return false;
}

bool OutputSplitInfo::Initialize() {
  FE_MAKE_SHARED(split_impl_ = std::make_shared<OutputSplitInfoImpl>(), return false);
  if (split_impl_== nullptr) {
    return false;
  }
  return true;
}

size_t OutputSplitInfo::GetIndex() const {
  return split_impl_->GetIndex();
}

std::vector<int64_t> OutputSplitInfo::GetAxis() const {
  return split_impl_->GetAxis();
}

void OutputSplitInfo::SetIndex(const size_t& idx) {
  split_impl_->SetIndex(idx);
}

void OutputSplitInfo::SetAxis(std::vector<int64_t>& axis) {
  split_impl_->SetAxis(axis);
}

class InputReduceInfoImpl {
public:
  size_t GetIndex() const { return idx_; }
  std::vector<int64_t> GetAxis() { return axis_; }
  void SetIndex(const size_t& idx) { idx_ = idx; }
  void SetAxis(std::vector<int64_t>& axis) { axis_ = axis; }

private:
  size_t idx_ = 0;
  std::vector<int64_t> axis_;
};

InputReduceInfo::InputReduceInfo() {}

InputReduceInfo::~InputReduceInfo() {}

bool InputReduceInfo::IsPtrNull() const {
  if (reduce_impl_ == nullptr) {
    return true;
  }
  return false;
}

bool InputReduceInfo::Initialize() {
  FE_MAKE_SHARED(reduce_impl_ = std::make_shared<InputReduceInfoImpl>(), return false);
  if (reduce_impl_ == nullptr) {
    return false;
  }
  return true;
}

InputReduceInfo::InputReduceInfo(const InputReduceInfo &input_reduce_info) {
  this->reduce_impl_ = input_reduce_info.reduce_impl_;
}

InputReduceInfo &InputReduceInfo::operator = (const InputReduceInfo &input_reduce_info) {
  this->reduce_impl_ = input_reduce_info.reduce_impl_;
  return *this;
}

size_t InputReduceInfo::GetIndex() const {
  return reduce_impl_->GetIndex();
}

std::vector<int64_t> InputReduceInfo::GetAxis() const {
  return reduce_impl_->GetAxis();
}

void InputReduceInfo::SetIndex(const size_t& idx) {
  reduce_impl_->SetIndex(idx);
}

void InputReduceInfo::SetAxis(std::vector<int64_t>& axis) {
  reduce_impl_->SetAxis(axis);
}

class OutputReduceInfoImpl {
public:
  size_t GetIndex() const { return idx_; }
  OpReduceType GetReduceType() const { return reduce_type_; }
  bool GetIsAtomic() const { return is_atomic_; }
  void SetIndex(const size_t& idx) { idx_ = idx; }
  void SetReduceType(const OpReduceType& reduce_type) { reduce_type_ = reduce_type; }
  void SetIsAtomic(const bool& is_atomic) { is_atomic_ = is_atomic; }
private:
  size_t idx_ = 0;
  OpReduceType reduce_type_;
  bool is_atomic_{false};
};

OutputReduceInfo::OutputReduceInfo() {}

OutputReduceInfo::~OutputReduceInfo() {}

OutputReduceInfo::OutputReduceInfo(const OutputReduceInfo &output_reduce_info) {
  this->reduce_impl_ = output_reduce_info.reduce_impl_;
}

OutputReduceInfo &OutputReduceInfo::operator = (const OutputReduceInfo &output_reduce_info) {
  this->reduce_impl_ = output_reduce_info.reduce_impl_;
  return *this;
}

bool OutputReduceInfo::IsPtrNull() const {
  if (reduce_impl_ == nullptr) {
    return true;
  }
  return false;
}

bool OutputReduceInfo::Initialize() {
  FE_MAKE_SHARED(reduce_impl_ = std::make_shared<OutputReduceInfoImpl>(), return false);
  if (reduce_impl_ == nullptr) {
    return false;
  }
  return true;
}

size_t OutputReduceInfo::GetIndex() const {
  return reduce_impl_->GetIndex();
}

OpReduceType OutputReduceInfo::GetReduceType() const {
  return reduce_impl_->GetReduceType();
}

bool OutputReduceInfo::GetIsAtomic() const {
  return reduce_impl_->GetIsAtomic();
}

void OutputReduceInfo::SetIndex(const size_t& idx) {
  reduce_impl_->SetIndex(idx);
}

void OutputReduceInfo::SetReduceType(const OpReduceType& reduce_type) {
  reduce_impl_->SetReduceType(reduce_type);
}

void OutputReduceInfo::SetIsAtomic(const bool& is_atomic) {
  reduce_impl_->SetIsAtomic(is_atomic);
}

class AxisSplitMapImpl {
public:
  std::vector<InputSplitInfoPtr> GetInputSplitInfos() { return input_split_vec_; }
  std::vector<OutputSplitInfoPtr> GetOutputSplitInfos() { return output_split_vec_; }
  void AddInputSplitInfo(const InputSplitInfo& input_split_info) {
    InputSplitInfoPtr input_split_info_ptr = nullptr;
    FE_MAKE_SHARED(input_split_info_ptr = std::make_shared<InputSplitInfo>(input_split_info), return);
    if (input_split_info_ptr == nullptr) {
      return;
    }
    if (input_split_info_ptr->IsPtrNull()) {
      if (!input_split_info_ptr->Initialize()) {
        return;
      }
    }

    input_split_vec_.push_back(input_split_info_ptr);
  }

  void SetInputSplitInfos(std::vector<InputSplitInfo>& input_split_vec) {
    input_split_vec_.clear();
    for (InputSplitInfo &input_split_info : input_split_vec) {
      AddInputSplitInfo(input_split_info);
    }
  }

  void SetInputSplitInfos(std::vector<InputSplitInfoPtr>& input_split_vec) {
    input_split_vec_ = input_split_vec;
  }

  void AddOutputSplitInfo(const OutputSplitInfo& output_split_info) {
    OutputSplitInfoPtr output_split_info_ptr = nullptr;
    FE_MAKE_SHARED(output_split_info_ptr = std::make_shared<OutputSplitInfo>(output_split_info), return);
    if (output_split_info_ptr == nullptr) {
      return;
    }

    if (output_split_info_ptr->IsPtrNull()) {
      if (!output_split_info_ptr->Initialize()) {
        return;
      }
    }

    output_split_vec_.push_back(output_split_info_ptr);
  }

  void SetOutputSplitInfos(std::vector<OutputSplitInfo>& output_split_vec) {
    output_split_vec_.clear();
    for (OutputSplitInfo &output_split_info : output_split_vec) {
      AddOutputSplitInfo(output_split_info);
    }
  }

  void SetOutputSplitInfos(std::vector<OutputSplitInfoPtr>& output_split_vec) {
    output_split_vec_ = output_split_vec;
  }

private:
  std::vector<InputSplitInfoPtr> input_split_vec_;
  std::vector<OutputSplitInfoPtr> output_split_vec_;
};

AxisSplitMap::AxisSplitMap() {}

AxisSplitMap::~AxisSplitMap() {}

AxisSplitMap::AxisSplitMap(const AxisSplitMap &axis_split_map) {
  this->aixs_split_impl_ = axis_split_map.aixs_split_impl_;
}

AxisSplitMap &AxisSplitMap::operator = (const AxisSplitMap &axis_split_map) {
  this->aixs_split_impl_ = axis_split_map.aixs_split_impl_;
  return *this;
}

bool AxisSplitMap::IsPtrNull() const {
  if (aixs_split_impl_ == nullptr) {
    return true;
  }
  return false;
}

bool AxisSplitMap::Initialize() {
  FE_MAKE_SHARED(aixs_split_impl_ = std::make_shared<AxisSplitMapImpl>(), return false);
  if (aixs_split_impl_ == nullptr) {
    return false;
  }
  return true;
}

std::vector<InputSplitInfoPtr> AxisSplitMap::GetInputSplitInfos() const {
  return aixs_split_impl_->GetInputSplitInfos();
}

std::vector<InputSplitInfo> AxisSplitMap::GetInputSplitInfoVec() const {
  std::vector<InputSplitInfo> ret;
  for (InputSplitInfoPtr info_ptr : aixs_split_impl_->GetInputSplitInfos()) {
    ret.push_back(*info_ptr);
  }
  return ret;
}

std::vector<OutputSplitInfoPtr> AxisSplitMap::GetOutputSplitInfos() const {
  return aixs_split_impl_->GetOutputSplitInfos();
}

std::vector<OutputSplitInfo> AxisSplitMap::GetOutputSplitInfoVec() const {
  std::vector<OutputSplitInfo> ret;
  for (OutputSplitInfoPtr info_ptr : aixs_split_impl_->GetOutputSplitInfos()) {
    ret.push_back(*info_ptr);
  }
  return ret;
}

void AxisSplitMap::AddInputSplitInfo(InputSplitInfo& input_split_info) {
  aixs_split_impl_->AddInputSplitInfo(input_split_info);
}

void AxisSplitMap::SetInputSplitInfos(std::vector<InputSplitInfo>& input_split_vec) {
  aixs_split_impl_->SetInputSplitInfos(input_split_vec);
}

void AxisSplitMap::SetInputSplitInfos(std::vector<InputSplitInfoPtr>& input_split_vec) {
  aixs_split_impl_->SetInputSplitInfos(input_split_vec);
}

void AxisSplitMap::AddOutputSplitInfo(OutputSplitInfo& output_split_info) {
  aixs_split_impl_->AddOutputSplitInfo(output_split_info);
}

void AxisSplitMap::SetOutputSplitInfos(std::vector<OutputSplitInfo>& output_split_vec) {
  aixs_split_impl_->SetOutputSplitInfos(output_split_vec);
}

void AxisSplitMap::SetOutputSplitInfos(std::vector<OutputSplitInfoPtr>& output_split_vec) {
  aixs_split_impl_->SetOutputSplitInfos(output_split_vec);
}

class AxisReduceMapImpl {
public:
  std::vector<InputReduceInfoPtr> GetInputReduceInfos() { return input_reduce_vec_; }
  std::vector<OutputReduceInfoPtr> GetOutputReduceInfos() { return output_reduce_vec_; }

  void AddInputReduceInfo(const InputReduceInfo& input_reduce_info) {
    InputReduceInfoPtr input_reduce_info_ptr = nullptr;
    FE_MAKE_SHARED(input_reduce_info_ptr = std::make_shared<InputReduceInfo>(input_reduce_info), return);
    if (input_reduce_info_ptr == nullptr) {
      return;
    }

      if(input_reduce_info_ptr->IsPtrNull()) {
        if (!input_reduce_info_ptr->Initialize()) {
          return;
        }
    }

    input_reduce_vec_.push_back(input_reduce_info_ptr);
  }

  void SetInputReduceInfos(std::vector<InputReduceInfo>& input_reduce_vec) {
    input_reduce_vec_.clear();
    for (InputReduceInfo &input_reduce_info : input_reduce_vec) {
      AddInputReduceInfo(input_reduce_info);
    }
  }

  void SetInputReduceInfos(std::vector<InputReduceInfoPtr>& input_reduce_vec) {
    input_reduce_vec_ = input_reduce_vec;
  }

  void AddOutputReduceInfo(const OutputReduceInfo& output_reduce_info) {
    OutputReduceInfoPtr output_reduce_info_ptr = nullptr;
    FE_MAKE_SHARED(output_reduce_info_ptr = std::make_shared<OutputReduceInfo>(output_reduce_info), return);
    if (output_reduce_info_ptr == nullptr) {
      return;
    }

    if (output_reduce_info_ptr->IsPtrNull())  {
      if (!output_reduce_info_ptr->Initialize()) {
        return;
      }
    }
    output_reduce_vec_.push_back(output_reduce_info_ptr);
  }

  void SetOutputReduceInfos(std::vector<OutputReduceInfo>& output_reduce_vec) {
    output_reduce_vec_.clear();
    for (OutputReduceInfo &output_reduce_info : output_reduce_vec) {
      AddOutputReduceInfo(output_reduce_info);
    }
  }

  void SetOutputReduceInfos(std::vector<OutputReduceInfoPtr>& output_reduce_vec) {
    output_reduce_vec_ = output_reduce_vec;
  }

private:
  std::vector<InputReduceInfoPtr> input_reduce_vec_;
  std::vector<OutputReduceInfoPtr> output_reduce_vec_;
};

AxisReduceMap::AxisReduceMap() {}

AxisReduceMap::~AxisReduceMap() {}

AxisReduceMap::AxisReduceMap(const AxisReduceMap &axis_reduce_map) {
  this->aixs_reduce_impl_ = axis_reduce_map.aixs_reduce_impl_;
}

AxisReduceMap &AxisReduceMap::operator = (const AxisReduceMap &axis_reduce_map) {
  this->aixs_reduce_impl_ = axis_reduce_map.aixs_reduce_impl_;
  return *this;
}

bool AxisReduceMap::IsPtrNull() const {
  if (aixs_reduce_impl_ == nullptr) {
    return true;
  }
  return false;
}

bool AxisReduceMap::Initialize() {
  FE_MAKE_SHARED(aixs_reduce_impl_ = std::make_shared<AxisReduceMapImpl>(), return false);
  if (aixs_reduce_impl_ == nullptr) {
    return false;
  }
  return true;
}

std::vector<InputReduceInfoPtr> AxisReduceMap::GetInputReduceInfos() const {
  return aixs_reduce_impl_->GetInputReduceInfos();
}

std::vector<InputReduceInfo> AxisReduceMap::GetInputReduceInfoVec() const {
  std::vector<InputReduceInfo> ret;
  for (InputReduceInfoPtr info_ptr : aixs_reduce_impl_->GetInputReduceInfos()) {
    ret.push_back(*info_ptr);
  }
  return ret;
}

std::vector<OutputReduceInfoPtr> AxisReduceMap::GetOutputReduceInfos() const {
  return aixs_reduce_impl_->GetOutputReduceInfos();
}

std::vector<OutputReduceInfo> AxisReduceMap::GetOutputReduceInfoVec() const {
  std::vector<OutputReduceInfo> ret;
  for (OutputReduceInfoPtr info_ptr : aixs_reduce_impl_->GetOutputReduceInfos()) {
    ret.push_back(*info_ptr);
  }
  return ret;
}

void AxisReduceMap::AddInputReduceInfo(InputReduceInfo& input_reduce_info) {
  aixs_reduce_impl_->AddInputReduceInfo(input_reduce_info);
}

void AxisReduceMap::SetInputReduceInfos(std::vector<InputReduceInfo>& input_reduce_vec) {
  aixs_reduce_impl_->SetInputReduceInfos(input_reduce_vec);
}

void AxisReduceMap::SetInputReduceInfos(std::vector<InputReduceInfoPtr>& input_reduce_vec) {
  aixs_reduce_impl_->SetInputReduceInfos(input_reduce_vec);
}

void AxisReduceMap::AddOutputReduceInfo(OutputReduceInfo& output_reduce_info) {
  aixs_reduce_impl_->AddOutputReduceInfo(output_reduce_info);
}

void AxisReduceMap::SetOutputReduceInfos(std::vector<OutputReduceInfo>& output_reduce_vec) {
  aixs_reduce_impl_->SetOutputReduceInfos(output_reduce_vec);
}

void AxisReduceMap::SetOutputReduceInfos(std::vector<OutputReduceInfoPtr>& output_reduce_vec) {
  aixs_reduce_impl_->SetOutputReduceInfos(output_reduce_vec);
}

class OpCalcInfoImpl {
public:
  std::vector<AxisSplitMapPtr> GetAxisSplitMaps() { return axis_split_vec_; }
  std::vector<AxisReduceMapPtr> GetAxisReduceMaps() { return axis_reduce_vec_; }
  OpL1FusionType GetL1FusionEnable() const { return l1_fusion_enable_; }
  int64_t GetMinTbeL1Space() const { return min_tbe_l1_space_; }

  void AddAxisSplitMap(const AxisSplitMap& axis_split_map) {
    AxisSplitMapPtr axis_split_map_ptr = nullptr;
    FE_MAKE_SHARED(axis_split_map_ptr = std::make_shared<AxisSplitMap>(axis_split_map), return);
    if (axis_split_map_ptr == nullptr) {
      return;
    }
    axis_split_vec_.push_back(axis_split_map_ptr);
  }

  void SetAxisSplitMaps(std::vector<AxisSplitMap>& axis_split_vec) {
    axis_split_vec_.clear();
    for (AxisSplitMap &axis_split_map : axis_split_vec) {
      AddAxisSplitMap(axis_split_map);
    }
  }

  void SetAxisSplitMaps(std::vector<AxisSplitMapPtr>& axis_split_vec) { axis_split_vec_ = axis_split_vec; }

  void AddAxisReduceMap(const AxisReduceMap& axis_reduce_map) {
    AxisReduceMapPtr axis_reduce_map_ptr = nullptr;
    FE_MAKE_SHARED(axis_reduce_map_ptr = std::make_shared<AxisReduceMap>(axis_reduce_map), return);
    if (axis_reduce_map_ptr == nullptr) {
      return;
    }
    axis_reduce_vec_.push_back(axis_reduce_map_ptr);
  }

  void SetAxisReduceMaps(std::vector<AxisReduceMap>& axis_reduce_vec) {
    axis_reduce_vec_.clear();
    for (AxisReduceMap &axis_reduce_map : axis_reduce_vec) {
      AddAxisReduceMap(axis_reduce_map);
    }
  }

  void SetAxisReduceMaps(std::vector<AxisReduceMapPtr>& axis_reduce_vec) { axis_reduce_vec_ = axis_reduce_vec; }
  void SetL1FusionEnable(const OpL1FusionType& l1_fusion_enable) { l1_fusion_enable_ = l1_fusion_enable; }
  void SetMinTbeL1Space(const int64_t& min_tbe_l1_space) { min_tbe_l1_space_ = min_tbe_l1_space; }

  void DelAxisSplitMapBaseAxis(std::vector<int64_t>& axis) {
    AxisSplitMapPtr temp_axis_split_map;
    for (AxisSplitMapPtr axis_split : axis_split_vec_) {
      for (InputSplitInfoPtr input_split : axis_split->GetInputSplitInfos()) {
        if (input_split->GetAxis() == axis) {
          temp_axis_split_map = axis_split;
        }
      }
    }

    std::vector<AxisSplitMapPtr>::iterator iter =
            std::find(axis_split_vec_.begin(), axis_split_vec_.end(), temp_axis_split_map);
    if (iter != axis_split_vec_.end()) {
      std::ignore = axis_split_vec_.erase(iter);
    }
  }

private:
  std::vector<AxisSplitMapPtr> axis_split_vec_;
  std::vector<AxisReduceMapPtr> axis_reduce_vec_;
  OpL1FusionType l1_fusion_enable_ = L1FUSION_DISABLE;
  int64_t min_tbe_l1_space_ = 0;
};

OpCalcInfo::OpCalcInfo() {}

OpCalcInfo::~OpCalcInfo() {}

bool OpCalcInfo::IsPtrNull() const {
  if (op_calc_info_impl_ == nullptr) {
    return true;
  }
  return false;
}

bool OpCalcInfo::Initialize() {
  FE_MAKE_SHARED(op_calc_info_impl_ = std::make_shared<OpCalcInfoImpl>(), return false);
  if (op_calc_info_impl_ == nullptr) {
    return false;
  }
  return true;
}

std::vector<AxisSplitMapPtr> OpCalcInfo::GetAxisSplitMaps() const {
  return op_calc_info_impl_->GetAxisSplitMaps();
}

std::vector<AxisReduceMapPtr> OpCalcInfo::GetAxisReduceMaps() const {
  return op_calc_info_impl_->GetAxisReduceMaps();
}

std::vector<AxisSplitMap> OpCalcInfo::GetAxisSplitMapVec() const {
  std::vector<AxisSplitMap> ret;
  for (AxisSplitMapPtr map_ptr : op_calc_info_impl_->GetAxisSplitMaps()) {
    ret.push_back(*map_ptr);
  }
  return ret;
}

std::vector<AxisReduceMap> OpCalcInfo::GetAxisReduceMapVec() const {
  std::vector<AxisReduceMap> ret;
  for (AxisReduceMapPtr map_ptr : op_calc_info_impl_->GetAxisReduceMaps()) {
    ret.push_back(*map_ptr);
  }
  return ret;
}

OpL1FusionType OpCalcInfo::GetL1FusionEnable() const {
  return op_calc_info_impl_->GetL1FusionEnable();
}

int64_t OpCalcInfo::GetMinTbeL1Space() const {
  return op_calc_info_impl_->GetMinTbeL1Space();
}

void OpCalcInfo::AddAxisSplitMap(AxisSplitMap& axis_split_map) {
  op_calc_info_impl_->AddAxisSplitMap(axis_split_map);
}

void OpCalcInfo::SetAxisSplitMaps(std::vector<AxisSplitMap>& axis_split_vec) {
  op_calc_info_impl_->SetAxisSplitMaps(axis_split_vec);
}

void OpCalcInfo::SetAxisSplitMaps(std::vector<AxisSplitMapPtr>& axis_split_vec) {
  op_calc_info_impl_->SetAxisSplitMaps(axis_split_vec);
}

void OpCalcInfo::AddAxisReduceMap(AxisReduceMap& axis_reduce_map) {
  op_calc_info_impl_->AddAxisReduceMap(axis_reduce_map);
}

void OpCalcInfo::SetAxisReduceMaps(std::vector<AxisReduceMap>& axis_reduce_vec) {
  op_calc_info_impl_->SetAxisReduceMaps(axis_reduce_vec);
}

void OpCalcInfo::SetAxisReduceMaps(std::vector<AxisReduceMapPtr>& axis_reduce_vec) {
  op_calc_info_impl_->SetAxisReduceMaps(axis_reduce_vec);
}

void OpCalcInfo::SetL1FusionEnable(const OpL1FusionType& l1_fusion_enable) {
  op_calc_info_impl_->SetL1FusionEnable(l1_fusion_enable);
}

void OpCalcInfo::SetMinTbeL1Space(const int64_t& min_tbe_l1_space) {
  op_calc_info_impl_->SetMinTbeL1Space(min_tbe_l1_space);
}

void OpCalcInfo::DelAxisSplitMapBaseAxis(std::vector<int64_t>& axis) {
  op_calc_info_impl_->DelAxisSplitMapBaseAxis(axis);
}

}  // namespace fe
