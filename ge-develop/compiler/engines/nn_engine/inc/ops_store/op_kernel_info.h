/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_INC_OPS_STORE_OP_KERNEL_INFO_H_
#define FUSION_ENGINE_INC_OPS_STORE_OP_KERNEL_INFO_H_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <array>
#include "graph_optimizer/graph_optimize_register_error_codes.h"
#include "common/aicore_util_types.h"
#include "common/opskernel/ops_kernel_info_types.h"

#define GenerateOpKernelFlagFunc(flag_item)                             \
bool Is##flag_item() const {                                            \
  return op_flag_vec_[static_cast<size_t>(OP_KERNEL_FLAG::flag_item)];  \
}

#define GenerateOpKernelParamFunc(param_type, param_item)                                           \
param_type Get##param_item() const {                                                                \
  return static_cast<param_type>(op_param_vec_[static_cast<size_t>(OP_KERNEL_PARAM::param_item)]);  \
}

#define GenerateOpKernelStrParamFunc(param_item)                                  \
const std::string& Get##param_item() const {                                      \
  return op_str_param_vec_[static_cast<size_t>(OP_KERNEL_STR_PARAM::param_item)]; \
}

#define GenerateOpKernelStrParamVecFunc(param_item)                                  \
const std::vector<std::string>& Get##param_item##Vec() const {                            \
  return op_str_param_2_vec_[static_cast<size_t>(OP_KERNEL_STR_PARAM::param_item)];  \
}

namespace fe {
using AttrTypePair = std::pair<std::string, ge::GeAttrValue::ValueType>;
enum class OP_KERNEL_FLAG {
  NeedCheckSupport = 0,
  DynamicFormat,
  HeavyOp,
  SoftSyncOp,
  SupportDynamicShape,
  InputMemContinues,
  OutputMemContinues,
  OpFlagBottom
};

enum class OP_KERNEL_PARAM {
  OpPattern = 0,
  PrecisionPolicy,
  OpSlicePattern,
  JitCompileType,
  RangeLimitType,
  DynamicRankType,
  DynamicCompileStatic,
  AclnnSupportType,
  MultiKernelSupportType,
  EnableVectorCore,
  OpParamBottom
};

enum class OP_KERNEL_STR_PARAM {
  OpImpPath = 0,
  CoreType,
  OpImplSwitch,
  PrebuildPattern,
  PromoteType,
  OpStrParamBottom
};

struct PromoteTypeVal {
  std::vector<std::vector<std::string>> promote_val;
  bool is_dynamic = false;
  bool promote_flag = false;
};

struct FallbackFlags {
  std::vector<bool> fallbacks;
  std::vector<bool> unknown_shape_fallbacks;
};

class InputOrOutputInfo {
  /* Set OpsKernelInfoConstructor as a friend class of InputOrOutputInfo to
   * conveniently set the private attributes of InputOrOutputInfo.
   */
  friend class OpKernelInfoConstructor;

 public:
  explicit InputOrOutputInfo(std::string name);
  ~InputOrOutputInfo();

  const std::string& GetName() const;
  const bool& GetIsInput() const;
  uint32_t GetIndex() const;
  const vector<ge::DataType>& GetDataType() const;
  const vector<ge::Format>& GetFormat() const;
  const vector<uint32_t>& GetSubformat() const;
  const vector<ge::DataType>& GetUnknownShapeDataType() const;
  const vector<ge::Format>& GetUnknownShapeFormat() const;
  OpParamType GetParamType() const;
  bool GetTuneFormatSwitch() const;
  OpConstValueDepend GetConstValueDepend() const;
  const std::string& GetReshapeType() const;
  void SetReshapeType(const std::string &reshape_type);
  const std::string& GetUniqueName() const;

 private:
  bool is_input_;    /* whether it's input or output. true means input */
  std::string name_; /* name of input or output */
  std::string unique_name_;

  /* The index of input or output; Index is generated from json files,
   * like "input0", "input1" ....If the number after input is not consecutive or absent,
   * program will report error when loading op kernel information
   */
  uint32_t index_;

  /* Reshape type is a method which is used when we transfer this input into 4D if it is
   * is 1,2,3D or more than 6D. */
  std::string reshape_type_;

  /* Reshape type which needed to be transfered which is used when we transfer
   * this input into 4D if it is 1,2,3D or more than 6D. */
  std::string propagate_reshape_type_;

  OpParamType op_param_type_; /* Type of input or output, can be one of {dynamic, required, optional} */

  bool op_tune_format_switch_; /* weather input or output need tuneformat */

  OpConstValueDepend op_const_value_depend_; /* Whether the node of other end of input or output is const or constant. */

  /* Data formats that are supported by this op. The size of supported_formats_
   * should strictly be the same as supported_dtypes_. If not, program will report error
   * when initialize this op.
   */
  std::vector<ge::Format> supported_formats_;

  /* Data sub_formats that are supported by this op. The size of supported_formats_
   * do not need to strictly be the same as supported_formats_.
   */
  std::vector<uint32_t> supported_sub_formats_;

  /* Data types that are supported by this op. There not exists a flag
   * called "all_data_types_supported", because that does not make sense.
   */
  std::vector<ge::DataType> supported_dtypes_;

  /* Data formats that are supported by this op. The size of supported_formats_
   * should strictly be the same as supported_dtypes_. If not, program will report error
   * when initialize this op.
   */
  std::vector<ge::Format> supported_unknown_shape_formats_;

  /* Data types that are supported by this op. There not exists a flag
   * called "all_data_types_supported", because that does not make sense.
   */
  std::vector<ge::DataType> supported_unknown_shape_dtypes_;
};

using InputOrOutputInfoPtr = std::shared_ptr<InputOrOutputInfo>;

class AttrInfo {
  /* Set OpsKernelInfoConstructor as a friend class of AttrInfo to
   * conveniently set the private attributes of AttrInfo.
   */
  friend class OpKernelInfoConstructor;

 public:
  explicit AttrInfo(const std::string &attr_name);
  ~AttrInfo();

  const std::string& GetAttrName() const;

  const ge::GeAttrValue::ValueType& GetAttrDType() const;

  const bool& GetSupportAllValue() const;

  bool GetIsRequired() const;

  const std::vector<ge::GeAttrValue>& GetSupportedAttrValueVector() const;

  const bool& GetDefaultValueDefinedFlag() const;

  const ge::GeAttrValue& GetDefaultValue() const;

 private:
  std::string attr_name_;
  /* Whether this attribute is necessary or not. If is_required_ = true,
   * this attribute is required. On the contrary, it is optional.
   * For optional attribute, if we failed to find this attribute is ops_desc,
   * we will loop it up from default_value_. If is_default_value_defined_ = false,
   * the program will report error when doing operation "CheckSupport".
   */
  bool is_required_;

  ge::GeAttrValue::ValueType dtype_;

  /* is_support_all_value_ tells Whether this attribute supports all kinds of value.
   * If is_support_all_value_ = false, all supported values are stored in vector supported_values_.
   */
  bool is_support_all_value_;

  std::vector<ge::GeAttrValue> supported_values_;

  bool is_default_value_defined_;

  /* If default value is not set, is_default_value_defined_ will be false.
   * In that situation, default_value_ will automatically calls its default constructor and will
   * be constructed as an unknown value. At the same time, it is not appropriate to assign a
   * meaningless default value to default_value_ because the meaningless one will be used int some
   * special cases. So we introduce a flag called "isDefaultValueDefined_" to explicitly elucidate
   * whether default_value_ is set by the user.
   */
  ge::GeAttrValue default_value_;
};
using AttrInfoPtr = std::shared_ptr<AttrInfo>;

class OpKernelInfo {
  /* We separate the definition and initialization in two classes because
   * there exists more than one method to initialize op kernel.
   * Set OpsKernelInfoConstructor as a friend class of OpKernelInfo to
   * conveniently set the private attributes of OpKernelInfo.
   */
  friend class OpKernelInfoConstructor;
  friend class SubOpsStore;

 public:
  explicit OpKernelInfo(const std::string &op_type);
  OpKernelInfo(const std::string &op_type, const OpImplType &impl_type);
  ~OpKernelInfo();

  const std::string& GetOpType() const;
  OpImplType GetOpStoreImplType() const;

  const std::string& GetOpsPathNamePrefix() const;

  /* Now all AttrInfo are stored in a vector, so we offer a find function(loop in vector)
   * for specific attribute with name attr_name to get its Attr Value Type or Attr
   * Value
   */
  Status GetAttrTypeByAttrName(const std::string &attr_name, ge::GeAttrValue::ValueType &dtype) const;
  Status GetAttrValueByAttrName(const std::string &attr_name, std::vector<ge::GeAttrValue> &attr_value) const;

  /* In other cases, we only provide an interface to get vector of whole attr information.
   * Outside callers can do loop in this vector and get information. */
  const std::vector<AttrInfoPtr>& GetVecAttrInfo() const;

  const ge::OpInfo& GetOpInfo() const;

  const std::vector<InputOrOutputInfoPtr>& GetAllInputInfo() const;
  const std::vector<InputOrOutputInfoPtr>& GetAllOutputInfo() const;
  /* The following function is an interface for operations
   * which is the same for input and output. */
  Status GetTensorInfoByName(const bool &isinput, const std::string &tensor_name,
                             InputOrOutputInfoPtr &tensor_info_ptr) const;

  Status GetInputInfoByName(const std::string &input_name, InputOrOutputInfoPtr &input_info_ptr) const;
  Status GetOutputInfoByName(const std::string &output_name, InputOrOutputInfoPtr &output_info_ptr) const;
  void UpdatePrebuildPattern(const std::string &prebuild_pattern);
  void SetPromoteInfo(const PromoteTypeVal &promote_val);
  void GetPromoteInfo(PromoteTypeVal &promote_val) const;
  const FallbackFlags &GetFallbackFlags() const;
  std::vector<std::vector<int64_t>> GetOutputIplaceInfo() const;
  void SetOutputIplaceInfo(std::vector<std::vector<int64_t>> &output_inplace_ability);

  GenerateOpKernelFlagFunc(NeedCheckSupport)
  GenerateOpKernelFlagFunc(HeavyOp)
  GenerateOpKernelFlagFunc(SoftSyncOp)
  GenerateOpKernelFlagFunc(SupportDynamicShape)
  GenerateOpKernelFlagFunc(InputMemContinues)
  GenerateOpKernelFlagFunc(OutputMemContinues)

  GenerateOpKernelParamFunc(OpPattern, OpPattern)
  GenerateOpKernelParamFunc(JitCompile, JitCompileType)
  GenerateOpKernelParamFunc(SlicePattern, OpSlicePattern)
  GenerateOpKernelParamFunc(PrecisionPolicy, PrecisionPolicy)
  GenerateOpKernelParamFunc(RangeLimitType, RangeLimitType)
  GenerateOpKernelParamFunc(DynamicRankType, DynamicRankType)
  GenerateOpKernelParamFunc(DynamicCompileStatic, DynamicCompileStatic)
  GenerateOpKernelParamFunc(AclnnSupportType, AclnnSupportType)
  GenerateOpKernelParamFunc(MultiKernelSupportType, MultiKernelSupportType)
  GenerateOpKernelParamFunc(VectorCoreType, EnableVectorCore)

  GenerateOpKernelStrParamFunc(OpImpPath)
  GenerateOpKernelStrParamFunc(CoreType)
  GenerateOpKernelStrParamFunc(OpImplSwitch)
  GenerateOpKernelStrParamFunc(PrebuildPattern)

  GenerateOpKernelStrParamVecFunc(CoreType)
  GenerateOpKernelStrParamVecFunc(OpImplSwitch)

  bool IsDynamicCompileStatic() const;
  bool IsSupportDynamicRank() const;
  void SetOpImpPath(const std::string &op_imp_path);
  const std::vector<std::string>& GetReferTensorNameVec() const;
  bool IsOpFileNull() const;
  bool IsMultiKernelSupport() const;
 private:
  bool init_flag_;
  std::string op_type_;
  std::string ops_path_name_prefix_;
  OpImplType impl_type_;
  ge::OpInfo op_info_;
  std::string op_impl_path_;
  std::vector<InputOrOutputInfoPtr> input_infos_;
  std::vector<InputOrOutputInfoPtr> output_infos_;
  std::vector<AttrInfoPtr> attrs_info_;
  std::array<bool, static_cast<size_t>(OP_KERNEL_FLAG::OpFlagBottom)> op_flag_vec_;
  std::array<int64_t, static_cast<size_t>(OP_KERNEL_PARAM::OpParamBottom)> op_param_vec_;
  std::array<std::string, static_cast<size_t>(OP_KERNEL_STR_PARAM::OpStrParamBottom)> op_str_param_vec_;
  std::array<std::vector<string>, static_cast<size_t>(OP_KERNEL_STR_PARAM::OpStrParamBottom)> op_str_param_2_vec_;
  std::vector<std::string> refer_tensor_name_vec_;
  PromoteTypeVal promote_type_;
  FallbackFlags fallback_flags_;
  std::vector<std::vector<int64_t>> output_inplace_ability_;
};
using OpKernelInfoPtr = std::shared_ptr<OpKernelInfo>;
}  // namespace fe
#endif  // FUSION_ENGINE_INC_OPS_STORE_OP_KERNEL_INFO_H_
