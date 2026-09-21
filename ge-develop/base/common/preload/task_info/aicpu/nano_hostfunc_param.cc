/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/preload/task_info/aicpu/nano_hostfunc_param.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/graph_utils.h"
#include "common/preload/task_info/aicpu/nano_hostfunc_task_info.h"
#include "common/preload/model/pre_model_utils.h"
#include "common/preload/model/pre_model_types.h"
#include "common/preload/model/pre_model_partition_utils.h"
#include "common/preload/task_info/pre_task_status.h"
#include "graph/debug/ge_attr_define.h"
#include "framework/common/tlv/pre_model_desc.h"
#include "graph_metadef/common/ge_common/util.h"

namespace ge {

namespace {
uint32_t GetAttrListStrLen(const GeAttrValue &attr_value) {
  std::vector<string> vec;
  (void)attr_value.GetValue<std::vector<string>>(vec);
  if (vec.empty()) {
    return 0U;
  }

  uint32_t len = 0U;
  for (const string &vec_str : vec) {
    // get len of AttrParamVal.value_list_info
    len += static_cast<uint32_t>(sizeof(uint32_t));
    // get len of AttrParamVal.value
    len += static_cast<uint32_t>(vec_str.size());
  }

  return len;
}

uint32_t GetAttrListFloatLen(const GeAttrValue &attr_value) {
  std::vector<float> vec;
  (void)attr_value.GetValue<std::vector<float>>(vec);
  // get len of AttrParamVal.value
  return static_cast<uint32_t>(sizeof(float) * vec.size());
}

uint32_t GetAttrListBoolLen(const GeAttrValue &attr_value) {
  std::vector<bool> vec;
  (void)attr_value.GetValue<std::vector<bool>>(vec);
  // get len of AttrParamVal.value
  return static_cast<uint32_t>(sizeof(bool) * vec.size());
}

uint32_t GetAttrListIntLen(const GeAttrValue &attr_value) {
  std::vector<int64_t> vec;
  (void)attr_value.GetValue<std::vector<int64_t>>(vec);
  // get len of AttrParamVal.value
  return static_cast<uint32_t>(sizeof(int64_t) * vec.size());
}

uint32_t GetAttrStrLen(const GeAttrValue &attr_value) {
  string s = "";
  (void)attr_value.GetValue<string>(s);
  // get len of AttrParamVal.value
  return static_cast<uint32_t>(s.size());
}

uint32_t GetAttrFloatLen(const GeAttrValue &attr_value) {
  float32_t f = 0.0F;
  (void)attr_value.GetValue<float>(f);
  // get len of AttrParamVal.value
  return static_cast<uint32_t>(sizeof(float));
}

uint32_t GetAttrBoolLen(const GeAttrValue &attr_value) {
  bool b = false;
  (void)attr_value.GetValue<bool>(b);
  // get len of AttrParamVal.value
  return static_cast<uint32_t>(sizeof(bool));
}

uint32_t GetAttrIntLen(const GeAttrValue &attr_value) {
  int64_t i = 0;
  (void)attr_value.GetValue<int64_t>(i);
  // get len of AttrParamVal.value
  return static_cast<uint32_t>(sizeof(int64_t));
}

uint32_t GetAttrListListIntLen(const GeAttrValue &attr_value) {
  std::vector<std::vector<int64_t>> vec;
  (void)attr_value.GetValue<std::vector<std::vector<int64_t>>>(vec);

  uint32_t len = 0U;
  for (const auto &vec_int : vec) {
    // get len of AttrParamVal.value_list_info
    len += static_cast<uint32_t>(sizeof(uint32_t));
    // get len of AttrParamVal.value
    len += static_cast<uint32_t>(sizeof(int64_t) * vec_int.size());
  }
  // get len of AttrParamVal.value
  return len;
}

uint32_t GetAttrListListFloatLen(const GeAttrValue &attr_value) {
  std::vector<std::vector<float>> vec;
  (void)attr_value.GetValue<std::vector<std::vector<float>>>(vec);

  uint32_t len = 0U;
  for (const auto &vec_int : vec) {
    // get len of AttrParamVal.value_list_info
    len += static_cast<uint32_t>(sizeof(uint32_t));
    // get len of AttrParamVal.value
    len += static_cast<uint32_t>(sizeof(float) * vec_int.size());
  }
  // get len of AttrParamVal.value
  return len;
}

using GetAttrLenFunc = std::function<uint32_t(const GeAttrValue &)>;
static const std::map<ge::GeAttrValue::ValueType, GetAttrLenFunc> attr_len_map = {
    {ge::GeAttrValue::VT_LIST_LIST_FLOAT, &GetAttrListListFloatLen},
    {ge::GeAttrValue::VT_LIST_LIST_INT, &GetAttrListListIntLen},
    {ge::GeAttrValue::VT_LIST_STRING, &GetAttrListStrLen},
    {ge::GeAttrValue::VT_LIST_FLOAT, &GetAttrListFloatLen},
    {ge::GeAttrValue::VT_LIST_BOOL, &GetAttrListBoolLen},
    {ge::GeAttrValue::VT_LIST_INT, &GetAttrListIntLen},
    {ge::GeAttrValue::VT_STRING, &GetAttrStrLen},
    {ge::GeAttrValue::VT_FLOAT, &GetAttrFloatLen},
    {ge::GeAttrValue::VT_BOOL, &GetAttrBoolLen},
    {ge::GeAttrValue::VT_INT, &GetAttrIntLen},
};

static std::atomic<std::uint32_t> g_kernel_id(0U);
}  // namespace

NanoHostfuncParam::NanoHostfuncParam(const domi::TaskDef &task_def, const OpDescPtr &op_desc,
                                     const PreTaskInput &pre_task_input)
    : op_desc_(op_desc), task_def_(task_def), pre_task_input_(pre_task_input) {}

uint32_t NanoHostfuncParam::DataSize() const {
  GELOGD("NanoHostfuncParam buff_size_:%u", buff_size_);
  return buff_size_;
}

std::shared_ptr<uint8_t> NanoHostfuncParam::Data() const {
  GE_ASSERT_NOTNULL(buff_, "buff is nullptr, size[%u]", buff_size_);
  return buff_;
}

Status NanoHostfuncParam::UpdateBuffer(const void *src, const size_t count) {
  if (count == 0U) {
    GELOGD("nothing to memcpy");
    return SUCCESS;
  }
  const auto ret = memcpy_s(des_addr_, static_cast<size_t>(des_size_), src, count);
  GE_ASSERT_EOK(ret, "memcpy_s failed, return: %d", ret);

  des_addr_ = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(des_addr_) + static_cast<uint32_t>(count)));
  des_size_ -= static_cast<uint32_t>(count);
  return SUCCESS;
}

Status NanoHostfuncParam::GenHostFuncParamBufDesc(rtParamBufDesc_t &param_buf_desc) {
  GELOGD("generate hostfunc param buffer desc begin");
  // 1. init tlv data
  GE_ASSERT_TRUE((ParamBufTlvInit() == SUCCESS), "init param buf tlv data failed");

  // 2. get tlv len
  GenParamBufTlvLen();

  // 3. format tlv data
  buff_.reset(new (std::nothrow) uint8_t[buff_size_], std::default_delete<uint8_t[]>());
  GE_CHECK_NOTNULL(buff_);
  GE_ASSERT_TRUE((GenParamBufTlvData() == SUCCESS), "generate param buf tlv data failed");

  param_buf_desc.bufInfo = buff_.get();
  param_buf_desc.bufSize = buff_size_;
  GELOGD("generate hostfunc param buffer desc end");
  return SUCCESS;
}

void NanoHostfuncParam::GenInOutPutCommonDesc(const GeTensorDescPtr &tensor_desc,
                                              HostFuncTensorDescInfo &desc_info) const {
  desc_info.dtype = tensor_desc->GetDataType();
  desc_info.format = GetPrimaryFormat(static_cast<int32_t>(tensor_desc->GetFormat()));
  desc_info.name_len = static_cast<uint32_t>(tensor_desc->GetName().size());
  desc_info.name = tensor_desc->GetName();

  desc_info.dims.clear();
  desc_info.dims = tensor_desc->GetShape().GetDims();
  desc_info.dims_num = static_cast<uint32_t>(desc_info.dims.size());

  // shape_range 预埋 暂时不用
  desc_info.shape_range = {};
  desc_info.shape_range_num = static_cast<uint32_t>(desc_info.shape_range.size());
}

void NanoHostfuncParam::ConvertOffset(const uint64_t offset, const KernelArgsParam &args_param,
                                      HostFuncTensorDescInfo &input_desc) const {
  input_desc.mem_offset = offset;
  input_desc.mem_base_type = static_cast<uint8_t>(HOSTFUNC_TENSOR_MEM_BASE_TYPE::HOSTFUNC_TENSOR_MEM_BASE_TYPE_WEIGHT);
  const uint64_t args_ddr_type = args_param.para;
  const auto zero_copy_offset_to_ids = PreModelPartitionUtils::GetInstance().GetZeroCopyTable();
  if ((args_ddr_type == KERNEL_ARG_UPADTE_ADDR_TYPE_ARGS) || (args_ddr_type == KERNEL_ARG_UPADTE_ADDR_TYPE_WORKSPACE)) {
    input_desc.mem_base_type =
        static_cast<uint8_t>(HOSTFUNC_TENSOR_MEM_BASE_TYPE::HOSTFUNC_TENSOR_MEM_BASE_TYPE_WORKSPACE);
    for (const auto &pair : zero_copy_offset_to_ids) {
      if (pair.second == offset) {
        input_desc.mem_base_type =
            static_cast<uint8_t>(HOSTFUNC_TENSOR_MEM_BASE_TYPE::HOSTFUNC_TENSOR_MEM_BASE_TYPE_DATA);
        GELOGD("offset[%llu] mem_type[%llu] is io offset, label_id[%u].", pair.first, args_ddr_type, pair.second);
        break;
      }
    }
  }
  return;
}

Status NanoHostfuncParam::GenInputDesc() {
  const size_t inputs_size = op_desc_->GetAllInputsSize();
  std::vector<KernelArgsParam> args_param;
  std::vector<uint64_t> args_offset_values;
  const auto input_data_addr_offset =
      PreModelUtils::GetInputDataAddrOffset(pre_task_input_.rts_param, op_desc_, args_param, args_offset_values);
  GE_ASSERT_TRUE(((inputs_size == args_param.size()) && (inputs_size == args_offset_values.size())),
                 "Input size check fail, op:%s", op_desc_->GetName().c_str());
  for (size_t i = 0U; i < inputs_size; ++i) {
    const GeTensorDescPtr tensor_desc = op_desc_->MutableInputDesc(static_cast<uint32_t>(i));
    if (tensor_desc == nullptr) {
      GELOGD("Op: %s, Index: %zu, has no input", op_desc_->GetName().c_str(), i);
      continue;
    }
    HostFuncTensorDescInfo input_temp;
    GenInOutPutCommonDesc(tensor_desc, input_temp);
    ConvertOffset(args_offset_values.at(i), args_param.at(i), input_temp);
    GELOGD("input info memory type[%u], offset[%llu]", static_cast<uint32_t>(input_temp.mem_base_type),
           input_temp.mem_offset);

    param_desc_.input_tensor.push_back(input_temp);
  }
  return SUCCESS;
}

Status NanoHostfuncParam::GenOutputDesc() {
  const size_t outputs_size = op_desc_->GetOutputsSize();
  std::vector<KernelArgsParam> args_param;
  std::vector<uint64_t> args_offset_values;
  const auto input_data_addr_offset =
      PreModelUtils::GetOutputDataAddrOffset(pre_task_input_.rts_param, op_desc_, args_param, args_offset_values);
  GE_ASSERT_TRUE(((outputs_size == args_param.size()) && (outputs_size == args_offset_values.size())),
                 "Output size check fail, op:%s", op_desc_->GetName().c_str());
  for (size_t i = 0U; i < outputs_size; ++i) {
    const GeTensorDescPtr tensor_desc = op_desc_->MutableOutputDesc(static_cast<uint32_t>(i));
    if (tensor_desc == nullptr) {
      GELOGD("Op: %s, Index: %zu, has no output", op_desc_->GetName().c_str(), i);
      continue;
    }
    HostFuncTensorDescInfo output_temp;
    GenInOutPutCommonDesc(tensor_desc, output_temp);
    ConvertOffset(args_offset_values.at(i), args_param.at(i), output_temp);
    GELOGD("output info memory type[%u], offset[%llu]", static_cast<uint32_t>(output_temp.mem_base_type),
           output_temp.mem_offset);

    param_desc_.output_tensor.push_back(output_temp);
  }
  return SUCCESS;
}

Status NanoHostfuncParam::ParamBufTlvInit() {
  GELOGD("tlv init begin");
  const domi::KernelDef &kernel_def = task_def_.kernel();

  // ext_info暂时预留
  param_desc_.ext_info.kernel_id = g_kernel_id++;
  param_desc_.ext_info.session_id = 0U;
  GELOGD("host func ext_info kernel id : %u", param_desc_.ext_info.kernel_id);
  // so_name
  param_desc_.so_name = kernel_def.so_name();
  // kernel function name
  param_desc_.kernel_name = kernel_def.kernel_name();

  // input tensor
  GE_ASSERT_TRUE((GenInputDesc() == SUCCESS), "generate input tensor data failed");

  // output tensor
  GE_ASSERT_TRUE((GenOutputDesc() == SUCCESS), "generate output tensor data failed");

  param_desc_.all_attrs = op_desc_->GetAllAttrs();
  GELOGD("tlv init end, so_name=%s, kernel_name=%s, input_tensor_size=%zu, output_tensor_size=%zu, attr_size=%zu",
         param_desc_.so_name.c_str(), param_desc_.kernel_name.c_str(), param_desc_.input_tensor.size(),
         param_desc_.output_tensor.size(), param_desc_.all_attrs.size());
  return SUCCESS;
}

Status NanoHostfuncParam::SaveExtInfoTlv() {
  GELOGD("save ext info tlv begin");
  const uint8_t *curr_buff_addr = des_addr_;
  TlvHead head;
  head.type = L1_TLV_EXT_INFO;
  head.len = static_cast<uint32_t>(sizeof(ExtInfoTlv1));
  GE_ASSERT_TRUE((UpdateBuffer(static_cast<void *>(&head), sizeof(TlvHead)) == SUCCESS), "UpdateBuffer TlvHead failed");

  GE_ASSERT_TRUE((UpdateBuffer(static_cast<void *>(&(param_desc_.ext_info)), sizeof(ExtInfoTlv1)) == SUCCESS),
                 "UpdateBuffer ExtInfoTlv1 failed");
  ParseSubBuffer(curr_buff_addr, static_cast<uint32_t>(sizeof(TlvHead) + sizeof(ExtInfoTlv1)));
  GELOGD("save ext info tlv end, size : %zu", sizeof(TlvHead) + sizeof(ExtInfoTlv1));
  return SUCCESS;
}

Status NanoHostfuncParam::SaveSoNameTlv() {
  GELOGD("save so name tlv begin");
  const uint8_t *curr_buff_addr = des_addr_;
  TlvHead head;
  head.type = L1_TLV_KERNEL_SO_NAME;
  head.len = static_cast<uint32_t>(param_desc_.so_name.size());
  GE_ASSERT_TRUE((UpdateBuffer(static_cast<void *>(&head), sizeof(TlvHead)) == SUCCESS), "UpdateBuffer TlvHead failed");

  GE_ASSERT_TRUE((UpdateBuffer(param_desc_.so_name.c_str(), param_desc_.so_name.size()) == SUCCESS),
                 "UpdateBuffer so_name failed");
  ParseSubBuffer(curr_buff_addr, static_cast<uint32_t>(sizeof(TlvHead) + param_desc_.so_name.size()));
  GELOGD("save so name tlv end, size : %zu", sizeof(TlvHead) + param_desc_.so_name.size());
  return SUCCESS;
}

Status NanoHostfuncParam::SaveFuncNameTlv() {
  GELOGD("save func name tlv begin");
  const uint8_t *curr_buff_addr = des_addr_;
  TlvHead head;
  head.type = L1_TLV_KERNEL_FUNC_NAME;
  head.len = static_cast<uint32_t>(param_desc_.kernel_name.size());
  GE_ASSERT_TRUE((UpdateBuffer(static_cast<void *>(&head), sizeof(TlvHead)) == SUCCESS), "UpdateBuffer TlvHead failed");

  GE_ASSERT_TRUE((UpdateBuffer(param_desc_.kernel_name.c_str(), param_desc_.kernel_name.size()) == SUCCESS),
                 "UpdateBuffer func_name failed");
  ParseSubBuffer(curr_buff_addr, static_cast<uint32_t>(sizeof(TlvHead) + param_desc_.kernel_name.size()));
  GELOGD("save func name tlv end, size : %zu", sizeof(TlvHead) + param_desc_.kernel_name.size());
  return SUCCESS;
}

Status NanoHostfuncParam::SaveTensorDescTlv(const bool is_input) {
  GELOGD("save tensorDesc tlv begin, is_input=%u", static_cast<uint32_t>(is_input));
  const uint8_t *curr_buff_addr = des_addr_;
  const vector<HostFuncTensorDescInfo> tensor_desc_list =
      (is_input) ? param_desc_.input_tensor : param_desc_.output_tensor;
  TlvHead *tlv_head_ptr = PtrToPtr<uint8_t, TlvHead>(des_addr_);  // for update tlv len

  TlvHead head;
  head.type = (is_input) ? static_cast<uint32_t>(L1_TLV_TYPE_INPUT_TENSOR_DESC)
                         : static_cast<uint32_t>(L1_TLV_TYPE_OUTPUT_TENSOR_DESC);
  head.len = 0U;  // will be update later
  GE_ASSERT_TRUE((UpdateBuffer(static_cast<void *>(&head), sizeof(TlvHead)) == SUCCESS), "UpdateBuffer TlvHead failed");
  const uint32_t data_pos = des_size_;  // for update tlv len, since this pos, the behind is TLV value

  TensorDescTlv1 tensor_desc_tlv1;
  tensor_desc_tlv1.tensor_num = static_cast<uint32_t>(tensor_desc_list.size());
  GE_ASSERT_TRUE((UpdateBuffer(static_cast<void *>(&tensor_desc_tlv1), sizeof(TensorDescTlv1)) == SUCCESS),
                 "UpdateBuffer TensorDescTlv1 failed");

  GELOGD("tensor_num=%u", tensor_desc_tlv1.tensor_num);
  for (const auto &tensor_desc : tensor_desc_list) {
    GE_ASSERT_TRUE(
        (UpdateBuffer(PtrToPtr<const uint32_t, const void>(&(tensor_desc.name_len)), sizeof(uint32_t)) == SUCCESS),
        "UpdateBuffer tensor name_len failed");
    GE_ASSERT_TRUE((UpdateBuffer(tensor_desc.name.c_str(), static_cast<size_t>(tensor_desc.name_len)) == SUCCESS),
                   "UpdateBuffer tensor name failed");
    GE_ASSERT_TRUE(
        (UpdateBuffer(PtrToPtr<const int32_t, const void>(&(tensor_desc.dtype)), sizeof(int32_t)) == SUCCESS),
        "UpdateBuffer tensor dtype failed");
    GE_ASSERT_TRUE(
        (UpdateBuffer(PtrToPtr<const int32_t, const void>(&(tensor_desc.format)), sizeof(int32_t)) == SUCCESS),
        "UpdateBuffer tensor format failed");
    GE_ASSERT_TRUE(
        (UpdateBuffer(PtrToPtr<const uint8_t, const void>(&(tensor_desc.mem_base_type)), sizeof(uint8_t)) == SUCCESS),
        "UpdateBuffer tensor mem_base_type failed");
    GE_ASSERT_TRUE(
        (UpdateBuffer(PtrToPtr<const uint64_t, const void>(&(tensor_desc.mem_offset)), sizeof(uint64_t)) == SUCCESS),
        "UpdateBuffer tensor mem_offset failed");
    GE_ASSERT_TRUE(
        (UpdateBuffer(PtrToPtr<const uint32_t, const void>(&(tensor_desc.dims_num)), sizeof(uint32_t)) == SUCCESS),
        "UpdateBuffer tensor dims_num failed");
    for (const int64_t &dim : tensor_desc.dims) {
      GE_ASSERT_TRUE((UpdateBuffer(PtrToPtr<const int64_t, const void>(&dim), sizeof(int64_t)) == SUCCESS),
                     "UpdateBuffer tensor dim[%lld] failed", dim);
    }
    GE_ASSERT_TRUE((UpdateBuffer(PtrToPtr<const uint32_t, const void>(&(tensor_desc.shape_range_num)),
                                 sizeof(uint32_t)) == SUCCESS),
                   "UpdateBuffer tensor shape_range_len failed");
    for (const int64_t &shape_range : tensor_desc.shape_range) {
      GE_ASSERT_TRUE((UpdateBuffer(PtrToPtr<const int64_t, const void>(&shape_range), sizeof(int64_t)) == SUCCESS),
                     "UpdateBuffer tensor shape_range[%lld] failed", shape_range);
    }
  }
  tlv_head_ptr->len = data_pos - des_size_;
  ParseSubBuffer(curr_buff_addr, static_cast<uint32_t>(sizeof(TlvHead) + tlv_head_ptr->len));
  GELOGD("save tensorDesc tlv end, size : %zu", sizeof(TlvHead) + tlv_head_ptr->len);
  return SUCCESS;
}

template <typename T>
Status NanoHostfuncParam::SaveAttrValue(const uint32_t type, const GeAttrValue &attr_value) {
  T val;
  (void)attr_value.GetValue<T>(val);
  const uint32_t value_list_num = 0U;
  const uint32_t value_len = static_cast<uint32_t>(sizeof(T));

  GE_ASSERT_TRUE((UpdateBuffer(PtrToPtr<const uint32_t, const void>(&type), sizeof(uint32_t)) == SUCCESS),
                 "UpdateBuffer type failed");
  GE_ASSERT_TRUE((UpdateBuffer(PtrToPtr<const uint32_t, const void>(&value_list_num), sizeof(uint32_t)) == SUCCESS),
                 "UpdateBuffer value_list_num failed");
  GE_ASSERT_TRUE((UpdateBuffer(PtrToPtr<const uint32_t, const void>(&value_len), sizeof(uint32_t)) == SUCCESS),
                 "UpdateBuffer value_len failed");
  GE_ASSERT_TRUE((UpdateBuffer(static_cast<void *>(&val), static_cast<size_t>(value_len)) == SUCCESS),
                 "UpdateBuffer value failed");
  return SUCCESS;
}

Status NanoHostfuncParam::SaveAttrStringValue(const uint32_t type, const GeAttrValue &attr_value) {
  string val;
  (void)attr_value.GetValue<string>(val);
  const uint32_t value_list_num = 0U;
  const uint32_t value_len = static_cast<uint32_t>(val.size());

  GE_ASSERT_TRUE((UpdateBuffer(PtrToPtr<const uint32_t, const void>(&type), sizeof(uint32_t)) == SUCCESS),
                 "UpdateBuffer type failed");
  GE_ASSERT_TRUE((UpdateBuffer(PtrToPtr<const uint32_t, const void>(&value_list_num), sizeof(uint32_t)) == SUCCESS),
                 "UpdateBuffer value_list_num failed");
  GE_ASSERT_TRUE((UpdateBuffer(PtrToPtr<const uint32_t, const void>(&value_len), sizeof(uint32_t)) == SUCCESS),
                 "UpdateBuffer value_len failed");
  GE_ASSERT_TRUE((UpdateBuffer(val.c_str(), static_cast<size_t>(value_len)) == SUCCESS), "UpdateBuffer value failed");
  return SUCCESS;
}

template <typename T>
Status NanoHostfuncParam::SaveAttrListValue(const uint32_t type, const GeAttrValue &attr_value) {
  std::vector<T> values;
  (void)attr_value.GetValue<std::vector<T>>(values);
  const uint32_t value_list_num = 0U;
  const uint32_t value_len = static_cast<uint32_t>(sizeof(T) * values.size());

  GE_ASSERT_TRUE((UpdateBuffer(PtrToPtr<const uint32_t, const void>(&type), sizeof(uint32_t)) == SUCCESS),
                 "UpdateBuffer type failed");
  GE_ASSERT_TRUE((UpdateBuffer(PtrToPtr<const uint32_t, const void>(&value_list_num), sizeof(uint32_t)) == SUCCESS),
                 "UpdateBuffer value_list_num failed");
  GE_ASSERT_TRUE((UpdateBuffer(PtrToPtr<const uint32_t, const void>(&value_len), sizeof(uint32_t)) == SUCCESS),
                 "UpdateBuffer value_len failed");

  for (const T &value : values) {  // vector<bool> is a special vector, can not be memcpy
    GE_ASSERT_TRUE((UpdateBuffer(PtrToPtr<const T, const void>(&value), sizeof(T)) == SUCCESS),
                   "UpdateBuffer value failed");
  }
  return SUCCESS;
}

Status NanoHostfuncParam::SaveAttrStringListValue(const uint32_t type, const GeAttrValue &attr_value) {
  std::vector<std::string> values;
  (void)attr_value.GetValue<std::vector<std::string>>(values);
  size_t value_len = 0U;
  vector<uint32_t> value_list_info;
  for (const auto &val : values) {
    value_len += val.size();
    value_list_info.push_back(static_cast<uint32_t>(val.size()));
  }
  const uint32_t value_list_num = static_cast<uint32_t>(values.size());

  GE_ASSERT_TRUE((UpdateBuffer(PtrToPtr<const uint32_t, const void>(&type), sizeof(uint32_t)) == SUCCESS),
                 "UpdateBuffer type failed");
  GE_ASSERT_TRUE((UpdateBuffer(PtrToPtr<const uint32_t, const void>(&value_list_num), sizeof(uint32_t)) == SUCCESS),
                 "UpdateBuffer value_list_num failed");
  GE_ASSERT_TRUE(
      (UpdateBuffer(static_cast<void *>(&(value_list_info[0])), value_list_num * sizeof(uint32_t)) == SUCCESS),
      "UpdateBuffer value_list_info failed");
  GE_ASSERT_TRUE((UpdateBuffer(static_cast<void *>(&value_len), sizeof(uint32_t)) == SUCCESS),
                 "UpdateBuffer value_len failed");
  GE_ASSERT_TRUE((UpdateBuffer(static_cast<void *>(&(values[0])), value_len) == SUCCESS), "UpdateBuffer value failed");
  return SUCCESS;
}

template <typename T>
Status NanoHostfuncParam::SaveAttrListListValue(const uint32_t type, const GeAttrValue &attr_value) {
  std::vector<std::vector<T>> values;
  (void)attr_value.GetValue<std::vector<std::vector<T>>>(values);
  size_t value_len = 0U;
  vector<uint32_t> value_list_info;
  for (const auto &val : values) {
    value_len += (val.size() * sizeof(T));
    value_list_info.push_back(static_cast<uint32_t>(val.size()));
  }
  const uint32_t value_list_num = static_cast<uint32_t>(values.size());

  GE_ASSERT_TRUE((UpdateBuffer(PtrToPtr<const uint32_t, const void>(&type), sizeof(uint32_t)) == SUCCESS),
                 "UpdateBuffer type failed");
  GE_ASSERT_TRUE((UpdateBuffer(PtrToPtr<const uint32_t, const void>(&value_list_num), sizeof(uint32_t)) == SUCCESS),
                 "UpdateBuffer value_list_num failed");
  GE_ASSERT_TRUE(
      (UpdateBuffer(static_cast<void *>(&(value_list_info[0])), value_list_num * sizeof(uint32_t)) == SUCCESS),
      "UpdateBuffer value_list_info failed");
  GE_ASSERT_TRUE((UpdateBuffer(static_cast<void *>(&value_len), sizeof(uint32_t)) == SUCCESS),
                 "UpdateBuffer value_len failed");
  GE_ASSERT_TRUE((UpdateBuffer(static_cast<void *>(&(values[0])), value_len) == SUCCESS), "UpdateBuffer value failed");
  return SUCCESS;
}

Status NanoHostfuncParam::SaveAttrValueTlv(const GeAttrValue &attr_value) {
  Status ret = FAILED;
  switch (attr_value.GetValueType()) {
    case ge::GeAttrValue::VT_INT:
      ret = SaveAttrValue<int64_t>(L2_TLV_TYPE_ATTR_INT, attr_value);
      break;
    case ge::GeAttrValue::VT_FLOAT:
      ret = SaveAttrValue<float32_t>(L2_TLV_TYPE_ATTR_FLOAT, attr_value);
      break;
    case ge::GeAttrValue::VT_BOOL:
      ret = SaveAttrValue<bool>(L2_TLV_TYPE_ATTR_BOOL, attr_value);
      break;
    case ge::GeAttrValue::VT_STRING:
      ret = SaveAttrStringValue(L2_TLV_TYPE_ATTR_STRING, attr_value);
      break;
    case ge::GeAttrValue::VT_LIST_INT:
      ret = SaveAttrListValue<int64_t>(L2_TLV_TYPE_ATTR_LIST_INT, attr_value);
      break;
    case ge::GeAttrValue::VT_LIST_FLOAT:
      ret = SaveAttrListValue<float32_t>(L2_TLV_TYPE_ATTR_LIST_FLOAT, attr_value);
      break;
    case ge::GeAttrValue::VT_LIST_BOOL:
      ret = SaveAttrListValue<bool>(L2_TLV_TYPE_ATTR_LIST_BOOL, attr_value);
      break;
    case ge::GeAttrValue::VT_LIST_STRING:
      ret = SaveAttrStringListValue(L2_TLV_TYPE_ATTR_LIST_STRING, attr_value);
      break;
    case ge::GeAttrValue::VT_LIST_LIST_INT:
      ret = SaveAttrListListValue<int64_t>(L2_TLV_TYPE_ATTR_LIST_LIST_INT, attr_value);
      break;
    case ge::GeAttrValue::VT_LIST_LIST_FLOAT:
      ret = SaveAttrListListValue<float32_t>(L2_TLV_TYPE_ATTR_LIST_LIST_FLOAT, attr_value);
      break;
    default:
      GELOGD("unsupported type %d", attr_value.GetValueType());
      break;
  }
  return ret;
}

Status NanoHostfuncParam::SaveAttrTlv() {
  GELOGD("save attr tlv begin");
  const uint8_t *curr_buff_addr = des_addr_;
  TlvHead *tlv_head_ptr = PtrToPtr<uint8_t, TlvHead>(des_addr_);  // for update tlv len

  TlvHead head;
  head.type = L1_TLV_ATTR;
  head.len = 0U;
  GE_ASSERT_TRUE((UpdateBuffer(static_cast<void *>(&head), sizeof(TlvHead)) == SUCCESS), "UpdateBuffer TlvHead failed");
  const uint32_t data_pos = des_size_;  // for update tlv len, since this pos, the behind is TLV value

  AttrDescTlv1 *attr_tlv_ptr = PtrToPtr<uint8_t, AttrDescTlv1>(des_addr_);
  AttrDescTlv1 attr_desc_tlv1;
  attr_desc_tlv1.num = static_cast<uint32_t>(param_desc_.all_attrs.size());
  GE_ASSERT_TRUE((UpdateBuffer(static_cast<void *>(&attr_desc_tlv1), sizeof(TensorDescTlv1)) == SUCCESS),
                 "UpdateBuffer TensorDescTlv1 failed");

  uint32_t attrs_num = 0;
  for (const auto &attr : param_desc_.all_attrs) {
    const uint32_t name_len = static_cast<uint32_t>(attr.first.size());
    const ge::GeAttrValue::ValueType v_t = attr.second.GetValueType();
    const auto iter = attr_len_map.find(v_t);
    if (iter == attr_len_map.end()) {
      GELOGD("Skip attr type: %d", v_t);
      continue;
    }
    GE_ASSERT_TRUE((UpdateBuffer(PtrToPtr<const uint32_t, const void>(&name_len), sizeof(uint32_t)) == SUCCESS),
                   "UpdateBuffer name_len failed");
    GE_ASSERT_TRUE((UpdateBuffer(attr.first.c_str(), static_cast<size_t>(name_len)) == SUCCESS),
                   "UpdateBuffer name failed");
    GE_ASSERT_TRUE((SaveAttrValueTlv(attr.second) == SUCCESS), "UpdateBuffer attr value failed");
    attrs_num++;
  }

  tlv_head_ptr->len = data_pos - des_size_;
  attr_tlv_ptr->num = attrs_num;
  ParseSubBuffer(curr_buff_addr, static_cast<uint32_t>(sizeof(TlvHead) + tlv_head_ptr->len));
  GELOGD("save attr tlv end, size : %zu, attr num : %u", sizeof(TlvHead) + tlv_head_ptr->len, attrs_num);
  return SUCCESS;
}

Status NanoHostfuncParam::GenParamBufTlvData() {
  des_addr_ = buff_.get();
  des_size_ = buff_size_;

  uint64_t tlv_len = 0U;
  const uint32_t data_pos = des_size_;                             // for update tlv len
  uint64_t *tlv_len_ptr = PtrToPtr<uint8_t, uint64_t>(des_addr_);  // for update tlv len
  GE_ASSERT_TRUE((UpdateBuffer(static_cast<void *>(&tlv_len), sizeof(HostFuncHead)) == SUCCESS),
                 "UpdateBuffer tlv_len failed");
  GE_ASSERT_TRUE((SaveSoNameTlv() == SUCCESS), "SaveSoNameTlv failed");
  GE_ASSERT_TRUE((SaveFuncNameTlv() == SUCCESS), "SaveFuncNameTlv failed");
  GE_ASSERT_TRUE((SaveExtInfoTlv() == SUCCESS), "SaveExtInfoTlv failed");
  GE_ASSERT_TRUE((SaveTensorDescTlv(true) == SUCCESS), "SaveTensorDescTlv input tensor failed");
  GE_ASSERT_TRUE((SaveTensorDescTlv(false) == SUCCESS), "SaveTensorDescTlv output tensor failed");
  GE_ASSERT_TRUE((SaveAttrTlv() == SUCCESS), "SaveExtInfoTlv failed");
  *tlv_len_ptr = static_cast<uint64_t>(data_pos - des_size_);
  return SUCCESS;
}

void NanoHostfuncParam::GenParamBufTlvLen() {
  GELOGD("generate hostfunc param buffer tlv len begin");
  buff_size_ += static_cast<uint32_t>(sizeof(HostFuncHead));

  buff_size_ += static_cast<uint32_t>(sizeof(TlvHead));
  buff_size_ += static_cast<uint32_t>(sizeof(ExtInfoTlv1));

  buff_size_ += static_cast<uint32_t>(sizeof(TlvHead));
  buff_size_ += static_cast<uint32_t>(param_desc_.so_name.size());

  buff_size_ += static_cast<uint32_t>(sizeof(TlvHead));
  buff_size_ += static_cast<uint32_t>(param_desc_.kernel_name.size());

  buff_size_ += static_cast<uint32_t>(sizeof(TlvHead));
  GenTensorLen(param_desc_.input_tensor);

  buff_size_ += static_cast<uint32_t>(sizeof(TlvHead));
  GenTensorLen(param_desc_.output_tensor);

  buff_size_ += static_cast<uint32_t>(sizeof(TlvHead));
  GenAllAttrsLen();
  GELOGD("generate hostfunc param buffer tlv len : %u", buff_size_);
}

void NanoHostfuncParam::GenTensorLen(const vector<HostFuncTensorDescInfo> &tensor_vec) {
  buff_size_ += static_cast<uint32_t>(sizeof(TensorDescTlv1));
  for (const auto &tensor : tensor_vec) {
    buff_size_ += static_cast<uint32_t>(sizeof(TensorDescParamTlv1));
    buff_size_ += tensor.name_len;
    buff_size_ += static_cast<uint32_t>(tensor.dims_num * sizeof(int64_t));
    buff_size_ += static_cast<uint32_t>(tensor.shape_range_num * sizeof(int64_t));
  }
}

void NanoHostfuncParam::GenAllAttrsLen() {
  const uint32_t tmp_size = buff_size_;
  buff_size_ += static_cast<uint32_t>(sizeof(AttrDescTlv1));
  for (const auto &attr : param_desc_.all_attrs) {
    const ge::GeAttrValue::ValueType v_t = attr.second.GetValueType();
    const auto iter = attr_len_map.find(v_t);
    if (iter == attr_len_map.end()) {
      GELOGD("Skip attr type: %d", v_t);
      continue;
    }
    buff_size_ += static_cast<uint32_t>(sizeof(AttrParamVal));
    // get len of AttrParamVal.name
    buff_size_ += static_cast<uint32_t>(attr.first.size());
    buff_size_ += iter->second(attr.second);
  }
  GELOGD("all attes len = %u", buff_size_ - tmp_size);
}

void NanoHostfuncParam::ParseSubBuffer(const uint8_t *buffer, const uint32_t count) const {
  std::stringstream ss;
  constexpr int32_t print_num = 2;
  for (uint32_t i = 0U; i < count; i++) {
    ss << std::hex << std::setw(print_num) << std::setfill('0') << static_cast<int32_t>(buffer[i]) << " ";
  }
  GELOGD("sub buffer [%s]", ss.str().c_str());
}
}  // namespace ge