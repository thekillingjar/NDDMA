/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_COMMON_PRELOAD_NANO_HOSTFUNC_PARAM_H_
#define GE_COMMON_PRELOAD_NANO_HOSTFUNC_PARAM_H_
#include <vector>
#include <sstream>
#include <iomanip>
#include "common/preload/task_info/pre_task_status.h"
#include "framework/common/tlv/nano_aicpu_hostfunc_tlv.h"
#include "common/preload/task_info/aicpu/nano_hostfunc_task_info.h"

namespace ge {
#pragma pack(1)
struct HostFuncTensorDescInfo {
  uint32_t name_len;
  std::string name;
  int32_t dtype;
  int32_t format;
  uint8_t mem_base_type;  // data, weight, workspace
  uint64_t mem_offset;
  uint32_t dims_num;
  vector<int64_t> dims;
  uint32_t shape_range_num;
  vector<int64_t> shape_range;
};

struct ParamHostFuncDesc {
  uint64_t len;
  ExtInfoTlv1 ext_info;
  std::string so_name;
  std::string kernel_name;
  vector<HostFuncTensorDescInfo> input_tensor;
  vector<HostFuncTensorDescInfo> output_tensor;
  map<string, GeAttrValue> all_attrs;
};
#pragma pack()

enum class HOSTFUNC_TENSOR_MEM_BASE_TYPE : uint8_t {
  HOSTFUNC_TENSOR_MEM_BASE_TYPE_DATA,
  HOSTFUNC_TENSOR_MEM_BASE_TYPE_WORKSPACE,
  HOSTFUNC_TENSOR_MEM_BASE_TYPE_WEIGHT
};

class NanoHostfuncParam {
 public:
  NanoHostfuncParam(const domi::TaskDef &task_def, const OpDescPtr &op_desc, const PreTaskInput &pre_task_input);
  ~NanoHostfuncParam() = default;
  Status GenHostFuncParamBufDesc(rtParamBufDesc_t &param_buf_desc);
  std::shared_ptr<uint8_t> Data() const;
  uint32_t DataSize() const;

 private:
  Status ParamBufTlvInit();
  void GenParamBufTlvLen();
  void GenAllAttrsLen();
  void GenTensorLen(const vector<HostFuncTensorDescInfo> &tensor_vec);
  void GenInOutPutCommonDesc(const GeTensorDescPtr &tensor_desc, HostFuncTensorDescInfo &desc_info) const;
  void ConvertOffset(const uint64_t offset, const KernelArgsParam &args_param,
                     HostFuncTensorDescInfo &input_desc) const;
  Status GenInputDesc();
  Status GenOutputDesc();

  // save buffer
  Status UpdateBuffer(const void *src, const size_t count);
  Status GenParamBufTlvData();
  Status SaveExtInfoTlv();
  Status SaveSoNameTlv();
  Status SaveFuncNameTlv();
  Status SaveTensorDescTlv(const bool is_input);
  Status SaveAttrTlv();
  Status SaveAttrValueTlv(const GeAttrValue &attr_value);

  // parse
  void ParseSubBuffer(const uint8_t *buffer, const uint32_t count) const;

  // the flowing function should write [type, value_list_num, value_list_info, value_len, value]
  template <typename T>
  Status SaveAttrValue(const uint32_t type, const GeAttrValue &attr_value);
  Status SaveAttrStringValue(const uint32_t type, const GeAttrValue &attr_value);
  template <typename T>
  Status SaveAttrListValue(const uint32_t type, const GeAttrValue &attr_value);
  Status SaveAttrStringListValue(const uint32_t type, const GeAttrValue &attr_value);
  template <typename T>
  Status SaveAttrListListValue(const uint32_t type, const GeAttrValue &attr_value);

  const OpDescPtr op_desc_;
  const domi::TaskDef &task_def_;
  const PreTaskInput pre_task_input_;
  ParamHostFuncDesc param_desc_;

  // common member
  std::shared_ptr<uint8_t> buff_ = nullptr;
  uint32_t buff_size_ = 0U;
  uint8_t *des_addr_ = nullptr;
  uint32_t des_size_ = 0U;
};
}  // namespace ge
#endif  // GE_COMMON_PRELOAD_NANO_HOSTFUNC_PARAM_H_