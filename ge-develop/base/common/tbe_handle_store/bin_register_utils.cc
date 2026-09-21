/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/tbe_handle_store/bin_register_utils.h"

#include "runtime/rt.h"
#include "common/plugin/ge_make_unique_util.h"
#include "common/util.h"
#include "common/tbe_handle_store/kernel_store.h"
#include "framework/common/debug/log.h"
#include "common/tbe_handle_store/tbe_handle_store.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "base/err_msg.h"

namespace ge {
namespace {
ge::Status GetDevBinFromOpDesc(const OpDesc &op_desc, const TBEKernelPtr &tbe_kernel, rtDevBinary_t &binary,
                               const std::string &kTvmMagicName) {
  const std::string *json_string = AttrUtils::GetStr(op_desc, kTvmMagicName);
  GE_ASSERT_NOTNULL(json_string, "Get json_string of tvm_magic from op_desc is null.");
  if (*json_string == "RT_DEV_BINARY_MAGIC_ELF_AICPU") {
    binary.magic = RT_DEV_BINARY_MAGIC_ELF_AICPU;
  } else if (*json_string == "RT_DEV_BINARY_MAGIC_ELF") {
    binary.magic = RT_DEV_BINARY_MAGIC_ELF;
  } else if (*json_string == "RT_DEV_BINARY_MAGIC_ELF_AIVEC") {
    binary.magic = RT_DEV_BINARY_MAGIC_ELF_AIVEC;
  } else if (*json_string == "RT_DEV_BINARY_MAGIC_ELF_AICUBE") {
    binary.magic = RT_DEV_BINARY_MAGIC_ELF_AICUBE;
  } else {
    GELOGE(PARAM_INVALID, "[Check][JsonStr]Attr:%s in op:%s(%s), value:%s check invalid", TVM_ATTR_NAME_MAGIC.c_str(),
           op_desc.GetName().c_str(), op_desc.GetType().c_str(), json_string->c_str());
    REPORT_INNER_ERR_MSG("E19999", "Attr:%s in op:%s(%s), value:%s check invalid", TVM_ATTR_NAME_MAGIC.c_str(),
                       op_desc.GetName().c_str(), op_desc.GetType().c_str(), json_string->c_str());
    return PARAM_INVALID;
  }
  binary.version = 0U;
  binary.data = tbe_kernel->GetBinData();
  binary.length = tbe_kernel->GetBinDataSize();
  GELOGI("TBE: binary.length: %lu", binary.length);
  return SUCCESS;
}
} // namespace

Status BinRegisterUtils::RegisterBin(const OpDesc &op_desc, const std::string &stub_name,
                                     const AttrNameOfBinOnOp &attr_names, void *&stub_func) {
  const rtError_t rt_ret = rtQueryFunctionRegistered(stub_name.c_str());
  if (rt_ret != RT_ERROR_NONE) {
    const auto op_desc_ptr = MakeShared<OpDesc>(op_desc);
    GE_CHECK_NOTNULL(op_desc_ptr);
    
    TBEHandleStore &kernel_store = TBEHandleStore::GetInstance();
    void *bin_handle = nullptr;
    if (!kernel_store.FindTBEHandle(stub_name.c_str(), bin_handle)) {
      GELOGI("TBE: can't find the binfile_key[%s] in HandleMap", stub_name.c_str());
      const auto tbe_kernel = op_desc_ptr->TryGetExtAttr(attr_names.kTbeKernel, TBEKernelPtr());
      if (tbe_kernel == nullptr) {
        REPORT_INNER_ERR_MSG("E19999", "%s(%s) can't find tvm bin file!", op_desc_ptr->GetName().c_str(),
                           op_desc_ptr->GetType().c_str());
        GELOGE(INTERNAL_ERROR, "[TryGet][ExtAttr]TBE: %s(%s) can't find tvm bin file!", op_desc_ptr->GetName().c_str(),
               op_desc_ptr->GetType().c_str());
        return INTERNAL_ERROR;
      }
      rtDevBinary_t binary;
      GE_CHK_STATUS_RET_NOLOG(GetDevBinFromOpDesc(op_desc, tbe_kernel, binary, attr_names.kTvmMagicName));
      GE_CHK_RT_RET(rtDevBinaryRegister(&binary, &bin_handle));
      std::string tvm_meta_data;
      GE_IF_BOOL_EXEC(AttrUtils::GetStr(op_desc_ptr, attr_names.kTvmMetaData, tvm_meta_data),
                      GELOGI("Get original type of json_string"));
      GELOGI("TBE: meta data: [%s]", tvm_meta_data.c_str());
      GE_IF_BOOL_EXEC(!tvm_meta_data.empty(), GE_CHK_RT_RET(rtMetadataRegister(bin_handle, tvm_meta_data.c_str())));
      GE_ASSERT_SUCCESS(kernel_store.StoreTBEHandle(stub_name.c_str(), bin_handle, tbe_kernel));
    } else {
      GELOGI("TBE: find the binfile_key[%s] in HandleMap", stub_name.c_str());
      kernel_store.ReferTBEHandle(stub_name.c_str());
    }
    std::string kernel_name;
    const std::string key_for_kernel_name = op_desc.GetName() + attr_names.kKernelNameSuffix;
    GE_IF_BOOL_EXEC(AttrUtils::GetStr(op_desc_ptr, key_for_kernel_name, attr_names.kKernelNameSuffix, kernel_name),
                    GELOGI("Get original type of kernel_name"));
    GELOGI("TBE: binfile_key=%s, kernel_name=%s", stub_name.c_str(), kernel_name.c_str());
    GE_CHK_RT_RET(rtFunctionRegister(bin_handle, stub_name.c_str(), stub_name.c_str(), kernel_name.c_str(), 0U));
  }
  (void)KernelBinRegistry::GetInstance().GetUnique(stub_name);
  GE_CHK_RT_RET(rtGetFunctionByName(stub_name.c_str(), &stub_func));
  return SUCCESS;
}

Status BinRegisterUtils::RegisterBinWithHandle(const OpDesc &op_desc, const AttrNameOfBinOnOp &attr_names,
                                               void *&handle) {
  const auto tbe_kernel = op_desc.TryGetExtAttr(attr_names.kTbeKernel, TBEKernelPtr());
  if (tbe_kernel == nullptr) {
    GELOGE(INTERNAL_ERROR, "[Invoke][TryGetExtAttr]TBE: %s(%s) can't find tvm bin file!", op_desc.GetName().c_str(),
           op_desc.GetType().c_str());
    REPORT_INNER_ERR_MSG("E19999", "TBE: %s(%s) can't find tvm bin file.", op_desc.GetName().c_str(),
                      op_desc.GetType().c_str());
    return INTERNAL_ERROR;
  }

  void *bin_handle = nullptr;
  GELOGD("Start to register kernel for node: [%s].", op_desc.GetName().c_str());
  rtDevBinary_t binary;
  GE_CHK_STATUS_RET_NOLOG(GetDevBinFromOpDesc(op_desc, tbe_kernel, binary, attr_names.kTvmMagicName));
  GE_CHK_RT_RET(rtRegisterAllKernel(&binary, &bin_handle));
  handle = bin_handle;
  return SUCCESS;
}
}  // namespace ge
