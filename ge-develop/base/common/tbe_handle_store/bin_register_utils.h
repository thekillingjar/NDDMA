/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_HYBRID_KERNEL_AICORE_BIN_REGISTER_UTILS_H_
#define GE_HYBRID_KERNEL_AICORE_BIN_REGISTER_UTILS_H_

#include "graph/op_desc.h"
#include "ge/ge_api_types.h"

namespace ge {
struct AttrNameOfBinOnOp {
  std::string kTbeKernel;
  std::string kTvmMetaData;
  std::string kKernelNameSuffix;
  std::string kTvmMagicName;
};

class BinRegisterUtils {
  public:
  /**
   * @brief Call rtDevBinaryRegister to register bin
   * @param op_desc
   * @param stub_name
   *        Name of stub func. For op in SingleOp model, stub name is log_id + stub_func(defined in kernel_def).
   *        For op in hybrid model, stub name is stub_func(defined in kernel_def).
   * @param stub_func
   * @return Status
   */
   static Status RegisterBin(const OpDesc &op_desc, const std::string &stub_name, const AttrNameOfBinOnOp &attr_names,
                             void *&stub_func);

   /**
    * @brief Call rtRegisterAllKernel to register bin
    * @param op_desc
    *        input param
    * @param handle
    *        outpu param. After register bin, handle will be update.
    * @return Status
    */
   static Status RegisterBinWithHandle(const OpDesc &op_desc, const AttrNameOfBinOnOp &attr_names, void *&handle);
};
}  // namespace ge
#endif  // GE_HYBRID_KERNEL_AICORE_BIN_REGISTER_UTIL_H_
