/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef H7992249B_058D_40A1_94EA_52BBCB76434E
#define H7992249B_058D_40A1_94EA_52BBCB76434E

#include "fake_ns.h"
#include "common/opskernel/ops_kernel_info_types.h"

FAKE_NS_BEGIN

struct InfoStoreHolder {
  InfoStoreHolder();
  InfoStoreHolder(const std::string&);
  virtual ~InfoStoreHolder() = default;
  void EngineName(std::string engine_name);
  void RegistOp(std::string op_type);
  std::string GetLibName();

 protected:
  std::map<std::string, ge::OpInfo> op_info_map_;
  std::string kernel_lib_name_;
  std::string engine_name_;
};

FAKE_NS_END

#endif
