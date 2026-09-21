/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/fe_log.h"
#include "common/resource_def.h"

namespace fe {
struct funcStateInfo {
  bool isConfiged;
  bool isOpen;
};
using funcState = struct funcStateInfo;

class FuncSetting {
 public:
  static FuncSetting *Instance();
  bool SetFuncState(FuncParamType type, bool isOpen);
  bool GetFuncState(FuncParamType type);

 protected:
  FuncSetting()
  {
    for (int i = 0; i < static_cast<int>(FuncParamType::MAX_NUM); i++) {
      state[i].isConfiged = false;
      state[i].isOpen = false;
    }
  }
  ~FuncSetting() {}

 private:
  funcState state[static_cast<size_t>(FuncParamType::MAX_NUM)];
};

FuncSetting *FuncSetting::Instance()
{
  static FuncSetting instance;
  return &instance;
}

bool FuncSetting::SetFuncState(FuncParamType type, bool isOpen)
{
  if (type < FuncParamType::MAX_NUM) {
    state[static_cast<size_t>(type)].isConfiged = true;
    state[static_cast<size_t>(type)].isOpen = isOpen;
    return true;
  } else {
    FE_LOGE("SetFuncState type[%d] is invalid!", static_cast<int32_t>(type));
    return false;
  }
}

bool FuncSetting::GetFuncState(FuncParamType type)
{
  if (type < FuncParamType::MAX_NUM) {
    if (state[static_cast<size_t>(type)].isConfiged)
    {
      return state[static_cast<size_t>(type)].isOpen;
    }
  }
  return false;
}

bool SetFunctionState(FuncParamType type, bool isOpen)
{
  return FuncSetting::Instance()->SetFuncState(type, isOpen);
}

bool GetFunctionState(FuncParamType type)
{
  return FuncSetting::Instance()->GetFuncState(type);
}

}
