/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_PNE_DATA_FLOW_INNER_PP_H_
#define INC_PNE_DATA_FLOW_INNER_PP_H_

#include <map>
#include "ge/ge_api_error_codes.h"
#include "graph/ascend_string.h"
#include "flow_graph/process_point.h"

namespace ge {
namespace dflow {
class InnerPpImpl;
class GE_FUNC_VISIBILITY InnerPp : public ProcessPoint {
 public:
  InnerPp(const char_t *pp_name, const char_t *inner_type);
  ~InnerPp() override = default;
  void Serialize(ge::AscendString &str) const override;
 protected:
  virtual void InnerSerialize(std::map<ge::AscendString, ge::AscendString> &serialize_map) const = 0;

 private:
  std::shared_ptr<InnerPpImpl> impl_;
};
}  // namespace dflow
}  // namespace ge
#endif  // INC_PNE_DATA_FLOW_INNER_PP_H_
