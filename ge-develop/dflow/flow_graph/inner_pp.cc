/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "data_flow/flow_graph/inner_pp.h"
#include "common/util/mem_utils.h"
#include "framework/common/debug/ge_log.h"
#include "data_flow_attr_define.h"
#include "proto/dflow.pb.h"
#include "base/err_msg.h"

namespace ge {
namespace dflow {
class InnerPpImpl {
 public:
  explicit InnerPpImpl(const std::string &inner_type) : inner_type_(inner_type) {}
  const std::string &GetInnerType() {
    return inner_type_;
  }

 private:
  std::string inner_type_;
};

InnerPp::InnerPp(const char_t *pp_name, const char_t *inner_type) : ProcessPoint(pp_name, ProcessPointType::INNER) {
  if (inner_type == nullptr) {
    impl_ = nullptr;
    GELOGE(GRAPH_FAILED, "inner type is nullptr.");
    return;
  }
  impl_ = MakeShared<InnerPpImpl>(inner_type);
  if (impl_ == nullptr) {
    GELOGE(GRAPH_FAILED, "InnerPpImpl make shared failed.");
  }
}

void InnerPp::Serialize(ge::AscendString &str) const {
  if (impl_ == nullptr) {
    GELOGE(GRAPH_PARAM_INVALID, "[Check][Param] InnerPpImpl is nullptr, check failed");
    REPORT_INNER_ERR_MSG("E18888", "Serialize failed: InnerPp can not be used, impl is nullptr.");
    return;
  }
  dataflow::ProcessPoint process_point;
  process_point.set_name(this->GetProcessPointName());
  process_point.set_type(dataflow::ProcessPoint_ProcessPointType_INNER);
  process_point.set_compile_cfg_file(this->GetCompileConfig());
  std::map<ge::AscendString, ge::AscendString> serialize_map;
  InnerSerialize(serialize_map);
  auto *const pp_extend_attrs = process_point.mutable_pp_extend_attrs();
  for (const auto &serialize_iter : serialize_map) {
    (*pp_extend_attrs)[std::string(serialize_iter.first.GetString(), serialize_iter.first.GetLength())] =
        std::string(serialize_iter.second.GetString(), serialize_iter.second.GetLength());
  }
  (*pp_extend_attrs)[std::string(INNER_PP_CUSTOM_ATTR_INNER_TYPE)] = impl_->GetInnerType();
  std::string target_str;
  process_point.SerializeToString(&target_str);
  str = ge::AscendString(target_str.c_str(), target_str.length());
}

}  // namespace dflow
}  // namespace ge
