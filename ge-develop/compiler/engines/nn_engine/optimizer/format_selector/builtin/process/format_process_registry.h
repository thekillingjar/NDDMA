/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_OPTIMIZER_FORMAT_SELECTOR_BUILTIN_PROCESS_FORMAT_PROCESS_REGISTRY_H_
#define FUSION_ENGINE_OPTIMIZER_FORMAT_SELECTOR_BUILTIN_PROCESS_FORMAT_PROCESS_REGISTRY_H_

#include <functional>
#include <memory>
#include <vector>
#include "common/fe_inner_error_codes.h"
#include "common/fe_log.h"
#include "common/util/op_info_util.h"
#include "graph/ge_tensor.h"
#include "graph/node.h"
#include "graph/op_desc.h"

namespace fe {
struct OriginInfo {
  vector<ge::Format> input_formats;
  vector<ge::DataType> input_dtypes;
  vector<ge::GeShape> input_shapes;
  vector<ge::Format> output_formats;
  vector<ge::GeShape> output_shapes;
};

using OriginInfoPtr = std::shared_ptr<OriginInfo>;

struct FormatProccessArgs {
  OpPattern op_pattern;
  ge::Format support_format;
  OriginInfoPtr origin_info_ptr;
  ge::Format propagat_primary_format = ge::FORMAT_RESERVED;
  uint32_t propagat_sub_format = 1;
};

struct FormatProccessResult {
  vector<vector<ge::Format>> input_format_res;
  vector<vector<ge::Format>> output_format_res;
  vector<uint32_t> input_subformat_res;
  vector<uint32_t> output_subformat_res;
};

struct FormatProccessInputArg {
  ge::Format input_format_;
  ge::DataType input_dtype_;
  ge::GeShape input_shape_;
  ge::Format propagat_primary_format_ = ge::FORMAT_RESERVED;
  int32_t propagat_sub_format_ = 0;
  FormatProccessInputArg(ge::Format input_format, ge::DataType input_dtype, ge::GeShape input_shape,
                         ge::Format propagat_primary_format, int32_t propagat_sub_format)
      : input_format_(input_format),
        input_dtype_(input_dtype),
        input_shape_(input_shape),
        propagat_primary_format_(propagat_primary_format),
        propagat_sub_format_(propagat_sub_format) {}
};

class FormatProcessBase {
 public:
  FormatProcessBase(){};
  virtual ~FormatProcessBase(){};
  virtual Status Process(const ge::OpDesc &op_desc, const FormatProccessArgs &args, FormatProccessResult &result) = 0;
};

using FormatProcessBasePtr = std::shared_ptr<FormatProcessBase>;
using FormatProcessBuildFunc = std::function<std::shared_ptr<FormatProcessBase>()>;

struct FormatProcessRegistry {
  Status RegisterBuilder(OpPattern op_pattern, ge::Format support_format, FormatProcessBuildFunc builder) {
    FE_LOGD("[GraphOpt][GenBuiltInFmt][GetFmtProcsReg]RegisterBuilder to register the Builder");
    src_dst_builder[op_pattern][support_format] = std::move(builder);
    return SUCCESS;
  }

  std::map<OpPattern, std::map<ge::Format, FormatProcessBuildFunc>> src_dst_builder;
};

FormatProcessRegistry &GetFormatProcessRegistry();

class FormatProcessRegister {
 public:
  FormatProcessRegister(FormatProcessBuildFunc builder, OpPattern op_pattern, ge::Format format);
  ~FormatProcessRegister() = default;
};

FormatProcessBasePtr BuildFormatProcess(const FormatProccessArgs &args);
bool FormatProcessExists(const FormatProccessArgs &args);

#define REGISTER_FORMAT_PROCESS(clazz, op_pattern, format_str, format)                                                 \
  FormatProcessBasePtr create_##op_pattern##format_str##_process(); \
  FormatProcessBasePtr create_##op_pattern##format_str##_process() { return std::make_shared<clazz>(); } \
  FormatProcessRegister f_##op_pattern##format_str##_register(create_##op_pattern##format_str##_process, op_pattern,   \
                                                              format)
}  // namespace fe
#endif  // FUSION_ENGINE_OPTIMIZER_FORMAT_SELECTOR_BUILTIN_PROCESS_FORMAT_PROCESS_REGISTRY_H_
