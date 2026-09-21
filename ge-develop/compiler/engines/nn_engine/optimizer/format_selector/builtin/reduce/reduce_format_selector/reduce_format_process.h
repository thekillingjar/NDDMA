/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_OPTIMIZER_FORMAT_SELECTOR_BUILTIN_REDUCE_REDUCE_FORMAT_SELECTOR_REDUCE_FORMAT_PROCESS_H_
#define FUSION_ENGINE_OPTIMIZER_FORMAT_SELECTOR_BUILTIN_REDUCE_REDUCE_FORMAT_SELECTOR_REDUCE_FORMAT_PROCESS_H_

#include "common/fe_inner_error_codes.h"
#include "common/fe_log.h"
#include "common/format/axis_util.h"
#include "common/fe_op_info_common.h"
#include "graph/types.h"
#include "format_selector/builtin/process/format_process_registry.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_tensor.h"
#include "graph/node.h"
#include "graph/op_desc.h"
#include "graph/utils/type_utils.h"

namespace fe {
class ReduceFormatProcess : public FormatProcessBase {
 public:
  ReduceFormatProcess(){};
  ~ReduceFormatProcess() override {};

  /**
   * Check whether the dimNum of the origin shape is equal to dim_min,
   * and check whether the origin format is identifiable.
   * @param formats origin formats
   * @param shapes origin shapes
   * @param dim refernce value
   * @return
   */
  virtual bool CheckOriginFormatAndShape(const vector<ge::Format> &input_formats,
                                         const vector<ge::Format> &output_formats,
                                         const vector<ge::GeShape> &shapes, const size_t &dim);

 protected:
  bool CheckContainReduceAxis(const ge::OpDesc &op_desc, const vector<ge::Format> &formats,
                              const vector<ge::GeShape> &shapes, const string &axis_name);

  void GenerateFormats(const size_t &in_shape_size, const size_t &out_shape_size,
                       const vector<ge::Format> &support_in_formats, const vector<ge::Format> &support_out_formats,
                       FormatProccessResult &result) const;

  static bool CheckNdFormat(const vector<ge::Format>& formats);

  const string LAST_AXIS_NAME = "last";
  const string LASTBUTONE_AXIS_NAME = "lastButOne";

 private:
  Status GetOriginAxisIndexByName(const ge::Format &format, const ge::GeShape &shape, const std::string &axis_name,
                                  int64_t &axis_index);
};
}  // namespace fe

#endif  // FUSION_ENGINE_OPTIMIZER_FORMAT_SELECTOR_BUILTIN_REDUCE_REDUCE_FORMAT_SELECTOR_REDUCE_FORMAT_PROCESS_H_
