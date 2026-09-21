/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_OPTIMIZER_FORMAT_SELECTOR_BUILTIN_BROADCAST_FORMAT_PROCESS_BROADCAST_FORMAT_PROCESS_H_
#define FUSION_ENGINE_OPTIMIZER_FORMAT_SELECTOR_BUILTIN_BROADCAST_FORMAT_PROCESS_BROADCAST_FORMAT_PROCESS_H_

#include "common/fe_inner_error_codes.h"
#include "common/fe_log.h"
#include "common/format/axis_util.h"
#include "common/fe_op_info_common.h"
#include "graph/types.h"
#include "format_selector/builtin/process/format_process_registry.h"
#include "graph/ge_tensor.h"
#include "graph/node.h"
#include "graph/op_desc.h"

namespace fe {
class BroadcastFormatProcess : public FormatProcessBase {
 public:
  BroadcastFormatProcess(){};
  ~BroadcastFormatProcess() override {};
  Status Process(const ge::OpDesc &op_desc, const FormatProccessArgs &args, FormatProccessResult &result) override;

  virtual bool CheckOriginFormat(const std::vector<ge::Format> &input_formats,
                                 const vector<ge::GeShape> &input_shapes);
  virtual bool CheckOriginShape(const std::vector<ge::GeShape> &shapes);
  Status Check6HDShape(const std::vector<ge::GeShape> &shapes, const ge::Format &supprt_format,
                       size_t &dim_value) const;

  virtual bool CheckPartNonScalarInputs(const FormatProccessInputArg &input_arg) = 0;
  virtual bool CheckAllNonScalarInputs(const FormatProccessArgs &args) = 0;

 protected:
  void GetDimValue(const std::string &dim, const ge::Format &format, const ge::GeShape &shape,
                   int64_t &dim_value) const;
  bool CheckAxisNeedBroadcast(const std::string &dim, const std::vector<ge::Format> &formats,
                              const std::vector<ge::GeShape> &shapes) const;

 private:
  void GetScalarInputIndex(const std::vector<ge::GeShape> &shapes, std::set<size_t> &scalar_input_index) const;
  void GenerateOutputFormats(const vector<ge::GeShape> &output_shapes, const ge::Format &format,
                             vector<ge::Format> &output_formats) const;
  void InsertFormatVec(const size_t &size, const ge::Format &format, vector<vector<ge::Format>> &formats) const;
  void InsertSubformatVec(const ge::Format &format, const uint32_t &sub_format,
                          vector<uint32_t> &input_subformat_res, vector<uint32_t> &output_subformat_res) const;
  ge::GeShape GetNewShapeWithNewFormat(const ge::OpDesc &op_desc, const ge::GeShape &old_shape,
                                       const ge::Format &old_format, const ge::Format &new_format,
                                       const ge::DataType &current_data_type, const int32_t &group) const;
  bool CheckShapeWithSubformatSupportBroadcast(vector<ge::GeShape> &shapes) const;
  bool CheckNewShapeSupportBroadcast(const ge::OpDesc &op_desc, const vector<ge::Format> &input_formats,
                                     const vector<ge::DataType> &input_dtypes, const vector<ge::GeShape> &input_shapes,
                                     const ge::Format &new_format, uint32_t sub_format) const;
};
}  // namespace fe

#endif  // FUSION_ENGINE_OPTIMIZER_FORMAT_SELECTOR_BUILTIN_BROADCAST_FORMAT_PROCESS_BROADCAST_FORMAT_PROCESS_H_
