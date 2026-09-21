/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_OPTIMIZER_FORMAT_SELECTOR_BUILTIN_REDUCE_REDUCE_FORMAT_SELECTOR_REDUCE_PROCESS_NZ_H_
#define FUSION_ENGINE_OPTIMIZER_FORMAT_SELECTOR_BUILTIN_REDUCE_REDUCE_FORMAT_SELECTOR_REDUCE_PROCESS_NZ_H_
#include "format_selector/builtin/reduce/reduce_format_selector/reduce_format_process.h"

namespace fe {

class ReduceProcessNz : public ReduceFormatProcess {
 public:
  ReduceProcessNz(){};

  ~ReduceProcessNz() override {};

  Status Process(const ge::OpDesc &op_desc, const FormatProccessArgs &args, FormatProccessResult &result) override;

  bool CheckOriginFormatAndShape(const vector<ge::Format> &input_formats, const vector<ge::Format> &output_formats,
                                 const vector<ge::GeShape> &shapes, const size_t &dim) override;

 private:
  // check if the last axis is C0 aligend and if the lastbutone is 16 aligned
  bool CheckNzAxisIsAligned(const ge::OpDesc &op_desc, const vector<ge::DataType> &dtypes,
                            const vector<ge::GeShape> &shapes, const bool &is_reduce_last,
                            const bool &is_reduce_lastbutone) const;
};

}  // namespace fe

#endif  // FUSION_ENGINE_OPTIMIZER_FORMAT_SELECTOR_BUILTIN_REDUCE_REDUCE_FORMAT_SELECTOR_REDUCE_PROCESS_NZ_H_
