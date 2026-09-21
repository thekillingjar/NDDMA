/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_HEAVY_FORMAT_PROPAGATION_HEAVY_FORMAT_SUPPORT_FORMATS_UPDATER_H_
#define FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_HEAVY_FORMAT_PROPAGATION_HEAVY_FORMAT_SUPPORT_FORMATS_UPDATER_H_
#include <memory>
#include <vector>
#include "common/fe_inner_error_codes.h"
#include "common/fe_log.h"
#include "common/fe_utils.h"
#include "common/math_util.h"
#include "common/fe_op_info_common.h"
#include "common/util/op_info_util.h"
#include "format_selector/manager/format_dtype_querier.h"
#include "format_selector/manager/format_dtype_setter.h"
#include "graph/compute_graph.h"

namespace fe {
using FormatDtypeSetterPtr = std::shared_ptr<FormatDtypeSetter>;
using FormatDtypeQuerierPtr = std::shared_ptr<FormatDtypeQuerier>;

class HeavyFormatSupportFormatsUpdater {
 public:
  explicit HeavyFormatSupportFormatsUpdater(FormatDtypeQuerierPtr format_dtype_querier_ptr,
                                            FormatDtypeSetterPtr format_dtype_setter_ptr);
  ~HeavyFormatSupportFormatsUpdater();

  Status UpdateSupportFormats(const ge::NodePtr& node_ptr, const OpKernelInfoPtr& op_kernel_info_ptr,
                              const std::vector<IndexNameMap>& tensor_map, const HeavyFormatInfo& heavy_format_info);

 private:
  bool NeedUpdateSupportFormats(const ge::OpDescPtr& op_desc_ptr, const HeavyFormatInfo& heavy_format_info,
                                const vector<ge::Format>& kernel_formats, ge::Format propaga_heavy_format);
  bool IsFzRelaFormat(const HeavyFormatInfo& heavy_format_info) const;
  bool IsSelectFormatOrBroadcast(const ge::OpDescPtr& op_desc_ptr, const OpKernelInfoPtr& op_kernel_info_ptr);

  void UpdateSubFormatForTensors(const ge::OpDescPtr &op_desc_ptr, const HeavyFormatInfo& heavy_format_info) const;
  FormatDtypeQuerierPtr format_dtype_querier_ptr_;
  FormatDtypeSetterPtr format_dtype_setter_ptr_;
};
}  // namespace fe

#endif  // FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_HEAVY_FORMAT_PROPAGATION_HEAVY_FORMAT_SUPPORT_FORMATS_UPDATER_H_
