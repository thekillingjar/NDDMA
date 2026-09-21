/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dvpp_builder.h"
#include "common/dvpp_ops_checker.h"
#include "common/dvpp_ops_lib.h"
#include "util/dvpp_constexpr.h"
#include "util/dvpp_define.h"
#include "util/dvpp_error_code.h"
#include "util/dvpp_log.h"
#include "util/util.h"
#include "graph/node.h"

namespace dvpp {
DvppErrorCode DvppBuilder::CalcTotalSizeByDimsAndType(
    const std::vector<int64_t>& dims, const ge::DataType& data_type,
    int64_t& total_size) const {
  total_size = 1;
  for (auto& dim : dims) {
    // dim 包含 -1 或 -2 为动态算子 不计算
    DVPP_CHECK_IF_THEN_DO((dim == -1) || (dim == -2),
        DVPP_ENGINE_LOG_DEBUG("dim is [%ld], the op is dynamic", dim);
        break);

    DVPP_CHECK_IF_THEN_DO(dim < -2,
        DVPP_REPORT_INNER_ERR_MSG("dim[%ld] less than -2 is invalid", dim);
        return DvppErrorCode::kInvalidParam);

    // 芯片限制保证结果不会出现翻转
    total_size *= dim;
  }

  int32_t data_type_size = 0;
  bool ret = GetDataTypeSize(data_type, data_type_size);
  DVPP_CHECK_IF_THEN_DO(!ret,
      DVPP_REPORT_INNER_ERR_MSG("Invalid data type[%s]",
          ge::TypeUtils::DataTypeToSerialString(data_type).c_str());
      return DvppErrorCode::kInvalidParam);

  // 芯片限制保证结果不会出现翻转
  total_size *= data_type_size;
  return DvppErrorCode::kSuccess;
}

DvppErrorCode DvppBuilder::CalcOutputMemorySize(
    ge::GeTensorDesc& output_desc, int64_t& output_memory_size) const {
  auto dims = output_desc.GetShape().GetDims();
  DVPP_CHECK_IF_THEN_DO(dims.empty(), DVPP_REPORT_INNER_ERR_MSG("dims is empty");
    return DvppErrorCode::kInputParamNull);

  // 后续YUV格式需要根据format额外计算 GE需要同步新增格式
  // ge::Format data_format = output_desc.GetFormat()
  ge::DataType output_data_type = output_desc.GetDataType();
  auto error_code = CalcTotalSizeByDimsAndType(
      dims, output_data_type, output_memory_size);
  DVPP_CHECK_IF_THEN_DO(error_code != DvppErrorCode::kSuccess,
      DVPP_REPORT_INNER_ERR_MSG("calculate output memory size failed");
      return error_code);

  // shape_range 在动态shape场景才会使用
  std::vector<std::pair<int64_t, int64_t>> shape_range;
  for (size_t dim_index = 0; dim_index < dims.size(); ++dim_index) {
    shape_range.emplace_back(
        std::make_pair(dims[dim_index], dims[dim_index]));
  }
  output_desc.SetShapeRange(shape_range);
  return DvppErrorCode::kSuccess;
}

DvppErrorCode DvppBuilder::SetOutPutMemorySize(
    ge::OpDescPtr& op_desc_ptr) const {
  std::vector<int64_t> output_memory;
  std::vector<int64_t> output_alignment;
  uint32_t output_index = 0;
  for (auto& output_desc : op_desc_ptr->GetAllOutputsDesc()) {
    int64_t output_memory_size = 0;
    auto error_code = CalcOutputMemorySize(output_desc, output_memory_size);
    DVPP_CHECK_IF_THEN_DO(error_code != DvppErrorCode::kSuccess,
        DVPP_REPORT_INNER_ERR_MSG("op[%s] calculate output memory size failed",
                                op_desc_ptr->GetType().c_str());
        return error_code);

    DVPP_ENGINE_LOG_INFO("op[%s] output[%u] memory size is [%ld]",
        op_desc_ptr->GetType().c_str(), output_index, output_memory_size);

    // GE会根据每个output申请内存
    ge::TensorUtils::SetSize(output_desc, output_memory_size);

    (void)output_memory.emplace_back(output_memory_size);
    (void)output_alignment.emplace_back(kAlignmentValue);

    // 前面属性修改都是从GE获取的备份 需要调用update刷回去
    auto status = op_desc_ptr->UpdateOutputDesc(output_index, output_desc);
    DVPP_CHECK_IF_THEN_DO(status != ge::SUCCESS, return DvppErrorCode::kFailed);

    ++output_index;
  }

  // GE实际未使用 kOutputMemsizeVector & kOutputAlignmentVector 这里失败仅打印
  bool ret = ge::AttrUtils::SetListInt(
      op_desc_ptr, kOutputMemsizeVector, output_memory);
  DVPP_CHECK_IF_THEN_DO(!ret,
      DVPP_REPORT_INNER_ERR_MSG(
          "call ge::AttrUtils::SetListInt failed to set op[%s] attr[%s]",
          op_desc_ptr->GetType().c_str(), kOutputMemsizeVector.c_str()));

  ret = ge::AttrUtils::SetListInt(
      op_desc_ptr, kOutputAlignmentVector, output_alignment);
  DVPP_CHECK_IF_THEN_DO(!ret,
      DVPP_REPORT_INNER_ERR_MSG(
          "call ge::AttrUtils::SetListInt failed to set op[%s] attr[%s]",
          op_desc_ptr->GetType().c_str(), kOutputAlignmentVector.c_str()));

  return DvppErrorCode::kSuccess;
}

DvppErrorCode DvppBuilder::CalcOpRunningParam(ge::Node& node) const {
  auto op_desc_ptr = node.GetOpDesc();
  DVPP_CHECK_IF_THEN_DO(op_desc_ptr == nullptr,
      DVPP_REPORT_INNER_ERR_MSG("op_desc_ptr is nullptr");
      return DvppErrorCode::kInputParamNull);

  // name为算子名 在GE图中需要保持唯一 但可能存在同类型算子
  // 所以命名可能是 ResizeBilinearV2_0 ResizeBilinearV2_1
  std::string node_name = node.GetName();
  // type为算子类型 比如ResizeBilinearV2
  std::string node_type = node.GetType();
  DVPP_ENGINE_LOG_INFO("dvpp op %s[%s] run CalcOpRunningParam",
                       node_name.c_str(), node_type.c_str());

  // set outputs memory size 仅计算输出内存即可 输入内存为前面算子的输出内存
  auto error_code = SetOutPutMemorySize(op_desc_ptr);
  DVPP_CHECK_IF_THEN_DO(error_code != DvppErrorCode::kSuccess,
      DVPP_REPORT_INNER_ERR_MSG("op[%s] SetOutPutMemorySize failed",
                              node_type.c_str());
      return error_code);

  // set resource channel
  error_code = SetAttrResource(op_desc_ptr);
  DVPP_CHECK_IF_THEN_DO(error_code != DvppErrorCode::kSuccess,
      DVPP_REPORT_INNER_ERR_MSG("op[%s] SetAttrResource failed",
                              node_type.c_str());
      return error_code);

  DVPP_ENGINE_LOG_INFO("dvpp op %s[%s] run CalcOpRunningParam success",
                       node_name.c_str(), node_type.c_str());
  return DvppErrorCode::kSuccess;
}
} // namespace dvpp
