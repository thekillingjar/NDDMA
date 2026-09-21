/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dvpp_ops_lib.h"
#include "util/dvpp_constexpr.h"
#include "util/dvpp_define.h"
#include "util/dvpp_error_code.h"
#include "util/dvpp_log.h"
#include "util/json_file_operator.h"
#include "util/util.h"
#include "graph/types.h"
#include "yuv_subformat.h"
#include "graph/utils/op_desc_utils_ex.h"
#include "mmpa/mmpa_api.h"

namespace dvpp {
DvppOpsLib& DvppOpsLib::Instance() {
  static DvppOpsLib instance;
  return instance;
}

DvppErrorCode DvppOpsLib::ReadOpsInfoFromJsonFile(std::string& file_name) {
  std::string dvpp_ops_file_path_new;
  std::string dvpp_ops_file_path;
  char* path_env = nullptr;
  MM_SYS_GET_ENV(MM_ENV_ASCEND_OPP_PATH, path_env);
  if (path_env != nullptr) {
    std::string path = path_env;
    std::string file_path = RealPath(path);
    DVPP_CHECK_IF_THEN_DO(file_path.empty(),
        DVPP_REPORT_INNER_ERR_MSG("opp file path is empty.");
        return DvppErrorCode::kFailed);
    dvpp_ops_file_path_new = path + kDvppOpsJsonFilePathBaseOnEnvPathNew;
    dvpp_ops_file_path = path + kDvppOpsJsonFilePathBaseOnEnvPath;
  } else {
    dvpp_ops_file_path_new = kDvppOpsJsonFilePathNew;
    dvpp_ops_file_path = kDvppOpsJsonFilePath;
  }
  dvpp_ops_file_path_new += file_name;
  dvpp_ops_file_path += file_name;
  DVPP_ENGINE_LOG_INFO("new dvpp ops file path is %s.",
                       dvpp_ops_file_path_new.c_str());
  DVPP_ENGINE_LOG_INFO("dvpp ops file path is %s", dvpp_ops_file_path.c_str());

  bool ret = JsonFileOperator::ReadJsonFile(
      dvpp_ops_file_path_new, dvpp_ops_file_path, dvpp_ops_info_json_file_);
  DVPP_CHECK_IF_THEN_DO(!ret,
      DVPP_REPORT_INNER_ERR_MSG("call ReadJsonFile failed");
      return DvppErrorCode::kFailed);

  DVPP_ENGINE_LOG_INFO("read ops info from json file success.");
  return DvppErrorCode::kSuccess;
}

DvppErrorCode DvppOpsLib::GetAllOpsInfo(DvppOpsInfoLib& dvpp_ops_info_lib) {
  for (const auto& dvpp_op : dvpp_ops_info_lib.ops) {
    DVPP_CHECK_IF_THEN_DO(dvpp_op.name.empty(), continue);

    auto ret = all_ops_.emplace(
        std::pair<std::string, DvppOpInfo>(dvpp_op.name, dvpp_op.info));
    DVPP_CHECK_IF_THEN_DO(!ret.second,
        DVPP_ENGINE_LOG_WARNING("insert a pair of op[%s] and OpInfo failed",
                                dvpp_op.name.c_str()));

    DVPP_ENGINE_LOG_DEBUG("read op[%s] from json file.", dvpp_op.name.c_str());
  }

  PrintDvppOpInfo(all_ops_);
  return DvppErrorCode::kSuccess;
}

DvppErrorCode DvppOpsLib::Initialize(std::string& file_name) {
  // read dvpp ops json file
  auto error_code = ReadOpsInfoFromJsonFile(file_name);
  DVPP_CHECK_IF_THEN_DO(error_code != DvppErrorCode::kSuccess,
      DVPP_REPORT_INNER_ERR_MSG("call ReadOpsInfoFromJsonFile failed");
      return error_code);

  try {
    // will call from_json
    DvppOpsInfoLib dvpp_ops_info_lib = dvpp_ops_info_json_file_;
    DVPP_ENGINE_LOG_INFO("read dvpp ops info json file success, op size is %lu",
                         dvpp_ops_info_lib.ops.size());
    return GetAllOpsInfo(dvpp_ops_info_lib);
  } catch (const nlohmann::json::exception& e) {
    DVPP_REPORT_INNER_ERR_MSG("read json file failed, %s", e.what());
    return DvppErrorCode::kFailed;
  }
}

std::map<std::string, DvppOpInfo> DvppOpsLib::AllOps() const {
  return all_ops_;
}

bool DvppOpsLib::GetDvppOpInfo(
    std::string& op_name, DvppOpInfo& dvpp_op_info) const {
  auto iter = all_ops_.find(op_name);
  DVPP_CHECK_IF_THEN_DO(iter == all_ops_.end(),
      DVPP_ENGINE_LOG_EVENT("dvpp ops info store not include op[%s]",
                            op_name.c_str());
      return false);

  dvpp_op_info = iter->second;
  return true;
}

std::string DvppOpsLib::CreateDvppOpName(const std::string& op_type) {
  // op_num回绕也没有问题 只要保证一张图中的op_name是唯一即可
  static uint16_t op_num = 1;
  std::string op_name;
  const std::lock_guard<std::mutex> lock(op_num_mutex_);
  op_name = op_type + "_dvpp_" + std::to_string(op_num++);
  return op_name;
}

void DvppOpsLib::CreateAdjustContrastWithMeanDesc(ge::OpDescPtr& op_desc_ptr) {
  ge::GeTensorDesc tensor_x(ge::GeShape({kDynamicShape}),
  ge::FORMAT_ND, ge::DT_FLOAT);
  op_desc_ptr->AddInputDesc("image", tensor_x);

  ge::GeTensorDesc tensor_mean(ge::GeShape({kDynamicShape}),
                               ge::FORMAT_ND, ge::DT_FLOAT);
  op_desc_ptr->AddInputDesc("mean", tensor_mean);

  ge::GeTensorDesc tensor_ratio(ge::GeShape({kDynamicShape}),
                                ge::FORMAT_ND, ge::DT_FLOAT);
  op_desc_ptr->AddInputDesc("contrast_factor", tensor_ratio);

  ge::GeTensorDesc tensor_y(ge::GeShape({kDynamicShape}),
                            ge::FORMAT_ND, ge::DT_FLOAT);
  op_desc_ptr->AddOutputDesc("y", tensor_y);
}

void DvppOpsLib::CreateRgbToGrayscaleDesc(ge::OpDescPtr& op_desc_ptr) {
  ge::GeTensorDesc tensor_x(ge::GeShape({kDynamicShape}),
  ge::FORMAT_ND, ge::DT_FLOAT);
  op_desc_ptr->AddInputDesc("x", tensor_x);

  ge::GeTensorDesc tensor_y(ge::GeShape({kDynamicShape}),
                            ge::FORMAT_ND, ge::DT_FLOAT);
  op_desc_ptr->AddOutputDesc("y", tensor_y);
}

DvppErrorCode DvppOpsLib::CreateDvppOpDesc(
    const std::string& op_type, ge::OpDescPtr& op_desc_ptr) {
  std::string op_name = CreateDvppOpName(op_type);
  DVPP_TRY_CATCH(op_desc_ptr = std::make_shared<ge::OpDesc>(op_name, op_type),
      DVPP_REPORT_INNER_ERR_MSG("make shared op[%s] desc failed.",
                              op_name.c_str());
      return DvppErrorCode::kFailed);

  if (op_type == kDvppOpDecodeJpegPre) {
    ge::GeTensorDesc tensor_x(ge::GeShape({kDynamicShape}),
                              ge::FORMAT_ND, ge::DT_STRING);
    op_desc_ptr->AddInputDesc("contents", tensor_x);
    op_desc_ptr->AddRequiredAttr("w_range");
    op_desc_ptr->AddRequiredAttr("h_range");

    ge::GeTensorDesc tensor_y(ge::GeShape(std::vector<int64_t>{}),
                              ge::FORMAT_ND, ge::DT_BOOL);
    op_desc_ptr->AddOutputDesc("dvpp_support", tensor_y);
  } else if (op_type == kNodeData) {
    ge::GeTensorDesc tensor_x(ge::GeShape({kDynamicShape}),
                              ge::FORMAT_ND, ge::DT_STRING);
    op_desc_ptr->AddInputDesc("x", tensor_x);

    ge::GeTensorDesc tensor_y(ge::GeShape({kDynamicShape}),
                              ge::FORMAT_ND, ge::DT_STRING);
    op_desc_ptr->AddOutputDesc("y", tensor_y);
  } else if (op_type == kNodeNetOutput) {
    ge::GeTensorDesc tensor_x(ge::GeShape({kDynamicShape}),
                              ge::FORMAT_ND, ge::DT_UINT8);
    op_desc_ptr->AddInputDesc(0, tensor_x);
    ge::OpDescUtilsEx::SetType(op_desc_ptr, kNodeNetOutput);
  } else if (op_type == kDvppOpNormalizeV2) {
    ge::GeTensorDesc tensor_x(ge::GeShape({kDynamicShape}),
                              ge::FORMAT_ND, ge::DT_FLOAT);
    op_desc_ptr->AddInputDesc("x", tensor_x);

    ge::GeTensorDesc tensor_mean(ge::GeShape({kDynamicShape}),
                                 ge::FORMAT_ND, ge::DT_FLOAT);
    op_desc_ptr->AddInputDesc("mean", tensor_mean);

    ge::GeTensorDesc tensor_variance(ge::GeShape({kDynamicShape}),
                                     ge::FORMAT_ND, ge::DT_FLOAT);
    op_desc_ptr->AddInputDesc("variance", tensor_variance);

    ge::GeTensorDesc tensor_y(ge::GeShape({kDynamicShape}),
                              ge::FORMAT_ND, ge::DT_FLOAT);
    op_desc_ptr->AddOutputDesc("y", tensor_y);
  } else if (op_type == kDvppOpAdjustContrastWithMean) {
    CreateAdjustContrastWithMeanDesc(op_desc_ptr);
  } else if (op_type == kDvppOpRgbToGrayscale) {
    CreateRgbToGrayscaleDesc(op_desc_ptr);
  } else {
    DVPP_REPORT_INNER_ERR_MSG("do not support create op[%s].", op_type.c_str());
    return DvppErrorCode::kFailed;
  }

  return DvppErrorCode::kSuccess;
}

DvppErrorCode DvppOpsLib::CreateAICpuReduceMeanDesc(
    ge::OpDescPtr& op_desc_ptr) {
  std::string op_name = CreateDvppOpName(kAICpuOpReduceMean);
  DVPP_TRY_CATCH(op_desc_ptr =
      std::make_shared<ge::OpDesc>(op_name, kAICpuOpReduceMean),
      DVPP_REPORT_INNER_ERR_MSG("make shared op[%s] desc failed.",
                              op_name.c_str());
      return DvppErrorCode::kFailed);

  ge::GeTensorDesc tensor_x(ge::GeShape({kDynamicShape}),
  ge::FORMAT_ND, ge::DT_FLOAT);
  op_desc_ptr->AddInputDesc("x", tensor_x);

  ge::GeTensorDesc tensor_axes(ge::GeShape({kDynamicShape}),
  ge::FORMAT_ND, ge::DT_FLOAT);
  op_desc_ptr->AddInputDesc("axes", tensor_axes);

  ge::GeTensorDesc tensor_y(ge::GeShape({kDynamicShape}),
  ge::FORMAT_ND, ge::DT_FLOAT);
  op_desc_ptr->AddOutputDesc("y", tensor_y);

  return DvppErrorCode::kSuccess;
}

DvppErrorCode DvppOpsLib::CreateIfOpDesc(
    const std::string& op_type, ge::OpDescPtr& op_desc_ptr) {
  std::string op_name = CreateDvppOpName(kDvppOpIf);
  DVPP_TRY_CATCH(op_desc_ptr = std::make_shared<ge::OpDesc>(op_name, kDvppOpIf),
      DVPP_REPORT_INNER_ERR_MSG("make shared op[%s] desc failed",
                              op_name.c_str());
      return DvppErrorCode::kFailed);

  ge::GeTensorDesc tensor_x(ge::GeShape(std::vector<int64_t>{}),
                           ge::FORMAT_ND, ge::DT_BOOL);
  op_desc_ptr->AddInputDesc("cond", tensor_x);
  op_desc_ptr->AddDynamicOutputDesc("output", 1);

  if (op_type == kDvppOpDecodeJpeg) {
    op_desc_ptr->AddDynamicInputDesc("input", kNum1);
  }

  if (op_type == kDvppOpDecodeAndCropJpeg) {
    op_desc_ptr->AddDynamicInputDesc("input", kNum2);
    auto dy_input1 = op_desc_ptr->MutableInputDesc("input1");
    DVPP_CHECK_IF_THEN_DO(dy_input1 == nullptr,
        DVPP_ENGINE_LOG_EVENT("op[%s] get input1 desc is nullptr.",
                              op_type.c_str());
        return DvppErrorCode::kFailed);
    dy_input1->SetDataType(ge::DT_INT32);
    dy_input1->SetShape(ge::GeShape(std::vector<int64_t>{kNum4}));

    auto dy_output0 = op_desc_ptr->MutableOutputDesc("output0");
    DVPP_CHECK_IF_THEN_DO(dy_output0 == nullptr,
        DVPP_ENGINE_LOG_EVENT("op[%s] get output0 desc is nullptr.",
                              op_type.c_str());
        return DvppErrorCode::kFailed);
    dy_output0->SetShape(ge::GeShape({kDynamicShape}));
  }

  return DvppErrorCode::kSuccess;
}
} // namespace dvpp
