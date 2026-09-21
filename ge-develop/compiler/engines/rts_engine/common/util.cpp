/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_context.h"
#include "external/ge/ge_api_types.h"
#include "util/log.h"
#include <cstring>
#include <string.h>
#include "securec.h"
using namespace ge;
using namespace std;
namespace cce {
namespace runtime {
ge::Status GetSocVersion(char_t *version, int32_t socVersionLen) {
  if (version == nullptr || socVersionLen <= 0) {
    RTS_LOGE("Invalid input parameters");
    return ge::FAILED;
  }

  std::string soc_version;
  graphStatus ret = ge::GetContext().GetOption(ge::SOC_VERSION, soc_version);
  if (ret != ge::SUCCESS) {
    RTS_LOGE("GetOption for SOC_VERSION failed");
    return ge::FAILED;
  }

  size_t copy_len = std::min(static_cast<size_t>(socVersionLen - 1), soc_version.length());
  auto rc = strncpy_s(version, socVersionLen, soc_version.c_str(), copy_len);
  if (rc != ge::SUCCESS) {
    RTS_LOGE("memcpy_s failed when copying SOC_VERSION");
    return ge::FAILED;
  }
  return ge::SUCCESS;
}

ge::Status IsNeedCalcOpRunningParam(const ge::Node &geNode, bool &isNeed) {
  ge::OpDescPtr opDesc = geNode.GetOpDesc();
  if (opDesc == nullptr) {
    RTS_REPORT_CALL_ERROR("calculate op running param failed, as op desc is null");
    return ge::FAILED;
  }
  bool unknownShape = false;
  if (ge::NodeUtils::GetNodeUnknownShapeStatus(geNode, unknownShape) == ge::GRAPH_SUCCESS) {
    if (unknownShape) {
      bool isNoTiling = false;
      (void)ge::AttrUtils::GetBool(opDesc, ATTR_NAME_OP_NO_TILING, isNoTiling);
      if (!isNoTiling) {
        RTS_LOGI("op: %s is unknownShape node, does not need to calc tensor size.", geNode.GetName().c_str());
        isNeed = false;  // unknownshape && not notiling
        return ge::SUCCESS;
      }
      RTS_LOGI("op: %s is unknownShape node, but is no tiling, need to calc tensor size.", geNode.GetName().c_str());
      isNeed = true;  // unknownshape && notiling
      return ge::SUCCESS;
    }
    isNeed = true;  // not unknownshape
  } else {
    RTS_REPORT_CALL_ERROR("op: %s get unknown shape status failed.", geNode.GetName().c_str());
    return ge::FAILED;
  }
  return ge::SUCCESS;
}

ge::Status IsSupportFftsPlus(bool &isSupportFlag) {
  isSupportFlag = false;
  const int32_t kSocVersionLen = 50;
  char_t version[kSocVersionLen] = {0};
  auto ret = GetSocVersion(version, kSocVersionLen);
  if (ret != SUCCESS) {
    RTS_LOGE("GetSocVersion failed.");
    return FAILED;
  }

  if ((strncmp(version, "Ascend910B", strlen("Ascend910B")) != 0) &&
      (strncmp(version, "Ascend910_93", strlen("Ascend910_93")) != 0)) {
    RTS_LOGI("SOC version is [%s], does not support ffts plus engine", version);
    return SUCCESS;
  }

  isSupportFlag = true;
  return SUCCESS;
}
}  // namespace runtime
}  // namespace cce