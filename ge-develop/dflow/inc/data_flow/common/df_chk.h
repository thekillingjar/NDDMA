/* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#ifndef DFLOW_INC_COMMON_DF_CHK_H_
#define DFLOW_INC_COMMON_DF_CHK_H_

#include "common/ge_common/debug/ge_log.h"
#include "acl/acl.h"

// -----------------runtime related macro definitions-------------------------------
// If expr is not ACL_ERROR_NONE, print the log
#define DF_CHK_ACL(expr)                                                \
  do {                                                                 \
    const aclError _acl_err = (expr);                                  \
    if (_acl_err != ACL_ERROR_NONE) {                                    \
      GELOGE(ge::RT_FAILED, "Call aclrt api failed, ret: 0x%X", _acl_err); \
    }                                                                  \
  } while (false)

// If expr is not ACL_ERROR_NONE, print the log and return
#define DF_CHK_ACL_RET(expr)                                                   \
  do {                                                                        \
    const aclError _acl_ret = (expr);                                         \
    if (_acl_ret != ACL_ERROR_NONE) {                                           \
      REPORT_INNER_ERR_MSG("E19999", "Call %s fail, ret: 0x%X", #expr, static_cast<uint32_t>(_acl_ret)); \
      GELOGE(ge::RT_FAILED, "Call aclrt api failed, ret: 0x%X", static_cast<uint32_t>(_acl_ret)); \
      return RT_ERROR_TO_GE_STATUS(_acl_ret);                                  \
    }                                                                         \
  } while (false)

#define DF_FREE_ACL_RT_LOG(addr)                                        \
  do {                                                              \
    if ((addr) != nullptr) {                                        \
      const aclError error = aclrtFree(addr);                         \
      if (error != ACL_ERROR_NONE) {                                 \
        GELOGE(ge::RT_FAILED, "Call aclrtFree failed, error: %#x", error); \
      }                                                             \
      (addr) = nullptr;                                             \
    }                                                               \
  } while (false)

#define DF_MAKE_GUARD_ACLSTREAM(var)    \
  GE_MAKE_GUARD(var, [&var]() {           \
    if ((var) != nullptr) {            \
      DF_CHK_ACL(aclrtDestroyStream(var)); \
    }                                  \
  })

#endif  // DFLOW_INC_COMMON_DF_CHK_H_
