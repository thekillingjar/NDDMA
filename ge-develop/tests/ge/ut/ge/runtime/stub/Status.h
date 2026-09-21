/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_STATUS_H
#define AIR_CXX_STATUS_H
namespace grpc {
enum StatusCode {
  OK = 0,
  CANCELLED = 1
};

class Status {
 public:
  Status() 
    : code_(StatusCode::OK) {}
  bool ok() {
    return true;
  }

  int error_code() {
    return 0;
  }

  std::string error_message() {
    return "ok";
  }

  static const Status &OK;
  static const Status &CANCELLED;
  StatusCode code_;
};
}
#endif //AIR_CXX_STATUS_H
