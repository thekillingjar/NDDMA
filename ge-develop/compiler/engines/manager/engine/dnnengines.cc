/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "engines/manager/engine/dnnengines.h"

#include <string>

namespace ge {
CustomDNNEngine::CustomDNNEngine(const std::string &engine_name) {
  engine_attribute_.engine_name = engine_name;
  engine_attribute_.compute_cost = PriorityEnum::COST_0;
  engine_attribute_.runtime_type = DEVICE;
  engine_attribute_.engine_input_format = FORMAT_RESERVED;
  engine_attribute_.engine_output_format = FORMAT_RESERVED;
}

AICoreDNNEngine::AICoreDNNEngine(const std::string &engine_name) {
  engine_attribute_.engine_name = engine_name;
  engine_attribute_.compute_cost = PriorityEnum::COST_1;
  engine_attribute_.runtime_type = DEVICE;
  engine_attribute_.engine_input_format = FORMAT_RESERVED;
  engine_attribute_.engine_output_format = FORMAT_RESERVED;
}

VectorCoreDNNEngine::VectorCoreDNNEngine(const std::string &engine_name) {
  engine_attribute_.engine_name = engine_name;
  engine_attribute_.compute_cost = PriorityEnum::COST_2;
  engine_attribute_.runtime_type = DEVICE;
  engine_attribute_.engine_input_format = FORMAT_RESERVED;
  engine_attribute_.engine_output_format = FORMAT_RESERVED;
}

AICpuDNNEngine::AICpuDNNEngine(const std::string &engine_name) {
  engine_attribute_.engine_name = engine_name;
  engine_attribute_.compute_cost = PriorityEnum::COST_3;
  engine_attribute_.runtime_type = DEVICE;
  engine_attribute_.engine_input_format = FORMAT_RESERVED;
  engine_attribute_.engine_output_format = FORMAT_RESERVED;
}

DvppDNNEngine::DvppDNNEngine(const std::string &engine_name) {
  engine_attribute_.engine_name = engine_name;
  engine_attribute_.compute_cost = PriorityEnum::COST_5;
  engine_attribute_.runtime_type = DEVICE;
  engine_attribute_.engine_input_format = FORMAT_RESERVED;
  engine_attribute_.engine_output_format = FORMAT_RESERVED;
}

AICpuTFDNNEngine::AICpuTFDNNEngine(const std::string &engine_name) {
  engine_attribute_.engine_name = engine_name;
  engine_attribute_.compute_cost = PriorityEnum::COST_4;
  engine_attribute_.runtime_type = DEVICE;
  engine_attribute_.engine_input_format = FORMAT_RESERVED;
  engine_attribute_.engine_output_format = FORMAT_RESERVED;
}

GeLocalDNNEngine::GeLocalDNNEngine(const std::string &engine_name) {
  engine_attribute_.engine_name = engine_name;
  engine_attribute_.engine_input_format = FORMAT_RESERVED;
  engine_attribute_.engine_output_format = FORMAT_RESERVED;
}

HostCpuDNNEngine::HostCpuDNNEngine(const std::string &engine_name) {
  engine_attribute_.engine_name = engine_name;
  engine_attribute_.compute_cost = PriorityEnum::COST_10;
  engine_attribute_.runtime_type = HOST;
  engine_attribute_.engine_input_format = FORMAT_RESERVED;
  engine_attribute_.engine_output_format = FORMAT_RESERVED;
}

RtsDNNEngine::RtsDNNEngine(const std::string &engine_name) {
  engine_attribute_.engine_name = engine_name;
  engine_attribute_.engine_input_format = FORMAT_RESERVED;
  engine_attribute_.engine_output_format = FORMAT_RESERVED;
}

RtsFftsPlusDNNEngine::RtsFftsPlusDNNEngine(const std::string &engine_name) {
  engine_attribute_.engine_name = engine_name;
  engine_attribute_.engine_input_format = FORMAT_RESERVED;
  engine_attribute_.engine_output_format = FORMAT_RESERVED;
}

HcclDNNEngine::HcclDNNEngine(const std::string &engine_name) {
  engine_attribute_.engine_name = engine_name;
  engine_attribute_.engine_input_format = FORMAT_RESERVED;
  engine_attribute_.engine_output_format = FORMAT_RESERVED;
}

FftsPlusDNNEngine::FftsPlusDNNEngine(const std::string &engine_name) {
  engine_attribute_.engine_name = engine_name;
  engine_attribute_.engine_input_format = FORMAT_RESERVED;
  engine_attribute_.engine_output_format = FORMAT_RESERVED;
}

DSADNNEngine::DSADNNEngine(const std::string &engine_name) {
  engine_attribute_.engine_name = engine_name;
  engine_attribute_.compute_cost = PriorityEnum::COST_2;
  engine_attribute_.runtime_type = DEVICE;
  engine_attribute_.engine_input_format = FORMAT_RESERVED;
  engine_attribute_.engine_output_format = FORMAT_RESERVED;
}
}  // namespace ge
