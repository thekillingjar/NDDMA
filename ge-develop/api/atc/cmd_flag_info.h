/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_COMPILER_OFFLINE_GEFLAGS_H_
#define AIR_CXX_COMPILER_OFFLINE_GEFLAGS_H_

#include <string>
#include <unordered_map>
#include <cstdint>

namespace ge {
namespace flgs {
using GfStatus = uint32_t;

const GfStatus GF_SUCCESS = 0x00U;
const GfStatus GF_FAILED = 0x01U;
const GfStatus GF_HELP = 0x02U;

/*
 * @brief      : register string command line parameter
 * @return     : string reference
 */
std::string& RegisterParamString(const std::string &name, const std::string &default_val, const std::string &msg);

/*
 * @brief      : register bool command line parameter
 * @return     : bool reference
 */
bool& RegisterParamBool(const std::string &name, bool default_val, const std::string &msg);

/*
 * @brief      : register int32_t command line parameter
 * @return     : int32_t reference
 */
int32_t& RegisterParamInt32(const std::string &name, int32_t default_val, const std::string &msg);

/*
 * @brief      : set usage message
 * @return     : NA
 */
void SetUsageMessage(const std::string &usage);

/*
 * @brief      : get command line string
 * @return     : argv string
 */
std::string& GetArgv();

/*
 * @brief      : parse command line
 * @param [in] : int32_t argc      cmdline param
 * @param [in] : char *argv[]      cmdline param
 * @return     : !=0 failure; ==0 success
 */
GfStatus ParseCommandLine(int32_t argc, char* argv[]);

/*
 * @brief      : get flag name and value which is user set
 * @return     : unordered_map<string, string>, key is flag name, value is flag value
 */
std::unordered_map<std::string, std::string> &GetUserOptions();
} // namespace flgs
} // namespace ge

#define DEFINE_bool(flag_name, default_val, msg) \
  bool &FLAGS_##flag_name = ge::flgs::RegisterParamBool((#flag_name), (default_val), (msg))

#define DEFINE_string(flag_name, default_val, msg) \
  std::string &FLAGS_##flag_name = ge::flgs::RegisterParamString((#flag_name), (default_val), (msg))

#define DEFINE_int32(flag_name, default_val, msg) \
  int32_t &FLAGS_##flag_name = ge::flgs::RegisterParamInt32((#flag_name), (default_val), (msg))

#define DECLARE_bool(flag_name) \
  extern bool &FLAGS_##flag_name

#define DECLARE_string(flag_name) \
  extern std::string &FLAGS_##flag_name

#define DECLARE_int32(flag_name) \
  extern int32_t &FLAGS_##flag_name

#endif // AIR_CXX_COMPILER_OFFLINE_GEFLAGS_H_
