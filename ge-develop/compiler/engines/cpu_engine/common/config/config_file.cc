/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "config_file.h"

#include <algorithm>
#include <fstream>

#include "util/log.h"
#include "util/util.h"

using namespace std;

namespace {
const string kConfigItemSeparator = ",";
}

namespace aicpu {
ConfigFile &ConfigFile::GetInstance() {
  static ConfigFile instance;
  return instance;  //lint !e1413
}

string ConfigFile::Trim(const string &source, const char delims) const {
  string result(source);
  string::size_type index = result.find_last_not_of(delims);
  if (index != string::npos) {
    result.erase(++index);
  }

  index = result.find_first_not_of(delims);
  if (index != string::npos) {
    result.erase(0, index);
  }
  return result;
}

aicpu::State ConfigFile::LoadFile(const string &file_name) {
  if (!is_load_file_) {
    std::fstream f;
    f.open(file_name.c_str(), std::fstream::in);
    if (!f.is_open()) {
      return aicpu::State(ge::FAILED, Stringcat("open file[", file_name, "] failed"));
    }
    string line;
    while (getline(f, line)) {
      // Skip comments and empty lines
      if (line.empty()) {
        AICPUE_LOGI("Skip empty line");
        continue;
      }
      string timed_line = Trim(Trim(Trim(line), '\r'), '\n');
      if (timed_line[0] == '#') {
        AICPUE_LOGI("Skip annotation line %s", timed_line.c_str());
        continue;
      }
      // parse "key = value" pair, if "=" non-exist,skip this line
      auto pos_trenner = timed_line.find('=');  //lint !e1101
      if (pos_trenner == string::npos) {
        AICPU_REPORT_INNER_ERR_MSG(
            "invalid item[%s], should be contain '=', file[%s]",
            timed_line.c_str(), file_name.c_str());
        continue;
      }

      string key = Trim(timed_line.substr(0, pos_trenner));
      if (data_map_.find(key) != data_map_.end()) {
        AICPUE_LOGW("repeated key %s.", key.c_str());
        continue;
      }

      string value = Trim(timed_line.substr(pos_trenner + 1));
      data_map_[key] = value;
    }
    f.close();

    if (data_map_.empty()) {
      return aicpu::State(ge::FAILED, Stringcat("cannot parse out the normative",
          " configuration k-v items from file[", file_name, "]"));
    }
    is_load_file_ = true;
  }
  return aicpu::State(ge::SUCCESS);
}

bool ConfigFile::GetValue(const string &key, string &value) const {
  auto iter = data_map_.find(key);  //lint !e1101
  if (iter == data_map_.end()) {
    return false;
  }
  value = iter->second;
  return true;
}

void ConfigFile::SetValue(const string &key, string &value) {
  auto iter = data_map_.find(key);
  if (iter != data_map_.end()) {
    data_map_[key] = value;
  }
}

void ConfigFile::SplitValue(const string &str, vector<string> &result,
                            const set<string> &blacklist) const {
  // Easy to intercept the last piece of data
  string strs = str + kConfigItemSeparator;

  size_t pos = strs.find(kConfigItemSeparator);
  size_t size = strs.size();

  while (pos != string::npos) {
    string x = strs.substr(0, pos);
    if (!x.empty() && blacklist.find(x) == blacklist.end()) {
      auto it = find(result.begin(), result.end(), x);
      if (it == result.end()) {
        result.push_back(x);
      }
    }
    strs = strs.substr(pos + kConfigItemSeparator.length(), size);
    pos = strs.find(kConfigItemSeparator);
  }
}
}  // namespace aicpu
