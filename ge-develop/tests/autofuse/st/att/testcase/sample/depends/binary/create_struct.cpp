/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <regex>
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace fs = std::filesystem;
constexpr int32_t TILING_DATA_FILENAME_SUFFIX_LENGTH = 14;

int main() {
  std::string tiling_data_filename = "tiling_data.h";
  try {
    for (const auto &entry : fs::directory_iterator(".")) {
      if (entry.is_regular_file()) {
        std::string filename = entry.path().filename().string();
        if (filename.size() >= TILING_DATA_FILENAME_SUFFIX_LENGTH &&
            filename.substr(filename.size() - TILING_DATA_FILENAME_SUFFIX_LENGTH) == "_tiling_data.h") {
          tiling_data_filename = filename;
          break;
        }
      }
    }
  } catch (const fs::filesystem_error &e) {
    std::cerr << "Filesystem error: " << e.what() << '\n';
  } catch (const std::exception &e) {
    std::cerr << "Standard exception: " << e.what() << '\n';
  }
  std::string struct_name;
  std::string line;
  std::string content;
  std::string idx;
  std::string print_data;
  std::vector<std::string> pass_data;
  std::ifstream file(tiling_data_filename);
  std::regex pattern1("BEGIN_TILING_DATA_DEF\\(\\s*(.*?)\\s*\\)");
  std::regex pattern2("TILING_DATA_FIELD_DEF\\(\\s*(.*?)\\s*,\\s*(.*?)\\s*\\)");
  std::regex pattern3("TILING_DATA_FIELD_DEF_ARR\\(\\s*(.*?)\\s*,\\s*(.*?)\\s*,\\s*(.*?)\\s*\\)");
  std::regex pattern4("TILING_DATA_FIELD_DEF_STRUCT\\(\\s*(.*?)\\s*,\\s*(.*?)\\s*\\)");
  std::regex pattern5("END_TILING_DATA_DEF");

  while (std::getline(file, line)) {
    std::smatch match;
    if (std::regex_search(line, match, pattern1)) {
      content += "#pragma pack(1)\n";
      struct_name = "Pack" + match[1].str();
      content += "struct " + struct_name + " {\n";
    } else if (std::regex_search(line, match, pattern2)) {
      content += "  " + match[1].str() + " " + match[2].str() + " = 0;\n";
      pass_data.emplace_back("(tilingData)." + match[2].str() + " = tilingDataPointer->" + match[2].str() + ";");
      print_data += "  std::cout << \" " + match[2].str() + ": \" << tilingData." + match[2].str() + " << std::endl;\n";
    } else if (std::regex_search(line, match, pattern3)) {
      content += "  " + match[1].str() + " " + match[3].str() + "[" + match[2].str() + "] = 0;";
      for (uint16_t i = 0; i < std::stoul(match[2].str()); ++i) {
        idx = std::to_string(i);
        pass_data.emplace_back("(tilingData)." + match[2].str() + "[" + idx + "] = tilingDataPointer->" +
                               match[2].str() + "[" + idx + "];");
      }
    } else if (std::regex_search(line, match, pattern4)) {
      content += "  " + match[1].str() + " " + match[2].str() + " = {};\n";
      pass_data.emplace_back("(tilingData)." + match[2].str() + " = tilingDataPointer->" + match[2].str() + ";");
    } else if (std::regex_search(line, match, pattern5)) {
      content += "};\n";
      content += "#pragma pack()\n";
      content += "\n";
      content += "#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \\\n";
      content += "  tilingStruct *tilingDataPointer = reinterpret_cast<tilingStruct *>((uint8_t *)(tilingPointer));\n";
      content += "\n";
      content += "#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \\\n";
      content += "  CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);\n";
      content += "\n";
      content += "#define GET_TILING_DATA(tilingData, tilingPointer)                    \\\n";
      content += "  " + struct_name + " tilingData;                                      \\\n";
      content += "  INIT_TILING_DATA(" + struct_name + ", tilingDataPointer, tilingPointer);    \\\n";
      for (uint16_t i = 0; i < pass_data.size(); ++i) {
        if (i != pass_data.size() - 1) {
          content += "  " + pass_data[i] + "                   \\\n";
        } else {
          content += "  " + pass_data[i];
        }
      }
      content += "\n";
      content += "void PrintTilingData(" + struct_name + "& tilingData) {\n";
      content += "  std::cout << \"=======================================\" << std::endl;\n";
      content += print_data;
      content += "  std::cout << \"=======================================\" << std::endl;\n";
      content += "}\n";
      content += "\n";
    }
  }
  std::ofstream output_file("struct_info.h");
  if (output_file.is_open()) {
    output_file << content;
    output_file.close();
    std::cout << "Tiling data struct info has been written to struct_info.h" << std::endl;
  }
}