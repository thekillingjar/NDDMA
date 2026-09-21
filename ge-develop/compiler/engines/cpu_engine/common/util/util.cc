/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "util.h"

#include <dlfcn.h>
#include <regex.h>
#include <cstdlib>
#include <sys/time.h>

#include <algorithm>
#include <climits>
#include <fstream>

#include "constant.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"
#include "common/sgt_slice_type.h"

using namespace std;
using namespace ge;

namespace {
// max file size limit
constexpr int kMaxFileSizeLimit = INT_MAX;

const string kOutputMemsizeVector = "output_memsize_vector";
const string kOutputAlignmentVector = "output_alignment_vector";
const string kFrameworkOp = "FrameworkOp";
const string kOriginalType = "original_type";
const string kOpNoTiling = "_op_no_tiling";
const uint64_t kAlignmentValue = 64;

// data type map
const map<string, DataType> DataTypeMap = {
    {"DT_BOOL", DataType::DT_BOOL},
    {"DT_INT8", DataType::DT_INT8},
    {"DT_UINT8", DataType::DT_UINT8},
    {"DT_INT16", DataType::DT_INT16},
    {"DT_UINT16", DataType::DT_UINT16},
    {"DT_INT32", DataType::DT_INT32},
    {"DT_UINT32", DataType::DT_UINT32},
    {"DT_INT64", DataType::DT_INT64},
    {"DT_UINT64", DataType::DT_UINT64},
    {"DT_FLOAT16", DataType::DT_FLOAT16},
    {"DT_BF16", DataType::DT_BF16},
    {"DT_FLOAT", DataType::DT_FLOAT},
    {"DT_DOUBLE", DataType::DT_DOUBLE},
    {"DT_COMPLEX32", DataType::DT_COMPLEX32},
    {"DT_COMPLEX64", DataType::DT_COMPLEX64},
    {"DT_COMPLEX128", DataType::DT_COMPLEX128},
    {"DT_QINT8", DataType::DT_QINT8},
    {"DT_QUINT8", DataType::DT_QUINT8},
    {"DT_QINT16", DataType::DT_QINT16},
    {"DT_QUINT16", DataType::DT_QUINT16},
    {"DT_QINT32", DataType::DT_QINT32},
    {"DT_STRING", DataType::DT_STRING},
    {"DT_STRING_REF", DataType::DT_STRING_REF},
    {"DT_RESOURCE", DataType::DT_RESOURCE},
    {"DT_VARIANT", DataType::DT_VARIANT},
    {"DT_UINT1", DataType::DT_UINT1},
    {"DT_HIFLOAT8", DataType::DT_HIFLOAT8},
    {"DT_FLOAT8_E4M3FN", DataType::DT_FLOAT8_E4M3FN},
    {"DT_FLOAT8_E5M2", DataType::DT_FLOAT8_E5M2},
    {"DT_FLOAT8_E8M0", DataType::DT_FLOAT8_E8M0},
    {"DT_FLOAT4_E2M1", DataType::DT_FLOAT4_E2M1},
    {"DT_FLOAT4_E1M2", DataType::DT_FLOAT4_E1M2}};

// need delete when update libgraph.so in blue zone
const std::string AICPU_ATTR_NAME_TENSOR_MAX_SHAPE = "_tensor_max_shape";
}  // namespace

namespace aicpu {
const string GetSoPath(const void *instance) {
  Dl_info dl_info;
  char resoved_path[PATH_MAX] = {0x00};
  string real_file_path;
  AICPU_IF_BOOL_EXEC((dladdr(instance, &dl_info) == 0),
      AICPU_REPORT_INNER_ERR_MSG("Call dladdr failed.");
      return real_file_path);
  string so_path = dl_info.dli_fname;
  AICPUE_LOGI("So file path=%s", so_path.c_str());
  AICPU_IF_BOOL_EXEC(so_path.empty(), AICPUE_LOGI("So file path is empty.");
                     return real_file_path);
  AICPU_IF_BOOL_EXEC(realpath(so_path.c_str(), resoved_path) == nullptr,
      AICPU_REPORT_INNER_ERR_MSG("realpath [%s] failed, %s.",
          so_path.c_str(), strerror(errno));
      return real_file_path);
  string so_file_path = resoved_path;
  string::size_type pos_so = so_file_path.rfind('/');
  AICPU_IF_BOOL_EXEC(
      pos_so == string::npos,
      AICPU_REPORT_INNER_ERR_MSG("Invalid path[%s] not contain /.",
          so_file_path.c_str());
      return real_file_path);
  real_file_path = so_file_path.substr(0, pos_so + 1);

  AICPUE_LOGI("Real config File path is %s", real_file_path.c_str());
  return real_file_path;
}

bool IsSameShape(const ge::GeShape &first_shape, const ge::GeShape &second_shape) {
  if (first_shape.GetDimNum() != second_shape.GetDimNum()) {
    return false;
  }
  size_t dim_value = first_shape.GetDimNum();
  for (size_t i = 0; i < dim_value; i++) {
    if (first_shape.GetDim(i) == UNKNOWN_DIM && second_shape.GetDim(i) == UNKNOWN_DIM) {
      return false;
    }
    if (first_shape.GetDim(i) != second_shape.GetDim(i)) {
      return false;
    }
  }
  return true;
}

string CurrentTimeInStr() {
  time_t now = time(nullptr);
  tm *ptm = localtime(&now);
  if (ptm == nullptr) {
    AICPU_REPORT_INNER_ERR_MSG("Call localtime function failed.");
    return "";
  }

  const int time_buffer_len = 32;
  char buffer[time_buffer_len] = {0};
  // format: 20171122042550
  (void)strftime(buffer, time_buffer_len, "%Y%m%d%H%M%S", ptm);
  return string(buffer);
}

aicpu::State ReadJsonFile(const string &file_path, nlohmann::json &json_read) {
  AICPUE_LOGI("Read %s json file", file_path.c_str());
  ifstream ifs(file_path);
  AICPU_IF_BOOL_EXEC(
      !ifs.is_open(),
      return aicpu::State(ge::FAILED, Stringcat("open [", file_path, "] failed")));

  try {
    ifs >> json_read;
    ifs.close();
  } catch (const nlohmann::json::exception &e) {
    AICPU_IF_BOOL_EXEC(ifs.is_open(), ifs.close();)
    return aicpu::State(ge::FAILED, e.what());
  }

  AICPUE_LOGD("Read %s json file success.", file_path.c_str());
  return aicpu::State(ge::SUCCESS);
}

aicpu::State StringToBool(const string &str, bool &result) {
  result = false;
  string buff = str;
  try {
    (void)transform(buff.begin(), buff.end(), buff.begin(), ::tolower);
    if ((buff == "false") || (buff == "true")) {
      istringstream(buff) >> boolalpha >> result;
      return aicpu::State(ge::SUCCESS);
    }
  } catch (std::exception &e) {
    // something else reason
    return aicpu::State(ge::FAILED, e.what());
  }
  return aicpu::State(ge::FAILED, "unknown reason");
}

void SplitSequence(const string &str, const string &pattern,
                   set<string> &result) {
  // Easy to intercept the last piece of data
  string strs = str + pattern;

  size_t pos = strs.find(pattern);
  size_t size = strs.size();

  while (pos != string::npos) {
    string x = strs.substr(0, pos);
    if (!x.empty()) {
      (void)result.emplace(x);
    }
    strs = strs.substr(pos + pattern.length(), size);
    pos = strs.find(pattern);
  }
}

void SplitSequence(const string &str, const string &pattern,
                   vector<string> &result) {
  // Easy to intercept the last piece of data
  string strs = str + pattern;

  size_t pos = strs.find(pattern);
  size_t size = strs.size();

  while (pos != string::npos) {
    string x = strs.substr(0, pos);
    if (!x.empty()) {
      result.push_back(x);
    }
    strs = strs.substr(pos + pattern.length(), size);
    pos = strs.find(pattern);
  }
}

void GetDataType(const map<string, string> &data_types,
                 const string &data_type_name, set<DataType> &data_type) {
  auto type_iter = data_types.find(data_type_name);
  if (type_iter != data_types.end()) {
    string data_type_str = type_iter->second;
    set<string> data_type_set;
    SplitSequence(data_type_str, kConfigItemSeparator, data_type_set);

    for (auto &elem : data_type_set) {
      auto it = DataTypeMap.find(elem);
      // only data_type in configure file exists in DataTypeMap, save it and
      // return
      if (it != DataTypeMap.end()) {
        (void)data_type.insert(it->second);
      }
    }
  } else {
    data_type = {DT_UNDEFINED};
  }
}

void GetDataType(const map<string, string> &data_types,
                 const string &data_type_name, vector<DataType> &data_type) {
  auto type_iter = data_types.find(data_type_name);
  if (type_iter != data_types.end()) {
    string data_type_str = type_iter->second;
    vector<string> data_type_set;
    SplitSequence(data_type_str, kConfigItemSeparator, data_type_set);

    for (auto &elem : data_type_set) {
      auto it = DataTypeMap.find(elem);
      // only data_type in configure file exists in DataTypeMap, save it and
      // return
      if (it != DataTypeMap.end()) {
        data_type.push_back(it->second);
      }
    }
  }
}

bool ConvertDataType2String(string &elem, DataType data_type) {
  static const map<DataType, string> DataTypeConvertMap = {
      {DataType::DT_FLOAT, "DT_FLOAT"},
      {DataType::DT_FLOAT16, "DT_FLOAT16"},
      {DataType::DT_BF16, "DT_BF16"},
      {DataType::DT_INT8, "DT_INT8"},
      {DataType::DT_UINT8, "DT_UINT8"},
      {DataType::DT_INT16, "DT_INT16"},
      {DataType::DT_UINT16, "DT_UINT16"},
      {DataType::DT_INT32, "DT_INT32"},
      {DataType::DT_INT64, "DT_INT64"},
      {DataType::DT_UINT32, "DT_UINT32"},
      {DataType::DT_UINT64, "DT_UINT64"},
      {DataType::DT_BOOL, "DT_BOOL"},
      {DataType::DT_DOUBLE, "DT_DOUBLE"},
      {DataType::DT_COMPLEX32, "DT_COMPLEX32"},
      {DataType::DT_COMPLEX64, "DT_COMPLEX64"},
      {DataType::DT_COMPLEX128, "DT_COMPLEX128"},
      {DataType::DT_STRING, "DT_STRING"},
      {DataType::DT_STRING_REF, "DT_STRING_REF"},
      {DataType::DT_RESOURCE, "DT_RESOURCE"},
      {DataType::DT_QINT8, "DT_QINT8"},
      {DataType::DT_QINT16, "DT_QINT16"},
      {DataType::DT_QINT32, "DT_QINT32"},
      {DataType::DT_QUINT8, "DT_QUINT8"},
      {DataType::DT_QUINT16, "DT_QUINT16"},
      {DataType::DT_DUAL_SUB_INT8, "DT_DUAL_SUB_INT8"},
      {DataType::DT_DUAL_SUB_UINT8, "DT_DUAL_SUB_UINT8"},
      {DataType::DT_DUAL, "DT_DUAL"},
      {DataType::DT_VARIANT, "DT_VARIANT"},
      {DataType::DT_UINT1, "DT_UINT1"},
      {DataType::DT_HIFLOAT8, "DT_HIFLOAT8"}, {DataType::DT_FLOAT8_E4M3FN, "DT_FLOAT8_E4M3FN"}, 
      {DataType::DT_FLOAT8_E5M2, "DT_FLOAT8_E5M2"}, {DataType::DT_FLOAT8_E8M0, "DT_FLOAT8_E8M0"}, 
      {DataType::DT_FLOAT4_E2M1, "DT_FLOAT4_E2M1"}, {DataType::DT_FLOAT4_E1M2, "DT_FLOAT4_E1M2"}};
  auto it = DataTypeConvertMap.find(data_type);
  // only data_type in configure file exists in DataTypeMap_convert, save it and return
  if (it != DataTypeConvertMap.end()) {
    elem = it->second;
    return true;
  }
  elem = "DT_INVALID";
  return false;
}

int32_t GetDataTypeSize(DataType data_type) {
  constexpr uint32_t kSizeOfFloat16 = 2;
  // DT_RESOURCE: ge allocate 8 bytes
  constexpr int32_t kSizeOfResource = 8;
  // DT_STRING_REF: ge allocate 16 bytes
  // store mutex pointer and tensor pointer
  constexpr int32_t kSizeOfStringRef = 16;
  // DT_STRING: ge allocate 8 bytes
  // store string rawdata pointer on device ddr momery
  constexpr int32_t kSizeOfString = 16;
  constexpr int32_t kSizeOfComplex32 = 4;
  constexpr int32_t kSizeOfComplex64 = 8;
  constexpr int32_t kSizeOfComplex128 = 16;
  constexpr int32_t kSizeOfVariant = 8;

  static const map<DataType, int32_t> DataTypeSizeConvertMap = {
      {DataType::DT_FLOAT16, kSizeOfFloat16},
      {DataType::DT_BF16, kSizeOfFloat16},
      {DataType::DT_FLOAT, sizeof(float)},
      {DataType::DT_DOUBLE, sizeof(double)},
      {DataType::DT_INT8, sizeof(int8_t)},
      {DataType::DT_UINT8, sizeof(uint8_t)},
      {DataType::DT_INT16, sizeof(int16_t)},
      {DataType::DT_UINT16, sizeof(uint16_t)},
      {DataType::DT_INT32, sizeof(int32_t)},
      {DataType::DT_UINT32, sizeof(uint32_t)},
      {DataType::DT_INT64, sizeof(int64_t)},
      {DataType::DT_UINT64, sizeof(uint64_t)},
      {DataType::DT_BOOL, sizeof(bool)},
      {DataType::DT_RESOURCE, kSizeOfResource},
      {DataType::DT_STRING, kSizeOfString},
      {DataType::DT_STRING_REF, kSizeOfStringRef},
      {DataType::DT_COMPLEX32, kSizeOfComplex32},
      {DataType::DT_COMPLEX64, kSizeOfComplex64},
      {DataType::DT_COMPLEX128, kSizeOfComplex128},
      {DataType::DT_QINT16, sizeof(int16_t)},
      {DataType::DT_QUINT16, sizeof(uint16_t)},
      {DataType::DT_QINT8, sizeof(int8_t)},
      {DataType::DT_QUINT8, sizeof(uint8_t)},
      {DataType::DT_QINT32, sizeof(int32_t)},
      {DataType::DT_VARIANT, kSizeOfVariant},
      {DataType::DT_UINT1, sizeof(uint8_t)},
      {DataType::DT_HIFLOAT8, sizeof(int8_t)},
      {DataType::DT_FLOAT8_E4M3FN, sizeof(int8_t)},
      {DataType::DT_FLOAT8_E5M2, sizeof(int8_t)},
      {DataType::DT_FLOAT8_E8M0, sizeof(int8_t)},
      {DataType::DT_FLOAT4_E2M1, sizeof(int8_t)},
      {DataType::DT_FLOAT4_E1M2, sizeof(int8_t)}};

  map<DataType, int32_t>::const_iterator iter = DataTypeSizeConvertMap.find(data_type);
  if (iter != DataTypeSizeConvertMap.end()) {
    return iter->second;
  } else {
    return 0;
  }
}

bool CheckInt64MulOverflow(int64_t a, int64_t b) {
  // Not overflow
  if (a == 0) {
    return false;
  }
  if (b <= (LLONG_MAX / a)) {
    return false;
  }
  return true;
}

bool CheckUint64AddOverflow(uint64_t a, uint64_t b) {
  // Not overflow
  if (b <= (ULLONG_MAX - a)) {
    return false;
  }
  return true;
}

bool CheckInt64AddOverflow(int64_t a, int64_t b) {
  // Not overflow
  if (b <= (LLONG_MAX - a)) {
    return false;
  }
  return true;
}

bool CheckUint32AddOverflow(uint32_t a, uint32_t b) {
  // Not overflow
  if (b <= (UINT32_MAX - a)) {
    return false;
  }
  return true;
}

NodePtr GenGeNode(const string &name, const string &type, int in_count,
                  int out_count, Format format, DataType data_type,
                  vector<int64_t> shape) {
  ComputeGraphPtr compute_graph_ptr =
      shared_ptr<ComputeGraph>(new (nothrow) ComputeGraph("ComputeGraph"));
  AICPU_CHECK_NOTNULL_ERRCODE(compute_graph_ptr, nullptr);
  auto tensor_desc = make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape(move(shape)));
  tensor_desc->SetFormat(format);
  tensor_desc->SetDataType(data_type);

  auto op_desc = make_shared<OpDesc>(name, type);
  AICPU_CHECK_NOTNULL_ERRCODE(op_desc, nullptr);
  for (int i = 0; i < in_count; ++i) {
    (void)op_desc->AddInputDesc(tensor_desc->Clone());
  }
  for (int i = 0; i < out_count; ++i) {
    (void)op_desc->AddOutputDesc(tensor_desc->Clone());
  }
  return compute_graph_ptr->AddNode(op_desc);
}

const string RealPath(const string &path) {
  // PATH_MAX is the system marco，indicate the maximum length for file path
  // pclint check，one param in stack can not exceed 1K bytes
  char resoved_path[PATH_MAX] = {0x00};

  std::string res = "";

  // path not exists or not allowed to read，return nullptr
  // path exists and readable, return the resoved path
  if (realpath(path.c_str(), resoved_path) != nullptr) {
    res = resoved_path;
  } else {
    AICPUE_LOGI("Path %s is not exist.", path.c_str());
  }
  return res;
}

bool ReadBytesFromBinaryFile(const std::string &file_name,
                             std::vector<char> &buffer) {
  AICPU_IF_BOOL_EXEC(file_name.empty(),
      AICPU_REPORT_INNER_ERR_MSG("file name is empty.");
      return false)

  std::string real_path = RealPath(file_name);
  AICPU_IF_BOOL_EXEC(real_path.empty(),
      AICPU_REPORT_INNER_ERR_MSG("Invalid path[%s].", file_name.c_str());
      return false);

  std::ifstream file(real_path.c_str(), std::ios::binary | std::ios::ate);
  AICPU_IF_BOOL_EXEC(!file.is_open(),
      AICPU_REPORT_INNER_ERR_MSG("open file[%s] failed.", file_name.c_str());
      return false);

  std::streamsize size = file.tellg();

  AICPU_IF_BOOL_EXEC(size <= 0, file.close();
      AICPU_REPORT_INNER_ERR_MSG("Empty file[%s].", file_name.c_str());
      return false);

  AICPU_IF_BOOL_EXEC(size > kMaxFileSizeLimit, file.close();
      AICPU_REPORT_INNER_ERR_MSG("File[%s] size[%ld] is out of limit[%d].",
          file_name.c_str(), size, kMaxFileSizeLimit);
      return false);

  file.seekg(0, std::ios::beg);

  buffer.resize(size);
  file.read(&buffer[0], size);
  file.close();
  AICPUE_LOGI("Binary file size is[%ld]", size);
  return true;
}

bool ValidateStr(const std::string &str, const std::string &mode) {
  regex_t reg;
  int cflags = REG_EXTENDED | REG_NOSUB;
  int ret = regcomp(&reg, mode.c_str(), cflags);
  if (ret != 0) {
    return false;
  }

  ret = regexec(&reg, str.c_str(), 0, nullptr, 0);
  if (ret != 0) {
    regfree(&reg);
    return false;
  }

  regfree(&reg);
  return true;
}

/**
 * Get kernel lib name
 * @param op_type op type string
 * @param all_op_info op info
 * @return string lib name
 */
ge::string GetKernelLibNameByOpType(const string &op_type, const std::map<string, OpFullInfo> &all_op_info) {
  auto iter = all_op_info.find(op_type);
  if (iter == all_op_info.end()) {
    AICPUE_LOGW("Op[%s] is not exist in all kernel librarys.", op_type.c_str());
    return "";
  }
  return (iter->second).opKernelLib;
}

/**
 * Check is known shape
 * @param op_desc_ptr op desc ptr
 * @return bool is known or not
 */
bool IsUnknowShape(const OpDescPtr &op_desc_ptr) {
  string op_type = op_desc_ptr->GetType();
  // check inputs
  for (const auto &desc : op_desc_ptr->GetAllInputsDescPtr()) {
    AICPU_IF_BOOL_EXEC(desc == nullptr,
        AICPUE_LOGW("InputsDescPtr is empty");
        return true);
    auto ge_shape = desc->GetShape();
    for (const auto &dim : ge_shape.GetDims()) {
      if (dim == UNKNOWN_DIM || dim == UNKNOWN_DIM_NUM) {
        AICPUE_LOGI("Op type[%s]: shape is [%ld], which is unknown.", op_type.c_str(),
                    dim);
        return true;
      }
    }
  }
  // check outputs
  for (const auto &desc : op_desc_ptr->GetAllOutputsDescPtr()) {
    AICPU_IF_BOOL_EXEC(desc == nullptr, AICPUE_LOGW("OutputsDescPtr is empty");
                       return true);
    auto ge_shape = desc->GetShape();
    for (const auto &dim : ge_shape.GetDims()) {
      if (dim == UNKNOWN_DIM || dim == UNKNOWN_DIM_NUM) {
        AICPUE_LOGI("Op type[%s]: shape is %ld, which is unknown.", op_type.c_str(),
                    dim);
        return true;
      }
    }
  }
  return false;
}

// calculate and set outputs size
Status SetOutPutsSize(shared_ptr<OpDesc> &op_desc_ptr) {
  AICPU_CHECK_NOTNULL_ERRCODE(op_desc_ptr, INPUT_PARAM_NULL)
  bool is_no_tiling = IsNotiling(op_desc_ptr);
  int32_t shape_type = 0;
  if (AttrUtils::HasAttr(op_desc_ptr, kAttrNameUnknownShape)) {
    // unknow shape
    CHECK_RES_BOOL(
        AttrUtils::GetInt(op_desc_ptr, ATTR_NAME_UNKNOWN_SHAPE_TYPE, shape_type),
        INVOKE_GRAPH_ITF_FAILED,
        AICPU_REPORT_INNER_ERR_MSG(
            "Call ge::AttrUtils::GetInt failed to get attr[%s], op[%s].",
            ATTR_NAME_UNKNOWN_SHAPE_TYPE.c_str(), op_desc_ptr->GetName().c_str()))
    if (shape_type == DEPEND_COMPUTE && !is_no_tiling) {
      // unknow type 4, output is ResultSummary
      AICPU_CHECK_RES_WITH_LOG(SetOutSizeForSummary(op_desc_ptr),
          "Call SetOutSizeForSummary function failed, op[%s].",
          op_desc_ptr->GetName().c_str())
      return SUCCESS;
    }
  }

  vector<int64_t> output_memsize_vec;
  vector<int64_t> output_alignment_vec;
  size_t output_size = op_desc_ptr->GetOutputsSize();
  for (size_t i = 0; i < output_size; i++) {
    int64_t output_mem_size = 0;
    GeTensorDesc output_tensor = op_desc_ptr->GetOutputDesc(i);
    DataType output_data_type = output_tensor.GetDataType();
    vector<int64_t> dims;
    (void)AttrUtils::GetListInt(output_tensor, AICPU_ATTR_NAME_TENSOR_MAX_SHAPE, dims);
    if (!dims.empty()) {
      aicpu::State state = GetTotalSizeByDimsAndType(output_data_type, dims, output_mem_size);
      AICPU_IF_BOOL_EXEC(state.state != ge::SUCCESS,
                         AICPU_REPORT_INNER_ERR_MSG("Overflow occurred when calculate total "
                                                  "bytes of output[%zu], %s, op[%s]",
                                                  i, state.msg.c_str(), op_desc_ptr->GetName().c_str());
                         return state.state)
      vector<pair<int64_t, int64_t>> shape_range;
      for (size_t dim_index = 0; dim_index < dims.size(); ++dim_index) {
        shape_range.emplace_back(make_pair(dims[dim_index], dims[dim_index]));
      }
      output_tensor.SetShapeRange(shape_range);
    } else if (shape_type == DEPEND_SHAPE_RANGE) {
      vector<pair<int64_t, int64_t>> shape_range;
      // try get shape range from tensor desc
      AICPU_CHECK_RES_WITH_LOG(output_tensor.GetShapeRange(shape_range),
          "Call GetShapeRange function failed for %zuth output, op[%s].",
          i, op_desc_ptr->GetName().c_str())
      if (!shape_range.empty()) {
        // unknow type3, calc outputs size by ShapeRange
        AICPU_CHECK_RES_WITH_LOG(
            GetOutSizeByShapeRange(output_data_type, shape_range, output_mem_size),
            "Call GetOutSizeByShapeRange function failed for %zuth output, op[%s].",
            i, op_desc_ptr->GetName().c_str())
      } else {
        const GeShape &output_shape = output_tensor.GetShape();
        aicpu::State state = GetTotalSizeByShapeAndType(
            output_data_type, output_shape, output_mem_size);
        AICPU_IF_BOOL_EXEC(state.state != ge::SUCCESS,
            AICPU_REPORT_INNER_ERR_MSG("Overflow occurred when calculate total "
                "bytes of output[%zu], %s, op[%s]", i, state.msg.c_str(),
                op_desc_ptr->GetName().c_str());
            return state.state)
      }
    } else {
      const GeShape &output_shape = output_tensor.GetShape();
      aicpu::State state = GetTotalSizeByShapeAndType(
          output_data_type, output_shape, output_mem_size);
      AICPU_IF_BOOL_EXEC(state.state != ge::SUCCESS,
          AICPU_REPORT_INNER_ERR_MSG("Overflow occurred when calculate total "
              "bytes of output[%zu], %s, op[%s]", i, state.msg.c_str(),
              op_desc_ptr->GetName().c_str());
          return state.state)
    }

    AICPUE_LOGI("After the output [%zu]'s calc, the total size is [%ld]", i,
                output_mem_size);
    if (output_mem_size > LONG_MAX) {
      AICPU_REPORT_INNER_ERR_MSG("Overflow occured when calculate %zuth output "
          "total bytes, data type[%s], shape[%s], op[%s].", i,
          ge::TypeUtils::DataTypeToSerialString(output_data_type).c_str(),
          DebugString(output_tensor.GetShape().GetDims()).c_str(),
          op_desc_ptr->GetName().c_str());
      return DATA_OVERFLOW;
    }
    TensorUtils::SetSize(output_tensor, output_mem_size);
    output_memsize_vec.emplace_back(output_mem_size);
    output_alignment_vec.emplace_back(kAlignmentValue);
    AICPU_CHECK_RES(op_desc_ptr->UpdateOutputDesc(i, output_tensor));
  }

  CHECK_RES_BOOL(
      AttrUtils::SetListInt(op_desc_ptr, kOutputMemsizeVector, output_memsize_vec),
      INVOKE_GRAPH_ITF_FAILED,
      AICPU_REPORT_INNER_ERR_MSG(
          "Call ge::AttrUtils::SetListInt failed to set attr[%s], op[%s]",
          kOutputMemsizeVector.c_str(), op_desc_ptr->GetName().c_str()))
  CHECK_RES_BOOL(
      AttrUtils::SetListInt(op_desc_ptr, kOutputAlignmentVector,
                            output_alignment_vec),
      INVOKE_GRAPH_ITF_FAILED,
      AICPU_REPORT_INNER_ERR_MSG(
          "Call ge::AttrUtils::SetListInt failed to set attr[%s], op[%s]",
          kOutputAlignmentVector.c_str(), op_desc_ptr->GetName().c_str()))
  return SUCCESS;
}

// Set outputs size for unknow type 4 which output is result summary
Status SetOutSizeForSummary(shared_ptr<OpDesc> &op_desc_ptr) {
  vector<int64_t> output_memsize_vec;
  vector<int64_t> output_alignment_vec;

  size_t output_size = op_desc_ptr->GetOutputsSize();
  for (size_t i = 0; i < output_size; i++) {
    // output is ResultSummary, 4 means number of fields for ResultSummary
    int64_t output_mem_size = 4 * sizeof(uint64_t);
    GeTensorDesc output_tensor = op_desc_ptr->GetOutputDesc(i);
    TensorUtils::SetSize(output_tensor, static_cast<uint32_t>(output_mem_size));
    (void)output_memsize_vec.emplace_back(output_mem_size);
    (void)output_alignment_vec.emplace_back(kAlignmentValue);
    AICPU_CHECK_RES(op_desc_ptr->UpdateOutputDesc(i, output_tensor));
  }
  CHECK_RES_BOOL(
      AttrUtils::SetListInt(op_desc_ptr, kOutputMemsizeVector, output_memsize_vec),
      INVOKE_GRAPH_ITF_FAILED,
      AICPU_REPORT_INNER_ERR_MSG(
          "Call ge::AttrUtils::SetListInt failed to set attr[%s], op[%s]",
          kOutputMemsizeVector.c_str(), op_desc_ptr->GetName().c_str()))
  CHECK_RES_BOOL(AttrUtils::SetListInt(op_desc_ptr, kOutputAlignmentVector,
                                       output_alignment_vec),
      INVOKE_GRAPH_ITF_FAILED,
      AICPU_REPORT_INNER_ERR_MSG(
          "Call ge::AttrUtils::SetListInt failed to set attr[%s], op[%s]",
          kOutputAlignmentVector.c_str(), op_desc_ptr->GetName().c_str()))

  AICPUE_LOGI(
      "Set output size with ResultSummary for unknow type 4 ok, op: %s.",
      op_desc_ptr->GetName().c_str());
  return SUCCESS;
}

State GetTotalSizeByShapeAndType(const DataType &data_type, const GeShape &ge_shape,
                                 int64_t &total_size) {
  vector<int64_t> dims = ge_shape.GetDims();
  return GetTotalSizeByDimsAndType(data_type, dims, total_size);
}

State GetTotalSizeByDimsAndType(const ge::DataType &data_type,
                                const vector<int64_t> &dims,
                                int64_t &total_size) {
  int32_t data_size = GetDataTypeSize(data_type);
  AICPU_CHECK_GREATER_THAN_ZERO(data_size, DATA_TYPE_UNDEFILED,
      "Invalid data type[%s].",
      ge::TypeUtils::DataTypeToSerialString(data_type).c_str())
  int64_t total_num = 1;
  size_t dim_size = dims.size();
  for (size_t i = 0; i < dim_size; i++) {
    if (dims[i] == UNKNOWN_DIM || dims[i] == UNKNOWN_DIM_NUM) {
      AICPUE_LOGW("Unknow output shape [%ld], skip it.", dims[i]);
      break;
    }
    if (dims[i] < 0) {
      string err_msg =
          Stringcat("dim[", i, "] is invalid, shape is ", DebugString(dims));
      aicpu::State state(GE_SHAPE_SIZE_INVAILD, err_msg);
      return state;
    }
    CHECK_INT64_MUL_OVERFLOW(total_num, dims[i], aicpu::State(DATA_OVERFLOW),
        "Overflow when calculate tensor total bytes, shape[%s], data type[%s],",
        DebugString(dims).c_str(),
        ge::TypeUtils::DataTypeToSerialString(data_type).c_str())
    total_num *= dims[i];
  }

  CHECK_INT64_MUL_OVERFLOW(total_num, data_size, aicpu::State(DATA_OVERFLOW),
      "Overflow when calculate tensor total bytes, shape[%s], data type[%s],",
      DebugString(dims).c_str(),
      ge::TypeUtils::DataTypeToSerialString(data_type).c_str())
  total_size = total_num * data_size;
  return aicpu::State(SUCCESS);
}

// calculate output size for unknow type 3 which output shape is range
Status GetOutSizeByShapeRange(const DataType &data_type, const vector<pair<int64_t, int64_t>> &shape_range,
                              int64_t &total_size) {
  int32_t data_size = GetDataTypeSize(data_type);
  AICPU_CHECK_GREATER_THAN_ZERO(data_size, DATA_TYPE_UNDEFILED,
      "Invalid data type[%s].",
      ge::TypeUtils::DataTypeToSerialString(data_type).c_str())

  int64_t total_num = 1;
  for (size_t index = 0; index < shape_range.size(); ++index) {
    const auto &dim_item = shape_range[index];
    // unknow shape
    if (dim_item.second == UNKNOWN_DIM || dim_item.second == UNKNOWN_DIM_NUM) {
      AICPUE_LOGW("Unknow output shape [%ld], skip it.", dim_item.second);
      break;
    }
    AICPU_CHECK_SHAPE_RANGE_GREATER_THAN_OR_EQUAL_TO_ZERO(dim_item.second, GE_SHAPE_SIZE_INVAILD,
        "Invalid value[%ld] of dim[%zu].", dim_item.second, index)
    CHECK_INT64_MUL_OVERFLOW(total_num, dim_item.second, DATA_OVERFLOW,
        "Overflow when calculate tensor shape range total bytes")
    total_num *= dim_item.second;
  }
  CHECK_INT64_MUL_OVERFLOW(total_num, data_size, DATA_OVERFLOW,
      "Overflow when calculate tensor shape range total bytes")
  total_size = total_num * data_size;
  AICPUE_LOGI("Calc output size by shape range ok, total_size[%ld].", total_size);
  return SUCCESS;
}

std::vector<std::string> SplitString(const std::string str, const std::string split)
{
  size_t curIndex = 0;
  size_t nextSplit = 0;
  std::vector<std::string> strs;
  while (curIndex < str.size()) {
    nextSplit = str.find(split, curIndex);
    if (nextSplit == std::string::npos) {
      nextSplit = str.size();
    }
    strs.emplace_back(str.substr(curIndex, nextSplit));
    curIndex = nextSplit + split.size();
  }
  return strs;
}

void GetThreadTensorShape(const vector<vector<vector<ffts::DimRange>>> &shape_range,
                          vector<vector<vector<int64_t>>> &shape) {
  for (size_t i = 0; i < shape_range.size(); i++) {
    std::vector<std::vector<int64_t>> temp_input;
    for (size_t j = 0; j < shape_range[i].size(); j++) {
      std::vector<int64_t> dim;
      for (size_t k = 0; k < shape_range[i][j].size(); k++) {
        dim.push_back(shape_range[i][j][k].higher - shape_range[i][j][k].lower);
      }
      temp_input.push_back(dim);
    }
    shape.push_back(temp_input);
  }
  return;
}

ge::Status GetOriginalType(const ge::OpDescPtr &op_desc_ptr, std::string &type) {
  AICPU_CHECK_NOTNULL(op_desc_ptr)
  type = op_desc_ptr->GetType();
  AICPU_IF_BOOL_EXEC(type != kFrameworkOp, return SUCCESS)
  const std::string *type_ptr = ge::AttrUtils::GetStr(op_desc_ptr, kOriginalType);
  if (type_ptr == nullptr) {
    AICPU_REPORT_INNER_ERR_MSG("Get Attr:%s fail from op:%s(%s)",
                             kOriginalType.c_str(), op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str());
    AICPUE_LOGE("[Get][Attr] %s fail from op:%s(%s)",
                kOriginalType.c_str(), op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str());
    return GET_ATTR_FAILED;
  }
  type = *type_ptr;
  AICPUE_LOGD("Get FrameWorkOp original type [%s]", type.c_str());
  return SUCCESS;
}

bool IsNotiling(const ge::OpDescPtr &op_desc_ptr) {
  bool is_no_tiling = false;
  (void) ge::AttrUtils::GetBool(op_desc_ptr, kOpNoTiling, is_no_tiling);
  return is_no_tiling;
}

bool IsPathExist(const std::string &path) {
  std::string real_path = path;
  real_path = RealPath(real_path);
  if (real_path.empty()) {
    return false;
  }
  return true;
}

string Trim(const string &source, const char delims) {
  std::string result(source);
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

void SplitLine(const string &str, const string &pattern,
               std::vector<string> &result) {
  // Easy to intercept the last piece of data
  string strs = str + pattern;

  size_t pos = strs.find(pattern);
  size_t size = strs.size();

  while (pos != string::npos) {
    string x = strs.substr(0, pos);
    if (!x.empty()) {
      result.push_back(x);
    }
    strs = strs.substr(pos + pattern.length(), size);
    pos = strs.find(pattern);
  }
}

bool ReadConfigFile(const string &file_path, std::vector<string> &result) {
  std::string real_path = RealPath(file_path);
  AICPUE_LOGI("The real path of config.ini is %s.", real_path.c_str());
  std::ifstream ifs(real_path);
  if (!ifs.is_open()) {
    AICPUE_LOGI("Custom config.ini is not exist.");
    return false;
  }
  std::string line;
  while (std::getline(ifs, line)) {
    line = Trim(Trim(line, '\r'), '\n');
    if (line.empty() || line.find('#') == 0) {
      continue;
    }
    if (line.find("load_priority") == std::string::npos) {
      continue;
    }
    size_t pos_of_equal = line.find('=');
    if (pos_of_equal == std::string::npos) {
      ifs.close();
      AICPU_REPORT_INNER_ERR_MSG("[ReadConfigFile] Config format is error.");
      return false;
    }
    std::string value = line.substr(pos_of_equal + 1);
    SplitLine(value, kConfigItemSeparator, result);
    AICPUE_LOGI("Get number %zu custom path.", result.size());
    break;
  }
  ifs.close();
  AICPUE_LOGI("End to load opp custom config.");
  return true;
}
}
