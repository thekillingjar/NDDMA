/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/helper/model_parser_base.h"

#include <fstream>
#include <string>
#include <cstring>
#include <array>

#include "securec.h"
#include "framework/common/helper/model_helper.h"
#include "mmpa/mmpa_api.h"
#include "graph/def_types.h"
#include "common/math/math_util.h"
#include "base/err_msg.h"
#include "graph_metadef/common/ge_common/util.h"

namespace {
constexpr int64_t kMaxFileSizeLimit = INT64_MAX;
constexpr size_t kMaxErrorStringLen = 128U;
constexpr uint32_t kStatiOmFileModelNum = 1U;
constexpr uint8_t kDynamicOmFlag = 1U;
}  // namespace

namespace ge {
Status ModelParserBase::LoadFromFile(const char_t *const model_path, const int32_t priority, ModelData &model_data) {
  const std::string real_path = RealPath(model_path);
  if (real_path.empty()) {
    std::array<char_t, kMaxErrorStringLen + 1U> err_buf = {};
    const auto err_msg = mmGetErrorFormatMessage(mmGetErrorCode(), err_buf.data(), kMaxErrorStringLen);
    std::string reason = "[Error " + std::to_string(mmGetErrorCode()) + "] " + err_msg;
    (void)REPORT_PREDEFINED_ERR_MSG("E13000", std::vector<const char *>({"path", "errmsg"}),
                       std::vector<const char *>({model_path, reason.c_str()}));
    GELOGE(ACL_ERROR_GE_EXEC_MODEL_PATH_INVALID, "[Check][Param]Model file path %s is invalid",
           model_path);
    return ACL_ERROR_GE_EXEC_MODEL_PATH_INVALID;
  }

  const int64_t length = GetFileLength(model_path);
  if (length == -1) {
    (void)REPORT_PREDEFINED_ERR_MSG("E13015", std::vector<const char *>({"file", "size", "maxsize"}),
      std::vector<const char *>({model_path, std::to_string(length).c_str(), std::to_string(kMaxFileSizeLimit).c_str()}));
    GELOGE(ACL_ERROR_GE_EXEC_MODEL_PATH_INVALID, "[Check][Param]File [%s] size %lld is out if range (0, %lld].",
           model_path, length, kMaxFileSizeLimit);
    return ACL_ERROR_GE_EXEC_MODEL_PATH_INVALID;
  }

  std::ifstream fs(real_path.c_str(), std::ifstream::binary);
  if (!fs.is_open()) {
    std::array<char_t, kMaxErrorStringLen + 1U> err_buf = {};
    const auto err_msg = mmGetErrorFormatMessage(mmGetErrorCode(), &err_buf[0], kMaxErrorStringLen);
    GELOGE(ACL_ERROR_GE_EXEC_MODEL_PATH_INVALID, "[Open][File]Failed, file %s, error %s",
           model_path, err_msg);
    REPORT_INNER_ERR_MSG("E19999", "Open file %s failed, error %s", model_path, err_msg);
    return ACL_ERROR_GE_EXEC_MODEL_PATH_INVALID;
  }

  // get length of file:
  (void)fs.seekg(0, std::ifstream::end);
  const uint64_t len = static_cast<uint64_t>(fs.tellg());

  GE_CHECK_GE(len, 1U);

  (void)fs.seekg(0, std::ifstream::beg);
  char_t *const data = new (std::nothrow) char_t[len];
  if (data == nullptr) {
    GELOGE(ACL_ERROR_GE_MEMORY_ALLOCATION, "[Load][ModelFromFile]Failed, "
           "bad memory allocation occur(need %" PRIu64 "), file %s", len, model_path);
    REPORT_INNER_ERR_MSG("E19999", "Load model from file %s failed, "
                      "bad memory allocation occur(need %" PRIu64 ")", model_path, len);
    return ACL_ERROR_GE_MEMORY_ALLOCATION;
  }

  // read data as a block:
  (void)fs.read(data, static_cast<std::streamsize>(len));
  const ModelHelper model_helper;
  (void)model_helper.GetBaseNameFromFileName(model_path, model_data.om_name);
  // Set the model data parameter
  model_data.model_data = data;
  model_data.model_len = len;
  model_data.priority = priority;

  return SUCCESS;
}

Status ModelParserBase::ParseModelContent(const ModelData &model, uint8_t *&model_data, uint64_t &model_len) {
  // Parameter validity check
  GE_CHECK_NOTNULL(model.model_data);

  // Model length too small
  GE_CHK_BOOL_EXEC(model.model_len >= sizeof(ModelFileHeader),
      std::string reason = "Invalid om file. The model data size " + std::to_string(model.model_len) +
                           " is smaller than " + std::to_string(sizeof(ModelFileHeader)) + ".";
      (void)REPORT_PREDEFINED_ERR_MSG("E10001", std::vector<const char *>({"parameter", "value", "reason"}),
                                      std::vector<const char *>({"om", model.om_name.c_str(), reason.c_str()}));
      GELOGE(ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID,
             "[Check][Param] Invalid model. Model data size %" PRIu64 " must be "
             "greater than or equal to %zu.",
             model.model_len, sizeof(ModelFileHeader));
      return ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID);
  // Get file header
  const auto file_header = static_cast<ModelFileHeader *>(model.model_data);
  // Determine whether the file length and magic number match
  model_len =
      (file_header->model_length == 0UL) ? static_cast<uint64_t>(file_header->length) : file_header->model_length;
  GE_CHK_BOOL_EXEC((model_len == (model.model_len - sizeof(ModelFileHeader))) &&
                   (file_header->magic == MODEL_FILE_MAGIC_NUM),
                   (void)REPORT_PREDEFINED_ERR_MSG("E10001", std::vector<const char_t *>({"parameter", "value", "reason"}),
                                      std::vector<const char_t *>({"om", model.om_name.c_str(), "Invalid om file."}));
                   GELOGE(ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID,
                          "[Check][Param] Invalid model, file_header->(model)length[%" PRIu64 "]"
                          " + sizeof(ModelFileHeader)[%zu] != model->model_len[%" PRIu64 "]"
                          "|| MODEL_FILE_MAGIC_NUM[%u] != file_header->magic[%u]",
                          model_len, sizeof(ModelFileHeader), model.model_len,
                          MODEL_FILE_MAGIC_NUM, file_header->magic);
                   return ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID);

  // Get data address
  model_data = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(model.model_data) + sizeof(ModelFileHeader)));
  GELOGD("Model_len is %" PRIu64 ", model_file_head_len is %zu.", model_len, sizeof(ModelFileHeader));
  return SUCCESS;
}

bool ModelParserBase::IsDynamicModel(const ModelFileHeader &file_header) {
  return (file_header.version >= ge::MODEL_VERSION) &&
         ((file_header.model_num > kStatiOmFileModelNum) || (file_header.is_unknow_model == kDynamicOmFlag));
}

static Status GenInOutTensorDesc(const uint8_t * const data, size_t &offset,
                                 const ModelTensorDescBaseInfo &tensor_base_info,
                                 ModelInOutTensorDesc &desc, const size_t size) {
  std::string name(PtrToPtr<void, const char>(ValueToPtr(PtrToValue(data) + static_cast<uint64_t>(offset))),
                    static_cast<size_t>(tensor_base_info.name_len));
  desc.name = std::move(name);
  GE_ASSERT_SUCCESS(CheckUint64AddOverflow(offset, static_cast<uint64_t>(tensor_base_info.name_len)),
                 "[Check][Param] offset:%lu is beyond the UINT64_MAX, info size %u",
                 offset, tensor_base_info.name_len);
  GE_CHECK_LE((offset + tensor_base_info.name_len), size);
  offset += tensor_base_info.name_len;
  GE_ASSERT_SUCCESS(CheckUint64AddOverflow(offset, static_cast<uint64_t>(tensor_base_info.dims_len)),
                 "[Check][Param] offset:%lu is beyond the UINT64_MAX, info size %u",
                 offset, tensor_base_info.dims_len);
  GE_CHECK_LE((offset + tensor_base_info.dims_len), size);
  for (uint32_t i = 0U; i < tensor_base_info.dims_len / sizeof(int64_t); ++i) {
    (void)i;
    int64_t dim = *PtrToPtr<void, const int64_t>(ValueToPtr(PtrToValue(data) + static_cast<uint64_t>(offset)));
    offset += sizeof(int64_t);
    (void)desc.dims.emplace_back(dim);
  }
  GE_ASSERT_SUCCESS(CheckUint64AddOverflow(offset, static_cast<uint64_t>(tensor_base_info.dimsV2_len)),
                 "[Check][Param] offset:%lu is beyond the UINT64_MAX, info size %u",
                 offset, tensor_base_info.dimsV2_len);
  GE_CHECK_LE((offset + tensor_base_info.dimsV2_len), size);
  for (uint32_t i = 0U; i < tensor_base_info.dimsV2_len / sizeof(int64_t); ++i) {
    (void)i;
    int64_t dim = *PtrToPtr<void, const int64_t>(ValueToPtr(PtrToValue(data) + static_cast<uint64_t>(offset)));
    offset += sizeof(int64_t);
    (void)desc.dimsV2.emplace_back(dim);
  }
  GE_ASSERT_SUCCESS(CheckUint64AddOverflow(offset, static_cast<uint64_t>(tensor_base_info.shape_range_len)),
                 "[Check][Param] offset:%lu is beyond the UINT64_MAX, info size %u",
                 offset, tensor_base_info.shape_range_len);
  GE_CHECK_LE((offset + tensor_base_info.shape_range_len), size);
  constexpr size_t shape_range_el_len = sizeof(std::pair<int64_t, int64_t>);
  for (uint32_t i = 0U; i < tensor_base_info.shape_range_len / shape_range_el_len; ++i) {
    (void)i;
    int64_t range_left = *PtrToPtr<void, const int64_t>(ValueToPtr(PtrToValue(data) + static_cast<uint64_t>(offset)));
    offset += sizeof(int64_t);
    int64_t range_right =
                    *PtrToPtr<void, const int64_t>(ValueToPtr(PtrToValue(data) + static_cast<uint64_t>(offset)));
    offset += sizeof(int64_t);
    (void)desc.shape_ranges.emplace_back(std::make_pair(range_left, range_right));
  }
  return SUCCESS;
}

static Status GetModelInOutDesc(const uint8_t * const data, const size_t size, const bool is_input,
                                ModelInOutInfo &info) {
  size_t offset = 0U;
  GE_ASSERT_SUCCESS(CheckUint64AddOverflow(offset, sizeof(uint32_t)),
                 "[Check][Param] offset:%lu is beyond the UINT64_MAX", offset);
  GE_CHECK_LE((offset + sizeof(uint32_t)), size);
  const uint32_t desc_num = *PtrToPtr<const uint8_t, const uint32_t>(data);
  GELOGD("get desc num is %u, is_input is %d.", desc_num, static_cast<int32_t>(is_input));
  offset += sizeof(uint32_t);
  ModelTensorDescBaseInfo tensor_base_info;
  for (uint32_t index = 0U; index < desc_num; ++index) {
    GE_ASSERT_SUCCESS(CheckUint64AddOverflow(offset, sizeof(ModelTensorDescBaseInfo)),
                   "[Check][Param] offset:%lu is beyond the UINT64_MAX, info size %lu",
                   offset, sizeof(ModelTensorDescBaseInfo));
    GE_CHECK_LE((offset + sizeof(ModelTensorDescBaseInfo)), size);
    tensor_base_info =
        *PtrToPtr<void, const ModelTensorDescBaseInfo>(ValueToPtr(PtrToValue(data) + static_cast<uint64_t>(offset)));
    GELOGD("current index is %u, size is %zu, fomat is %d, dt id %d, name len si %u,"
           "dims len is %u, dimsV2 is %u, shape_range is %u.",
           index, tensor_base_info.size, tensor_base_info.format, tensor_base_info.dt, tensor_base_info.name_len,
           tensor_base_info.dims_len, tensor_base_info.dimsV2_len, tensor_base_info.shape_range_len);
    offset += sizeof(ModelTensorDescBaseInfo);
    ModelInOutTensorDesc desc;
    desc.size = tensor_base_info.size;
    desc.dataType = tensor_base_info.dt;
    desc.format = tensor_base_info.format;
    GE_ASSERT_SUCCESS(GenInOutTensorDesc(data, offset, tensor_base_info, desc, size),
                       "GenInOutTensorDesc failed");
    if (is_input) {
      (void)info.input_desc.emplace_back(std::move(desc));
    } else {
      (void)info.output_desc.emplace_back(std::move(desc));
    }
  }
  return SUCCESS;
}

Status ModelParserBase::GetModelInputDesc(const uint8_t * const data, const size_t size, ModelInOutInfo &info) {
  return GetModelInOutDesc(data, size, true, info);
}

Status ModelParserBase::GetModelOutputDesc(const uint8_t * const data, const size_t size, ModelInOutInfo &info) {
  return GetModelInOutDesc(data, size, false, info);
}

Status ModelParserBase::GetDynamicBatch(const uint8_t * const data, const size_t size, ModelInOutInfo &info) {
  size_t offset = 0U;
  GE_ASSERT_SUCCESS(CheckUint64AddOverflow(offset, sizeof(uint32_t)),
                   "[Check][Param] offset:%lu is beyond the UINT64_MAX, info size %lu",
                   offset, sizeof(uint32_t));
  GE_CHECK_LE((offset + sizeof(uint32_t)), size);
  const uint32_t num = *PtrToPtr<const uint8_t, uint32_t>(data);
  GELOGD("get num is %u", num);
  offset += sizeof(uint32_t);
  GE_ASSERT_SUCCESS(CheckUint64AddOverflow(offset, sizeof(uint64_t) * num),
                 "[Check][Param] offset:%lu is beyond the UINT64_MAX, info size %u", offset, num);
  GE_CHECK_LE((offset + sizeof(uint64_t) * num), size);
  for (uint32_t i = 0U; i < num; ++i) {
    const uint64_t batch =
                      *PtrToPtr<void, const uint64_t>(ValueToPtr(PtrToValue(data) + static_cast<uint64_t>(offset)));
    GELOGD("current i is %u, batch is %lu", i, batch);
    (void)info.dynamic_batch.emplace_back(batch);
    offset += sizeof(uint64_t);
  }
  return SUCCESS;
}

static Status GetListListIntInfo(const uint8_t * const data, const size_t size,
                                 std::vector<std::vector<uint64_t>> &list_list_val) {
  size_t offset = 0U;
  GE_ASSERT_SUCCESS(CheckUint64AddOverflow(offset, sizeof(uint32_t)),
                 "[Check][Param] offset:%lu is beyond the UINT64_MAX, info size %lu",
                 offset, sizeof(uint32_t));
  GE_CHECK_LE((offset + sizeof(uint32_t)), size);
  const uint32_t num = *PtrToPtr<const uint8_t, const uint32_t>(data);
  GELOGD("get num is %u", num);
  offset += sizeof(uint32_t);
  std::vector<uint32_t> second_vec;
  GE_ASSERT_SUCCESS(CheckUint64AddOverflow(offset, sizeof(uint32_t) * num),
                 "[Check][Param] offset:%lu is beyond the UINT64_MAX, info size %u",
                 offset, num);
  GE_CHECK_LE((offset + sizeof(uint32_t) * num), size);
  for (uint32_t i = 0U; i < num; ++i) {
    (void)i;
    const uint32_t part_vec_num =
                        *PtrToPtr<void, const uint32_t>(ValueToPtr(PtrToValue(data) + static_cast<uint64_t>(offset)));
    (void)second_vec.emplace_back(part_vec_num);
    offset += sizeof(uint32_t);
  }
  for (uint32_t i = 0U; i < num; ++i) {
    std::vector<uint64_t> tmp_vec;
    for (uint32_t j = 0U; j < second_vec[static_cast<size_t>(i)]; ++j) {
      (void)j;
      GE_ASSERT_SUCCESS(CheckUint64AddOverflow(offset, sizeof(uint64_t)),
                     "[Check][Param] offset:%lu is beyond the UINT64_MAX, info size %lu",
                     offset, sizeof(uint64_t));
      GE_CHECK_LE((offset + sizeof(uint64_t)), size);
      uint64_t dim = *PtrToPtr<void, const uint64_t>(ValueToPtr(PtrToValue(data) + static_cast<uint64_t>(offset)));
      offset += sizeof(uint64_t);
      (void)tmp_vec.emplace_back(dim);
    }
    (void)list_list_val.emplace_back(std::move(tmp_vec));
  }
  return SUCCESS;
}

Status ModelParserBase::GetDynamicHW(const uint8_t * const data, const size_t size, ModelInOutInfo &info) {
  return GetListListIntInfo(data, size, info.dynamic_hw);
}

Status ModelParserBase::GetDynamicDims(const uint8_t * const data, const size_t size, ModelInOutInfo &info) {
  return GetListListIntInfo(data, size, info.dynamic_dims);
}

static Status GetListStrInfo(const uint8_t * const data, const size_t size, std::vector<std::string> &list_str) {
  size_t offset = 0U;
  GE_ASSERT_SUCCESS(CheckUint64AddOverflow(offset, sizeof(uint32_t)),
                 "[Check][Param] offset:%lu is beyond the UINT64_MAX, info size %lu",
                 offset, sizeof(uint32_t));
  GE_CHECK_LE((offset + sizeof(uint32_t)), size);
  const uint32_t num = *PtrToPtr<uint8_t, uint32_t>(data);
  GELOGD("get num is %u", num);
  offset += sizeof(uint32_t);
  std::vector<uint32_t> second_vec;
  GE_ASSERT_SUCCESS(CheckUint64AddOverflow(offset, sizeof(uint32_t) * num),
                 "[Check][Param] offset:%lu is beyond the UINT64_MAX, info size %u",
                 offset, num);
  GE_CHECK_LE((offset + sizeof(uint32_t)) * num, size);

  for (size_t i = 0U; i < static_cast<size_t>(num); ++i) {
    const uint32_t part_vec_num =
                        *PtrToPtr<void, const uint32_t>(ValueToPtr(PtrToValue(data) + static_cast<uint64_t>(offset)));
    (void)second_vec.emplace_back(part_vec_num);
    GELOGD("current i is %zu, strlen is %u", i, part_vec_num);
    offset += sizeof(uint32_t);
  }
  for (size_t i = 0U; i < static_cast<size_t>(num); ++i) {
    GE_ASSERT_SUCCESS(CheckUint64AddOverflow(offset, static_cast<uint64_t>(second_vec[i])),
                 "[Check][Param] offset:%lu is beyond the UINT64_MAX, info size %u",
                 offset, second_vec[i]);
    GE_CHECK_LE((offset + second_vec[i]), size);
    std::string tmp_vec(PtrToPtr<void, const char>(ValueToPtr(PtrToValue(data) + static_cast<uint64_t>(offset))),
                        static_cast<size_t>(second_vec[i]));
    GELOGD("current i is %zu, str is %s", i, tmp_vec.c_str());
    offset += second_vec[i];
    (void)list_str.emplace_back(std::move(tmp_vec));
  }
  return SUCCESS;
}

Status ModelParserBase::GetDataNameOrder(const uint8_t * const data, const size_t size, ModelInOutInfo &info) {
  return GetListStrInfo(data, size, info.data_name_order);
}

Status ModelParserBase::GetDynamicOutShape(const uint8_t * const data, const size_t size, ModelInOutInfo &info) {
  return GetListStrInfo(data, size, info.dynamic_output_shape);
}
}  // namespace ge
