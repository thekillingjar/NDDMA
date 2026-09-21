/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "message2operator.h"

#include <vector>

#include "common/convert/pb2json.h"
#include "common/util.h"
#include "framework/common/debug/ge_log.h"
#include "base/err_msg.h"

namespace ge {
namespace {
const int kMaxParseDepth = 5;
const uint32_t kInteval = 2;
}  // namespace

Status Message2Operator::ParseOperatorAttrs(const google::protobuf::Message *message, int depth, ge::Operator &ops) {
  GE_CHECK_NOTNULL(message);
  if (depth > kMaxParseDepth) {
    REPORT_INNER_ERR_MSG("E19999", "Message depth:%d can not exceed %d.", depth, kMaxParseDepth);
    GELOGE(FAILED, "[Check][Param]Message depth can not exceed %d.", kMaxParseDepth);
    return FAILED;
  }

  const google::protobuf::Reflection *reflection = message->GetReflection();
  GE_CHECK_NOTNULL(reflection);
  std::vector<const google::protobuf::FieldDescriptor *> field_desc;
  reflection->ListFields(*message, &field_desc);

  for (auto &field : field_desc) {
    GE_CHECK_NOTNULL(field);
    if (field->is_repeated()) {
      if (ParseRepeatedField(reflection, message, field, ops) != SUCCESS) {
        GELOGE(FAILED, "[Parse][RepeatedField] %s failed.", field->name().c_str());
        return FAILED;
      }
    } else {
      if (ParseField(reflection, message, field, depth, ops) != SUCCESS) {
        GELOGE(FAILED, "[Parse][Field] %s failed.", field->name().c_str());
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

Status Message2Operator::ParseField(const google::protobuf::Reflection *reflection,
                                    const google::protobuf::Message *message,
                                    const google::protobuf::FieldDescriptor *field, int depth, ge::Operator &ops) {
  GELOGD("Start to parse field: %s.", field->name().c_str());
  switch (field->cpp_type()) {
#define CASE_FIELD_TYPE(cpptype, method, valuetype, logtype)                    \
    case google::protobuf::FieldDescriptor::CPPTYPE_##cpptype: {                \
      valuetype value = reflection->Get##method(*message, field);               \
      GELOGD("Parse result(%s : %" #logtype ")", field->name().c_str(), value); \
      (void)ops.SetAttr(field->name().c_str(), value);                          \
      break;                                                                    \
    }
    CASE_FIELD_TYPE(INT32, Int32, int32_t, d);
    CASE_FIELD_TYPE(UINT32, UInt32, uint32_t, u);
    CASE_FIELD_TYPE(INT64, Int64, int64_t, ld);
    CASE_FIELD_TYPE(FLOAT, Float, float, f);
    CASE_FIELD_TYPE(BOOL, Bool, bool, d);
#undef CASE_FIELD_TYPE
    case google::protobuf::FieldDescriptor::CPPTYPE_ENUM: {
      GE_CHECK_NOTNULL(reflection->GetEnum(*message, field));
      int value = reflection->GetEnum(*message, field)->number();
      GELOGD("Parse result(%s : %d)", field->name().c_str(), value);
      (void)ops.SetAttr(field->name().c_str(), value);
      break;
    }
    case google::protobuf::FieldDescriptor::CPPTYPE_STRING: {
      string value = reflection->GetString(*message, field);
      GELOGD("Parse result(%s : %s)", field->name().c_str(), value.c_str());
      (void)ops.SetAttr(field->name().c_str(), value);
      break;
    }
    case google::protobuf::FieldDescriptor::CPPTYPE_MESSAGE: {
      const google::protobuf::Message &sub_message = reflection->GetMessage(*message, field);
      if (ParseOperatorAttrs(&sub_message, depth + 1, ops) != SUCCESS) {
        GELOGE(FAILED, "[Parse][OperatorAttrs] of %s failed.", field->name().c_str());
        return FAILED;
      }
      break;
    }
    default: {
      REPORT_PREDEFINED_ERR_MSG("E11032", std::vector<const char *>({"message_type", "name", "reason"}),
                                std::vector<const char *>({"model", field->name().c_str(), "Unsupported field type"}));
      GELOGE(FAILED, "[Check][FieldType]Unsupported field type, name: %s.", field->name().c_str());
      return FAILED;
    }
  }
  GELOGD("Parse field: %s success.", field->name().c_str());
  return SUCCESS;
}

Status Message2Operator::ParseRepeatedField(const google::protobuf::Reflection *reflection,
                                            const google::protobuf::Message *message,
                                            const google::protobuf::FieldDescriptor *field,
                                            ge::Operator &ops) {
  GELOGD("Start to parse field: %s.", field->name().c_str());
  int field_size = reflection->FieldSize(*message, field);
  if (field_size <= 0) {
    REPORT_INNER_ERR_MSG("E19999", "Size of repeated field %s must bigger than 0", field->name().c_str());
    GELOGE(FAILED, "[Check][Size]Size of repeated field %s must bigger than 0", field->name().c_str());
    return FAILED;
  }

  switch (field->cpp_type()) {
#define CASE_FIELD_TYPE_REPEATED(cpptype, method, valuetype)                   \
    case google::protobuf::FieldDescriptor::CPPTYPE_##cpptype: {               \
      std::vector<valuetype> attr_value;                                       \
      for (int i = 0; i < field_size; i++) {                                   \
        valuetype value = reflection->GetRepeated##method(*message, field, i); \
        attr_value.push_back(value);                                           \
      }                                                                        \
      (void)ops.SetAttr(field->name().c_str(), attr_value);                    \
      break;                                                                   \
    }
    CASE_FIELD_TYPE_REPEATED(INT32, Int32, int32_t);
    CASE_FIELD_TYPE_REPEATED(UINT32, UInt32, uint32_t);
    CASE_FIELD_TYPE_REPEATED(INT64, Int64, int64_t);
    CASE_FIELD_TYPE_REPEATED(FLOAT, Float, float);
    CASE_FIELD_TYPE_REPEATED(BOOL, Bool, bool);
    CASE_FIELD_TYPE_REPEATED(STRING, String, string);
#undef CASE_FIELD_TYPE_REPEATED
    case google::protobuf::FieldDescriptor::CPPTYPE_MESSAGE: {
      nlohmann::json message_json;
      Pb2Json::RepeatedMessage2Json(*message, field, reflection, std::set<string>(), message_json[field->name()],
                                    false);
      std::string repeated_message_str;
      try {
        repeated_message_str = message_json.dump(kInteval, ' ', false, Json::error_handler_t::ignore);
      } catch (std::exception &e) {
        REPORT_INNER_ERR_MSG("E19999", "Failed to convert JSON to string, reason: %s.", e.what());
        GELOGE(FAILED, "[Parse][JSON]Failed to convert JSON to string, reason: %s.", e.what());
        return FAILED;
      } catch (...) {
        REPORT_INNER_ERR_MSG("E19999", "Failed to convert JSON to string.");
        GELOGE(FAILED, "[Parse][JSON]Failed to convert JSON to string.");
        return FAILED;
      }
      (void)ops.SetAttr(field->name().c_str(), repeated_message_str);
      break;
    }
    default: {
      REPORT_PREDEFINED_ERR_MSG("E11032", std::vector<const char *>({"message_type", "name", "reason"}),
                                std::vector<const char *>({"model", field->name().c_str(), "Unsupported field type"}));
      GELOGE(FAILED, "[Check][FieldType]Unsupported field type, name: %s.", field->name().c_str());
      return FAILED;
    }
  }
  GELOGD("Parse repeated field: %s success.", field->name().c_str());
  return SUCCESS;
}
}  // namespace ge
