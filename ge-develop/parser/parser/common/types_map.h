/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_TYPES_MAP_H
#define GE_TYPES_MAP_H

#include "graph/types.h"
#include "proto/tensorflow/graph.pb.h"

namespace ge {
// Correspondence between data_type in GE and tensorflow
static map<int32_t, int32_t> GE_TENSORFLOW_DATA_TYPE_MAP = {
    {ge::DataType::DT_UNDEFINED, domi::tensorflow::DT_INVALID},
    {ge::DataType::DT_FLOAT, domi::tensorflow::DT_FLOAT},
    {ge::DataType::DT_FLOAT16, domi::tensorflow::DT_HALF},
    {ge::DataType::DT_INT8, domi::tensorflow::DT_INT8},
    {ge::DataType::DT_INT16, domi::tensorflow::DT_INT16},
    {ge::DataType::DT_UINT16, domi::tensorflow::DT_UINT16},
    {ge::DataType::DT_UINT8, domi::tensorflow::DT_UINT8},
    {ge::DataType::DT_INT32, domi::tensorflow::DT_INT32},
    {ge::DataType::DT_INT64, domi::tensorflow::DT_INT64},
    {ge::DataType::DT_UINT32, domi::tensorflow::DT_UINT32},
    {ge::DataType::DT_UINT64, domi::tensorflow::DT_UINT64},
    {ge::DataType::DT_STRING, domi::tensorflow::DT_STRING},
    {ge::DataType::DT_RESOURCE, domi::tensorflow::DT_RESOURCE},
    {ge::DataType::DT_BOOL, domi::tensorflow::DT_BOOL},
    {ge::DataType::DT_DOUBLE, domi::tensorflow::DT_DOUBLE},
    {ge::DataType::DT_COMPLEX64, domi::tensorflow::DT_COMPLEX64},
    {ge::DataType::DT_COMPLEX128, domi::tensorflow::DT_COMPLEX128},
    {ge::DataType::DT_QINT8, domi::tensorflow::DT_QINT8},
    {ge::DataType::DT_QINT16, domi::tensorflow::DT_QINT16},
    {ge::DataType::DT_QINT32, domi::tensorflow::DT_QINT32},
    {ge::DataType::DT_QUINT8, domi::tensorflow::DT_QUINT8},
    {ge::DataType::DT_QUINT16, domi::tensorflow::DT_QUINT16},
    {ge::DataType::DT_VARIANT, domi::tensorflow::DT_VARIANT},
    {ge::DataType::DT_DUAL, domi::tensorflow::DT_INVALID},
    {ge::DataType::DT_DUAL_SUB_INT8, domi::tensorflow::DT_INVALID},
    {ge::DataType::DT_DUAL_SUB_UINT8, domi::tensorflow::DT_INVALID},
};

static map<domi::tensorflow::DataType, std::string> TF_DATATYPE_TO_GE_SYMBOL_MAP = {
  {domi::tensorflow::DT_INVALID,      "DT_UNDEFINED"},
  {domi::tensorflow::DT_FLOAT,        "DT_FLOAT"},
  {domi::tensorflow::DT_HALF,         "DT_FLOAT16"},
  {domi::tensorflow::DT_INT8,         "DT_INT8"},
  {domi::tensorflow::DT_INT16,        "DT_INT16"},
  {domi::tensorflow::DT_UINT16,       "DT_UINT16"},
  {domi::tensorflow::DT_UINT8,        "DT_UINT8"},
  {domi::tensorflow::DT_INT32,        "DT_INT32"},
  {domi::tensorflow::DT_INT64,        "DT_INT64"},
  {domi::tensorflow::DT_UINT32,       "DT_UINT32"},
  {domi::tensorflow::DT_UINT64,       "DT_UINT64"},
  {domi::tensorflow::DT_STRING,       "DT_STRING"},
  {domi::tensorflow::DT_RESOURCE,     "DT_RESOURCE"},
  {domi::tensorflow::DT_BOOL,         "DT_BOOL"},
  {domi::tensorflow::DT_DOUBLE,       "DT_DOUBLE"},
  {domi::tensorflow::DT_COMPLEX64,    "DT_COMPLEX64"},
  {domi::tensorflow::DT_COMPLEX128,   "DT_COMPLEX128"},
  {domi::tensorflow::DT_QINT8,        "DT_QINT8"},
  {domi::tensorflow::DT_QINT16,       "DT_QINT16"},
  {domi::tensorflow::DT_QINT32,       "DT_QINT32"},
  {domi::tensorflow::DT_QUINT8,       "DT_QUINT8"},
  {domi::tensorflow::DT_QUINT16,      "DT_QUINT16"},
  {domi::tensorflow::DT_VARIANT,      "DT_VARIANT"},
};

static unordered_map<std::string, std::pair<std::string, std::string>> attr_type_map = {
  {"int",          {"Int",          "0"}},
  {"float",        {"Float",        "0.0"}},
  {"bool",         {"Bool",         "false"}},
  {"string",       {"String",       "\"\""}},
  {"type",         {"DataType",     "DT_FLOAT"}},
  {"shape",        {"Shape",        "\"[]\""}},
  {"list(int)",    {"IntList",      "{}"}},
  {"list(float)",  {"FloatList",    "{}"}},
  {"list(bool)",   {"BoolList",     "{}"}},
  {"list(string)", {"StringList",   "{}"}},
  {"list(type)",   {"DataTypeList", "{DT_FLOAT}"}},
  {"list(shape)",  {"ShapeList",    "{}"}},
  {"func",         {"Func",         "\"\""}},
  {"tensor",       {"Tensor",       "\"\""}}
};

}  // namespace ge
#endif  // GE_TYPES_MAP_H
