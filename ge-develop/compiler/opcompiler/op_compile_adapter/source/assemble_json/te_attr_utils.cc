/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "assemble_json/te_attr_utils.h"
#include "inc/te_fusion_util_constants.h"
#include "inc/te_fusion_log.h"
#include "common/fusion_common.h"

namespace te {
namespace fusion {
using nlohmann::json;
namespace {
template<typename T>
void GetNullDesc(const T value, NullDesc &nullDesc)
{
    if (std::isinf(value)) {
        nullDesc.nullType = NullType::SINGLE_VALUE;
        if (std::signbit(value)) {
            nullDesc.nullDesc.push_back(KEY_NEGTIVE_INF);
        } else {
            nullDesc.nullDesc.push_back(KEY_INF);
        }
    } else if (std::isnan(value)) {
        nullDesc.nullType = NullType::SINGLE_VALUE;
        nullDesc.nullDesc.push_back(KEY_NAN);
    } else {
        nullDesc.nullType = NullType::NORMAL;
        nullDesc.nullDesc.push_back(KEY_NULL);
    }
}

template<typename T>
void ParseTbeAttrValue(json &jsonData, const std::string &attrType, const TbeAttrValue &attr, NullDesc &)
{
    T value = {};
    attr.GetValue(value);
    jsonData[attrType] = value;
}

void ParseTbeAttrValueForFloat(json &jsonData, const std::string &attrType, const TbeAttrValue &attr,
                               NullDesc &nullDesc)
{
    float value = {};
    attr.GetValue(value);
    jsonData[attrType] = value;

    GetNullDesc<float>(value, nullDesc);
}

void ParseTbeAttrValueForDouble(json &jsonData, const std::string &attrType, const TbeAttrValue &attr,
                                NullDesc &nullDesc)
{
    double value = {};
    attr.GetValue(value);
    jsonData[attrType] = value;

    GetNullDesc<double>(value, nullDesc);
}

template<typename T>
void GetVectorNullDesc(const T &value, NullDesc &nullDesc)
{
    nullDesc.nullType = NullType::NORMAL;
    for (auto iter = value.begin(); iter != value.end(); iter ++) {
        if (std::isinf(*iter)) {
            nullDesc.nullType = NullType::LIST_VALUE;
            if (std::signbit(*iter)) {
                nullDesc.nullDesc.push_back(KEY_NEGTIVE_INF);
            } else {
                nullDesc.nullDesc.push_back(KEY_INF);
            }
        } else if (std::isnan(*iter)) {
            nullDesc.nullType = NullType::LIST_VALUE;
            nullDesc.nullDesc.push_back(KEY_NAN);
        } else {
            nullDesc.nullDesc.push_back(KEY_NULL);
        }
    }
}

void ParseTbeAttrValueForVectorFloat(json &jsonData, const std::string &attrType, const TbeAttrValue &attr,
                                     NullDesc &nullDesc)
{
    std::vector<float> value = {};
    attr.GetValue(value);
    jsonData[attrType] = value;

    GetVectorNullDesc<std::vector<float>>(value, nullDesc);
}

void ParseTbeAttrValueForVectorDouble(json &jsonData, const std::string &attrType, const TbeAttrValue &attr,
                                      NullDesc &nullDesc)
{
    std::vector<double> value = {};
    attr.GetValue(value);
    jsonData[attrType] = value;

    GetVectorNullDesc<std::vector<double>>(value, nullDesc);
}

struct TbeAttrParserParam {
    std::string attrType;
    std::function<void(nlohmann::json&, std::string&, const TbeAttrValue &, NullDesc &)> parser;
};
std::map<ATTR_DTYPE, TbeAttrParserParam> TbeAttrValueParser = {
    {ATTR_INT8,         {"ATTR_INT8",           ParseTbeAttrValue<int8_t>}},
    {ATTR_UINT8,        {"ATTR_UINT8",          ParseTbeAttrValue<uint8_t>}},
    {ATTR_INT16,        {"ATTR_INT16",          ParseTbeAttrValue<int16_t>}},
    {ATTR_UINT16,       {"ATTR_UINT16",         ParseTbeAttrValue<uint16_t>}},
    {ATTR_INT32,        {"ATTR_INT32",          ParseTbeAttrValue<int32_t>}},
    {ATTR_UINT32,       {"ATTR_UINT32",         ParseTbeAttrValue<uint32_t>}},
    {ATTR_INT64,        {"ATTR_INT64",          ParseTbeAttrValue<int64_t>}},
    {ATTR_UINT64,       {"ATTR_UINT64",         ParseTbeAttrValue<uint64_t>}},
    {ATTR_FLOAT32,      {"ATTR_FLOAT32",        ParseTbeAttrValueForFloat}},
    {ATTR_DOUBLE,       {"ATTR_DOUBLE",         ParseTbeAttrValueForDouble}},
    {ATTR_BOOL,         {"ATTR_BOOL",           ParseTbeAttrValue<bool>}},
    {ATTR_STR,          {"ATTR_STR",            ParseTbeAttrValue<std::string>}},
    {ATTR_LIST_INT8,    {"ATTR_LIST_INT8",      ParseTbeAttrValue<std::vector<int8_t>>}},
    {ATTR_LIST_UINT8,   {"ATTR_LIST_UINT8",     ParseTbeAttrValue<std::vector<uint8_t>>}},
    {ATTR_LIST_INT16,   {"ATTR_LIST_INT16",     ParseTbeAttrValue<std::vector<int16_t>>}},
    {ATTR_LIST_UINT16,  {"ATTR_LIST_UINT16",    ParseTbeAttrValue<std::vector<uint16_t>>}},
    {ATTR_LIST_INT32,   {"ATTR_LIST_INT32",     ParseTbeAttrValue<std::vector<int32_t>>}},
    {ATTR_LIST_UINT32,  {"ATTR_LIST_UINT32",    ParseTbeAttrValue<std::vector<uint32_t>>}},
    {ATTR_LIST_INT64,   {"ATTR_LIST_INT64",     ParseTbeAttrValue<std::vector<int64_t>>}},
    {ATTR_LIST_UINT64,  {"ATTR_LIST_UINT64",    ParseTbeAttrValue<std::vector<uint64_t>>}},
    {ATTR_LIST_FLOAT32, {"ATTR_LIST_FLOAT32",   ParseTbeAttrValueForVectorFloat}},
    {ATTR_LIST_DOUBLE,  {"ATTR_LIST_DOUBLE",    ParseTbeAttrValueForVectorDouble}},
    {ATTR_LIST_BOOL,    {"ATTR_LIST_BOOL",      ParseTbeAttrValue<std::vector<bool>>}},
    {ATTR_LIST_STR,     {"ATTR_LIST_STR",       ParseTbeAttrValue<std::vector<std::string>>}},
    {ATTR_LIST_LIST_INT64, {"ATTR_LIST_LIST_INT64", ParseTbeAttrValue<std::vector<std::vector<int64_t>>>}}
};
}
void SetAttrDtype2Json(const TbeAttrValue &cattr, nlohmann::json &jsonData)
{
    ATTR_DTYPE dtype = cattr.GetType();
    if (TbeAttrValueParser.find(dtype) == TbeAttrValueParser.end()) {
        TE_DBGLOGF("Can not find dtype[%d] in TbeAttrValueParser", dtype);
        return;
    }

    std::string attrTypeStr;
    (void)TbeAttrDtypeToString(dtype, attrTypeStr);
    jsonData["dtype"] = attrTypeStr;
}

void GetAttrValueToJson(const TbeAttrValue &attrValue, nlohmann::json &attrValueJson, NullDesc &nullDesc)
{
    ATTR_DTYPE dtype = attrValue.GetType();
    if (TbeAttrValueParser.find(dtype) == TbeAttrValueParser.end()) {
        TE_WARNLOGF("dtype[%d] is invalid", dtype);
        return;
    }
    auto &attrTypeStr = TbeAttrValueParser[dtype].attrType;
    auto &attrParser = TbeAttrValueParser[dtype].parser;

    json jsonAttr;
    attrParser(jsonAttr, attrTypeStr, attrValue, nullDesc);
    attrValueJson = jsonAttr[attrTypeStr];
}

void GetAttrValueToJson(const TbeAttrValue &attrValue, nlohmann::json &attrValueJson)
{
    NullDesc nullDesc;
    GetAttrValueToJson(attrValue, attrValueJson, nullDesc);
}
}
}