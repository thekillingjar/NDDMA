/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TE_FUSION_ERROR_CODE_H
#define TE_FUSION_ERROR_CODE_H

namespace te {
namespace fusion {
constexpr int ERROR_CODE_MAX_LENGTH = 16;
constexpr int ERROR_DIED_PROCESS_STATUS_CODE = 3;

/* Path [%s] for [%s] is invalid. Result: %s. Reason: %s.
 * path,arg,result,reason
 */
constexpr const char *EM_PATH_INVALID_WARNING = "W40010";

/* Failed to create disk cache directory [%s]. Result: %s. Reason: %s.
 * path,result,reason
 */
constexpr const char *EM_DIRECTORY_CREATE_FAILED_WARNING = "W40011";

/* Value [%s] for parameter [%s] is invalid. The value must be in the range of [%s] and defaults to [%s].
 * invalid_value,argument,valid_range,default_value
 */
constexpr const char *EM_PARAMETER_INVALID_WARNING = "W40012";

/* Value [%s] for environment variable [%s] is invalid when %s.
 * value,env,situation
 */
constexpr const char *EM_ENVIRONMENT_INVALID_ERROR = "E40001";

/*  The current Python version is [%s]. The system supports only Python [%s] or later.
 * currentVersion,supportVersion
 */
constexpr const char *EM_PYTHON_VERSION_INVALID_ERROR = "E40002";

/* Failed to open Json file: %s
 * file_name
 */
constexpr const char *EM_OPEN_JSON_FILE_ERROR = "E40003";

/* Failed to import Python module [%s].
 * result
 */
constexpr const char *EM_IMPORT_PYTHON_FAILED_ERROR = "E40020";

/* Failed to compile Op [%s]. (oppath: [%s], optype: [%s]).
 * op_name,opp_path,op_type
 */
constexpr const char *EM_COMPILE_OP_FAILED_ERROR = "E40021";

/* Value [%s] for parameter [%s] is invalid. The value must be in the range of [%s].
 * invalid_value,argument,valid_range
 */
constexpr const char *EM_PARAMETER_INVALID_ERROR = "E40022";

/* Path [%s] for [%s] is invalid. Result: %s. Reason: %s.
 * path,arg,result,reason
 */
constexpr const char *EM_PATH_INVALID_ERROR = "E40023";

constexpr const char *EM_CALL_FUNC_MATHOD_ERROR = "E40024";

constexpr const char *EM_INNER_WARNING = "W49999";

constexpr const char *EM_INNER_ERROR = "E49999";

constexpr const char *EM_UNKNOWN_PROCESS_DIED_ERROR = "EZ9999";

} // namespace fusion
} // namespace te

#endif /* TE_FUSION_ERROR_CODE_H */
