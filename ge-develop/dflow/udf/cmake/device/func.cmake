# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

function(protobuf_generate comp c_var h_var)
    if(NOT ARGN)
        message(SEND_ERROR "Error: protobuf_generate() called without any proto files")
        return()
    endif()
    set(${c_var})
    set(${h_var})
    set(_add_target FALSE)

    set(extra_option "")
    foreach(arg ${ARGN})
        if ("${arg}" MATCHES "--proto_path")
            set(extra_option ${arg})
        endif()
    endforeach()

    foreach(file ${ARGN})
        if("${file}" STREQUAL "TARGET")
            set(_add_target TRUE)
            continue()
        endif()

        if ("${file}" MATCHES "--proto_path")
            continue()
        endif()

        get_filename_component(abs_file ${file} ABSOLUTE)
        get_filename_component(file_name ${file} NAME_WE)
        get_filename_component(file_dir ${abs_file} PATH)
        get_filename_component(parent_subdir ${file_dir} NAME)

        if("${parent_subdir}" STREQUAL "proto")
            set(proto_output_path ${CMAKE_BINARY_DIR}/proto/${comp}/proto)
        else()
            set(proto_output_path ${CMAKE_BINARY_DIR}/proto/${comp}/proto/${parent_subdir})
        endif()
        list(APPEND ${c_var} "${proto_output_path}/${file_name}.pb.cc")
        list(APPEND ${h_var} "${proto_output_path}/${file_name}.pb.h")

        add_custom_command(
                OUTPUT "${proto_output_path}/${file_name}.pb.cc" "${proto_output_path}/${file_name}.pb.h"
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                COMMAND ${CMAKE_COMMAND} -E make_directory "${proto_output_path}"
                COMMAND ${CMAKE_COMMAND} -E echo "generate proto cpp_out ${comp} by ${abs_file}"
                COMMAND host_protoc -I${file_dir} ${extra_option} --cpp_out=${proto_output_path} ${abs_file}
                DEPENDS ${abs_file} host_protoc
                COMMENT "Running C++ protocol buffer compiler on ${file}" VERBATIM )
    endforeach()

    if(_add_target)
        add_custom_target(
                ${comp} DEPENDS ${${c_var}} ${${h_var}}
        )
    endif()

    set_source_files_properties(${${c_var}} ${${h_var}} PROPERTIES GENERATED TRUE)
    set(${c_var} ${${c_var}} PARENT_SCOPE)
    set(${h_var} ${${h_var}} PARENT_SCOPE)
endfunction()

# =============================================================================
# Function: pack_targets_and_files
#
# Packs targets and/or files into a flat tar.gz archive (no directory structure).
# Optionally generates a SHA256 manifest file and includes it in the archive.
#
# Usage:
#   pack_targets_and_files(
#       OUTPUT <output.tar.gz>           # e.g., "cann-tsch-compat.tar.gz"
#       [TARGETS target1 [target2 ...]]
#       [FILES file1 [file2 ...]]
#       [MANIFEST <manifest_filename>]   # e.g., "aicpu_compat_bin_hash.cfg"
#   )
#
# Examples:
#   # With manifest
#   pack_targets_and_files(
#       OUTPUT cann-tsch-compat.tar.gz
#       TARGETS app server
#       FILES "LICENSE" "config/default.json"
#       MANIFEST "aicpu_compat_bin_hash.cfg"
#   )
#
#   # Without manifest
#   pack_targets_and_files(
#       OUTPUT cann-tsch-compat.tar.gz
#       TARGETS app
#       FILES "README.md"
#   )
# =============================================================================
function(pack_targets_and_files)
    cmake_parse_arguments(ARG
            ""
            "OUTPUT;MANIFEST;OUTPUT_TARGET"
            "TARGETS;FILES"
            ${ARGN}
    )

    # --- Validation ---
    if(NOT ARG_OUTPUT)
        message(FATAL_ERROR "[pack_targets_and_files] OUTPUT is required")
    endif()

    if(NOT IS_ABSOLUTE "${ARG_OUTPUT}")
        set(ARG_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${ARG_OUTPUT}")
    endif()

    if(NOT ARG_OUTPUT_TARGET)
        message(FATAL_ERROR "[pack_targets_and_files] OUTPUT_TARGET is required")
    endif()

    # Generate safe target name
    get_filename_component(tar_basename "${ARG_OUTPUT}" NAME_WE)
    string(MAKE_C_IDENTIFIER "pack_${tar_basename}" safe_name)
    set(staging_dir "${CMAKE_CURRENT_BINARY_DIR}/_${safe_name}_stage")

    # --- Collect all source items (as generator expressions) ---
    set(src_items "")
    foreach(tgt IN LISTS ARG_TARGETS)
        if(NOT TARGET ${tgt})
            message(FATAL_ERROR "[pack_targets_and_files] Target '${tgt}' does not exist")
        endif()
        list(APPEND src_items "$<TARGET_FILE:${tgt}>")
    endforeach()
    list(APPEND src_items ${ARG_FILES})

    if(NOT src_items)
        message(FATAL_ERROR "[pack_targets_and_files] No targets or files specified to pack")
    endif()

    set(manifest_arg "")
    if(ARG_MANIFEST)
        if("${ARG_MANIFEST}" STREQUAL "")
            message(FATAL_ERROR "[pack] MANIFEST filename cannot be empty")
        endif()
        if(IS_ABSOLUTE "${ARG_MANIFEST}")
            message(FATAL_ERROR "[pack] MANIFEST must be relative (e.g., 'sha256sums.cfg')")
        endif()
        set(manifest_arg -D_MANIFEST_FILE=${staging_dir}/${ARG_MANIFEST})
    endif()

    add_custom_command(
            OUTPUT ${staging_dir}
            COMMAND ${CMAKE_COMMAND} -E make_directory "${staging_dir}"
            VERBATIM
    )

    add_custom_command(
            OUTPUT "${ARG_OUTPUT}"
            COMMAND ${CMAKE_COMMAND}
            -D _STAGING_DIR=${staging_dir}
            ${manifest_arg}
            -D "_ITEMS=$<JOIN:${src_items},;>"
            -P "${CMAKE_CURRENT_LIST_DIR}/_pack_stage.cmake"
            COMMAND ${CMAKE_COMMAND} -E tar "czf" "${ARG_OUTPUT}" .
            WORKING_DIRECTORY ${staging_dir}
            DEPENDS ${ARG_TARGETS} ${staging_dir}
            COMMENT "Packing with ${ARG_OUTPUT}"
            VERBATIM
    )

    add_custom_target(${ARG_OUTPUT_TARGET} ALL DEPENDS "${ARG_OUTPUT}")
endfunction()

# sign_file.cmake
# =============================================================================
# Function: sign_file
#
# Signs a file and places signature in a standard directory.
#
# Usage:
#   sign_file(
#       INPUT <input_file>
#       SCRIPT <sign_script>
#       [SCRIPT_ARGS ...]
#       [RESULT_VAR <output_var>]   # ← returns generated sig path
#       [DEPENDS ...]
#       [WORKING_DIRECTORY ...]
#   )
#
# Output location: ${CMAKE_BINARY_DIR}/signatures/<basename>.sig
# Signature script interface:
#   $1 = input file (absolute)
#   $2 = output signature file (absolute, auto-generated)
#   [optional] extra args
#
# Example:
#   sign_file(INPUT $<TARGET_FILE:app> SCRIPT sign.sh RESULT_VAR APP_SIG)
#   message(STATUS "Signature: ${APP_SIG}")
# =============================================================================
function(sign_file)
    cmake_parse_arguments(
            ARG
            ""
            "OUTPUT_TARGET;INPUT;CONFIG;RESULT_VAR"
            "DEPENDS"
            ${ARGN}
    )

    # --- Validation ---
    if(DEFINED CUSTOM_SIGN_SCRIPT AND NOT CUSTOM_SIGN_SCRIPT STREQUAL "")
        set(SIGN_SCRIPT ${CUSTOM_SIGN_SCRIPT})
    else()
        set(SIGN_SCRIPT)
    endif()

    if(ENABLE_SIGN)
        set(sign_flag "true")
    else()
        set(sign_flag "false")
    endif()

    foreach(var INPUT CONFIG RESULT_VAR)
        if(NOT ARG_${var})
            message(FATAL_ERROR "[sign_file] Missing required: ${var}")
        endif()
    endforeach()

    if(NOT EXISTS "${ARG_CONFIG}")
        message(FATAL_ERROR "[sign_file] Sign config not found: ${ARG_CONFIG}")
    endif()

    # Normalize input
    if(NOT IS_ABSOLUTE "${ARG_INPUT}")
        set(ARG_INPUT "${CMAKE_CURRENT_BINARY_DIR}/${ARG_INPUT}")
    endif()

    # Auto output path: ${CMAKE_CURRENT_BINARY_DIR}/signatures
    set(signatures_dir "${CMAKE_CURRENT_BINARY_DIR}/signatures")
    get_filename_component(input_name "${ARG_INPUT}" NAME)
    set(output_sig "${signatures_dir}/${input_name}")

    if(EXISTS "${SIGN_SCRIPT}")
        get_filename_component(EXT ${SIGN_SCRIPT} EXT) # 获取文件扩展名
        if (${EXT} STREQUAL ".sh")
            set(sign_cmd bash ${SIGN_SCRIPT} ${output_sig} ${ARG_CONFIG} ${sign_flag})
        elseif (${EXT} STREQUAL ".py")
            message(STATUS "Detected VERSION_INFO:${VERSION_INFO}, TOP_DIR:${TOP_DIR}")
            set(sign_cmd python3 ${TOP_DIR}/scripts/sign/add_header_sign.py ${signatures_dir} ${sign_flag} --bios_check_cfg=${ARG_CONFIG} --sign_script=${SIGN_SCRIPT} --version=${VERSION_INFO})
        else ()
            message(FATAL_ERROR "unsupported sign script file type, only support '.sh' or '.py', but got ${EXT} in ${SIGN_SCRIPT}.")
        endif()
    else ()
        set(sign_cmd)
    endif()

    # Ensure dir exists
    file(MAKE_DIRECTORY "${signatures_dir}")
    if (ARG_OUTPUT_TARGET)
        set(sign_target "${ARG_OUTPUT_TARGET}")
    else ()
        # Target name
        get_filename_component(sign_basename "${ARG_INPUT}" NAME_WE)
        string(MAKE_C_IDENTIFIER "${sign_basename}" safe_name)
        set(sign_target "sign_${safe_name}")
    endif ()

    add_custom_command(
            OUTPUT "${output_sig}"
            COMMAND ${CMAKE_COMMAND} -E make_directory ${signatures_dir}
            COMMAND ${CMAKE_COMMAND} -E copy ${ARG_INPUT} ${output_sig}
            COMMAND ${sign_cmd}
            DEPENDS "${ARG_INPUT}" "${SIGN_SCRIPT}" ${ARG_DEPENDS} ${ARG_CONFIG}
            COMMENT "Signing: ${ARG_INPUT} → ${output_sig}"
            VERBATIM
    )

    add_custom_target(${sign_target} ALL DEPENDS "${output_sig}")

    # Return path via RESULT_VAR
    if(ARG_RESULT_VAR)
        set(${ARG_RESULT_VAR} "${output_sig}" PARENT_SCOPE)
    endif()
endfunction()
