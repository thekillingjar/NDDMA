#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

USER_ID=$(id -u)
CURRENT_DIR=$(dirname $(readlink -f $0))
SOURCE_DIR=${CURRENT_DIR}/packages

if [ "${USER_ID}" != "0" ]; then
    DEFAULT_TOOLKIT_INSTALL_DIR="${HOME}/Ascend/ascend-toolkit/latest"
    DEFAULT_INSTALL_DIR="${HOME}/Ascend/latest"
else
    DEFAULT_TOOLKIT_INSTALL_DIR="/usr/local/Ascend/ascend-toolkit/latest"
    DEFAULT_INSTALL_DIR="/usr/local/Ascend/latest"
fi

function log() {
    local current_time=`date +"%Y-%m-%d %H:%M:%S"`
    echo "[$current_time] "$1
}

function get_version_dir() {
    local _outvar="$1"
    local _version_info_path="$2"
    local _result

    if [ ! -f "${_version_info_path}" ]; then
        eval "${_outvar}=\"\""
        return 1
    fi

    _result="$(grep "^version_dir=" "${_version_info_path}" | cut -d= -f2-)"
    eval "${_outvar}=\"${_result}\""
}

function get_version_info() {
    local _outvar="$1"
    local _version_info_path="$2"
    local _result

    if [ ! -f "${_version_info_path}" ]; then
        eval "${_outvar}=\"\""
        return 1
    fi

    _result="$(grep "^Version=" "${_version_info_path}" | cut -d= -f2-)"
    eval "${_outvar}=\"${_result}\""
}

function get_backup_dir() {
    local _outvar="$1"
    local _version_info_path="$2"
    local _result

    if [ ! -f "${_version_info_path}" ]; then
        eval "${_outvar}=\"\""
        return 1
    fi

    _result="$(grep "^backup_dir=" "${_version_info_path}" | cut -d= -f2-)"
    eval "${_outvar}=\"${_result}\""
}

function get_base_package() {
    local _outvar="$1"
    local _version_info_path="$2"
    local _result

    if [ ! -f "${_version_info_path}" ]; then
        eval "${_outvar}=\"\""
        return 1
    fi

    _result="$(grep "^base_package=" "${_version_info_path}" | cut -d= -f2-)"
    eval "${_outvar}=\"${_result}\""
}

function get_version_file() {
    local _outvar="$1"
    local _root_dir="$2"
    local _package_list="$3"
    local _result=""

    temp_ifs=$IFS
    IFS=";"
    for package in $_package_list; do
        version_file=${_root_dir}/${package}/version.info
        if [ -f "${version_file}" ];then
            _result=${version_file}
            break
        fi
    done
    IFS=$temp_ifs

    if [ "${_result}" == "" ]; then
        eval "${_outvar}=\"\""
        return 1
    else
        eval "${_outvar}=\"${_result}\""
    fi
}

function delete_backup() {
    local manifest_file="$1"
    local version_info_file="$2"
    local backup_dir="$3"
    if [ -n "${manifest_file}" ] && [ -f "${manifest_file}" ];then
        rm -rf ${manifest_file}
    fi

    if [ -n "${version_info_file}" ] && [ -f "${version_info_file}" ];then
        rm -rf ${version_info_file}
    fi

    if [ -n "${backup_dir}" ] && [ -d "${backup_dir}" ];then
        rm -rf ${backup_dir}
    fi

}

function cp_file() {
    local src_dir="$1"
    local dest_dir="$2"
    local relative_file="$3"
    local src_file="${src_dir}/${relative_file}"
    local dest_file="${dest_dir}/${relative_file}"
    local install_dir=$(dirname ${dest_file})
    if [ "${IS_QUIET}" == "n" ];then
        local _option="-v"
    fi
    if [ ! -f "${install_dir}" ];then
        mkdir -p ${install_dir}
    fi
    cp -rf ${_option} ${src_file} ${dest_file}
}

function install_patches() {
    local mode="$1"
    local src_dir="$2"
    local dest_dir="$3"
    local install_manifest="$4"
    local backup_dir="$5"
    local original_perms=""

    # 获取源目录中所有子目录
    local src_subdirectorys=$(find ${src_dir} -mindepth 1 -type d | sort)

    # 为目的目录添加写权限，并记录原始目录权限
    for file in ${src_subdirectorys}; do
        relative_directory="${file#$src_dir/}"
        actual_directory=${dest_dir}/${relative_directory}
        if [ -d "${actual_directory}" ];then
            perm=$(stat -c '%a %n' $actual_directory)
            original_perms+=" ${perm}"
            chmod u+w ${actual_directory}
        fi
    done

    if [ "${mode}" == "install" ];then
        # install模式根据生成manifest文件
        install_manifest_dir=$(dirname ${install_manifest})
        if [ ! -f "${install_manifest_dir}" ];then
            mkdir -p ${install_manifest_dir}
        fi
        touch ${install_manifest}
    else
        while read line
        do
            rm -f ${dest_dir}/${line}
        done < ${install_manifest}
    fi

    # 获取源目录中所有文件
    local file_list=$(find ${src_dir} -xtype f | sort)
    for file in ${file_list}; do
        relative_file="${file#$src_dir/}"
        actual_file=${dest_dir}/${relative_file}

        # 记录安装文件列表
        if [ "${mode}" == "install" ];then
            echo "${relative_file}" >> ${install_manifest}
            if [ -f "${actual_file}" ];then
                chmod u+w ${actual_file}
                cp_file ${dest_dir} ${backup_dir} ${relative_file}
            fi
        fi
        cp_file ${src_dir} ${dest_dir} ${relative_file}
    done

    # 恢复目的目录及其子目录的原始权限
    original_perm_arr=($original_perms)
    for((i=0;i<${#original_perm_arr[@]};i+=2))
    do
        chmod ${original_perm_arr[i]} ${original_perm_arr[i+1]}
    done
}

function check_version_compatiable() {
    local base_version="$1"
    local current_version="$2"
    _base_version=$(echo ${base_version} | cut -d'.' -f1-4)
    _current_version=$(echo ${current_version} | cut -d'.' -f1-4)

    _lower_base_version=$(echo "${_base_version}" | tr '[:upper:]' '[:lower:]')
    _lower_current_version=$(echo "${_current_version}" | tr '[:upper:]' '[:lower:]')

    if [ "$_lower_base_version" != "$_lower_current_version" ]; then
        log "[ERROR] the version number of the incremental package is ${_lower_current_version}, and the version number of the cann package used is ${_lower_base_version}. Please install version ${_lower_current_version} of the cann package."
        exit 1
    fi
}

function copy_version_info() {
    local src_file="$1"
    local dst_file="$2"

    dst_dir=$(dirname ${dst_file})
    if [ ! -d ${dst_dir} ];then
        mkdir -p ${dst_dir}
    fi
    cp -rf ${src_file} ${dst_file}
}

function create_softlink_for_libge_runner_v2 () {
    local target_root_dir="$1"
    local target_version_dir="$2"
    local package="compiler"
    local arch_linux="$(find $target_root_dir/$target_version_dir -maxdepth 1 -type d -name '*-linux' | xargs basename)"

    (cd $target_root_dir/$target_version_dir/$package/lib64 && chmod u+w . && \
        ln -sf ../../x86_64-linux/lib64/libge_runner_v2.so libge_runner_v2.so && \
        log "[INFO] create softlink libge_runner_v2.so in dir: $target_root_dir/$target_version_dir/$package/lib64.")

    (cd $target_root_dir/$target_version_dir/atc/lib64 && chmod u+w . && \
        ln -sf ../../x86_64-linux/lib64/libge_runner_v2.so libge_runner_v2.so && \
        log "[INFO] create softlink libge_runner_v2.so in dir: $target_root_dir/$target_version_dir/atc/lib64.")

    (cd $target_root_dir/$target_version_dir/fwkacllib/lib64 && chmod u+w . && \
        ln -sf ../../x86_64-linux/lib64/libge_runner_v2.so libge_runner_v2.so && \
        log "[INFO] create softlink libge_runner_v2.so in dir: $target_root_dir/$target_version_dir/fwkacllib/lib64.")

    (cd $target_root_dir/latest/$arch_linux/lib64 && chmod u+w . && \
        ln -sf ../../../$target_version_dir/$arch_linux/lib64/libge_runner_v2.so && \
        log "[INFO] create softlink libge_runner_v2.so in dir: $target_root_dir/latest/$arch_linux/lib64.")
}

IS_QUIET="n"
IS_ROLLBACK="n"
CUSTOM_INSTALL_DIR=""

while true
do
    case $1 in
    --quiet)
        IS_QUIET="y"
        shift
        ;;
    --rollback)
        IS_ROLLBACK="y"
        shift
        ;;
    --install-path=*)
        temp_install_path=$(echo $1 | cut -d"=" -f2-)
        CUSTOM_INSTALL_DIR=${temp_install_path%*/}
        shift
        ;;
    --full)
        shift
        ;;
    --*)
        shift
        ;;
    *)
        break
        ;;
    esac
done

if [ -n "${CUSTOM_INSTALL_DIR}" ]; then
    fstr="$(expr substr "$CUSTOM_INSTALL_DIR" 1 1)"
    if [ "$fstr" = "~" ]; then
        INSTALL_DIR="${HOME}$(echo "${CUSTOM_INSTALL_DIR}" | cut -d'~' -f 2-)"
    elif [ "$fstr" != "/" ]; then
        log "[ERROR] --install-path parameter requires an absolute path."
        exit 1
    else
        INSTALL_DIR=${CUSTOM_INSTALL_DIR}/latest
    fi
    log "[INFO] The custom installation directory path is ${INSTALL_DIR} ."
elif [ -n "${ASCEND_HOME_PATH}" ];then
    INSTALL_DIR=${ASCEND_HOME_PATH}
    log "[INFO] Use ASCEND_HOME_PATH installation directory ${ASCEND_HOME_PATH} ."
elif [ -d "${DEFAULT_TOOLKIT_INSTALL_DIR}" ]; then
    INSTALL_DIR=${DEFAULT_TOOLKIT_INSTALL_DIR}
    log "[INFO] Use default installation directory ${INSTALL_DIR} ."
elif [ -d "${DEFAULT_INSTALL_DIR}" ]; then
    INSTALL_DIR=${DEFAULT_INSTALL_DIR}
    log "[INFO] Use default installation directory ${INSTALL_DIR} ."
else
    log "[INFO] Please set the installation directory through --install-path"
fi

TARGET_LATEST_DIR=${INSTALL_DIR}

if [ ! -d "${TARGET_LATEST_DIR}" ]; then
    log "[ERROR] the $TARGET_LATEST_DIR dose not exist, please install the CANN package and set environment variables."
    exit 1
fi

TARGET_ROOT_DIR=$(dirname ${TARGET_LATEST_DIR})

get_base_package "base_package" "./version.info"
if [ "${base_package}" == "" ]; then
    log "[ERROR] get base_package failed."
    exit 1
fi

get_version_file "version_file" ${TARGET_LATEST_DIR} ${base_package}
if [ "${version_file}" == "" ]; then
    log "[ERROR] please install the basic package ${base_package} first."
    exit 1
fi

get_version_dir "version_dir" "${version_file}"
if [ "${version_dir}" == "" ]; then
    log "[ERROR] get version_dir failed."
    exit 1
fi

get_version_info "base_version_info" "${version_file}"
if [ "${base_version_info}" == "" ]; then
    log "[ERROR] get base_version_info failed."
    exit 1
fi

TARGET_DIR=${TARGET_ROOT_DIR}/${version_dir}

if [ ! -d "${TARGET_DIR}" ];then
    log "[ERROR] ${TARGET_DIR} does not exist."
    exit 1
fi

get_backup_dir "backup_dir" "./version.info"
if [ "${backup_dir}" == "" ]; then
    log "[ERROR] get backup_dir failed."
    exit 1
fi

get_version_info "current_version_info" "./version.info"
if [ "${current_version_info}" == "" ]; then
    log "[ERROR] get current_version_info failed."
    exit 1
fi

BACKUP_DIR=${TARGET_DIR}/backup/${backup_dir}
MANIFEST_FILE=${BACKUP_DIR}-manifest.ini
VERSION_INFO_FILE=${BACKUP_DIR}-version.info

if [ "${IS_ROLLBACK}" == "y" ];then
    if [ ! -d "${BACKUP_DIR}" ]; then
        log "[ERROR] the incremental package ${BACKUP_DIR} is not installed, rollback cannot be performed."
        exit 1
    fi
    log "[INFO] the rollback path is ${TARGET_DIR}."
    install_patches "backup" "${BACKUP_DIR}" "${TARGET_DIR}" "${MANIFEST_FILE}"
    delete_backup "${MANIFEST_FILE}" "${VERSION_INFO_FILE}" "${BACKUP_DIR}"
    log "[INFO] package rollback successfully!"
else
    if [ ! -d "${SOURCE_DIR}" ]; then
        log "[ERROR] the source code package ${SOURCE_DIR} does not exist, please check."
        exit 1
    fi
    check_version_compatiable "${base_version_info}" "${current_version_info}"
    log "[INFO] the installation path is ${TARGET_DIR}."
    delete_backup "${MANIFEST_FILE}" "${VERSION_INFO_FILE}" "${BACKUP_DIR}"
    copy_version_info "./version.info" "${VERSION_INFO_FILE}"
    create_softlink_for_libge_runner_v2 "${TARGET_ROOT_DIR}" "${version_dir}"
    install_patches "install" "${SOURCE_DIR}" "${TARGET_DIR}" "${MANIFEST_FILE}" "${BACKUP_DIR}"
    log "[INFO] package install successfully!"
fi




