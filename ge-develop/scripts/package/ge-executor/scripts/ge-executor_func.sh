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

get_pkg_arch_name() {
    local scene_info="$curpath/../scene.info"
    if [ ! -f "$scene_info" ]; then
        local cur_date="$(date +'%Y-%m-%d %H:%M:%S')"
        echo "[ge-executor] [$cur_date] [ERROR]: $scene_info file cannot be found!"
        exit 1
    fi
    local arch="$(grep -iw arch "$scene_info" | cut -d"=" -f2- | awk '{print tolower($0)}')"
    if [ -z "$arch" ]; then
        local cur_date="$(date +'%Y-%m-%d %H:%M:%S')"
        echo "[ge-executor] [$cur_date] [ERROR]: var arch cannot be found in file $scene_info!"
        exit 1
    fi
    echo $arch
}

get_dir_mod() {
    local path="$1"
    stat -c %a "$path"
}

remove_dir_recursive() {
    local dir_start="$1"
    local dir_end="$2"
    if [ "$dir_end" = "$dir_start" ]; then
        return 0
    fi
    if [ ! -e "$dir_end" ]; then
        return 0
    fi
    if [ "x$(ls -A $dir_end 2>&1)" != "x" ]; then
        return 0
    fi
    local up_dir="$(dirname $dir_end)"
    local oldmod="$(get_dir_mod $up_dir)"
    chmod u+w "$up_dir"
    [ -n "$dir_end" ] && rm -rf "$dir_end"
    if [ $? -ne 0 ]; then
        chmod "$oldmod" "$up_dir"
        return 1
    fi
    chmod "$oldmod" "$up_dir"
    remove_dir_recursive "$dir_start" "$up_dir"
}

get_stub_libs_from_filelist() {
    awk -v arch_name="$arch_name" 'BEGIN {
        FS=","
        prefix=sprintf("^%s-linux/devlib/", arch_name)
        pat=sprintf("^%s-linux/devlib/(linux/%s/[^/]+\\.(so|a)$)", arch_name, arch_name)
    }
    {
        if (match($4, pat)) {
            sub(prefix, "", $4)
            print($4)
        }
    }' $curpath/filelist.csv
}

create_stub_softlink() {
    local install_path="$1"
    if [ ! -d "$install_path" ]; then
        return
    fi
    local arch_name="$pkg_arch_name"
    ([ -d "$install_path/${arch_name}-linux/devlib" ] && cd "$install_path/${arch_name}-linux/devlib" && {
        chmod u+w . && \
        for lib in $(get_stub_libs_from_filelist); do
            [ -f "$lib" ] && ln -sf "$lib" "$(basename $lib)"
        done
        chmod u-w .
    })
}

remove_stub_softlink() {
    local install_path="$1"
    if [ ! -d "$install_path" ]; then
        return
    fi
    local arch_name="$pkg_arch_name"
    ([ -d "$install_path/${arch_name}-linux/devlib" ] && cd "$install_path/${arch_name}-linux/devlib" && {
        chmod u+w . && basename --multiple $(get_stub_libs_from_filelist) | xargs --no-run-if-empty rm -rf
        chmod u-w .
    })
}

gen_acl_header() {
    local header="$1"
    local bname="$(basename $header)"
    if [ "$bname" = "acl_base_mdl.h" ] || [ "$bname" = "acl_mdl.h" ]; then
        cat - << 'EOF' > "$header" 2> /dev/null
/*!
 * This is an empty file, included for compatibility with old version.
 * If you want to use aclmdl api, please install the corresponding package to overrite this file.
 * */
EOF
    elif [ "$bname" = "acl_op.h" ]; then
        cat - << 'EOF' > "$header" 2> /dev/null
/*!
 * This is an empty file, included for compatibility with old version.
 * If you want to use aclop api, please install the corresponding package to overrite this file.
 * */
EOF
    fi
}

create_acl_empty_headers() {
    local install_dir="$1"
    local pkg_version_dir="$2"
    local arch_name="$pkg_arch_name"
    if [ -f "$install_dir/share/info/runtime/version.info" ] && \
       [ -f "$install_dir/share/info/runtime/ascend_install.info" ]; then # if runtime package installed
        local acl_headers_dir="$install_dir/$arch_name-linux/include/acl"
        [ -d "$acl_headers_dir" ] && chmod u+w "$acl_headers_dir" && \
        for header in acl_base_mdl.h acl_mdl.h acl_op.h; do
            gen_acl_header "$acl_headers_dir/$header"
        done
        chmod u-w "$acl_headers_dir"
    fi
}

pkg_arch_name="$(get_pkg_arch_name)"
