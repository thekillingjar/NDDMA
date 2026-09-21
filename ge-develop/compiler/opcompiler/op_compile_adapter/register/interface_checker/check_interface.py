#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
# Get the catalog of interface specifications
specs_dir = os.path.realpath(os.path.dirname(__file__) + "/../interface_spec")
# Get the root directory of TBE
tbe_root = os.path.realpath(os.path.dirname(__file__) + "/../")


class FuncIntfSpec:
    """
    The specification definition of the function API, which mainly includes
    the name of the function, and the input parameters of the function
    Temporarily unable to verify the parameters
    """
    def __init__(self, func_name, param_list):
        self.func_name = func_name
        self.param_list = param_list

    def print_detail(self):
        print("func name: %s, param_list: [%s]" % (self.func_name, ",".join(self.param_list)))


class ClassIntfSpec:
    """
    The interface specification definition of the class type, for example,
    the various APIs of Tik are encapsulated in the class types mainly include:
    the name of the class, the parent class of the class, and the api function
    interface provided by the class
    """
    def __init__(self, name, supper_classes):
        self.class_name = name
        self.supper_classes = supper_classes
        self.func_list = {}
        self.param_list = {}

    def add_func_spec(self, func_info: FuncIntfSpec):
        self.func_list[func_info.func_name] = func_info

    def print_detail(self):
        print("-------------------------------------------")
        print("class name: %s, supper classes: %s" % (self.class_name, ",".join(self.supper_classes)))
        print("func list:")
        for (idx, func_spec) in self.func_list.items():
            func_spec.print_detail()
        print("params list:")
        for param_spec in self.param_list:
            param_spec.print_detail()
        print("-------------------------------------------")


class GlobalVarSpec:
    """
    Global variable definition, such as NAME_INDEX in the elewise_compute interface
    Types mainly include: global variable name, global variable value
    """
    def __init__(self, name, values):
        self.global_var_name = name
        self.global_var_values = values
        self.func_list = {}
        self.param_list = {}


class FileSpec:
    """
    File specification definition, corresponding to a source file
    """
    def __init__(self, spec_file_name, source_file_name=""):
        self.spec_file_name = spec_file_name
        self.source_file_name = source_file_name
        self.class_specs = {}
        self.func_specs = {}
        self.global_var_spec = {}

    def add_class_spec(self, spec: ClassIntfSpec):
        self.class_specs[spec.class_name] = spec

    def add_func_spec(self, spec: FuncIntfSpec):
        self.func_specs[spec.func_name] = spec

    def add_global_var_spec(self, spec: GlobalVarSpec):
        self.global_var_spec[spec.global_var_name] = spec

    def print_detail(self):
        print("===========================================")
        print("file name: %s" % self.spec_file_name)
        print("class specs:")
        for _, class_spec in self.class_specs.items():
            class_spec.print_detail()
        print("func specs:")
        for _, func_spec in self.func_specs.items():
            func_spec.print_detail()
        print("===========================================")


def get_tree_idx(str_info: str):
    """
    The level of python code is guaranteed by indentation,
    and here is a few levels of indentation.
    There is actually no requirement for the indentation
    of python code, as long as the indentation at the same
    level is the same.
    It is assumed here that our code format is standardized,
    according to 4 spaces.
    """
    TAB_STR = "    "
    if str_info.startswith(TAB_STR):
        return 1 + get_tree_idx(str_info[4:])
    else:
        return 1


def build_file_spec(spec_lines, file_spec: FileSpec):
    """
    Analyze the specification definition according to the code in the file.
    """
    spec_tree = [file_spec, ]
    for spec_line in spec_lines:
        spec_line = spec_line.rstrip()

        if spec_line.startswith("#"):
            continue
        elif "class " in spec_line:
            # Indicates that a class is defined
            tree_idx = get_tree_idx(spec_line)
            if "(" in spec_line:
                class_name = spec_line[6:spec_line.index("(")]
                super_classes = spec_line[spec_line.index("(")+1: spec_line.index(")")].split(",")
            else:
                class_name = spec_line[6:-1]
                super_classes = []
            class_spec = ClassIntfSpec(class_name, super_classes)
            if len(spec_tree) < (tree_idx + 1):
                spec_tree.append(class_spec)
            else:
                spec_tree[tree_idx] = class_spec
            spec_tree[tree_idx-1].add_class_spec(class_spec)
        elif "def " in spec_line:
            # Indicates that a function api is defined
            tree_idx = get_tree_idx(spec_line)
            spec_line = spec_line.strip()
            func_name = spec_line[4:spec_line.index("(")]
            param_list = [x.strip() for x in spec_line[spec_line.index("(")+1:spec_line.index(")")].split(",")]
            func_spec = FuncIntfSpec(func_name, param_list)
            if len(spec_tree) < (tree_idx + 1):
                spec_tree.append(func_spec)
            else:
                spec_tree[tree_idx] = func_spec
            spec_tree[tree_idx-1].add_func_spec(func_spec)
        elif is_global_variable(spec_line):
            tree_idx = get_tree_idx(spec_line)
            name = spec_line[:spec_line.index("=")].rstrip()
            values = spec_line[(len(name) + 3):]
            global_var_spec = GlobalVarSpec(name, values)
            if len(spec_tree) < (tree_idx + 1):
                spec_tree.append(global_var_spec)
            else:
                spec_tree[tree_idx] = global_var_spec
            spec_tree[tree_idx - 1].add_global_var_spec(global_var_spec)


def is_global_variable(spec_line):
    if spec_line.startswith("    "):
        return False
    if "=" not in spec_line:
        return False
    name = spec_line[:spec_line.index("=")].rstrip()
    name_split = name.split("_")
    for i in name_split:
        if not i.isupper():
            return False
    return True


def get_spec_info_list():
    """
    Scan out all interface specification definitions
    from the interface specification folder
    """
    def _get_class_spec_info(spec_line, spec_tree):
        tree_idx = get_tree_idx(spec_line)
        if "(" in spec_line:
            class_name = spec_line[6:spec_line.index("(")]
            super_classes = spec_line[spec_line.index("(") + 1: spec_line.index(")")].split(",")
        else:
            class_name = spec_line[6:-1]
            super_classes = []
        class_spec = ClassIntfSpec(class_name, super_classes)
        if len(spec_tree) < (tree_idx + 1):
            spec_tree.append(class_spec)
        else:
            spec_tree[tree_idx] = class_spec
        spec_tree[tree_idx - 1].add_class_spec(class_spec)

    def _get_def_spec_info(spec_line, spec_tree):
        tree_idx = get_tree_idx(spec_line)
        spec_line = spec_line.strip()
        func_name = spec_line[4:spec_line.index("(")]
        param_list = [x.strip() for x in spec_line[spec_line.index("(") + 1:spec_line.index(")")].split(",")]
        func_spec = FuncIntfSpec(func_name, param_list)
        if len(spec_tree) < (tree_idx + 1):
            spec_tree.append(func_spec)
        else:
            spec_tree[tree_idx] = func_spec
        spec_tree[tree_idx - 1].add_func_spec(func_spec)

    def _get_global_var_spec_info(spec_line, spec_tree):
        tree_idx = get_tree_idx(spec_line)
        name = spec_line[:spec_line.index("=")].rstrip()
        values = spec_line[(len(name) + 3):]
        global_var_spec = GlobalVarSpec(name, values)
        if len(spec_tree) < (tree_idx + 1):
            spec_tree.append(global_var_spec)
        else:
            spec_tree[tree_idx] = global_var_spec
        spec_tree[tree_idx - 1].add_global_var_spec(global_var_spec)

    def _process_dir(specs, path, file_spec_list):
        for spec_file in specs:
            spec_tree = []
            if not spec_file.endswith("pyh"):
                continue
            with open(os.path.join(path, spec_file)) as sf:
                spec_lines = sf.readlines()

            for spec_line in spec_lines:
                spec_line = spec_line.rstrip()
                if "# source file:" in spec_line:
                    spec_source_file = spec_line[14:].strip()
                    file_spec = FileSpec(spec_file, spec_source_file)
                    file_spec_list.append(file_spec)
                    if len(spec_tree) < 1:
                        spec_tree.append(file_spec)
                    else:
                        spec_tree[0] = file_spec
                elif spec_line.startswith("#"):
                    continue

                elif "class " in spec_line:
                    _get_class_spec_info(spec_line, spec_tree)

                elif "def " in spec_line:
                    _get_def_spec_info(spec_line, spec_tree)

                elif is_global_variable(spec_line):
                    _get_global_var_spec_info(spec_line, spec_tree)

    specs = os.listdir(specs_dir)
    sub_dirs = traversal_dir(specs_dir)
    file_spec_list = []
    _process_dir(specs, specs_dir, file_spec_list)
    for sub_dir in sub_dirs:
        _process_dir(os.listdir(sub_dir), sub_dir, file_spec_list)
        sub_sub_dirs = traversal_dir(sub_dir)
        for sub_sub_dir in sub_sub_dirs:
            _process_dir(os.listdir(sub_sub_dir), sub_sub_dir, file_spec_list)
    return file_spec_list


def traversal_dir(path):
    sub_dirs = []
    if os.path.exists(path):
        temp_dirs = os.listdir(path)
        for file in temp_dirs:
            temp_path = os.path.join(path, file)
            if os.path.isdir(temp_path):
                sub_dirs.append(temp_path)

    return sub_dirs


def get_spec_from_file(file_path):
    with open(file_path) as ff:
        lines = ff.readlines()

    def _get_new_lines(line, line_end, new_lines):
        if "def " in line or "class " in line:
            if not line.rstrip().endswith(":"):
                line_end = False
                new_lines.append(line.rstrip())
            else:
                new_lines.append(line.rstrip())
        elif is_global_variable(line):
            new_lines.append(line.rstrip())
        return line_end

    new_lines = []
    block_comment = False
    line_end = True
    for line in lines:
        tmp_line = line.strip()
        if block_comment:
            # The front is a block comment, until the end of the block comment is obtained
            if "\"\"\"" in tmp_line:
                block_comment = False
        else:
            if tmp_line.startswith("#"):
                continue
            if "\"\"\"" in tmp_line and tmp_line.count("\"") == 3:
                # Contains 3 consecutive quotation marks, which are block comments,
                # where block comments are generally independent lines
                block_comment = True
                continue
            if "#" in line:
                # Remove end-of-line comments
                line = line[:line.index("#")]
            if line_end:
                # Because the function and class declaration may wrap,
                # if the definition of the previous function has not ended,
                # continue to splice it up, remove the wrap
                line_end = _get_new_lines(line, line_end, new_lines)
            else:
                new_lines[-1]+=line.strip()
                # Both function definitions and class definitions
                # end with a colon, check the colon
                if line.rstrip().endswith(":"):
                    line_end = True

    return new_lines


def remove_sub_func_under_func(lines):
    """
    Because get_spec_from_file extracts the interface
    and class definitions in the file.
    Since the function can also be defined inside the function,
    it will also be extracted here, so the function
    inside the function is removed.
    """
    tree_info = [""]
    new_lines = []
    for line in lines:
        idx = get_tree_idx(line)
        if line.strip().startswith("def "):
            type_name = "func"
        elif line.strip().startswith("class "):
            type_name = "class"
        elif is_global_variable(line):
            name = line[:line.index("=")].rstrip()
            type_name = name
        if type_name == "func" and tree_info[min(len(tree_info)-1, idx-1)] == "func":
            continue
        if len(tree_info) <= idx:
            tree_info.append(type_name)
        else:
            tree_info[idx] = type_name
        new_lines.append(line)
    return new_lines


def check_source_file_match(defined_spec: FileSpec):
    """
    Verify that the source code complies with the specification definition
    """
    source_file_path = os.path.realpath(os.path.join(tbe_root, defined_spec.source_file_name))
    lines = get_spec_from_file(source_file_path)
    new_lines = remove_sub_func_under_func(lines)

    spec_in_source = FileSpec("", source_file_path)
    build_file_spec(new_lines, spec_in_source)

    return compare_file_spec(defined_spec, spec_in_source)


def compare_func_spec(spec1: FuncIntfSpec, spec2: FuncIntfSpec, file_name1="", file_name2=""):
    params_1 = sorted([x.strip() for x in spec1.param_list])
    params_2 = sorted([x.strip() for x in spec2.param_list])
    if params_1 != params_2:
        print("[EEEE] compare \"%s\" func failed" % spec1.func_name)
        print("[EEEE] file path: \"%s\"" % file_name1)
        print("[EEEE] param list: \"%s\"" % ",".join(params_1))
        print("[EEEE] file path: \"%s\"" % file_name2)
        print("[EEEE] param list: \"%s\"" % ",".join(params_2))
        return False
    else:
        print("[====] compare \"%s\" func success" % spec1.func_name)
        return True


def build_diff_list_result(func_names_1, func_names_2):
    func_names_1 = func_names_1[:]
    func_names_2 = func_names_2[:]
    diff_print_name1 = []
    diff_print_name2 = []
    len_1 = len(func_names_1)
    len_2 = len(func_names_2)
    i = 0
    j = 0
    while i<len_1 or j<len_2:
        if i < len_1 and j < len_2:
            if func_names_1[i] == func_names_2[j]:
                diff_print_name1.append(func_names_1[i])
                diff_print_name2.append(func_names_2[j])
                i += 1
                j += 1
            else:
                if func_names_1[i] in func_names_2:
                    diff_print_name1.append("_"*len(func_names_2[j]))
                    diff_print_name2.append(func_names_2[j])
                    j += 1
                elif func_names_2[j] in func_names_1:
                    diff_print_name2.append("_"*len(func_names_1[i]))
                    diff_print_name1.append(func_names_1[i])
                    i += 1
                else:
                    diff_print_name1.append(func_names_1[i])
                    diff_print_name2.append("_" * len(func_names_1[i]))
                    diff_print_name1.append("_" * len(func_names_2[j]))
                    diff_print_name2.append(func_names_2[j])
                    i += 1
                    j += 1
        else:
            if i == len_1:
                for item in func_names_2[j:]:
                    diff_print_name1 += ["_" * len(item)]
                    diff_print_name2.append(item)
                j = len_2
            elif j == len_2:
                for item in func_names_1[i:]:
                    diff_print_name1.append(item)
                    diff_print_name2 += ["_" * len(item)]
                i = len_1

    return diff_print_name1, diff_print_name2


def compare_class_spec(spec1: ClassIntfSpec, spec2: ClassIntfSpec, file_name1="", file_name2=""):
    compare_matched = True
    print("[====] compare class: \"%s\"" % spec1.class_name)
    func_list_1 = spec1.func_list
    func_list_2 = spec2.func_list
    # remove private func api
    func_names_1 = sorted([x for x in func_list_1.keys() if not x.startswith("__") and not x.startswith("_")])
    func_names_2 = sorted([x for x in func_list_2.keys() if not x.startswith("__") and not x.startswith("_")])
    diff_print_name1 = func_names_1[:]
    if func_names_1 != func_names_2:
        compare_matched = False
        diff_print_name1, diff_print_name2 = build_diff_list_result(func_names_1, func_names_2)

        print("[EEEE] func list in class is different")
        print("[EEEE] file path: \"%s\"" % file_name1)
        print("[EEEE] func list is: \"%s\"" % ",".join(diff_print_name1))
        print("[EEEE] file path: \"%s\"" % file_name2)
        print("[EEEE] func list is: \"%s\"" % ",".join(diff_print_name2))

    for func_name in diff_print_name1:
        if not func_name.startswith("_"):
            if not compare_func_spec(func_list_1[func_name], func_list_2[func_name], file_name1, file_name2):
                compare_matched = False
    print("[====] compare class: \"%s\" end, result: \"%s\"" % (spec1.class_name, "Success" if compare_matched else "Fail"))
    return compare_matched


def compare_global_var_spec(spec1: GlobalVarSpec, spec2: GlobalVarSpec, file_name1="", file_name2=""):
    values_1 = sorted([x.strip() for x in spec1.global_var_values])
    values_2 = sorted([x.strip() for x in spec2.global_var_values])
    if values_1 != values_2:
        print("[EEEE] compare \"%s\" global_var failed" % spec1.global_var_name)
        print("[EEEE] file path: \"%s\"" % file_name1)
        print("[EEEE] param list: \"%s\"" % ",".join(values_1))
        print("[EEEE] file path: \"%s\"" % file_name2)
        print("[EEEE] param list: \"%s\"" % ",".join(values_2))
        return False
    else:
        print("[====] compare \"%s\" global_var success" % spec1.global_var_name)
        return True


def compare_file_spec(spec1: FileSpec, spec2: FileSpec):
    print("\n[====] compare interface define: \"%s\", source file: \"%s\"" % (spec1.spec_file_name, spec1.source_file_name))
    compare_matched = True

    def _compare_global_var(compare_matched, spec1, spec2):
        global_var_list_1 = spec1.global_var_spec
        global_var_list_2 = spec2.global_var_spec
        global_var_name1 = sorted([x.strip() for x in global_var_list_1.keys()])
        global_var_name2 = sorted([x.strip() for x in global_var_list_2.keys()])
        diff_global_var_name_1 = global_var_name1
        diff_global_var_name_2 = global_var_name2
        if global_var_name1 != global_var_name2:
            compare_matched = False
            diff_global_var_name_1, diff_global_var_name_2 = build_diff_list_result(global_var_name1, global_var_name2)
            print("[EEEE] global_var in file is different")
            print("[EEEE] file path: \"%s\"" % spec1.spec_file_name)
            print("[EEEE] class list is: %s" % ",".join(diff_global_var_name_1))
            print("[EEEE] file path: \"%s\"" % spec1.source_file_name)
            print("[EEEE] class list is: %s" % ",".join(diff_global_var_name_2))
        for name1, name2 in zip(diff_global_var_name_1, diff_global_var_name_2):
            if name1 != name2:
                compare_matched = False
            elif not compare_global_var_spec(global_var_list_1[name1], global_var_list_2[name2], spec1.spec_file_name, spec1.source_file_name):
                compare_matched = False
        return compare_matched

    def _compare_func(compare_matched, spec1, spec2):
        func_list_1 = spec1.func_specs
        func_list_2 = spec2.func_specs
        func_names_1 = sorted([x for x in func_list_1.keys() if not x.startswith("__") and not x.startswith("_")])
        func_names_2 = sorted([x for x in func_list_2.keys() if not x.startswith("__") and not x.startswith("_")])
        diff_print_name1 = func_names_1
        diff_print_name2 = func_names_2
        if func_names_1 != func_names_2:
            compare_matched = False
            diff_print_name1, diff_print_name2 = build_diff_list_result(func_names_1, func_names_2)
            print("[EEEE] func list in file is different")
            print("[EEEE] file path: \"%s\"" % spec1.spec_file_name)
            print("[EEEE] func list is: \"%s\"" % ",".join(diff_print_name1))
            print("[EEEE] file path: \"%s\"" % spec1.source_file_name)
            print("[EEEE] func list is: \"%s\"" % ",".join(diff_print_name2))
        for func_name1, func_name2 in zip(diff_print_name1, diff_print_name2):
            if func_name1 != func_name2:
                compare_matched = False
            elif not compare_func_spec(func_list_1[func_name1], func_list_2[func_name2], spec1.spec_file_name, spec1.source_file_name):
                compare_matched = False
        return compare_matched

    def _compare_class(compare_matched, spec1, spec2):
        class_list_1 = spec1.class_specs
        class_list_2 = spec2.class_specs
        class_name1 = sorted([x.strip() for x in class_list_1.keys()])
        class_name2 = sorted([x.strip() for x in class_list_2.keys()])
        diff_class_name_1 = class_name1
        if len(class_name1) == 0:
            for class_ in class_name2:
                if not class_.startswith("_"):
                    compare_matched = False
                    return compare_matched
            return compare_matched
        if class_name1 != class_name2:
            compare_matched = False
            diff_class_name_1, diff_class_name2 = build_diff_list_result(class_name1, class_name2)
            print("[EEEE] class in file is different")
            print("[EEEE] file path: \"%s\"" % spec1.spec_file_name)
            print("[EEEE] class list is: %s" % ",".join(diff_class_name_1))
            print("[EEEE] file path: \"%s\"" % spec1.source_file_name)
            print("[EEEE] class list is: %s" % ",".join(diff_class_name2))
            return compare_matched
        for class_name in diff_class_name_1:
            if not compare_class_spec(class_list_1[class_name], class_list_2[class_name], spec1.spec_file_name, spec1.source_file_name):
                compare_matched = False
        return compare_matched

    compare_matched = _compare_global_var(compare_matched, spec1, spec2)

    compare_matched = _compare_func(compare_matched, spec1, spec2)

    compare_matched = _compare_class(compare_matched, spec1, spec2)

    print("[====] compare result: \"%s\". interface define: \"%s\", source file: \"%s\"" % ("Success" if compare_matched else "Fail", spec1.spec_file_name, spec1.source_file_name))
    return compare_matched


def check_all():
    check_result = True
    spec_define_list = get_spec_info_list()
    for spec in spec_define_list:
        if not check_source_file_match(spec):
            check_result = False
    return check_result


if __name__ == "__main__":
    if not check_all():
        exit(-1)
