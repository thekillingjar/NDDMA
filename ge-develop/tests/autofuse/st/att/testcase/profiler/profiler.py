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
import re
import csv
import sys
import argparse
import numpy as np

sim_map = {
    "AIC_MTE1" : "aic_mte1_ratio",
    "AIC_MTE2" : "aic_mte2_ratio",
    "AIC_FIXPIPE" : "aic_fixpipe_ratio",
    "AIC_MAC" : "aic_mac_ratio",
    "AICORE_VEC" : "aiv_vec_ratio",
    "AIV_MTE2" : "aiv_mte2_ratio",
    "AIV_MTE3" : "aiv_mte3_ratio",
    "AIV_VEC" : "aiv_vec_ratio"
}

cycle_ref = {
    "AIC_MTE1" : "aic_total_cycles",
    "AIC_MTE2" : "aic_total_cycles",
    "AIC_FIXPIPE" : "aic_total_cycles",
    "AIC_MAC" : "aic_total_cycles",
    "AICORE_VEC" : "aiv_total_cycles",
    "AIV_MTE2" : "aiv_total_cycles",
    "AIV_MTE3" : "aiv_total_cycles",
    "AIV_VEC" : "aiv_total_cycles"
}

def GetCase(info):
    res = dict()
    pattern_getcase = r"\[INFO\][^\n]*\[([^\n\[\]]*?)\]Start GetTiling\.(.*?)(\[INFO\][^\n]*\[([^\n\[\]]*?)\]End GetTiling\.|\[ERROR\])"
    matches = re.findall(pattern_getcase, info, re.DOTALL)
    for match in matches:
        op_name = match[0]
        if op_name not in res:
            res[op_name] = []
        res[op_name].append(match[1])
    return res

def GetSchedGroup(op_name, infos):
    res = dict()
    pattern_getgraph = r"\[INFO\][^\n]*\[{}\]Start tiling for sched group (.*?)\.(.*?)\[INFO\][^\n]*\[{}\]End tiling for sched group\.".format(op_name, op_name)
    matches = re.findall(pattern_getgraph, infos, re.DOTALL)
    if len(matches) > 0:
        for match in matches:
            res[match[0]] = match[1]
    else:
        res["uniq_group"] = infos
    return res

def GetSchedResult(op_name, infos):
    res = dict()
    pattern_getcase = r"\[INFO\][^\n]*\[{}\]\[PROF\]Among all schedule results, (.*?) is the best choice\.".format(op_name)
    match = re.findall(pattern_getcase, infos, re.DOTALL)
    if len(match) > 0:
        return match[0][0]
    else:
        return None

def IsSchedResult(sched_group, sched_result):
    if sched_result is None:
        return True
    return sched_group.startswith("ScheduleResult" + sched_result)

def GetGraph(op_name, infos):
    res = dict()
    pattern_getgraph = r"\[DEBUG\][^\n]*\[{}\]Calculating the tiling data for tilingCaseId (\d*?)\.(.*?)\[DEBUG\][^\n]*\[{}\]Finish calculating the tiling data for tilingCaseId (\d*?)\.".format(op_name, op_name)
    matches = re.findall(pattern_getgraph, infos, re.DOTALL)
    res_getgraph = r"\[INFO\][^\n]*\[{}\]\[PROF\]Among the templates, tiling case (\d*?) of (.*?) is the best choice\.".format(op_name)
    result = re.findall(res_getgraph, infos, re.DOTALL)
    for match in matches:
        if (match[0] == match[2]):
            case_id = int(match[0])
            if case_id not in res:
                res[case_id] = [case_id == int(result[0][0]), match[1]]
    return res

def PassValidCheck(op_name, infos):
    pattern_keylog = r"\[DEBUG\][^\n]*\[{}\]Execute DoTiling\.".format(op_name)
    if re.search(pattern_keylog, infos):
        return True
    else:
        return False

def CheckContext(op_name, infos):
    pattern_keylog = r"\[INFO\][^\n]*\[{}\]Start context tiling\.".format(op_name)
    if re.search(pattern_keylog, infos):
        return True
    else:
        return False

def GetInputVars(op_name, tiling_key, infos):
    input_vars = dict()
    pattern_getinput = r"\[DEBUG\][^\n]*\[{}\]Start setting axis size for {}\.(.*?)\[DEBUG\][^\n]*\[{}\]End setting axis size for {}\.".format(op_name, tiling_key, op_name, tiling_key)
    matches = re.findall(pattern_getinput, infos, re.DOTALL)
    initiate_info = matches[0]
    pattern_getvar = r"Initiate (.*?) to (\d*?)\."
    var_info = re.findall(pattern_getvar, initiate_info)
    for var in var_info:
        input_vars[var[0]] = int(var[1])
    return input_vars

def FindIterNum(op_name, infos):
    pattern_keylog = r"iter : (\d+?)"
    matches = re.findall(pattern_keylog, infos)
    return len(matches)

def FindSolution(info):
    pattern_keylog = r"Feasible solution"
    if re.search(pattern_keylog, info):
        return True
    else:
        return False

def ObtainErrorLog(op_name, info):
    pattern_keylog = r"\[WARNING\][^\n]*\[{}\](.*?)\n".format(op_name)
    matches = re.findall(pattern_keylog, info)
    return matches[0]

def GetHardwareInfo(op_name, info):
    params = dict()
    pattern_getstatus = r"\[DEBUG\][^\n]*\[{}\]Set hardware params. (.*?)\n".format(op_name)
    pattern_getparams = r"(.*?) = (\d*?)\."
    matches = re.findall(pattern_getstatus, info)
    if len(matches) > 0:
        status = re.findall(pattern_getparams, matches[0])
        for item in status:
            params[item[0]] = int(item[1])
    return params

def SolutionAnalysis(op_name, tiling_key, tiling_info, tiling_res, df):
    tiling_data = dict()
    sim_cost = dict()
    sim_expr = dict()
    real_cost = dict()
    occupy = dict()
    cycle_map = dict()
    pattern_getsolution = r"\[DEBUG\][^\n]*\[{}\]Filling tilingdata for case{}\.\n(.*?)\[DEBUG\][^\n]*\[{}\]Objective value for case{} is (\d*?\.*?\d*?)\.\n".format(op_name, tiling_key, op_name, tiling_key)
    pattern_getstatus = r"\[DEBUG\][^\n]*\[{}\](.*?) = (\d*?\.*?\d*?)\n".format(op_name)
    pattern_split = r"\[DEBUG\][^\n]*\[{}\]Simulate the cost\.\n".format(op_name)
    pattern_expr = r"\[DEBUG\][^\n]*\[{}\]The expression of (.*?) is (.*?)\n".format(op_name)
    matches = re.findall(pattern_getsolution, tiling_info, re.DOTALL)[0]
    split_strs = re.split(pattern_split, matches[0], re.DOTALL)
    status = re.findall(pattern_getstatus, split_strs[0])
    for item in status:
        occupy[item[0]] = float(item[1])
    status = re.findall(pattern_getstatus, split_strs[1])
    for item in status:
        sim_cost[item[0]] = float(item[1])
        if df is not None:
            cur_df = df[op_name]
            if (tiling_res["best_case"]):
                real_cost[item[0]] = []
                for case in cur_df:
                    real_cost[item[0]].append(case[sim_map[item[0]]] * case[cycle_ref[item[0]]])
    exprs = re.findall(pattern_expr, split_strs[1])
    for item in exprs:
        sim_expr[item[0]] = item[1]
    pattern_res = r"\[DEBUG\][^\n]*\[{}\]The output of the solver for tilingCaseId case{} is:\n(.*?)\[DEBUG\][^\n]*\[{}\]The solver executed successfully\.".format(op_name, tiling_key, op_name)
    res_info = re.findall(pattern_res, tiling_info, re.DOTALL)[0]
    val_res = re.findall(pattern_getstatus, res_info)
    for item in val_res:
        tiling_data[item[0]] = int(item[1])
    tiling_res["obj"] = float(matches[1])
    tiling_res["solution"] = tiling_data
    tiling_res["occupy"] = occupy
    tiling_res["sim_cost"] = sim_cost
    tiling_res["sim_expr"] = sim_expr
    tiling_res["real_cost"] = real_cost

def FormatNumber(num, simplify = False):
    ret_num = num
    if isinstance(num, float):
        if int(num) == num:
            ret_num = int(num)
        else:
            ret_num = round(num, 2)
    if simplify:
        if (ret_num >= 1024 * 1024):
            return str(round(ret_num / 1024 / 1024, 2)) + "m"
        if (ret_num >= 1024):
            return str(round(ret_num / 1024, 2)) + "k"
    return str(ret_num)

def PrintDict(used_dict, compare_dict = None, header = [], simplify = False):
    print("------------------------------------------------")
    if (len(header) > 0):
        header_str = "      "
        for header_item in header:
            header_str += "{:<15}".format(header_item)
        print(header_str)
        print("------------------------------------------------")
    for key, value in used_dict.items():
        if compare_dict is not None:
            print("      {:<15}{:<15}{:<15}".format(key, FormatNumber(value, simplify), FormatNumber(compare_dict[key], simplify)))
        else:
            print("      {:<15}{:<15}".format(key, FormatNumber(value, simplify)))
    print("------------------------------------------------")
    print("")

def DisplayRes(tiling_ret, has_summary):
    for op_name, op_info in tiling_ret.items():
        print("Op Name: {:<20}".format(op_name))
        for case_id, case_info in op_info.items():
            print(" Case: {:<20}".format(case_id))
            for sched_group, sched_info in case_info.items():
                if sched_group != "uniq_group":
                    sched_str = "  Sched Group: {}".format(sched_group)
                    if sched_info[0]:
                        sched_str += "(chosen)"
                    print(sched_str)
                for tiling_res in sched_info[1].values():
                    print("   tilingCaseId: {:<20}".format(tiling_res["tilingCaseId"]))
                    if tiling_res["pass_check"]:
                        if "input_var" in tiling_res:
                            print("    input vars:")
                            PrintDict(tiling_res["input_var"])
                        print("    hardware params:")
                        PrintDict(tiling_res["hardware_param"], simplify = True)
                        print("    exe_iter: {:<20}".format(tiling_res["exec_iter"]))
                        if tiling_res["has_solution"]:
                            print("    Solution:")
                            PrintDict(tiling_res["solution"])
                            print("    Estimate:")
                            if has_summary:
                                PrintDict(tiling_res["sim_cost"], compare_dict = tiling_res["real_cost"], header = ["pipe", "sim_cost", "real_cost"])
                            else:
                                PrintDict(tiling_res["sim_cost"], header = ["pipe", "sim_cost"])
                            PrintDict(tiling_res["occupy"], header = ["loc", "occupy"], simplify = True)
                        else:
                            print("    Status: {}".format(tiling_res["error_msg"]))
                    else:
                        print("    Status: {}".format(tiling_res["error_msg"]))
    print("")

def AnalysisInfo(info, df):
    case_infos = GetCase(info)
    op_ret = dict()
    for op_name, case_list in case_infos.items():
        input_ret = dict()
        for i, case_info in enumerate(case_list):
            cur_input = "case{}".format(i + 1)
            case_ret = dict()
            sched_result = GetSchedResult(op_name, case_info)
            sched_groups = GetSchedGroup(op_name, case_info)
            for sched_group, sched_info in sched_groups.items():
                sched_ret = dict()
                tiling_infos = GetGraph(op_name, case_info)
                for tiling_case, tiling_infomation in tiling_infos.items():
                    cur_case = "tilingCase{}".format(tiling_case)
                    tiling_res = {}
                    best_case, tiling_info = tiling_infomation
                    tiling_res["best_case"] = best_case
                    tiling_res["tilingCaseId"] = tiling_case
                    tiling_res["error_msg"] = ""
                    tiling_res["hardware_param"] = GetHardwareInfo(op_name, tiling_info)
                    
                    tiling_res["pass_check"] = True
                    if (CheckContext(op_name, tiling_info)):
                        tiling_res["input_var"] = GetInputVars(op_name, tiling_case, tiling_info)
                    tiling_res["exec_iter"] = FindIterNum(op_name, tiling_info)
                    tiling_res["has_solution"] = FindSolution(tiling_info)
                    if (tiling_res["has_solution"]):
                        SolutionAnalysis(op_name, tiling_case, tiling_info, tiling_res, df)
                    else:
                        tiling_res["error_msg"] = ObtainErrorLog(op_name, tiling_info)
                    sched_ret[cur_case] = tiling_res
                case_ret[sched_group] = [IsSchedResult(sched_group, sched_result), sched_ret]
            input_ret[cur_input] = case_ret
        op_ret[op_name] = input_ret
    return op_ret

def GetDataFrame(folder_path):
    csv_path = os.path.join(folder_path, "mindstudio_profiler_output")
    if os.path.exists(csv_path):
        for file in os.listdir(csv_path):
            if file.endswith("op_summary"):
                profiler_path = os.path.join(csv_path, file)
                data_frame = dict()
                header = dict()
                with open(profiler_path, 'r') as file:
                    reader = csv.reader(file)
                    for i, row in enumerate(reader):
                        if i == 0:
                            for j, col_name in enumerate(row):
                                header[col_name] = j
                        else:
                            if row[header["op_name"]] not in data_frame:
                                data_frame[row[header["op_name"]]] = []
                            col_info = dict()
                            for key_name in sim_map.values():
                                col_info[key_name] = row[header[key_name]]
                            for key_name in cycle_ref.values():
                                col_info[key_name] = row[header[key_name]]
                            data_frame[row[header["op_name"]]].append(col_info)
                return data_frame
    return None

def OutputRes(op_ret, has_summary, output_path):
    header = ["op_name", "sched group", "tilingCaseId", "tilingData", "block_dim"]
    for pipetype in sim_map.keys():
        header += ["sim_" + pipetype, pipetype + "_expr"]
        if has_summary:
            header.append("real_" + pipetype)
    data = [header]
    for op_name, input_ret in op_ret.items():
        for cur_input, case_ret in input_ret.items():
            for cur_sched, sched_ret in case_ret.items():
                if not sched_ret[0]:
                    continue
                for cur_case, tiling_res in sched_ret[1].items():
                    if (not tiling_res["best_case"]) or (len(tiling_res["sim_cost"]) == 0):
                        continue
                    cur_cost = []
                    col_info = [op_name, cur_sched, tiling_res["tilingCaseId"]]
                    tiling_info = ""
                    for key, value in tiling_res["solution"].items():
                        if len(tiling_info) > 0:
                            tiling_info += ","
                        tiling_info += "{} = {}".format(key, value)
                    col_info.append(tiling_info)
                    if ("block_dim" in tiling_res["occupy"]):
                        col_info.append(tiling_res["occupy"]["block_dim"])
                    else:
                        col_info.append(0)
                    if has_summary:
                        for value in tiling_res["real_cost"].values():
                            test_num = len(value)
                            break
                        for i in range(test_num):
                            cur_col_info = [item for item in has_summary]
                            for pipetype in sim_map.keys():
                                if pipetype in tiling_res["sim_cost"]:
                                    cur_col_info += [tiling_res["sim_cost"][pipetype], tiling_res["sim_expr"][pipetype]]
                                else:
                                    cur_col_info += [0, 0]
                                if pipetype in tiling_res["real_cost"]:
                                    cur_col_info.append(tiling_res["real_cost"][pipetype][i])
                                else:
                                    cur_col_info.append(0)
                            data.append(cur_col_info)
                    else:
                        for pipetype in sim_map.keys():
                            if pipetype in tiling_res["sim_cost"]:
                                col_info += [tiling_res["sim_cost"][pipetype], tiling_res["sim_expr"][pipetype]]
                            else:
                                col_info += [0, 0]
                        data.append(col_info)
                    break
    with open(output_path, mode = "w", newline = '') as f:
        writer = csv.writer(f)
        writer.writerows(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', type=str, help='Path to the tiling log')
    parser.add_argument('--summary_path', type=str, default='./PROF', help='Path to the folder containing the profiling data')
    args = parser.parse_args()
    with open(args.log_path, 'r', encoding = 'utf-8') as file:
        info = file.read()
    data_frame = GetDataFrame(args.summary_path)
    if info is not None:
        has_summary = (data_frame is not None)
        op_ret = AnalysisInfo(info, data_frame)
        DisplayRes(op_ret, has_summary)
        OutputRes(op_ret, has_summary, "./output.csv")