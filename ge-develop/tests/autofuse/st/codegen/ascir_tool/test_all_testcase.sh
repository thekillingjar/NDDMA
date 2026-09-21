# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

rm test_all.log
bash test_ascir.sh --mode=0 --case=reduce_max_ar_4dim_align                 | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_max_ar_4dim_noalign               | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_max_arar_4dim_r_align             | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_max_arar_4dim_r_noalign           | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_max_arar_4dim_after_fuse          | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_max_arar_4dim_after_fuse          | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_min_arar_4dim_r_align             | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_min_arar_4dim_r_noalign           | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_max_ra_4dim_align                 | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_max_ra_4dim_noalign               | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_min_ra_4dim_align                 | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_min_ra_4dim_noalign               | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_max_ara_3dim_a_align              | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_max_ara_3dim_a_noalign            | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_max_ara_3dim_a_align              | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_max_ara_3dim_a_noalign            | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_max_rar_3dim_r_align              | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_max_rar_3dim_r_noalign            | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_min_ar_2dim_r_align               | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_max_ar_2dim_r_noalign             | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_min_ar_2dim_r_align               | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_min_ar_2dim_r_noalign             | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_max_ra_2dim_a_align               | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_max_ra_2dim_a_align               | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_max_arararar_8dim_r_align         | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_max_arararar_8dim_r_noalign       | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_any_arar_4dim_r_align             | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_any_arar_4dim_r_noalign           | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_any_arar_4dim_r_align             | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_any_arar_4dim_r_noalign           | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_all_arar_4dim_r_align             | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_all_arar_4dim_r_noalign           | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_any_rara_4dim_a_align             | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_any_rara_4dim_a_noalign           | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_all_rara_4dim_a_align             | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_all_rara_4dim_a_noalign           | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_max_brc_elewise_noalign           | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_max_two_input_noalign             | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_sum_arar_4dim_align               | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_sum_arar_4dim_not_align           | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_sum_arar_4dim_after_fuse_align    | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_sum_arar_4dim_after_fuse_noalign  | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_mean_arar_4dim_r_align            | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_mean_arar_4dim_r_noalign          | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_sum_rara_4dim_align               | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_sum_rara_4dim_noalign             | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_mean_rara_4dim_a_align            | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_mean_rara_4dim_a_noalign          | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_sum_ara_3dim_a_align              | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_sum_ara_3dim_a_noalign            | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_mean_ara_3dim_a_align             | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_mean_ara_3dim_a_noalign           | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_sum_rar_3dim_r_align              | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_sum_rar_3dim_r_noalign            | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_mean_rar_3dim_r_align             | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_mean_rar_3dim_r_noalign           | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_sum_ar_2dim_align                 | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_sum_ar_2dim_not_align             | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_mean_ar_2dim_r_align              | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_mean_ar_2dim_r_noalign            | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_sum_ra_2dim_a_align               | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_sum_ra_2dim_a_noalign             | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_mean_ra_2dim_a_align              | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_mean_ra_2dim_a_noalign            | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_sum_arararar_8dim_r_align         | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_sum_arararar_8dim_r_noalign       | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_sum_rararara_8dim_a_align         | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_sum_rararara_8dim_a_noalign       | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_prod_arar_4dim_align              | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_prod_arar_4dim_not_align          | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_prod_arar_4dim_after_fuse_align   | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_prod_arar_4dim_after_fuse_noalign | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_prod_rara_4dim_align              | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_prod_rara_4dim_not_align          | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_prod_ara_3dim_a_align             | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_prod_ara_3dim_a_noalign           | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_prod_rar_3dim_r_align             | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_prod_rar_3dim_r_noalign           | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_prod_ar_2dim_align                | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_prod_ar_2dim_not_align            | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_prod_ra_2dim_a_align              | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_prod_ra_2dim_a_noalign            | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_prod_arararar_8dim_r_align        | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_prod_arararar_8dim_r_noalign      | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_prod_rararara_8dim_a_noalign      | tee -a test_all.log
#bash test_ascir.sh --mode=0 --case=reduce_sum_brc_elewise_noalign           | tee -a test_all.log  # reduceSum + broadcast 不支持，会挂掉
#bash test_ascir.sh --mode=0 --case=reduce_sum_two_input_noalign             | tee -a test_all.log  # reduceSum + broadcast 不支持，会挂掉
bash test_ascir.sh --mode=0 --case=reduce_max_norm_2dim_align               | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_max_norm_2dim_noalign             | tee -a test_all.log
#bash test_ascir.sh --mode=0 --case=reduce_sum_norm_2dim_align               | tee -a test_all.log  # reduceSum + broadcast 不支持，会挂掉
#bash test_ascir.sh --mode=0 --case=reduce_sum_norm_2dim_noalign             | tee -a test_all.log  # reduceSum + broadcast 不支持，会挂掉
bash test_ascir.sh --mode=0 --case=reduce_max_norm_2out_3dim_align          | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_max_norm_2out_3dim_noalign        | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_sum_norm_2out_3dim_align          | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_sum_norm_2out_3dim_noalign        | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_011_div                           | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_011_after                         | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=reduce_012                               | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=lastbrc_reduce_max                       | tee -a test_all.log
bash test_ascir.sh --mode=0 --case=lastbrc_reduce_mean                      | tee -a test_all.log


echo "==================[SUMMARY] follow is all testcase result==================="
grep "Result:" test_all.log
n=`grep "Result:" test_all.log  |wc -l`
if [ $n != 102 ]; then
  echo "[FAIL] output ${n} result, may sametestcase run abnormal"
  exit -1
fi
e=`grep "Result: ERROR" test_all.log  |wc -l`
if [ $n == 0 ]; then
  echo "[SUCCESS] all testcase run success"
  exit 0 
fi
echo "==================[FAIL] please check fail testcase==================="
exit -1
