set -e
cd "$(dirname "$0")/.."
PY="${PY:-python}"
W="${WORKERS:-10}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
M="$PY -u -m unrolled_em"
CELLS=F32_500,F64_500,F128_500,P32_500,P64_500

$M.experiments.run --part all --workers $W
$M.experiments.run --part timing
$M.experiments.run --part percall
$M.experiments.export

$M.experiments.group3 --part all --workers $W

$M.experiments.exp_ab A
$M.experiments.exp_ab Call $W
$M.experiments.exp_ab_summary

$M.experiments.exp_lam --part all --cells F32_500,F64_500,F128_500 --workers $W

$M.experiments.exp_class --part train --workers $W
$M.experiments.exp_class --part test --workers $W
$M.experiments.exp_class --part summary

$M.experiments.exp_sqbt --part all --workers $W
$M.experiments.exp_sqbt --part all --lt supp --cells F32_500,F64_500 --workers $W

$M.experiments.exp_sqpkg_rest --part runs --workers $W
$M.experiments.exp_sqpkg_rest --part timing
$M.experiments.exp_sqpkg_rest --part summary

$M.experiments.smax_exp --part train --workers 3
$M.experiments.smax_exp --part test --workers $W
$M.experiments.smax_exp --part summary

$M.experiments.exp_cont --part all --workers $W
$M.experiments.exp_cont --part timing2
$M.experiments.ana_timing2 > results/v2c/timing2_summary.json

for p in switch sq16 qn stoprule d512; do
  $M.experiments.exp_review --part $p --workers $W
done
$M.experiments.exp_review --part t512 --workers 1
$M.experiments.exp_review --part summary

$M.experiments.exp_time16 --part hits256 --workers $W
$M.experiments.exp_time16 --part timing
$M.experiments.exp_time16 --part hits512 --workers $W
$M.experiments.exp_time16 --part t512
$M.experiments.exp_time16 --part summary

$M.experiments.sq_diag --part jac --workers $W
$M.experiments.sq_diag --part jac --cells F128_500 --js 0,1,2 --workers 3
$M.experiments.sq_diag --part jac --cells P64_500 --js 0,1,2,3,4 --workers 5
$M.experiments.sq_diag --part arms --workers $W
$M.experiments.sq_diag --part summary

$M.experiments.nf_sens --part run --workers $W
$M.experiments.nf_sens --part summary

$M.experiments.dare_check all
$M.experiments.check_switch $CELLS $W > results/v2c/switch_check.jsonl

$M.preliminary.classical --part lambda --workers $W
$M.preliminary.classical --part main --workers $W
$M.preliminary.classical --part verdict
$M.preliminary.direct --part main --workers $W
$M.preliminary.direct --part verdict

$M.experiments.paper

echo "all steps finished; numbers in paper/numbers.txt"
