set -e
cd "$(dirname "$0")/.."
PY="${PY:-python}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
$PY -u -m unrolled_em.experiments.export
$PY -u -m unrolled_em.experiments.exp_ab_summary
$PY -u -m unrolled_em.experiments.smax_exp --part summary
$PY -u -m unrolled_em.experiments.nf_sens --part summary
$PY -u -m unrolled_em.experiments.exp_time16 --part summary
$PY -u -m unrolled_em.experiments.paper
echo "numbers written to paper/numbers.txt, figure data to paper/data"
