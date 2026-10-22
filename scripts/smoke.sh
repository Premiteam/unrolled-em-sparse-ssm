set -e
cd "$(dirname "$0")/.."
PY="${PY:-python}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 V2_SMOKE=1
$PY -u -m unrolled_em.experiments.run --part all --workers "${WORKERS:-2}"
$PY -u -m unrolled_em.experiments.export
echo "smoke test finished: results/v2_smoke"
