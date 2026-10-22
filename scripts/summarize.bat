@echo off
setlocal
cd /d "%~dp0.."
if "%PY%"=="" set PY=python
set PYTHONDONTWRITEBYTECODE=1
set PYTHONIOENCODING=utf-8
%PY% -u -m unrolled_em.experiments.export || exit /b 1
%PY% -u -m unrolled_em.experiments.exp_ab_summary || exit /b 1
%PY% -u -m unrolled_em.experiments.smax_exp --part summary || exit /b 1
%PY% -u -m unrolled_em.experiments.nf_sens --part summary || exit /b 1
%PY% -u -m unrolled_em.experiments.exp_time16 --part summary || exit /b 1
%PY% -u -m unrolled_em.experiments.paper || exit /b 1
echo numbers written to paper\numbers.txt, figure data to paper\data
exit /b 0
