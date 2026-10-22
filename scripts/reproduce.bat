@echo off
setlocal
cd /d "%~dp0.."
if "%PY%"=="" set PY=python
if "%WORKERS%"=="" set WORKERS=10
set PYTHONDONTWRITEBYTECODE=1
set PYTHONIOENCODING=utf-8
set M=%PY% -u -m unrolled_em
set CELLS=F32_500,F64_500,F128_500,P32_500,P64_500

%M%.experiments.run --part all --workers %WORKERS% || exit /b 1
%M%.experiments.run --part timing || exit /b 1
%M%.experiments.run --part percall || exit /b 1
%M%.experiments.export || exit /b 1

%M%.experiments.group3 --part all --workers %WORKERS% || exit /b 1

%M%.experiments.exp_ab A || exit /b 1
%M%.experiments.exp_ab Call %WORKERS% || exit /b 1
%M%.experiments.exp_ab_summary || exit /b 1

%M%.experiments.exp_lam --part all --cells F32_500,F64_500,F128_500 --workers %WORKERS% || exit /b 1

%M%.experiments.exp_class --part train --workers %WORKERS% || exit /b 1
%M%.experiments.exp_class --part test --workers %WORKERS% || exit /b 1
%M%.experiments.exp_class --part summary || exit /b 1

%M%.experiments.exp_sqbt --part all --workers %WORKERS% || exit /b 1
%M%.experiments.exp_sqbt --part all --lt supp --cells F32_500,F64_500 --workers %WORKERS% || exit /b 1

%M%.experiments.exp_sqpkg_rest --part runs --workers %WORKERS% || exit /b 1
%M%.experiments.exp_sqpkg_rest --part timing || exit /b 1
%M%.experiments.exp_sqpkg_rest --part summary || exit /b 1

%M%.experiments.smax_exp --part train --workers 3 || exit /b 1
%M%.experiments.smax_exp --part test --workers %WORKERS% || exit /b 1
%M%.experiments.smax_exp --part summary || exit /b 1

%M%.experiments.exp_cont --part all --workers %WORKERS% || exit /b 1
%M%.experiments.exp_cont --part timing2 || exit /b 1
%M%.experiments.ana_timing2 > results\v2c\timing2_summary.json || exit /b 1

for %%p in (switch sq16 qn stoprule d512) do (
  %M%.experiments.exp_review --part %%p --workers %WORKERS% || exit /b 1
)
%M%.experiments.exp_review --part t512 --workers 1 || exit /b 1
%M%.experiments.exp_review --part summary || exit /b 1

%M%.experiments.exp_time16 --part hits256 --workers %WORKERS% || exit /b 1
%M%.experiments.exp_time16 --part timing || exit /b 1
%M%.experiments.exp_time16 --part hits512 --workers %WORKERS% || exit /b 1
%M%.experiments.exp_time16 --part t512 || exit /b 1
%M%.experiments.exp_time16 --part summary || exit /b 1

%M%.experiments.sq_diag --part jac --workers %WORKERS% || exit /b 1
%M%.experiments.sq_diag --part jac --cells F128_500 --js 0,1,2 --workers 3 || exit /b 1
%M%.experiments.sq_diag --part jac --cells P64_500 --js 0,1,2,3,4 --workers 5 || exit /b 1
%M%.experiments.sq_diag --part arms --workers %WORKERS% || exit /b 1
%M%.experiments.sq_diag --part summary || exit /b 1

%M%.experiments.nf_sens --part run --workers %WORKERS% || exit /b 1
%M%.experiments.nf_sens --part summary || exit /b 1

%M%.experiments.dare_check all || exit /b 1
if not exist results\v2c mkdir results\v2c
%M%.experiments.check_switch %CELLS% %WORKERS% > results\v2c\switch_check.jsonl || exit /b 1

%M%.preliminary.classical --part lambda --workers %WORKERS% || exit /b 1
%M%.preliminary.classical --part main --workers %WORKERS% || exit /b 1
%M%.preliminary.classical --part verdict || exit /b 1
%M%.preliminary.direct --part main --workers %WORKERS% || exit /b 1
%M%.preliminary.direct --part verdict || exit /b 1

%M%.experiments.paper || exit /b 1

echo all steps finished; numbers in paper\numbers.txt
exit /b 0
