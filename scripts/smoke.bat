@echo off
setlocal
cd /d "%~dp0.."
if "%PY%"=="" set PY=python
if "%WORKERS%"=="" set WORKERS=2
set PYTHONDONTWRITEBYTECODE=1
set PYTHONIOENCODING=utf-8
set V2_SMOKE=1
%PY% -u -m unrolled_em.experiments.run --part all --workers %WORKERS% || exit /b 1
%PY% -u -m unrolled_em.experiments.export || exit /b 1
echo smoke test finished: results/v2_smoke
exit /b 0
