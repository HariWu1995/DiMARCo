@echo off

SET REPO_DIR="C:/Users/Mr. RIAH/Documents/GenAI/AI3R"
SET EXT_DIR=%REPO_DIR%\extensions

SET CONDA="C:\ProgramData\miniconda3"
CALL %CONDA%\Scripts\activate.bat %CONDA%

::CHDIR %REPO_DIR%

::CALL conda install pytorch==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
::CALL pip install opencv-contrib-python
::CALL pip install spatial-correlation-sampler --no-build-isolation

::CALL jupyter labextension install jupyterlab-plotly
::CALL jupyter labextension install @jupyter-widgets/jupyterlab-manager plotlywidget
CALL jupyter lab

:EXIT
echo.
echo Exiting.
pause

::TIMEOUT /T 1
