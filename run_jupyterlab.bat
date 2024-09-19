@echo off

SET CONDA="C:\ProgramData\miniconda3"
CALL %CONDA%\Scripts\activate.bat %CONDA%

::CALL pip uninstall opencv-python
::CALL pip uninstall opencv-contrib-python
::CALL pip install opencv-contrib-python

CD "C:/Users/Mr. RIAH/Documents/GenAI/AI3R"
::CALL jupyter labextension install jupyterlab-plotly
::CALL jupyter labextension install @jupyter-widgets/jupyterlab-manager plotlywidget
CALL jupyter lab

:EXIT
echo.
echo Exiting.
pause

::TIMEOUT /T 1
