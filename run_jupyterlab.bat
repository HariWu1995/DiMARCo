@echo off

SET CONDA="C:\ProgramData\miniconda3"
CALL %CONDA%\Scripts\activate.bat %CONDA%

CD "C:/Users/Mr. RIAH/Documents/GenAI/AI3R"
CALL jupyter lab

:EXIT
echo.
echo Exiting.
pause

::TIMEOUT /T 1
