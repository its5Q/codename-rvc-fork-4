@echo off
:: get hostname of your computer and save it to variable host.
FOR /F "usebackq" %%i IN (`hostname`) DO SET host=%%i

:: use port 25565 as the tensorboard port.
set port=25565

:: the link to the local tensorboard webpage is as follows
set address="http://%host%:%port%"

:: display the address in the command prompt
echo %address%

:: show the dragged folder path
echo %1

::start tensorboard
start "" tensorboard --logdir=%1 --host="%host%" --port=25565 

TIMEOUT /T 3 /NOBREAK >nul

:: use default browser to open tensorboard webpage
explorer %address%
