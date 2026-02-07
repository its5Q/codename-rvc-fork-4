@echo off
:: get hostname of your computer and save it to variable host.
FOR /F "usebackq" %%i IN (`hostname`) DO SET host=%%i

:: use port 25565 as the tensorboard port.
set port=25565

:: the link to the local tensorboard webpage is as follows
set address="http://%host%:%port%"

:: display the address in the command prompt
echo %address%

:: ask user to key in the saved model directory, example "C:\tmp\mnist_model"
set /p UserInputPath=Key in model saved directory:

::start tensorboard
start "" tensorboard --logdir="%UserInputPath%" --host="%host%" --port=25565 

TIMEOUT /T 3 /NOBREAK >nul

:: use default browser to open tensorboard webpage
explorer %address%
