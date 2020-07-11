@REM generate the VS project
cmake -G "Visual Studio 15 2017 Win64" -DENABLE_CUDA=ON -S base -B _build
@REM also build it
cmake --build _build --config Release
cmake --build _build --config Debug
pause
