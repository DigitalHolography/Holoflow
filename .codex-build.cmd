@call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul
@cmake --preset msvc-multi
@if errorlevel 1 exit /b %errorlevel%
@cmake --build --preset build-Release
