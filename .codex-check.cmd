@call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul
@cmake --build --preset build-Release > build\msvc-multi\verification-build.log 2>&1
@if errorlevel 1 exit /b %errorlevel%
@ctest --preset test-Release > build\msvc-multi\verification-tests.log 2>&1
@exit /b %errorlevel%
