@echo off
setlocal enabledelayedexpansion

:: Define the build directory
set BUILD_DIR=bin\win

:: Create the build directory if it doesn't exist
if not exist %BUILD_DIR% mkdir %BUILD_DIR%

:: Navigate to the build directory
cd %BUILD_DIR%

:: Remove previous CMake cache and files if necessary
if exist CMakeCache.txt del CMakeCache.txt
if exist CMakeFiles rmdir /s /q CMakeFiles

:: Run CMake to generate Visual Studio project files
cmake -G "Visual Studio 17 2022" ../..

:: Return to the original directory
cd ../..

@echo on
