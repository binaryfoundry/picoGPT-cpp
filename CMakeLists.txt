cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME picoGPT)

# Set project name and version
project(${PROJECT_NAME} VERSION 1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set the startup project for Visual Studio
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})

# Add nlohmann library
add_library(nlohmann INTERFACE)
target_include_directories(nlohmann INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/lib)

# Add sources
set(MAIN_SOURCES
    src/gpt2_pico.cpp
    src/encoder.cpp)

# Add sources
set(MAIN_HEADERS
    src/encoder.hpp)

# Define the executable
add_executable(${PROJECT_NAME} ${MAIN_SOURCES} ${MAIN_HEADERS})

# Link the static library to the executable
target_link_libraries(${PROJECT_NAME} PRIVATE nlohmann)

# Enable the use of folders in Visual Studio and some other IDEs that support CMake-generated project files.
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Group headers and source files in Visual Studio's Solution Explorer
source_group("Source Files" FILES ${MAIN_SOURCES})
source_group("Header Files" FILES ${MAIN_HEADERS})

# Set compiler-specific options
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /W4 /permissive-)
    target_compile_options(${PROJECT_NAME} PRIVATE /W4 /permissive-)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/Debug)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/Release)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/Debug)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/Release)
elseif(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic -Wno-missing-field-initializers)
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic -Wno-missing-field-initializers)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
endif()

# Define build types explicitly
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
endif()
