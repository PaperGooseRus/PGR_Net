cmake_minimum_required(VERSION 3.20)
project(PGR_Neural_NN VERSION 0.4.0 LANGUAGES CXX)

# Это самостоятельный проект. Зависимостей от Qt и PGR_Make нет.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CTest)
add_subdirectory(math)
add_subdirectory(model)
add_subdirectory(training)
add_subdirectory(storage)
add_subdirectory(demo)
if(BUILD_TESTING)
    add_subdirectory(tests)
else()
    message(STATUS "PGR Neural: tests disabled")
endif()
