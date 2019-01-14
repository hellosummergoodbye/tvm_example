cmake_minimum_required(VERSION 3.4.3)



project(tvm_example LANGUAGES CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

set(SOURCE source/main.cpp)

set(LOC ${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath -Wl,/usr/local/lib")

add_executable(${PROJECT_NAME} ${SOURCE})

target_link_libraries(${PROJECT_NAME} tvm_runtime) 
