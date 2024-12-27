cmake_minimum_required(VERSION 3.22)

set(CMAKE_CXX_COMPILER acpp)

project(sycl_test VERSION 1.0 LANGUAGES CXX)
find_package(AdaptiveCpp CONFIG REQUIRED)

add_executable(sycl_test src/main.cpp)
target_include_directories(sycl_test PUBLIC include)
target_compile_features(sycl_test PRIVATE cxx_std_17)
set_target_properties(sycl_test PROPERTIES
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)
add_sycl_to_target(TARGET sycl_test SOURCES src/main.cpp)
