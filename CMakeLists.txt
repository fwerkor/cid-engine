cmake_minimum_required(VERSION 3.22)
project(cid_engine LANGUAGES CXX)

find_package(Torch REQUIRED)

add_library(cid_engine_core STATIC csrc/ops.cpp)
target_include_directories(cid_engine_core PUBLIC csrc/include)
target_link_libraries(cid_engine_core PUBLIC "${TORCH_LIBRARIES}")
target_compile_features(cid_engine_core PUBLIC cxx_std_20)
set_target_properties(cid_engine_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

option(CID_ENGINE_BUILD_TESTS "Build native C++ tests" ON)
if(CID_ENGINE_BUILD_TESTS)
  enable_testing()
  add_executable(cid_engine_cpp_tests tests/cpp/test_ops.cpp)
  target_link_libraries(cid_engine_cpp_tests PRIVATE cid_engine_core)
  target_compile_features(cid_engine_cpp_tests PRIVATE cxx_std_20)
  add_test(NAME cid_engine_cpp_tests COMMAND cid_engine_cpp_tests)
endif()
