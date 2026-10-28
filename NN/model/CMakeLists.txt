add_library(pgr_neural_model STATIC src/network.cpp src/example.cpp)
add_library(PGR::neural_model ALIAS pgr_neural_model)
target_include_directories(pgr_neural_model PUBLIC include)
target_link_libraries(pgr_neural_model PUBLIC PGR::neural_math)
