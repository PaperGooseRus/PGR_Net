add_library(pgr_neural_storage STATIC src/model_io.cpp src/file_io.cpp)
add_library(PGR::neural_storage ALIAS pgr_neural_storage)
target_include_directories(pgr_neural_storage PUBLIC include)
target_link_libraries(pgr_neural_storage PUBLIC PGR::neural_model)
