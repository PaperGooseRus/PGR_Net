add_library(pgr_neural_math STATIC src/math.cpp)
add_library(PGR::neural_math ALIAS pgr_neural_math)
target_include_directories(pgr_neural_math PUBLIC include)
target_compile_features(pgr_neural_math PUBLIC cxx_std_17)
