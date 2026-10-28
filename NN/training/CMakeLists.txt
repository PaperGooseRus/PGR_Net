add_library(pgr_neural_training STATIC src/gradients.cpp src/initialization.cpp src/sgd.cpp src/trainer.cpp)
add_library(PGR::neural_training ALIAS pgr_neural_training)
target_include_directories(pgr_neural_training PUBLIC include)
target_link_libraries(pgr_neural_training PUBLIC PGR::neural_model)
