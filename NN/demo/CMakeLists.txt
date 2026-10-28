add_executable(pgr_neural_demo main.cpp)
target_link_libraries(pgr_neural_demo PRIVATE PGR::neural_model)
add_executable(pgr_neural_gradient_demo gradient_main.cpp)
target_link_libraries(pgr_neural_gradient_demo PRIVATE PGR::neural_training)

add_executable(pgr_neural_training_demo training_main.cpp)
target_link_libraries(pgr_neural_training_demo PRIVATE PGR::neural_training PGR::neural_storage)
add_executable(pgr_neural_model_demo model_main.cpp)
target_link_libraries(pgr_neural_model_demo PRIVATE PGR::neural_storage)
if(WIN32)
    # Unicode-аргументы CLI через CommandLineToArgvW, без потери кириллицы.
    target_link_libraries(pgr_neural_training_demo PRIVATE shell32)
    target_link_libraries(pgr_neural_model_demo PRIVATE shell32)
endif()
