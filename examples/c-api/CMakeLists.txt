add_executable(example-c-api main.c)
target_link_libraries(example-c-api PRIVATE parakeet_lib)
set_target_properties(example-c-api PROPERTIES
    C_STANDARD 99
    LINKER_LANGUAGE CXX
)
