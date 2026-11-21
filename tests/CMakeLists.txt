# Model-free unit tests. Each executable provides its own doctest main.
# doctest headers are included as SYSTEM so the framework's own code does
# not trip Sime's strict warning flags.
function(sime_add_test name)
    add_executable(${name} ${name}.cc)
    target_link_libraries(${name} PRIVATE sime_core)
    target_include_directories(${name} SYSTEM PRIVATE ${doctest_SOURCE_DIR})
    target_compile_options(${name} PRIVATE -Wall -Wextra -pedantic -Wconversion)
    doctest_discover_tests(${name})
endfunction()

sime_add_test(test_user_sentence)
sime_add_test(test_pinyin_correct)
sime_add_test(test_dict_static)
sime_add_test(test_ustr)
sime_add_test(test_config)
sime_add_test(test_helpers)

# Gated integration tests: need a real dict+LM model dir. Without it the
# test binary builds but every case self-skips.
sime_add_test(test_decode)
if(SIME_TEST_MODEL_DIR)
    target_compile_definitions(test_decode PRIVATE
        SIME_TEST_MODEL_DIR="${SIME_TEST_MODEL_DIR}")
endif()