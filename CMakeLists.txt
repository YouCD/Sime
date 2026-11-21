cmake_minimum_required(VERSION 3.20)
project(sime_new LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(NEW_ROOT ${CMAKE_CURRENT_LIST_DIR})
set(NEW_SRC_DIR ${NEW_ROOT}/src)
set(NEW_INC_DIR ${NEW_ROOT}/include)

add_library(sime_core
    ${NEW_SRC_DIR}/ustr.cc
    ${NEW_SRC_DIR}/count.cc
    ${NEW_SRC_DIR}/construct.cc
    ${NEW_SRC_DIR}/construct_smooth.cc
    ${NEW_SRC_DIR}/construct_io.cc
    ${NEW_SRC_DIR}/convert.cc
    ${NEW_SRC_DIR}/compact.cc
    ${NEW_SRC_DIR}/score.cc
    ${NEW_SRC_DIR}/score_query.cc
    ${NEW_SRC_DIR}/dict.cc
    ${NEW_SRC_DIR}/state.cc
    ${NEW_SRC_DIR}/user.cc
    ${NEW_SRC_DIR}/user_decay.cc
    ${NEW_SRC_DIR}/user_persist.cc
    ${NEW_SRC_DIR}/sime.cc
    ${NEW_SRC_DIR}/sime_net_pinyin.cc
    ${NEW_SRC_DIR}/sime_net_num.cc
    ${NEW_SRC_DIR}/sime_penalty.cc
    ${NEW_SRC_DIR}/sime_english.cc
    ${NEW_SRC_DIR}/sime_search.cc
    ${NEW_SRC_DIR}/sime_decode.cc
    ${NEW_SRC_DIR}/sime_predict.cc
    ${NEW_SRC_DIR}/sime_gru.cc
    ${NEW_SRC_DIR}/trie.cc
    ${NEW_SRC_DIR}/trie_serialize.cc
    ${NEW_SRC_DIR}/trie_pinyin.cc
    ${NEW_SRC_DIR}/trie_build.cc
    ${NEW_SRC_DIR}/cut.cc
    ${NEW_SRC_DIR}/gru_reranker.cc
    ${NEW_SRC_DIR}/pinyin_correct.cc
)

target_include_directories(sime_core PUBLIC ${NEW_INC_DIR} ${NEW_SRC_DIR})
target_compile_features(sime_core PUBLIC cxx_std_20)
target_compile_options(sime_core PRIVATE -Wall -Wextra -pedantic -Wconversion)
set_target_properties(sime_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

option(SIME_ENABLE_NCNN "Enable the ncnn GRU reranker backend" OFF)
if(SIME_ENABLE_NCNN)
    include(${NEW_ROOT}/cmake/Ncnn.cmake)
    sime_add_ncnn()
    target_link_libraries(sime_core PRIVATE ncnn)
    target_compile_definitions(sime_core PUBLIC SIME_ENABLE_NCNN=1)
    # Android ncnn deliberately exports -fno-exceptions/-fno-rtti. Those
    # flags must not leak into Sime, whose public API reports malformed
    # UTF-8 and model errors with C++ exceptions.
    if(ANDROID)
        target_compile_options(sime_core PRIVATE -fexceptions -frtti)
    endif()
endif()

option(SIME_BUILD_TOOLS "Build Sime command-line and model tools" ON)
if(SIME_BUILD_TOOLS)
    add_executable(sime-count ${NEW_ROOT}/bin/counter.cc)
    target_link_libraries(sime-count PRIVATE sime_core)

    add_executable(sime-construct ${NEW_ROOT}/bin/constructor.cc)
    target_link_libraries(sime-construct PRIVATE sime_core)

    add_executable(sime-converter ${NEW_ROOT}/bin/converter.cc)
    target_link_libraries(sime-converter PRIVATE sime_core)

    add_executable(sime ${NEW_ROOT}/bin/sime.cc)
    target_link_libraries(sime PRIVATE sime_core)

    add_executable(sime-dump ${NEW_ROOT}/bin/dump.cc)
    target_link_libraries(sime-dump PRIVATE sime_core)

    add_executable(sime-cut ${NEW_ROOT}/bin/cutter.cc)
    target_link_libraries(sime-cut PRIVATE sime_core)

    add_executable(sime-import ${NEW_ROOT}/bin/sime-import.cc)
    target_link_libraries(sime-import PRIVATE sime_core)
    find_path(LEVELDB_INCLUDE_DIR leveldb/db.h)
    find_library(LEVELDB_LIBRARY NAMES leveldb)
    if(LEVELDB_INCLUDE_DIR AND LEVELDB_LIBRARY)
        target_include_directories(sime-import PRIVATE ${LEVELDB_INCLUDE_DIR})
        target_link_libraries(sime-import PRIVATE ${LEVELDB_LIBRARY})
    else()
        message(FATAL_ERROR "sime-import requires libleveldb (leveldb/db.h + leveldb lib)")
    endif()

    add_executable(sime-compact ${NEW_ROOT}/bin/compact.cc)
    target_link_libraries(sime-compact PRIVATE sime_core)

    set(_sime_targets
        sime-count
        sime-construct
        sime-converter
        sime-dump
        sime-compact
        #sime-cut
        sime-import
        sime
    )

    foreach(target_name IN LISTS _sime_targets)
        target_compile_options(${target_name} PRIVATE -Wall -Wextra -pedantic -Wconversion)
    endforeach()
endif()

option(SIME_BUILD_TESTS "Build Sime unit tests (doctest)" OFF)
set(SIME_TEST_MODEL_DIR "" CACHE PATH "Model dir for gated decode integration tests")
if(SIME_BUILD_TESTS)
    include(${NEW_ROOT}/cmake/Doctest.cmake)
    enable_testing()
    add_subdirectory(tests)
endif()
