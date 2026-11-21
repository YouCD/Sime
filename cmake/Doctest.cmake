include_guard(GLOBAL)

include(FetchContent)

# doctest is a single-header test framework. We only need the header and the
# CTest discovery script, not doctest's own CMake project (its 2.4.11
# cmake_minimum_required predates CMake 4 compatibility). SOURCE_SUBDIR points
# at the header folder, which has no CMakeLists.txt, so MakeAvailable only
# populates the source instead of processing doctest's project.
FetchContent_Declare(doctest
    URL
        "https://github.com/doctest/doctest/archive/refs/tags/v2.4.11.zip"
    URL_HASH
        "SHA256=6745e17682d6e2be1ea31ec8bf1814a3d7cb17d61d120e5f2ee5a075926634ad"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SOURCE_SUBDIR doctest
)
FetchContent_MakeAvailable(doctest)

# doctest_discover_tests() for CTest registration.
include(${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake)