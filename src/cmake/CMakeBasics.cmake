################################
# Standard CMake Options
################################


# Fail if someone tries to config an in-source build.
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
   message(FATAL_ERROR "In-source builds are not supported. Please remove "
                       "CMakeCache.txt from the 'src' dir and configure an "
                       "out-of-source build in another directory.")
endif()

# enable creation of compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# always use position independent code
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

message(STATUS "CMake build tool name: ${CMAKE_BUILD_TOOL}")

macro(ENABLE_WARNINGS)
    # set the warning levels we want to abide by
    if("${CMAKE_BUILD_TOOL}" MATCHES "(msdev|devenv|nmake|MSBuild)")
        add_definitions(/W4)
    else()
        if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" OR
            "${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU"   OR
            "${CMAKE_CXX_COMPILER_ID}" MATCHES "Intel")
            # use these flags for clang, gcc, or icc
            add_definitions(-Wall -Wextra)
        endif()
    endif()
endmacro()


################################
# Shared vs Static Libs
################################
if(BUILD_SHARED_LIBS)
    message(STATUS "Building shared libraries (BUILD_SHARED_LIBS == ON)")
else()
    message(STATUS "Building static libraries (BUILD_SHARED_LIBS == OFF)")
endif()

################################
# Coverage Flags
################################
if(ENABLE_COVERAGE)
    message(STATUS "Building using coverage flags (ENABLE_COVERAGE == ON)")
    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} --coverage")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
else()
    message(STATUS "Building without coverage flags (ENABLE_COVERAGE == OFF)")
endif()

################################
# Win32 Output Dir Settings
################################
# On windows we place all of the libs and execs in one dir.
# dlls need to be located next to the execs since there is no
# rpath equiv on windows. I tried some gymnastics to extract
# and append the output dir of each dependent lib to the PATH for
# each of our tests and bins, but that was futile.
################################
if(WIN32)
    set(EXECUTABLE_OUTPUT_PATH  ${CMAKE_BINARY_DIR}/bin)
    set(ARCHIVE_OUTPUT_PATH     ${CMAKE_BINARY_DIR}/bin)
    set(LIBRARY_OUTPUT_PATH     ${CMAKE_BINARY_DIR}/bin)
endif()

################################
# Standard CTest Options
################################
if(ENABLE_TESTS)
    set(MEMORYCHECK_SUPPRESSIONS_FILE "${CMAKE_SOURCE_DIR}/cmake/valgrind.supp" CACHE PATH "")
    include(CTest)
endif()

##############################################################################
# Try to extract the current git sha
#
# This solution is derived from:
#  http://stackoverflow.com/a/21028226/203071
#
# This does not have full dependency tracking - it wont auto update when the
# git HEAD changes or when a branch is checked out, unless a change causes
# cmake to reconfigure.
#
# However, this limited approach will still be useful in many cases,
# including building and for installing ascent as a tpl
#
##############################################################################
find_package(Git)
if(GIT_FOUND)
  message("git executable: ${GIT_EXECUTABLE}")
  execute_process(COMMAND
    "${GIT_EXECUTABLE}" describe --match=NeVeRmAtCh --always --abbrev=40 --dirty
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    OUTPUT_VARIABLE CONDUIT_GIT_SHA1
    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
  message(STATUS "Repo SHA1:" ${CONDUIT_GIT_SHA1})
endif()


###############################################################################
# This macro converts a cmake path to a platform specific string literal
# usable in C++. (For example, on windows C:/Path will be come C:\\Path)
###############################################################################

macro(convert_to_native_escaped_file_path path output)
    file(TO_NATIVE_PATH ${path} ${output})
    string(REPLACE "\\" "\\\\"  ${output} "${${output}}")
endmacro()


