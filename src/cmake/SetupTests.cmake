##------------------------------------------------------------------------------
## - Builds and adds a test that uses gtest
##
## add_cpp_test( TEST test DEPENDS_ON dep1 dep2... )
##------------------------------------------------------------------------------
function(add_cpp_test)

    set(options)
    set(singleValueArgs TEST)
    set(multiValueArgs DEPENDS_ON)

    # parse our arguments
    cmake_parse_arguments(arg
                         "${options}" 
                         "${singleValueArgs}" 
                         "${multiValueArgs}" ${ARGN} )

    message(STATUS " [*] Adding Unit Test: ${arg_TEST}")

    blt_add_executable( NAME ${arg_TEST}
                        SOURCES ${arg_TEST}.cpp ${fortran_driver_source}
                        OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}
                        DEPENDS_ON "${arg_DEPENDS_ON}" gtest)

    blt_add_test( NAME ${arg_TEST}
                  COMMAND ${arg_TEST}
                    )

endfunction()


##------------------------------------------------------------------------------
## - Builds and adds a test that uses gtest
##
## add_cuda_test( TEST test DEPENDS_ON dep1 dep2... )
##------------------------------------------------------------------------------
function(add_cuda_test)

    set(options)
    set(singleValueArgs TEST)
    set(multiValueArgs DEPENDS_ON)

    # parse our arguments
    cmake_parse_arguments(arg
                         "${options}" 
                         "${singleValueArgs}" 
                         "${multiValueArgs}" ${ARGN} )

    message(STATUS " [*] Adding CUDA Unit Test: ${arg_TEST}")

    blt_add_executable( NAME ${arg_TEST}
                        SOURCES ${arg_TEST}.cpp ${fortran_driver_source}
                        OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}
                        DEPENDS_ON "${arg_DEPENDS_ON}" gtest cuda)

    blt_add_test( NAME ${arg_TEST}
                  COMMAND ${arg_TEST}
                    )

endfunction()


##------------------------------------------------------------------------------
## - Builds and adds a test that uses gtest and mpi
##
## add_cpp_mpi_test( TEST test NUM_MPI_TASKS 2 DEPENDS_ON dep1 dep2... )
##------------------------------------------------------------------------------
function(add_cpp_mpi_test)

    set(options)
    set(singleValueArgs TEST NUM_MPI_TASKS)
    set(multiValueArgs DEPENDS_ON)

    # parse our arguments
    cmake_parse_arguments(arg
                         "${options}" 
                         "${singleValueArgs}" 
                         "${multiValueArgs}" ${ARGN} )

    message(STATUS " [*] Adding Unit Test: ${arg_TEST}")
    
    
    blt_add_executable( NAME ${arg_TEST}
                        SOURCES ${arg_TEST}.cpp ${fortran_driver_source}
                        OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}
                        DEPENDS_ON "${arg_DEPENDS_ON}" gtest mpi)
                        
    blt_add_test( NAME ${arg_TEST}
                  COMMAND ${arg_TEST}
                  NUM_MPI_TASKS ${arg_NUM_MPI_TASKS})

endfunction()


##------------------------------------------------------------------------------
## - Adds a python based unit test
##
## add_python_test( TEST test)
##------------------------------------------------------------------------------
function(add_python_test TEST)
            
    message(STATUS " [*] Adding Python-based Unit Test: ${TEST}")
    add_test( NAME ${TEST}
              COMMAND ${PYTHON_EXECUTABLE} -B -m unittest -v ${TEST})

    # make sure python can pick up the modules we built
    set(PYTHON_TEST_PATH "${CMAKE_BINARY_DIR}/python-modules/:${CMAKE_CURRENT_SOURCE_DIR}")
    if(EXTRA_PYTHON_MODULE_DIRS)
        set(PYTHON_TEST_PATH "${EXTRA_PYTHON_MODULE_DIRS}:${PYTHON_TEST_PATH}")
    endif()
    set_property(TEST ${TEST} PROPERTY ENVIRONMENT  "PYTHONPATH=${PYTHON_TEST_PATH}")
endfunction(add_python_test)

##------------------------------------------------------------------------------
## - Adds a fortran based unit test
##
## add_fortran_test( TEST test DEPENDS_ON dep1 dep2... )
##------------------------------------------------------------------------------
macro(add_fortran_test)
    set(options)
    set(singleValueArgs TEST)
    set(multiValueArgs DEPENDS_ON)

    # parse our arguments
    cmake_parse_arguments(arg
                         "${options}" 
                         "${singleValueArgs}" 
                         "${multiValueArgs}" ${ARGN} )

    message(STATUS " [*] Adding Fortran Unit Test: ${arg_TEST}")
    blt_add_executable( NAME ${arg_TEST}
                        SOURCES ${arg_TEST}.f
                        OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}
                        DEPENDS_ON fruit "${arg_DEPENDS_ON}")

    blt_add_test( NAME ${arg_TEST}
                  COMMAND ${arg_TEST})

endmacro(add_fortran_test)



