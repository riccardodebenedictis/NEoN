file( GLOB SOLVER_SOURCES test0.cpp ${CMAKE_SOURCE_DIR}/core/*.cpp )

add_executable( solver_tests ${SOLVER_SOURCES} )

if(WIN32)
  target_link_libraries( solver_tests wsock32 ws2_32 )
endif()