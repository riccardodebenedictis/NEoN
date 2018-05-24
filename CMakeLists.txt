cmake_minimum_required(VERSION 3.1)

project ( NEoN VERSION 1.0 LANGUAGES CXX )

# Set a default build type if none was specified
set( default_build_type "Release" )
if( EXISTS "${CMAKE_SOURCE_DIR}/.git" )
  set( default_build_type "Debug" )
endif()
 
if( NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES )
  message( STATUS "Setting build type to '${default_build_type}' as none was specified." )
  set( CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Choose the type of build." FORCE )
  # Set the possible values of build type for cmake-gui
  set_property( CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo" )
endif()

set ( CMAKE_CXX_STANDARD 14 )

file( GLOB SOURCES main.cpp nn/*.cpp )
if( CMAKE_BUILD_TYPE MATCHES Debug )
  list( APPEND SOURCES nn/listeners/socket_listener.cpp ) 
endif()

include_directories( nn nn/listeners )

add_executable( ${PROJECT_NAME} ${SOURCES} )

if( CMAKE_BUILD_TYPE MATCHES Debug )
  if( WIN32 )
    target_link_libraries( ${PROJECT_NAME} wsock32 ws2_32 #[[-PROFILE]] )
  endif()
  include( CTest )
  #add_subdirectory( core/tests )
endif()