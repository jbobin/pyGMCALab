# Install script for directory: /Users/jbobin/Downloads/GMCAlab-python/pygmca/pyredwave

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/Users/jbobin/Downloads/GMCAlab-python/pygmca/pyredwave")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/pyredwave" TYPE SHARED_LIBRARY FILES "/Users/jbobin/Downloads/GMCAlab-python/pygmca/pyredwave/build/redwavecxx.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/pyredwave/redwavecxx.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/pyredwave/redwavecxx.so")
    execute_process(COMMAND "/opt/local/bin/install_name_tool"
      -id "redwavecxx.so"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/pyredwave/redwavecxx.so")
    execute_process(COMMAND /opt/local/bin/install_name_tool
      -delete_rpath "/Users/jbobin/Downloads/GMCAlab-python/pygmca/pyredwave/build/extern/lib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/pyredwave/redwavecxx.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/opt/local/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/pyredwave/redwavecxx.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/Users/jbobin/Downloads/GMCAlab-python/pygmca/pyredwave/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
