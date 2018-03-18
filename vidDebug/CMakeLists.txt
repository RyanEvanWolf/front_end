project(de)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_FILE_OFFSET_BITS=64")

set(DC1394_FOUND TRUE)
set(DC1394_LIBRARIES dc1394)
set(DC1394_LIB_DIR /usr/lib/arm-linux-gnueabihf)

set(_FILE_OFFSET_BITS 64)


#add_executable(record ../time.cpp)
#target_link_libraries(record ${DC1394_LIBRARIES} pthread)

add_executable(record ../buffer.cpp)
target_link_libraries(record ${DC1394_LIBRARIES} pthread)


