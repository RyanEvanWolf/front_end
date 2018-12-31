if(${BUILD_ONLY_INTERFACES} STREQUAL "ON")
#include_directories(${OpenCV_INCLUDE_DIRS})
#add_executable(feature_node_cpp feature_node_cpp.cpp)
#add_dependencies(feature_node_cpp ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
#target_link_libraries(feature_node_cpp 
#	${catkin_LIBRARIES}
#	${OpenCV_LIBS}
#	${BasicSLAMStructures_LIBS}
#)
include_directories(${G2O_INCLUDE_DIR})
find_package(Cholmod REQUIRED)
link_directories(/usr/local/lib)


include_directories(${CHOLMOD_INCLUDE_DIR})


MESSAGE(STATUS "libs ${G2O_CORE_LIBRARY}")


# add_executable(sba_demo sba_demo.cpp)
# add_dependencies(sba_demo ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
# target_link_libraries(sba_demo g2o_core g2o_types_icp g2o_types_sba g2o_solver_cholmod g2o_solver_csparse g2o_stuff ${CHOLMOD_LIBRARIES})

# add_executable(sba_node sba_node.cpp nodeTypes.cpp sba_solver.cpp)
# add_dependencies(sba_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
# target_link_libraries(sba_node  g2o_core g2o_types_icp g2o_types_sba g2o_solver_cholmod g2o_solver_csparse g2o_stuff g2o_types_slam3d ${CHOLMOD_LIBRARIES} ${catkin_LIBRARIES})
else()

add_library(non_free nonfree_init.cpp sift.cpp surf.cpp)
add_dependencies(non_free ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(non_free 
	${catkin_LIBRARIES}
	${OpenCV_LIBS}
	${BasicSLAMStructures_LIBS}

)


# Create projects
include_directories("lib/")

# AKAZE libraries
set(AKAZE_SRCS
    lib/AKAZEConfig.h
    lib/AKAZE.h                  lib/AKAZE.cpp
    lib/fed.h                    lib/fed.cpp
    lib/nldiffusion_functions.h  lib/nldiffusion_functions.cpp
    lib/utils.h                  lib/utils.cpp)

add_library(AKAZE ${AKAZE_SRCS})
target_link_libraries(AKAZE ${OpenCV_LIBS} ${catkin_LIBRARIES})

# add_executable(front_end_node front_end_node.cpp StereoCamera.cpp)
# add_dependencies(front_end_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
# target_link_libraries(front_end_node 
# 	${catkin_LIBRARIES}
# 	${OpenCV_LIBS}
# 	${BasicSLAMStructures_LIBS}
# )

# add_executable(front_end_window_node front_end_window_node.cpp WindowMatcher.cpp)
# add_dependencies(front_end_window_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
# target_link_libraries(front_end_window_node 
# 	${catkin_LIBRARIES}
# 	${OpenCV_LIBS}
# 	${BasicSLAMStructures_LIBS}
# 	${RelativePoseEstimation_LIBS}
# )





add_executable(feature_node_cpp feature_node_cpp.cpp utils.cpp)
add_dependencies(feature_node_cpp ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(feature_node_cpp
	${catkin_LIBRARIES}
	${OpenCV_LIBS}
	${BasicSLAMStructures_LIBS}
	${RelativePoseEstimation_LIBS}
	non_free
  AKAZE
)


add_executable(detect_node_cpp live_stereo.cpp )
add_dependencies(detect_node_cpp ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(detect_node_cpp
	${catkin_LIBRARIES}
	${OpenCV_LIBS}
	${BasicSLAMStructures_LIBS}
	${RelativePoseEstimation_LIBS}
	non_free
  AKAZE
)

endif()



#foreach(file ${files})
	#get_filename_component(tempName ${file} NAME_WE)
	#add_executable(${tempName} ${file})
	#target_link_libraries(${tempName}
	#	${catkin_LIBRARIES}
	#	${OpenCV_LIBS}
	#	${MRPT_LIBS}
	#	${RelativePoseEstimation_LIBS}
	#	${Structures_LIBS}
	#)
