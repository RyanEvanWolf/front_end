if(${BUILD_ONLY_INTERFACES} STREQUAL "ON")
else()
add_executable(front_end_node front_end_node.cpp StereoCamera.cpp)
add_dependencies(front_end_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(front_end_node 
	${catkin_LIBRARIES}
	${OpenCV_LIBS}
	${BasicSLAMStructures_LIBS}
)

add_executable(front_end_window_node front_end_window_node.cpp WindowMatcher.cpp)
add_dependencies(front_end_window_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(front_end_window_node 
	${catkin_LIBRARIES}
	${OpenCV_LIBS}
	${BasicSLAMStructures_LIBS}
	${RelativePoseEstimation_LIBS}
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
