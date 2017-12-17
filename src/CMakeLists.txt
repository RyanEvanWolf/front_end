add_executable(front_end_node front_end_node.cpp StereoCamera.cpp)
target_link_libraries(front_end_node 
	${catkin_LIBRARIES}
	${OpenCV_LIBS}
	${BasicSLAMStructures_LIBS}
)	


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
