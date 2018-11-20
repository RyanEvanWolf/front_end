# front_end
ros based package for stereo feature extraction and matching with the opencv_tegra libraries

vidDebug must still be integrated into the cmake build, but it contains recording binaries
that dont drop frames when recording to USB


-----------------------------
--motion Simulation related binaries


rosrun front_end extractPCL 


"--speeds",default="Slow,Medium,Fast",type=str
"--motionType",default="straight",type=str
"--extractMethods",default="PCL",type=str
"--ideal",default=True,type=bool
"--outlier",default=True,type=bool
"--gaussian",default=True,type=bool

currently has a hard coded file directory in which to search for simulated motion files