path=$1 
# extract the file name without extension
path=${path%.*}
# save as .txt
cat $1 | grep -v " 0 $" | grep "^3 " > $path.txt
mv $path.txt ~/SKEL_WS/ros2_ws/src/smplx_from_skeleton_tracking_ROS2/virtual_fixture/skel_regions
