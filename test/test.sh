#! /bin/bash
current_dir="$(pwd)"
echo $current_dir
sub_folders="$(find $current_dir -maxdepth 1 -type d)"
for j in $sub_folders
do
        mat_name="$(find $j -maxdepth 1 -type f -name '*.mat')"
        echo $mat_name
        if [ ! -z "$mat_name" ]
        then
                python /home/watson/Documents/MiniScope_analysis/test.py --directory $j  --motion_correction
        fi
done