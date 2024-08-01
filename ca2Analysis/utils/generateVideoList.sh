#! /bin/bash
current_dir="$(pwd)"
echo $current_dir
avi_names="$(find $current_dir -maxdepth 1 -type f -name '*.avi' -printf '%f\n' | sort -n)"
increment=1
output_file_name='mylist'
output_file_extension='.txt'
#file_index=1
start_index=0
if [ -z "$avi_names" ]
then
		echo "\$avi_names is empty"
else
	if test -f "$output_file_name$file_index$output_file_extension"
	then
		rm $output_file_name$file_index$output_file_extension
	fi

	for j in $avi_names
	do
		echo -e 'file' '\t' $j >> $output_file_name$file_index$output_file_extension
		#if [ $((increment%50)) -eq 0 ] 
		#then
		#	#ffmpeg -f concat -safe -0 -i $output_file_name$file_index$output_file_extension -c copy -r 30 'output_'$start_index'_'$((increment-1))'.avi'
		#	file_index=$((file_index+1))
		#if test -f "$output_file_name$file_index$output_file_extension"
		#then
		#	rm $output_file_name$file_index$output_file_extension
		#fi
		#	start_index=$increment
		#else
		#	echo $increment $j
		#fi
		echo $increment $j
		increment=$((increment+1))
	done
fi
