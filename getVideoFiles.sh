#! /bin/bash
echo "hello"
dir_prefix="/media/watson/UbuntuHDD/feng_Xin/Xin/Miniscope/5133207132023/reward_seeking/days_with_miniscope_recording/day1_poke_lick/session1/"
dir_names="$(find $dir_prefix -type d)"
increment=1
for i in $dir_names
do
	echo $i
	avi_names="$(find $i -maxdepth 1 -type f -name '*.avi')"
	if [ -z "$avi_names" ]
	then
		echo "\$avi_names is empty"
	else
		for j in $avi_names
		do
			echo -e $increment '\t' $i '\t' $j >> output.txt
			increment=$((increment+1))
		done
	fi
done
