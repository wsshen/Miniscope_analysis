#! /bin/bash -l
#SBATCH -t 10:00:00
#SBATCH -n 1
#SBATCH --mem=300G
source /etc/profile.d/modules.sh
source ~/.bashrc
conda init bash
conda activate caiman
current_dir="/om2/user/shenwang/miniscope/72769/reward_seeking"
sub_folders="$(find $current_dir -maxdepth 1 -type d)"
for j in $sub_folders
do
        mat_name="$(find $j -maxdepth 1 -type f -name '*[^_].mmap')"
        echo $mat_name
        if [ -n "$mat_name" ]
        then
                echo "pass"
                #python /home/watson/Documents/MiniScope_analysis/caiman_addNoiseToMMAP_batch.py --directory $j
        fi
done