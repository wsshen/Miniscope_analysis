#! /bin/bash -l
#SBATCH -t 3:00:00
#SBATCH -n 1
#SBATCH --mem=300G
source /etc/profile.d/modules.sh
source ~/.bashrc
conda init bash
conda activate caiman
current_dir="/media/watson/UbuntuHDD/feng_Xin/Xin/Miniscope/newcohort_03242024/reward_seeking/7276902252024"
sub_folders="$(find $current_dir -maxdepth 1 -type d)"
for j in $sub_folders
do
        mat_name="$(find $j -maxdepth 1 -type f -name 'output_rescaled.pickle')"
        echo $j $mat_namere
        if [ -z "$mat_name" ]
        then
                echo "pass"
                #python /home/watson/Documents/MiniScope_analysis/basicAnalysis_batch.py --directory $j  --motion_correction
        fi
done
