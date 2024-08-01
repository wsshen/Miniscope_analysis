ccc = ['/media/watson/UbuntuHDD/feng_Xin/Xin/Miniscope/4914202252023/05192023/10_48_49/My_V4_Miniscope/output_0_rig__d1_600_d2_600_d3_1_order_F_frames_10000.mmap',
       '/media/watson/UbuntuHDD/feng_Xin/Xin/Miniscope/4914202252023/05192023/10_48_49/My_V4_Miniscope/output_1_rig__d1_600_d2_600_d3_1_order_F_frames_10000.mmap',
'/media/watson/UbuntuHDD/feng_Xin/Xin/Miniscope/4914202252023/05192023/10_48_49/My_V4_Miniscope/output_2_rig__d1_600_d2_600_d3_1_order_F_frames_10000.mmap',
'/media/watson/UbuntuHDD/feng_Xin/Xin/Miniscope/4914202252023/05192023/10_48_49/My_V4_Miniscope/output_3_rig__d1_600_d2_600_d3_1_order_F_frames_10000.mmap',
'/media/watson/UbuntuHDD/feng_Xin/Xin/Miniscope/4914202252023/05192023/10_48_49/My_V4_Miniscope/output_4_rig__d1_600_d2_600_d3_1_order_F_frames_10000.mmap',
'/media/watson/UbuntuHDD/feng_Xin/Xin/Miniscope/4914202252023/05192023/10_48_49/My_V4_Miniscope/output_5_rig__d1_600_d2_600_d3_1_order_F_frames_10000.mmap',
'/media/watson/UbuntuHDD/feng_Xin/Xin/Miniscope/4914202252023/05192023/10_48_49/My_V4_Miniscope/output_6_rig__d1_600_d2_600_d3_1_order_F_frames_10000.mmap',
'/media/watson/UbuntuHDD/feng_Xin/Xin/Miniscope/4914202252023/05192023/10_48_49/My_V4_Miniscope/output_7_rig__d1_600_d2_600_d3_1_order_F_frames_10000.mmap',
'/media/watson/UbuntuHDD/feng_Xin/Xin/Miniscope/4914202252023/05192023/10_48_49/My_V4_Miniscope/output_8_rig__d1_600_d2_600_d3_1_order_F_frames_10000.mmap',
'/media/watson/UbuntuHDD/feng_Xin/Xin/Miniscope/4914202252023/05192023/10_48_49/My_V4_Miniscope/output_9_rig__d1_600_d2_600_d3_1_order_F_frames_10000.mmap',
'/media/watson/UbuntuHDD/feng_Xin/Xin/Miniscope/4914202252023/05192023/10_48_49/My_V4_Miniscope/output_10_rig__d1_600_d2_600_d3_1_order_F_frames_10000.mmap',
'/media/watson/UbuntuHDD/feng_Xin/Xin/Miniscope/4914202252023/05192023/10_48_49/My_V4_Miniscope/output_11_rig__d1_600_d2_600_d3_1_order_F_frames_8805.mmap',
]
import caiman as cm
# if 'dview' in locals():
#     cm.stop_server(dview=dview)
#     print("inside dview")
# c, dview, n_processes = cm.cluster.setup_cluster(
#     backend='local', n_processes=None, single_thread=False)
from caiman.mmapping import *
#from caiman.mmapping_cupy import save_memmap_join
save_memmap_join(ccc, base_name='memmap_merged', dview=None)
