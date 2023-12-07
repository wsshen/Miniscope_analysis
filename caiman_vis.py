from mesmerize_core import *


set_parent_raw_data_path('/home/watson/Documents/caiman_fromCluster/vis/')
df = create_batch("/home/watson/Documents/caiman_fromCluster/vis/vis_41_80.pickle")
cnmf_params =\
{
    'main':
        {
            'p':1,
            'K':None,
            'gSig':(3,3),
            'gSiz':(13,13),
            'Ain':None,
            'merge_thr': 0.7,
            'rf': 40,
            'stride':20,
            'tsub':2,
            'ssub':1,
            'low_rank_background':None,
            'nb':0,
            'nb_patch':0,
            'min_corr':0.5,
            'min_pnr':10,
            'ssub_B':2,
            'ring_size_factor':1.4,
            'method_init':'corr_pnr',
            'only_init':True,
            'method_deconvolution':'oasis',
            'update_background_components':True,
            'normalize_init':False,
            'center_psf':True,
            'del_duplicates':True,
            'memory_fact':10,
            'border_pix':0
        },
}
df.caiman.add_item(
    algo='cnmf',
    item_name='output_41_80',
    input_movie_path='/home/watson/Documents/caiman_fromCluster/vis/memmap_d1_600_d2_600_d3_1_order_C_frames_40000.mmap',
    params=cnmf_params
)