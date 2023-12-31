from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/home/yz/tcsvt/MMTrack/data/got10k_lmdb'
    settings.got10k_path = '/home/yz/tcsvt/MMTrack/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/home/yz/tcsvt/MMTrack/data/itb'
    settings.lasot_extension_subset_path = '/home/yz/tcsvt/MMTrack/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/home/yz/tcsvt/MMTrack/data/lasot_lmdb'
    settings.lasot_path = '/home/yz/tcsvt/MMTrack/data/lasot'
    settings.network_path = '/home/yz/tcsvt/MMTrack/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/home/yz/tcsvt/MMTrack/data/nfs'
    settings.otb_lang_path = '/home/yz/tcsvt/MMTrack/data/otb_lang'
    settings.otb_path = '/home/yz/tcsvt/MMTrack/data/otb'
    settings.prj_dir = '/home/yz/tcsvt/MMTrack'
    settings.result_plot_path = '/home/yz/tcsvt/MMTrack/output/test/result_plots'
    settings.results_path = '/home/yz/tcsvt/MMTrack/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/yz/tcsvt/MMTrack/output'
    settings.segmentation_path = '/home/yz/tcsvt/MMTrack/output/test/segmentation_results'
    settings.tc128_path = '/home/yz/tcsvt/MMTrack/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/home/yz/tcsvt/MMTrack/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/yz/tcsvt/MMTrack/data/trackingnet'
    settings.uav_path = '/home/yz/tcsvt/MMTrack/data/uav'
    settings.vot18_path = '/home/yz/tcsvt/MMTrack/data/vot2018'
    settings.vot22_path = '/home/yz/tcsvt/MMTrack/data/vot2022'
    settings.vot_path = '/home/yz/tcsvt/MMTrack/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

