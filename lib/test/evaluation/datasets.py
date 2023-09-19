from collections import namedtuple
import importlib
from lib.test.evaluation.data import SequenceList

DatasetInfo = namedtuple('DatasetInfo', ['module', 'class_name', 'kwargs'])

pt = "lib.test.evaluation.%sdataset"  # Useful abbreviations to reduce the clutter
seg = "lib.train.dataset.%s"

dataset_dict = dict(
    otb=DatasetInfo(module=pt % "otb", class_name="OTBDataset", kwargs=dict()),
    nfs=DatasetInfo(module=pt % "nfs", class_name="NFSDataset", kwargs=dict()),
    uav=DatasetInfo(module=pt % "uav", class_name="UAVDataset", kwargs=dict()),
    tc128=DatasetInfo(module=pt % "tc128", class_name="TC128Dataset", kwargs=dict()),
    tc128ce=DatasetInfo(module=pt % "tc128ce", class_name="TC128CEDataset", kwargs=dict()),
    trackingnet=DatasetInfo(module=pt % "trackingnet", class_name="TrackingNetDataset", kwargs=dict()),
    got10k_test=DatasetInfo(module=pt % "got10k", class_name="GOT10KDataset", kwargs=dict(split='test')),
    got10k_val=DatasetInfo(module=pt % "got10k", class_name="GOT10KDataset", kwargs=dict(split='val')),
    got10k_ltrval=DatasetInfo(module=pt % "got10k", class_name="GOT10KDataset", kwargs=dict(split='ltrval')),
    lasot=DatasetInfo(module=pt % "lasot", class_name="LaSOTDataset", kwargs=dict()),
    lasot_lmdb=DatasetInfo(module=pt % "lasot_lmdb", class_name="LaSOTlmdbDataset", kwargs=dict()),
    tnl2k=DatasetInfo(module=pt % "tnl2k", class_name="TNL2kDataset", kwargs=dict()),

    vot18=DatasetInfo(module=pt % "vot", class_name="VOTDataset", kwargs=dict()),
    vot22=DatasetInfo(module=pt % "vot", class_name="VOTDataset", kwargs=dict(year=22)),
    itb=DatasetInfo(module=pt % "itb", class_name="ITBDataset", kwargs=dict()),
    lasot_extension_subset=DatasetInfo(module=pt % "lasotextensionsubset", class_name="LaSOTExtensionSubsetDataset", kwargs=dict()),

    # Segmentation Datasets
    dv2017_val=DatasetInfo(module=seg % "davis", class_name="Davis", kwargs=dict(version='2017', split='val')),
    dv2016_val=DatasetInfo(module=seg % "davis", class_name="Davis", kwargs=dict(version='2016', split='val')),
    dv2017_test_dev=DatasetInfo(module=seg % "davis", class_name="Davis", kwargs=dict(version='2017', split='test-dev')),
    dv2017_test_chal=DatasetInfo(module=seg % "davis", class_name="Davis", kwargs=dict(version='2017', split='test-challenge')),
    yt2019_test=DatasetInfo(module=seg % "youtubevos", class_name="YouTubeVOS", kwargs=dict(version='2019', split='test')),
    yt2019_valid=DatasetInfo(module=seg % "youtubevos", class_name="YouTubeVOS", kwargs=dict(version='2019', split='valid')),
    yt2019_valid_all=DatasetInfo(module=seg % "youtubevos", class_name="YouTubeVOS", kwargs=dict(version='2019', split='valid', all_frames=True)),
    yt2018_valid_all=DatasetInfo(module=seg % "youtubevos", class_name="YouTubeVOS", kwargs=dict(version='2018', split='valid', all_frames=True)),
    yt2018_jjval=DatasetInfo(module=seg % "youtubevos", class_name="YouTubeVOS", kwargs=dict(version='2018', split='jjvalid')),
    yt2019_jjval=DatasetInfo(module=seg % "youtubevos", class_name="YouTubeVOS", kwargs=dict(version='2019', split='jjvalid', cleanup=['starts'])),
    yt2019_jjval_all=DatasetInfo(module=seg % "youtubevos", class_name="YouTubeVOS", kwargs=dict(version='2019', split='jjvalid', all_frames=True, cleanup=['starts'])),

    # Vision Language Datasets
    tnl2k_lang=DatasetInfo(module=pt % "tnl2k", class_name="TNL2k_LangDataset", kwargs=dict()),
    otb_lang=DatasetInfo(module=pt % "otb_lang", class_name="OTB_LangDataset", kwargs=dict()),
    lasot_lang=DatasetInfo(module=pt % "lasot_lang", class_name="LaSOT_LangDataset", kwargs=dict()),
    lasot_extension_subset_lang=DatasetInfo(module=pt % "lasotextensionsubset_lang", class_name="LaSOTExtensionSubset_LangDataset", kwargs=dict()),

)


def load_dataset(name: str):
    """ Import and load a single dataset."""
    name = name.lower()
    dset_info = dataset_dict.get(name)
    if dset_info is None:
        raise ValueError('Unknown dataset \'%s\'' % name)

    m = importlib.import_module(dset_info.module)
    dataset = getattr(m, dset_info.class_name)(**dset_info.kwargs)  # Call the constructor
    return dataset.get_sequence_list()


def get_dataset(*args):
    """ Get a single or set of datasets."""
    dset = SequenceList()
    for name in args:
        dset.extend(load_dataset(name))
    return dset