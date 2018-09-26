"""
Script to prepare small CREMI example dataset to put on public syntist github repos.

"""
import os
import h5py


def load_hdf5(inputfilename, dataset, offset_z=0):
    f = h5py.File(inputfilename, 'r')
    if len(f[dataset].shape) == 3:
        data = f[dataset][z1+offset_z:z2+offset_z, y1:y2, x1:x2]
    elif len(f[dataset].shape) == 4:
        data = f[dataset][:, z1+offset_z:z2+offset_z, y1:y2, x1:x2]

    f.close()
    print (dataset, data.shape)
    return data

SAMPLE = 'C'

z1, y1, x1 = 81, 500, 500
z2, y2, x2 = 95, 1000, 1000



# IO Settings
runpath = '20180213/190158' # Model pre-->post rej. prob = 0.95, trained on A,B,half C
iteration = 210000
base_dir = ''
cremifile = 'data/cremi/sample_%s_20160501.hdf' % SAMPLE.upper() # CREMI dataset from website
inputfilename_aff = os.path.join(base_dir, runpath, 'inference', 'sample_%s' % SAMPLE.lower(),
                                 'inference_iteration%0.8i.hdf' % iteration)

segdatasetname = '/volumes/labels/neuron_ids'
datasetname = 'volumes/pred/pre_lr_affinities'
rawdatasetname = '/volumes/raw'


labels = load_hdf5(cremifile , segdatasetname)
affs_map  = load_hdf5(inputfilename_aff, datasetname)
raw = load_hdf5(cremifile , rawdatasetname)
print raw.shape

outputfilename = 'cremi_exampledataset.h5'

f = h5py.File(outputfilename, 'w')
f.create_dataset('raw', data=raw)
f.create_dataset('segmentation', data=labels)
f.create_dataset('prediction', data=affs_map)
f.create_dataset('bb_offset', data=(z1, y1, x1))
f.create_dataset('bb_end', data=(z2, y2, x2))
f.close()



