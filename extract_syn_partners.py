"""
Script to extract synaptic partners from probability maps of directed edges.

This script has some CREMI specific modifications that would need to be adapted if run on different datasets.
"""
from utils import synapses
import time
import numpy as np
import sys


from cremi.io import CremiFile
from cremi.evaluation import SynapticPartners

SAMPLE = 'c'

# ------------ CREMI specific settings------------
# Handle section loss, this is prob only needed for ssTEM data.
zsectiondic = ({'a': [], 'b': [16-1, 17-1, 78-1], 'c': [74]}) # CREMI segmentation GT data is used here
# and has segmentation cuts at those positions.
voxel_size = np.asarray([40., 4., 4.])

# ------------ Hyperparameter ------------
initial_threshold = 0.5 # Parameter t1 in paper.
score_threshold = 2500 # Parameter t2 in paper. Parameters from the paper, optimized on a,b,c
score_type = 'sum'


remove_reciprocal_synapses = False # Simple heuristic, does not consider loops.
reciprocal_distance_threshold = 100
splitting_distance = 60 # For finding connected components.
affinity_vectors = [[-80, 0, 0],
                    [-40, -60, -60],
                    [-40, -60, 60],
                    [-40, 60, -60],
                    [-40, 60, 60],
                    [0, -120, 0],
                    [0, 0, -120],
                    [0, 0, 120],
                    [0, 120, 0],
                    [40, -60, -60],
                    [40, -60, 60],
                    [40, 60, -60],
                    [40, 60, 60],
                    [80, 0, 0]]
affinity_vectors = np.array(affinity_vectors)
affinity_vectors = np.negative(affinity_vectors)
affinity_vectors_vx = affinity_vectors / voxel_size


# ------------ IO Settings ------------
examplecremi = 'data/cremi_exampledataset.h5'  # Download exampledataset, contains network output prediction,
# raw data and Ground truth segmentation.
cremigt_filename = 'sample_%s_20160501.hdf' % SAMPLE.upper() # Download data from the CREMI website.


# ------------ Load data ------------
affs_map = synapses.load_hdf5(examplecremi, 'prediction')
neuronseg = synapses.load_hdf5(examplecremi, 'segmentation')
z1, y1, x1 = synapses.load_hdf5(examplecremi, 'bb_offset') # bounding box relative to CREMI dataset
z2, y2, x2 = synapses.load_hdf5(examplecremi, 'bb_end')
offset_locs_vx = z1, y1, x1

# Prepare Segmentation data
zindeces = zsectiondic[SAMPLE]
for zindex in zindeces:
    rel_zindex = zindex-z1
    if rel_zindex > 0 and rel_zindex <= neuronseg.shape[0]:
        neuronseg[zindex-z1, :, :] = 0 #mask out the section in segmentation data (at least in the ground truth case,
    # otherwise this gives many wrongly segmented segments)


# ------------ Extract synapses ------------
print 'generating synlocs'
generating_synloc_time = time.time()
synlocs = synapses.generate_synlocs(initial_threshold, affs_map, neuronseg, affinity_vectors_vx,
                 offset_vx=offset_locs_vx)
print 'generating synlocs took %0.2f' % (time.time() - generating_synloc_time)


# Split postsynaptic point clouds based on connected components.
synlocs = synapses.split_synlocs(synlocs, splitting_distance, voxel_size, outputfilename=None,
              offset=offset_locs_vx, labels=neuronseg)

print 'filtering with ', score_threshold
synapses.filter_synlocs_based_on_affinities(synlocs, score_threshold, score_type=score_type)
# Remove reciprocal synapses
if remove_reciprocal_synapses:
    synlocs, _ = synapses.remove_reciprocal_synlocs(synlocs, dist_threshold=reciprocal_distance_threshold ,
                                                           voxel_size=voxel_size)




# ------------ CREMI evaluation ------------
bb_min = np.array([z1, y1, x1])
bb_max = np.array([z2, y2,x2])
cremifilename = 'output/synpredictions.h5'
cremifilename_crop = 'output/synpredictions_crop.h5'
cremifilename_crop_gt = 'output/syngt_crop.h5'
synapses.reformat_for_cremi_eval(synlocs, offset=[0, 0, 0],
                                 voxel_size=voxel_size, outputfilename=cremifilename)
test = CremiFile(cremifilename, 'r')
truth = CremiFile(cremigt_filename, 'r')
print 'cropping predicted partners to bounding box'
testannotation_crop = synapses.crop_cremi_annotation(test.read_annotations(), bb_min * voxel_size,
                                                     bb_max * voxel_size,
                                                     offset=np.array([0, 0, 0]))
testannotation_crop.offset = [0., 0., 0.]
print 'cropping ground truth partners to bounding box'
truthannotation_crop = synapses.crop_cremi_annotation(truth.read_annotations(), bb_min * voxel_size,
                                                      bb_max * voxel_size, offset=[0, 0, 0])
print truthannotation_crop.offset, 'gt offset'
truthannotation_crop.offset = tuple(truthannotation_crop.offset)
crop_output = CremiFile(cremifilename_crop, 'w')
crop_output.write_annotations(testannotation_crop)

crop_output = CremiFile(cremifilename_crop_gt, 'w')
crop_output.write_annotations(truthannotation_crop)
cremi_eval_time = time.time()
synaptic_partners_evaluation = SynapticPartners(matching_threshold=300)
fscore, precision, recall, fpcount, fncount, matches = synaptic_partners_evaluation.fscore(
    testannotation_crop,
    truthannotation_crop,
    truth.read_neuron_ids(),
    all_stats=True)

print "Synaptic partners"
print "================="
print "\tfscore: " + str(fscore)
print "\tprecision " + str(precision)
print "\trecall " + str(recall)
print "\t # false positive: " + str(fpcount)
print "\t # false negative: " + str(fncount)
print "\t total number of partners: " + str(len(truthannotation_crop.pre_post_partners))
print "\t total number of predicted partners: " + str(len(testannotation_crop.pre_post_partners))
print '\t evaluation took %0.2f' % (time.time() - cremi_eval_time)


# Expected fscore on small crop dataset with unchanged parameters above! : 0.68