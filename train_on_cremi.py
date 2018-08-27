from __future__ import print_function

import os
import math
import sys
import json
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Gunpowder imports
from gunpowder import *
from gunpowder.tensorflow import *
from gunpowder.nodes.batch_filter import BatchFilter

# Custom imports
from gunpowdernodes import CremiSource
from tf_utils import generate_tinder_net_cremi

data_dir = 'put here cremi data direction' # publicly available ground truth set from: https://cremi.org/

def train(output_path, hyperparameter_dic, affinity_vectors):
    print('writing everything to %s' % output_path)
    meta_graph_filename = output_path + '/unet'
    jsonfilename = output_path + '/net_io_names.json'

    max_iteration = 500000
    min_masked = hyperparameter_dic['min_masked']
    reject_probability = hyperparameter_dic['reject_probability']
    blob_radius = hyperparameter_dic['blob_radius']

    with open(jsonfilename, 'r') as f:
        net_io_names = json.load(f)

    register_volume_type('RAW')
    register_volume_type('ALPHA_MASK')
    register_volume_type('GT_LABELS')
    register_volume_type('GT_SCALE')
    register_volume_type('PREDICTED_AFFS')
    register_volume_type('LOSS_GRADIENT')

    # Points
    register_points_type('PRESYN')
    register_points_type('POSTSYN')

    # Blob
    register_volume_type('PRESYN_BLOB')
    register_volume_type('POSTSYN_BLOB')
    register_volume_type('PRE_LR_AFFINITIES')
    register_volume_type('POST_LR_AFFINITIES')
    register_volume_type('PREDICTED_AFFS')
    register_volume_type('LOSS_GRADIENT')

    request = BatchRequest()
    voxel_size = (40, 4, 4)
    input_size = Coordinate((42, 403, 403)) * voxel_size
    output_size = Coordinate((14, 191, 191)) * voxel_size  # Output size depends on network architecture.
    request.add_centered(VolumeTypes.RAW, input_size, voxel_size=voxel_size)
    request.add_centered(VolumeTypes.GT_LABELS, output_size, voxel_size=voxel_size)
    request.add_centered(VolumeTypes.LOSS_GRADIENT, output_size, voxel_size=voxel_size)

    # adjust for affinity type trained on
    aff_type = hyperparameter_dic['aff_type']
    if aff_type == 'pre':
        gt_volume = VolumeTypes.PRE_LR_AFFINITIES
    elif aff_type == 'post':
        gt_volume = VolumeTypes.POST_LR_AFFINITIES
    else:
        raise Exception('Unknown affinity type given, should be "pre" or "post"')

    snapshot_dir = os.path.join(output_path, 'snapshots')

    ## Load Cremi Data A and B and half of C ------------
    volume_phys_offset = [(1520, 3644, 3644), (1520, 3644, 3644), (1520, 3644, 3644)]
    synapse_shapes = [(125, 1250, 1250), (125, 1250, 1250),
                      (60, 1250, 1250)]  # Second half of sampleC is used for testing
    voxel_size_np = np.array(voxel_size)
    data_sources = []
    samples = ['sample_A_padded_20160501.hdf', 'sample_B_padded_20160501.hdf', 'sample_C_padded_20160501.hdf']
    for ii, sample in enumerate(samples):
        data_sources.append(
            CremiSource(
                os.path.join(data_dir, sample),
                datasets={
                    VolumeTypes.RAW: 'volumes/raw',
                    VolumeTypes.GT_LABELS: 'volumes/labels/neuron_ids',
                },
                points_types=[PointsTypes.PRESYN, PointsTypes.POSTSYN],

                points_rois={
                    PointsTypes.PRESYN: [np.array(volume_phys_offset[ii]),
                                         synapse_shapes[ii] * voxel_size_np],

                    PointsTypes.POSTSYN: [np.array(volume_phys_offset[ii]),
                                          synapse_shapes[ii] * voxel_size_np]
                }
            ) +
            RandomLocation(focus_points_type=PointsTypes.PRESYN) +
            Normalize()
        )

    data_sources = tuple(data_sources)

    ### AddBlobsFromPoints ------------------------------------------------------------------------
    output_dtype = 'uint64'
    mapper = Int64BitIdEnconder()
    add_blob_data = {
        'POSTSYN':
            {
                'point_type': PointsTypes.POSTSYN,
                'output_volume_type': VolumeTypes.POSTSYN_BLOB,
                'output_volume_dtype': output_dtype,
                'radius': blob_radius,
                'output_voxel_size': voxel_size,
                'restrictive_mask_type': VolumeTypes.GT_LABELS,
                'id_mapper': mapper,
                'partner_points': PointsTypes.PRESYN
            },
        'PRESYN':
            {
                'point_type': PointsTypes.PRESYN,
                'output_volume_type': VolumeTypes.PRESYN_BLOB,
                'output_volume_dtype': output_dtype,
                'radius': blob_radius,
                'output_voxel_size': voxel_size,
                'restrictive_mask_type': VolumeTypes.GT_LABELS,
                'id_mapper': mapper,
                'partner_points': PointsTypes.POSTSYN
            },
    }

    add_blob_from_points_node = AddBlobsFromPoints(add_blob_data)
    long_range_affinities_node = AddLongRangeAffinities(affinity_vectors, VolumeTypes.PRESYN_BLOB,
                                                        VolumeTypes.POSTSYN_BLOB, VolumeTypes.PRE_LR_AFFINITIES,
                                                        VolumeTypes.POST_LR_AFFINITIES)

    ### Snapshot ----------------------------------------------------------------------------------
    snapshot_request = BatchRequest()
    snapshot_request.add_centered(VolumeTypes.RAW, input_size, voxel_size=voxel_size)
    snapshot_request.add_centered(VolumeTypes.GT_LABELS, output_size, voxel_size=voxel_size)
    snapshot_request.add_centered(VolumeTypes.GT_SCALE, output_size, voxel_size=voxel_size)
    snapshot_request.add_centered(VolumeTypes.PRESYN_BLOB, output_size, voxel_size=voxel_size)
    snapshot_request.add_centered(VolumeTypes.POSTSYN_BLOB, output_size, voxel_size=voxel_size)
    snapshot_request.add_centered(VolumeTypes.PRE_LR_AFFINITIES, output_size, voxel_size=voxel_size)
    snapshot_request.add_centered(VolumeTypes.POST_LR_AFFINITIES, output_size, voxel_size=voxel_size)
    snapshot_request.add_centered(VolumeTypes.PREDICTED_AFFS, output_size, voxel_size=voxel_size)
    snapshot_request.add_centered(VolumeTypes.LOSS_GRADIENT, output_size, voxel_size=voxel_size)

    snapshot_node = Snapshot(
        {
            VolumeTypes.RAW: '/volumes/raw',
            VolumeTypes.GT_LABELS: '/volumes/labels/neuron_ids',
            VolumeTypes.PRESYN_BLOB: '/volumes/labels/pre_synaptic_blobs',
            VolumeTypes.POSTSYN_BLOB: '/volumes/labels/post_synaptic_blobs',
            VolumeTypes.PRE_LR_AFFINITIES: '/volumes/labels/pre_lr_affinities',
            VolumeTypes.POST_LR_AFFINITIES: '/volumes/labels/post_lr_affinities',
            VolumeTypes.PREDICTED_AFFS: '/volumes/pred/pre_lr_affinities',
            VolumeTypes.LOSS_GRADIENT: '/volumes/train/loss_grad',
            VolumeTypes.GT_SCALE: '/volumes/train/scale'
        },
        every=30000,
        output_dir=snapshot_dir,
        additional_request=snapshot_request,
        output_filename='batch_{iteration}.hdf',
        compression_type='gzip'
    )

    ### Gunpowder BATCH TREE --------------------------------------------------------------------------------
    target_volume = gt_volume
    print('Defining Tree')
    batch_provider_tree = (
        data_sources +
        RandomProvider() +
        ElasticAugment([4, 40, 40], [0, 2, 2], [0, math.pi / 2.0], prob_slip=0.05, prob_shift=0.05,
                       max_misalign=10, subsample=8) +
        SimpleAugment(transpose_only_xy=True) +
        add_blob_from_points_node +
        long_range_affinities_node +
        Reject(min_masked=min_masked, mask_volume_type=target_volume, reject_probability=reject_probability) +
        BalanceLabels(labels=target_volume,
                      scales=VolumeTypes.GT_SCALE,
                      slab=(1, -1, -1, -1)) +
        IntensityAugment(0.9, 1.1, -0.1, 0.1, z_section_wise=True) +
        IntensityScaleShift(2, -1) +
        PreCache(
            cache_size=40,
            num_workers=15) +
        Train(
            meta_graph_filename=meta_graph_filename,
            checkpoint_dir=output_path,
            optimizer=net_io_names['optimizer'],
            loss=net_io_names['loss'],
            save_every=30000,  # 10000
            inputs={
                net_io_names['raw']: VolumeTypes.RAW,
                net_io_names['gt_affs']: target_volume,
                net_io_names['loss_weights']: VolumeTypes.GT_SCALE
            },
            outputs={
                net_io_names['affs']: VolumeTypes.PREDICTED_AFFS
            },
            gradients={
                net_io_names['affs']: VolumeTypes.LOSS_GRADIENT
            }) +
        LogLossNode() +
        snapshot_node
    )

    print("Requesting", max_iteration, "batches to train")

    with build(batch_provider_tree):
        for i in range(max_iteration):
            batch = batch_provider_tree.request_batch(request)

    print("Finished Test")


class Int64BitIdEnconder:
    def __init__(self):
        self._map = {}

    def get_map(self):
        return self._map.copy()

    def make_map(self, all_points):
        self._map = {}
        all_synapse_ids = []

        for point_type, points in all_points.items():
            for point_id, point_data in points.data.items():
                all_synapse_ids.append(point_data.synapse_id)

        unique_synapse_ids = np.unique(all_synapse_ids)

        assert len(unique_synapse_ids) <= 64, "This map cannot reliably enconde more than 64 different IDs"

        for i, synapse_id in enumerate(unique_synapse_ids):
            new_synapse_id = np.power(2, i)
            self._map[synapse_id] = new_synapse_id

    def __call__(self, synapse_id):
        assert synapse_id in self._map.keys(), 'Unexpected point ID %s' % synapse_id
        return self._map[synapse_id]


class LogLossNode(BatchFilter):
    def process(self, batch, request):
        if batch.loss:
            logger.critical('it: %s, loss: %s', batch.iteration, batch.loss)


if __name__ == '__main__':
    import logging

    logging.basicConfig(level=logging.INFO)  # Needed, otherwise, gunpowder does not output anything!

    # vector r in the paper
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

    output_dir = 'output/'
    print('Base dir: ', output_dir)

    # Parameters used for model presented in the paper.
    hyperparameter_dic = {
        'learning_rate': 0.5e-5,
        'reg_scale': 10 ** -20,
        'min_masked': 10 ** -4,  # gunpowder reject node option
        'reject_probability': 0.9,  # gunpowder reject node option
        'blob_radius': 100,  # length of the vector r (in paper)
        'aff_type': 'pre'  # whether to predict edge directions from pre->post or post->pre
    }

    # Generate the U-Network
    generate_tinder_net_cremi(output_dir, learning_rate=hyperparameter_dic['learning_rate'],
                              reg_scale=hyperparameter_dic['reg_scale'],
                              num_output=affinity_vectors.shape[0])
    with open(output_dir + '/hyperparameterdic.json', 'w') as f:
        json.dump(hyperparameter_dic, f)

    # Start training
    train(output_dir, hyperparameter_dic, affinity_vectors)
