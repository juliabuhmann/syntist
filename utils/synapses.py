import h5py
import numpy as np
import time
import networkx as nx
from scipy import spatial

import sys
sys.path.insert(0, '/groups/funke/home/buhmannj/src/cremi_python')
from cremi import Annotations

def load_hdf5(inputfilename, dataset, bb_offset=None, bb_end=None):
    f = h5py.File(inputfilename, 'r')
    if bb_offset is None:
        data = f[dataset].value
    else:
        z1, x1, y1 = bb_offset
        z2, x2, y2 = bb_end
        if len(f[dataset].shape) == 3:
            data = f[dataset][z1:z2, y1:y2, x1:x2]
        elif len(f[dataset].shape) == 4:
            data = f[dataset][:, z1:z2, y1:y2, x1:x2]
    f.close()
    # print (dataset, data.shape)
    return data


class SynLocations(object):
    def __init__(self, id_pre_segm, id_post_segm, all_locs_pre=None, all_locs_post=None,
                 center_pre=None, center_post=None, affinities=None):
        self.id_pre_segm = id_pre_segm
        self.id_post_segm = id_post_segm
        self.all_locs_pre = all_locs_pre
        self.all_locs_post = all_locs_post
        self.center_pre = center_pre
        self.center_post = center_post
        self.affinities = affinities


def extract_locations_with_high_affs(min_max_aff, aff_map, verbose=False):
    ''' Get max affinity per voxel, apply threshold (min_max_aff) and return all non-zero voxel locations

    threshold_max_aff: float, minimal size of max affinty per voxel
    aff_map:           array, network output of affinity predictions
    '''
    max_aff_map = np.max(aff_map, axis=0)
    total_number_ofnonzero = np.count_nonzero(max_aff_map)
    max_aff_map[max_aff_map < min_max_aff] = 0
    nonzero_locs_affs = np.nonzero(max_aff_map)
    if verbose:
        print 'Num nonzero before thresholding:', total_number_ofnonzero
        print 'Num nonzero after thresholding:', len(nonzero_locs_affs[0])
        print 'reduced by %0.2f' % (1. - len(nonzero_locs_affs[0]) / float(total_number_ofnonzero))
    # returns a (N, 3) array, with N = number of locations with highest affinity greater than the provided threshold
    return np.reshape(nonzero_locs_affs, (3, -1)).T


def generate_synlocs(initial_aff_threshold, affs_map, labels, affinity_vectors_vx,
                     offset_vx=np.array([0, 0, 0]), verbose=True):
    ''' extract those locations (voxels) where max. affinity prediction is above a defined threshold '''
    # to avoid running over the entire map during evaluation and instead only consider locations with higher affinities
    locs_pre_to_eval = extract_locations_with_high_affs(initial_aff_threshold, affs_map, verbose)

    ''' create/format affinities to synapse locations'''
    all_synlocs = create_synlocs(presyn_locs=locs_pre_to_eval, aff_map_pre=affs_map, labels=labels,
                                 aff_threshold=initial_aff_threshold, aff_vectors_vx=affinity_vectors_vx,
                                 offset=offset_vx)

    return all_synlocs


def create_synlocs(presyn_locs, aff_map_pre, labels, aff_threshold,
                   aff_vectors_vx, offset=np.array([0, 0, 0])):
    all_synlocs = []
    print 'number of voxel locations to look at: %i' % len(presyn_locs)
    counter = 0
    execution_time_start = time.time()
    for loc in presyn_locs:
        if counter % 100000 == 0:
            ratio_done = counter / float(len(presyn_locs))
            total_time = time.time() - execution_time_start
            if not ratio_done == 0:
                estimated_time = (total_time / ratio_done) * (1 - ratio_done)
                print 'estimated time left: %0.2f sec ; ' \
                      'processed %0.2f percent (%i out of %i)' % (estimated_time,
                                                                  ratio_done * 100, counter, len(presyn_locs))
        counter += 1
        for nr_aff, aff in enumerate(aff_map_pre[:, loc[0], loc[1], loc[2]]):
            if aff >= aff_threshold:

                id_start_segm = labels[loc[0], loc[1], loc[2]]
                aff_vector = aff_vectors_vx[nr_aff]
                end_location = loc + aff_vector
                end_location = end_location.astype(np.int)
                # check if end location lies within the labelled volume
                if (end_location == np.clip(end_location, a_min=(0, 0, 0),
                                            a_max=np.array(labels.shape) - (1, 1, 1))).all():
                    id_end_segm = labels[end_location[0], end_location[1], end_location[2]]
                    if id_end_segm == 0 or id_start_segm == 0:
                        continue
                    # check if end_location lies outside of start segment
                    if id_start_segm != id_end_segm:
                        synloc_exists_already = False
                        for synloc in all_synlocs:
                            if synloc.id_pre_segm == id_start_segm and synloc.id_post_segm == id_end_segm:
                                synloc.all_locs_pre.append(loc + offset)
                                synloc.all_locs_post.append(end_location + offset)
                                synloc.affinities.append(aff)
                                synloc_exists_already = True

                        if not synloc_exists_already:
                            new_synloc = SynLocations(id_pre_segm=id_start_segm, id_post_segm=id_end_segm,
                                                      all_locs_pre=[loc + offset],
                                                      all_locs_post=[end_location + offset], affinities=[aff])
                            all_synlocs.append(new_synloc)
    print('synloc extraction took %0.2f sec' % (time.time() - execution_time_start))
    return all_synlocs


def split_pointcloud_into_clusters(points, dist_threshold, verbose=False):
    # Create spatial kdtree
    kdtreeloc = spatial.cKDTree(points)

    # Find clusters in point cloud
    total_num_points = len(kdtreeloc.data)
    if verbose:
        print 'total number of points', total_num_points

    seed_list = range(total_num_points)
    looseclusters = []
    while len(seed_list) != 0:
        seed = seed_list.pop(0)
        cur_loc_seed = kdtreeloc.data[seed]
        nodes = set(kdtreeloc.query_ball_point(cur_loc_seed, dist_threshold))

        seeds_to_remove = kdtreeloc.query_ball_point(cur_loc_seed, int(dist_threshold * 0.7))
        for seed_to_remove in seeds_to_remove:
            if seed_to_remove in seed_list:
                seed_list.remove(seed_to_remove)

        looseclusters.append(nodes)

    baseclusters = find_connected_components(looseclusters)
    baseclusters = [list(basecluster) for basecluster in baseclusters]
    if verbose:
        print 'point cloud split into %i unique clusters' % len(baseclusters)
    return baseclusters


def find_connected_components(clusters):
    g = nx.Graph()
    for cluster in clusters:
        cluster = list(cluster)
        g.add_path(cluster)
    ccs = nx.connected_components(g)
    connected_components = []
    for cc in ccs:
        connected_components.append(list(cc))
    return connected_components


def compute_centers_per_synloc(all_synlocs, segvolume=None, voxel_size=None, offset=np.array([0, 0, 0])):
    # Take Center of Mass as an estimate for the location of pre and postsynapse.
    # To obtain original connectivity, do not use center of mass if change of connectivity is detected.
    wrong_seg_id_counter_pre = 0
    wrong_seg_id_counter_post = 0
    rel_z, rel_y, rel_x = offset
    for synloc in all_synlocs:
        segidpre = synloc.id_pre_segm
        segidpost = synloc.id_post_segm

        all_locs_pre = synloc.all_locs_pre
        if voxel_size is not None:
            all_locs_pre = np.array(all_locs_pre)
            all_locs_pre *= voxel_size
        if len(all_locs_pre) >= 1:
            center_pre = np.mean(np.reshape(all_locs_pre, (-1, 3)), axis=0)
        else:
            center_pre = None
        if segvolume is not None:

            seg_id_from_loc = segvolume[
                int(center_pre[0]) - rel_z, int(center_pre[1]) - rel_y, int(center_pre[2]) - rel_x]
            if not seg_id_from_loc == segidpre:
                wrong_seg_id_counter_pre += 1
                center_pre = all_locs_pre[0]

        synloc.center_pre = center_pre
        all_locs_post = synloc.all_locs_post
        if len(all_locs_post) >= 1:
            center_post = np.mean(np.reshape(all_locs_post, (-1, 3)), axis=0)
        else:
            center_post = None
        if segvolume is not None:
            seg_id_from_loc = segvolume[
                int(center_post[0]) - rel_z, int(center_post[1]) - rel_y, int(center_post[2]) - rel_x]
            if not seg_id_from_loc == segidpost:
                wrong_seg_id_counter_post += 1
                center_post = all_locs_post[0]  # really bad heuristic, find something smarter!
        synloc.center_post = center_post
    print 'total of shifted pre segs', wrong_seg_id_counter_pre
    print 'total of shifted post segs', wrong_seg_id_counter_post

    print('Done with computing centers from list of voxel locations')


def split_synlocs(all_synlocs, splitting_distance_vx, voxel_size,
                  outputfilename=None, labels=None, offset=np.array([0, 0, 0])):
    '''split point clouds'''
    # This step is necessary to account for multiple connections/synapses between the same neuron pair. So far,
    # each neuron pair (two segments) has a list of locations associated with. With the splitting,
    # the locations are distangled.

    # Split synlocs based on distance threshold
    splitting_time = time.time()
    splitted_synlocs = []

    print 'splitting %i' % len(all_synlocs)
    for ii, synloc in enumerate(all_synlocs):
        splittet_synlocs_with_same_nsegpair = split_synlocation_based_on_distance_helper(synloc,
                                                                                         dist_threshold=
                                                                                         splitting_distance_vx,
                                                                                         voxel_size=np.asarray(
                                                                                             voxel_size))
        splitted_synlocs.extend(splittet_synlocs_with_same_nsegpair)
        if ii%10 == 0:
            print 'splitted %i out of %i' % (ii, len(all_synlocs))

    end_splitting = time.time() - splitting_time
    print 'splitting took %0.3f' % end_splitting

    # Calculating new center of mass for point clouds
    compute_centers_per_synloc(splitted_synlocs, segvolume=labels, offset=offset)
    return splitted_synlocs


def split_synlocation_based_on_distance_helper(synloc, dist_threshold, voxel_size):
    '''
    Args:

    Returns:
    '''
    voxel_size = np.array(voxel_size)

    # For performance reason, filter out redundant locations and carry out clustering on a unique list of locations.
    loc_to_indeces = {}
    for ii, loc in enumerate(synloc.all_locs_post):
        locphys = loc * voxel_size
        if tuple(locphys) in loc_to_indeces:
            loc_to_indeces[tuple(locphys)].append(ii)
        else:
            loc_to_indeces[tuple(locphys)] = [ii]

    locs = np.asarray(loc_to_indeces.keys())
    cluster_of_indeces = split_pointcloud_into_clusters(locs, dist_threshold, verbose=False)
    new_synlocs = []
    for cluster in cluster_of_indeces:
        new_locations_pre = []
        new_locations_post = []
        new_affinities = []
        for node_id in cluster:
            loc = locs[node_id, :]
            ori_node_ids = loc_to_indeces[tuple(loc)]
            for ori_node_id in ori_node_ids:
                new_locations_pre.append(synloc.all_locs_pre[ori_node_id])
                new_locations_post.append(synloc.all_locs_post[ori_node_id])
                new_affinities.append(synloc.affinities[ori_node_id])

        new_synloc = SynLocations(synloc.id_pre_segm, synloc.id_post_segm, all_locs_pre=new_locations_pre,
                                  all_locs_post=new_locations_post, affinities=new_affinities)
        new_synlocs.append(new_synloc)
    return new_synlocs


class Synlocspairs(object):
    def __init__(self, prepostsynlocs, postpresynlocs, id_pre_segm, id_post_segm):
        self.prepostsynlocs = prepostsynlocs
        self.postpresynlocs = postpresynlocs
        self.id_pre_segm = id_pre_segm
        self.id_post_segm = id_post_segm


def create_synloc_dics(synlocs):
    prepostdic = {}
    unique_identi_dic = {}
    for ii, synloc in enumerate(synlocs):
        prepostpair = tuple([synloc.id_pre_segm, synloc.id_post_segm])
        if prepostpair in prepostdic:
            prepostdic[prepostpair].append(synloc)
        else:
            prepostdic[prepostpair] = [synloc]
        synloc.unique_identifier = ii
        unique_identi_dic[ii] = synloc
    return prepostdic, unique_identi_dic


def find_synloc_pairs(prepostdic):
    # Identify synlocs with potential bidrectionality eg. if there is mapping from seg1 to seg2 and
    # from seg2 to seg1.
    samesegpairlist = []
    prepostdictoremove = prepostdic.copy()
    for prepostpair, synlocsprepost in prepostdic.iteritems():
        postprepair = tuple([prepostpair[1], prepostpair[0]])
        if postprepair in prepostdictoremove:
            synlocspostpre = prepostdic[postprepair]
            synlocpair = Synlocspairs(synlocsprepost, synlocspostpre, prepostpair[0], prepostpair[1])
            samesegpairlist.append(synlocpair)
            del prepostdictoremove[prepostpair]
    return samesegpairlist

def match_synlocs(synlocs1, synlocs2, voxel_size, dist_threshold):
    # Synlocs1 and synlocs2 are two lists of locations. Partners are identified between the two lists,
    # when the two postsynaptics locations are in closer distance than dist_threshold.
    identified_pairs = []
    for synloc1 in synlocs1:
        prelocation = synloc1.center_post * voxel_size
        for synloc2 in synlocs2:
            prelocation2 = synloc2.center_pre * voxel_size
            dist = np.linalg.norm(prelocation - prelocation2)
            if dist < dist_threshold:
                identified_pairs.append([synloc1.unique_identifier, synloc2.unique_identifier])
                break
    return identified_pairs

def remove_reciprocal_synlocs(synlocs, dist_threshold, voxel_size, score_type='count'):

    total_num_synlocs = len(synlocs)
    print 'removing bidirectional synapses, total number: %i' %total_num_synlocs
    prepostdic, unique_id_dic = create_synloc_dics(synlocs)
    samesegpairlist = find_synloc_pairs(prepostdic)

    identified_pairs = []
    for samesegpair in samesegpairlist:
        # samesegpair is an Synlocpair opbject representing a list of synlocs
        identified_pairs.extend(match_synlocs(samesegpair.prepostsynlocs,
                                              samesegpair.postpresynlocs,
                                              voxel_size, dist_threshold))
    identified_synloc_pairs = []
    for unique_id1, unique_id2 in identified_pairs:
        if score_type == 'count':
            score1 = len(synlocs[unique_id1].affinities)
            score2 = len(synlocs[unique_id2].affinities)
        elif score_type == 'sum':
            score1 = np.sum(synlocs[unique_id1].affinities)
            score2 = np.sum(synlocs[unique_id2].affinities)

        if score1 > score2:
            if unique_id2 in unique_id_dic:
                del unique_id_dic[unique_id2]
        else:
            if unique_id1 in unique_id_dic:
                del unique_id_dic[unique_id1]
        identified_synloc_pairs.append((synlocs[unique_id1], synlocs[unique_id2]))
    print 'removed %i of bidirectional synlocs' %(total_num_synlocs-len(unique_id_dic.values()))
    return unique_id_dic.values(), identified_synloc_pairs



def check_point_inside_bb(point, bb_min, bb_max):
    check_inside = True
    for dim in range(len(bb_min)):
        pos = point[dim]
        if pos < bb_min[dim]:
            check_inside = False
        elif pos >= bb_max[dim]:
            check_inside = False
    return check_inside

def crop_cremi_annotation(annotations, bb_min, bb_max, offset=None, verbose=False):
    ids_to_stay = []
    ids_to_go = []
    cropped_ann = Annotations(offset=annotations.offset)
    pre_to_post = {}
    post_to_pre = {}
    prepostlist = list(annotations.pre_post_partners) #copies the list
    for partner in annotations.pre_post_partners:
        pre_to_post[partner[0]] = partner[1]
        post_to_pre[partner[1]] = partner[0]

    for id in annotations.ids():
        loc = annotations.get_annotation(id)[1]
        if check_point_inside_bb(loc, bb_min, bb_max):
            ids_to_stay.append(id)
        else:
            ids_to_go.append(id)

    # Partners are removed when either pre or post side are outside of bounding box....
    for id_to_go in ids_to_go:
        if id_to_go in pre_to_post.keys():
            if pre_to_post[id_to_go] in ids_to_stay:
                ids_to_stay.remove(pre_to_post[id_to_go])
            partnertuple = (id_to_go, pre_to_post[id_to_go])
        elif id_to_go in post_to_pre.keys():
            if post_to_pre[id_to_go] in ids_to_stay:
                ids_to_stay.remove(post_to_pre[id_to_go])
            partnertuple = (post_to_pre[id_to_go], id_to_go)
        else:
            if verbose:
                print annotations.pre_post_partners
                print 'id %i not ni prepostpartners' %id_to_go
        if partnertuple in prepostlist:
            prepostlist.remove(partnertuple)

    for id in ids_to_stay:
        cur_type, loc = annotations.get_annotation(id)
        if offset is not None:
            loc = np.asarray(loc)
            loc += offset
        cropped_ann.add_annotation(id, cur_type, loc)
    cropped_ann.pre_post_partners = prepostlist
    if verbose:
        print 'cropping: removed %i synaptic partners from a total of %i' %(len(annotations.pre_post_partners)-
                                                                        len(prepostlist), len(annotations.pre_post_partners))
    return cropped_ann

def reformat_for_cremi_eval(all_synlocs, offset, voxel_size,
                            outputfilename=None, swap_xz=False, voxel=True):
    id_nr, ids, locations, partners, types = 0, [], [], [], []
    for synloc in all_synlocs:
        if synloc.center_pre is not None and synloc.center_post is not None:
            id_nr += 2
            ids.extend([id_nr - 2, id_nr - 1])
            partners.extend([np.array((id_nr - 2, id_nr - 1))])
            # types.extend(['presynaptic_site', 'postsynaptic_site'])
            types.extend(['postsynaptic_site', 'presynaptic_site'])

            if swap_xz:
                center_pre = (synloc.center_pre + offset) * voxel_size
                center_pre_swapped = [center_pre[2], center_pre[1], center_pre[0]]
                center_post = (synloc.center_post + offset) * voxel_size
                center_post_swapped = [center_post[2], center_post[1], center_post[0]]
                locations.extend([center_pre_swapped, center_post_swapped])
            else:
                if voxel:
                    locations.extend(
                        [(synloc.center_pre + offset) * voxel_size, (synloc.center_post + offset) * voxel_size])
                else:
                    locations.extend(
                        [synloc.center_pre + offset, synloc.center_post + offset])

                # locations.extend([synloc.center_pre, synloc.center_post])

    if outputfilename is not None:
        h5_file = h5py.File(outputfilename, 'w')
        dset = h5_file.create_dataset('annotations/ids', data=ids, compression='gzip')
        dset = h5_file.create_dataset('annotations/locations', data=np.stack(locations, axis=0), compression='gzip')
        dset = h5_file.create_dataset('annotations/presynaptic_site/partners',
                                      data=np.stack(partners, axis=0), compression='gzip')
        dset = h5_file.create_dataset('annotations/types', data=types, compression='gzip')

        h5_file.attrs['resolution'] = voxel_size
        h5_file['annotations'].attrs['offset'] = [ 1480., 3644., 3644.] # CREMI offset of padded raw dataset
        h5_file.close()
    else:
        return ids, locations, partners, types


def get_affinity_score(affinities, score_type):
    if score_type == 'mean':
        return np.mean(affinities)
    elif score_type == 'sum':
        return np.sum(affinities)
    elif score_type == 'median':
        return np.median(affinities)
    elif score_type == 'max':
        return np.max(affinities)
    elif score_type == '90percentile':
        return np.percentile(affinities, 90)
    else:
        print 'score type not defined'

def filter_synlocs_based_on_affinities(all_synlocs, threshold, score_type):
    synlocs_to_remove = []
    for synloc in all_synlocs:
        score = get_affinity_score(np.array(synloc.affinities), score_type)
        if score < threshold:
            synlocs_to_remove.append(synloc)

    for synloc_rm in synlocs_to_remove:
        all_synlocs.remove(synloc_rm)