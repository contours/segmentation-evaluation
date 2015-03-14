import os
import math
import segeval as se
import numpy as np
import itertools as it
import collections
import random

from decimal import Decimal

# Directory containing our segmentation files
DATA_DIR = '/Users/ryanshaw/Code/u-series-segmentations'

# Boundary similarity parameter: Maximum distance (in potential boundary positions) that a transposition may span
MAX_TRANSPOSE_DISTANCE = 9

def datapath(path):
    return os.path.join(DATA_DIR, path)

def remove_coder(dataset, coder):
    for item, coder_segmentations in dataset.items():
        dataset[item].pop(coder)
    dataset.coders.discard(coder)
    return dataset

def load_dataset(name):
    dataset = se.input_linear_mass_json(datapath(name))
    dataset.coders = tuple(sorted(set.intersection(
        *[ set(data.keys()) for data in dataset.values() ])))
    return dataset

def rename_coders(dataset, old_names, suffix):
    if len(set.intersection(set(old_names), set(dataset.coders))) == 0:
        return dataset
    rename = lambda x: x+suffix if x in old_names else x
    d = se.data.Dataset(
        { item: { rename(c):s for c,s in segmentations.items() }
          for item, segmentations in dataset.items() })
    d.coders = tuple(map(rename, dataset.coders))
    return d

def merge_datasets(*datasets):
    repeated_coders = set.intersection(*[ set(d.coders)
                                          for d in datasets ])
    ds = [datasets[0]] + [ rename_coders(d, repeated_coders, '-%s' % i)
                           for i, d in enumerate(datasets[1:], start=1) ]
    items = set.intersection(*[ set(d.keys()) for d in ds ])
    merged = se.data.Dataset(
        { item: { c:s for c,s in it.chain.from_iterable(
            [ d[item].items() for d in ds ]) }
          for item in items })
    merged.coders = tuple(it.chain(*[ d.coders for d in ds ]))
    return merged

def micro_mean(ratios):
    num, den = [ sum(values) for values in zip(*ratios) ]
    return num / den    

def micro_mean_std_n(ratios):
    mean = micro_mean(ratios)
    scores = [ (n/d) for n,d in ratios ]
    squared_deviations = [ abs(s - mean)**2 for s in scores ]
    std = np.sqrt(np.mean(squared_deviations))
    return (mean, std, len(ratios))

def micro_mean_std_n_str(ratios):
    return '{0:.4f} ±{1:.4f}, n={2}'.format(*micro_mean_std_n(ratios))

def mean_std_n(values):
    floats = np.array(list(values), dtype=float)
    return (np.mean(floats), np.std(floats), len(floats))

def mean_std_n_str(values):
    return '{0:.4f} ±{1:.4f}, n={2}'.format(*mean_std_n(values))

def b_statistics(dataset):
    return se.boundary_statistics(dataset, n_t=MAX_TRANSPOSE_DISTANCE)

def b(*datasets, return_parts=False):
    return se.boundary_similarity(*datasets, n_t=MAX_TRANSPOSE_DISTANCE, return_parts=return_parts)

def wd(dataset):
    return se.window_diff(dataset)

def actual_agreement(dataset, return_parts=False):
    return se.actual_agreement_linear(dataset, n_t=MAX_TRANSPOSE_DISTANCE, return_parts=return_parts)

def pi(dataset, return_parts=False):
    return se.fleiss_pi_linear(dataset, n_t=MAX_TRANSPOSE_DISTANCE, return_parts=return_parts)

def pi_variance(dataset):
    # This uses formula 13 in Fleiss et al. 1979 (single-category π* variance)
    n_coders = len(dataset.coders)
    n_possible_boundaries = potential_boundaries(dataset)
    return Decimal(2) / (n_possible_boundaries * n_coders * (n_coders-1))

def pi_z_test(dataset):
    # This uses formula 16 in Fleiss et al. 1979 (single-category π* z score)
    n_coders = len(dataset.coders)
    n_possible_boundaries = potential_boundaries(dataset)
    n_actual_boundaries = boundaries_placed(dataset)
    p_boundary = Decimal(n_actual_boundaries) / (n_possible_boundaries * n_coders)

    a = pi(dataset) + (Decimal(1) / (n_possible_boundaries*(n_coders-1)))
    b = ( (n_possible_boundaries*n_coders*(n_coders-1)) / Decimal(2) )**Decimal(0.5)
    
    return a * b

def kappa(dataset):
    return se.fleiss_kappa_linear(dataset, n_t=MAX_TRANSPOSE_DISTANCE)

def bias(dataset):
    return se.artstein_poesio_bias_linear(dataset, n_t=MAX_TRANSPOSE_DISTANCE)

def pairwise_similarity(dataset, coder, item):
    ratios = [ v[:2] for k,v in b(dataset, return_parts=True).items() 
               if (coder in k) and (item in k) ]
    return micro_mean(ratios)
    
def strip_prefixes(strings):
    return [ s.split(':')[-1] for s in strings ]

def total_mass(dataset, item):
    sums = [ sum(s) for s in dataset[item].values() ]
    assert(all(s == sums[0] for s in sums))
    return sums[0]

def total_masses_for(dataset):
    return { item: total_mass(dataset, item) for item in dataset.keys() }

def total_boundaries_laid(dataset, item):
    return sum([ (len(s) - 1) for s in dataset[item].values() ])

def total_segments_created(dataset, item):
    return sum([ len(s) for s in dataset[item].values() ])

def boundaries_placed(dataset, coder=None):
    if coder is not None:
        return sum([ (len(v[coder]) - 1) for v in dataset.values() ])
    else:
        return sum([ boundaries_placed(dataset, c) for c in dataset.coders ])

def potential_boundaries(dataset):
    return sum([ (total_mass(dataset, item) - 1) for item in dataset.keys() ])

def boundary_placement_rate(dataset, coder):
    return boundaries_placed(dataset, coder) / potential_boundaries(dataset)

def boundary_ratios_for(dataset):
    d = collections.defaultdict(dict)
    for item, segmentations in dataset.items():
        for coder, masses in segmentations.items():
            actual_boundaries = len(masses) - 1
            possible_boundaries = sum(masses) - 1
            d[item][coder] = (actual_boundaries, possible_boundaries)
    return d

def all_segmentations_of(dataset, item):
    return list(it.chain.from_iterable(dataset[item].values()))

def null_segmentations_for(dataset):
    return se.data.Dataset({ item: { 'NULL': (total_mass(dataset, item),) } 
                             for item in dataset.keys() })

def random_segmentation_of(dataset, item, ratio):
    mass = 0
    masses = []
    for x in range(total_mass(dataset, item)):
        mass += 1
        if (random.random() < ratio):
            masses.append(mass)
            mass = 0
    masses.append(mass)
    return masses

def random_segmentations_for(dataset):
    d = collections.defaultdict(dict)
    boundary_ratios = boundary_ratios_for(dataset)
    for item, coder_ratios in boundary_ratios.items():
        p_of_b = micro_mean(coder_ratios.values())
        d[item]['RANDOM'] = random_segmentation_of(dataset, item, p_of_b)
    return se.data.Dataset(d)

def regular_segmentation_of(dataset, item, num_segments):
    total_m = total_mass(dataset, item)
    segment_m = math.floor(total_m / num_segments)
    remainder = total_m % num_segments
    masses = [segment_m] * num_segments
    for i in range(remainder):
        masses[i] += 1
    return masses

def regular_segmentations_for(dataset):
    d = collections.defaultdict(dict)
    for item, coder_segmentations in dataset.items():
        num_segments = int(round(np.mean(np.array([ len(s) for s in coder_segmentations.values() ]))))
        d[item]['REGULAR'] = regular_segmentation_of(dataset, item, num_segments)
    return se.data.Dataset(d)

