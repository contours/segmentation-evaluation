import os
import math
import segeval as se
import numpy as np
import itertools as it
import functools
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

def mean_std_n(values):
    floats = np.array(list(values), dtype=float)
    # ddof=1 means calculate sample std dev
    return (np.mean(floats), np.std(floats, ddof=1), len(floats))

def mean_std_n_str(values):
    return '{0:.4f} ±{1:.4f}, n={2}'.format(*mean_std_n(values))

def mean_ci_n(values):
    mean, sd, n = mean_std_n(values)
    se = sd / np.sqrt(n)
    ci = 1.96 * se
    return mean, ci, n

def mean_ci_n_str(values):
    return '{0:.4f} ±{1:.4f} (95% CI), n={2}'.format(*mean_ci_n(values))

def b_confusion(hypothesis, reference):
    def sum_matrices(a,b):
        m = se.ConfusionMatrix()
        classes = a.classes() | b.classes()
        for c1,c2 in it.product(classes,classes):
            m[c1][c2] = a[c1][c2] + b[c1][c2]
        return m
    cms = se.boundary_confusion_matrix(hypothesis, reference, n_t=MAX_TRANSPOSE_DISTANCE)
    return functools.reduce(sum_matrices, cms.values())

def b_statistics(*datasets):
    return se.boundary_statistics(*datasets, n_t=MAX_TRANSPOSE_DISTANCE)

def b(*datasets, return_parts=False):
    return se.boundary_similarity(*datasets, n_t=MAX_TRANSPOSE_DISTANCE, return_parts=return_parts)

def boundary_pair_scores(total_score, n_pairs, additions, _, transpositions):
    n_matches = n_pairs - len(additions) - len(transpositions)
    scores = (((Decimal(1),) * n_matches) +
              ((Decimal(0),) * len(additions)) +
              tuple([ (1 - (abs(t[0]-t[1]) / Decimal(MAX_TRANSPOSE_DISTANCE))) for t in transpositions ]))
    return scores

def b_pairwise(dataset):
    return { label: boundary_pair_scores(*parts)
             for label, parts in b(dataset, return_parts=True).items() }

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
    scores = tuple(it.chain(*[ v for k,v in b_pairwise(dataset).items()
                               if (coder in k) and (item in k) ]))
    return mean_std_n(scores)[0]

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

def boundary_placement_rate(dataset, coder=None):
    if coder is not None:
        return (boundaries_placed(dataset, coder)
                / potential_boundaries(dataset))
    else:
        return (boundaries_placed(dataset)
                / (potential_boundaries(dataset) * len(dataset.coders)))

def boundary_ratios_for(dataset):
    d = collections.defaultdict(dict)
    for item, segmentations in dataset.items():
        for coder, masses in segmentations.items():
            actual_boundaries = len(masses) - 1
            possible_boundaries = sum(masses) - 1
            d[item][coder] = (actual_boundaries, possible_boundaries)
    return d

def all_segments_of(dataset, item=None):
    if item is not None:
        return tuple(it.chain.from_iterable(
            dataset[item].values()))
    else:
        return tuple(it.chain.from_iterable(
            [ all_segments_of(dataset, i) for i in dataset.keys() ]))

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
    p_b = boundary_placement_rate(dataset)
    for item in dataset.keys():
        d[item]['RANDOM'] = random_segmentation_of(dataset, item, p_b)
    return se.data.Dataset(d)

def uniform_segmentation_of(dataset, item, target_mass):
    total_m = total_mass(dataset, item)
    n_segments = round(total_m / target_mass)
    segment_m = round(total_m / n_segments)
    masses = [segment_m] * n_segments
    remainder = math.floor((total_m - (n_segments * segment_m)) / n_segments)
    for i in range(len(masses)):
        masses[i] += remainder
    masses[-1] += (total_m - (n_segments * segment_m)) % n_segments
    assert sum(masses) == total_m
    return masses

def uniform_segmentations_for(dataset):
    uniform_segment_mass = math.ceil(np.median(all_segments_of(dataset)))
    d = collections.defaultdict(dict)
    for item in dataset.keys():
        d[item]['UNIFORM'] = uniform_segmentation_of(
            dataset, item, uniform_segment_mass)
    return se.data.Dataset(d)

def keep(dataset, items):
    return se.data.Dataset({ item: segmentations
                             for item, segmentations in dataset.items()
                             if item in items })
