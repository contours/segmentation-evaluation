# Parameters:
# w: pseudosentence (token sequence) size, 14–26
# k: block size, 6–14
# m: depth score cutoff, -1.0–+1.0
#    -0.75 means a cutoff of (mean(depth) - 0.75*(stddev(depth))
#    +0.75 means a cutoff of (mean(depth) + 0.75*(stddev(depth))
# n: number of rounds of smoothing, 1–2
# s: smoothing width, 1–2
w = [ 'w%s' % i                for i in range(14, 27, 2) ]
k = [ 'k%s' % i                for i in range(6, 15, 2) ]
m = [ 'm{:+.2f}'.format(i/100) for i in range(25,101,25) ]
n = [ 'n%s' % i                for i in range(1, 4) ]
s = [ 's%s' % i                for i in range(1, 4) ]

param_combos = [ '-'.join(params) for params in it.product(w,k,m,n,s) ]

texttiling = { params: load_dataset('texttiling-parameter-sweep/TextTiling-%s.json' % params) 
               for params in param_combos}

--------------------------------------------------------------------------------

datasets = [ [params, merge_datasets(human_segmentations, texttiling[params])] 
             for params in param_combos ]
print('%s parameter combinations\n' % len(datasets))

for i,l in enumerate(datasets):
    dataset = l[1]
    l.append(pi(dataset))
    if i % 20 == 0: print('.', end='')

--------------------------------------------------------------------------------

for params, dataset, agreement in list(sorted(datasets, key=lambda t: t[2], reverse=True))[:50]:
    print(params)
    show_similarity_and_agreement(dataset)
    print()
