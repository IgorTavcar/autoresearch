# Evaluating Experiment Results at Scale

After running 50+ experiments, eyeballing val_bpb deltas in results.tsv stops working.
This guide covers noise floor estimation, Pareto efficiency, and practical one-liners.

## Understanding val_bpb

val_bpb (validation bits per byte) measures how well the model compresses held-out text.
Lower is better. The metric is vocab-size-independent, so results are comparable across
tokenizers. Typical range for this model size: 0.9–1.3.

Small deltas matter: a 0.005 improvement may represent genuine architectural progress.
But GPU nondeterminism means consecutive identical runs will produce slightly different
scores. You need to distinguish real improvements from noise.

## Noise floor estimation

The noise floor is the typical random variation between runs with no code change.
Estimate it from consecutive keeps in results.tsv:

```bash
# Median absolute pairwise difference between consecutive keeps
awk -F'\t' '$6=="keep" {if(prev!="") print (prev-$3<0?$3-prev:prev-$3); prev=$3}' results.tsv | \
  sort -n | awk '{a[NR]=$1} END {print a[int(NR/2)+1]}'
```

With fewer than 5 keeps, this estimate is unreliable. Use an absolute threshold of
0.003 as a conservative starting point.

## When to trust an improvement

| Delta vs noise floor | Verdict |
|---------------------|---------|
| > 1.5× noise floor | Probably real — keep |
| 1.0–1.5× noise floor | Ambiguous — lean on simplicity (keep if simpler, discard if more complex) |
| < 1.0× noise floor | Probably noise — discard unless the change is a simplification |

## Pareto efficiency

Track the val_bpb vs memory_gb tradeoff. An experiment is Pareto-dominated if another
experiment achieves both lower val_bpb AND lower memory usage. Focus on the Pareto
frontier — experiments that aren't dominated by any other.

## Useful one-liners

```bash
# Best val_bpb achieved
awk -F'\t' '$6=="keep" {print $3}' results.tsv | sort -n | head -1

# Total / keep / discard / crash counts
awk -F'\t' 'NR>1 {t++; s[$6]++} END {printf "total=%d keep=%d discard=%d crash=%d\n",t,s["keep"],s["discard"],s["crash"]}' results.tsv

# Memory range across keeps
awk -F'\t' '$6=="keep" {print $5}' results.tsv | sort -n | awk 'NR==1{lo=$1} END{print "mem: "lo" - "$1" GB"}'

# Last 10 experiments
tail -10 results.tsv | column -t -s$'\t'

# Biggest single improvement (between consecutive keeps)
awk -F'\t' '$6=="keep" {if(prev!="") print prev-$3, NR; prev=$3}' results.tsv | sort -rn | head -5

# Success rate over time (rolling window of 10)
awk -F'\t' 'NR>1 {a[NR]=$6} END {for(i=11;i<=NR;i++){k=0;for(j=i-9;j<=i;j++)if(a[j]=="keep")k++;print i, k"/10"}}' results.tsv

# Keeps-only view
awk -F'\t' 'NR==1 || $6=="keep"' results.tsv | column -t -s$'\t'
```
