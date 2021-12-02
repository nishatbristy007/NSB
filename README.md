# NSB: Genome-wide alignment-free distance estimation under a no strand-bias model.

## Short Description

**NSB** is a tool for alignment-free phylogenetic distance estimation, under a no strand-bias, time reversible GTR model, TK4. 

Alignment-free methods are useful because of their simplification of the pipeline/process of phylogenetic inference. However, despite the appeal, the accuracies of the alignment-based methods most often surpass the ones with an alignment-free setting, as the alignment-free methods use simple base-substitution GTR models, like Jukes-Cantor. 

NSB uses a base-substitution technique on **k-mers** to identify the frequencies of transitions and transversions, and thus allows the use of more complex sequence evaluation models. This enables NSB to estimate more accurate phylogenetic distances, even when the true distances are high. 

## Execution and dependencies

- Python 3
- NumPy, Pandas, Pickle

## Input and Output formats of NSB

### Input
- NSB takes input in **fasta** and **fna** formats.
- All the sequence files are needed to be saved in a directory, i.e. see **ref_dir** folder for example. 

### Output
The output to NSB is a n\*n distance matrix, saved under the name **ref-dist-mat-nsb-ref_dir.txt**.

## Commands for running **NSB**

**For estimating distances with Jellyfish**

```
python NSB.py -m 2 -k 31 -s 100000 -p 4 ref_dir
```

