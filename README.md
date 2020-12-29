# Skmer_extended_tool

## Commands
**For estimating distances with Mash**
```
python Tool.py -m 3 -k 31 -s 100000 ref_dir
```

**For estimating distances with Jellyfish**
Jellyfish has two options. '-m 1' takes less memory and more time. '-m 2' takes more memory, and thus less time.
```
python Tool.py -m 1 -k 31 -s 100000 ref_dir

or, 

python Tool.py -m 2 -k 31 -s 100000 ref_dir
```
