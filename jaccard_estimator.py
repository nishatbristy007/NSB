#from numba import jitclass, njit, int32, int64, float32

import numpy as np
import time
import os
import pickle
import os
from os import listdir
from os.path import isfile, join
from subprocess import call, check_output, STDOUT
kmerdir="kmer_dir"
kmerstatfile="kmer_stats.txt"
half_size=32
base_count=4

w = np.array(list('ACGT'))
v = np.array([3,2,1,0])
s = np.flip(1 << (2 * np.arange(32)))
base_str=['A','C','G','T']
def encode(kmer):
    global w, v, s
    n = len(kmer)
    kmer = np.array(list(kmer))[:,np.newaxis]
    p = (kmer == w)
    b = np.sum(s[-n:] @ (p * v))
    return np.int64(b)

# Saves the k-mer encodings of each file in the 'folderpath'.
# First runs Jellyfish with canonical or non-canonical setting, specified by 
# first base will be substitituted by second base
def replace_encoding(kmer,first_base,second_base,ksize):
    mask=int('1'*ksize,2)
    b_str="{0:b}".format(kmer)
    b_first,b_second=b_str[::2],b_str[1::2]
    x1=int(b_first,2)
    x2=int(b_second,2)
    if first_base=='A' and second_base=='C':
        xnew=x1 ^ x2
        x2=x2 & xnew
    elif first_base=='C' and second_base=='A':
        xnew=(x1 ^ x2)
        x2=x2 | xnew
    elif first_base=='A' and second_base=='G':
        xnew=x1 ^ x2
        x1=x1 & xnew
    elif first_base=='G' and second_base=='A':
        xnew=(x1 ^ x2)
        x1=x1 | xnew
    elif first_base=='A' and second_base=='T':
        xnew=(x1 ^ x2)
        x1=x1 & xnew
        x2=x2 & xnew
    elif first_base=='T' and second_base=='A':
        xnew=(x1 | x2) ^ mask
        x1=x1 ^ xnew
        x2=x2 ^ xnew
    elif first_base=='C' and second_base=='G':
        xnew=x2 ^ mask
        xnew=x1 & xnew
        x1=x1 ^ xnew
        x2=x2 ^ xnew
    elif first_base=='G' and second_base=='C':
        xnew=x1 ^ mask
        xnew=x2 & xnew
        x1=x1 ^ xnew
        x2=x2 ^ xnew
    elif first_base=='C' and second_base=='T':
        xnew=x2 ^ mask
        xnew=x1 & xnew
        x1=x1 ^ xnew
    elif first_base=='T' and second_base=='C':
        xnew=(x1 | x2) ^ mask
        x1=x1 ^ xnew
    elif first_base=='G' and second_base=='T':
        xnew=x1 ^ mask
        xnew=x2 & xnew
        x2=x2 ^ xnew
    elif first_base=='T' and second_base=='G':
        xnew=(x1 | x2) ^ mask
        x2=x2 ^ xnew    
    x1="{0:b}".format(x1)
    x2="{0:b}".format(x2)
    x1=x1.zfill(half_size)
    x2=x2.zfill(half_size)
    originalstring = ''.join([''.join(x) for x in zip(x1,x2)])
    return int(originalstring,2)
def save_encoding(folderpath,ksize):
    call(["mkdir","-p","encodedfiles"],stderr=open(os.devnull, 'w'))
    files = os.listdir(folderpath)
    for file in files:
        print("Encoding starting for "+file)
        taxaname=file.split(".")[0]
        kmers = open(folderpath+"/"+file, 'r')
        kmerlinecounts=open(kmerstatfile,'r').readlines()
        for line in kmerlinecounts:
            if line.split()[0]==taxaname+".txt":
                len_taxa=int(line.split()[1])
        print(len_taxa)
        count = 0
        set1 = np.array([encode(kmer.split()[0]) for kmer in kmers], dtype =  np.int64)
        new_dict={taxaname:set1}
        with open("encodedfiles/"+taxaname+'.pickle', 'wb') as f:
            pickle.dump(new_dict,f)
        print(len(set1))
        for i in range(base_count):
            for j in range(base_count):
                if i!=j:
                    set2=np.array([replace_encoding(kmer,base_str[i],base_str[j],ksize) for kmer in set1])
                    tmp_name=taxaname+base_str[i]+base_str[j]
                    new_dict={tmp_name: set2}
                    with open("encodedfiles/"+taxaname+base_str[i]+base_str[j]+'.pickle', 'wb') as f:
                        pickle.dump(new_dict,f)
                    print(base_str[i]+base_str[j]+" encoding saved")

def calculateJaccardPickle(taxa1_name,taxa2_name):
    jaccard_matrix=np.zeros((base_count,base_count))
    taxa1_kmer=taxa1_name.split(".")[0]
    taxa2_kmer=taxa2_name.split(".")[0]
    taxa1_kmerfile=taxa1_kmer+".txt"
    taxa2_kmerfile=taxa2_kmer+".txt"
    kmerlinecounts=open(kmerstatfile,'r').readlines()
    for line in kmerlinecounts:
        if line.split()[0]==taxa1_kmerfile:
            len_taxa1=int(line.split()[1])
        if line.split()[0]==taxa2_kmerfile:
            len_taxa2=int(line.split()[1])
    print(len_taxa1)
    print(len_taxa2)
    with open("encodedfiles/"+taxa1_kmer+".pickle",'rb') as f:
        set1 = pickle.load(f)[taxa1_kmer]
    with open("encodedfiles/"+taxa2_kmer+".pickle",'rb') as f:
        set2 = pickle.load(f)[taxa2_kmer]
    intersec = np.intersect1d(set1,set2)
    intersec_len=len(intersec)
    union_len = len_taxa1 + len_taxa2 - intersec_len
    normal_jaccard=intersec_len/union_len
    print("Intersection: ", intersec_len, "\nUnion: ", union_len)
    for i in range(base_count):
        for j in range(base_count):
            if i==j:
                jaccard_matrix[i,j]=normal_jaccard
            else:
                print("calculate"+base_str[i]+base_str[j]+" jaccard")
                with open("encodedfiles/"+taxa1_kmer+base_str[i]+base_str[j]+".pickle",'rb') as f:
                    set1 = pickle.load(f)[taxa1_kmer+base_str[i]+base_str[j]]
                with open("encodedfiles/"+taxa2_kmer+base_str[i]+base_str[j]+".pickle",'rb') as f:
                    set2 = pickle.load(f)[taxa2_kmer+base_str[i]+base_str[j]]
                intersec_len=len(np.intersect1d(set1,set2))
                union_len = len_taxa1 + len_taxa2 - intersec_len
                replace_jaccard=intersec_len/union_len
                print("Intersection: ", intersec_len, "\nUnion: ", union_len)
                jaccard_matrix[i,j]=replace_jaccard
    return jaccard_matrix
#save_encoding("kmer_dir",31)
#calculateJaccardPickle("taxa1_2Way.fasta","taxa2_2Way.fasta")
'''with open('encodedfiles/taxa1_AC.pickle','rb') as f:
    new_dict=pickle.load(f)
print(len(new_dict['taxa1_AC']))'''
def calculateJaccard(taxa1_name,taxa2_name,ksize):
    jaccard_matrix=np.zeros((base_count,base_count))
    taxa1_kmer=taxa1_name.split(".")[0]+".txt"
    taxa2_kmer=taxa2_name.split(".")[0]+".txt"
    taxa1 = open(kmerdir+"/"+taxa1_kmer, 'r') .readlines()
    taxa2 = open(kmerdir+"/"+taxa2_kmer, 'r') .readlines()
    kmerlinecounts=open(kmerstatfile,'r').readlines()
    for line in kmerlinecounts:
        if line.split()[0]==taxa1_kmer:
            len_taxa1=int(line.split()[1])
        if line.split()[0]==taxa2_kmer:
            len_taxa2=int(line.split()[1])
    print(len_taxa1)
    print(len_taxa2)   
    print("starting..")
    count = 0
    
    start = time.time()
    
    set1 = np.array([encode(kmer.split()[0]) for kmer in taxa1], dtype =  np.int64)
    print("set1 done")
    set1_time = time.time()
    set2 = np.array([encode(kmer.split()[0]) for kmer in taxa2], dtype =  np.int64)
    set2_time = time.time()
    print("set2 done")
    
    intersec = np.intersect1d(set1,set2)
    intersec_len=len(intersec)
    intersec_time = time.time()
    
    union_len = len_taxa1 + len_taxa2 - intersec_len
    normal_jaccard=intersec_len/union_len
    print("Intersection: ", intersec_len, "\nUnion: ", union_len)
    print("Set1 took: ", (set1_time-start))
    print("Set2 took: ", (set2_time-set1_time))
    print("intersection took: ", (intersec_time-set2_time))
    for i in range(base_count):
        for j in range(base_count):
            if i==j:
                jaccard_matrix[i,j]=normal_jaccard
            else:
                print("calculate"+base_str[i]+base_str[j]+" jaccard")
                set3=np.array([replace_encoding(kmer,base_str[i],base_str[j],ksize) for kmer in set1])
                print("set3 done")
                set4=np.array([replace_encoding(kmer,base_str[i],base_str[j],ksize) for kmer in set2])
                print("set4 done")
                intersec_len=len(np.intersect1d(set3,set4))
                union_len = len_taxa1 + len_taxa2 - intersec_len
                replace_jaccard=intersec_len/union_len
                print("Intersection: ", intersec_len, "\nUnion: ", union_len)
                jaccard_matrix[i,j]=replace_jaccard
    return jaccard_matrix
#print(base_str[3])    
#calculateJaccard("taxa1_.fasta","taxa2_.fasta",31)
# kmers Taxa1 = 148243322
# kmers Taxa2 = 157387762