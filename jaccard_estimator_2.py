import numpy as np
import time
import os
import pickle
import os
from os import listdir
from os.path import isfile, join
from subprocess import call, check_output, STDOUT
import shutil
import re
import sys
import multiprocessing as mp
kmerdir="kmer_dir"
encode_dir = "encodedfiles"
encode_dir_replaced = "encodedfiles_replaced"
kmerstatfile="kmer_stats.txt"
half_size=32
base_count=4
base_subs = 13
base_str=['A','C','G','T']
encode_str = ['11','10','01','00']
tmp_name = "tmp"  
def encodeKmer(kmer):
    kmer = re.sub(r'A','11',kmer)
    kmer = re.sub(r'C','10',kmer)
    kmer = re.sub(r'G','01',kmer)
    kmer = re.sub(r'T','00',kmer)
    return np.int64(int(kmer,2))

def replaceEncodedKmer(kmer,regex_,encode_base,k_fmt):
    return np.int64(int(regex_.sub( encode_base, format(kmer,k_fmt)),2))

def replace_encoding(kmer,first_base,second_base,ksize):
    mask=int('1'*ksize,2)
    b_str="{0:b}".format(kmer)
    b_str=b_str.zfill(ksize*2)
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
    x1=x1.zfill(ksize)
    x2=x2.zfill(ksize)
    originalstring = ''.join([''.join(x) for x in zip(x1,x2)])
     
    
    return np.int64(int(originalstring,2))
    
def saveEncoding(k):
    global tmp_name
    folderpath="kmer_dir"
    files = os.listdir(folderpath)
    for file in files:
        taxaname=file.split(".")[0]
        sys.stderr.write('File read for {0}...\n'.format(file))
        
        kmers = open(folderpath+"/"+taxaname+".txt", 'r')
        sys.stderr.write('Encoding starting for {0}...\n'.format(file))
        start=time.time()
        set1 = np.array([encodeKmer(kmer.split()[0]) for kmer in kmers], dtype =  np.int64)
        new_dict={tmp_name:set1}
        with open("encodedfiles/"+taxaname+'.pickle', 'wb') as f:
            pickle.dump(new_dict,f)
        end=time.time()

        sys.stderr.write('Encoding done for {0}.\n'.format(file))
        sys.stderr.write('Time taken {0}.\n'.format(end-start))

        # Deleteing the kmer file from kmer_dir
        call(["rm",folderpath+"/"+file],stderr=open(os.devnull, 'w'))
    
def saveReplacedEncoding(k, i, j):
    global encode_dir, encode_dir_replaced, tmp_name, base_str
    if os.path.exists(encode_dir_replaced):
        shutil.rmtree(encode_dir_replaced)
    os.mkdir(encode_dir_replaced)

    files = os.listdir(encode_dir)
    for file in files:
        taxaname=file.split(".")[0]
        sys.stderr.write("opening pickle file\n")
        start = time.time()
        with open("encodedfiles/"+taxaname+".pickle",'rb') as f:
            set1 = pickle.load(f)[tmp_name]
        end = time.time()
        sys.stderr.write('Time taken for opening {0}.\n'.format(end-start))

        sys.stderr.write('Encoding starting for {0}...\n'.format(file+base_str[i]+base_str[j]))
        start=time.time()
        regex_ = re.compile(encode_str[i]+'(?=(?:[\d0-1]{2})*$)')
        encode_base = encode_str[j]
        k_fmt = '0'+str(2*k)+'b'
        from_str = base_str[i]
        to_str = base_str[j]
        #set2=np.array([np.int64(int(re.sub(regex_, encode_base, format(kmer,k_fmt)),2)) for kmer in set1])
        set2 = np.array([replace_encoding(kmer, from_str, to_str, k) for kmer in set1])
        #set2 = np.array([replaceEncodedKmer(kmer,regex_, encode_base,k_fmt) for kmer in set1])
        
        #pool_sketch = mp.Pool(mp.cpu_count() -1)
        #results_sketch = [pool_sketch.apply_async(replaceEncodedKmer, args=(kmer,regex_, encode_base,k_fmt)) for kmer in set1]
        sys.stderr.write("Encoding done\n")
        #set2 = np.array([result.get(9999999) for result in results_sketch], dtype=np.int64)#(len(pathnames))]
        #for result in results_sketch:
        #    result.get(9999999)
        #pool_sketch.close()
        #pool_sketch.join()
        
        
        new_dict={tmp_name: set2}
        with open("encodedfiles_replaced/"+taxaname+base_str[i]+base_str[j]+'.pickle', 'wb') as f:
            pickle.dump(new_dict,f)
        end=time.time()
        sys.stderr.write('Encoding saved for {0}...\n'.format(file+base_str[i]+base_str[j]))
        sys.stderr.write('Time taken {0}.\n'.format(end-start))

def estimateJaccard(folderpath, n_taxa):
    global tmp_name
    files = os.listdir(folderpath)
    jaccard_matrix = np.zeros((n_taxa,n_taxa))
    for  i in range(n_taxa-1):
        for j in range(i+1, n_taxa):
            with open(folderpath+'/'+files[i],'rb') as f:
                set1 = pickle.load(f)[tmp_name]
            with open(folderpath+'/'+files[j],'rb') as f:
                set2 = pickle.load(f)[tmp_name]
            start = time.time()
            intersec = np.intersect1d(set1,set2)
            intersec_len=len(intersec)
            union_len = len(np.union1d(set1,set2))
            end = time.time()
            sys.stderr.write('Time taken for intersection and union {0}.\n'.format(end-start))
            jaccard=intersec_len/union_len
            jaccard_matrix[i][j] = jaccard_matrix[j][i] = jaccard
    return jaccard_matrix
            
def estimateDistance(matrix, k):
    return 1 - ((2*matrix/(1+matrix))**(1/k))

def distEstimatorMaster(k, n_taxa):
    global base_str, encode_dir, encode_dir_replaced
    # -- Creating directories to save encodings--
    if os.path.exists(encode_dir):
        shutil.rmtree(encode_dir)
    
    os.mkdir(encode_dir)
    # --   --

    # -- Distance (n,n) of the actual encodings --
    dist_matrices = np.zeros((base_subs,n_taxa,n_taxa))
    saveEncoding(k)
    dist_matrices[0] = estimateDistance(estimateJaccard(encode_dir,n_taxa),k)
    # --   --

    alphabetSize = len(base_str)
    # -- Jaccard (n,n) of the 12 replaced encodings --
    count = 1
    for i in range(alphabetSize):
        for j in range(alphabetSize):
            if i!=j:
                saveReplacedEncoding(k,i,j)
                dist_matrices[count] = estimateDistance(estimateJaccard(encode_dir_replaced,n_taxa),k)
                count += 1
    # --   --
    shutil.rmtree(encode_dir)
    shutil.rmtree(encode_dir_replaced)
    return dist_matrices