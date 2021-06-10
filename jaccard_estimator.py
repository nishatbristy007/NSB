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
base_str=['A','C','G','T']
encode_str = ['11','10','01','00']
tmp_name = "tmp"  

first_bit_trans = str.maketrans('ACGT', '0110')
second_bit_trans = str.maketrans('ACGT', '1010')


def encodeKmer2(kmer):
    kmer = re.sub(r'A','11',kmer)
    kmer = re.sub(r'C','10',kmer)
    kmer = re.sub(r'G','01',kmer)
    kmer = re.sub(r'T','00',kmer)
    return np.int64(int(kmer,2))

def replaceEncodedKmer(kmer,regex_,encode_base,k_fmt):
    return np.int64(int(re.sub(regex_, encode_base, format(kmer,k_fmt)),2))

def encodeKmer(kmer):
    global first_bit_trans, second_bit_trans
    return np.int64(int(kmer.translate(first_bit_trans)+kmer.translate(second_bit_trans), 2))

def A_to_C(x,size,mask):
    x1= (x & (mask<<size)) >> size
    x2 = x & mask
    xnew=x1 ^ mask
    xnew=x2 & xnew 
    x1=x1 ^ xnew
    x2=x2 ^ xnew
    return np.int64((x1 << size) + x2)
    #return int((x1 << size) + x2)

def C_to_A(x,size,mask):
    x1= (x & (mask<<size)) >> size
    x2 = x & mask
    xnew=x2 ^ mask
    xnew=x1 & xnew 
    x1=x1 ^ xnew
    x2=x2 ^ xnew
    return np.int64((x1 << size) + x2)

def A_to_G(x,size,mask):
    #mask=int(('1'*size),2)
    x1= (x & (mask<<size)) >> size
    x2 = x & mask
    xnew=x1 ^ mask
    xnew=x2 & xnew 
    x1=x1 ^ xnew
    return np.int64((x1 << size) + x2)
    #return int((x1 << size) + x2)

def G_to_A(x,size,mask):
    #mask=int(('1'*size),2)
    #print("{0:b}".format(mask))
    x1= (x & (mask<<size)) >> size
    x2 = x & mask
    
    x1 = x1 & (x1 ^ x2)
    return np.int64((x1 << size) + x2)
    
def A_to_T(x,size,mask):
    #mask=int(('1'*size),2)
    x1= (x & (mask<<size)) >> size
    x2 = x & mask
    xnew=x1 ^ mask
    xnew=x2 & xnew 
    x2=x2 ^ xnew
    return np.int64((x1 << size) + x2)
    #return int((x1 << size) + x2)

def T_to_A(x,size,mask):
    #mask=int(('1'*size),2)
    x1= (x & (mask<<size)) >> size
    x2 = x & mask
    xnew=(x1 | x2) ^ mask
    x2=x2 ^ xnew
    return np.int64((x1 << size) + x2)

def C_to_G(x,size,mask):
    #mask=int(('1'*size),2)
    x1= (x & (mask<<size)) >> size
    x2 = x & mask
    xnew=x2 ^ mask
    xnew=x1 & xnew 
    x2=x2 ^ xnew
    return np.int64((x1 << size) + x2)
    #return int((x1 << size) + x2)

def G_to_C(x,size,mask):
    #mask=int(('1'*size),2)
    #print("{0:b}".format(mask))
    x1= (x & (mask<<size)) >> size
    x2 = x & mask

    x2 = x2 & (x1 ^ x2)
    return np.int64((x1 << size) + x2)

def C_to_T(x,size,mask):
    #mask=int(('1'*size),2)
    x1= (x & (mask<<size)) >> size
    x2 = x & mask
    xnew=x2 ^ mask
    xnew=x1 & xnew 
    x1=x1 ^ xnew
    return np.int64((x1 << size) + x2)

def T_to_C(x,size,mask):
    #mask=int(('1'*size),2)
    x1= (x & (mask<<size)) >> size
    x2 = x & mask
    xnew=(x1 | x2) ^ mask
    x1=x1 ^ xnew
    return np.int64((x1 << size) + x2)
    
def G_to_T(x,size,mask):
    #mask=int(('1'*size),2)
    #print("{0:b}".format(mask))
    x1= (x & (mask<<size)) >> size
    x2 = x & mask
    xnew=x1 ^ x2
    x1= x1 & xnew
    x2 = x2 & xnew
    return np.int64((x1 << size) + x2)

def T_to_G(x,size,mask):
    #mask=int(('1'*size),2)
    x1= (x & (mask<<size)) >> size
    x2 = x & mask
    xnew=(x1 | x2) ^ mask
    x1=x1 ^ xnew
    x2=x2 ^ xnew
    return np.int64((x1 << size) + x2)

def saveEncoding_(folderpath, file):
    sys.stderr.write('Encoding {0}...\n'.format(file))
    taxaname=file.split(".")[0]
    kmers = open(folderpath+"/"+taxaname+".txt", 'r')
    start=time.time()
    set1 = np.array([encodeKmer(kmer.split()[0]) for kmer in kmers], dtype = np.int64)
    sys.stderr.write('Time taken for encoding {0}.\n'.format(time.time()-start))
    start = time.time()        
    with open(encode_dir+"/"+taxaname+'.pickle', 'wb') as f:
        pickle.dump({tmp_name:set1},f)
    end=time.time()

    sys.stderr.write('Encoding done for {0}.\n'.format(file))
    # Deleteing the kmer file from kmer_dir
    call(["rm",folderpath+"/"+file],stderr=open(os.devnull, 'w'))

def saveEncoding(n_pool):
    
    if os.path.exists(encode_dir):
        shutil.rmtree(encode_dir)
    os.mkdir(encode_dir)
    
    folderpath="kmer_dir"
    files = sorted(os.listdir(folderpath))
    
    pool_encode = mp.Pool(n_pool)
    results_encode = [pool_encode.apply_async(saveEncoding_, args=(folderpath,file,)) for file in files]#(len(pathnames))]
    for result in results_encode:
        result.get(9999999)
    pool_encode.close()
    pool_encode.join()    
    
def selectFunction(i,j):
    if i == 0 and j == 1:
        func = A_to_C
    elif i == 0  and j == 2:
        func = A_to_G
    elif i ==0 and j == 3:
        func = A_to_T
    elif i == 1 and j == 0:
        func = C_to_A
    elif i == 1 and j == 2:
        func = C_to_G
    elif i == 1 and j == 3:
        func = C_to_T
    elif i == 2 and j == 0:
        func = G_to_A
    elif i == 2 and j == 1:
        func = G_to_C
    elif i == 2 and j == 3:
        func = G_to_T
    elif i == 3 and j == 0:
        func = T_to_A
    elif i == 3 and j == 1:
        func = T_to_C
    elif i == 3 and j == 2:
        func = T_to_G
    return func
    
   

def replaceEncoding(file,func,k,mask,i,j,encode_dir_replaced):
    taxaname=file.split(".")[0]
    start = time.time()
    with open(encode_dir+"/"+taxaname+".pickle",'rb') as f:
        set1 = pickle.load(f)[tmp_name]
    
    sys.stderr.write('Encoding starting for {0}...\n'.format(file+base_str[i]+base_str[j]))
    start=time.time()
    set2=np.array([func(kmer, k,mask) for kmer in set1],dtype=np.int64)
    sys.stderr.write('Time taken for encoding {0}.\n'.format(time.time()-start))

    start = time.time()
    with open(encode_dir_replaced+"/"+taxaname+base_str[i]+base_str[j]+'.pickle', 'wb') as f:
        pickle.dump({tmp_name: set2},f)
    sys.stderr.write('Encoding saved for {0}...\n'.format(file+base_str[i]+base_str[j]))
    sys.stderr.write('Time taken for saving pickle {0}.\n'.format(time.time()-start))

def saveReplacedEncoding(k, i, j, n_pool):
    global encode_dir
    encode_dir_replaced = "encodedfiles_replaced"+base_str[i]+base_str[j]
    if os.path.exists(encode_dir_replaced):
        shutil.rmtree(encode_dir_replaced)
    os.mkdir(encode_dir_replaced)
    mask=int(('1'*k),2)
    func = selectFunction(i,j)		
    files = sorted(os.listdir(encode_dir))
    
    pool_sketch = mp.Pool(n_pool)
    #print(sequences)
    results_sketch = [pool_sketch.apply_async(replaceEncoding, args=(file,func,k,mask,i,j,encode_dir_replaced,)) for file in files]#(len(pathnames))]
    for result in results_sketch:
        result.get(9999999)
    pool_sketch.close()
    pool_sketch.join()
            
def estimateDistance(J, k):
    return 1 - ((2*J/(1+J))**(1/k))

def estimateDistance2(J1, J2, k):
    return 1 - (((4*J2/(1+J2)) - (2*J1/(1+J1)))**(1/k))


def sortintersection(x,y):
    i=0
    j=0
    lenx=len(x)
    leny=len(y)
    while i < lenx and j < leny:
        if x[i] == y[j]:
            yield x[i]
            i += 1
            j += 1
        elif x[i] > y[j]:
            j += 1
        else:
            i += 1  

def readmatrix(filename, n_taxa):
        f = open(filename,'r')
        lines = f.readlines()
        J = np.zeros((n_taxa, n_taxa))
        for line in lines:
                token = line.split(" ")
                if len(token)<3:
                        continue
                i = int(token[1].strip())
                j = int(token[2].strip())
                jac = float(token[0].strip())
                print(i,j,jac)
                J[i][j] = jac
                J[j][i] = jac
        f.close()
        return J

def clcJaccard(folderpath, files, i, j,filename):
    with open(folderpath+'/'+files[i],'rb') as f:
        set1 = pickle.load(f)[tmp_name]
    with open(folderpath+'/'+files[j],'rb') as f:
        set2 = pickle.load(f)[tmp_name]
    
    start = time.time()
    intersec = np.intersect1d(set1,set2)
    intersec_len=len(intersec)
    sys.stderr.write('Time taken for intersection {0}.\n'.format(time.time()-start))
    union_len = len(set1)+len(set2) - intersec_len
    jaccard=intersec_len/union_len 
    fjac = open(filename,'a')
    fjac.write(str(jaccard)+" "+str(i)+" "+str(j)+"\n")
    fjac.close()

def estimateJaccard(folderpath, n_taxa,token, n_pool):
    sys.stderr.write("[Tool] Estimating jaccard using {0} processors\n".format(n_pool)) 
    files = sorted(os.listdir(folderpath))
    jaccard_matrix = np.zeros(( n_taxa,n_taxa))
    pool_sketch = mp.Pool(n_pool)
    
    filename = 'savejac'+token+'.txt'
    results_sketch = [pool_sketch.apply_async(clcJaccard, args=(folderpath,files,i,j,filename,)) for i in range(n_taxa-1) for j in range(i+1,n_taxa)]#(len(pathnames))]
    for result in results_sketch:
        result.get(9999999)
    pool_sketch.close()
    pool_sketch.join()
    
    return readmatrix(filename, n_taxa)

def preprocess(k, n_taxa, n_pool):
    # Building initial encoded library from k-mers
    saveEncoding(n_pool)
    alphabetSize = len(base_str)
    #Building encoded libraries from k-mers
    for i in range(alphabetSize):
        for j in range(alphabetSize):
            if (i ==1 and (j==0 or j ==3)) or i == 2 or i == 3:
                continue
            if i!=j:
                saveReplacedEncoding(k,i,j,n_pool)


def distEstimatorMaster(k, n_taxa, n_pool):
    global base_str, encode_dir, encode_dir_replaced
    
    #......... PREPROCESS .........
    preprocess(k,n_taxa, n_pool)
    #..............................
    base_subs = 13
    #......... DIST ...............
    dist_matrices = np.zeros((base_subs,n_taxa,n_taxa))
    
    J1 = estimateJaccard(encode_dir,n_taxa,"",n_pool)
    dist_matrices[0] = estimateDistance(J1,k)
    fsave = open('dist.txt','w')
    #sys.stderr.write("dist = {0}".format(dist_matrices[0][0][1]))
    fsave.write(str(dist_matrices[0]))
    fsave.write('\n')
    alphabetSize = len(base_str)
    # -- Jaccard (n,n) of the 12 replaced encodings --
    count = 1
    for i in range(alphabetSize):
        for j in range(alphabetSize):
            if (i ==1 and (j==0 or j ==3)) or i == 2 or i == 3:
                count += 1
                continue
            if i!=j:
                token = base_str[i]+base_str[j]
                encode_dir_replaced = "encodedfiles_replaced"+base_str[i]+base_str[j]
                J2 = estimateJaccard(encode_dir_replaced,n_taxa, token, n_pool)
                if (i == 0 and j == 3) or (i == 1 and j == 2):
                    dist_matrices[count] = estimateDistance(J2,k)
                else :
                    dist_matrices[count] = estimateDistance(J1, J2,k)
                #sys.stderr.write("dist = {0}".format(dist_matrices[count][0][1]))
                fsave.write(str(dist_matrices[count]))
                fsave.write("\n")
                count += 1
    print("jaccards done")
    # --   --                             #  1  2  3  4  5  6  7  8  9  10  11  12
    dist_matrices[4] = dist_matrices[1]   # AC,AG,AT,CA,CG,CT,GA,GC,GT, TA, TC, TG
    dist_matrices[9] = dist_matrices[1]
    dist_matrices[12] = dist_matrices[1]
    dist_matrices[6] = dist_matrices[2]
    dist_matrices[7] = dist_matrices[2]
    dist_matrices[11] = dist_matrices[2]
    dist_matrices[10] = dist_matrices[3]
    dist_matrices[8] = dist_matrices[5]
    
    print("here")
    
    #shutil.rmtree(encode_dir)
    #shutil.rmtree(encode_dir_replaced)
    return dist_matrices
