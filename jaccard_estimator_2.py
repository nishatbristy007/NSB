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

def saveEncoding(k):
    folderpath="kmer_dir"
    files = os.listdir(folderpath)
    print(files)
    for file in files:
        sys.stderr.write('Encoding starting for {0}...\n'.format(file))
        taxaname=file.split(".")[0]

        kmers = open(folderpath+"/"+taxaname+".txt", 'r')
        start=time.time()
        set1 = np.array([encodeKmer(kmer.split()[0]) for kmer in kmers], dtype = np.int64)
        #set1 = np.array([encodeKmer(kmers.pop().split()[0]) for i in range(len(kmers))], dtype = np.int64)
        sys.stderr.write('Time taken for encoding {0}.\n'.format(time.time()-start))

        start = time.time()
        
        with open("encodedfiles/"+taxaname+'.pickle', 'wb') as f:
            pickle.dump({tmp_name:set1},f)
        end=time.time()

        sys.stderr.write('Encoding done for {0}.\n'.format(file))
        #sys.stderr.write('Time taken for saving pickle file {0}.\n'.format(end-start))

        # Deleteing the kmer file from kmer_dir
        call(["rm",folderpath+"/"+file],stderr=open(os.devnull, 'w'))
    
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
            
def saveReplacedEncoding(k, i, j):
    global encode_dir, encode_dir_replaced
    if os.path.exists(encode_dir_replaced):
        shutil.rmtree(encode_dir_replaced)
    os.mkdir(encode_dir_replaced)
    mask=int(('1'*k),2)
    
    
    func = selectFunction(i,j)		
    files = os.listdir(encode_dir)
    for file in files:
        taxaname=file.split(".")[0]
        start = time.time()
        with open("encodedfiles/"+taxaname+".pickle",'rb') as f:
            set1 = pickle.load(f)[tmp_name]
        #sys.stderr.write('Time for loading pickle file {0}...\n'.format(time.time()-start))
        sys.stderr.write('Encoding starting for {0}...\n'.format(file+base_str[i]+base_str[j]))
        start=time.time()
        '''
        regex_ = encode_str[i]+'(?=(?:[\d0-1]{2})*$)'
        encode_base = encode_str[j]
        k_fmt = '0'+str(2*k)+'b'
        set2=np.array([np.int64(int(re.sub(regex_, encode_base, format(kmer,k_fmt)),2)) for kmer in set1])
        '''
        #set2=np.array([np.int64(int(re.sub(regex_, encode_base, format(kmer,k_fmt)),2)) for kmer in set1])
        #mask=int(('1'*k),2)
        lset1 = len(set1)
        #set2=np.array([func(set1.pop(), k,mask) for i in range(lset1)],dtype=np.int64)
        set2=np.array([func(kmer, k,mask) for kmer in set1],dtype=np.int64)
        
        sys.stderr.write('Time taken for encoding {0}.\n'.format(time.time()-start))

        start = time.time()
        with open("encodedfiles_replaced/"+taxaname+base_str[i]+base_str[j]+'.pickle', 'wb') as f:
            pickle.dump({tmp_name: set2},f)
        sys.stderr.write('Encoding saved for {0}...\n'.format(file+base_str[i]+base_str[j]))
        sys.stderr.write('Time taken for saving pickle {0}.\n'.format(time.time()-start))

def estimateJaccard(folderpath, n_taxa):
    sys.stderr.write("estimateJaccard\n") 
    files = os.listdir(folderpath)
    jaccard_matrix = np.zeros(( n_taxa,n_taxa))
    for  i in range(n_taxa-1):
        for j in range(i+1, n_taxa):
            #print(files[i], files[j])
            start = time.time()
            with open(folderpath+'/'+files[i],'rb') as f:
                set1 = pickle.load(f)[tmp_name]
            with open(folderpath+'/'+files[j],'rb') as f:
                set2 = pickle.load(f)[tmp_name]
            sys.stderr.write('Time taken for reading pickles {0}.\n'.format(time.time()-start))
            
            start = time.time()
            #set1 = set(set1)
            #set2 = set(set2)
            #sys.stderr.write('Time taken for making sets {0}.\n'.format(time.time()-start))
            #start = time.time()
            #xsorted = np.sort(set1)#sorted(set1)
            #ysorted = np.sort(set2)#sorted(set2)
            #sys.stderr.write('Time taken for sorting list {0}.\n'.format(time.time()-start))
            start = time.time()
            intersec = np.intersect1d(set1,set2)
            intersec_len=len(intersec)
            #intersec_len = len(list(sortintersection(xsorted,ysorted)))
            sys.stderr.write('Time taken for intersection {0}.\n'.format(time.time()-start))
            #start = time.time()
            union_len = len(set1)+len(set2) - intersec_len
            #sys.stderr.write('Time taken for len union {0}.\n'.format(time.time()-start))
            #intersec = np.intersect1d(set1,set2)
            #intersec_len=len(intersec)
            #start = time.time()
            #union_len = len(np.union1d(set1,set2))
            #end = time.time()
            #sys.stderr.write('Time taken for union1d union {0}.\n'.format(time.time()-start))

            jaccard=intersec_len/union_len
            jaccard_matrix[i][j] = jaccard_matrix[j][i] = jaccard
    return jaccard_matrix
            
def estimateDistance(J, k):
    return 1 - ((2*J/(1+J))**(1/k))

def estimateDistance2(J1, J2, k):
    return 1 - (((4*J2/(1+J2)) - (2*J1/(1+J1)))**(1/k))

def distEstimatorMaster(k, n_taxa):
    global base_str, encode_dir, encode_dir_replaced
    # -- Creating directories to save encodings--
    fsave = open("dist.txt", "a")
    if os.path.exists(encode_dir):
        shutil.rmtree(encode_dir)
       
    sys.stderr.write("n = {0}\n".format(n_taxa))
    os.mkdir(encode_dir)
    
    # --   --

    # -- Jaccard (n,n) of the actual encodings --
    dist_matrices = np.zeros((base_subs,n_taxa,n_taxa))
    saveEncoding(k)
    J1 = estimateJaccard(encode_dir,n_taxa)
    dist_matrices[0] = estimateDistance(J1,k)
    # --   --
    sys.stderr.write("dist = {0}".format(dist_matrices[0][0][1]))
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
                saveReplacedEncoding(k,i,j)
                J2 = estimateJaccard(encode_dir_replaced,n_taxa)
                if (i == 0 and j == 3) or (i == 1 and j == 2):
                    dist_matrices[count] = estimateDistance(J2,k)
                else :
                    dist_matrices[count] = estimateDistance(J2,k)
                sys.stderr.write("dist = {0}".format(dist_matrices[count][0][1]))
                fsave.write(str(dist_matrices[count]))
                fsave.write("\n")
                count += 1
    # --   --                             #  1  2  3  4  5  6  7  8  9  10  11  12
    dist_matrices[4] = dist_matrices[1]   # AC,AG,AT,CA,CG,CT,GA,GC,GT, TA, TC, TG
    dist_matrices[9] = dist_matrices[1]
    dist_matrices[12] = dist_matrices[1]
    dist_matrices[6] = dist_matrices[2]
    dist_matrices[7] = dist_matrices[2]
    dist_matrices[11] = dist_matrices[2]
    dist_matrices[10] = dist_matrices[3]
    dist_matrices[8] = dist_matrices[5]
    
    
    
    shutil.rmtree(encode_dir)
    shutil.rmtree(encode_dir_replaced)
    return dist_matrices
