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
import math
from math import comb
import pandas as pd


kmerdir="kmer_dir"
encode_dir = "encodedfiles"
base_str=['A','C','G','T']
encode_str = ['11','10','01','00']
tmp_name = "tmp"  
f_int = open("intersections.txt",'w')
f_int.close()
first_bit_trans = str.maketrans('ACGT', '0110')
second_bit_trans = str.maketrans('ACGT', '1010')

csv_path = "dist_params.csv"

if os.path.isfile(csv_path):
    file_csv = open(csv_path,"a")
else:
    file_csv = open(csv_path,"w")

    
def encode4(scaf,k):
    L=len(scaf)
    mask=int(('1'*k),2)
    enc_b1={'A':0b0,'C':0b1,'G':0b1,'T':0b0}
    enc_b2={'A':0b1,'C':0b0,'G':0b1,'T':0b0}
    if L < k:
        print(scaf)
        yield []
    else:
        z=0
        z_1=0
        z_2=0
        for i in range(0,k):
            z_1=(z_1<<1|enc_b1[scaf[i]])&mask
            z_2=(z_2<<1|enc_b2[scaf[i]])&mask
            z=(z_1<<k)|z_2
        yield z
        for i in range(k,L):
            z_1=(z_1<<1|enc_b1[scaf[i]])&mask
            z_2=(z_2<<1|enc_b2[scaf[i]])&mask
            z=(z_1<<k)|z_2
            yield z
    

def replaceEncodedKmer(kmer,regex_,encode_base,k_fmt):
    return np.int64(int(re.sub(regex_, encode_base, format(kmer,k_fmt)),2))


def A_to_C(x,size,mask):
    x1= (x & (mask<<size)) >> size
    x2 = x & mask
    xnew=x1 ^ mask
    xnew=x2 & xnew 
    x1=x1 ^ xnew
    x2=x2 ^ xnew
    return (x1 << size) + x2
    #return int((x1 << size) + x2)

def C_to_A(x,size,mask):
    x1= (x & (mask<<size)) >> size
    x2 = x & mask
    xnew=x2 ^ mask
    xnew=x1 & xnew 
    x1=x1 ^ xnew
    x2=x2 ^ xnew
    return (x1 << size) + x2

def A_to_G(x,size,mask):
    #mask=int(('1'*size),2)
    x1= (x & (mask<<size)) >> size
    x2 = x & mask
    xnew=x1 ^ mask
    xnew=x2 & xnew 
    x1=x1 ^ xnew
    return (x1 << size) + x2
    #return int((x1 << size) + x2)

def G_to_A(x,size,mask):
    #mask=int(('1'*size),2)
    ##print("{0:b}".format(mask))
    x1= (x & (mask<<size)) >> size
    x2 = x & mask
    
    x1 = x1 & (x1 ^ x2)
    return (x1 << size) + x2
    
def A_to_T(x,size,mask):
    #mask=int(('1'*size),2)
    x1= (x & (mask<<size)) >> size
    x2 = x & mask
    xnew=x1 ^ mask
    xnew=x2 & xnew 
    x2=x2 ^ xnew
    return (x1 << size) + x2
    #return int((x1 << size) + x2)

def T_to_A(x,size,mask):
    #mask=int(('1'*size),2)
    x1= (x & (mask<<size)) >> size
    x2 = x & mask
    xnew=(x1 | x2) ^ mask
    x2=x2 ^ xnew
    return (x1 << size) + x2

def C_to_G(x,size,mask):
    #mask=int(('1'*size),2)
    x1= (x & (mask<<size)) >> size
    x2 = x & mask
    xnew=x2 ^ mask
    xnew=x1 & xnew 
    x2=x2 ^ xnew
    return (x1 << size) + x2
    #return int((x1 << size) + x2)

def G_to_C(x,size,mask):
    #mask=int(('1'*size),2)
    ##print("{0:b}".format(mask))
    x1= (x & (mask<<size)) >> size
    x2 = x & mask

    x2 = x2 & (x1 ^ x2)
    return (x1 << size) + x2

def C_to_T(x,size,mask):
    #mask=int(('1'*size),2)
    x1= (x & (mask<<size)) >> size
    x2 = x & mask
    xnew=x2 ^ mask
    xnew=x1 & xnew 
    x1=x1 ^ xnew
    return (x1 << size) + x2

def T_to_C(x,size,mask):
    #mask=int(('1'*size),2)
    x1= (x & (mask<<size)) >> size
    x2 = x & mask
    xnew=(x1 | x2) ^ mask
    x1=x1 ^ xnew
    return (x1 << size) + x2
    
def G_to_T(x,size,mask):
    #mask=int(('1'*size),2)
    ##print("{0:b}".format(mask))
    x1= (x & (mask<<size)) >> size
    x2 = x & mask
    xnew=x1 ^ x2
    x1= x1 & xnew
    x2 = x2 & xnew
    return (x1 << size) + x2

def T_to_G(x,size,mask):
    #mask=int(('1'*size),2)
    x1= (x & (mask<<size)) >> size
    x2 = x & mask
    xnew=(x1 | x2) ^ mask
    x1=x1 ^ xnew
    x2=x2 ^ xnew
    return (x1 << size) + x2


def saveEncoding(n_pool,two_way_dir,k):
    print('[NSB] Encoding sequences with {0} processors..'.format(str(n_pool)))
    if os.path.exists(encode_dir):
        shutil.rmtree(encode_dir)
    os.mkdir(encode_dir)
    
    folderpath=two_way_dir
    files = sorted(os.listdir(folderpath))
    
    pool_encode = mp.Pool(n_pool)
    results_encode = [pool_encode.apply_async(saveEncoding2, args=(folderpath,k,file,)) for file in files]#(len(pathnames))]
    for result in results_encode:
        result.get(9999999)
    pool_encode.close()
    pool_encode.join()    

def saveEncoding2(two_way_dir,k,file):
    #if os.path.exists(encode_dir):
    #    shutil.rmtree(encode_dir)
    #os.mkdir(encode_dir)
    #taxafiles = sorted(os.listdir(two_way_dir))
    #for file in taxafiles:
    taxaname=file.split(".")[0]
    set1=[]
    start=time.time()
#    print(file," starting")    
    with open(two_way_dir+"/"+file) as f:
        if ".fasta" in file or ".fa" in file or ".fna" in file:
            for line in f:
                if line[0]!='>':
                    set1+=[i for i in encode4(line.rstrip(),k)]
        elif ".fq" in file or ".fastq" in file:
            fl=1
            Lines=f.readlines()
            for i in range(len(Lines)):
                if i%4==1:
                    #set1+=[j for j in encode4(Lines[i].rstrip(),k)]
                    for j in encode4(Lines[i].rstrip(),k):
                        if j !=[]:
                            set1.append(j)
#    print("Testing", file, len(set1))    
#    if file=="Saccharomyces_eubayanus.fq":
#        print(set1)
    set1 = np.unique(np.array(set1, dtype = np.int64))
    print(file)
    sys.stderr.write('Time taken for encoding {0}.\n'.format(time.time()-start))
    start = time.time()        
    with open(encode_dir+"/"+taxaname+'.pickle', 'wb') as f:
        pickle.dump({tmp_name:set1},f)
    ###end=time.time()

    ###sys.stderr.write('Encoding done for {0}.\n'.format(file))
    # Deleteing the kmer file from kmer_dir
    call(["rm",two_way_dir+"/"+file],stderr=open(os.devnull, 'w'))
    

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
    ###start=time.time()
    set2=np.unique(func(set1, k, mask))
    #set2=np.array([func(kmer, k,mask) for kmer in set1],dtype=np.int64)
    ###sys.stderr.write('Time taken for encoding {0}.\n'.format(time.time()-start))

    ###start = time.time()
    with open(encode_dir_replaced+"/"+taxaname+base_str[i]+base_str[j]+'.pickle', 'wb') as f:
        pickle.dump({tmp_name: set2},f)
    sys.stderr.write('Encoding saved for {0}...\n'.format(file+base_str[i]+base_str[j]))
    ###sys.stderr.write('Time taken for saving pickle {0}.\n'.format(time.time()-start))

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
    ##print(sequences)
    results_sketch = [pool_sketch.apply_async(replaceEncoding, args=(file,func,k,mask,i,j,encode_dir_replaced,)) for file in files]#(len(pathnames))]
    for result in results_sketch:
        result.get(9999999)
    pool_sketch.close()
    pool_sketch.join()
    
def dist_temp_func(cov, eps, k, l, cov_thres):
    if cov == "NA":
        return [1.0, 0]
    #print(k,eps,type(eps))
    p = np.exp(-1*k*eps)
    #print(p)
    copy_thres = int(1.0 * cov / cov_thres) + 1
    lam = 1.0 * cov * (l - k) / l
    if copy_thres == 1 or p == 1:
        return [1 - np.exp(-lam * p), lam * (1 - p)]
    else:
        s = [(lam * p) ** i / np.math.factorial(i) for i in range(copy_thres)]
        return [1 - np.exp(-lam * p) * sum(s), 0]
            
def estimateDistance(J, k,covs,seq_lens,seq_errs,read_lens,fl):
    print("J\n",J)
    if fl==0:
        return 1 - ((2*J/(1+J))**(1/k))
    D = np.zeros((len(J), len(J)))
    cov_thres=5.0
    for i in range(len(J)):
        for j in range(len(J)): 
            if i!=j:
                gl_1 = 2*int(seq_lens[i])
                gl_2 = 2*int(seq_lens[j])
                if gl_1 == "NA" or gl_2 == "NA":
                    gl_1 = 1
                    gl_2 = 1
                cov_1 = float(covs[i])
                cov_2 = float(covs[j])
                print("covs",cov_1,cov_2)
                eps_1 = float(seq_errs[i])
                eps_2 = float(seq_errs[j])
                l_1 = int(read_lens[i])
                l_2 = int(read_lens[j])
                r_1 = dist_temp_func(cov_1, eps_1, k, l_1, cov_thres)
                r_2 = dist_temp_func(cov_2, eps_2, k, l_2, cov_thres)
                wp = r_1[0] * r_2[0] * (gl_1 + gl_2) * 0.5
                zp = sum(r_1) * gl_1 + sum(r_2) * gl_2
                #d = max(0, 1 - (1.0 * zp * j / (wp * (1 + j))) ** (1.0 / k))

                D[i][j] =  1 - (1.0 * zp * J[i][j] / (wp * (1 + J[i][j]))) ** (1.0 / k)
            #D[i][j] = 1 - ((2*J[i][j]/(1+J[i][j]))**(1/k))
    print(D)
    return D

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
                #print(i,j,jac)
                J[i][j] = jac
                J[j][i] = jac
        f.close()
        return J

def clcJaccard(folderpath, files, taxa1, taxa2,filename,freqs_1,freqs_2,subs_1,subs_2,k,fl,l1,l2):
    with open(folderpath+'/'+files[taxa1],'rb') as f:
        set1 = pickle.load(f)[tmp_name]
    with open(folderpath+'/'+files[taxa2],'rb') as f:
        set2 = pickle.load(f)[tmp_name]
    
    start = time.time()
    intersec = np.intersect1d(set1,set2,assume_unique=True)
    intersec_len=len(intersec)
    f_int = open("intersections.txt",'a')
    f_int.write(str(subs_1)+" "+str(subs_2)+" "+str(len(set1))+" "+str(len(set2))+" "+str(intersec_len)+"\n")
    # -- Count random matches ---
    q = 0.25
    '''
    if fl==0:
        w1 = (freqs_1[0]+freqs_1[3])/freqs_1[4]
        w2 = (freqs_2[0]+freqs_2[3])/freqs_2[4]
        L1 = freqs_1[4]
        L2 = freqs_2[4]
    else:
        w1 = (freqs_1[0]+freqs_1[3])/(2*l1)
        w2 = (freqs_2[0]+freqs_2[3])/(2*l2)
        L1 = 2*l1
        L2 = 2*l2
    '''
    w1 = (freqs_1[0]+freqs_1[3])/freqs_1[4]
    w2 = (freqs_2[0]+freqs_2[3])/freqs_2[4]
    if fl==0:
        L1 = freqs_1[4]
        L2 = freqs_2[4]
    else:
        L1 = 2*l1
        L2 = 2*l2
    #w = 0.6
    #print("l1", L1)
    #print("l2", L2)
    expected_rand_match = 0

    
    #L1=2555170
    #L2=4339805
    if subs_1 ==0 and (subs_2==1 or subs_2==2):
        #q = (1/2)*w**2 + (1/2) - w/2
        #print(subs_1, subs_2,q)
        #expected_rand_match = round(freqs_1[4]*freqs_2[4]*q**k)
        for i in range(k+1):
            for j in range(k+1-i):
                expected_rand_match += (1-(1-((1/2)**i)*((w1/2)**j)*((1-w1)/2)**(k-i-j))**(L1))*(1-(1-((1/2)**i)*((w2/2)**j)*((1-w2)/2)**(k-i-j))**(L2))*comb(k,i)*comb(k-i,j)

    #    print("AC AG rand match = ",expected_rand_match)
    else:
    #    print("len = ",L1,L2)
        if subs_1 == 0 and subs_2 == 3:
            #q = (3/2)*w**2 + (1/2) - w   
            for i in range(0,k+1):
                expected_rand_match += (1-(1-w1**(k-i)*((1-w1)/2)**i)**(L1))*(1-(1-w2**(k-i)*((1-w2)/2)**i)**(L2))*comb(k,i)*(2**i)
    #        print("AT rand match = ",expected_rand_match)
        elif subs_1 == 1 and subs_2 == 2:
            #q = (3/2)*w**2 + 1 - 2*w
            for i in range(0,k+1):
                expected_rand_match += (1-(1-(1-w1)**(k-i)*((w1)/2)**i)**(L1))*(1-(1-(1-w2)**(k-i)*((w2)/2)**i)**(L2))*comb(k,i)*(2**i)
    #        print("CG rand match = ",expected_rand_match)
        #expected_rand_match = round(freqs_1[4]*freqs_2[4]*q**k)
    #print("total", intersec_len, "random", expected_rand_match)
    intersec_len -= expected_rand_match
    intersec_len = max(0,intersec_len)
    f_int.write(str(subs_1)+" "+str(subs_2)+" "+str(len(set1))+" "+str(len(set2))+" "+str(intersec_len)+"\n")
    f_int.close()
    #------------------------
    #sys.stderr.write('Time taken for intersection {0}.\n'.format(time.time()-start))
    union_len = len(set1)+len(set2) - intersec_len
    #print(taxa1,taxa2,len(set1),len(set2),intersec_len)
    jaccard=intersec_len/union_len 
    fjac = open(filename,'a')
    fjac.write(str(jaccard)+" "+str(taxa1)+" "+str(taxa2)+"\n")
    fjac.close()

def estimateJaccard(folderpath, n_taxa,token, n_pool,freqs,subs_1,subs_2,k,fl,seq_lens):
    sys.stderr.write("[Tool] Estimating jaccard using {0} processors\n".format(n_pool)) 
    files = sorted(os.listdir(folderpath))
    jaccard_matrix = np.zeros(( n_taxa,n_taxa))
    pool_sketch = mp.Pool(n_pool)
    
    filename = 'savejac'+token+'.txt'
    results_sketch = [pool_sketch.apply_async(clcJaccard, args=(folderpath,files,i,j,filename,freqs[i],freqs[j],subs_1, subs_2,k,fl,float(seq_lens[i]),float(seq_lens[j]),)) for i in range(n_taxa-1) for j in range(i+1,n_taxa)]#(len(pathnames))]
    for result in results_sketch:
        result.get(9999999)
    pool_sketch.close()
    pool_sketch.join()
    
    return readmatrix(filename, n_taxa)

def preprocess(k, n_taxa, n_pool,two_way_dir):
    # Building initial encoded library from k-mers
    saveEncoding(n_pool,two_way_dir,k)
    alphabetSize = len(base_str)
    #Building encoded libraries from k-mers
    for i in range(alphabetSize):
        for j in range(alphabetSize):
            if (i ==1 and (j==0 or j ==3)) or i == 2 or i == 3:
                continue
            if i!=j:
                saveReplacedEncoding(k,i,j,n_pool)

def estimateJC(dist):
    return -(3/4)*np.log(1-(4/3)*dist)
    
def distEstimatorMaster(k, n_taxa, n_pool,freqs,two_way_dir,covs,seq_lens,seq_errs,read_lens,fl):
    global base_str, encode_dir, encode_dir_replaced
    
    #......... PREPROCESS .........
    preprocess(k,n_taxa, n_pool,two_way_dir)
    #..............................
    base_subs = 13
    #......... DIST ...............
    dist_matrices = np.zeros((base_subs,n_taxa,n_taxa))
    
    J1 = estimateJaccard(encode_dir,n_taxa,"",n_pool,freqs,0,0,k,fl,seq_lens)
    dist_matrices[0] = estimateDistance(J1,k,covs,seq_lens,seq_errs,read_lens,fl)
    fsave = open('dist.txt','w')
    #sys.stderr.write("dist = {0}".format(dist_matrices[0][0][1]))
    #file_csv.write(str(J1[0][1])+",")
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
                J2 = estimateJaccard(encode_dir_replaced,n_taxa, token, n_pool,freqs,i,j,k,fl,seq_lens)
                #file_csv.write(str(J2[0][1])+",")
                dist_matrices[count] = estimateDistance(J2,k,covs,seq_lens,seq_errs,read_lens,fl)
                #print(i,j,J1[0][1], J2[0][1],dist_matrices[count][0][1])
                #sys.stderr.write("dist = {0}".format(dist_matrices[count][0][1]))
                fsave.write(str(dist_matrices[count]))
                fsave.write("\n")
                count += 1
    #print("jaccards done")
    # --   --                             #  1  2  3  4  5  6  7  8  9  10  11  12
    dist_matrices[4] = dist_matrices[1]   # AC,AG,AT,CA,CG,CT,GA,GC,GT, TA, TC, TG
    dist_matrices[9] = dist_matrices[1]
    dist_matrices[12] = dist_matrices[1]
    dist_matrices[6] = dist_matrices[2]
    dist_matrices[7] = dist_matrices[2]
    dist_matrices[11] = dist_matrices[2]
    dist_matrices[10] = dist_matrices[3]
    dist_matrices[8] = dist_matrices[5]
    
    d_skmer_jf=estimateJC(dist_matrices[0])
    dist_matrices[0] = (2*dist_matrices[1] + 2* dist_matrices[2] + dist_matrices[3] + dist_matrices[5])/5
    
    #file_csv.write(str(estimateJC(dist_matrices[0][0][1]))+",")
    shutil.rmtree(encode_dir)
    shutil.rmtree(encode_dir_replaced)
    return dist_matrices,d_skmer_jf
