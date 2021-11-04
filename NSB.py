#! /usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
from scipy.optimize import newton
import argparse
import os
import shutil
import fnmatch
import sys
import errno
import pandas as pd
from subprocess import call, check_output, STDOUT
import multiprocessing as mp
from os import listdir
from os.path import isfile, join
import jaccard_estimator as jac
import pandas as pd

__version__ = 'Tool 1.0.0'

csv_path = "dist_params.csv"

if os.path.isfile(csv_path):
    file_csv = open(csv_path,"a")
else:
    file_csv = open(csv_path,"w")


def two_way_genome(savedir, path, file):
    file1 = open(path+'/'+file, 'r')
    Lines = file1.readlines()
    file2 = open(savedir+'/'+file,'w')     						
    newseq = "" 
    oldseq = ""
    l1 = []
    l2 = []
    s1 = ['A','T','C','G','N']
    s2 = ['T','A','G','C','N']
    for line in Lines:
        oldseq += line
        if line.startswith(">") != True:
            temp = ""
            for i in line:
                if i in s1:
                    temp += s2[s1.index(i)]		# Replacing each character with it's RC
            
            temp = temp[::-1]                   # Reversing the RC string
            newseq += temp+"\n"
        else :
            newseq += line
    newseq = newseq.rstrip()
    oldseq = oldseq.rstrip()              

    newlist  = newseq.split('\n')
    for i in range(int(len(newlist)/2)):
        newlist[2*i],newlist[2*i+1] = newlist[2*i+1],newlist[2*i]   # Swaping each ith line with (i+1)th line in the RC.
    newlist = newlist[::-1]                                         # Reversing the swapped string. 

    newseq2 = oldseq+'\n'+'\n'.join(newlist)
    file2.write(newseq2)
    file1.close()
    file2.close()

def  two_way_genome_builder(path, dir_2way_path, args):	
    files = sorted(os.listdir(path))
    savedir = dir_2way_path						# Making 2way directory: 'path_2way'
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    pool_two_way = mp.Pool(args.p)
    result_two_way = [pool_two_way.apply_async(two_way_genome, args=(savedir, path,file,)) for file in files]
    for result in result_two_way:
    	result.get(9999999)
    pool_two_way.close()
    pool_two_way.join()
        
    return savedir

def genome_stats(input_dir, entry):
    i=0
    A_count=0
    C_count=0
    G_count=0
    T_count=0
    N_count=0
    space_count=0
    lent=0
    with open(input_dir+"/"+entry) as f:
        for line in f:
            i=i+1
            if line[0]!='>':
                #lent=lent+len(line)
                A_count=A_count+line.count("A")
                C_count=C_count+line.count("C")
                G_count=G_count+line.count("G")
                T_count=T_count+line.count("T")
                N_count=N_count+line.count("N")
                #space_count=space_count+line.count("\n")
                #s=set(line)
                #print(s)

    lent=A_count+C_count+T_count+G_count+N_count
    out_string=str(A_count)+" "+str(C_count)+" "+str(G_count)+" "+str(T_count)+" "+str(lent)+"\n"
    output = open(os.path.join(os.getcwd(),"stats.txt"),"a")
    output.write(entry+"\n")
    output.write(out_string)
    output.close()


def build_stats(input_dir, args):
    
    entries = sorted(os.listdir(input_dir+"/"))

    pool_stat = mp.Pool(args.p)
    result_stat = [pool_stat.apply_async(genome_stats, args=(input_dir, entry,)) for entry in entries]
    for result in result_stat:
    	result.get(9999999)
    pool_stat.close()
    pool_stat.join()
    

def clcFreqs():
    file1 = open(os.path.join(os.getcwd(),'stats.txt'),'r') 
    Lines = file1.readlines()
    freq = {}
    names = []
    for i in range(int(len(Lines)/2)):
        freq[Lines[2*i]] = Lines[2*i+1].split()
        names.append(Lines[2*i])
    freq_items = freq.items() #Get key-value pairs.
    freq_sorted = sorted(freq_items)
    names = sorted(names)
    freqs = []
    for item in freq_sorted:
        freqs.append(item[1])
    return np.array(freqs,dtype=float),names
	
#################################################
# TK4 takes a 4*4 matrix, m, indexes are A,C,G,T
# R = (AC+GT)/2
# P = (AG+CT)/2
# 2Q1 = AT/2
# 2Q2 = CG/2
# s1+s3 = w -(P+R)/2
#################################################
def TK4(m,w,Hamming):   
    #Hamming = 1-(2*jaccard/(1+jaccard))**(1/31)  
    if np.all(m==0):
        return 0
    sums = np.sum(m,axis = 1)
    #w = (freq[0]+freq[3])/freq[4]
    R = (m[0][1] + m[1][0] + m[2][3] + m[3][2])/4 	#R = (m[0][1]+m[2][3])/2
    P = (m[0][2] + m[2][0] + m[1][3] + m[3][1])/4	#P = (m[0][2]+m[1][3])/2
    TwoQ1 = (m[0][3] + m[3][0])/2				#TwoQ1 = m[0][3]/2
    TwoQ2 = (m[1][2] + m[2][1])/2					#TwoQ2 = m[1][2]/2
    s1_s3 = w - (P+R)/2 - TwoQ1   				#s1_s3 = f[0] + f[3] - sums[0] - sums[3] + m[0][0] + m[3][3] #
    s2_s4 = 1 - w - (P+R)/2 - TwoQ2					#s2_s4 = f[2] + f[3] - sums[1] - sums[2] + m[2][2] + m[1][1] #
    
    #print("w", w)
    #print("P", P)
    #print("Q1", TwoQ1)
    #print("Q2", TwoQ2)
    #print("R", R)
    #print("S1",s1_s3)
    #print("S2",s2_s4)
    ln1 = -1/4*np.log(((s1_s3-TwoQ1)*(s2_s4-TwoQ2)-((P-R)/2)**2)/(w*(1-w)))
    ln2 = -1/4*np.log((1-((P+R)/(2*w*(1-w))))**(8*w*(1-w)-1))
    file_csv.write(str(ln1+ln2)+","+str(w)+","+str(R)+","+str(P)+","+str(TwoQ1)+","+str(TwoQ2)+"\n")
    return ln1+ln2

def JC(dist):
    return -(3/4)*np.log(1-(4/3)*dist)

def replacer(replaced_dir,two_way_dir,x,y): 
        if os.path.exists(replaced_dir):
            shutil.rmtree(replaced_dir)
        os.mkdir(replaced_dir)

        samples = sorted(os.listdir(two_way_dir))

        for sample in samples:
            sample_1 = open(two_way_dir+'/'+sample, 'r')
            Lines = sample_1.readlines()
            sample_2 = open(replaced_dir+'/'+sample,'w') 
            newseq="" 
            for line in Lines:
                newseq+=line.replace(x, y)
            sample_2.write(newseq)

            sample_1.close()
            sample_2.close()
            
# For supporting the functionality of skmer and mash. 
class Mash_Skmer:
    def __init__(self, args,  file_names, freqs):
        self.k = args.k
        self.s = args.s
        self.file_names = file_names
        self.n_taxa = len(file_names)

        self.two_way_dir = os.path.join(os.getcwd(), 'ref_dir_2way')
        self.replaced_dir = os.path.join(os.getcwd(), 'ref_dir_replaced')
        self.mash_dir  = os.path.join(os.getcwd(), 'library/mash')
        self.n_pool = args.p
        
        self.freqs  = freqs
        
        if not os.path.exists(self.mash_dir):   
            os.mkdir(self.mash_dir)        
            
    def sketch(self,sequence):
        sample = os.path.basename(sequence).rsplit('.f', 1)[0]
        msh = os.path.join(self.mash_dir, sample)
        
        call(["mash", "sketch", "-k", str(self.k),"-n", "-s", str(self.s), "-o", msh, sequence], stderr=open(os.devnull, 'w'))

    def mash_dist(self,sample_1, sample_2):
        if sample_1 == sample_2:
            return sample_1, sample_2, 0.0
        sample_1 = os.path.basename(sample_1).rsplit('.f', 1)[0]
        sample_2 = os.path.basename(sample_2).rsplit('.f', 1)[0]

        msh_1 = os.path.join(self.mash_dir, sample_1+'.msh')
        msh_2 = os.path.join(self.mash_dir, sample_2+'.msh')
        
        dist_stderr = check_output(["mash", "dist", msh_1, msh_2], stderr=STDOUT, universal_newlines=True)
        j = float(dist_stderr.split()[-1].split("/")[0]) / float(dist_stderr.split()[-1].split("/")[1])
        dist = 1 - ((2*j/(1+j))**(1/self.k))
        return dist
        
    def mash_runner(self):
        base_subs = ['','AC','AG','AT','CG','CT','GT','CA','GA','TA','GC','TC','TG']
        d_ = np.zeros((len(base_subs),self.n_taxa,self.n_taxa))
        count = 0
        for bases in base_subs:
            if bases == '':
                dir = self.two_way_dir
            else:
                dir = self.replaced_dir
                replacer(dir, self.two_way_dir,bases[0],bases[1])
            sequences = [os.path.join(dir, f) for f in self.file_names]
            for seq in sequences:
                self.sketch(seq)
            for i in range(self.n_taxa-1):
                for j in range(i+1, self.n_taxa):
                    dist = self.mash_dist(sequences[i], sequences[j])
                    d_[count][i][j] = d_[count][j][i] = dist
            count += 1

        d = 2*(d_[0] - d_[1:])
        for i in [2,4,7,9]:
            d[i] /= 2
        
        phylo_dist_mat = np.ndarray((self.n_taxa,self.n_taxa),dtype=float)
        for i in range(self.n_taxa):
            phylo_dist_mat[i][i] = 0.0
        for i in range(self.n_taxa-1):
            for j in range(i+1,self.n_taxa):
                f = self.freqs[i] if self.freqs[i][4] > self.freqs[j][4] else self.freqs[j] #(freq[i]+freq[j])/2
                f /= f[4]
                print(f)
                m = [[f[0] - (d[0][i][j]+d[1][i][j]+d[2][i][j]),d[0][i][j],d[1][i][j],d[2][i][j]],
                    [d[6][i][j],f[1] - (d[6][i][j]+d[3][i][j]+d[4][i][j]),d[3][i][j],d[4][i][j]],
                    [d[7][i][j],d[9][i][j],f[2] - (d[7][i][j]+d[9][i][j]+d[5][i][j]),d[5][i][j]],
                    [d[8][i][j],d[10][i][j],d[11][i][j],f[3] - (d[8][i][j]+d[10][i][j]+d[11][i][j])]]
                f1 = self.freqs[i]/self.freqs[i][4]
                f2 = self.freqs[j]/self.freqs[j][4]
                w = (f1[0]+f1[3]+f2[0]+f2[3])/2
                phylo_dist_mat[i][j] = phylo_dist_mat[j][i] = TK4(m, w, d_[0][i][j])
        print(phylo_dist_mat)
        
        #shutil.rmtree(self.mash_dir)
        return phylo_dist_mat

class Jellyfish:
    def __init__(self, args, file_names, freqs):
        self.k = args.k
        self.s = args.s
        self.n_pool = args.p
        self.file_names =  file_names
        self.freqs = freqs
        self.method = args.m
        self.n_taxa = len(file_names)
        self.two_way_dir = os.path.join(os.getcwd(), 'ref_dir_2way')
        self.kmer_dir = os.path.join(os.getcwd(), 'kmer_dir')
        self.jf_dir  = os.path.join(os.getcwd(), 'library')
        
        if os.path.exists(self.jf_dir):  
            shutil.rmtree(self.jf_dir) 
        if os.path.exists(self.kmer_dir): 
            shutil.rmtree(self.kmer_dir)  
        os.mkdir(self.jf_dir) 
        os.mkdir(self.kmer_dir)
	      
        
      
    def blocks(self,files, size=65536):
        while True:
            b = files.read(size)
            if not b: break
            yield b
    

       
    def sketch(self, pathname, outputdir,taxaname):
        sys.stderr.write('Sketching {0}\n'.format(pathname))
        jffile = "library/"+taxaname+'.jf'
        call(["jellyfish", "count", "-m", str(self.k), "-s", str(self.s),pathname, "-o", jffile], stderr=open(os.devnull, 'w'))
        print(pathname+" - jellyfish count done")
        call(["jellyfish", "dump", "-c", jffile,"-o",outputdir], stderr=open(os.devnull, 'w'))
        print(pathname+" - Jellyfish dump done")
        os.remove(jffile)
        return 1
    
        
    
    def createKmerStats(self):
        kmerstatfile="kmer_stats.txt"
        stat_str=""
        kmer_dir=self.kmer_dir
        dir_name = self.two_way_dir        
        taxafiles = sorted(os.listdir(dir_name))
        print("here_:", taxafiles)
        outputfilenames = []
        pathnames = []
        
        for file in taxafiles:
            outputfilename='.'.join(file.split('.')[:-1])+".txt"
            pathname=dir_name+"/"+file
            outputdir=kmer_dir+"/"+outputfilename
            outputfilenames.append(outputdir)
            pathnames.append(pathname)
            
        sys.stderr.write('[Tool] Sketching sequences using {0} processors...\n'.format(self.n_pool))
        
        pool_sketch = mp.Pool(self.n_pool)
       
        results_sketch = [pool_sketch.apply_async(self.sketch, args=(pathnames[i],outputfilenames[i],'.'.join(taxafiles[i].split('.')[:-1]))) for i in range(len(taxafiles))]#(len(pathnames))]
        for result in results_sketch:
            result.get(9999999)
        pool_sketch.close()
        pool_sketch.join()
        
        return kmer_dir
    
    def jellyfish_runner(self):
        self.createKmerStats()
        
        d_,d_skmer_jf = jac.distEstimatorMaster(self.k, self.n_taxa, self.n_pool,self.freqs)
        
        d = 2*(d_[0] - d_[1:])
        for i in [2,4,7,9]:
            d[i] /= 2
    
        phylo_dist_mat = np.ndarray((self.n_taxa,self.n_taxa),dtype=float)
        jc_dist_mat = np.ndarray((self.n_taxa,self.n_taxa),dtype=float)
        for i in range(self.n_taxa):
            phylo_dist_mat[i][i] = 0.0
        for i in range(self.n_taxa-1):
            for j in range(i+1,self.n_taxa):
                f = self.freqs[i] if self.freqs[i][4] > self.freqs[j][4] else self.freqs[j] #(freq[i]+freq[j])/2
                f /= f[4]
                m = [[f[0] - (d[0][i][j]+d[1][i][j]+d[2][i][j]),d[0][i][j],d[1][i][j],d[2][i][j]],
                    [d[3][i][j],f[1] - (d[3][i][j]+d[4][i][j]+d[5][i][j]),d[4][i][j],d[5][i][j]],
                    [d[6][i][j],d[7][i][j],f[2] - (d[6][i][j]+d[7][i][j]+d[8][i][j]),d[8][i][j]],
                    [d[9][i][j],d[10][i][j],d[11][i][j],f[3] - (d[9][i][j]+d[10][i][j]+d[11][i][j])]]
                f1=self.freqs[i]/self.freqs[i][4]
                f2=self.freqs[j]/self.freqs[j][4]
                w=(f1[0]+f1[3]+f2[0]+f2[3])/2
                phylo_dist_mat[i][j] = phylo_dist_mat[j][i] = TK4(m, w, d_[0][i][j])
                jc_dist_mat[i][j] = jc_dist_mat[j][i] = JC(d_[0][i][j])
        print(phylo_dist_mat)
        
        #shutil.rmtree(self.jf_dir)
        return phylo_dist_mat,jc_dist_mat,d_skmer_jf
        
def file_writer(matrix,taxa_names,filename):
    fw= open(filename,'w')
    fw.write(str(len(matrix))+"\n")
    for i in range(len(matrix)):
        fw.write(taxa_names[i]+" ")
        for j in range(len(matrix[i])):
            fw.write(str(matrix[i][j])+" ")
        fw.write("\n")
    fw.close()
    
def reference(args):
    # Making a list of sample names
    formats = ['.fq', '.fastq', '.fa', '.fna', '.fasta']
    files_names = [f for f in sorted(os.listdir(args.input_dir))
                   if True in (fnmatch.fnmatch(f, '*' + form) for form in formats)]

    print('[Tool] Building 2way genome with {0} processors..'.format(str(args.p)))
    two_way_genome_builder(args.input_dir, os.path.join(os.getcwd(), 'ref_dir_2way'), args)
    print('[Tool] Building stats with {0} processors..'.format(str(args.p)))
    build_stats(os.path.join(os.getcwd(), 'ref_dir_2way'), args)
    
    freqs, names = clcFreqs()
    if args.m == 1 or args.m == 2:  # Run Jellyfish
        print('[Tool] Sketching sequences (Jellyfish) with {0} processors..'.format(str(args.p)))
        jellyfish1 = Jellyfish(args, files_names, freqs)
        dist_matrix,jc_matrix,skmer_jf_matrix = jellyfish1.jellyfish_runner()
        file_writer(jc_matrix,files_names,"ref-dist-mat-jc-"+args.input_dir.split('/')[0]+".txt")
        file_writer(skmer_jf_matrix,files_names,"ref-dist-mat-skmer-jf-"+args.input_dir.split('/')[0]+".txt")
    if args.m == 3:  # Run Mash 
        print('[Tool] Sketching sequences (Mash) with {0} processors..'.format(str(args.p)))
        mash = Mash_Skmer(args, files_names, freqs)
        dist_matrix = mash.mash_runner()
        
    file_writer(dist_matrix,files_names,"ref-dist-mat-nsb-"+args.input_dir.split('/')[0]+".txt")
    		

def main():
    # Input arguments parser
    parser = argparse.ArgumentParser(description='{0} - Estimating gonomic distances between '.format(__version__) +
                                                 'genome-skims',
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # parser.add_argument('-v', '--version', action='store_true', help='print the current version')
    parser.add_argument('--debug', action='store_true', help='Print the traceback when an exception is raised')
    subparsers = parser.add_subparsers(title='commands',
                                       description='reference   Process a library of reference genome-skims or assemblies\n'
                                                   'distance    Compute pairwise distances using Tk4\n'
                                                   'library',
                                       help='Run skmer {commands} [-h] for additional help\n\n\n'
                                           '1 - Jellyfish1 (takes less space, more time),\n 2 - Jellyfish2 (takes more space, less time), \n 3 - mash',
                                       dest='{commands}')

    # To make sure that subcommand is required in python >= 3.3
    python_version = sys.version_info
    if (python_version[0] * 10 + python_version[1]) >= 33:
        subparsers.required = True

    # Reference command subparser
    parser_ref = subparsers.add_parser('reference',
                                       description='Process a library of reference genome-skims or assemblies')
    parser_ref.add_argument('input_dir',
                            help='Directory of input genome-skims or assemblies (dir of .fastq/.fq/.fa/.fna/.fasta files)')
    parser_ref.add_argument('-l', default=os.path.join(os.getcwd(), 'library'),
                            help='Directory of output (reference) library. Default: working_directory/library')
    parser_ref.add_argument('-o', default='ref-dist-mat',
                            help='Output (distances) prefix. Default: ref-dist-mat')
    parser_ref.add_argument('-k', type=int, choices=list(range(1, 41)), default=31, help='K-mer length [1-40]. ' +
                                                                                         'Default: 31', metavar='K')
    parser_ref.add_argument('-s', type=int, default=10 ** 7, help='Sketch size. Default: 100000')
    parser_ref.add_argument('-p', type=int, choices=list(range(1, mp.cpu_count() + 1)), default=mp.cpu_count(),
                            help='Max number of processors to use [1-{0}]. '.format(mp.cpu_count()) +
                                 'Default for this machine: {0}'.format(mp.cpu_count()), metavar='P')
    parser_ref.add_argument('-m',type=int,choices=list(range(1,4)), default=1, 
                            help='1 - Jellyfish1 (takes less space, more time),\n 2 - Jellyfish2 (takes more space, less time), \n 3 - mash',metavar='M')
    parser_ref.set_defaults(func=reference)


    args = parser.parse_args()

    # Handling traceback on exceptions
    def exception_handler(exception_type, exception, traceback, debug_hook=sys.excepthook):
        if args.debug:
            debug_hook(exception_type, exception, traceback)
        else:
            print("{0}: {1}".format(exception_type.__name__, exception))

    sys.excepthook = exception_handler

    args.func(args)


if __name__ == "__main__":
    main()

