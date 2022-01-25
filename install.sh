if conda info --envs | grep "nsb" > /dev/null; then
    echo "conda environment nsb exists"
else
    conda create -y -c bioconda -c conda-forge --name nsb python=3.9 numpy scipy pandas skmer 
fi
conda activate nsb
