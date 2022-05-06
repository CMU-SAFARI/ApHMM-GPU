# ApHMM-GPU: A Profile Hidden Markov Model Acceleration Framework for Genome Analysis (GPU)

Here, we use the CUDA library for implementing and executing the Baum-Welch algorithm in GPUs. We integrate the Baum-Welch algorithm in an error correction tool, Apollo.

Apollo is an assembly polishing algorithm that corrects the errors in an assembly. It can take multiple set of reads in a single run and polish the assemblies of genomes of any size.

## Installing ApHMM-GPU

* Make sure you have a compiler that has support for C++14.
* Download the code from this GitHub repository.

```bash
git clone https://github.com/CMU-SAFARI/ApHMM-GPU.git aphmm-gpu
```
*  Change directory to `./aphmm-gpu` and run the Makefile. If everything goes well, you will have a binary called `aphmm-gpu` inside the `bin` folder. The gencode arguments are currently set for either NVIDIA Titan V (SM=70) or NVIDIA A100 (SM=80) GPUs in Makefile. You can change the SM value based on the GPU architecture you use from the following line in Makefile:

```bash
#Set to 80 for A100, 70 for Titan V
SM=70
```

To change the directory and run the Makefile:

```bash
cd ./aphmm-gpu
make
cd ./bin
```
Now you can copy this binary wherever you want (preferably under a directory that is included in your `$PATH`). Assuming that you are in the directory that the binary is located, you may run the command below to display the help message.

```bash
./aphmm-gpu -h
```

## Assembly polishing
Polishing using a single set of reads (i.e., non-hybrid):

Assume that you have 1) an assembly `assembly.fasta`, 2) a set of reads `reads.fasta`, 3) the alignment file `alignment.bam` that contains the alignment of the reads to the assembly, 4) and you would like to store polished assembly as `polished.fasta`. The command below uses `30` CPU threads and processes `256` observations in parallel in GPUs while polishing the assembly:

```bash
./aphmm-gpu -a assembly.fasta -r reads.fasta -m alignment.bam -t 30 -g 256 -o polished.fasta
```
Resulting fasta file `polished.fasta` will be the final output of Apollo.

## Supported and Required Input Files
### Alignment File

* Apollo supports alignment files in BAM format. If you have a SAM file you can easily convert your `input.sam` to `input.bam` using the following command:

```bash
samtools view -hb input.sam > input.bam
```
* Apollo requires the input BAM file to be coordinate sorted. You can sort your `input.bam` file using the following command:

```bash
samtools view -h -F4 input.bam | samtools sort -m 16G -l0 > input_sorted.bam
```

### Set of Reads

* Apollo supports the reads set in FASTA format. For each read (i.e., sequence), the number of characters per line has to be the same, except for the last line. For example, a sequence of length 1000 can either be represented in a single line with 1000 characters or can be split into multiple lines where each line include the equal number of characters. Only exception here is the last line, which can have any number of characters but no more than the characters that the prior lines have. An illustration of a sequence with a length of 10 would be:

>\>read1  
>TAT  
>TAT  
>ATT  
>A

or in a single line:

>\>read1  
TATTATATTA

The restriction on the number of characters per line is required as Apollo constructs the index file (i.e., FAI file) for the input read set. Further information about indexing and the requirements can be found at: https://seqan.readthedocs.io/en/master/Tutorial/InputOutput/IndexedFastaIO.html

* If there are *too* long reads in a input read set, we recommend dividing these reads into smaller chunks to reduce the memory requirements. Apollo supports chunking during run time. One can simply use command to divide the reads into chunks of size 650 (maximum) while polishing:

```bash
./aphmm-gpu -a assembly.fasta -r reads1.fasta -r reads2.fasta -m alignment1.bam -m alignment2.bam -t 30 -g 256 -o polished.fasta -c 650
```

## Example run

You may use the following test run. As aphmm-gpu runs on the Apollo application, it will output the time it took in relavent places that we use in the ApHMM paper.

```bash
#create a test folder
cd test
#download a read set that is publicly available by PacBio and only fetch small number of read set as this is a sanity check
tar -xzf SRR5413248_subreads.fasta.tar.gz
aphmm-gpu -a Ecoli.O157_assembly.contigs.fasta -r SRR5413248_subreads.fasta -m Ecoli.O157_assembly.contigs_tig00000864_1k.bam -o polished.fasta -c 650 -g 256
```

An example outout is as follows:

```bash
...
Number of observations processed in parallel: 256
forward: Number of forward calculations in parallel:256 	Total time (everything)=48s, 	Data Preparation=1s, 	CPU+GPU calculation=46s, 	GPU Calculation (all forwards)=0.020122s, 	GPU Calculation (single forward)=0.000079s, 	Avg. Kernel Time (single timestamp)=0.000031s, 	NumOf Kernel Calls=649
backward: Number of backward calculations in parallel:256 	Total time (everything)=44s, 	Data Preparation=0s, 	CPU+GPU calculation=44s, 	GPU Calculation (all backwards)=0.017541s, 	GPU Calculation (single backward)=0.000069s, 	Avg. Kernel Time (single timestamp)=0.000027s, 	NumOf Kernel Calls=649
forward: Number of forward calculations in parallel:256 	Total time (everything)=46s, 	Data Preparation=0s, 	CPU+GPU calculation=46s, 	GPU Calculation (all forwards)=0.020625s, 	GPU Calculation (single forward)=0.000081s, 	Avg. Kernel Time (single timestamp)=0.000032s, 	NumOf Kernel Calls=649
backward: Number of backward calculations in parallel:256 	Total time (everything)=43s, 	Data Preparation=0s, 	CPU+GPU calculation=43s, 	GPU Calculation (all backwards)=0.018279s, 	GPU Calculation (single backward)=0.000071s, 	Avg. Kernel Time (single timestamp)=0.000028s, 	NumOf Kernel Calls=649
forward: Number of forward calculations in parallel:256 	Total time (everything)=46s, 	Data Preparation=0s, 	CPU+GPU calculation=46s, 	GPU Calculation (all forwards)=0.020499s, 	GPU Calculation (single forward)=0.000080s, 	Avg. Kernel Time (single timestamp)=0.000032s, 	NumOf Kernel Calls=649
backward: Number of backward calculations in parallel:256 	Total time (everything)=41s, 	Data Preparation=0s, 	CPU+GPU calculation=41s, 	GPU Calculation (all backwards)=0.017356s, 	GPU Calculation (single backward)=0.000068s, 	Avg. Kernel Time (single timestamp)=0.000027s, 	NumOf Kernel Calls=649
avgForwardExec time (per read of a contig): 47512.980915 ms
avgBackwardExec time (per read of a contig): 43473.613462 ms
avgOverallPosteriorCalc time (per read of a contig): 735.908864 ms
Total processed read count for a contig: 853
Total processed read count so far: 853
avgcalcFBTime time (avg for a contig): 851456.011000 ms (cpu time) 851120.915286 ms (real time)
maximizeEM for a contig:  totalTime=8119ms, gpuPreparationTime=6380ms, gpuKernelTime=16.464001ms
avgmaximizeEMTime time (avg for a contig): 8325.599000 ms (cpu time) 8323.979719 ms (real time)
avgcalcFBTime time (avg of all contigs): 851456.011000 ms (cpu time)
avgmaximizeEMTime time (avg of all contigs): 8325.599000 ms (cpu time)

Results have been written under polished.fasta
Overall polishing time: 2760473.803000 ms (cpu time) 2759294.783717 ms (real time)
```

From the above example output, we are mainly interested in the following four values: `GPU Calculation (single forward)`, `GPU Calculation (single backward)`, `avgOverallPosteriorCalc`, and `avgmaximizeEMTime time (avg of all contigs)`. Apart from `avgOverallPosteriorCalc`, all of the values show the time it took for GPU to calcualte the relavent parts: forward calculation for a single observation, backward calculation for a single observation, and the expectation-maximization (EM) step. The value for `avgOverallPosteriorCalc` shows time for accumulating values of the previously calculated Forward/Backward values that are used in the EM step.

### Publication and citing ApHMM

If you would like to cite ApHMM, please cite the following publication:

> Can Firtina, Kamlesh Pillai, Gurpreet S. Kalsi, Bharathwaj Suresh, Damla Senol Cali, 
> Jeremie S. Kim,Taha Shahroodi, Meryem Banu Cavlak, Joel Lindegger, Mohammed Alser, 
> Juan Gomez Luna, Sreenivas Subramoney, and Onur Mutlu,
> "ApHMM: A Profile Hidden Markov Model Acceleration Framework for Genome Analysis",
> *arXiv*, May 2022.
