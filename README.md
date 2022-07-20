# ApHMM: Accelerating Profile Hidden Markov Models for Fast and Energy-Efficient Genome Analysis

ApHMM is the *first* flexible hardware-software co-designed acceleration framework that can significantly reduce the computational and energy overheads of the Baum-Welch algorithm for profile Hidden Markov Models (pHMMs). ApHMM is built on four key mechanisms. First, ApHMM is highly flexible that can use different pHMM designs with the ability to change certain parameter choices to enable the adoption of ApHMM for many pHMM-based applications. This enables 1) additional support for pHMM-based error correction that traditional pHMM design cannot efficiently and accurately support. Second, ApHMM exploits the spatial locality that pHMMs provide with the Baum-Welch algorithm by efficiently utilizing on-chip memories with memoizing techniques. Third, ApHMM efficiently eliminates negligible computations with a hardware-based filter design. Fourth, ApHMM avoids redundant floating-point operations by 1) providing a mechanism for efficiently reusing the most common products of multiplications in lookup tables (LUTs) and 2) identifying pipelining and broadcasting opportunities where certain computations are moved between multiple steps in the Baum-Welch algorithm without extra storage or computational overheads. Among these mechanisms, the fourth mechanism includes our software optimizations, while on-chip memory and hardware-based filter require a special and efficient hardware design.

In this repository, we provide *ApHMM-GPU*, our GPU implementation for ApHMM that includes the software optimizations. ApHMM-GPU the *first* GPU implementation of the Baum-Welch algorithm for pHMMs.

We integrate the ApHMM-GPU implementation in Apollo, an assembly polishing tool for correcting errors in *de novo* assemblies. Apollo builds pHMMs for each assembly and uses the Baum-Welch algorithm to update the parameters based on the alignment of sequences to each assembly. Apollo uses the Viterbi algorithm to decode the consensus sequence from trained pHMM, which provides the corrected assembly sequence. The original implementation of Apollo can be found at: https://github.com/CMU-SAFARI/Apollo.

## Citation

Please cite the following preliminary version of our paper if you find this repository useful:

> Can Firtina, Kamlesh Pillai, Gurpreet S. Kalsi, Bharathwaj Suresh, Damla Senol Cali, 
> Jeremie S. Kim, Taha Shahroodi, Meryem Banu Cavlak, Joel Lindegger, Mohammed Alser, 
> Juan Gómez-Luna, Sreenivas Subramoney, and Onur Mutlu,
> "ApHMM: Accelerating Profile Hidden Markov Models for Fast and Energy-Efficient Genome Analysis",
> *arXiv*, July 2022.

BibTeX entry for citation:

```
@article{firtina_aphmm_2022,
  title = {{ApHMM}: {Accelerating} {Profile} {Hidden} {Markov} {Models} for {Fast} and {Energy-Efficient} {Genome} {Analysis}},
  journal = {arXiv},
  author = {Firtina, Can and Pillai, Kamlesh and Kalsi, Gurpreet S. and Suresh, Bharathwaj and Senol Cali, Damla and Kim, Jeremie S. and Shahroodi, Taha and Cavlak, Meryem Banu and Lindegger, Joel and Alser, Mohammed and G{\'o}mez-Luna, Juan and Subramoney, Sreenivas and Mutlu, Onur},
  year = {2022},
  month = {July},
}
```

The link to the Apollo paper can be found at [this link](https://academic.oup.com/bioinformatics/article/36/12/3669/5804978?login=true). BibTeX entry for Apollo:

```
@article{firtina_apollo_2020,
  title = {Apollo: a sequencing-technology-independent, scalable and accurate assembly polishing algorithm},
  volume = {36},
  issn = {1367-4803},
  url = {https://doi.org/10.1093/bioinformatics/btaa179},
  doi = {10.1093/bioinformatics/btaa179},
  number = {12},
  urldate = {2022-04-13},
  journal = {Bioinformatics},
  author = {Firtina, Can and Kim, Jeremie S. and Alser, Mohammed and Senol Cali, Damla and Cicek, A Ercument and Alkan, Can and Mutlu, Onur},
  month = {June},
  year = {2020},
  pages = {3669--3679},
}
```

## Installing ApHMM-GPU
### Repository Structure

We point out important files and directories, which we will mention in the later sections in this `README.md`.

```
.
+-- README.md
+-- Makefile
+-- src/
|   +-- HMMTrainer.h
|   +-- HMMTrainer.cpp
|   +-- HMMTrainer.cuh
|   +-- HMMTrainer.cu
|   +-- *.h
|   +-- *.cpp
+-- test/
|   +-- Ecoli.O157_assembly.contigs_tig00000864_1k.bam
|   +-- Ecoli.O157_assembly.contigs.fasta
|   +-- SRR5413248_subreads_1k.fasta
+-- utils/
|   +-- bzip2-1.0.6.tar.gz
|   +-- seqan.tar.gz
|   +-- zlib-1.2.11.tar.gz

```

### Prerequisites

* ApHMM-GPU requires the [CUDA Library](https://developer.nvidia.com/cuda-downloads?target_os=Linux) (>= v11.6).
* Apollo requires the [SeqAn Library](https://seqan.readthedocs.io/en/master/) (v2.4). The SeqAn library is already included in `utils/seqan.tar.gz`.
* The SeqAn library requires a compiler with support for [C++14](https://en.cppreference.com/w/cpp/14).
* The SeqAn library requires the [bzip2](https://www.sourceware.org/bzip2/) and [zlib](https://zlib.net) libraries. The source code of these libraries are included in `utils/bzip2-1.0.6.tar.gz` and `utils/zlib-1.2.11.tar.gz`, respectively.
* We provide `Makefile` to compile the source code and generate the binary for using ApHMM-GPU. To compile using `Makefile`, GNU [make](https://www.gnu.org/software/make/manual/make.html) is required.


### Cloning the repository

To clone this repository from remote (this GitHub page) to your local machine under a directory called `aphmm-gpu`, run the following command in your local machine.

```bash
git clone https://github.com/CMU-SAFARI/ApHMM-GPU.git aphmm-gpu
```

### Compilation

To compile the source code, change directory to `aphmm-gpu` and run the Makefile. If you meet the prerequisites, the `aphmm-gpu` binary will be created under `aphmm-gpu/bin/`.

To change directory and run the Makefile, run the following commands:
```bash
cd ./aphmm-gpu
make
```

In `Makefile`, we configure the CUDA-related flags based on our experimental setup in two ways. First, we assume that the CUDA path is `/usr/local/cuda` in your machine. Second, we set the gencode flags for CUDA based on the GPUs we evaluate ApHMM-GPU with. ApHMM-GPU is evaluated using the NVIDIA Titan V and NVIDIA A100 GPUs. In `Makefile`, we include the configurations for these GPUs. Specifically, we set the gencode flags based on the compute capabilities. For NVIDIA Titan V, we set `-gencode arch=compute_70,code=compute_70` and for NVIDIA A100 we set `-gencode arch=compute_80,code=compute_80`
We suggest changing the following entries in `Makefile` if you want to customize the the CUDA path and gencode flags:
```bash
#CUDA Path. Set it to any other path if CUDA is located at somewhere else. The path should include lib, include, and bin directories for the CUDA library.
CUDA_PATH?=/usr/local/cuda

...

#Set to 80 for A100, 70 for Titan V. Set it to any other value for other GPU architectures
SM=70
```

From this point on, we assume that the `aphmm-gpu` binary can be found searching your `$PATH`.


## Running ApHMM-GPU
### Outputting the Help Message

ApHMM-GPU is implemented in Apollo. The help command will output the help message as outputted in Apollo. To generate the help message, following command should be executed:

```bash
aphmm-gpu -h
```

Different from Apollo, ApHMM-GPU also provides the `--gthread` option, which identifies the number of *observation* sequences to process in parallel. Each observation updates the parameters of a pHMM independently, also known as training. ApHMM-GPU processes `--gthread` many observation sequences concurrently to train a pHMM. Further, each CUDA core processes a single state in a pHMM. Thus, the size of a pHMM and `--gthread` determines the number of parallelism that ApHMM-GPU executes at each kernel run. When using default parameters, around 2,500 states are processed during a single training. We set `--gthread` to 256. Thus, ApHMM-GPU requests around 256 x 2,500 = 640,000 CUDA cores at each kernel execution. Number of states are determined by the number of insertion states, deletion transitions, and chunk sizes. The help message provides detailed information on how to set these parameters. We set `--gthread` to 256 because overall execution time of Apollo starts slowing down beyond this number.

### Running ApHMM-GPU with Apollo

Apollo corrects the errors in assembly sequences. To correct errors, Apollo completes its execution in three steps. First, Apollo processes the input data to generate pHMMs from assemblies and to identify observation sequences to use during training. The first step is fully implemented in CPUs. We assume assembly sequences are provided in a single FASTA file. Apollo uses the alignment of other sequences to an assembly sequence to use each alignment information as observation during training. Second, Apollo executes the Baum-Welch algorithm for each observation to update the probabilities in pHMMs. The Baum-Welch algorithm is implemented in `src/HMMTrainer.cpp` and `src/HMMTrainer.cu`. The `src/HMMTrainer.cpp` implementation includes the portion where ApHMM-GPU uses CPUs to execute the parts of the Baum-Welch algorithm (i.e., sorting, data synchronisation, preparing data for further kernel executions). The `src/HMMTrainer.cu` implementation includes the GPU implementation where ApHMM-GPU executes the Forward, Backward, and Parameter Update steps for each timestamp. Third, ApHMM-GPU uses the Viterbi algorithm to generate the consensus sequence from a trained pHMM. This step is fully executed in CPUs.


We show how to use aphmm-gpu to correct errors with Apollo. Assume that you have 1) assembly sequences stored in FASTA file `assembly.fasta`, 2) read sequences stored in `reads.fasta`, 3) the alignment file `alignment.bam` that contains the alignment of the read sequences in `reads.fasta` to assembly sequences in `assembly.fasta`, 4) an output file (corrected assembly sequence) `polished.fasta`. The command below uses `30` CPU threads and processes `256` observations in parallel in GPUs while polishing the assembly:

```bash
./aphmm-gpu -a assembly.fasta -r reads.fasta -m alignment.bam -t 30 -g 256 -o polished.fasta
```
Resulting FASTA file `polished.fasta` will be the final output of Apollo.

### File Format

* Apollo supports alignment files in BAM format. If you have a SAM file you can easily convert your `input.sam` to `input.bam` using the following command:

```bash
samtools view -hb input.sam > input.bam
```
* Apollo requires the input BAM file to be coordinate sorted. You can sort your `input.bam` file using the following command:

```bash
samtools view -h -F4 input.bam | samtools sort -m 16G -l0 > input_sorted.bam
```

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

We provide the input files in the `test/` directory and command we use to evaluate the GPU runtime. To test ApHMM-GPU using your GPUs, you may use the following test run. As ApHMM-GPU is implemented in Apollo, ApHMM-GPU provides the output related to both Apollo and execution time of the steps that ApHMM-GPU executes in a GPU. We use these execution times in the ApHMM paper.

We evaluate ApHMM-GPU using NVIDIA Titan V and NVIDIA A100 GPUs. Our GPU machines include the Intel(R) Xeon(R) Gold 5118 CPU @ 2.30GHz processors, which is used to execute the CPU parts of the ApHMM-GPU implementation. Our GPU machine has 192GB of main memory.


To execute ApHMM-GPU using the test dataset we provide, change to the `test/` directory and execute the `aphmm-gpu` as follows:
```bash
cd test
aphmm-gpu -a Ecoli.O157_assembly.contigs.fasta -r SRR5413248_subreads_1k.fasta -m Ecoli.O157_assembly.contigs_tig00000864_1k.bam -o polished.fasta -c 650 -g 256
```

An example output is as follows:

```bash
...
Number of observations processed in parallel: 256
Overall GPU threads to execute at each Forward kernel: 307200
Overall GPU threads to execute at each Backward kernel: 1126400
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

From the above example output, we are mainly interested in the following four values: `GPU Calculation (single forward)`, `GPU Calculation (single backward)`, `avgOverallPosteriorCalc`, and `avgmaximizeEMTime time (avg of all contigs)`. Apart from `avgOverallPosteriorCalc`, all of the values show the time it took for GPU to calculate the relevant parts: forward calculation for a single observation, backward calculation for a single observation, and the expectation-maximization (EM) step. The value for `avgOverallPosteriorCalc` shows time for accumulating values of the previously calculated Forward/Backward values that are used in the EM step.

Further, the above example output also shows the number of CUDA cores requested for each kernel execution during Forward and Backward calculations. For the above example, the Forward and Backward calculations request 307,200 and 1,126,400 CUDA cores for each kernel execution, respectively.

## Acknowledgments
We acknowledge support from the SAFARI Research Group’s industrial partners, especially Intel, Google, Huawei, Microsoft, VMware, and the Semiconductor Research Corporation.
