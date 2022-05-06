	/** @file HMMTrainer.cpp
* @brief One sentence brief
*
* More details
* In multiple lines
* Copyright © 2020 SAFARI
*
* @author Can Firtina
* @bug No known bug
*/
#include "HMMTrainer.h"
#include "HMMTrainer.cuh"
#include <limits>
#include <algorithm>
#include <chrono>

// to write matrix to file
#include <iostream>
#include <fstream>

// things for GPU

#define DEBUG_CPU_TIME

#ifdef DEBUG_CPU_TIME
double overallArrayAllocTime = 0;
cnt_prec cntArrayAllocTime = 0;

double overallForwardExec = 0;
cnt_prec cntForwardExec = 0;
double overallForwardKernel = 0;
cnt_prec cntForwardKernel = 0;
double overallForwardSortKernel = 0;
cnt_prec cntForwardSortKernel = 0;
double overallForwardTransitionKernel = 0;
cnt_prec cntForwardTransitionKernel = 0;

double overallBackwardExec = 0;
cnt_prec cntBackwardExec = 0;
double overallBackwardKernel = 0;
cnt_prec cntBackwardKernel = 0;
double overallBackwardSortKernel = 0;
cnt_prec cntBackwardSortKernel = 0;
double overallBackwardTransitionKernel = 0;
cnt_prec cntBackwardTransitionKernel = 0;

double overallCalcPosteriorKernel = 0;
cnt_prec cntCalcPosteriorKernel = 0;

double overallAddPosteriorKernel = 0;
cnt_prec cntAddPosteriorKernel = 0;

double overallPosteriorCalcPerRead = 0;
cnt_prec cntPosteriorCalcPerRead = 0;

int totreadcnt = 0;
int totreadcntcontig = 0;
#endif

#define gtind(t, toff, gt, gtoff, i) ((t*toff)+(gt*gtoff)+i)

typedef double prob_prec; //precision of the probabilities
typedef int cnt_prec; //precision of the count arrays -- relative to the avg. depth of coverage
typedef unsigned ind_prec; //precision of the size of contigs -- relative to genome size

HMMTrainer::HMMTrainer(){graph = NULL;}
HMMTrainer::HMMTrainer(HMMGraph* graph):graph(graph){}
HMMTrainer::~HMMTrainer(){}

void HMMTrainer::calculateFB(std::vector<seqan::BamFileIn>& alignmentSetsIn,
							std::vector<seqan::BamAlignmentRecord>& curRecords,
							const std::vector<seqan::FaiIndex>& readFAIs, unsigned thread, unsigned gthread){
	if(graph == NULL) return;

	totreadcntcontig = 0;

    overallForwardExec = 0;
    cntForwardExec = 0;

    overallBackwardExec = 0;
    cntBackwardExec = 0;

    overallCalcPosteriorKernel = 0;
    cntCalcPosteriorKernel = 0;
    overallAddPosteriorKernel = 0;
    cntAddPosteriorKernel = 0;
    overallPosteriorCalcPerRead = 0;
    cntPosteriorCalcPerRead = 0;

	//Starting F/B calculations in separate threads (i.e., for each read alignment)
	for(size_t curSet = 0; curSet < alignmentSetsIn.size(); ++curSet){
		if(atEnd(alignmentSetsIn[curSet]) || curRecords[curSet].rID != graph->contigId) continue;

		bool shouldPolish = true;
		while(shouldPolish){
			// ind_prec readIndex = 0; //for threads
			// std::vector<std::thread> threads;
			std::vector<Read> reads;

			cnt_prec maxGraphSize = MATCH_OFFSET(graph->params.chunkSize, graph->params.chunkSize*0.2, 
												 graph->params.maxInsertion);

			shouldPolish = fillBuffer(alignmentSetsIn[curSet], readFAIs[curSet],curRecords[curSet],
									  reads, graph->contigId, graph->params.mapQ, 50000,
									  graph->params.chunkSize, maxGraphSize);

			//we shuffle to reduce the race condiitons that may frequently arise in a multi-threaded run
            auto rng = std::default_random_engine {};
            std::shuffle(std::begin(reads), std::end(reads), rng);

			totreadcnt += reads.size();
			totreadcntcontig += reads.size();

			//alignedReads.size will be 0 if there is no more alignment record left to read in the current file
			// for(unsigned i = 0; i < thread && i < (unsigned)reads.size(); ++i)
				// threads.push_back(std::thread(&HMMTrainer::fbThreadPool, this, std::ref(reads), std::ref(readIndex)));
			fbThreadPool(reads, graph->params.chunkSize, thread, gthread);

			//buffer is cleared here. every thread needs to wait before the buffer gets reloaded again
			// for(size_t i = 0; i < threads.size(); ++i) threads[i].join();
		}
	}//F/B calculation is now done

    #ifdef DEBUG_CPU_TIME
    printf("avgForwardExec time (per read of a contig): %f ms\n" ,overallForwardExec/cntForwardExec);

    printf("avgBackwardExec time (per read of a contig): %f ms\n" ,overallBackwardExec/cntBackwardExec);

    printf("avgOverallPosteriorCalc time (per read of a contig): %f ms\n" ,overallPosteriorCalcPerRead/cntPosteriorCalcPerRead);

    std::cout << "Total processed read count for a contig: " << totreadcntcontig << std::endl;
    std::cout << "Total processed read count so far: " << totreadcnt << std::endl;
    #endif
}

//@IMPORTANT: log calculations are done here, instead of in the viterbi calculations
void HMMTrainer::maximizeEM(){
	if(graph == NULL) return;

	graph->maxEmissionProbs = new std::pair<prob_prec, char>[graph->numberOfStates];
	// prob_prec lastInsertionTransitionResetProb = graph->preCalculatedLogTransitionProbs[1];
	prob_prec lastInsertionTransitionProb = log10(graph->params.matchTransition + graph->params.insertionTransition);

	// start the time measure
	std::int64_t overallStartTime = currentTimeMillis();

	// prepare for the GPU kernel time timemeasure
	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	// define the array-sizes
	ind_prec numberOfStates = graph->numberOfStates;
	ind_prec halflength_bitvector =  graph->contigLength;
	ind_prec width_transitions_probs = graph->numOfTransitionsPerState * sizeof(prob_prec);
	ind_prec width_transitions_count = graph->numOfTransitionsPerState * sizeof(cnt_prec);
	ind_prec size_constantIntegers = 3 * sizeof(int);
	ind_prec size_constantDoubles = 2 * sizeof(double);
	ind_prec size_bitVector = 2 * halflength_bitvector * sizeof(bool);
	ind_prec size_stateProcessedCount = numberOfStates * sizeof(cnt_prec);
	ind_prec size_defaultEmissionProbs = 4 * sizeof(prob_prec);
	ind_prec size_emissionProbabilities = 4 * numberOfStates * sizeof(prob_prec);
	//QUESTION: can we divide size_bestEmissionProbPairs into two: double and char arrays
	ind_prec size_bestEmissionProbPairs = 2 * numberOfStates * sizeof(prob_prec);
	ind_prec s_lutTransition = numberOfStates * width_transitions_probs;
	ind_prec size_transitionProcessedCount = numberOfStates * width_transitions_count;

	int firstStateGPU = graph->params.maxInsertion+2; // calculated on GPU
	int lastStateGPU = numberOfStates - (graph->params.maxInsertion + 1) - 1; // calculated on GPU

	// create the bitvector for the complete Graph
	// TODO: reuse this bitvector for the forward and backward calculations
	//FIXED: currentPosition+1 -> currentPosition
	bool *bitvectorGraph = new bool[2 * halflength_bitvector];//(halflength_bitvector+1)];
	for(ind_prec currentPosition = 0; currentPosition < halflength_bitvector; ++currentPosition) {
		switch((char) graph->contig[currentPosition]) {
			case 'A': bitvectorGraph[2*(currentPosition)] = 0;
					bitvectorGraph[2*(currentPosition) + 1] = 0;
					break;
			case 'T': bitvectorGraph[2*(currentPosition)] = 0;
					bitvectorGraph[2*(currentPosition) + 1] = 1;
					break;
			case 'G': bitvectorGraph[2*(currentPosition)] = 1;
					bitvectorGraph[2*(currentPosition) + 1] = 0;
					break;
			case 'C': bitvectorGraph[2*(currentPosition)] = 1;
					bitvectorGraph[2*(currentPosition) + 1] = 1;
					break;
		}
	}

	// create and fill the vectors for the GPU
	double defaultEmissionProbs[] = { (double) graph->params.matchEmission
			, (double) graph->params.mismatchEmission
			, 0.0, (double) graph->params.insertionEmission };

	double constantDoubles[] = { (double) graph->params.matchTransition
			, (double) graph->params.insertionTransition};

	int constantIntegers[] = { (int) graph->params.maxInsertion, (int) graph->numOfTransitionsPerState
			, (int) lastStateGPU };

	int *transitionsProcessedCount = new int[(unsigned) numberOfStates*graph->numOfTransitionsPerState];
	double *transitionProbabilities = new double[(unsigned) numberOfStates*graph->numOfTransitionsPerState];

	for(unsigned i = 0; i < numberOfStates; i++) {
		for(unsigned j = 0; j < graph->numOfTransitionsPerState; j++) {
			transitionProbabilities[i*graph->numOfTransitionsPerState+j] = graph->transitionProbs[i][j];
			transitionsProcessedCount[i*graph->numOfTransitionsPerState+j] = graph->transitionProcessedCount[i][j];
		}
	}

	double *emissionProbabilities = new double[numberOfStates*4];
	for(int i = 0; i < numberOfStates; i++) {
		for(int j = 0; j < 4; j++) {
			emissionProbabilities[i*4+j] = graph->emissionProbs[i][j];
		}
	}

	// allocate 2D arrays, that are mapped onto 1D arrays, on GPU
	double *d_transitionProbs = NULL;
	cudaMalloc((void**)&d_transitionProbs, s_lutTransition);
	checkForCudaError();
	double *d_emissionProbs = NULL;
	cudaMalloc((void**)&d_emissionProbs, size_emissionProbabilities);
	checkForCudaError();
	int *d_transitionProcessedCount = NULL;
	cudaMalloc((void**)&d_transitionProcessedCount, size_transitionProcessedCount);
	checkForCudaError();

	// allocate 1D arrays on GPU
	double *d_preCalculatedLogTransitionProbs = NULL;
	cudaMalloc((void**)&d_preCalculatedLogTransitionProbs, width_transitions_probs);
	checkForCudaError();
	int *d_constantIntegers = NULL;
	cudaMalloc((void**)&d_constantIntegers, size_constantIntegers);
	checkForCudaError();
	int *d_stateProcessedCount = NULL;
	cudaMalloc((void**)&d_stateProcessedCount, size_stateProcessedCount);
	checkForCudaError();
	double *d_bestEmissionProbPairs = NULL; // this array will only be copied back from GPU
	cudaMalloc((void**)&d_bestEmissionProbPairs, size_bestEmissionProbPairs);
	checkForCudaError();
	bool *d_bitvectorGraph = NULL;
	cudaMalloc((void**)&d_bitvectorGraph, size_bitVector);
	checkForCudaError();
	double *d_constantDoubles = NULL;
	cudaMalloc((void**)&d_constantDoubles, size_constantDoubles);
	checkForCudaError();
	double *d_defaultEmissionProbs = NULL;
	cudaMalloc((void**)&d_defaultEmissionProbs, size_defaultEmissionProbs);
	checkForCudaError();

	// copy 1D vectors to GPU
	cudaMemcpy(d_preCalculatedLogTransitionProbs, graph->preCalculatedLogTransitionProbs
			, width_transitions_probs, cudaMemcpyHostToDevice);
	checkForCudaError();
	cudaMemcpy(d_constantIntegers, constantIntegers, size_constantIntegers, cudaMemcpyHostToDevice);
	checkForCudaError();
	cudaMemcpy(d_stateProcessedCount, graph->stateProcessedCount, size_stateProcessedCount
			, cudaMemcpyHostToDevice);
	checkForCudaError();
	cudaMemcpy(d_bitvectorGraph, bitvectorGraph, size_bitVector, cudaMemcpyHostToDevice);
	checkForCudaError();
	cudaMemcpy(d_constantDoubles, constantDoubles, size_constantDoubles, cudaMemcpyHostToDevice);
	checkForCudaError();
	cudaMemcpy(d_defaultEmissionProbs, defaultEmissionProbs, size_defaultEmissionProbs
			, cudaMemcpyHostToDevice);
	checkForCudaError();
	cudaMemcpy(d_transitionProbs, transitionProbabilities, s_lutTransition, cudaMemcpyHostToDevice);
	checkForCudaError();
	cudaMemcpy(d_emissionProbs, emissionProbabilities, size_emissionProbabilities, cudaMemcpyHostToDevice);
	checkForCudaError();
	cudaMemcpy(d_transitionProcessedCount, transitionsProcessedCount, size_transitionProcessedCount
			, cudaMemcpyHostToDevice);
	checkForCudaError();

	std::int64_t gpuPreparationTime = currentTimeMillis() - overallStartTime;

	// Launch the calculations on the CUDA Kernel
	int threadsPerBlock = 256;
	int blocksPerGrid = (numberOfStates + threadsPerBlock - 1) / threadsPerBlock;

	cudaEventRecord(start);
	gpu_maximizeEM(blocksPerGrid, threadsPerBlock,d_transitionProbs, d_emissionProbs, d_preCalculatedLogTransitionProbs, 		   d_constantIntegers, d_constantDoubles, d_defaultEmissionProbs, d_stateProcessedCount, 
				   d_transitionProcessedCount, d_bestEmissionProbPairs, d_bitvectorGraph);

	// wait for the CUDA kernel to complete the calculations
	cudaDeviceSynchronize();
	cudaEventRecord(end);
	checkForCudaError();

	// copy the result from GPU back to CPU
	cudaMemcpy(transitionProbabilities, d_transitionProbs, s_lutTransition
			, cudaMemcpyDeviceToHost);
	checkForCudaError();
	double *bestEmissionProbPairs = new double[2*numberOfStates];
	cudaMemcpy(bestEmissionProbPairs, d_bestEmissionProbPairs, size_bestEmissionProbPairs
			, cudaMemcpyDeviceToHost);
	checkForCudaError();

	// save kernel time
	float gpuKernelTime = 0;
	cudaEventElapsedTime(&gpuKernelTime, start, end);

	// free the memory on the GPU
	cudaFree(d_transitionProbs);
	checkForCudaError();
	cudaFree(d_emissionProbs);
	checkForCudaError();
	cudaFree(d_preCalculatedLogTransitionProbs);
	checkForCudaError();
	cudaFree(d_constantIntegers);
	checkForCudaError();
	cudaFree(d_constantDoubles);
	checkForCudaError();
	cudaFree(d_defaultEmissionProbs);
	checkForCudaError();
	cudaFree(d_stateProcessedCount);
	checkForCudaError();
	cudaFree(d_transitionProcessedCount);
	checkForCudaError();
	cudaFree(d_bestEmissionProbPairs);
	checkForCudaError();
	cudaFree(d_bitvectorGraph);
	checkForCudaError();
	// calculate the first states on the CPU
	maximizeEMCPU(0, firstStateGPU-1);

	// calculate the last states on the CPU
	//FIXED: lastStateGPU -> lastStateGPU+1
	maximizeEMCPU(lastStateGPU+1, numberOfStates-1);
	// create the emissionProbPairs from GPU
	char emissionCharacter;
	for(ind_prec curState = firstStateGPU; curState <= lastStateGPU; ++curState) {
		if(bestEmissionProbPairs[2*curState + 1] == 0) emissionCharacter = 'A';
		else if(bestEmissionProbPairs[2*curState + 1] == 1) emissionCharacter = 'T';
		else if(bestEmissionProbPairs[2*curState + 1] == 2) emissionCharacter = 'G';
		else if(bestEmissionProbPairs[2*curState + 1] == 3) emissionCharacter = 'C';
		else std::cerr << "ERROR: no emissionCharacter had the highest emission probability!" << std::endl;

		graph->maxEmissionProbs[curState] = std::make_pair(log10(bestEmissionProbPairs[2*curState]), emissionCharacter);
	}

	// delete arrays used for CPU
	delete[] bitvectorGraph;
	// delete[] defaultEmissionProbs;
	// delete[] constantDoubles;
	// delete[] constantIntegers;
	//FIXED: added delete[] bestEmissionProbPairs;
	delete[] bestEmissionProbPairs;
	delete[] transitionProbabilities;
	delete[] transitionsProcessedCount;
	delete[] emissionProbabilities;

	// end the overall timemeasure
	std::int64_t overallTime = currentTimeMillis() - overallStartTime;
	printf("maximizeEM for a contig:  totalTime=%dms, gpuPreparationTime=%dms, gpuKernelTime=%fms\n", overallTime
	 		, gpuPreparationTime, gpuKernelTime);

}

/**
 * values in graph->preCalculatedLogTransitionProbs will be changed temporarily
 * values in graph->transitionProbs[][] will be changed
 * values in graph->emissionProbs[][] will be changed
 * values in graph->maxEmissionProbs[] will be changed
 */
void HMMTrainer::maximizeEMCPU(int firstState, int lastState) {

	prob_prec lastInsertionTransitionResetProb = graph->preCalculatedLogTransitionProbs[1];
	prob_prec lastInsertionTransitionProb = log10(graph->params.matchTransition + graph->params.insertionTransition);

	for(ind_prec curState = firstState; curState <= lastState; ++curState){
		if(graph->seqGraph[curState].isLastInsertionState()){
			// for the last insertion state, the insertion probs change so that it wont have insertion transition
			graph->preCalculatedLogTransitionProbs[1] = lastInsertionTransitionProb;
			graph->transitionProbs[curState][0] = std::numeric_limits<int>::min()/100;
		}

		ind_prec curTransition = (graph->seqGraph[curState].isLastInsertionState())?1:0;
		prob_prec maxEmissionProb;
		char maxEmissionChar;
		// if this state ever processed then its probs may need to be updated
		if(graph->stateProcessedCount[curState] > 0){
			while(curTransition < graph->numOfTransitionsPerState){
				graph->transitionProbs[curState][curTransition] =
				(graph->transitionProcessedCount[curState][curTransition] > 0)?
				log10(graph->transitionProbs[curState][curTransition]/
					graph->transitionProcessedCount[curState][curTransition]):
				graph->preCalculatedLogTransitionProbs[curTransition];
				curTransition++;
			}

			graph->emissionProbs[curState][A] /= graph->stateProcessedCount[curState];
			graph->emissionProbs[curState][T] /= graph->stateProcessedCount[curState];
			graph->emissionProbs[curState][G] /= graph->stateProcessedCount[curState];
			graph->emissionProbs[curState][C] /= graph->stateProcessedCount[curState];
		}else{ //initial probs to be set unless this state has been processed
			while(curTransition < graph->numOfTransitionsPerState){
				graph->transitionProbs[curState][curTransition] = graph->preCalculatedLogTransitionProbs[curTransition];
				curTransition++;
			}

			graph->emissionProbs[curState][A] = graph->seqGraph[curState].getEmissionProb('A', graph->params);
			maxEmissionProb = graph->emissionProbs[curState][A]; maxEmissionChar = 'A';
			graph->emissionProbs[curState][T] = graph->seqGraph[curState].getEmissionProb('T', graph->params);
			graph->emissionProbs[curState][G] = graph->seqGraph[curState].getEmissionProb('G', graph->params);
			graph->emissionProbs[curState][C] = graph->seqGraph[curState].getEmissionProb('C', graph->params);
		}

		maxEmissionProb = graph->emissionProbs[curState][A]; maxEmissionChar = 'A';
		if(graph->emissionProbs[curState][T] > maxEmissionProb){
			maxEmissionProb = graph->emissionProbs[curState][T]; maxEmissionChar = 'T';
		}
		if(graph->emissionProbs[curState][G] > maxEmissionProb){
			maxEmissionProb = graph->emissionProbs[curState][G]; maxEmissionChar = 'G';
		}
		if(graph->emissionProbs[curState][C] > maxEmissionProb){
			maxEmissionProb = graph->emissionProbs[curState][C]; maxEmissionChar = 'C';
		}

		graph->maxEmissionProbs[curState] = std::make_pair(log10(maxEmissionProb), maxEmissionChar);

		if(graph->seqGraph[curState].isLastInsertionState())
			graph->preCalculatedLogTransitionProbs[1] = lastInsertionTransitionResetProb;
	}
}

void HMMTrainer::fbThreadPool(const std::vector<Read>& reads, cnt_prec chunkSize, unsigned thread, unsigned gthread){
	if(graph == NULL) return;

	// ind_prec readIndex;
	// ind_prec maxFilteredMatrixSize = graph->params.filterSize;
	// ind_prec maxFilteredTransitions = graph->params.filterSize*graph->numOfTransitionsPerState;
	// ind_prec maxTransitions = maxGraphSize*graph->numOfTransitionsPerState;
	// uint32_t gthread = (gthread+maxFilteredTransitions-1)/maxFilteredTransitions;
	printf("Number of observations processed in parallel: %u\n", gthread);
	printf("Overall GPU threads to execute at each Forward kernel: %u\n", gthread*graph->params.filterSize*graph->numOfTransitionsPerState);
	printf("Overall GPU threads to execute at each Backward kernel: %u\n", gthread*graph->params.filterSize*(graph->params.maxInsertion+1)*(graph->params.maxDeletion+1));
	
    double loc_overallForwardExec = 0; cnt_prec loc_cntForwardExec = 0;
    double loc_overallBackwardExec = 0; cnt_prec loc_cntBackwardExec = 0;
    double loc_overallPosteriorCalcPerRead = 0; cnt_prec loc_cntPosteriorCalcPerRead = 0;

	std::vector<TransitionInfoNode> curStateTransitions;
	prob_prec* curStateTransitionLikelihood = new prob_prec[graph->numOfTransitionsPerState];
	prob_prec* curStateEmissionProbs = new prob_prec[totalNuc];

	ind_prec curRead = 0; //0-based. refers to the current read id
	ind_prec numOfReads = (ind_prec)reads.size();
	// {//block for obtaining the next aligned read
	// 	std::lock_guard<std::mutex> lk(indexMutex);
	// 	curRead = readIndex++;
	// }

	ind_prec maxGraphSize = MATCH_OFFSET(chunkSize, chunkSize*0.2, graph->params.maxInsertion);
	uint64_t matrixSize = chunkSize*gthread*maxGraphSize;
	ind_prec toff = gthread*maxGraphSize;
	prob_prec* forwardMatrix = new prob_prec[matrixSize]; 
	prob_prec* backwardMatrix = new prob_prec[matrixSize];
	ind_prec* readLength = new ind_prec[gthread];
	ind_prec* maxDistanceOnAssembly = new ind_prec[gthread];
	ind_prec* offset = new ind_prec[gthread];
	ind_prec* endPosState = new ind_prec[gthread];
	ind_prec* fbMatrixSize = new ind_prec[gthread];
	char** readChars = new char*[gthread];

	while(curRead + gthread < numOfReads){

		std::fill_n(forwardMatrix, matrixSize, 0);
		std::fill_n(backwardMatrix, matrixSize, 0);
		std::fill_n(readLength, gthread, 0);
		std::fill_n(maxDistanceOnAssembly, gthread, 0);
		std::fill_n(offset, gthread, 0);
		std::fill_n(endPosState, gthread, 0);
		std::fill_n(fbMatrixSize, gthread, 0);
		// std::fill_n(readChars, gthread, NULL);

		for(auto gt = 0; gt < gthread && curRead+gt < numOfReads; ++gt){
			readLength[gt] = (ind_prec)seqan::length(reads[curRead+gt].read);
			//end pos exclusive
			maxDistanceOnAssembly[gt] = std::min(reads[curRead+gt].endPos, graph->contigLength);
			//states prior to this wont be processed. offset value is to ignore these states
			offset[gt] = (reads[curRead+gt].pos>0)?MATCH_DOFFSET(reads[curRead+gt].pos,1,graph->params.maxInsertion):0;
			//maximum number of states to be processed
			endPosState[gt] = MATCH_OFFSET(maxDistanceOnAssembly[gt], 0, graph->params.maxInsertion);
			if(endPosState[gt] > graph->numberOfStates) endPosState[gt] = graph->numberOfStates;
			fbMatrixSize[gt] = (endPosState[gt] > offset[gt])?endPosState[gt]-offset[gt]:0;
			readChars[gt] = toCString(reads[curRead+gt].read);

			assert(readLength[gt] == chunkSize && fbMatrixSize[gt] <= maxGraphSize);
		}

		ind_prec j; //j value in i,j transitions

        #ifdef DEBUG_CPU_TIME
        auto realstart = std::chrono::high_resolution_clock::now();
        #endif

        fillForwardMatrix(forwardMatrix,fbMatrixSize,readChars,offset,endPosState,readLength,chunkSize, maxGraphSize, 
        				  gthread);

        #ifdef DEBUG_CPU_TIME
        auto realend = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> dur = realend - realstart;
        loc_overallForwardExec += dur.count();
        loc_cntForwardExec++;
        #endif

        #ifdef DEBUG_CPU_TIME
        auto realstart2 = std::chrono::high_resolution_clock::now();
        #endif

        fillBackwardMatrix(backwardMatrix,fbMatrixSize,readChars,endPosState,offset,readLength,chunkSize, maxGraphSize,
         				   gthread);

        #ifdef DEBUG_CPU_TIME
        auto realend2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> dur2 = realend2 - realstart2;
        loc_overallBackwardExec += dur2.count();
        loc_cntBackwardExec++;
        #endif

        //this part can still be parallized in cpu
		for(auto gt = 0; gt < gthread && curRead+gt < numOfReads; ++gt){
			
			if(fbMatrixSize[gt] == 0) continue;

            #ifdef DEBUG_CPU_TIME
            auto realstart3 = std::chrono::high_resolution_clock::now();
            #endif
            
			//updating probabilities wrt the f/b matrices computed just now
			for(ind_prec curState = INSERTION_DOFFSET(reads[curRead+gt].pos, 1, 1, graph->params.maxInsertion); 
				curState < endPosState[gt]; ++curState){

				if(graph->seqGraph[curState].isLastInsertionState())
					//for the last insertion state, the insertion probs change
					graph->preCalculatedTransitionProbs[1] = graph->params.matchTransition + 
															 graph->params.insertionTransition;

				if(curState-offset[gt] < fbMatrixSize[gt]){
					ind_prec matchoff = MATCH_OFFSET(graph->seqGraph[curState].getCharIndex(), 0, 
													 graph->params.maxInsertion);
					std::fill_n(curStateTransitionLikelihood, graph->numOfTransitionsPerState,0);
					std::fill_n(curStateEmissionProbs, totalNuc, 0);
					insertNewForwardTransitions(&curStateTransitions,graph->seqGraph[curState],
												graph->params.maxDeletion, graph->params.maxInsertion);

					for(ind_prec t = 0; t < readLength[gt]; ++t){
						//transition probabilities
						if(t < readLength[gt]-1){
							for(size_t curTr = 0; curTr < curStateTransitions.size(); ++curTr){
								j = curStateTransitions[curTr].toState;
								if(j >= offset[gt] && j < endPosState[gt]){
									//0->insertion, 1-> match, 2,3...->deletions
									ind_prec transitionIndex = (j - matchoff)/(graph->params.maxInsertion+1);

									curStateTransitionLikelihood[transitionIndex] +=
									forwardMatrix[gtind(t, toff, gt, maxGraphSize, (curState-offset[gt]))]*
									backwardMatrix[gtind((t+1), toff, gt, maxGraphSize, (j-offset[gt]))]*
									graph->preCalculatedTransitionProbs[transitionIndex]*
									graph->seqGraph[j].getEmissionProb(readChars[gt][t+1], graph->params);
								}
							}
						}

						//emission probabilities
						char emitChar = (readChars[gt][t] != 'N')?readChars[gt][t]:
						(graph->seqGraph[curState].isMatchState())?graph->seqGraph[curState].getNucleotide():'\0';
						Nucleotide chosenNuc = (emitChar == 'A' || emitChar == 'a')?A:
						(emitChar == 'T' || emitChar == 't')?T:
						(emitChar == 'G' || emitChar == 'g')?G:
						(emitChar == 'C' || emitChar == 'c')?C:totalNuc;
						if(chosenNuc < totalNuc)
							curStateEmissionProbs[chosenNuc] +=
							forwardMatrix[gtind(t, toff, gt, maxGraphSize, (curState-offset[gt]))]*
							backwardMatrix[gtind(t, toff, gt, maxGraphSize, (curState-offset[gt]))];
					}
					curStateTransitions.clear();
					prob_prec totalEmissionProbs = curStateEmissionProbs[A] + curStateEmissionProbs[T] +
					curStateEmissionProbs[G] + curStateEmissionProbs[C];
					prob_prec totalTransitionLikelihoods = 0;
					prob_prec processedTransitionProb = 0;
					for(ind_prec i = (graph->seqGraph[curState].isLastInsertionState())?1:0;
						i < graph->numOfTransitionsPerState; ++i){
						if(curStateTransitionLikelihood[i] > 0 || i == 0){
							totalTransitionLikelihoods += curStateTransitionLikelihood[i];
							processedTransitionProb += graph->preCalculatedTransitionProbs[i];
						}
					}

					if(totalEmissionProbs != 0 && curState < endPosState[gt]){
						if(totalTransitionLikelihoods != 0){
							// {//block for updating the transition probs and the transition proccessed count
								// std::lock_guard<std::mutex> lk(transitionProbMutex);

							for(ind_prec i = (graph->seqGraph[curState].isLastInsertionState())?1:0;
								i < graph->numOfTransitionsPerState; ++i){
								if(curStateTransitionLikelihood[i] > 0 || i == 0){
									graph->transitionProbs[curState][i] +=
									(curStateTransitionLikelihood[i]/totalTransitionLikelihoods)*
									processedTransitionProb;
									graph->transitionProcessedCount[curState][i]++;
								}
							}
							// }
						}

						// {//block for updating the emission probs and the state processed count
							// std::lock_guard<std::mutex> lk(emissionProbMutex);

						graph->emissionProbs[curState][A] += curStateEmissionProbs[A]/totalEmissionProbs;
						graph->emissionProbs[curState][T] += curStateEmissionProbs[T]/totalEmissionProbs;
						graph->emissionProbs[curState][G] += curStateEmissionProbs[G]/totalEmissionProbs;
						graph->emissionProbs[curState][C] += curStateEmissionProbs[C]/totalEmissionProbs;
						graph->stateProcessedCount[curState]++;
						// }
					}
				}

				//for the last insertion state, the insertion probs change so that it wont have insertion
				//transition. putting it back to normal now
				if(graph->seqGraph[curState].isLastInsertionState())
					graph->preCalculatedTransitionProbs[1] = graph->params.matchTransition;
			}

			#ifdef DEBUG_CPU_TIME
            auto realend3 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> dur3 = realend3 - realstart3;
            loc_overallPosteriorCalcPerRead += dur3.count();
            loc_cntPosteriorCalcPerRead++;
            #endif
		}

		curRead += gthread;
	}

	// #ifdef DEBUG_CPU_TIME
    // {
        // std::lock_guard<std::mutex> lk(indexMutex);

    overallForwardExec += loc_overallForwardExec;
    cntForwardExec += loc_cntForwardExec;

    overallBackwardExec += loc_overallBackwardExec;
    cntBackwardExec += loc_cntBackwardExec;

    overallPosteriorCalcPerRead += loc_overallPosteriorCalcPerRead;
    cntPosteriorCalcPerRead += loc_cntPosteriorCalcPerRead;
    // }
	// #endif

	delete[] curStateTransitionLikelihood;
	delete[] curStateEmissionProbs;
	delete[] forwardMatrix;
	delete[] backwardMatrix;
	delete[] readLength;
	delete[] maxDistanceOnAssembly;
	delete[] offset;
	delete[] endPosState;
	delete[] fbMatrixSize;
	delete[] readChars;
}

void HMMTrainer::fillForwardMatrix(prob_prec* forwardMatrix,ind_prec* fbMatrixSize,char** read, ind_prec* startPosition,
								   ind_prec* maxDistanceOnContig, ind_prec* readLength, cnt_prec chunkSize, 
								   uint64_t maxGraphSize, uint32_t gthread){

	ind_prec toff = gthread*maxGraphSize;
	ind_prec maxFilteredTransitions = graph->params.filterSize*graph->numOfTransitionsPerState;

	//LUTs
	prob_prec* lutTransition = graph->preCalculatedTransitionProbs;
	prob_prec lutEmission[] = {graph->params.matchEmission, graph->params.mismatchEmission, 0.0, 
							   graph->params.insertionEmission};

	std::vector<TransitionInfoNode>* curTrSet[gthread];	//transitions from the previous time
	std::vector<TransitionInfoNode>* nextTrSet[gthread]; //transitions to be processed for the next time
	std::vector<TransitionInfoNode>* tmpTrSet[gthread];

	for(auto gt = 0; gt < gthread; ++gt){
		curTrSet[gt] = new std::vector<TransitionInfoNode>;
		nextTrSet[gt] = new std::vector<TransitionInfoNode>;
	}

	char curChars[gthread]; std::fill_n(curChars, gthread, 'N');
	bool* allowedParentStates = new bool[toff]; std::fill_n(allowedParentStates, toff, false);
	bool* hasStateBeenProcessedBefore = new bool[toff]; std::fill_n(hasStateBeenProcessedBefore, toff, false);

	//1-initialization (t = 1)
	ind_prec curTime = 0; //represents the current time (1...T)

	for(auto gt = 0; gt < gthread; ++gt){
		if(!fbMatrixSize[gt]) continue;
		
		insertNewForwardTransitions(curTrSet[gt], graph->seqGraph[startPosition[gt]], graph->params.maxDeletion,
									graph->params.maxInsertion);
		for(size_t curTransition = 0; curTransition < curTrSet[gt]->size(); ++curTransition){
			TransitionInfoNode& frontTr = curTrSet[gt]->at(curTransition);
			int from_ind = frontTr.from - startPosition[gt];
			int to_ind = frontTr.toState - startPosition[gt];
			if(frontTr.toState < maxDistanceOnContig[gt]){
				ind_prec matchoff = MATCH_OFFSET(graph->seqGraph[frontTr.from].getCharIndex(),0,
												 graph->params.maxInsertion);

				//0->insertion, 1-> match, 2,3...->deletions
				ind_prec transitionIndex=(frontTr.toState-matchoff)/(graph->params.maxInsertion+1);

				forwardMatrix[gtind(0,0,gt,maxGraphSize,to_ind)]+= lutTransition[transitionIndex]*
									graph->seqGraph[frontTr.toState].getEmissionProb(read[gt][curTime],graph->params);

				insertNewForwardTransitions(nextTrSet[gt], graph->seqGraph[frontTr.toState],graph->params.maxDeletion,
											graph->params.maxInsertion);
			}
		}
		curTrSet[gt]->clear();

		//find the most likely states that should be allowed to make the next transitions
		findMaxValues(&forwardMatrix[gtind(0,0,gt,maxGraphSize,0)], &allowedParentStates[gt*maxGraphSize], 0,
					  fbMatrixSize[gt], graph->params.filterSize);

		tmpTrSet[gt] = curTrSet[gt];
		curTrSet[gt] = nextTrSet[gt];
		nextTrSet[gt] = tmpTrSet[gt];
	}

	/**
	* prepare the arrays for the GPU
	*/
	ind_prec l_filtered_transitions = gthread*maxFilteredTransitions;
	cnt_prec* filtered_transitions = new cnt_prec[l_filtered_transitions];
	std::fill_n(filtered_transitions, l_filtered_transitions, 0);
	cnt_prec nrOfValidTransitions[gthread]; // counts how many transitions are valid/used from the array

	/**
	* Forward-Calculations for each transition on GPU in parallel
	*/
	ind_prec s_forwardValues = toff*sizeof(prob_prec);
	ind_prec s_curChars = gthread*sizeof(char);
	ind_prec s_readLength = gthread*sizeof(ind_prec);
	ind_prec s_startPosition = gthread*sizeof(ind_prec);
	ind_prec s_seq = graph->contigLength*sizeof(char);
	ind_prec s_filtered_transitions = l_filtered_transitions*sizeof(cnt_prec);
	// ind_prec s_transitions = l_transitions*sizeof(bool);
	ind_prec s_lutTransition = graph->numOfTransitionsPerState*sizeof(prob_prec);
	ind_prec s_lutEmission  = 4*sizeof(prob_prec);
	
	// start time measure for the preparation of the gpu data
	std::int64_t overallStartTime = currentTimeMillis();
	float gpuKernelTime = 0;
	int nrOfKernelCalls = 0;

	// prepare for the GPU kernel time timemeasure
	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	// Allocate the device input vectors
	ind_prec forw_from = 0;
	ind_prec forw_to = 1;
	prob_prec* d_forwardValues[] = {NULL, NULL};
	cudaMalloc((void**)&d_forwardValues[0], s_forwardValues); checkForCudaError();
	cudaMalloc((void**)&d_forwardValues[1], s_forwardValues); checkForCudaError();

	char* d_curChars = NULL;
	cudaMalloc((void**)&d_curChars, s_curChars); checkForCudaError();

	ind_prec* d_readLength = NULL;
	cudaMalloc((void**)&d_readLength, s_readLength); checkForCudaError();

	ind_prec* d_startPosition = NULL;
	cudaMalloc((void**)&d_startPosition, s_startPosition); checkForCudaError();

	char* d_seq = NULL;
	cudaMalloc((void**)&d_seq, s_seq); checkForCudaError();

	cnt_prec* d_filtered_transitions = NULL;
	cudaMalloc((void**)&d_filtered_transitions, s_filtered_transitions); checkForCudaError();

	// bool* d_transitions = NULL;
	// cudaMalloc((void**)&d_transitions, s_transitions); checkForCudaError();

	prob_prec* d_lutTransition = NULL;
	cudaMalloc((void**)&d_lutTransition, s_lutTransition); 
	checkForCudaError();

	prob_prec* d_lutEmission = NULL;
	cudaMalloc((void**)&d_lutEmission, s_lutEmission); checkForCudaError();

	// copy arrays to GPU that remain constant at all times
	cudaMemcpy(d_readLength, readLength, s_readLength, cudaMemcpyHostToDevice); checkForCudaError();
	cudaMemcpy(d_startPosition, startPosition, s_startPosition, cudaMemcpyHostToDevice); checkForCudaError();
	cudaMemcpy(d_seq, graph->contigChar, s_seq, cudaMemcpyHostToDevice); checkForCudaError();
	cudaMemcpy(d_lutTransition, lutTransition, s_lutTransition, cudaMemcpyHostToDevice); checkForCudaError();
	cudaMemcpy(d_lutEmission, lutEmission, s_lutEmission, cudaMemcpyHostToDevice); checkForCudaError();
	
	// end timemeasure for the gpu preparation and start it for the calculations
	std::int64_t gpuPreparationTime = currentTimeMillis() - overallStartTime;
	std::int64_t calculationStartTime = currentTimeMillis();

	// calculate the forward values for each time
	while(curTime < chunkSize-1){

		curTime++;
		for(auto gt = 0; gt < gthread; ++gt){
			if(!fbMatrixSize[gt] || readLength[gt] < curTime || curTrSet[gt]->empty()) continue;

			nrOfValidTransitions[gt] = 0;
			curChars[gt] = read[gt][curTime];
			ind_prec gt_off = gt*maxGraphSize;
			for(int currentTransition = 0; currentTransition < curTrSet[gt]->size(); ++currentTransition){
				TransitionInfoNode& frontTr = curTrSet[gt]->at(currentTransition);
				ind_prec from_ind = frontTr.from - startPosition[gt];
				ind_prec to_ind = frontTr.toState - startPosition[gt];
				if(from_ind >= maxGraphSize || to_ind >= maxGraphSize || to_ind >= fbMatrixSize[gt]) continue;

				// only add valid transitions to the transitionSet for the GPU
				if(allowedParentStates[gt_off+from_ind]){
					if(!hasStateBeenProcessedBefore[gt_off+to_ind]&&nrOfValidTransitions[gt]<maxFilteredTransitions){
						filtered_transitions[gt*maxFilteredTransitions + nrOfValidTransitions[gt]++] = to_ind;
						insertNewForwardTransitions(nextTrSet[gt], graph->seqGraph[frontTr.toState], 
													graph->params.maxDeletion,
													graph->params.maxInsertion);
						hasStateBeenProcessedBefore[gt_off+to_ind] = true;
					}
				}
			}
		}

		// copy the non constant data to the GPU

		cudaMemcpy(d_forwardValues[forw_from],&forwardMatrix[gtind((curTime-1), toff, 0, 0, 0)],s_forwardValues,
				   cudaMemcpyHostToDevice); checkForCudaError();
		cudaMemset(d_forwardValues[forw_to], 0, s_forwardValues); checkForCudaError();

		cudaMemcpy(d_curChars, curChars, s_curChars, cudaMemcpyHostToDevice);
		checkForCudaError();

		cudaMemcpy(d_filtered_transitions, filtered_transitions, s_filtered_transitions, cudaMemcpyHostToDevice);
		checkForCudaError();

		// Launch the calculations on the CUDA Kernel
		int threadsPerBlock = 256;
		int blocksPerGrid = (l_filtered_transitions+threadsPerBlock-1)/threadsPerBlock;

		cudaEventRecord(start, 0);
		gpu_forwardCalculation(blocksPerGrid, threadsPerBlock, d_forwardValues[forw_from], d_forwardValues[forw_to], 
							   d_curChars,d_readLength,d_startPosition,d_seq,d_filtered_transitions,d_lutTransition, 
							   d_lutEmission, graph->params.maxInsertion, graph->params.maxDeletion, maxGraphSize, 
							   maxFilteredTransitions, gthread, curTime);

		cudaDeviceSynchronize();
		cudaEventRecord(end,0);
		checkForCudaError();

		// copy the result from GPU back to CPU
		cudaMemcpy(&forwardMatrix[gtind(curTime,toff,0,0,0)],d_forwardValues[forw_to],s_forwardValues, 
				   cudaMemcpyDeviceToHost); checkForCudaError();

		// save kernel time
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, end);
		gpuKernelTime += milliseconds;
		nrOfKernelCalls++;

		// prepare for the next state
		std::fill_n(hasStateBeenProcessedBefore, toff, false);
		std::fill_n(allowedParentStates, toff, false);
		std::fill_n(filtered_transitions, l_filtered_transitions, 0);
		std::fill_n(curChars, gthread, 'N');

		//find the most likely states that should be allowed to make the next transitions
		for(auto gt = 0; gt < gthread; ++gt){
			if(!fbMatrixSize[gt] || readLength[gt] < curTime) continue;
			curTrSet[gt]->clear();

			findMaxValues(&forwardMatrix[gtind(curTime,toff,gt,maxGraphSize,0)],&allowedParentStates[gt*maxGraphSize],0,
						  fbMatrixSize[gt],graph->params.filterSize);
			tmpTrSet[gt] = curTrSet[gt];
			curTrSet[gt] = nextTrSet[gt];
			nextTrSet[gt] = tmpTrSet[gt];
		}

		//switching between two arrays in the GPU in order to minimize data movement
		forw_from = forw_to;
		forw_to = (forw_to+1)%2;
	}

	// end the timemeasure for calculations
	std::int64_t calculationTime = currentTimeMillis() - calculationStartTime;

	// free memory on GPU
	if(d_forwardValues[0]) cudaFree(d_forwardValues[0]); checkForCudaError(); 
	if(d_forwardValues[1]) cudaFree(d_forwardValues[1]); checkForCudaError(); 
	if(d_curChars) cudaFree(d_curChars); checkForCudaError(); 
	if(d_readLength) cudaFree(d_readLength); checkForCudaError(); 
	if(d_startPosition) cudaFree(d_startPosition); checkForCudaError(); 
	if(d_seq) cudaFree(d_seq); checkForCudaError(); 
	if(d_filtered_transitions) cudaFree(d_filtered_transitions); checkForCudaError(); 
	if(d_lutTransition) cudaFree(d_lutTransition); checkForCudaError(); 
	if(d_lutEmission) cudaFree(d_lutEmission); checkForCudaError(); 

	// free memory on CPU
	if(filtered_transitions) delete[] filtered_transitions;
	if(allowedParentStates) delete[] allowedParentStates;
	if(hasStateBeenProcessedBefore) delete[] hasStateBeenProcessedBefore; 
	for(auto gt = 0; gt < gthread; ++gt){
		if(curTrSet[gt]) delete curTrSet[gt];
		if(nextTrSet[gt]) delete nextTrSet[gt];
	}

	//end the overall timemeasure
	std::int64_t overallTime = currentTimeMillis() - overallStartTime;
	printf("forward: Number of forward calculations in parallel:%d \tTotal time (everything)=%ds, \tData Preparation=%ds, \tCPU+GPU calculation=%ds, \tGPU Calculation (all forwards)=%fs, \tGPU Calculation (single forward)=%fs, \tAvg. Kernel Time (single timestamp)=%fs, \tNumOf Kernel Calls=%d\n",gthread, (double)overallTime/1000, (double)gpuPreparationTime/1000, (double)calculationTime/1000, (double)gpuKernelTime/1000, (double)(gpuKernelTime/gthread)/1000, (double)(gpuKernelTime/nrOfKernelCalls)/1000, nrOfKernelCalls);
}

void HMMTrainer::fillBackwardMatrix(prob_prec* backwardMatrix, ind_prec* fbMatrixSize, char** read,
									ind_prec* startPosition, ind_prec* maxDistanceOnContig, ind_prec* readLength, 
									cnt_prec chunkSize, uint64_t maxGraphSize, uint32_t gthread){

	ind_prec toff = gthread*maxGraphSize;
	// ind_prec maxFilteredMatrixSize = (maxGraphSize<graph->params.filterSize)?maxGraphSize:graph->params.filterSize;
	ind_prec maxFilteredTransitions=graph->params.filterSize*(graph->params.maxInsertion+1)*(graph->params.maxDeletion+1);

	prob_prec* lutTransition = graph->preCalculatedTransitionProbs;
	prob_prec lutEmission[] = {graph->params.matchEmission, graph->params.mismatchEmission, 0.0, 
							   graph->params.insertionEmission };

	std::vector<TransitionInfoNode>* curTrSet[gthread];	//transitions from the previous time
	std::vector<TransitionInfoNode>* nextTrSet[gthread]; //transitions to be processed for the next time
	std::vector<TransitionInfoNode>* tmpTrSet[gthread];

	for(auto gt = 0; gt < gthread; ++gt){
		curTrSet[gt] = new std::vector<TransitionInfoNode>;
		nextTrSet[gt] = new std::vector<TransitionInfoNode>;
	}

   	char curChars[gthread]; std::fill_n(curChars, gthread, 'N');
	bool* allowedParentStates = new bool[toff]; std::fill_n(allowedParentStates, toff, false);
	bool* hasStateBeenProcessedBefore = new bool[toff]; std::fill_n(hasStateBeenProcessedBefore, toff, false);

	//1-initialization
	ind_prec curTime = chunkSize-1;

	for(auto gt = 0; gt < gthread; ++gt){

		if(!fbMatrixSize[gt]) continue;

		insertNewBackwardTransitions(curTrSet[gt], graph->seqGraph[startPosition[gt]], graph->params.maxDeletion,
									 graph->params.maxInsertion);

		for(size_t curTransition = 0; curTransition < curTrSet[gt]->size(); ++curTransition){
			TransitionInfoNode& frontTr = curTrSet[gt]->at(curTransition);
			int from_ind = frontTr.from - maxDistanceOnContig[gt];
			int to_ind = frontTr.toState - maxDistanceOnContig[gt];
			if(frontTr.toState > maxDistanceOnContig[gt]){
				ind_prec matchoff = MATCH_OFFSET(graph->seqGraph[frontTr.toState].getCharIndex(), 0, 
												 graph->params.maxInsertion);
				//0->insertion, 1-> match, 2,3...->deletions
				ind_prec transitionIndex = (frontTr.from - matchoff)/(graph->params.maxInsertion+1);
				//@IMPORTANT: check maxPrec here.

				backwardMatrix[gtind(curTime, toff, gt, maxGraphSize, to_ind)] += lutTransition[transitionIndex];

				insertNewBackwardTransitions(nextTrSet[gt], graph->seqGraph[frontTr.toState], graph->params.maxDeletion,
											 graph->params.maxInsertion);
			}
		}

		curTrSet[gt]->clear();

		findMaxValues(&backwardMatrix[gtind(curTime, toff, gt,maxGraphSize, 0)],&allowedParentStates[gt*maxGraphSize],0,
					  fbMatrixSize[gt],graph->params.filterSize);

		tmpTrSet[gt] = curTrSet[gt];
		curTrSet[gt] = nextTrSet[gt];
		nextTrSet[gt] = tmpTrSet[gt];
	}

	/**
	* prepare the arrays for the GPU
	*/
	ind_prec l_filtered_transitions = gthread*maxFilteredTransitions;
	cnt_prec* filtered_transitions = new cnt_prec[l_filtered_transitions];
	std::fill_n(filtered_transitions, l_filtered_transitions, 0);
	cnt_prec nrOfValidTransitions[gthread]; // counts how many transitions are valid/used from the array

	/**
	* Backward-Calculations for each transition on GPU in parallel
	*/
	ind_prec s_backwardValues = toff*sizeof(prob_prec);
	ind_prec s_curChars = gthread*sizeof(char);
	ind_prec s_readLength = gthread*sizeof(ind_prec);
	ind_prec s_startPosition = gthread*sizeof(ind_prec);
	ind_prec s_seq = graph->contigLength*sizeof(char);
	ind_prec s_filtered_transitions = l_filtered_transitions*sizeof(cnt_prec);
	ind_prec s_lutTransition = graph->numOfTransitionsPerState*sizeof(prob_prec);
	ind_prec s_lutEmission  = 4*sizeof(prob_prec);

	// start time measure for the preparation of the gpu data
	std::int64_t overallStartTime = currentTimeMillis();
	float gpuKernelTime = 0;
	int nrOfKernelCalls = 0;

	// prepare for the GPU kernel time timemeasure
	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	// Allocate the device input vectors
	ind_prec back_from = 0;
	ind_prec back_to = 1;
	prob_prec* d_backwardValues[] = {NULL, NULL};
	cudaMalloc((void**)&d_backwardValues[back_from], s_backwardValues); checkForCudaError();
	cudaMalloc((void**)&d_backwardValues[back_to], s_backwardValues); checkForCudaError();

	char* d_curChars = NULL;
	cudaMalloc((void**)&d_curChars, s_curChars); checkForCudaError();

	ind_prec* d_readLength = NULL;
	cudaMalloc((void**)&d_readLength, s_readLength); checkForCudaError();

	ind_prec* d_startPosition = NULL;
	cudaMalloc((void**)&d_startPosition, s_startPosition); checkForCudaError();

	char* d_seq = NULL;
	cudaMalloc((void**)&d_seq, s_seq); checkForCudaError();

	cnt_prec* d_filtered_transitions = NULL;
	cudaMalloc((void**)&d_filtered_transitions, s_filtered_transitions); checkForCudaError();

	prob_prec* d_lutTransition = NULL;
	cudaMalloc((void**)&d_lutTransition, s_lutTransition); 
	checkForCudaError();

	prob_prec* d_lutEmission = NULL;
	cudaMalloc((void**)&d_lutEmission, s_lutEmission); checkForCudaError();

	// copy arrays to GPU that remain constant at all times
	cudaMemcpy(d_readLength, readLength, s_readLength, cudaMemcpyHostToDevice); checkForCudaError();
	cudaMemcpy(d_startPosition, maxDistanceOnContig, s_startPosition, cudaMemcpyHostToDevice); checkForCudaError();
	cudaMemcpy(d_seq, graph->contigChar, s_seq, cudaMemcpyHostToDevice); checkForCudaError();
	cudaMemcpy(d_lutTransition, lutTransition, s_lutTransition, cudaMemcpyHostToDevice); checkForCudaError();
	cudaMemcpy(d_lutEmission, lutEmission, s_lutEmission, cudaMemcpyHostToDevice); checkForCudaError();

	// end timemeasure for the gpu preparation an start it for the calculations
	std::int64_t gpuPreparationTime = currentTimeMillis() - overallStartTime;
	std::int64_t calculationStartTime = currentTimeMillis();

	curTime = chunkSize-1;
	// calculate the backward values for each time
	while (curTime > 0){

		curTime--;
		for(auto gt = 0; gt < gthread; ++gt){
			if(fbMatrixSize[gt] == 0 || curTrSet[gt]->empty()) continue;

			nrOfValidTransitions[gt] = 0;
			curChars[gt] = read[gt][curTime+1];
			auto gt_off = gt*maxGraphSize;
			for(int currentTransition = 0; currentTransition < curTrSet[gt]->size(); ++currentTransition){
				TransitionInfoNode& frontTr = curTrSet[gt]->at(currentTransition);
				int from_ind = frontTr.from - maxDistanceOnContig[gt];
				int to_ind = frontTr.toState - maxDistanceOnContig[gt];
				if(from_ind <= 0 || to_ind <= 0 || to_ind >= fbMatrixSize[gt]) continue;

				// only add valid transitions to the transitionSet for the GPU
				if(allowedParentStates[gt_off+from_ind]){

					if(!hasStateBeenProcessedBefore[gt_off+to_ind]&&nrOfValidTransitions[gt]<maxFilteredTransitions){

						filtered_transitions[gt*maxFilteredTransitions + nrOfValidTransitions[gt]++] = to_ind;
						insertNewBackwardTransitions(nextTrSet[gt], graph->seqGraph[frontTr.toState], 
													 graph->params.maxDeletion,
													 graph->params.maxInsertion);
						hasStateBeenProcessedBefore[gt_off+to_ind] = true;
					}
				}
			}
		}

		cudaMemcpy(d_backwardValues[back_from],&backwardMatrix[gtind((curTime+1),toff,0,0,0)],s_backwardValues,
				   cudaMemcpyHostToDevice); checkForCudaError();
		cudaMemset(d_backwardValues[back_to], 0, s_backwardValues); checkForCudaError();

		cudaMemcpy(d_curChars, curChars, s_curChars, cudaMemcpyHostToDevice);
		checkForCudaError();

		cudaMemcpy(d_filtered_transitions, filtered_transitions, s_filtered_transitions, cudaMemcpyHostToDevice);
		checkForCudaError();

		//Launch the calculations on the CUDA Kernel
		int threadsPerBlock = 256;
		ind_prec blocksPerGrid = (l_filtered_transitions+threadsPerBlock-1)/threadsPerBlock;

		cudaEventRecord(start, 0);
		//here the start positions are the maxdistancontig (which gives the start positions from the forward direction)
		//here we allocate more threads than the GPU's overall threads so we should run in batches here.
		gpu_backwardCalculation(blocksPerGrid, threadsPerBlock, d_backwardValues[back_from], d_backwardValues[back_to], 
							   d_curChars,d_readLength,d_startPosition,d_seq,d_filtered_transitions,d_lutTransition, 
							   d_lutEmission, graph->params.maxInsertion, graph->params.maxDeletion, maxGraphSize, 
							   maxFilteredTransitions, gthread, curTime);

		cudaDeviceSynchronize();
		cudaEventRecord(end,0);
		checkForCudaError();

		// copy the result from GPU back to CPU
		cudaMemcpy(&backwardMatrix[gtind(curTime,toff,0,0,0)],d_backwardValues[back_to],s_backwardValues,
				   cudaMemcpyDeviceToHost); checkForCudaError();

		// save kernel time    
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, end);
		gpuKernelTime += milliseconds;
		nrOfKernelCalls++;

		// prepare for the next state
		std::fill_n(hasStateBeenProcessedBefore, toff, false);
		std::fill_n(allowedParentStates, toff, false);
		std::fill_n(filtered_transitions, l_filtered_transitions, 0);
		std::fill_n(curChars, gthread, 'N');

		//find the most likely states that should be allowed to make the next transitions
		for(auto gt = 0; gt < gthread; ++gt){
			if(!fbMatrixSize[gt]) continue;
			curTrSet[gt]->clear();

			findMaxValues(&backwardMatrix[gtind(curTime,toff,gt,maxGraphSize,0)],&allowedParentStates[gt*maxGraphSize],
						  0,fbMatrixSize[gt],graph->params.filterSize);
			tmpTrSet[gt] = curTrSet[gt];
			curTrSet[gt] = nextTrSet[gt];
			nextTrSet[gt] = tmpTrSet[gt];
		}

		//switching between two arrays in the GPU in order to minimize data movement
		back_from = back_to;
		back_to = (back_to+1)%2;
	}

	// end the timemeasure for the calculations
	std::int64_t calculationTime = currentTimeMillis() - calculationStartTime;

	
	// free memory on GPU
	if(d_backwardValues[0]) cudaFree(d_backwardValues[0]); checkForCudaError(); 
	if(d_backwardValues[1]) cudaFree(d_backwardValues[1]); checkForCudaError(); 
	if(d_curChars) cudaFree(d_curChars); checkForCudaError(); 
	if(d_readLength) cudaFree(d_readLength); checkForCudaError(); 
	if(d_startPosition) cudaFree(d_startPosition); checkForCudaError(); 
	if(d_seq) cudaFree(d_seq); checkForCudaError(); 
	if(d_filtered_transitions) cudaFree(d_filtered_transitions); checkForCudaError(); 
	if(d_lutTransition) cudaFree(d_lutTransition); checkForCudaError(); 
	if(d_lutEmission) cudaFree(d_lutEmission); checkForCudaError(); 
	
	if(filtered_transitions != NULL) delete[] filtered_transitions;
	if(allowedParentStates != NULL) delete[] allowedParentStates;
	if(hasStateBeenProcessedBefore != NULL) delete[] hasStateBeenProcessedBefore; 
	for(auto gt = 0; gt < gthread; ++gt){
		if(curTrSet[gt]) delete curTrSet[gt];
		if(nextTrSet[gt]) delete nextTrSet[gt];
	}

	// end the overall timemeasure
	std::int64_t overallTime = currentTimeMillis() - overallStartTime;
	printf("backward: Number of backward calculations in parallel:%d \tTotal time (everything)=%ds, \tData Preparation=%ds, \tCPU+GPU calculation=%ds, \tGPU Calculation (all backwards)=%fs, \tGPU Calculation (single backward)=%fs, \tAvg. Kernel Time (single timestamp)=%fs, \tNumOf Kernel Calls=%d\n", gthread, (double)overallTime/1000, (double)gpuPreparationTime/1000, (double)calculationTime/1000, (double)gpuKernelTime/1000, (double)(gpuKernelTime/gthread)/1000, (double)(gpuKernelTime/nrOfKernelCalls)/1000, nrOfKernelCalls);
}

void HMMTrainer::checkForCudaError() {
	cudaError_t cudaError = cudaGetLastError();
		if(cudaError != cudaSuccess) {
			fprintf(stderr, "There was a cuda error! error code: %s\n", cudaGetErrorString(cudaError));
		exit(EXIT_FAILURE);
	}
}

std::int64_t HMMTrainer::currentTimeMillis() {
	return std::chrono::duration_cast<std::chrono::milliseconds>(
			std::chrono::system_clock::now().time_since_epoch()).count();
}