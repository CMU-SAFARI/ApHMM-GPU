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
#include "HMMTrainer.cuh"

// things for GPU
// #include <cuda_runtime.h>
// #include <helper_cuda.h>

// typedef double prob_prec; //precision of the probabilities
// typedef int cnt_prec; //precision of the count arrays -- relative to the avg. depth of coverage
// typedef unsigned ind_prec; //precision of the size of contigs -- relative to genome size

// struct TransitionInfoNode{

//     ind_prec from, toState;
//     TransitionInfoNode(ind_prec from, ind_prec toState):from(from), toState(toState){}

//     bool operator==(const TransitionInfoNode& rhs) const{
//         return from == rhs.from && toState == rhs.toState;
//     }

//     bool operator<(const TransitionInfoNode& rhs) const{
//         return ((from < rhs.from) || (from == rhs.from && (toState < rhs.toState)));
//     }
// };

// typedef double prob_prec; //precision of the probabilities
// typedef int cnt_prec; //precision of the count arrays -- relative to the avg. depth of coverage
// typedef unsigned ind_prec; //precision of the size of contigs -- relative to genome size
// typedef std::vector<TransitionInfoNode> transition_vector;

/**
* CUDA Kernel Device code for the calculation of the forward matrix
*
* constantIntegers = { startPosition, maxInsertion, charFromRead_01, charFromRead_02 }
* lutEmission = { matchEmissionProbability, substitutionEmissionProbability, 0
* 						, insertionEmissionProbability }
*
* transitions = { ..., {j.x, j.y}, ... }
* lutTransition = { insertionProb, matchProb, deletionProb01, deletionProb02, ... }
*
* seq = { ..., contigJ01, contigJ02, ... } with ( A, T, G, C ) = ( 00, 01, 10, 11 )
*/

typedef double prob_prec; //precision of the probabilities
typedef int cnt_prec; //precision of the count arrays -- relative to the avg. depth of coverage
typedef unsigned ind_prec; //precision of the size of contigs -- relative to genome size

__global__ void
forwardCalculationGPU(const prob_prec* forwardValuesFrom, prob_prec* forwardValuesTo, const char* curObs,
					  const ind_prec* readLength, const ind_prec* startPosition, const char* seq, 
					  const cnt_prec* filtered_transitions,const prob_prec* lutTransition,const prob_prec* lutEmission,
					  const cnt_prec maxInsertion, const cnt_prec maxDel, const ind_prec maxGraphSize, 
					  const ind_prec maxFilteredTransitions, const unsigned gthread, const unsigned time) {

	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int gt = i/maxFilteredTransitions;

	// if transition is valid, calculate forward value
	if(gt < gthread && i < gthread*maxFilteredTransitions && filtered_transitions[i] > 0 && time < readLength[gt] && curObs[gt] != 'N'){

		char obs = curObs[gt];
		int numStates = 1+maxInsertion;
		int startoff = startPosition[gt]/numStates;

		int to_state = filtered_transitions[i];
		int matchoffTo = to_state/numStates;

		bool toStateIsMatchstate = to_state%numStates == 0;
		bool readIsEqualToContig = seq[matchoffTo+startoff] == obs;
		bool readIsEqualToNextContig = seq[matchoffTo+startoff+1] == obs;
		bool readMatchOffset = toStateIsMatchstate?readIsEqualToContig:readIsEqualToNextContig;
		int emInd = toStateIsMatchstate?0:2;
		emInd = readMatchOffset?(emInd):(emInd+1);
		prob_prec emissionProb = lutEmission[emInd];
		int range = toStateIsMatchstate?((matchoffTo-maxDel-1 > 0)?((matchoffTo-maxDel-1)*numStates):0):to_state-1;

		prob_prec deltaForw = 0;

		for(int ind = to_state-1; ind >= range; --ind)
			deltaForw += forwardValuesFrom[gt*maxGraphSize+ind] *
						 lutTransition[to_state/numStates - ind/numStates]*emissionProb;

		forwardValuesTo[gt*maxGraphSize+to_state] = deltaForw;
	}
}

/**
 * CUDA Kernel Device code for the calculation of the backward matrix
 *
 * constantIntegers = { maxDistanceOnContig, maxInsertion, nextCharFromRead_01, nextCharFromRead_02 }
 * lutEmission = { matchEmissionProbability, substitutionEmissionProbability, 0
 * 						, insertionEmissionProbability }
 *
 * transitions = { ..., {j.x, j.y}, ... }
 * lutTransition = { insertionProb, matchProb, deletionProb01, deletionProb02, ... }
 *
 * seq = { ..., contigJ01, contigJ02, ... } with ( A, T, G, C ) = ( 00, 01, 10, 11 )
 */
__global__ void
backwardCalculationGPU(const prob_prec* backwardValuesFrom, prob_prec* backwardValuesTo, const char* curObs,
					  const ind_prec* readLength, const ind_prec* startPosition, const char* seq, 
					  const cnt_prec* filtered_transitions,const prob_prec* lutTransition,const prob_prec* lutEmission,
					  const cnt_prec maxInsertion, const cnt_prec maxDel, const ind_prec maxGraphSize, 
					  const ind_prec maxFilteredTransitions, const unsigned gthread, const unsigned time) {

 	int i = blockDim.x * blockIdx.x + threadIdx.x;
 	int gt = i/maxFilteredTransitions;
 	
 	// if transition is valid, calculate backward value
 	if(gt < gthread && i < gthread*maxFilteredTransitions && filtered_transitions[i] > 0 && time < readLength[gt] && curObs[gt] != 'N'){

 		char obs = curObs[gt];
 		int numStates = 1+maxInsertion;
 		int startoff = startPosition[gt]/numStates;

 		// correct the transitions
 		int to_state = filtered_transitions[i];
 		int matchoffTo = to_state/numStates;

 		int range = (matchoffTo+maxDel+1 < readLength[gt])?matchoffTo+maxDel+1:readLength[gt];
 		range *= numStates;

 		int ind = to_state+1;
 		int matchoffFrom = ind/numStates;
		bool fromStateIsMatchstate = ind%numStates == 0;
		bool readIsEqualToContig = seq[matchoffFrom+startoff] == obs;
		bool readIsEqualToNextContig = seq[matchoffFrom+startoff+1] == obs;
		bool readMatchOffset = fromStateIsMatchstate?readIsEqualToContig:readIsEqualToNextContig;
		int emInd = fromStateIsMatchstate?0:2;
		emInd = readMatchOffset?(emInd):(emInd+1);

		prob_prec deltaBack = backwardValuesFrom[gt*maxGraphSize+ind] *
					 		  lutTransition[matchoffFrom - matchoffTo]*lutEmission[emInd];

 		for(ind = (matchoffFrom+1)*numStates; ind <= range; ind+=numStates){

 			matchoffFrom = ind/numStates;
			emInd = (seq[matchoffFrom+startoff] == obs)?0:1;

			deltaBack += backwardValuesFrom[gt*maxGraphSize+ind] *
						 lutTransition[matchoffFrom - matchoffTo]*lutEmission[emInd];
 		}

 		backwardValuesTo[gt*maxGraphSize+to_state] = deltaBack;
 	}
}

/**
 * CUDA Kernel Device code for the maximizeEM step
 *
 * constantIntegers = { maxInsertion, numOfTransitionsPerState, lastStateGPU }
 * constantDoubles = { graph->params.matchTransition, graph->params.insertionTransition }
 * defaultEmissionProbs = { matchEmissionProbability, substitutionEmissionProbability, 0
 * 						, insertionEmissionProbability }
 *
 * transitions = { ..., {j.x, j.y}, ... }
 * transitionProbabilities = { insertionProb, matchProb, deletionProb01, deletionProb02, ... }
 * bestEmissionProbPair = { emissionProbability, emissionCharacter }
 *
 * ( A, T, G, C ) = ( 00, 01, 10, 11 ) = ( 0, 1, 2, 3)
 *
 * return: new transitionProbabilities and the best emissionProbability with emissionCharacter
 */
__global__ void
maximizeEMGPU(double *transitionProbs, const double *emissionProbs,
			  const double *preCalculatedLogTransitionProbs, const int *constantIntegers,
			  const double *constantDoubles, const double *defaultEmissionProbs, 
			  const int *stateProcessedCount, const int *transitionProcessedCount,
			  double *bestEmissionProbPairs, const bool *bitVector) {

	// the state index
	int currentState = blockDim.x * blockIdx.x + threadIdx.x;
	int arrayIndex = currentState * constantIntegers[1];

	if(currentState <= constantIntegers[2] && currentState > constantIntegers[0]+1){

		// true if currentState+1 is match state
		bool isLastInsertionState = (currentState+1) % (constantIntegers[0]+1) == 0;
		//FIXED: lastInsertionTransitionResetProb is not used 
		// double lastInsertionTransitionResetProb = preCalculatedLogTransitionProbs[1];
		//QUESTION: Can we pass this as a parameter because this is also constant?
		//FIXED: commented lastInsertionTransitionProb
		// double lastInsertionTransitionProb = log10(constantDoubles[0] + constantDoubles[1]);

		double preCalculatedLogTransitionProbs01 = preCalculatedLogTransitionProbs[1];
		int currentTransition = 0;

		if(isLastInsertionState) {
			preCalculatedLogTransitionProbs01 = log10(constantDoubles[0] + constantDoubles[1]);
			transitionProbs[arrayIndex + 0] = -21474836; // std::numeric_limits<int>::min()/100

			currentTransition = 1;
		}

		// only these transitions will update the transition probabilities
		bool stateProcessedCountIsZero = stateProcessedCount[currentState] <= 0;

		while(currentTransition < constantIntegers[1]) {
			bool transitionProcessedCountNotZero = transitionProcessedCount[arrayIndex + currentTransition] > 0;
			bool updateLogTransitionProb = !stateProcessedCountIsZero && transitionProcessedCountNotZero;

			double preCalculatedLogProb = (currentTransition == 1) ? preCalculatedLogTransitionProbs01
					: preCalculatedLogTransitionProbs[currentTransition];
			//FIXED
			preCalculatedLogProb = (updateLogTransitionProb)?log10(transitionProbs[arrayIndex + currentTransition]
					/ transitionProcessedCount[arrayIndex + currentTransition]):preCalculatedLogProb;
			// double newTransitionProb = updateLogTransitionProb * log10(transitionProbs[arrayIndex + currentTransition]
			// 		/ transitionProcessedCount[arrayIndex + currentTransition])
			// 		+ !updateLogTransitionProb * preCalculatedLogProb;
			transitionProbs[arrayIndex + currentTransition] = preCalculatedLogProb;

			currentTransition++;
		}

		bool stateIsMatchstate = (currentState % (1+constantIntegers[0])) == 0;
		int character;
		int matchStateIndex = currentState / (constantIntegers[0]+1) - 1;

		if(stateIsMatchstate) {
			// current statecharacter
			character = bitVector[2*matchStateIndex] ? 2 : 0;
			character += bitVector[2*matchStateIndex + 1] ? 1 : 0;
		} else {
			// next character on contig
			character = bitVector[2*(matchStateIndex+1)] ? 2 : 0;
			character += bitVector[2*(matchStateIndex+1) + 1] ? 1 : 0;
		}

		// emission probabilities
		double emissionProb00, emissionProb01, emissionProb02, emissionProb03;
		if(stateProcessedCountIsZero) {
			// how the following indices work:
			// bool readMatchOffset = toStateIsMatchstate ? readIsEqualToContig : readIsEqualToNextContig;
			// emissionProbability = constantDoubles[2 * !toStateIsMatchstate + !readMatchOffset];

			int matchstateOffset = stateIsMatchstate ? 0 : 2;
			int characterOffset[4];
			characterOffset[0] = 1;
			characterOffset[1] = 1;
			characterOffset[2] = 1;
			characterOffset[3] = 1;
			characterOffset[character] = 0;

			emissionProb00 = defaultEmissionProbs[matchstateOffset + characterOffset[0]];
			emissionProb01 = defaultEmissionProbs[matchstateOffset + characterOffset[1]];
			emissionProb02 = defaultEmissionProbs[matchstateOffset + characterOffset[2]];
			emissionProb03 = defaultEmissionProbs[matchstateOffset + characterOffset[3]];
		} else {
			emissionProb00 = emissionProbs[4*currentState + 0] / stateProcessedCount[currentState];
			emissionProb01 = emissionProbs[4*currentState + 1] / stateProcessedCount[currentState];
			emissionProb02 = emissionProbs[4*currentState + 2] / stateProcessedCount[currentState];
			emissionProb03 = emissionProbs[4*currentState + 3] / stateProcessedCount[currentState];
		}

		// find the maximum emissionProbability
		double maxEmissionProb; double maxEmissionChar;
		maxEmissionProb = emissionProb00; maxEmissionChar = 0;
		if(emissionProb01 > maxEmissionProb) {maxEmissionProb = emissionProb01; maxEmissionChar = 1;}
		if(emissionProb02 > maxEmissionProb) {maxEmissionProb = emissionProb02; maxEmissionChar = 2;}
		if(emissionProb03 > maxEmissionProb) {maxEmissionProb = emissionProb03; maxEmissionChar = 3;}

		// save the best emission probability and character
		bestEmissionProbPairs[2*currentState] = maxEmissionProb;
		bestEmissionProbPairs[2*currentState+1] = maxEmissionChar;
	}
}


void
gpu_forwardCalculation(int blocks, int threads, const prob_prec* forwardValuesFrom, prob_prec* forwardValuesTo, 
					   const char* curObs, const ind_prec* readLength, const ind_prec* startPosition, const char* seq, 
					   const cnt_prec* filtered_transitions,const prob_prec* lutTransition,const prob_prec* lutEmission,
					   const cnt_prec maxInsertion, const cnt_prec maxDel, const ind_prec maxGraphSize, 
					   const ind_prec maxFilteredTransitions, const unsigned gthread, const unsigned time){

	forwardCalculationGPU<<<blocks, threads>>>(forwardValuesFrom, forwardValuesTo, curObs, readLength, startPosition,
											   seq, filtered_transitions, lutTransition, lutEmission, maxInsertion,
											   maxDel, maxGraphSize, maxFilteredTransitions, gthread, time);
}

void
gpu_backwardCalculation(int blocks, int threads, const prob_prec* backwardValuesFrom, prob_prec* backwardValuesTo, 
					    const char* curObs, const ind_prec* readLength, const ind_prec* startPosition, const char* seq, 
					   const cnt_prec* filtered_transitions,const prob_prec* lutTransition,const prob_prec* lutEmission,
					    const cnt_prec maxInsertion, const cnt_prec maxDel, const ind_prec maxGraphSize, 
					    const ind_prec maxFilteredTransitions, const unsigned gthread, const unsigned time){

	backwardCalculationGPU<<<blocks, threads>>>(backwardValuesFrom, backwardValuesTo, curObs, readLength, startPosition,
											    seq, filtered_transitions, lutTransition, lutEmission, maxInsertion,
											    maxDel, maxGraphSize, maxFilteredTransitions, gthread, time);
}


void
gpu_maximizeEM(int blocks, int threads, double *transitionProbs, const double *emissionProbs,
              const double *preCalculatedLogTransitionProbs, const int *constantIntegers,
              const double *constantDoubles, const double *defaultEmissionProbs, 
              const int *stateProcessedCount, const int *transitionProcessedCount,
              double *bestEmissionProbPairs, const bool *bitVector){

	maximizeEMGPU<<<blocks, threads>>>(transitionProbs, emissionProbs, preCalculatedLogTransitionProbs, 
									   constantIntegers, constantDoubles, defaultEmissionProbs, 
									   stateProcessedCount, transitionProcessedCount, bestEmissionProbPairs, bitVector);

}
