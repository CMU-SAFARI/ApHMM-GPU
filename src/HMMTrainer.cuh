/** @file HMMTrainer.h
* @brief One sentence brief
*
* More details
* In multiple lines
* Copyright © 2020 SAFARI
*
* @author Can Firtina
* @bug No known bug
*/
#ifndef HMMTrainer_cuh
#define HMMTrainer_cuh

#include <cuda_runtime.h>
#include <helper_cuda.h>

typedef double prob_prec; //precision of the probabilities
typedef int cnt_prec; //precision of the count arrays -- relative to the avg. depth of coverage
typedef unsigned ind_prec; //precision of the size of contigs -- relative to genome size

void
gpu_forwardCalculation(int blocks, int threads, const prob_prec* forwardValuesFrom, prob_prec* forwardValuesTo, 
                       const char* curObs, const ind_prec* readLength, const ind_prec* startPosition, const char* seq, 
                       const cnt_prec* filtered_transitions,const prob_prec* lutTransition,const prob_prec* lutEmission,
                       const cnt_prec maxInsertion, const cnt_prec maxDel, const ind_prec maxGraphSize, 
                       const ind_prec maxFilteredTransitions, const unsigned gthread, const unsigned time);

void
gpu_backwardCalculation(int blocks, int threads, const prob_prec* backwardValuesFrom, prob_prec* backwardValuesTo, 
                        const char* curObs, const ind_prec* readLength, const ind_prec* startPosition, const char* seq, 
                       const cnt_prec* filtered_transitions,const prob_prec* lutTransition,const prob_prec* lutEmission,
                        const cnt_prec maxInsertion, const cnt_prec maxDel, const ind_prec maxGraphSize, 
                        const ind_prec maxFilteredTransitions, const unsigned gthread, const unsigned time);


void
gpu_maximizeEM(int blocks, int threads, double *transitionProbs, const double *emissionProbs,
              const double *preCalculatedLogTransitionProbs, const int *constantIntegers,
              const double *constantDoubles, const double *defaultEmissionProbs, 
              const int *stateProcessedCount, const int *transitionProcessedCount,
              double *bestEmissionProbPairs, const bool *bitVector);

#endif /* HMMTrainer_h */

