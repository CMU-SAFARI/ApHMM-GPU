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
#ifndef HMMTrainer_h
#define HMMTrainer_h

#include <stdio.h>
#include "HMMGraph.h"

class HMMTrainer{
    
    void fbThreadPool(const std::vector<Read>& reads, cnt_prec chunkSize, unsigned thread, unsigned gthread);
    
    void fillForwardMatrix(prob_prec* forwardMatrix,ind_prec* fbMatrixSize,char** read, ind_prec* startPosition,
                           ind_prec* maxDistanceOnContig, ind_prec* readLength, cnt_prec chunkSize, 
                           uint64_t maxGraphSize, uint32_t gthread);
    
    void fillBackwardMatrix(prob_prec* backwardMatrix,ind_prec* fbMatrixSize,char** read, ind_prec* startPosition,
                           ind_prec* maxDistanceOnContig, ind_prec* readLength, cnt_prec chunkSize, 
                           uint64_t maxGraphSize, uint32_t gthread);
    
    void maximizeEMCPU(int firstState, int lastState);
    void checkForCudaError();
    std::int64_t currentTimeMillis();

    HMMGraph* graph;
    std::mutex indexMutex;
    std::mutex emissionProbMutex;
    std::mutex transitionProbMutex;
    
public:
    HMMTrainer();
    HMMTrainer(HMMGraph* graph);
    ~HMMTrainer();
    
    void calculateFB(std::vector<seqan::BamFileIn>& alignmentSetsIn, std::vector<seqan::BamAlignmentRecord>& curRecords,
                     const std::vector<seqan::FaiIndex>& readFAIs, unsigned thread, unsigned gthread);
    void maximizeEM();
};
#endif /* HMMTrainer_h */
