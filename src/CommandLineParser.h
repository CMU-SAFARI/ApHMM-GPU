/** @file main.cpp
* @brief One sentence brief
*
* More details
* In multiple lines
* Copyright © 2020 SAFARI
*
* @author Can Firtina
* @bug No known bug
*/

#ifndef COMMAND_LINE_PARSER_H_
#define COMMAND_LINE_PARSER_H_

#include <stdio.h>
#include <iostream>
#include <vector>
#include <seqan/arg_parse.h>

/** @brief Struct for holding command line options and their values.
 *
 *  @see parseCommandOptions()
 *  @see parseInitialCommand()
 */
struct CommandLineParser{
    
    CommandLineParser():
    filterSize(100), viterbiFilterSize(5), maxDeletion(10), maxInsertion(3), batchSize(5000), chunkSize(1000),
    matchTransition(0.85), insertionTransition(0.1), shouldQuite(false), matchEmission(0.97), maxThread(1), mapQ(0),
    deletionTransitionFactor(2.5){}

    unsigned filterSize;
    unsigned viterbiFilterSize;
    unsigned maxDeletion;
    unsigned maxInsertion;
    unsigned batchSize;
    unsigned chunkSize;
    double matchTransition;

    double insertionTransition;
    bool shouldQuite;
    double matchEmission;
    unsigned maxThread;
    unsigned gpuThread;
    unsigned mapQ;
    double deletionTransitionFactor;

    seqan::CharString assembly;
    std::vector<seqan::CharString > alignmentSets;
    std::vector<seqan::CharString > readSets;
    seqan::CharString output;
};

/** @brief Parse values in order to run either preprocessing step or correction step.
 *
 *  @param options Stores parsed values
 *  @param argc Number of arguments specified while running Apollo
 *  @param argv Argument values array
 *  @return seqan::ArgumentParser::PARSE_OK if everything went well
 */
seqan::ArgumentParser::ParseResult
parseCommandOptions(CommandLineParser& options, int argc, char const **argv){

    using namespace std;
    seqan::ArgumentParser parser("Apollo: A Sequencing-Technology-Independent, Scalable, and Accurate Assembly Polishing Algorithm");

    setVersion(parser, "2.0");
    setDate(parser, "May 2020");


    addOption(parser, seqan::ArgParseOption("a", "assembly", "The fasta file which contains the assembly",
                                            seqan::ArgParseArgument::INPUT_FILE, "FILE", false));
    setRequired(parser, "assembly");

    addOption(parser, seqan::ArgParseOption("r", "read", "A fasta file which contains a set of reads that are "
                                            "aligned to the assembly.", seqan::ArgParseArgument::INPUT_FILE, "FILE", true));
    setRequired(parser, "read");

    addOption(parser, seqan::ArgParseOption("m", "alignment", "{s,b}am file which contains alignments of the set of reads"
                                            " to the assembly.", seqan::ArgParseArgument::INPUT_FILE, "FILE", true));
    setRequired(parser, "alignment");

    addOption(parser, seqan::ArgParseOption("o", "output", "Output file to write the polished (i.e., corrected) assembly.",
                                            seqan::ArgParseArgument::OUTPUT_FILE, "FILE", false));
    setRequired(parser, "output");

    addOption(parser, seqan::ArgParseOption("q", "mapq", "Minimum mapping quality for a read-to-assembly alignment "
                                            "to be used in assembly polishing. Note that if the aligner reports multiple"
                                            "alignmentsvfor a read, then it may be setting mapping qualities of multiple"
                                            " alignments as 0.",
                                            seqan::ArgParseArgument::INTEGER, "INT", false));
    setDefaultValue(getOption(parser, "mapq"), options.mapQ);
    seqan::setMinValue(parser, "mapq", "0");
    seqan::setMaxValue(parser, "mapq", "255");

    addOption(parser, seqan::ArgParseOption("f", "filter", "Filter size that allows calculation of at most \"f\" "
                                            "many most probable transitions in each time step. This parameter is "
                                            "directly proportional to running time.",
                                            seqan::ArgParseArgument::INTEGER, "INT", false));
    setDefaultValue(getOption(parser, "filter"), options.filterSize);
    seqan::setMinValue(parser, "filter", "1");

    addOption(parser, seqan::ArgParseOption("v", "viterbi-filter", "Filter size for the Viterbi algorithm that allows calculation"
                                            " of at most \"vf\" many most probable states in each time step. This "
                                            "parameter is directly proportional to running time.",
                                            seqan::ArgParseArgument::INTEGER, "INT", false));
    setDefaultValue(getOption(parser, "viterbi-filter"), options.viterbiFilterSize);
    seqan::setMinValue(parser, "viterbi-filter", "1");

    addOption(parser, seqan::ArgParseOption("i", "maxi", "Maximum number of insertions in a row. This "
                                            "parameter is directly proportional to the running time.",
                                            seqan::ArgParseArgument::INTEGER, "INT", false));
    setDefaultValue(getOption(parser, "maxi"), options.maxInsertion);
    seqan::setMinValue(parser, "maxi", "0");

    addOption(parser, seqan::ArgParseOption("d", "maxd", "Maximum number of deletions in a row. This "
                                            "parameter is directly proportional to the running time.",
                                            seqan::ArgParseArgument::INTEGER, "INT", false));
    setDefaultValue(getOption(parser, "maxd"), options.maxDeletion);
    seqan::setMinValue(parser, "maxd", "0");

    addOption(parser, seqan::ArgParseOption("tm", "mtransition", "Initial transition probability to a match "
                                            "state. See --itransition as well.",
                                            seqan::ArgParseArgument::DOUBLE, "FLOAT", false));
    setDefaultValue(getOption(parser, "mtransition"), options.matchTransition);
    seqan::setMinValue(parser, "mtransition", "0");
    seqan::setMaxValue(parser, "mtransition", "1");

    addOption(parser, seqan::ArgParseOption("ti", "itransition", "Initial transition probability to an "
                                            "insertion state. Note that the deletion transition probability equals to: "
                                            "(1 - (matchTransition + insertionTransition)).",
                                            seqan::ArgParseArgument::DOUBLE, "FLOAT", false));
    setDefaultValue(getOption(parser, "itransition"), options.insertionTransition);
    seqan::setMinValue(parser, "itransition", "0");
    seqan::setMaxValue(parser, "itransition", "1");

    addOption(parser, seqan::ArgParseOption("df", "dfactor", "Factor for the polynomial distribution to calculate the "
                                            "each of the probabilities to delete 1 to \"d\" many basepairs. Note that "
                                            "unless \"df\" is set 1, the probability of the deleting k many characters "
                                            "will always going to be different than deleting n many characters where "
                                            "0<k<n<\"d\". A higher \"df\" value favors less deletions.",
                                            seqan::ArgParseArgument::DOUBLE, "FLOAT", false));
    setDefaultValue(getOption(parser, "dfactor"), options.deletionTransitionFactor);
    seqan::setMinValue(parser, "dfactor", "0.001");

    addOption(parser, seqan::ArgParseOption("em", "memission", "Initial emission probability of a match to a "
                                            "reference. Note that: mismatch emission probability equals to: "
                                            "((1-matchEmission)/3).", seqan::ArgParseArgument::DOUBLE, "FLOAT", false));
    setDefaultValue(getOption(parser, "memission"), options.matchEmission);
    seqan::setMinValue(parser, "memission", "0");
    seqan::setMaxValue(parser, "memission", "1");

    addOption(parser, seqan::ArgParseOption("b", "batch", "Number of consecutive basepairs that Viterbi decodes per "
                                            "thread. Setting it to zero will decode the entire contig with a single thread.",
                                            seqan::ArgParseArgument::INTEGER, "INT", false));
    setDefaultValue(getOption(parser, "batch"), options.batchSize);
    seqan::setMinValue(parser, "batch", "0");
    
    addOption(parser, seqan::ArgParseOption("c", "chunk", "If a read is longer than --chunk, it will be divided "
                                            "(i.e., chunked) into multiple shorter reads of size --chunk. Helps to "
                                            "reduce overall memory usage. Set this to 0 if you do not want chunking.",
                                            seqan::ArgParseArgument::INTEGER, "INT", false));
    setDefaultValue(getOption(parser, "chunk"), options.chunkSize);
    seqan::setMinValue(parser, "chunk", "0");

    addOption(parser, seqan::ArgParseOption("t", "thread", "Maximum number of threads to use.",
                                            seqan::ArgParseArgument::INTEGER, "INT", false));
    setDefaultValue(getOption(parser, "thread"), options.maxThread);
    seqan::setMinValue(parser, "thread", "1");

    addOption(parser, seqan::ArgParseOption("g", "gthread", "Maximum number of GPU threads to use.",
                                            seqan::ArgParseArgument::INTEGER, "INT", false));
    setDefaultValue(getOption(parser, "gthread"), options.maxThread);
    seqan::setMinValue(parser, "gthread", "1");

    addOption(parser, seqan::ArgParseOption("n", "no-verbose", "Apollo runs quitely with no informative output"));

    seqan::ArgumentParser::ParseResult res = seqan::parse(parser, argc, argv);
    if (res == seqan::ArgumentParser::PARSE_OK){

        getOptionValue(options.assembly, parser, "a");

        unsigned readCount = seqan::getOptionValueCount(parser, "r");
        const std::vector<std::string> readOptionValues = getOptionValues(parser, "r");
        for(unsigned i = 0; i < readCount; ++i) options.readSets.push_back(readOptionValues.at(i));

        unsigned alignmentCount = seqan::getOptionValueCount(parser, "m");
        if(readCount != alignmentCount){
            std::cerr << "ERROR: Number of the read sets provided is not equal to the number of alignment sets " << std::endl;
            return seqan::ArgumentParser::PARSE_ERROR;
        }
        const std::vector<std::string> alignmentOptionValues = getOptionValues(parser, "m");
        for(unsigned i = 0; i < alignmentCount; ++i) options.alignmentSets.push_back(alignmentOptionValues.at(i));

        getOptionValue(options.output, parser, "o");
        getOptionValue(options.mapQ, parser, "q");
        getOptionValue(options.chunkSize, parser, "c");
	    getOptionValue(options.filterSize, parser, "f");
        getOptionValue(options.viterbiFilterSize, parser, "v");
        getOptionValue(options.maxInsertion, parser, "i");
        getOptionValue(options.maxDeletion, parser, "d");
        getOptionValue(options.matchTransition, parser, "tm");
        getOptionValue(options.insertionTransition, parser, "ti");
        getOptionValue(options.deletionTransitionFactor, parser, "df");
        getOptionValue(options.matchEmission, parser, "em");
        getOptionValue(options.batchSize, parser, "b");
        getOptionValue(options.maxThread, parser, "t");
        getOptionValue(options.gpuThread, parser, "g");
        options.shouldQuite = isSet(parser, "n");

        if(options.matchTransition + options.insertionTransition > 1){
            std::cerr << "ERROR: (matchTransition + insertionTransition) cannot be larger than 1 but the sum "
            << "is now: " << options.matchTransition + options.insertionTransition << std::endl;
            return seqan::ArgumentParser::PARSE_ERROR;
        }
    }

    return res;
}

#endif
