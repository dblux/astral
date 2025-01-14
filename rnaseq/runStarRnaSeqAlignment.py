#!/usr/bin/env python3

import sys
import os
import string
import re
import argparse
import logging
import multiprocessing

"""
Script to run STAR RNA-Seq alignment process.
"""

# set up the logger
logger = logging.getLogger(os.path.basename(sys.argv[0]))

parser = argparse.ArgumentParser("Script to run STAR RNA-Seq alignment process.")
parser.add_argument("-o", "--output", action="store", dest="output", type=str, required=True, help="Output file basename.")
parser.add_argument("-r", "--reference", action="store", dest="reference", type=str, required=True, help="STAR reference index directory.")
parser.add_argument("-t", "--thread", action="store", dest="thread", type=int, default=multiprocessing.cpu_count()/2, help="Number of threads to use.")
parser.add_argument("-s", "--single", action="store_true", dest="single", help="Treat all FASTQ files as single ended.")
parser.add_argument("filenames", metavar="filenames", type=str, nargs='*', help="List of gzipped FASTQ files")
opt = parser.parse_args()

# set the logging levels
logger.setLevel(logging.DEBUG)

# set up the logging handlers
if os.path.exists("%s.log" % opt.output):
	os.unlink("%s.log" % opt.output)
fh = logging.FileHandler("%s.log" % opt.output)
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)

logger.info("CMD: %s" % " ".join(sys.argv))

# check that the reference directory exists
if not os.path.exists(opt.reference) or not os.path.isdir(opt.reference):
	logger.error("STAR reference directory %s does not exists." % opt.reference)
	sys.exit(1)

# if no files were specified, use all files in the current directory
if len(opt.filenames) == 0:
	filenames = list(filter(lambda x: os.path.isfile(x), os.listdir(".")))
else:
	filenames = list(filter(lambda x: os.path.isfile(x), opt.filenames))

# scan the filenames for the correct ones and arrange in sets (single or paired)
paired = {}
single = {}
for filename in filenames:
	# check whether there is a R? at the end of the filename
	mo = re.search("^(.+)_(R?\\d)\\.(fastq|fq)\\.gz$", filename)
	if mo != None and opt.single == False:
		(sample, pair, fq) = mo.groups()
		pair = pair.replace("R", "")
		if not sample in paired:
			paired[sample] = {}
		paired[sample][pair] = filename
	else:
		mo = re.search("^(.+)\\.(fastq|fq)\\.gz$", filename)
		if mo != None:
			(sample, fq) = mo.groups()
			single[sample] = filename
		else:
			logger.info("Unable to parse filename %s" % filename)

if len(paired) == 0 and len(single) == 0:
	logger.error("Unable to locate any FASTQ files to process.")
	sys.exit(1)
else:
	logger.info("Proceeding with %d FASTQ files: %s" % (len(filenames), ", ".join(filenames)))

# check the paired ends
psamples = list(filter(lambda x: not "1" in paired[x] or not "2" in paired[x], paired.keys()))
if len(psamples) > 0:
	logger.error("Not all the paired samples have 2 FASTQ files. Samples %s have problems." % ", ".join(psamples))
	sys.exit(1)

# print out the samples
logger.info("Found %d paired end samples" % len(paired))
logger.info("Found %d single end samples" % len(single))

# run the paired STAR runs
for sample in paired.keys():
	logger.info("Running STAR in paired end on sample %s" % sample)
	cmd = "STAR --runThreadN %d --genomeDir %s --readFilesIn %s %s --readFilesCommand gunzip -c --outFileNamePrefix %s_ --genomeLoad LoadAndKeep --outSAMtype BAM SortedByCoordinate --limitBAMsortRAM 30000000000" % (opt.thread, opt.reference, paired[sample]["1"], paired[sample]["2"], sample)
	os.system(cmd)

# run the single STAR runs
for sample in single.keys():
	logger.info("Running STAR in single end on sample %s" % sample)
	cmd = "STAR --runThreadN %d --genomeDir %s --readFilesIn %s --readFilesCommand gunzip -c --outFileNamePrefix %s_ --genomeLoad LoadAndKeep --outSAMtype BAM SortedByCoordinate --limitBAMsortRAM 30000000000" % (opt.thread, opt.reference, single[sample], sample)
	os.system(cmd)

# remove the reference from memory
logger.info("Removing genome from memory.")
cmd = "STAR --genomeDir %s --genomeLoad Remove --outFileNamePrefix removegenome" % opt.reference
os.system(cmd)
os.system("rm -rf removegenome*")

# rename the BAM file and index
for sample in list(single.keys()) + list(paired.keys()):
	logger.info("Indexing sample %s" % sample)
	cmd = "mv %s_Aligned.sortedByCoord.out.bam %s.bam" % (sample, sample)
	os.system(cmd)
	cmd = "samtools index %s.bam" % sample
	os.system(cmd)

logger.info("All done.")
