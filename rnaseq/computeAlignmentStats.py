#!/usr/bin/env python

"""
Script to compute the alignment statistics.
"""

import os
import subprocess
import sys
import re
import argparse
import logging
import string

# set up the logger
logger = logging.getLogger(os.path.basename(sys.argv[0]))

# set up the parser
parser = argparse.ArgumentParser("Script to compute alignment statistics.")
parser.add_argument("-o", action="store", dest="output", type=str, required=True, help="Basename of the output files")
parser.add_argument("bams", metavar="BAM_file", type=str, nargs='+', help="List of BAM files")
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

logger.info("CMD: %s" % string.join(sys.argv, " "))

# data
chromosomes = {}
stats = {}

# process the bam files
samples = opt.bams
for sample in samples:
	logger.info("Processing sample %s" % sample)
	stats[sample] = {
		"chromosomes" : {},
		"categories" : {},
		"total" : 0,
		"mapped" : 0,
		"unmapped" : 0,
		}
	# compute the chromosome stats
	logger.info("Computing chromosome stats")
	lines = subprocess.check_output("samtools idxstats %s" % sample, shell=True)
	lines = lines.split("\n")
	for line in lines:
		if line != "":
			(chrom, length, mapped, unmapped) = line.split("\t")
			length = int(length)
			mapped =  int(mapped)
			unmapped = int(unmapped)
			stats[sample]["mapped"] += mapped
			stats[sample]["unmapped"] += unmapped
			if chrom != "*":
				# check that the length of the chromosomes are still the same
				if chromosomes.has_key(chrom):
					if chromosomes[chrom] != length:
						print "There are differences in the length of the chromosomes, this is an error at sample %s" % sample
						sys.exit(1)
				else:
					chromosomes[chrom] = length
				stats[sample]["chromosomes"][chrom] = mapped
	stats[sample]["total"] = stats[sample]["mapped"] + stats[sample]["unmapped"]
	# compute the flag stats
	logger.info("Computing category stats")
	lines = subprocess.check_output("samtools flagstat %s" % sample, shell=True)
	lines = lines.split("\n")
	for line in lines:
		if line != "":
			mo = re.search("^(\\d+) \\+ (\\d+) (.+)$", line)
			if mo != None:
				(nreads, nfailed, category) = mo.groups()
				nreads = int(nreads)
				if category != "with mate mapped to a different chr (mapQ>=5)":
					category = re.sub("\\s+\\(.+\\)$", "", category)
				stats[sample]["categories"][category] = nreads
	# compute the insert size
	logger.info("Computing insert size")
	lines = subprocess.check_output("samtools view -f 67 -F 260 %s | cut -f9 | tr -d '-'" % sample, shell=True)
	lines = lines.split("\n")
	insert_sum = 0
	insert_no = 0
	for line in lines:
		try:
			line = int(line)
			insert_sum += line
			insert_no += 1
		except:
			pass
	if insert_no != 0:
		stats[sample]["average_insert_size"] = float(insert_sum) / insert_no
		stats[sample]["average_insert_no"] = insert_no
	else:
		stats[sample]["average_insert_size"] = 0 
		stats[sample]["average_insert_no"] = 0

# write out the mapping rate
logger.info("Writing out mapping rates")
output = [["sample", "mapped", "unmapped", "total", "mapped_percentage", "average_insert_size"]]
for sample in stats.keys():
	output.append([
		sample,
		stats[sample]["mapped"],
		stats[sample]["unmapped"],
		stats[sample]["total"],
		float(stats[sample]["mapped"]) / stats[sample]["total"] * 100,
		stats[sample]["average_insert_size"]
		])
outfile = open("%s_mapping_rates.txt" % opt.output, "w")
for line in output:
	outfile.write(string.join(map(str, line), "\t") + "\n")
outfile.close()

# write out the per chromosome mapping rates
logger.info("Writing out chromosome mapping rates")
gsize = sum(chromosomes.values())
logger.info("Genome size: %d" % gsize)
output = [["sample", "chromosome", "chromosome_length", "mapped", "mapped_percentage", "expected_reads", "mapped_over_expected"]]
for sample in stats.keys():
	for chrom in stats[sample]["chromosomes"].keys():
		expected = float(chromosomes[chrom]) / gsize * stats[sample]["mapped"]
		output.append([
			sample,
			chrom,
			chromosomes[chrom],
			stats[sample]["chromosomes"][chrom],
			float(stats[sample]["chromosomes"][chrom]) / stats[sample]["total"] * 100,
			expected,
			expected != 0 and float(stats[sample]["chromosomes"][chrom]) / expected or 1
			])
outfile = open("%s_chromosome_mapping_rates.txt" % opt.output, "w")
for line in output:
	outfile.write(string.join(map(str, line), "\t") + "\n")
outfile.close()

# write out the per category mapping rates
logger.info("Writing out category mapping rates")
output = [["sample", "category", "mapped", "mapped_percentage"]]
for sample in stats.keys():
	for category in stats[sample]["categories"].keys():
		expected = float(chromosomes[chrom]) / gsize * stats[sample]["mapped"]
		output.append([
			sample,
			category,
			stats[sample]["categories"][category],
			float(stats[sample]["categories"][category]) / stats[sample]["total"] * 100,
			])
outfile = open("%s_category_mapping_rates.txt" % opt.output, "w")
for line in output:
	outfile.write(string.join(map(str, line), "\t") + "\n")
outfile.close()
