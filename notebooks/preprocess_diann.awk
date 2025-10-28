#!/usr/bin/awk -f
BEGIN {
    FS = OFS = "\t"   # tab as input and output delimiter
}

# Always print header line
NR == 1 {
    print "Polypeptide.Novogene.ID", $3, $6, $14, $15, $17, $20, $28
    next
}

# Filters out QC samples and formats ID
$2 ~ /^Astral-1_8_/ {
    sub(/^Astral-1_8_/, "", $2)
    print $2, $3, $6, $14, $15, $17, $20, $28
}
