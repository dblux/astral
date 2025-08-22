#!/usr/bin/awk -f
BEGIN {
    FS = OFS = "\t"   # tab as input and output delimiter
}

NR == 1 {
    # always print header line
    print $1, $2, $3, $6, $14, $15, $17, $28
    next
}

# Keep rows where the Genes column (column 6) starts with ITIH
$6 ~ /^ITIH/ {
    print $1, $2, $3, $6, $14, $15, $17, $28
}
