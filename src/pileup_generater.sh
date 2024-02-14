# generate pileup files by split large BAM file into small trunks and process them seperately
# One output pileup file per chromosome
#!/usr/bin/bash
# Usage: ./pileup_parallel.sh <bam_file> <ref_file> <output_prefix> <slots>
# Example: ./pileup_parallel.sh input.bam output_pileup 4

# BAM file
BAM_FILE=$1

# Referecnce file
REF_FILE=$2

# Output file prefix
OUTPUT_PREFIX=$3

# Region file name
REGION_F=$4

# Number of slots/threads for parallel
SLOTS=$5



# Chunk length
CHUNK_LENGTH=100000

# Generate regions file
#REGIONS_FILE="./mid_out/pileup/regions_chr21.txt"
REGIONS_FILE=$REGION_F
> $REGIONS_FILE
samtools idxstats $BAM_FILE | grep -v '*' | awk '$3 > 0' | cut -f 1,2 | while read CHR LENGTH
do
  NUM_CHUNKS=$((($LENGTH + $CHUNK_LENGTH - 1) / $CHUNK_LENGTH))
  for ((i = 0; i < $NUM_CHUNKS; i++)); do
    START=$((i * $CHUNK_LENGTH + 1))
    END=$(((i + 1) * $CHUNK_LENGTH))
    if [ $END -gt $LENGTH ]; then
      END=$LENGTH
    fi
    echo "$CHR:$START-$END" >> $REGIONS_FILE
  done
done | sort -k1,1V -k2,2n > $REGIONS_FILE

# Function to run samtools mpileup on a region
run_mpileup(){
  REGION=$1
  CHR=$(echo $REGION | cut -d':' -f 1)
  PILEUP_FILE="${OUTPUT_PREFIX}_${CHR}_${REGION//[:\-]/_}.pileup"
  samtools mpileup \
    --max-depth 200 \
    --min-MQ 10 \
    --min-BQ 20 \
    --reverse-del \
    -f $REF_FILE \
    -r $REGION \
    -o $PILEUP_FILE \
    $BAM_FILE

  # Check if the pileup file is empty, and delete it if so
  if [ ! -s "$PILEUP_FILE" ]; then
    rm "$PILEUP_FILE"
  fi
}

# Export the function and variables so they are available to parallel
export -f run_mpileup
export BAM_FILE OUTPUT_PREFIX REF_FILE REGION_F

# Run samtools mpileup in parallel
cat $REGIONS_FILE | parallel -j $SLOTS run_mpileup

# Combine the resulting pileup files in order of regions
cut -d':' -f 1 $REGIONS_FILE | sort -u | while read CHR; do
  PILEUP_FILES=$(ls ${OUTPUT_PREFIX}_${CHR}_*.pileup 2> /dev/null)
  if [[ -f "$PILEUP_FILE" && -s "$PILEUP_FILE" ]]; then
    cat "$PILEUP_FILE" >> "${OUTPUT_PREFIX}_${CHR}.pileup"
    rm "$PILEUP_FILE" # Remove the file after its contents have been concatenated
  fi
done < $REGIONS_FILE

