#!/bin/bash
#
# Run benchmarks for all config files
#
# Usage:
#   ./run_all_configs.sh [approaches] [num_gpus]
#
# Examples:
#   ./run_all_configs.sh                                    # Use approaches from each config file (sequential)
#   ./run_all_configs.sh official_fa3,flashinfer_mixed_fa3  # Override config files with specific approaches (sequential)
#   ./run_all_configs.sh "" 4                               # Use config approaches, run in parallel on 4 GPUs
#   ./run_all_configs.sh official_fa3 2                     # Override approaches, run in parallel on 2 GPUs
#

set -e  # Exit on error

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate /scratch/anirudha/revati-vidur/env

# Get approaches from command line (optional - defaults to using config file)
APPROACHES="$1"

# Get number of GPUs (optional - defaults to 1 for sequential execution)
NUM_GPUS="${2:-1}"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Create logs directory
LOGS_DIR="logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGS_DIR"

echo -e "${BLUE}================================================================================================${NC}"
echo -e "${BLUE}FlashInfer Benchmark Suite - Running All Configs${NC}"
echo -e "${BLUE}================================================================================================${NC}"
echo ""
if [ -z "$APPROACHES" ]; then
    echo -e "${GREEN}Approaches:${NC} Using approaches from config files"
else
    echo -e "${GREEN}Approaches:${NC} $APPROACHES (overriding config files)"
fi
if [ "$NUM_GPUS" -gt 1 ]; then
    echo -e "${GREEN}Execution mode:${NC} Parallel (${NUM_GPUS} GPUs per config)"
else
    echo -e "${GREEN}Execution mode:${NC} Sequential (1 GPU)"
fi
echo -e "${GREEN}Logs directory:${NC} $LOGS_DIR"
echo ""

# Find all config files
CONFIGS=($(ls -1 configs/*.yaml | sort))

if [ ${#CONFIGS[@]} -eq 0 ]; then
    echo -e "${RED}Error: No config files found in configs/ directory${NC}"
    exit 1
fi

echo -e "${GREEN}Found ${#CONFIGS[@]} config files:${NC}"
for config in "${CONFIGS[@]}"; do
    echo "  - $(basename $config)"
done
echo ""

# Run benchmarks for each config
TOTAL=${#CONFIGS[@]}
SUCCESS=0
FAILED=0
FAILED_CONFIGS=()

for i in "${!CONFIGS[@]}"; do
    config="${CONFIGS[$i]}"
    config_name=$(basename "$config" .yaml)
    num=$((i + 1))

    echo -e "${BLUE}================================================================================================${NC}"
    echo -e "${BLUE}[$num/$TOTAL] Running: $config_name${NC}"
    echo -e "${BLUE}================================================================================================${NC}"

    log_file="$LOGS_DIR/${config_name}.log"

    # Choose between parallel and sequential execution
    if [ "$NUM_GPUS" -gt 1 ]; then
        # Parallel execution with multiple GPUs
        if [ -z "$APPROACHES" ]; then
            CMD="python3 run_parallel.py --config $config --num-gpus $NUM_GPUS"
        else
            CMD="python3 run_parallel.py --config $config --num-gpus $NUM_GPUS --approaches $APPROACHES"
        fi
    else
        # Sequential execution with single GPU
        if [ -z "$APPROACHES" ]; then
            CMD="python3 run_benchmark.py --config $config"
        else
            CMD="python3 run_benchmark.py --config $config --approaches $APPROACHES"
        fi
    fi

    # Run benchmark and capture output
    if $CMD 2>&1 | tee "$log_file"; then
        echo -e "${GREEN}✓ SUCCESS: $config_name${NC}"
        SUCCESS=$((SUCCESS + 1))
    else
        echo -e "${RED}✗ FAILED: $config_name${NC}"
        FAILED=$((FAILED + 1))
        FAILED_CONFIGS+=("$config_name")
    fi

    echo ""
done

# Summary
echo -e "${BLUE}================================================================================================${NC}"
echo -e "${BLUE}Summary${NC}"
echo -e "${BLUE}================================================================================================${NC}"
echo -e "${GREEN}Successful:${NC} $SUCCESS/$TOTAL"
echo -e "${RED}Failed:${NC} $FAILED/$TOTAL"

if [ $FAILED -gt 0 ]; then
    echo ""
    echo -e "${RED}Failed configs:${NC}"
    for config in "${FAILED_CONFIGS[@]}"; do
        echo "  - $config"
    done
fi

echo ""
echo -e "${GREEN}Logs saved to:${NC} $LOGS_DIR"
echo ""

# Exit with error if any failed
if [ $FAILED -gt 0 ]; then
    exit 1
fi
