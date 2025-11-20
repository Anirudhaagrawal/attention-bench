#!/bin/bash
#
# Run benchmarks for all config files
#
# Usage:
#   ./scripts/run_all_configs.sh [approaches] [num_gpus] [config_type]
#
# Examples:
#   ./scripts/run_all_configs.sh                                    # Use approaches from each config file (sequential)
#   ./scripts/run_all_configs.sh official_fa3,flashinfer_mixed_fa3  # Override with specific approaches (sequential)
#   ./scripts/run_all_configs.sh "" 4                               # Use config approaches, run in parallel on 4 GPUs
#   ./scripts/run_all_configs.sh official_fa3 2                     # Override approaches, run in parallel on 2 GPUs
#   ./scripts/run_all_configs.sh "" 4 decode                        # Run only decode configs on 4 GPUs
#   ./scripts/run_all_configs.sh "" 4 mixed                         # Run only mixed configs on 4 GPUs
#

set -e  # Exit on error

# Activate conda environment if needed
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "Using conda environment: $CONDA_DEFAULT_ENV"
elif [ -d "/scratch/anirudha/revati-vidur/env" ]; then
    eval "$(conda shell.bash hook)"
    conda activate /scratch/anirudha/revati-vidur/env
fi

# Get approaches from command line (optional - defaults to using config file)
APPROACHES="$1"

# Get number of GPUs (optional - defaults to 1 for sequential execution)
NUM_GPUS="${2:-1}"

# Get config type filter (optional - defaults to all types)
CONFIG_TYPE="${3:-all}"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Create timestamped logs directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGS_DIR="logs/${TIMESTAMP}"
mkdir -p "$LOGS_DIR"

echo -e "${BLUE}================================================================================================${NC}"
echo -e "${BLUE}Attention Benchmark Suite - Running All Configs${NC}"
echo -e "${BLUE}================================================================================================${NC}"
echo ""
if [ -z "$APPROACHES" ]; then
    echo -e "${GREEN}Approaches:${NC} Using approaches from config files"
else
    echo -e "${GREEN}Approaches:${NC} $APPROACHES (overriding config files)"
fi
if [ "$NUM_GPUS" -gt 1 ]; then
    echo -e "${GREEN}Execution mode:${NC} Parallel (${NUM_GPUS} GPUs per config via Ray)"
else
    echo -e "${GREEN}Execution mode:${NC} Sequential (1 GPU via Ray)"
fi
echo -e "${GREEN}Config type:${NC} $CONFIG_TYPE"
echo -e "${GREEN}Logs directory:${NC} $LOGS_DIR"
echo ""

# Find all config files based on type filter
if [ "$CONFIG_TYPE" = "all" ]; then
    CONFIGS=($(find configs -name "*.yaml" -type f | sort))
elif [ "$CONFIG_TYPE" = "decode" ] || [ "$CONFIG_TYPE" = "prefill" ] || [ "$CONFIG_TYPE" = "mixed" ]; then
    CONFIGS=($(find configs/$CONFIG_TYPE -name "*.yaml" -type f 2>/dev/null | sort))
else
    echo -e "${RED}Error: Invalid config type '$CONFIG_TYPE'. Must be 'all', 'decode', 'prefill', or 'mixed'${NC}"
    exit 1
fi

if [ ${#CONFIGS[@]} -eq 0 ]; then
    echo -e "${RED}Error: No config files found${NC}"
    if [ "$CONFIG_TYPE" != "all" ]; then
        echo -e "${RED}Searched in: configs/$CONFIG_TYPE/${NC}"
    else
        echo -e "${RED}Searched in: configs/${NC}"
    fi
    exit 1
fi

echo -e "${GREEN}Found ${#CONFIGS[@]} config file(s):${NC}"
for config in "${CONFIGS[@]}"; do
    echo "  - $config"
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
    echo -e "${BLUE}[$num/$TOTAL] Running: $config${NC}"
    echo -e "${BLUE}================================================================================================${NC}"

    log_file="$LOGS_DIR/${config_name}.log"

    # Build command using attention-bench-ray CLI
    CMD="attention-bench-ray --config $config --num-gpus $NUM_GPUS"

    # Add approaches override if specified
    if [ -n "$APPROACHES" ]; then
        CMD="$CMD --approaches $APPROACHES"
    fi

    # Display command
    echo -e "${YELLOW}Command:${NC} $CMD"
    echo ""

    # Run benchmark and capture output
    START_TIME=$(date +%s)
    if $CMD 2>&1 | tee "$log_file"; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        echo ""
        echo -e "${GREEN}✓ SUCCESS: $config_name (${DURATION}s)${NC}"
        SUCCESS=$((SUCCESS + 1))
    else
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        echo ""
        echo -e "${RED}✗ FAILED: $config_name (${DURATION}s)${NC}"
        FAILED=$((FAILED + 1))
        FAILED_CONFIGS+=("$config_name")
    fi

    echo ""
done

# Summary
SUMMARY_FILE="$LOGS_DIR/summary.txt"

echo -e "${BLUE}================================================================================================${NC}"
echo -e "${BLUE}Summary${NC}"
echo -e "${BLUE}================================================================================================${NC}"
echo -e "${GREEN}Successful:${NC} $SUCCESS/$TOTAL"
echo -e "${RED}Failed:${NC} $FAILED/$TOTAL"

# Write summary to file
{
    echo "Attention Benchmark Suite - Run Summary"
    echo "========================================"
    echo ""
    echo "Timestamp: $(date)"
    echo "Config type: $CONFIG_TYPE"
    echo "GPUs: $NUM_GPUS"
    echo "Approaches: ${APPROACHES:-default}"
    echo ""
    echo "Results:"
    echo "  Successful: $SUCCESS/$TOTAL"
    echo "  Failed: $FAILED/$TOTAL"
    echo ""
} > "$SUMMARY_FILE"

if [ $FAILED -gt 0 ]; then
    echo ""
    echo -e "${RED}Failed configs:${NC}"
    echo "Failed configs:" >> "$SUMMARY_FILE"
    for config in "${FAILED_CONFIGS[@]}"; do
        echo "  - $config"
        echo "  - $config" >> "$SUMMARY_FILE"
    done
    echo "" >> "$SUMMARY_FILE"
fi

# List all log files
echo "Log files:" >> "$SUMMARY_FILE"
for config in "${CONFIGS[@]}"; do
    config_name=$(basename "$config" .yaml)
    echo "  - ${config_name}.log" >> "$SUMMARY_FILE"
done

echo ""
echo -e "${GREEN}Logs directory:${NC} $LOGS_DIR"
echo -e "${GREEN}Summary file:${NC} $SUMMARY_FILE"
echo ""

# Exit with error if any failed
if [ $FAILED -gt 0 ]; then
    exit 1
fi
