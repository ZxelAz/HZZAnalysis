#!/bin/bash

# Script to run the Z+X yield calculator with predefined inputs

# Define parameters
: "${HZZ_ROOT:=/afs/cern.ch/work/z/zhiheng/Thesis/HZZAnalysis}"
SCRIPT="${HZZ_ROOT}/DataPreprocessing/functions/ZX_yield_calculator.py"
YEARS=("2022" "2022EE" "2023preBPix" "2023postBPix")
BASE_DIR="/eos/user/z/zhiheng/STXS_samples"

# Activate the virtual environment if needed (uncomment if necessary)
# source /path/to/venv/bin/activate

echo "Running Z+X yield calculator for all configured years..."
echo ""

FAILED=0

for YEAR in "${YEARS[@]}"; do
    YEAR_MC="$YEAR"

    case "$YEAR" in
        2022|2022EE)
            DATA_DIR="2022Data"
            ;;
        2023preBPix|2023postBPix)
            DATA_DIR="2023Data"
            ;;
        *)
            echo "✗ Error: Unsupported year in configuration: $YEAR"
            FAILED=1
            continue
            ;;
    esac

    INPUT_FILE="$BASE_DIR/$DATA_DIR/${YEAR}.root"
    OUTPUT_FILE="$BASE_DIR/$DATA_DIR/ZX_${YEAR}.root"

    echo "Running Z+X yield calculator with the following parameters:"
    echo "  Year: $YEAR"
    echo "  Year MC: $YEAR_MC"
    echo "  Input file: $INPUT_FILE"
    echo "  Output file: $OUTPUT_FILE"
    echo ""

    # Run the Python script with predefined arguments
    echo "Executing: python ${SCRIPT} ..."
    echo ""

    python "${SCRIPT}" \
        --year "$YEAR" \
        --year_mc "$YEAR_MC" \
        --input_file "$INPUT_FILE" \
        --output_file "$OUTPUT_FILE"

    EXIT_CODE=$?
    echo ""

    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ Z+X yield calculation completed successfully for $YEAR"
    else
        echo "✗ Error: Z+X yield calculation failed for $YEAR with exit code $EXIT_CODE"
        FAILED=1
    fi

    echo "----------------------------------------"
    echo ""
done

if [ $FAILED -eq 0 ]; then
    echo "✓ All Z+X yield calculations completed successfully!"
    exit 0
else
    echo "✗ One or more Z+X yield calculations failed."
    exit 1
fi

