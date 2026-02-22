#!/bin/bash
set -e

# ============================================================
# SHARE NL2SQL Pipeline on Arcwise Data with GPT 5.2
# ============================================================
# Usage: bash scripts/run_arcwise_inference.sh
#
# Required env vars (set before running or export them):
#   OPENAI_API_BASE  - API base URL (default: https://api.openai.com/v1)
#   OPENAI_API_KEY   - Your OpenAI API key
#   OPENAI_ENGINE_ID - Model ID (default: gpt-5.2)
# ============================================================

# Defaults (override by exporting before running)
export OPENAI_API_BASE="${OPENAI_API_BASE:-https://api.openai.com/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:?Error: OPENAI_API_KEY must be set}"
export OPENAI_ENGINE_ID="${OPENAI_ENGINE_ID:-gpt-5.2}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

CONFIG="configs/arcwise_data_config.json"
BASELINE_SQL="data/arcwise/baseline_sql.json"
OUTPUT_DIR="outputs/arcwise_infer"

# Model IDs (vLLM downloads from HuggingFace automatically)
BAM_MODEL="birdsql/share-bam"
SAM_MODEL="birdsql/share-sam"
LOM_MODEL="birdsql/share-lom"

echo "=========================================="
echo "SHARE Pipeline - Arcwise Data"
echo "=========================================="
echo "Config:     $CONFIG"
echo "API Base:   $OPENAI_API_BASE"
echo "Engine:     $OPENAI_ENGINE_ID"
echo "Output Dir: $OUTPUT_DIR"
echo "=========================================="

# Step 1: Prepare data (create dev.json if not exists)
if [ ! -f "data/arcwise/dev.json" ]; then
    echo "[Step 1] Creating data/arcwise/dev.json from arcwise_plat_all.json..."
    python3 -c "
import json
data = json.load(open('arcwise_plat_all.json'))
for item in data:
    if item.get('evidence') is None:
        item['evidence'] = ''
json.dump(data, open('data/arcwise/dev.json', 'w'), indent=2, ensure_ascii=False)
print(f'Created dev.json with {len(data)} entries')
"
else
    echo "[Step 1] data/arcwise/dev.json already exists, skipping."
fi

# Step 2: Generate baseline SQL with GPT 5.2
if [ ! -f "$BASELINE_SQL" ]; then
    echo "[Step 2] Generating baseline SQL with GPT 5.2..."
    python3 scripts/generate_baseline_sql.py \
        --data_config_path "$CONFIG" \
        --output_path "$BASELINE_SQL"
else
    echo "[Step 2] $BASELINE_SQL already exists, skipping. (Use --resume flag to continue partial runs)"
fi

# Verify baseline SQL
BASELINE_COUNT=$(python3 -c "import json; d=json.load(open('$BASELINE_SQL')); print(len(d))")
echo "Baseline SQL count: $BASELINE_COUNT"

# Step 3: Run full SHARE pipeline (BAM → SAM → LOM → GPT 5.2)
echo "[Step 3] Running SHARE pipeline (BAM/SAM/LOM + final SQL generation)..."
mkdir -p "$OUTPUT_DIR"

python3 src/infer/run.py \
    --data_config_path "$CONFIG" \
    --bam_model_path "$BAM_MODEL" \
    --sam_model_path "$SAM_MODEL" \
    --lom_model_path "$LOM_MODEL" \
    --input_sql_path "$BASELINE_SQL" \
    --output_dir "$OUTPUT_DIR"

# Verify outputs
echo ""
echo "=========================================="
echo "Pipeline Complete! Checking outputs..."
echo "=========================================="
for f in original_traj.jsonl masked_traj.jsonl intermediate_traj.jsonl augmented_schema.json final_traj.jsonl final_sql.json; do
    FPATH="$OUTPUT_DIR/$f"
    if [ -f "$FPATH" ]; then
        if [[ "$f" == *.jsonl ]]; then
            COUNT=$(wc -l < "$FPATH")
            echo "  ✓ $f ($COUNT lines)"
        else
            COUNT=$(python3 -c "import json; print(len(json.load(open('$FPATH'))))")
            echo "  ✓ $f ($COUNT entries)"
        fi
    else
        echo "  ✗ $f MISSING"
    fi
done
