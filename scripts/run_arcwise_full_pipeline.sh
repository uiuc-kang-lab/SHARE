#!/bin/bash
set -e
cd /root/SHARE

# Load env
set -a && source .env && set +a

echo "=== Step 1: Check/Generate baseline SQL ==="
python3 scripts/generate_baseline_sql.py \
    --data_config_path configs/arcwise_data_config.json \
    --output_path data/arcwise/baseline_sql.json \
    --resume

BASELINE_COUNT=$(python3 -c "import json; d=json.load(open('data/arcwise/baseline_sql.json')); print(len(d))")
echo "Baseline SQL count: $BASELINE_COUNT"
if [ "$BASELINE_COUNT" -lt 673 ]; then
    echo "ERROR: Not all 673 baseline SQLs generated. Got $BASELINE_COUNT."
    exit 1
fi

echo ""
echo "=== Step 2: Run SHARE inference (BAM -> SAM -> LOM -> GPT-5.2) ==="
mkdir -p outputs/arcwise_infer

python3 src/infer/run.py \
    --data_config_path configs/arcwise_data_config.json \
    --bam_model_path birdsql/share-bam \
    --sam_model_path birdsql/share-sam \
    --lom_model_path birdsql/share-lom \
    --input_sql_path data/arcwise/baseline_sql.json \
    --output_dir outputs/arcwise_infer

echo ""
echo "=== Pipeline complete! ==="
for f in original_traj.jsonl masked_traj.jsonl intermediate_traj.jsonl augmented_schema.json final_traj.jsonl final_sql.json; do
    FPATH="outputs/arcwise_infer/$f"
    if [ -f "$FPATH" ]; then
        if [[ "$f" == *.jsonl ]]; then
            COUNT=$(wc -l < "$FPATH")
        else
            COUNT=$(python3 -c "import json; print(len(json.load(open('$FPATH'))))")
        fi
        echo "  ✓ $f ($COUNT entries)"
    else
        echo "  ✗ $f MISSING"
    fi
done
