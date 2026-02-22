import sys
import os
BASE_DIR = os.getcwd()
sys.path.append(BASE_DIR)

import json
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from src.call_models.call_apis import api_infer_single
from src.utils import generate_pk_fk


def build_schema_prompt(db_id, table_json):
    table_info = [t for t in table_json if t['db_id'] == db_id][0]
    table_names = table_info['table_names_original']
    column_names = table_info['column_names_original']
    column_types = table_info['column_types']

    schema_parts = []
    for t_idx, t_name in enumerate(table_names):
        cols = []
        for col_idx, (col_table_idx, col_name) in enumerate(column_names):
            if col_table_idx == t_idx:
                col_type = column_types[col_idx] if col_idx < len(column_types) else ""
                cols.append(f"  {col_name} {col_type}".rstrip())
        if cols:
            schema_parts.append(f"CREATE TABLE {t_name} (\n" + ",\n".join(cols) + "\n);")

    return "\n\n".join(schema_parts)


def build_fk_prompt(db_id, question_info, table_json):
    _, fk_dict = generate_pk_fk(question_info, table_json)
    if not fk_dict:
        return ""
    lines = ["Foreign Keys:"]
    for src, tgt in fk_dict.items():
        lines.append(f"  {src} = {tgt}")
    return "\n".join(lines)


def build_prompt(question_info, table_json):
    db_id = question_info['db_id']
    schema = build_schema_prompt(db_id, table_json)
    fk = build_fk_prompt(db_id, question_info, table_json)
    question = question_info['question']
    evidence = question_info.get('evidence', '') or ''

    prompt = f"""You are an expert SQL developer. Given the following SQLite database schema, generate a SQL query to answer the question.

{schema}

{fk}

Question: {question}"""
    if evidence:
        prompt += f"\nEvidence: {evidence}"
    prompt += """

Generate a valid SQLite query that answers the question. Return only the SQL inside a code block:
```sql
[Your SQL]
```"""
    return prompt


def extract_sql(response):
    import re
    for pattern in [r"```sqlite(.*?)```", r"```sql(.*?)```", r"```(.*?)```"]:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()
    return response.strip()


def process_one(idx, info, table_json):
    prompt = build_prompt(info, table_json)
    _, response, token_usage = api_infer_single(prompt, max_token_length=1024)
    sql = extract_sql(response)
    return idx, sql + '\t----- bird -----\t' + info['db_id'], token_usage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_config_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, default='data/arcwise/baseline_sql.json')
    parser.add_argument('--resume', action='store_true', help='Resume from existing partial output')
    parser.add_argument('--workers', type=int, default=16, help='Number of parallel API workers')
    parser.add_argument('--limit', type=int, default=None, help='Only process first N entries (for testing)')
    args = parser.parse_args()

    config = json.load(open(args.data_config_path))
    tables_path = config['dev_tables']
    if not os.path.isabs(tables_path):
        tables_path = os.path.join(BASE_DIR, tables_path)
    data_path = config['dev_data']
    if not os.path.isabs(data_path):
        data_path = os.path.join(BASE_DIR, data_path)

    table_json = json.load(open(tables_path))
    data_json = json.load(open(data_path))

    # Load existing results if resuming
    existing = {}
    if args.resume and os.path.exists(args.output_path):
        existing = json.load(open(args.output_path))
        print(f"Resuming: loaded {len(existing)} existing results")

    results = dict(existing)
    save_lock = threading.Lock()

    # Token counters
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cached_tokens = 0
    total_reasoning_tokens = 0

    # Build list of work to do
    if args.limit:
        data_json = data_json[:args.limit]
    todo = [(idx, info) for idx, info in enumerate(data_json) if str(idx) not in results]
    print(f"Generating {len(todo)} baseline SQLs with {args.workers} parallel workers...")

    completed_count = 0

    def save_results():
        os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)
        json.dump(results, open(args.output_path, 'w'), indent=4)

    def print_token_summary(n):
        print(
            f"[{n} done] Tokens — prompt: {total_prompt_tokens}, "
            f"completion: {total_completion_tokens}, "
            f"cached: {total_cached_tokens}, "
            f"reasoning: {total_reasoning_tokens}, "
            f"total: {total_prompt_tokens + total_completion_tokens}"
        )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_one, idx, info, table_json): idx for idx, info in todo}
        with tqdm(total=len(todo), desc="Generating baseline SQL") as pbar:
            for future in as_completed(futures):
                idx, sql_with_db, token_usage = future.result()
                with save_lock:
                    results[str(idx)] = sql_with_db
                    total_prompt_tokens += token_usage.get('prompt_tokens', 0)
                    total_completion_tokens += token_usage.get('completion_tokens', 0)
                    total_cached_tokens += token_usage.get('cached_tokens', 0)
                    total_reasoning_tokens += token_usage.get('reasoning_tokens', 0)
                    completed_count += 1
                    if completed_count % 50 == 0:
                        save_results()
                        print_token_summary(completed_count)
                pbar.update(1)

    save_results()
    print(f"Done. Saved {len(results)} baseline SQLs to {args.output_path}")
    print_token_summary(len(todo))


if __name__ == '__main__':
    main()
