import sys
import os
BASE_DIR = os.getcwd()
sys.path.append(BASE_DIR)

import json
import argparse
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_config_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, default='data/arcwise/baseline_sql.json')
    parser.add_argument('--resume', action='store_true', help='Resume from existing partial output')
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
    for idx, info in enumerate(tqdm(data_json, desc="Generating baseline SQL")):
        if str(idx) in results:
            continue
        prompt = build_prompt(info, table_json)
        _, response, _ = api_infer_single(prompt, max_token_length=1024)
        sql = extract_sql(response)
        sql_with_db = sql + '\t----- bird -----\t' + info['db_id']
        results[str(idx)] = sql_with_db

        # Save incrementally every 50 entries
        if (idx + 1) % 50 == 0:
            os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)
            json.dump(results, open(args.output_path, 'w'), indent=4)
            print(f"Saved {len(results)} results so far")

    os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)
    json.dump(results, open(args.output_path, 'w'), indent=4)
    print(f"Done. Saved {len(results)} baseline SQLs to {args.output_path}")


if __name__ == '__main__':
    main()
