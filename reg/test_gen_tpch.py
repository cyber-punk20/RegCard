"""
Generate TPC-H test workload for monotonicity evaluation.
Reads a TPC-H train-format CSV, expands queries by varying selected predicate columns,
and writes -card.csv (query templates) and -pairs.csv (comparison pairs), same format as test_gen.py.
"""

import argparse
import os
import re
from pathlib import Path

import pandas as pd
import numpy as np
import pickle
import csv


# Columns we can vary for monotonicity chains: (predicate_pattern, list of values or (min, max, step))
# Pattern is the column part that appears in predicates, e.g. "l.l_quantity" or "o.o_orderdate"
VARYABLE_NUMERIC = [
    ("l.l_quantity", list(range(5, 51, 5))),           # 5,10,...,50
    ("p.p_size", list(range(5, 51, 5))),
    ("n.n_nationkey", list(range(0, 25, 2))),        # 0,2,...,24
    ("r.r_regionkey", list(range(0, 5))),             # 0..4
    ("c.c_custkey", list(range(10000, 150000, 20000))),
    ("s.s_suppkey", list(range(10000, 100001, 15000))),
]
# Date columns: use a small set of dates (as string literals in predicates)
VARYABLE_DATES = [
    ("o.o_orderdate", ["'1992-01-01'", "'1994-01-01'", "'1995-06-01'", "'1996-12-01'", "'1998-12-01'"]),
    ("l.l_shipdate", ["'1992-01-01'", "'1994-01-01'", "'1995-06-01'", "'1996-12-01'", "'1998-12-01'"]),
    ("l.l_commitdate", ["'1992-01-01'", "'1994-01-01'", "'1996-01-01'", "'1998-01-01'"]),
]

# TPC-H schema: (table_name, alias)
TPCH_TABLES = [
    ("region", "r"),
    ("nation", "n"),
    ("supplier", "s"),
    ("customer", "c"),
    ("part", "p"),
    ("partsupp", "ps"),
    ("orders", "o"),
    ("lineitem", "l"),
]
ALIAS_TO_TABLE = {alias: tbl for tbl, alias in TPCH_TABLES}


def _match_predicate(parts, col_pattern):
    """Find triple (col, op, val) in parts where col matches col_pattern. parts = predicates.split(',')."""
    i = 0
    while i + 2 < len(parts):
        col, op, val = parts[i], parts[i + 1], parts[i + 2]
        if col_pattern in col or col == col_pattern:
            return i, col, op, val
        i += 3
    return None


def generate_new_line_tpch(row, base_idx):
    """Generate variant rows by varying one varyable column in predicates. Returns (rows, rows_with_meta)."""
    new_rows = []
    new_rows_with_meta = []
    predicates = row["predicates"]
    if not predicates or not str(predicates).strip():
        return new_rows, new_rows_with_meta

    parts = str(predicates).split(",")
    if len(parts) < 3:
        return new_rows, new_rows_with_meta

    # Check numeric columns
    for col_pattern, values in VARYABLE_NUMERIC:
        match = _match_predicate(parts, col_pattern)
        if match is None:
            continue
        idx, col, op, _ = match
        base_predicates = str(base_idx)
        if op == "=":
            for rel_ord, val in enumerate(values):
                new_parts = parts[:]
                new_parts[idx + 2] = str(val)
                new_pred = ",".join(new_parts)
                new_rows.append([row["tables"], row["joins"], new_pred, None])
                new_rows_with_meta.append([row["tables"], row["joins"], new_pred, base_predicates, "=", val, val, rel_ord])
        elif op == "<":
            for rel_ord, val in enumerate(values):
                new_parts = parts[:]
                new_parts[idx + 2] = str(val)
                new_pred = ",".join(new_parts)
                new_rows.append([row["tables"], row["joins"], new_pred, None])
                new_rows_with_meta.append([row["tables"], row["joins"], new_pred, base_predicates, "<", 0, val, rel_ord])
        elif op == ">":
            for rel_ord, val in enumerate(values):
                new_parts = parts[:]
                new_parts[idx + 2] = str(val)
                new_pred = ",".join(new_parts)
                new_rows.append([row["tables"], row["joins"], new_pred, None])
                new_rows_with_meta.append([row["tables"], row["joins"], new_pred, base_predicates, ">", val, 1000000, rel_ord])
        # Only vary first matching column per row
        break
    else:
        # Check date columns
        for col_pattern, values in VARYABLE_DATES:
            match = _match_predicate(parts, col_pattern)
            if match is None:
                continue
            idx, col, op, _ = match
            base_predicates = str(base_idx)
            if op == "=":
                for rel_ord, val in enumerate(values):
                    new_parts = parts[:]
                    new_parts[idx + 2] = val
                    new_pred = ",".join(new_parts)
                    new_rows.append([row["tables"], row["joins"], new_pred, None])
                    new_rows_with_meta.append([row["tables"], row["joins"], new_pred, base_predicates, "=", rel_ord, rel_ord, rel_ord])
            elif op == "<":
                for rel_ord, val in enumerate(values):
                    new_parts = parts[:]
                    new_parts[idx + 2] = val
                    new_pred = ",".join(new_parts)
                    new_rows.append([row["tables"], row["joins"], new_pred, None])
                    new_rows_with_meta.append([row["tables"], row["joins"], new_pred, base_predicates, "<", 0, rel_ord, rel_ord])
            elif op == ">":
                for rel_ord, val in enumerate(values):
                    new_parts = parts[:]
                    new_parts[idx + 2] = val
                    new_pred = ",".join(new_parts)
                    new_rows.append([row["tables"], row["joins"], new_pred, None])
                    new_rows_with_meta.append([row["tables"], row["joins"], new_pred, base_predicates, ">", rel_ord, 1000, rel_ord])
            break

    return new_rows, new_rows_with_meta


def get_cmp(df):
    """Build comparison pairs from expanded query metadata (same logic as test_gen.get_cmp)."""
    cmp = []
    df = df.sort_values(by=["base_predicates", "relative_order"])
    df["relative_order"] = df["relative_order"].astype(str)
    gp = df.groupby(["base_predicates", "type"])
    for (base_predicate, ptype), group in gp:
        if ptype in ("range", "<", ">"):
            for i in range(len(group.index)):
                if i == 0:
                    cmp.append(str(group.index[i]) + "=" + str(group.index[i]))
                    df.at[group.index[i], "relative_order"] = str(group.index[i]) + "=" + str(group.index[i])
                else:
                    cmp.append(str(group.index[i]) + ">" + str(group.index[i - 1]))
                    df.at[group.index[i], "relative_order"] = str(group.index[i]) + ">" + str(group.index[i - 1])
        if ptype == "=":
            for i in range(len(group.index) - 1, -1, -1):
                cmp.append(str(group.index[i]) + "=" + str(group.index[i]))
                df.at[group.index[i], "relative_order"] = str(group.index[i]) + "=" + str(group.index[i])
    return df, cmp


def _connect_psql(db_user="postgres", db_host="localhost", db_port="5432", db_password="", db_name="tpch"):
    try:
        import psycopg2
    except ImportError as e:
        raise ImportError("Install psycopg2: pip install psycopg2-binary") from e
    conn = psycopg2.connect(
        user=db_user, host=db_host, port=db_port, password=db_password or None, database=db_name
    )
    conn.autocommit = True
    return conn


def _ensure_materialized_views(cursor, num_samples=1000):
    """Create alias_view for each TPC-H table. Drop if exists first."""
    for table_name, alias in TPCH_TABLES:
        view_name = f"{alias}_view"
        cursor.execute(f"DROP MATERIALIZED VIEW IF EXISTS {view_name};")
        cursor.execute(
            f"CREATE MATERIALIZED VIEW {view_name} AS "
            f"SELECT * FROM {table_name} AS {alias} ORDER BY RANDOM() LIMIT {num_samples};"
        )
    print("Materialized views created.")


def _get_bitmap(tables_str, predicates_str, cursor, num_samples=1000):
    """
    Build packed bitmaps for the query.
    Semantics matches scripts/gen_tpch_bitmaps_and_column_stats.py: per-table bitmap from evaluating each predicate on *_view.
    """
    tables = tables_str.split(",") if tables_str else []
    table_abbrs = [t.split()[1] for t in tables if t.strip()]
    all_bitmaps = np.zeros((len(table_abbrs), num_samples), dtype=int)

    if not predicates_str or not str(predicates_str).strip():
        return np.packbits(all_bitmaps, axis=1)

    parts = str(predicates_str).split(",")
    num_predicates = len(parts) // 3
    for i in range(num_predicates):
        col, op, val = parts[3 * i], parts[3 * i + 1], parts[3 * i + 2]
        table_abbr = col.split(".")[0]
        pred_expr = f"{col}{op}{val}"
        view_name = f"{table_abbr}_view"
        sql = f"SELECT CASE WHEN {pred_expr} THEN 1 ELSE 0 END AS bitmap FROM {view_name} AS {table_abbr};"
        try:
            cursor.execute(sql)
            record = np.array([r[0] for r in cursor.fetchall()], dtype=int)
            if len(record) < num_samples:
                record = np.pad(record, (0, num_samples - len(record)), constant_values=0)
            idx = table_abbrs.index(table_abbr)
            all_bitmaps[idx] = record[:num_samples]
        except Exception as e:
            print(f"Bitmap predicate failed for {pred_expr}: {e}")
            # leave bitmap zeros for this predicate
            continue

    return np.packbits(all_bitmaps, axis=1)


def _get_cardinality(tables_str, joins_str, predicates_str, cursor):
    join_clause = " AND ".join(str(joins_str).split(",")) if joins_str else "1=1"
    pred_parts = str(predicates_str).split(",") if predicates_str else []
    n = len(pred_parts) // 3
    pred_clause = " AND ".join(" ".join(pred_parts[3 * k : 3 * k + 3]) for k in range(n)) if n else "1=1"
    where_clause = f"({join_clause}) AND ({pred_clause})"
    sql = f"SELECT COUNT(*) FROM {tables_str} WHERE {where_clause}"
    cursor.execute(sql)
    return int(cursor.fetchone()[0])


def _remap_cmp_pairs(cmp_pairs, kept_old_to_new):
    out = []
    for s in cmp_pairs:
        op = ">" if ">" in s else "="
        left_s, right_s = s.split(op)
        left = int(left_s)
        right = int(right_s)
        if left in kept_old_to_new and right in kept_old_to_new:
            out.append(f"{kept_old_to_new[left]}{op}{kept_old_to_new[right]}")
    return out


def generate_new_queries_tpch(input_path, output_path_prefix, include_passthrough=True):
    """
    input_path: path to CSV (train format: tables#joins#predicates#cardinality)
    output_path_prefix: e.g. data/tpch7k_final-cmp -> writes ...-cmp-card.csv and ...-cmp-pairs.csv
    include_passthrough: if True, also append rows that had no varyable column as single-query groups (self=self pair).
    """
    input_path = Path(input_path)
    prefix = Path(output_path_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, sep="#", header=None, names=["tables", "joins", "predicates", "count"])
    # Drop cardinality for output
    df = df[["tables", "joins", "predicates"]]

    new_rows = []
    new_rows_with_meta = []
    passthrough_rows = []

    total_in = len(df)
    progress_every = max(1, total_in // 20)  # ~5% increments
    for i in range(len(df)):
        row = df.iloc[i].to_dict()
        rows, rows_meta = generate_new_line_tpch(row, base_idx=i)
        if rows:
            new_rows.extend(rows)
            new_rows_with_meta.extend(rows_meta)
        elif include_passthrough:
            passthrough_rows.append([row["tables"], row["joins"], row["predicates"]])
        if (i + 1) % progress_every == 0 or (i + 1) == total_in:
            print(
                f"[test_gen_tpch] Expanded {i+1}/{total_in} input queries "
                f"(variants={len(new_rows)}, passthrough={len(passthrough_rows)})"
            )

    # Queries only (no cardinality column)
    out_card = str(prefix) + "-card.csv"
    out_pairs = str(prefix) + "-pairs.csv"

    if new_rows_with_meta:
        # new_rows are list rows like [tables, joins, predicates, None]
        new_df = pd.DataFrame(new_rows, columns=["tables", "joins", "predicates", "_"])\
            [["tables", "joins", "predicates"]]
        meta_df = pd.DataFrame(
            new_rows_with_meta,
            columns=["tables", "joins", "predicates", "base_predicates", "type", "val1", "val2", "relative_order"],
        )
        _, cmp = get_cmp(meta_df.copy())
        new_df.to_csv(out_card, sep="#", index=False, header=False)
        pd.Series(cmp).to_csv(out_pairs, index=False, header=False)
        print(f"Wrote {len(new_rows)} variant queries -> {out_card}, {out_pairs} ({len(cmp)} pairs)")
    else:
        cmp = []

    if include_passthrough and passthrough_rows:
        pass_df = pd.DataFrame(passthrough_rows, columns=["tables", "joins", "predicates"])
        if new_rows:
            pass_df.to_csv(out_card, sep="#", index=False, header=False, mode="a")
            start_idx = len(new_rows)
            extra_pairs = [f"{start_idx + j}={start_idx + j}" for j in range(len(passthrough_rows))]
            pd.Series(cmp + extra_pairs).to_csv(out_pairs, index=False, header=False)
        else:
            pass_df.to_csv(out_card, sep="#", index=False, header=False)
            extra_pairs = [f"{j}={j}" for j in range(len(passthrough_rows))]
            pd.Series(extra_pairs).to_csv(out_pairs, index=False, header=False)
        print(f"Appended {len(passthrough_rows)} passthrough queries (no varyable column).")
    elif not new_rows:
        # No variants and no passthrough, or passthrough only
        if not include_passthrough or not passthrough_rows:
            print("No rows had a varyable predicate column. No output written.")
            return

    total_queries = len(new_rows) + (len(passthrough_rows) if include_passthrough else 0)
    print(f"Total queries: {total_queries}")


def main():
    parser = argparse.ArgumentParser(description="Generate TPC-H test workload (query variants + pairs) for monotonicity eval")
    parser.add_argument("-f", "--file", type=str, default="data/tpch7k_final.csv", help="Input CSV (train format, # sep)")
    parser.add_argument("-o", "--output", type=str, default="data/tpch7k_final-cmp", help="Output prefix: writes {prefix}-card.csv and {prefix}-pairs.csv")
    parser.add_argument("--no-passthrough", action="store_true", help="Do not include rows without a varyable column")
    parser.add_argument("--workload-name", type=str, default=None, help="If set, write evaluation-ready files under workloads/{name}.csv/.bitmaps/.cmp")
    parser.add_argument("--gen-bitmaps", action="store_true", help="When used with --workload-name, generate workloads/{name}.bitmaps from PostgreSQL")
    parser.add_argument("--gen-cardinality", action="store_true", help="When used with --workload-name, compute and write non-zero cardinalities into workloads/{name}.csv")
    parser.add_argument("--num-materialized-samples", type=int, default=1000, help="Rows per table view for bitmaps (default: 1000)")
    parser.add_argument("--db-user", default="postgres")
    parser.add_argument("--db-host", default="localhost")
    parser.add_argument("--db-port", default="5432")
    parser.add_argument("--db-password", default="")
    parser.add_argument("--db-name", default="tpch")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent
    input_path = Path(args.file) if os.path.isabs(args.file) else base / args.file
    output_prefix = Path(args.output) if os.path.isabs(args.output) else base / args.output

    if not input_path.exists():
        print(f"Input not found: {input_path}")
        return
    generate_new_queries_tpch(input_path, output_prefix, include_passthrough=not args.no_passthrough)

    # Optional: write evaluation-ready workload under workloads/{workload_name}.*
    if args.workload_name:
        # Read back the just-generated -card and -pairs to avoid duplicating generation logic above.
        card_path = Path(str(output_prefix) + "-card.csv")
        pairs_path = Path(str(output_prefix) + "-pairs.csv")
        if not card_path.exists():
            print(f"Expected card file not found: {card_path}")
            return

        # Load generated queries (no cardinality)
        dfq = pd.read_csv(card_path, sep="#", header=None, names=["tables", "joins", "predicates"])
        cmp_pairs = []
        if pairs_path.exists():
            cmp_pairs = list(pd.read_csv(pairs_path, header=None)[0].astype(str).values)

        workloads_dir = base / "workloads"
        workloads_dir.mkdir(parents=True, exist_ok=True)
        out_csv = workloads_dir / f"{args.workload_name}.csv"
        out_bitmaps = workloads_dir / f"{args.workload_name}.bitmaps"
        out_cmp = workloads_dir / f"{args.workload_name}.cmp"

        # If generating bitmaps or cardinalities, we need DB connection and views.
        need_db = args.gen_bitmaps or args.gen_cardinality
        if need_db:
            conn = _connect_psql(
                db_user=args.db_user,
                db_host=args.db_host,
                db_port=args.db_port,
                db_password=args.db_password,
                db_name=args.db_name,
            )
            cur = conn.cursor()
            try:
                _ensure_materialized_views(cur, num_samples=args.num_materialized_samples)
            except Exception as e:
                print("Creating views failed:", e)
                cur.close()
                conn.close()
                return
        else:
            conn = None
            cur = None

        kept_rows = []
        kept_bitmaps = []
        kept_old_to_new = {}
        dropped = 0
        total_gen = len(dfq)
        progress_every = max(1, total_gen // 20)  # ~5% increments

        # Compute requested artifacts; always drop 0-card rows if we compute cardinality (required by mscn.data.load_data).
        for old_idx in range(len(dfq)):
            row = dfq.iloc[old_idx]
            tables_str = str(row["tables"])
            joins_str = "" if pd.isna(row["joins"]) else str(row["joins"])
            predicates_str = "" if pd.isna(row["predicates"]) else str(row["predicates"])

            card = None
            bitmap = None

            try:
                if args.gen_cardinality:
                    card = _get_cardinality(tables_str, joins_str, predicates_str, cur)
                    if card <= 0:
                        dropped += 1
                        continue
                if args.gen_bitmaps:
                    bitmap = _get_bitmap(tables_str, predicates_str, cur, num_samples=args.num_materialized_samples)
            except Exception as e:
                print(f"Failed on query {old_idx}: {e}")
                dropped += 1
                continue

            new_idx = len(kept_rows)
            kept_old_to_new[old_idx] = new_idx

            # train.py expects 4 columns: tables#joins#predicates#cardinality, and cardinality must be non-zero.
            if card is None:
                # If user didn't request cardinality, we still cannot create a train.py-evaluable workload.
                # Keep placeholder 1 to satisfy loader, but strongly recommend --gen-cardinality.
                card = 1

            kept_rows.append([tables_str, joins_str, predicates_str, str(card)])
            if args.gen_bitmaps:
                kept_bitmaps.append(bitmap)
            if (old_idx + 1) % progress_every == 0 or (old_idx + 1) == total_gen:
                print(
                    f"[test_gen_tpch] Built {old_idx+1}/{total_gen} "
                    f"(kept={len(kept_rows)}, dropped={dropped})"
                )

        # Write workload CSV
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f, delimiter="#", quoting=csv.QUOTE_NONE)
            for r in kept_rows:
                w.writerow(r)
        print(f"Wrote {len(kept_rows)} workload rows to {out_csv} (dropped {dropped})")

        # Write bitmaps if requested
        if args.gen_bitmaps:
            with open(out_bitmaps, "wb") as f:
                pickle.dump(kept_bitmaps, f)
            print(f"Wrote {len(kept_bitmaps)} bitmaps to {out_bitmaps}")

        # Write cmp (remapped) if we have any pairs
        if cmp_pairs:
            remapped = _remap_cmp_pairs(cmp_pairs, kept_old_to_new)
            with open(out_cmp, "w") as f:
                for s in remapped:
                    f.write(str(s) + "\n")
            print(f"Wrote {len(remapped)} cmp constraints to {out_cmp}")

        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    main()
