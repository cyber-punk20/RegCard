#!/usr/bin/env python3
"""
Generate 5000 TPC-H queries in train.csv format: tables#joins#predicates#cardinality.
Requires PostgreSQL with TPC-H data loaded. Run against the DB to fill cardinalities.
"""

import argparse
import csv
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root for imports if needed
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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

# Join graph: (alias1, alias2) -> (col1 = col2) for FROM t1, t2 WHERE ...
# We store (alias_a, alias_b, "a.col = b.col") so we can emit join condition.
JOIN_EDGES = [
    ("n", "r", "n.n_regionkey=r.r_regionkey"),
    ("s", "n", "s.s_nationkey=n.n_nationkey"),
    ("c", "n", "c.c_nationkey=n.n_nationkey"),
    ("ps", "p", "ps.ps_partkey=p.p_partkey"),
    ("ps", "s", "ps.ps_suppkey=s.s_suppkey"),
    ("o", "c", "o.o_custkey=c.c_custkey"),
    ("l", "o", "l.l_orderkey=o.o_orderkey"),
    ("l", "p", "l.l_partkey=p.p_partkey"),
    ("l", "s", "l.l_suppkey=s.s_suppkey"),
]

# (alias, column, type: 'int'|'decimal'|'date', min_val, max_val or None for dates)
# For dates we use string literals; for int/decimal we use numbers.
PREDICATE_COLUMNS = [
    ("r", "r_regionkey", "int", 0, 4),
    ("n", "n_nationkey", "int", 0, 24),
    ("s", "s_suppkey", "int", 1, 100000),
    ("s", "s_acctbal", "decimal", -1000, 10000),
    ("c", "c_custkey", "int", 1, 150000),
    ("c", "c_acctbal", "decimal", -1000, 10000),
    ("p", "p_partkey", "int", 1, 200000),
    ("p", "p_size", "int", 1, 50),
    ("p", "p_retailprice", "decimal", 1, 2000),
    ("ps", "ps_availqty", "int", 1, 9999),
    ("ps", "ps_supplycost", "decimal", 1, 1000),
    ("o", "o_orderkey", "int", 1, 6000000),
    ("o", "o_totalprice", "decimal", 100, 500000),
    ("o", "o_orderdate", "date", "1992-01-01", "1998-12-31"),
    ("l", "l_quantity", "int", 1, 50),
    ("l", "l_extendedprice", "decimal", 1, 100000),
    ("l", "l_discount", "decimal", 0, 0.1),
    ("l", "l_shipdate", "date", "1992-01-01", "1998-12-31"),
    ("l", "l_commitdate", "date", "1992-01-01", "1998-12-31"),
]

# Build alias -> table name
ALIAS_TO_TABLE = {alias: tbl for tbl, alias in TPCH_TABLES}
TABLE_TO_ALIAS = {tbl: alias for tbl, alias in TPCH_TABLES}

# Join index: (a,b) and (b,a) both map to the same condition
JOIN_MAP = {}
for a, b, cond in JOIN_EDGES:
    JOIN_MAP[(a, b)] = cond
    JOIN_MAP[(b, a)] = cond


def get_connected_tables(start_alias, num_tables, join_edges_list):
    """Return a set of aliases that form a connected subgraph of size num_tables including start."""
    if num_tables <= 0:
        return set()
    chosen = {start_alias}
    # edges as adjacency: alias -> set of neighbors
    adj = {}
    for a, b, _ in join_edges_list:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    while len(chosen) < num_tables:
        candidates = set()
        for a in chosen:
            for b in adj.get(a, []):
                if b not in chosen:
                    candidates.add(b)
        if not candidates:
            break
        chosen.add(random.choice(list(candidates)))
    return chosen


def build_joins(aliases):
    aliases = sorted(aliases)
    joins = []
    for i, a in enumerate(aliases):
        for b in aliases[i + 1 :]:
            key = (a, b)
            if key in JOIN_MAP:
                joins.append(JOIN_MAP[key])
    return joins


def format_tables(aliases):
    return ",".join(f"{ALIAS_TO_TABLE[a]} {a}" for a in sorted(aliases))


def sample_predicate_value(alias, col, typ, vmin, vmax):
    if typ == "int":
        return str(random.randint(int(vmin), int(vmax)))
    if typ == "decimal":
        return str(round(random.uniform(float(vmin), float(vmax)), 2))
    if typ == "date":
        dmin = datetime.strptime(vmin, "%Y-%m-%d")
        dmax = datetime.strptime(vmax, "%Y-%m-%d")
        delta = (dmax - dmin).days
        d = dmin if delta <= 0 else dmin + timedelta(days=random.randint(0, delta))
        return "'" + d.strftime("%Y-%m-%d") + "'"
    return str(vmin)


def generate_one_query(num_predicates_min_max=(1, 4), num_tables_min_max=(1, 5), single_table_only=False):
    if single_table_only:
        # One table, no joins
        alias = random.choice([t[1] for t in TPCH_TABLES])
        aliases = {alias}
        joins_str = ""
    else:
        num_tables = random.randint(num_tables_min_max[0], num_tables_min_max[1])
        start = random.choice([t[1] for t in TPCH_TABLES])
        aliases = get_connected_tables(start, num_tables, JOIN_EDGES)
        if len(aliases) < num_tables_min_max[0]:
            aliases = get_connected_tables(random.choice([t[1] for t in TPCH_TABLES]), num_tables_min_max[0], JOIN_EDGES)
        joins_str = ",".join(build_joins(aliases))

    tables_str = format_tables(aliases)

    # Predicates: only on columns that belong to chosen tables; require at least one
    available = [(a, col, typ, vmin, vmax) for a, col, typ, vmin, vmax in PREDICATE_COLUMNS if a in aliases]
    if not available:
        pred_parts = []
    else:
        lo, hi = num_predicates_min_max[0], num_predicates_min_max[1]
        n_pred = random.randint(max(1, lo), min(hi, len(available)))
        chosen_preds = random.sample(available, n_pred)
        pred_parts = []
        for a, col, typ, vmin, vmax in chosen_preds:
            op = random.choice(["=", "<", ">"])
            val = sample_predicate_value(a, col, typ, vmin, vmax)
            pred_parts.extend([f"{a}.{col}", op, val])
    predicates_str = ",".join(pred_parts)

    return tables_str, joins_str, predicates_str


def generate_queries(n=5000, seed=42, out_path=None, single_table_only=True):
    random.seed(seed)
    rows = []
    for _ in range(n):
        t, j, p = generate_one_query(single_table_only=single_table_only)
        # Cardinality placeholder; will be filled by DB script
        rows.append((t, j, p, ""))
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f, delimiter="#", quoting=csv.QUOTE_NONE)
            for r in rows:
                w.writerow(r)
        print(f"Wrote {len(rows)} query templates to {out_path}")
    return rows


def run_cardinalities(csv_path, db_user="postgres", db_host="localhost", db_port="5432", db_password="", db_name="tpch"):
    """Run each query against PostgreSQL and fill cardinality. Overwrites the CSV."""
    try:
        import psycopg2
    except ImportError:
        print("Install psycopg2: pip install psycopg2-binary")
        return
    csv_path = Path(csv_path)
    rows = []
    with open(csv_path, "r") as f:
        for line in f:
            parts = line.strip().split("#")
            if len(parts) < 4:
                parts.extend([""] * (4 - len(parts)))
            rows.append(parts)

    conn = psycopg2.connect(
        user=db_user, host=db_host, port=db_port, password=db_password or None, database=db_name
    )
    conn.autocommit = True
    cur = conn.cursor()

    for i, row in enumerate(rows):
        tables, joins, predicates, _ = row[0], row[1], row[2], row[3]
        join_clause = " AND ".join(joins.split(",")) if joins else "1=1"
        pred_triplets = predicates.split(",") if predicates else []
        n = len(pred_triplets) // 3
        pred_clause = " AND ".join(
            " ".join(pred_triplets[3 * k : 3 * k + 3]) for k in range(n)
        ) if n else "1=1"
        where_clause = f"({join_clause}) AND ({pred_clause})"
        sql = f"SELECT COUNT(*) FROM {tables} WHERE {where_clause}"
        try:
            cur.execute(sql)
            card = cur.fetchone()[0]
            row[3] = str(card)
        except Exception as e:
            print(f"Query {i} failed: {e}")
            row[3] = "0"
    cur.close()
    conn.close()

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f, delimiter="#", quoting=csv.QUOTE_NONE)
        for r in rows:
            w.writerow(r)
    print(f"Updated {csv_path} with cardinalities.")


def main():
    ap = argparse.ArgumentParser(description="Generate TPC-H queries in train.csv format")
    ap.add_argument("-n", "--num", type=int, default=5000, help="Number of queries")
    ap.add_argument("-o", "--output", type=str, default="data/tpch5k.csv", help="Output CSV path")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--with-joins", action="store_true", dest="with_joins", help="Allow multi-table queries with joins (default: single-table only, no joins)")
    ap.add_argument("--fill-cardinality", action="store_true", help="Run queries against PostgreSQL to fill cardinality")
    ap.add_argument("--db-user", default="postgres", help="PostgreSQL user")
    ap.add_argument("--db-host", default="localhost", help="PostgreSQL host")
    ap.add_argument("--db-port", default="5432", help="PostgreSQL port")
    ap.add_argument("--db-password", default="", help="PostgreSQL password")
    ap.add_argument("--db-name", default="tpch", help="Database name (TPC-H)")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    out = Path(args.output)
    if not out.is_absolute():
        out = project_root / out

    generate_queries(n=args.num, seed=args.seed, out_path=out, single_table_only=not args.with_joins)
    if args.fill_cardinality:
        run_cardinalities(
            out,
            db_user=args.db_user,
            db_host=args.db_host,
            db_port=args.db_port,
            db_password=args.db_password,
            db_name=args.db_name,
        )


if __name__ == "__main__":
    main()
