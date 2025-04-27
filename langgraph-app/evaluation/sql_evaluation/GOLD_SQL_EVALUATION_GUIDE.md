# Gold SQL Reference Guide

This document explains how to work with the `gold_sql.txt` file for evaluating SQL query generation.

## Format

The `gold_sql.txt` file contains reference SQL queries for evaluation. Each line follows this format:

```
<sql_query>\t<database_name>
```

For example:
```
SELECT COUNT(*) FROM patient_incidents;	bt5153_gp
```

## Purpose

These gold SQL queries serve as the ground truth for evaluating model-generated SQL queries. The evaluation compares predicted SQL queries against these reference queries to measure accuracy and correctness.

## Running Evaluation

To evaluate predicted SQL queries against the gold standard:

1. Open a terminal and navigate to the SQL evaluation directory
2. Run the evaluation script with the following command:

```bash
python evaluation.py \
  --gold gold_sql.txt \
  --pred predicted_sql.txt \
  --db . \
  --table tables.json \
  --etype match
```

Parameters:
- `--gold`: Path to the gold SQL file (ground truth)
- `--pred`: Path to the predicted SQL file to evaluate
- `--db`: Path to the database directory
- `--table`: Path to the tables.json schema file
- `--etype`: Evaluation type (usually "match")

## Integration with Evaluation Process

1. The `gold_sql.txt` file is read by the evaluation script during comparison
2. Reference queries are parsed and normalized for fair comparison
3. Semantic matching is performed to account for equivalent SQL queries with different syntax

## Creating Gold SQL Files

When creating your own gold SQL file:

1. Ensure each query is valid and tested against your database
2. Include the database name after each query, separated by a tab character
3. Verify that each query returns the expected results
4. Keep the format consistent: one query per line with the tab separator

## Common Issues

- Missing tab separators between query and database name
- SQL syntax errors in reference queries
- Queries that are logically equivalent but syntactically different
- Inconsistent database names

## Best Practices

- Use consistent SQL formatting across all reference queries
- Include a diverse range of query types (SELECT, JOIN, GROUP BY, etc.)
- Verify all queries run successfully on the target database
- Document any assumptions or special cases for particular queries 