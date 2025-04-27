# SQL Evaluation

This directory contains tools and scripts for evaluating SQL query generation performance.

## Prerequisites
1. Database setup
2. `gold_sql.txt` - Reference SQL queries
3. `predicted_sql.txt` - Generated from `sql_trim_langgraph.py`
4. `tables.json` - Database schema information
5. `evaluation.py` - Core evaluation script
6. `process_sql.py` - SQL processing utility

## Running Evaluations

Navigate to this directory in your terminal and run:

```bash
python evaluation.py \
  --gold gold_sql.txt \
  --pred predicted_sql.txt \
  --db . \
  --table tables.json \
  --etype match
```

## Additional Features
- The `sql_agent_evaluator.py` script provides extended evaluation capabilities.
- See `EVALUATOR_GUIDE.md` for detailed instructions on advanced evaluation options.
- For information about gold SQL format and best practices, see `GOLD_SQL_GUIDE.md`.

## References
- [Spider: A Large-Scale Human-Labeled Dataset for Text-to-SQL Tasks](https://arxiv.org/pdf/1809.08887)
- [Spider GitHub Repository](https://github.com/taoyds/spider) 