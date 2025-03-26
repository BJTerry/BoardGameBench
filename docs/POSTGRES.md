# PostgreSQL Migration Guide

This document provides instructions on setting up PostgreSQL for BoardGameBench.

## Installation

### macOS

Install PostgreSQL using Homebrew:

```bash
brew install postgresql@15
brew services start postgresql@15
```

### Linux

Install PostgreSQL using your package manager:

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install postgresql postgresql-contrib

# RHEL/Fedora
sudo dnf install postgresql-server postgresql-contrib
sudo postgresql-setup --initdb
sudo systemctl start postgresql
```

## Database Setup

1. Create the database and user:

```bash
# Connect to PostgreSQL
psql postgres

# In the PostgreSQL console
CREATE DATABASE bgbench;
CREATE USER bgbench_user WITH PASSWORD 'bgbench_password';
GRANT ALL PRIVILEGES ON DATABASE bgbench TO bgbench_user;

# Connect to the bgbench database to set schema permissions
\c bgbench
GRANT ALL ON SCHEMA public TO bgbench_user;
```

2. Configure the application to use PostgreSQL by creating a `.env` file in the project root:

```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=bgbench
DB_USER=bgbench_user
DB_PASSWORD=bgbench_password
```

## Migrating from SQLite

If you have existing data in SQLite that you want to migrate to PostgreSQL, you can use the migration script:

```bash
poetry run python -m bgbench.migrate_db
```

This will:
1. Read data from your existing SQLite database
2. Create the schema in PostgreSQL
3. Insert the data into PostgreSQL

### Migration Notes

The migration process may skip some records due to foreign key constraints that weren't enforced in SQLite. This is normal and expected. The script will report which records were skipped and why.

For detailed information about the migration, use the `--debug` flag:

```bash
poetry run python -m bgbench.migrate_db --debug
```

If you need to drop existing tables before migrating, use the `--drop` flag:

```bash
poetry run python -m bgbench.migrate_db --drop
```

PostgreSQL enforces data integrity through foreign key constraints. This means:
- Game states must reference valid games
- LLM interactions must reference valid games and players

If SQLite had orphaned records (e.g., game states referring to non-existent games), these will be skipped during migration.

- **Foreign Key Constraints**: PostgreSQL ensures data integrity (e.g., game states must reference valid games). Records violating constraints will be skipped.

## Sequence Synchronization

PostgreSQL uses sequence objects to generate values for auto-incrementing primary keys. When migrating data from SQLite, these sequences need to be synchronized with the existing data to prevent duplicate key errors.

The migration and initialization scripts automatically handle sequence synchronization by:

1. Identifying the maximum ID value for each table
2. Setting the corresponding sequence to start after this maximum value
3. Ensuring new records receive correctly incremented IDs

If you encounter duplicate key errors after migration, you may need to manually reset the sequences:

```sql
-- Example for experiments table
SELECT setval('experiments_id_seq', (SELECT MAX(id) FROM experiments), true);
```

This should be done for all tables with auto-incrementing IDs.

## Troubleshooting

- **Connection Issues**: Ensure PostgreSQL is running with `pg_isready` or `brew services list`
- **Permission Issues**: Check that your user has proper permissions with `\du` in psql
- **Schema Issues**: If you encounter schema errors during migration, you may need to drop and recreate tables
- **Duplicate Key Errors**: These indicate that sequences weren't properly synchronized; use the sequence reset commands above