CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS manifest_loads (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    project_name text NOT NULL,
    dbt_version text NOT NULL,
    schema_version text NOT NULL,
    generated_at timestamptz,
    invocation_id text,
    node_count integer NOT NULL,
    edge_count integer NOT NULL,
    loaded_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS nodes (
    unique_id text PRIMARY KEY,
    project_name text NOT NULL,
    resource_type text NOT NULL CHECK (resource_type IN ('model', 'source', 'seed', 'snapshot')),
    name text NOT NULL,
    source_name text,
    package_name text NOT NULL,
    database_name text,
    schema_name text,
    relation_name text,
    materialization text,
    description text NOT NULL DEFAULT '',
    file_path text,
    tags text[] NOT NULL DEFAULT '{}',
    owner text,
    owner_source text CHECK (owner_source IN ('dbt_meta', 'dbt_group', 'owners_file')),
    raw_code text
);

CREATE INDEX IF NOT EXISTS nodes_project_idx ON nodes (project_name);
CREATE INDEX IF NOT EXISTS nodes_name_idx ON nodes (name);
CREATE INDEX IF NOT EXISTS nodes_name_trgm_idx ON nodes USING gin (name gin_trgm_ops);

CREATE TABLE IF NOT EXISTS lineage_edges (
    parent_id text NOT NULL REFERENCES nodes (unique_id) ON DELETE CASCADE,
    child_id text NOT NULL REFERENCES nodes (unique_id) ON DELETE CASCADE,
    PRIMARY KEY (parent_id, child_id),
    CHECK (parent_id <> child_id)
);

CREATE INDEX IF NOT EXISTS lineage_edges_child_idx ON lineage_edges (child_id);

CREATE TABLE IF NOT EXISTS node_columns (
    node_id text NOT NULL REFERENCES nodes (unique_id) ON DELETE CASCADE,
    name text NOT NULL,
    description text NOT NULL DEFAULT '',
    data_type text,
    PRIMARY KEY (node_id, name)
);

CREATE TABLE IF NOT EXISTS data_tests (
    unique_id text PRIMARY KEY,
    project_name text NOT NULL,
    name text NOT NULL,
    test_type text NOT NULL,
    column_name text,
    attached_node_id text REFERENCES nodes (unique_id) ON DELETE CASCADE,
    severity text
);

CREATE INDEX IF NOT EXISTS data_tests_attached_idx ON data_tests (attached_node_id);
