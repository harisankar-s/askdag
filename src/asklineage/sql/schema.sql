CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS vector;

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

-- /api/ask conversations. LangGraph's checkpoint tables hold each thread's state; this table
-- says which threads exist and when they were last used, for 404s and pruning (ADR 0009).
CREATE TABLE IF NOT EXISTS ask_threads (
    thread_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at timestamptz NOT NULL DEFAULT now(),
    last_used_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ask_threads_last_used_idx ON ask_threads (last_used_at);

-- Embedded chunks of node text for semantic search (ADR 0012). No foreign key to nodes:
-- load-manifest replaces every node, and a cascade would throw the embeddings away each time.
-- `asklineage embed` re-embeds only changed chunks and deletes those of nodes that are gone.
-- The dimension must match EMBED_DIMENSIONS; changing it needs a manual migration.
CREATE TABLE IF NOT EXISTS node_chunks (
    node_id text NOT NULL,
    project_name text NOT NULL,
    kind text NOT NULL CHECK (kind IN ('summary', 'columns', 'sql')),
    ordinal integer NOT NULL CHECK (ordinal >= 0),
    content text NOT NULL,
    content_sha text NOT NULL,
    model text NOT NULL,
    embedding vector(768) NOT NULL,
    embedded_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (node_id, kind, ordinal)
);

CREATE INDEX IF NOT EXISTS node_chunks_project_idx ON node_chunks (project_name);

-- A search index of incident write-ups, not their system of record (ADR 0015). In production
-- the record is a ticketing tool; here the Markdown corpus stands in for its export (ADR 0014).
-- node_ids are names detected in the text, never the front-matter: that is the eval's ground
-- truth, and reading it here would grade the system against its own input. No foreign key to
-- nodes, for the same reason as node_chunks. `load-docs` replaces a project's rows at once.
CREATE TABLE IF NOT EXISTS documents (
    project_name text NOT NULL,
    doc_id text NOT NULL,
    source text NOT NULL,
    url text NOT NULL,
    doc_type text NOT NULL,
    title text NOT NULL,
    doc_date date NOT NULL,
    severity text,
    node_ids text[] NOT NULL DEFAULT '{}',
    body text NOT NULL,
    loaded_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (project_name, doc_id)
);

CREATE INDEX IF NOT EXISTS documents_node_ids_idx ON documents USING gin (node_ids);

-- One chunk per section. Vectors are optional so EMBED_PROVIDER=none still gets keyword search;
-- the dimension must match node_chunks and EMBED_DIMENSIONS.
CREATE TABLE IF NOT EXISTS doc_chunks (
    project_name text NOT NULL,
    doc_id text NOT NULL,
    ordinal integer NOT NULL CHECK (ordinal >= 0),
    section text NOT NULL,
    content text NOT NULL,
    search tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    model text,
    embedding vector(768),
    PRIMARY KEY (project_name, doc_id, ordinal),
    FOREIGN KEY (project_name, doc_id) REFERENCES documents ON DELETE CASCADE,
    CHECK ((model IS NULL) = (embedding IS NULL))
);

CREATE INDEX IF NOT EXISTS doc_chunks_search_idx ON doc_chunks USING gin (search);
