# 11. Web UI: one static HTML page served by FastAPI

**Status:** accepted, 2026-11-21

## Context
`/api/ask` works, but the only ways to try it are `curl` and the eval CLI. A reader of the repo (or an interviewer) should be able to open a page, ask a question and see the answer, the table and how it was produced. The UI must not become a second product: no new runtime, no second copy of the answer logic.

## Decision
- **One file, `src/asklineage/web/index.html`**, with plain JavaScript and about 80 lines of CSS (dark mode through `prefers-color-scheme`). Shipped in the package and read with `importlib.resources`, the same way as `sql/schema.sql`. `GET /` returns it.
- **It only calls the public API.** `POST /api/ask` for everything, and `GET /api/info` for the project names and the configured LLM (provider and model), so the page can say why free text abstains when `LLM_PROVIDER=none`. `/api/info` never returns the API key or the base URL; a test checks both are absent.
- **Two input modes.** Free text needs an LLM. A structured mode (node, direction, depth) works with `LLM_PROVIDER=none`, so a fresh clone has a working demo without a model.
- **Conversation toggle.** The first question sends `start_thread`, later ones send the `thread_id` the server issued (ADR 0009). A clarification button sends `choose` when the thread is waiting for it (`resumable`), otherwise asks again with `node=<unique_id>`, keeping the earlier direction and depth.
- **Every response kind is rendered from the response fields**: lineage (summary, highlights, table, truncation note), clarification (candidate buttons), abstain (reason and message). A collapsible panel lists each attempt (stage, passed, latency, tokens, violations, reply), which is the trace ADR 0010 added.
- **No `innerHTML`.** Server values (node names, model replies, descriptions) go in with `textContent` or `createElement`. A model reply is untrusted text; a node name from a manifest is too. A test greps the page for `innerHTML`, `outerHTML` and `insertAdjacentHTML`.

## Options compared
| | **Static page (chosen)** | Streamlit / Gradio | React + Vite |
|---|---|---|---|
| New dependencies | none | a framework with a large tree | Node, npm, a build toolchain, many packages |
| Processes to run | none extra (FastAPI serves it) | a second server | dev server, or a build step before serving |
| Uses `/api/ask` | yes | usually calls Python directly, bypassing the API, thread locks and response models | yes |
| Effort | one file | least | most |
| Looks | plain but clean | good defaults | whatever you build |
| Fits "clone and run" | yes | needs its own run command and port | needs Node installed |

Streamlit was rejected mainly because it is a second server that would call the flow in-process: the page would then prove something different from what a real client gets.

## Consequences
- **No automated tests for the JavaScript.** Python tests cover `GET /`, `/api/info` and the `innerHTML` rule only. The page is checked by hand against the browser steps in the 8a hand-off. A rendering bug can ship; the API contract it renders is covered by the API tests.
- **No authentication.** A thread ID is a bearer token (ADR 0009), and `/api/info` is public. Auth is Phase 6.
- **One file will not scale.** If the UI grows (project picker, history, column lineage), move to modules or a framework then; the API contract does not change.
- **Not yet in the page:** a project selector (the API accepts `project`), pagination of long tables, and graph drawing.

## Addendum (2026-11-21): edges in `/api/ask` for the graph view
The page will draw the lineage as a graph, but a lineage response only had `table` (each row's `unique_id` and `depth`), which says how far a node is, not what it reads from. Nothing to draw arrows with.

**Added:** `LineageResponse.edges: list[EdgeInfo]`, `EdgeInfo = {parent_id, child_id}` (both `unique_id`s), in the order `Lineage.edges` has them (by parent, then child). The page change is a separate step; `index.html` is unchanged.

1. **The walk's own edges, not a second query.** `lineage()` already runs `EDGES_BETWEEN`: every edge between two of the root and the reached nodes. The eval's ground truth (`TruthWalk.edges`, a stdlib BFS over `depends_on`) computes the same set independently, and `check_lineage` now requires the response's edge set to equal it, with no duplicates; a difference makes `lineage_ok` false and the turn a silent wrong answer, like a wrong table. So the graph is as correct as the table. A query of its own (for example only the edges the walk travelled) would be a second definition to keep in step with the eval.
   *Trade-off:* "between reached nodes" includes cross edges the walk did not travel to reach a node (a depth-2 node that also reads a depth-1 node). For a graph that is what you want, since it is how the models really connect; it makes a dense graph denser.
2. **Shape: a list of `{parent_id, child_id}` by `unique_id`.** IDs, not display names, because display names are not unique across projects and sources. The root is `node.unique_id`; it is not a `table` row, but edges can name it, so the page joins on `{node} ∪ table`.
3. **Always parent -> child**, for upstream and downstream walks. An edge is a fact about the project, not about the question; the page decides which way to lay it out. Flipping for upstream would make the same edge read two ways depending on the request.
4. **No cap in the API.** The walk is already bounded by depth (ADR 0005, at most 25) and its node set; the edge count follows from those. The page will cap what it draws (about 40 nodes) and say so when it does, because what fits on a screen is a display decision, not a contract. On a hub model at a large depth the response can carry a few thousand edges; that is the same order as the table and fine for JSON.
5. **A required field, no default `[]`.** A construction site that forgets edges fails loudly instead of returning a lineage answer with no arrows that the eval would then have to catch. `EdgeInfo` is in the checkpoint serializer allowlist (`STATE_TYPES`, ADR 0009), because the response is stored in the thread state; the walk itself is still not checkpointed.

## Addendum (2026-11-21): the graph view
The page draws each lineage answer as an SVG graph above the table, from `table` and `edges` only.

- **Hand-written SVG, no library.** Cytoscape or dagre would lay out messy graphs better and add zoom and pan, but they come from a CDN (the page would break offline) and need their own ADR. Revisit in Phase 5 if real graphs prove too crowded.
- **Layout: one column per depth.** The start node is at the left for downstream and at the right for upstream, so arrows (always parent -> child) point right in both views. A cross edge to the same or an earlier column leaves the parent's right side and enters the child's right side with a small loop. Labels are cut at 26 characters; the full name is the tooltip.
- **Capped at 40 nodes** (the start node plus the first 39 table rows, which are ordered by depth), with a note saying how many nodes there are. The table always lists all. What fits on a screen is a display decision, not an API one.
- **Click a node to ask about it**: sends `node=<unique_id>` with the same direction and depth (and the thread, if a conversation is on). Enter works on a focused node.
- **Still no JavaScript tests.** The graph is checked by hand; the edges it draws are the ones the eval compares with the manifest.
- **Known limit:** a dense graph (many cross edges) gets tangled, and edges can cross boxes. No edge routing.

## Addendum (2026-11-21): styling pass
The first version showed both input rows at once: `.row { display: flex }` overrides the `hidden` attribute, so the mode toggle did nothing visible. Fixed with `[hidden] { display: none !important; }`. A hand check caught it; no Python test can.
- A segmented Question / Structured control, project and LLM shown as pills (the LLM pill is amber when none is configured), monospace node names, muted "no owner recorded", row hover, focus rings, and a light and dark palette from CSS variables only. About 90 lines of CSS; the "no framework" rule stands.
- Backtick-marked node names in the summary and highlights are shown as `<code>` by splitting the text on backticks and building DOM nodes. Still no `innerHTML`.
