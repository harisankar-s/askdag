# 6. A narrow LLM provider interface, OpenAI-compatible first, off by default

**Status:** accepted, 2026-11-21

## Context
Step 5 adds an LLM-written summary around the lineage table. The original plan started with AWS Bedrock, but the author has no Bedrock account, and anyone cloning a public reference repo should be able to run it without one or a credit card. Enterprises adopting it will bring their own model: Bedrock, Azure OpenAI, an internal gateway. The model has to be a configuration choice, not a code change.

## Decision
- The answer layer depends on one Protocol, `LLMProvider.complete_json(system, prompt, output)`, which returns a `Completion` holding a Pydantic object validated against `output`. Adapters turn their SDK errors into `LLMError`, and a reply that doesn't validate into `LLMOutputError`.
- The first real adapter speaks the OpenAI chat-completions protocol. That one adapter covers local Ollama (free), vLLM, LiteLLM and many hosted APIs, chosen by `LLM_BASE_URL`, `LLM_MODEL` and `LLM_API_KEY`.
- `LLM_PROVIDER=none` is the default: summaries come from a fixed template built from the query result. A fresh clone works with no model at all, and so does CI.
- `ScriptedProvider` replays canned replies through the same parser, so answer and grounding logic is tested without network calls.
- The documented local model is `qwen2.5:7b` on Ollama.

## Consequences
- Switching models is a `.env` change. Answer *quality* is not portable: small local models fail the grounding check more often. The check and its template fallback (ADR 0007) keep wrong text away from users, and the eval reports a first-attempt failure rate per model.
- Not every OpenAI-compatible server enforces a JSON schema, so valid structure is checked by us (Pydantic), not guaranteed by the server. An invalid reply is treated like a failed grounding check.
- The JSON Schema is always written into the system prompt as well as sent as `response_format`, so servers that ignore the format hint still see it. `LLM_RESPONSE_FORMAT` drops to `json_object` or `off` for servers that reject `json_schema`.
- Requests send `temperature: 0` to make eval runs as repeatable as the server allows. Some reasoning models reject `temperature`; a setting to omit it gets added when someone hits that.
- The client retries once, not twice: a slow local model can already use most of the 60-second timeout.
- Native adapters (Bedrock, Anthropic) can be added behind the same Protocol when someone needs features the OpenAI protocol doesn't expose.

## Alternatives considered
- **Bedrock first:** needs an AWS account and model access before anyone can try the repo.
- **LangChain chat models as the interface:** convenient once LangGraph arrives in Step 6, but it ties the answer layer and eval harness to LangChain's message types for a one-call need.
- **LiteLLM as a library:** many providers out of the box, but a large dependency, and the OpenAI protocol already reaches LiteLLM's proxy if a team runs one.
- **A hosted API as the default:** cheap per call, but still needs an account and a key, so not zero-setup.
