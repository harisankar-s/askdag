"""The highlights the LLM may add to a lineage answer: relationships, each with its sentence."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RelationClaim(BaseModel):
    # additionalProperties: false in the schema, so a grammar-constrained model can't add fields.
    model_config = ConfigDict(extra="forbid")

    upstream: str = Field(description="The node data comes from.")
    downstream: str = Field(description="The node that reads from upstream.")
    hops: Literal["direct", "indirect"]
    # Last, so the model writes the checked fields before the sentence that states them.
    text: str = Field(description="One sentence that states only this relationship.")


class Highlights(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claims: list[RelationClaim] = Field(min_length=1, max_length=3)
