"""A small graph and canned LLM replies shared by the /api/ask tests."""

from helpers import make_node

RAW_ORDERS = make_node("orders", "source", source_name="raw")
ORDERS = make_node("orders")
STG_ORDERS = make_node("stg_orders", owner="finance")
FCT_ORDERS = make_node("fct_orders", owner="finance")
MART_WEEKLY = make_node("mart_weekly")

NODES = [RAW_ORDERS, ORDERS, STG_ORDERS, FCT_ORDERS, MART_WEEKLY]
EDGES = [
    (RAW_ORDERS.unique_id, STG_ORDERS.unique_id),
    (STG_ORDERS.unique_id, FCT_ORDERS.unique_id),
    (FCT_ORDERS.unique_id, MART_WEEKLY.unique_id),
]

QUESTION = "What breaks if I change stg_orders?"


def parse_reply(entity="stg_orders", direction="downstream", depth=None, *, same_node=False):
    return {
        "is_lineage": True,
        "same_node": same_node,
        "entity": entity,
        "direction": direction,
        "depth": depth,
    }


PARSE = parse_reply()
NOT_LINEAGE = {
    "is_lineage": False,
    "same_node": False,
    "entity": None,
    "direction": None,
    "depth": None,
}
HIGHLIGHTS = {
    "claims": [
        {
            "upstream": "stg_orders",
            "downstream": "fct_orders",
            "hops": "direct",
            "text": "`fct_orders` reads directly from `stg_orders`.",
        }
    ]
}
