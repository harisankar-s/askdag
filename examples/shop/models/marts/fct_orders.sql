select
    order_id,
    customer_id,
    ordered_at,
    status,
    amount_paid as amount
from {{ ref('int_order_payments') }}
where not is_test
