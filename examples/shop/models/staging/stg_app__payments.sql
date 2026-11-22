select
    id as payment_id,
    order_id,
    amount_cents / 100.0 as amount,
    method,
    status as payment_status
from {{ source('app', 'payments') }}
