select
    id as order_id,
    customer_id,
    cast(created_at as timestamp) as ordered_at,
    status,
    is_test
from {{ source('app', 'orders') }}
