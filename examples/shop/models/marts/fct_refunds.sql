select
    payment_id,
    order_id,
    amount as refunded_amount
from {{ ref('stg_app__payments') }}
where payment_status = 'refunded'
