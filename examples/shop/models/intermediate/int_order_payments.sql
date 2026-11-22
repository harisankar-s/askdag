select
    orders.order_id,
    orders.customer_id,
    orders.ordered_at,
    orders.status,
    orders.is_test,
    sum(case when payments.payment_status = 'success' then payments.amount end) as amount_paid
from {{ ref('stg_app__orders') }} as orders
left join {{ ref('stg_app__payments') }} as payments
    on payments.order_id = orders.order_id
group by 1, 2, 3, 4, 5
