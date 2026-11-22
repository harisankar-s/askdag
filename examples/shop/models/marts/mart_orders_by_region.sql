select
    customers.region,
    cast(orders.ordered_at as date) as order_date,
    count(*) as orders,
    sum(orders.amount) as revenue
from {{ ref('fct_orders') }} as orders
join {{ ref('dim_customers') }} as customers
    on customers.customer_id = orders.customer_id
group by 1, 2
