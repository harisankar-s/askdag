with sales as (
    select cast(ordered_at as date) as revenue_date, sum(amount) as gross_revenue
    from {{ ref('fct_orders') }}
    group by 1
),

refunds as (
    select cast(orders.ordered_at as date) as revenue_date, sum(refunds.refunded_amount) as refunds
    from {{ ref('fct_refunds') }} as refunds
    join {{ ref('fct_orders') }} as orders
        on orders.order_id = refunds.order_id
    group by 1
)

select
    sales.revenue_date,
    sales.gross_revenue,
    coalesce(refunds.refunds, 0) as refunds,
    sales.gross_revenue - coalesce(refunds.refunds, 0) as net_revenue
from sales
left join refunds
    on refunds.revenue_date = sales.revenue_date
