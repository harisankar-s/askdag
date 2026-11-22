with orders as (
    select
        customer_id,
        count(*) as order_count,
        sum(amount) as lifetime_value
    from {{ ref('fct_orders') }}
    group by 1
)

select
    customers.customer_id,
    customers.email,
    customers.country_code,
    countries.country_name,
    countries.region,
    customers.signed_up_at,
    coalesce(orders.order_count, 0) as order_count,
    coalesce(orders.lifetime_value, 0) as lifetime_value
from {{ ref('stg_app__customers') }} as customers
left join {{ ref('country_codes') }} as countries
    on countries.country_code = customers.country_code
left join orders
    on orders.customer_id = customers.customer_id
