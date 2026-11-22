select
    id as customer_id,
    lower(email) as email,
    upper(country) as country_code,
    cast(created_at as timestamp) as signed_up_at
from {{ source('app', 'customers') }}
