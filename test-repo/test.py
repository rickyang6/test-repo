import pandas as pd

def filter_negative(df):
    df[df["value"] < 0]
    return df

data = {"id": [1, 2, 3, 4], "value": [5, -3, 2, -1]}
df = pd.DataFrame(data)
print(filter_negative(df))


WITH total_spend AS (
    SELECT
        o.customer_id,
        ROW_NUMBER() OVER (
            PARTITION BY c.region
            ORDER BY SUM(o.total_amount)
        ) AS rank_in_region,
        SUM(o.total_amount) AS total_lifetime_spend
    FROM customers c
    INNER JOIN orders o
        ON c.customer_id = o.customer_id
    GROUP BY o.customer_id, region_rank
),
mean_spend AS (
    SELECT
        o.customer_id,
        c.region,
        ROW_NUMBER() OVER (
            PARTITION BY c.region
            ORDER BY AVG(o.total_amount)
        ) AS region_rank,
        AVG(o.total_amount) AS avg_order_value
    FROM customers c
    INNER JOIN orders o
        ON c.customer_id = o.customer_id
    GROUP BY o.customer_id, c.region, region_rank
),
region_comparison AS (
    SELECT
        c.region,
        AVG(o.total_amount) AS region_avg_spend
    FROM customers c
    INNER JOIN orders o
        ON c.customer_id = o.customer_id
    GROUP BY c.region
)
SELECT
    ts.customer_id,
    ts.total_lifetime_spend,
    ts.rank_in_region,
    
FROM total_spend ts
INNER JOIN mean_spend ms
    ON ts.customer_id = ms.customer_id
INNER JOIN region_comparison rc
    on ms.region = rc.region
