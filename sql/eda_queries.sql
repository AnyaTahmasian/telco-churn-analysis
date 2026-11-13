-- Churn is 0/1, so AVG(Churn) is the churn rate.

-- name: overall
SELECT COUNT(*)                     AS customers,
       SUM(Churn)                   AS churned,
       ROUND(AVG(Churn) * 100, 1)   AS churn_rate_pct
FROM customers;

-- name: churn_by_contract
SELECT Contract,
       COUNT(*)                     AS customers,
       ROUND(AVG(Churn) * 100, 1)   AS churn_rate_pct,
       ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charges
FROM customers
GROUP BY Contract
ORDER BY churn_rate_pct DESC;

-- name: churn_by_tenure_band
SELECT CASE
           WHEN tenure < 12 THEN '0-11 months'
           WHEN tenure < 24 THEN '12-23 months'
           WHEN tenure < 48 THEN '24-47 months'
           ELSE '48+ months'
       END                          AS tenure_band,
       COUNT(*)                     AS customers,
       ROUND(AVG(Churn) * 100, 1)   AS churn_rate_pct
FROM customers
GROUP BY tenure_band
ORDER BY MIN(tenure);

-- name: churn_by_payment_method
SELECT PaymentMethod,
       COUNT(*)                     AS customers,
       ROUND(AVG(Churn) * 100, 1)   AS churn_rate_pct
FROM customers
GROUP BY PaymentMethod
ORDER BY churn_rate_pct DESC;

-- name: fiber_without_support
SELECT TechSupport,
       COUNT(*)                     AS customers,
       ROUND(AVG(Churn) * 100, 1)   AS churn_rate_pct
FROM customers
WHERE InternetService = 'Fiber optic'
GROUP BY TechSupport
ORDER BY churn_rate_pct DESC;

-- name: monthly_revenue_at_risk
SELECT Contract,
       ROUND(SUM(CASE WHEN Churn = 1 THEN MonthlyCharges ELSE 0 END), 0) AS lost_monthly_revenue,
       ROUND(100.0 * SUM(CASE WHEN Churn = 1 THEN MonthlyCharges ELSE 0 END)
             / SUM(MonthlyCharges), 1)                                   AS pct_of_contract_revenue
FROM customers
GROUP BY Contract
ORDER BY lost_monthly_revenue DESC;
