-- Illustrative SQL only. Table and column names are fictional.
-- Anchor the observation window to a scoring date and exclude later events.

WITH activity_30d AS (
    SELECT
        subscriber_id,
        COUNT(DISTINCT activity_date) AS active_days_30d,
        COUNT(*) AS activity_events_30d,
        MAX(activity_date) AS last_activity_date
    FROM synthetic_activity_events
    WHERE activity_date >= :window_start
      AND activity_date <  :scoring_date
    GROUP BY subscriber_id
)
SELECT
    s.subscriber_id,
    s.tenure_days,
    COALESCE(a.active_days_30d, 0) AS active_days_30d,
    COALESCE(a.activity_events_30d, 0) AS activity_events_30d,
    CASE
        WHEN a.last_activity_date IS NULL THEN NULL
        ELSE DATE_DIFF('day', a.last_activity_date, :scoring_date)
    END AS days_since_last_activity
FROM synthetic_subscribers s
LEFT JOIN activity_30d a
    ON s.subscriber_id = a.subscriber_id;
