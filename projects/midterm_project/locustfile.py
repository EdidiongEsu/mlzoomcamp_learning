import numpy as np
from locust import task
from locust import between
from locust import HttpUser

sample = {
    "age": 12,
    "gender": "m",
    "region_category": "missing",
    "membership_category": "silver_membership",
    "joined_through_referral": "no",
    "preferred_offer_types": "credit/debit_card_offers",
    "medium_of_operation": "smartphone",
    "internet_option": "fiber_optic",
    "last_visit_time": "13:38:19",
    "days_since_last_login": 24.0,
    "avg_time_spent": 121.82,
    "avg_transaction_value": 30385.88,
    "avg_frequency_login_days": 25.0,
    "points_in_wallet": 698.62,
    "used_special_discount": "no",
    "offer_application_preference": "yes",
    "past_complaint": "yes",
    "complaint_status": "solved",
    "feedback": "too_many_ads"
}


class CreditRiskTestUser(HttpUser):
    """
    Usage:
        Start locust load testing client with:
            locust -H http://localhost:3000
        Open browser at http://0.0.0.0:8089, adjust desired number of users and spawn
        rate for the load test from the Web UI and start swarming.
    """

    @task
    def classify(self):
        self.client.post("/classify", json=sample)

    wait_time = between(0.01, 2)
