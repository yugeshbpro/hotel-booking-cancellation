"""Production-style scoring for the hotel cancellation model.

The same cleaning and feature code as the notebook, packaged so a booking
system (for example a FastAPI service) can import it.

Usage (after running the notebook once, which saves the model):
    python scoring.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MONTHS = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"], start=1)}

# Columns filled in AFTER the outcome (at check-in or cancellation): never used as inputs
LEAKAGE = ["reservation_status", "reservation_status_date", "assigned_room_type",
           "booking_changes", "required_car_parking_spaces"]


def clean_bookings(df):
    """Rule-based cleaning. Learns nothing from the data, so it is safe for training AND live bookings."""
    df = df.copy()
    df["meal"] = df["meal"].replace({"Undefined": "SC"})            # data dictionary: Undefined = SC (no meal)
    for col in ["market_segment", "distribution_channel"]:
        df[col] = df[col].replace({"Undefined": np.nan})
    df["children"] = pd.to_numeric(df["children"], errors="coerce").fillna(0)
    df["agent"] = df["agent"].fillna("none")                        # no agent = booked directly
    df["has_company"] = df["company"].notna().astype(int)           # 94% missing -> keep only a flag
    df.loc[(df["adr"] < 0) | (df["adr"] > 1000), "adr"] = np.nan    # impossible prices
    df["arrival_date"] = pd.to_datetime(pd.DataFrame({
        "year": df["arrival_date_year"],
        "month": df["arrival_date_month"].map(MONTHS),
        "day": df["arrival_date_day_of_month"]}))
    return df.drop(columns=LEAKAGE + ["company"], errors="ignore")


def add_features(df):
    """Business features that are known at booking time."""
    df = df.copy()
    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]
    df["has_kids"] = ((df["children"] + df["babies"]) > 0).astype(int)
    df["has_agent"] = (df["agent"] != "none").astype(int)
    df["arrival_month"] = df["arrival_date"].dt.month
    df["arrival_weekday"] = df["arrival_date"].dt.day_name()
    return df


class CancellationScorer:
    """Loads the saved pipeline once, then scores one booking at a time."""

    def __init__(self, model_dir="models"):
        model_dir = Path(model_dir)
        self.model = joblib.load(model_dir / "hotel_cancel_model.joblib")
        card = json.loads((model_dir / "model_card.json").read_text())
        self.features = card["features"]
        self.threshold = card["threshold"]

    def score(self, booking: dict) -> dict:
        row = add_features(clean_bookings(pd.DataFrame([booking])))
        prob = float(self.model.predict_proba(row[self.features])[0, 1])
        return {
            "cancel_probability": round(prob, 3),
            "high_risk": prob >= self.threshold,
            "action": "ask for deposit / send reminder" if prob >= self.threshold else "none",
        }


if __name__ == "__main__":
    new_booking = {
        "hotel": "City Hotel", "lead_time": 210, "arrival_date_year": 2017,
        "arrival_date_month": "October", "arrival_date_day_of_month": 14,
        "stays_in_weekend_nights": 1, "stays_in_week_nights": 2, "adults": 2,
        "children": None, "babies": 0, "meal": "BB", "country": "IND",
        "market_segment": "Online TA", "distribution_channel": "TA/TO",
        "is_repeated_guest": 0, "previous_cancellations": 0,
        "previous_bookings_not_canceled": 0, "reserved_room_type": "A",
        "deposit_type": "No Deposit", "agent": "9", "company": None,
        "days_in_waiting_list": 0, "customer_type": "Transient", "adr": 120.0,
        "total_of_special_requests": 0,
    }
    scorer = CancellationScorer()
    print(scorer.score(new_booking))
