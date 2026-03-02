"""
Synthetic Expense Data Generator
Generates realistic Indian expense transaction data for the AI Expense Intelligence System
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

random.seed(42)
np.random.seed(42)

CATEGORIES = {
    "Food": [
        "Swiggy Order", "Zomato Delivery", "Dominos Pizza", "McDonald's",
        "Cafe Coffee Day", "Starbucks", "BigBasket Groceries", "DMart",
        "Blinkit Order", "Restaurant Bill", "Mess Payment", "Canteen",
        "Zepto Groceries", "Subway Sandwich", "KFC Order"
    ],
    "Rent": [
        "House Rent Transfer", "Rent Payment NEFT", "Monthly Rent",
        "PG Accommodation", "Hostel Fee", "Room Rent", "Apartment Rent"
    ],
    "Transport": [
        "Ola Cab", "Uber Ride", "Rapido Bike", "Metro Card Recharge",
        "BMTC Bus Pass", "Petrol Bunk", "Indian Oil Fuel", "HP Petrol",
        "Auto Rickshaw", "Indigo Airlines", "Air India Ticket", "IRCTC Booking",
        "Yulu Bike", "BluSmart EV"
    ],
    "Subscriptions": [
        "Netflix Monthly", "Amazon Prime", "Spotify Premium", "Hotstar",
        "YouTube Premium", "Apple Music", "LinkedIn Premium", "Notion Pro",
        "Adobe Creative", "Microsoft 365", "Zee5 Subscription", "SonyLiv"
    ],
    "Entertainment": [
        "PVR Cinema Tickets", "INOX Movies", "BookMyShow", "Gaming Purchase",
        "Steam Games", "PlayStation Store", "Amusement Park", "Concert Tickets",
        "Sports Event", "IPL Tickets", "Club Entry", "Pub Hopping"
    ],
    "Utilities": [
        "BESCOM Electricity Bill", "Jio Recharge", "Airtel Bill", "BSNL Broadband",
        "Tata Sky DTH", "Water Bill", "Gas Cylinder", "Internet Bill",
        "BBMP Property Tax", "Insurance Premium", "Jio Fiber"
    ],
    "Others": [
        "Amazon Shopping", "Flipkart Order", "Myntra Purchase", "Medical Store",
        "Apollo Pharmacy", "Hospital Bill", "Education Fee", "Book Purchase",
        "Charity Donation", "Gift Purchase", "Home Decor", "ATM Withdrawal"
    ]
}

CATEGORY_AMOUNTS = {
    "Food": (150, 1200),
    "Rent": (8000, 25000),
    "Transport": (50, 800),
    "Subscriptions": (99, 999),
    "Entertainment": (200, 3000),
    "Utilities": (200, 5000),
    "Others": (100, 5000)
}

def generate_transactions(n_months=12, monthly_income=75000):
    transactions = []
    start_date = datetime.now() - timedelta(days=365)
    
    for month in range(n_months):
        month_start = start_date + timedelta(days=30 * month)
        
        # Rent: once a month
        rent_date = month_start + timedelta(days=random.randint(1, 5))
        transactions.append({
            "date": rent_date,
            "description": random.choice(CATEGORIES["Rent"]),
            "amount": round(random.uniform(12000, 18000), 2),
            "category": "Rent"
        })
        
        # Food: 20-30 times/month
        for _ in range(random.randint(20, 30)):
            transactions.append({
                "date": month_start + timedelta(days=random.randint(0, 29)),
                "description": random.choice(CATEGORIES["Food"]),
                "amount": round(random.uniform(*CATEGORY_AMOUNTS["Food"]), 2),
                "category": "Food"
            })
        
        # Transport: 15-25 times/month
        for _ in range(random.randint(15, 25)):
            transactions.append({
                "date": month_start + timedelta(days=random.randint(0, 29)),
                "description": random.choice(CATEGORIES["Transport"]),
                "amount": round(random.uniform(*CATEGORY_AMOUNTS["Transport"]), 2),
                "category": "Transport"
            })
        
        # Subscriptions: 3-6 fixed
        for _ in range(random.randint(3, 6)):
            sub_date = month_start + timedelta(days=random.randint(0, 5))
            transactions.append({
                "date": sub_date,
                "description": random.choice(CATEGORIES["Subscriptions"]),
                "amount": round(random.uniform(*CATEGORY_AMOUNTS["Subscriptions"]), 2),
                "category": "Subscriptions"
            })
        
        # Entertainment: 2-8 times
        for _ in range(random.randint(2, 8)):
            transactions.append({
                "date": month_start + timedelta(days=random.randint(0, 29)),
                "description": random.choice(CATEGORIES["Entertainment"]),
                "amount": round(random.uniform(*CATEGORY_AMOUNTS["Entertainment"]), 2),
                "category": "Entertainment"
            })
        
        # Utilities: 3-5 bills
        for _ in range(random.randint(3, 5)):
            transactions.append({
                "date": month_start + timedelta(days=random.randint(0, 10)),
                "description": random.choice(CATEGORIES["Utilities"]),
                "amount": round(random.uniform(*CATEGORY_AMOUNTS["Utilities"]), 2),
                "category": "Utilities"
            })
        
        # Others: 4-10 misc
        for _ in range(random.randint(4, 10)):
            transactions.append({
                "date": month_start + timedelta(days=random.randint(0, 29)),
                "description": random.choice(CATEGORIES["Others"]),
                "amount": round(random.uniform(*CATEGORY_AMOUNTS["Others"]), 2),
                "category": "Others"
            })
        
        # Inject anomalies in some months
        if month in [3, 7, 10]:
            transactions.append({
                "date": month_start + timedelta(days=random.randint(10, 20)),
                "description": "Electronics Store Purchase",
                "amount": round(random.uniform(15000, 45000), 2),
                "category": "Others",
                "is_anomaly": True
            })
    
    df = pd.DataFrame(transactions)
    df["is_anomaly"] = df.get("is_anomaly", False).fillna(False)
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M")
    df["day_of_week"] = df["date"].dt.dayofweek
    df["monthly_income"] = monthly_income
    df = df.sort_values("date").reset_index(drop=True)
    df["transaction_id"] = [f"TXN{str(i).zfill(5)}" for i in range(len(df))]
    return df

if __name__ == "__main__":
    df = generate_transactions()
    df.to_csv("transactions.csv", index=False)
    print(f"Generated {len(df)} transactions")
    print(df.groupby("category")["amount"].sum().sort_values(ascending=False))
