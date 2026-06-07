"""
Sales Prediction Project - Dataset Generator
Generates a realistic synthetic dataset for sales prediction modeling.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

np.random.seed(42)

def generate_sales_dataset(n_samples=2000):
    """Generate a comprehensive synthetic sales dataset."""

    # --- Date range (2 years of data) ---
    start_date = datetime(2022, 1, 1)
    dates = [start_date + timedelta(days=i * (730 // n_samples)) for i in range(n_samples)]

    # --- Marketing platforms ---
    platforms = ['Social Media', 'Google Ads', 'Email', 'TV', 'Radio', 'Print']
    platform_weights = [0.30, 0.25, 0.20, 0.12, 0.08, 0.05]

    # --- Customer segments ---
    segments = ['Youth (18-25)', 'Young Adults (26-35)', 'Middle-Aged (36-50)', 'Seniors (51+)', 'Corporate']
    segment_weights = [0.20, 0.30, 0.25, 0.15, 0.10]

    # --- Campaign types ---
    campaign_types = ['Brand Awareness', 'Lead Generation', 'Retargeting', 'Seasonal Promo', 'Product Launch']
    campaign_weights = [0.25, 0.25, 0.20, 0.20, 0.10]

    # --- Regions ---
    regions = ['North', 'South', 'East', 'West', 'Central']

    # --- Base columns ---
    platform_col     = np.random.choice(platforms, n_samples, p=platform_weights)
    segment_col      = np.random.choice(segments, n_samples, p=segment_weights)
    campaign_col     = np.random.choice(campaign_types, n_samples, p=campaign_weights)
    region_col       = np.random.choice(regions, n_samples)

    # --- Advertising spend (USD) ---
    ad_spend = np.random.lognormal(mean=9.5, sigma=1.0, size=n_samples)
    ad_spend = np.clip(ad_spend, 500, 150000)

    # --- Platform multiplier on spend effectiveness ---
    platform_multiplier = {
        'Social Media': 1.30, 'Google Ads': 1.25, 'Email': 1.15,
        'TV': 1.40, 'Radio': 1.05, 'Print': 0.90
    }

    # --- Seasonal index ---
    month = [d.month for d in dates]
    seasonality = np.array([
        0.75, 0.70, 0.85, 0.90, 0.95, 1.00,
        1.05, 1.00, 1.10, 1.15, 1.30, 1.50
    ])
    seasonal_factor = np.array([seasonality[m - 1] for m in month])

    # --- Campaign performance score (0-100) ---
    campaign_score = np.random.beta(a=5, b=2, size=n_samples) * 100

    # --- Competitor index (0-1, higher = more competition) ---
    competitor_index = np.random.uniform(0.1, 0.9, size=n_samples)

    # --- Economic index (macro factor) ---
    economic_index = np.random.normal(loc=1.0, scale=0.1, size=n_samples)
    economic_index = np.clip(economic_index, 0.7, 1.3)

    # --- Discount percentage offered ---
    discount_pct = np.random.uniform(0, 30, size=n_samples)

    # --- Number of marketing channels used simultaneously ---
    channels_used = np.random.randint(1, 6, size=n_samples)

    # --- Customer satisfaction score (from previous campaigns) ---
    csat_score = np.random.normal(loc=3.8, scale=0.5, size=n_samples)
    csat_score = np.clip(csat_score, 1.0, 5.0)

    # --- Website traffic (sessions) ---
    website_traffic = np.random.lognormal(mean=8.5, sigma=1.0, size=n_samples).astype(int)

    # --- Compute Sales (target variable) ---
    p_mult = np.array([platform_multiplier[p] for p in platform_col])
    seg_mult = {
        'Youth (18-25)': 1.00, 'Young Adults (26-35)': 1.20,
        'Middle-Aged (36-50)': 1.15, 'Seniors (51+)': 0.90, 'Corporate': 1.40
    }
    s_mult = np.array([seg_mult[s] for s in segment_col])
    camp_mult = {
        'Brand Awareness': 1.00, 'Lead Generation': 1.15,
        'Retargeting': 1.25, 'Seasonal Promo': 1.30, 'Product Launch': 1.35
    }
    c_mult = np.array([camp_mult[c] for c in campaign_col])

    sales = (
        500
        + 0.35 * ad_spend * p_mult * s_mult * c_mult
        + 120  * seasonal_factor
        + 18   * campaign_score
        - 200  * competitor_index
        + 300  * economic_index
        + 5    * discount_pct
        + 50   * channels_used
        + 80   * csat_score
        + 0.002 * website_traffic
        + np.random.normal(0, 1500, n_samples)  # noise
    )
    sales = np.clip(sales, 100, None)

    # --- Introduce ~3% missing values ---
    df = pd.DataFrame({
        'Date'              : dates,
        'Month'             : month,
        'Platform'          : platform_col,
        'Customer_Segment'  : segment_col,
        'Campaign_Type'     : campaign_col,
        'Region'            : region_col,
        'Ad_Spend'          : ad_spend.round(2),
        'Campaign_Score'    : campaign_score.round(2),
        'Competitor_Index'  : competitor_index.round(3),
        'Economic_Index'    : economic_index.round(3),
        'Discount_Pct'      : discount_pct.round(2),
        'Channels_Used'     : channels_used,
        'CSAT_Score'        : csat_score.round(2),
        'Website_Traffic'   : website_traffic,
        'Seasonal_Factor'   : seasonal_factor.round(3),
        'Sales'             : sales.round(2),
    })

    # Randomly null ~3% of some columns
    for col in ['Ad_Spend', 'Campaign_Score', 'CSAT_Score', 'Discount_Pct']:
        null_idx = np.random.choice(df.index, size=int(0.03 * n_samples), replace=False)
        df.loc[null_idx, col] = np.nan

    # Add ~1% duplicates
    dup_rows = df.sample(frac=0.01, random_state=1)
    df = pd.concat([df, dup_rows], ignore_index=True)

    return df


if __name__ == '__main__':
    df = generate_sales_dataset(2000)
    out_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sales_data.csv')
    df.to_csv(out_path, index=False)
    print(f"Dataset saved: {out_path}  |  Shape: {df.shape}")
    print(df.head())
