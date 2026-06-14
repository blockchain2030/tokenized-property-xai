NUMERICAL_FEATURES = [
    "property_value_usd", "annual_rent_usd", "rental_yield_pct", "occupancy_rate",
    "liquidity_ratio", "volatility_index", "trading_volume_usd", "market_momentum",
]

TEXTUAL_FEATURES = [
    "document_type", "document_length_words", "whitepaper_risk_score",
    "disclosure_sentiment", "dominant_risk_clause",
]

VISUAL_FEATURES = [
    "image_count", "property_condition_score", "listing_source",
]

BLOCKCHAIN_FEATURES = [
    "blockchain_network", "token_holders", "monthly_transactions",
    "token_holder_concentration", "transaction_graph_density",
    "average_gas_fee_usd", "wallet_age_days",
]

ALL_MODALITY_FEATURES = NUMERICAL_FEATURES + TEXTUAL_FEATURES + VISUAL_FEATURES + BLOCKCHAIN_FEATURES
TARGET = "investment_suitability"
ID_COLUMNS = ["asset_id", "collection_date"]
