"""Check that supervised features cannot see their own held-out labels."""

import numpy as np
import pandas as pd

from house_prices.demo import make_data
from house_prices.features import FrameEncoder, HouseFeatures, PeerComparables
from house_prices.schema import RAW_FEATURES


def test_target_encoding_uses_only_inner_training_labels():
    X = pd.DataFrame({"cat": [f"unique_{i}" for i in range(20)], "number": np.arange(20)})
    y = np.arange(20, dtype=float)
    first = FrameEncoder(folds=4, seed=7).fit_transform(X, y)
    changed = y.copy()
    changed[0] = 1000000
    second = FrameEncoder(folds=4, seed=7).fit_transform(X, changed)
    assert first.loc[0, "cat_encoded"] == second.loc[0, "cat_encoded"]
    encoder = FrameEncoder(folds=4).fit(X, y)
    new = X.iloc[:1].copy()
    new["cat"] = "unseen"
    assert encoder.transform(new).iloc[0]["cat_encoded"] == y.mean()


def test_neighborhood_statistics_are_frozen_for_inference():
    frame = make_data(50)
    X = frame[RAW_FEATURES]
    features = HouseFeatures("extended").fit(X)
    a = X.iloc[:2].copy()
    a["Neighborhood"] = "NewNeighborhood"
    before = features.transform(a)
    a.iloc[1, a.columns.get_loc("TotalBsmtSF")] = 1000000
    after = features.transform(a)
    assert before.iloc[0]["SF_Rel_to_Nbhd"] == after.iloc[0]["SF_Rel_to_Nbhd"]
    assert features.neighborhood_sf_ == HouseFeatures("extended").fit(X).neighborhood_sf_


def test_peer_price_excludes_own_label():
    frame = make_data(30)
    X = frame[RAW_FEATURES]
    y = frame.SalePrice.to_numpy()
    first = PeerComparables(folds=3, log_target=False).fit_transform(X, y)
    changed = y.copy()
    changed[0] *= 100
    second = PeerComparables(folds=3, log_target=False).fit_transform(X, changed)
    assert first.iloc[0]["Peer_Comp_Price"] == second.iloc[0]["Peer_Comp_Price"]


def test_missing_columns_are_dropped_from_training_policy_only():
    frame = make_data(30)[RAW_FEATURES]
    frame["LotFrontage"] = np.nan
    transformer = HouseFeatures("original", "drop").fit(frame)
    assert "LotFrontage" not in transformer.transform(frame)
    assert "SalePrice" not in transformer.transform(frame)
    assert "Id" not in transformer.transform(frame)
