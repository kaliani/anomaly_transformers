import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from anomaly_predictor import AnomalyPredictor

# ========== DATA PREP ==========

games_for_train = pd.read_csv("games_for_train.csv")

time_to_predict = "2024-12-24 22:00:00"
start_date = pd.to_datetime(time_to_predict).tz_localize('UTC') - pd.Timedelta(days=7)

games_for_train["Date"] = pd.to_datetime(games_for_train["Date"])
games_to_predict = games_for_train[games_for_train["Date"] == time_to_predict].copy()

agg_col = ["BetsCount", "TurnoverInEur", "WinningAmountInEur", "GGRInEur", "NumberOfPlayers"]

last_week_activity = games_for_train[(games_for_train["Date"] > start_date) & (games_for_train["Date"] <= time_to_predict)].copy()
last_week_games = last_week_activity["GameId"].unique()
historical_week_activity = games_for_train[(games_for_train["Date"] <= time_to_predict) & (games_for_train["GameId"].isin(last_week_games))].copy()

# ========== FUNCTIONS ==========

def get_missing_hours_df(raw_df: pd.DataFrame, start_date: str, end_date: str, date_col: str = "Date") -> pd.DataFrame:
    df = raw_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    if df[date_col].dt.tz is None:
        df[date_col] = df[date_col].dt.tz_localize('UTC')
    else:
        df[date_col] = df[date_col].dt.tz_convert('UTC')
    full_range = pd.date_range(start=start_date, end=end_date, freq='1H', tz='UTC')
    present_dates = df[date_col].unique()
    missing_dates = pd.DatetimeIndex(full_range.difference(present_dates))
    missing_df = pd.DataFrame({date_col: missing_dates})
    for col in df.columns:
        if col != date_col:
            missing_df[col] = pd.NA
    all_df = pd.concat([df, missing_df]).fillna(0)
    all_df = all_df.infer_objects(copy=False)
    return all_df

def get_main_info(df, GameId):
    min_date = df[df["GameId"] == GameId]["Date"].min()
    max_date = df[df["GameId"] == GameId]["Date"].max()
    return min_date, max_date

def get_x(df):
    window_size = 1
    step_size = 1
    X = []
    for i in range(0, len(df) - window_size + 1, step_size):
        window = df[agg_col].iloc[i:i+window_size].values
        X.append(window)
    X = np.array(X)
    return X

def process_game(cl):
    import warnings
    warnings.filterwarnings('ignore')
    general_train = get_missing_hours_df(
        historical_week_activity.query(f"GameId == {cl}"),
        get_main_info(historical_week_activity, cl)[0],
        get_main_info(historical_week_activity, cl)[1]
    ).sort_values(by=["Date"])
    X = get_x(general_train)
    y_train = np.zeros(X.shape[0])
    predictor = AnomalyPredictor(
        model_path="checkpoints/gms_checkpoint.pth",
        data_path="dataset/gms",
        dataset="gms",
        data_arrays=(X, X, y_train),
        win_size=1,
        step=1,
        batch_size=8192,
        num_workers=0,
        anomaly_ratio=0.01
    )
    results_df = predictor.predict()
    general_train["prediction"] = results_df["prediction"].to_list()
    return general_train

# ========== MAIN ==========

if __name__ == "__main__":
    results = Parallel(n_jobs=4)(
        delayed(process_game)(cl) 
        for cl in tqdm(games_to_predict["GameId"].unique()[:10])
    )
    big_check = pd.concat(results, ignore_index=True)
    # Зберегти результати
    big_check.to_csv("all_predictions.csv", index=False)
    print("Done! Results saved to all_predictions.csv")
