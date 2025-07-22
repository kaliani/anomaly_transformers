import pandas as pd
import numpy as np



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

    return all_df


def get_main_info(df, GameId):
   
    min_date = df[df["GameId"] == GameId]["Date"].min()
    max_date = df[df["GameId"] == GameId]["Date"].max()

    return min_date, max_date #get_main_info(historical_week_activity, 420033296)


agg_col = ["BetsCount", "TurnoverInEur", "WinningAmountInEur", "GGRInEur", "NumberOfPlayers"]

time_to_predict = "2024-12-24 22:00:00"
end_date = pd.to_datetime("2024-12-24 22:00:00").tz_localize('UTC')
start_date = end_date - pd.Timedelta(days=7)

games_for_train = pd.read_csv("games_for_train.csv")

games_for_train["Date"] = pd.to_datetime(games_for_train["Date"])
games_for_train = games_for_train[games_for_train["ProductName"] == "Slots"].copy()

games_to_predict = games_for_train[games_for_train["Date"] == time_to_predict].copy()



last_week_activity = games_for_train[(games_for_train["Date"] > start_date) & (games_for_train["Date"] <= end_date)].copy()
last_week_games = last_week_activity["GameId"].unique()
historical_week_activity = games_for_train[(games_for_train["Date"] <= end_date) & (games_for_train["GameId"].isin(last_week_games))].copy()

game_profile = (
    historical_week_activity
    .groupby("GameId")[agg_col]
    .agg(['mean', 'std', 'sum'])
    .fillna(0)
)

game_profile.columns = ["_".join(c) for c in game_profile.columns]

hd = historical_week_activity.groupby(["GameId"])["Date"].agg(["min","max"]).copy()
hd['days_delta'] = (hd["max"] - hd["min"]).dt.days
game_profile  = game_profile.join(hd)

turnover_sum_q = game_profile["TurnoverInEur_sum"].quantile([0.25, 0.5, 0.75])
turnover_std_q = game_profile["TurnoverInEur_std"].quantile([0.25, 0.5, 0.75])

def rule_segment(row):
    if row['TurnoverInEur_sum'] > turnover_sum_q[0.75]:
        if row['TurnoverInEur_std'] > turnover_std_q[0.75]:
            return 'top_volatile'
        else:
            return 'top_stable'
    elif row['TurnoverInEur_sum'] > turnover_sum_q[0.5]:
        return 'mid'
    else:
        return 'low'
game_profile['segment'] = game_profile.apply(rule_segment, axis=1)
game_profile = game_profile.reset_index()


def activity_status(i):
    if i < 48:
        return "not enough"
    else:
        return "enough"
    

def get_x(df):
    window_size = 1
    step_size = 1
    X = []
    y = []

    for i in range(0, len(general_train) - window_size + 1, step_size):
        window = general_train[agg_col].iloc[i:i+window_size].values
        X.append(window)

    X = np.array(X)

    return X
    
game_profile["preprocessing_status"] = game_profile["days_delta"].apply(activity_status)

general_train = get_missing_hours_df(historical_week_activity.query("GameId == 3300891.0"), 
                     get_main_info(historical_week_activity, 3300891.0)[0], 
                     get_main_info(historical_week_activity, 3300891.0)[1]).sort_values(by=["Date"]).copy()


X = get_x(general_train)


np.save("game_train.npy", X)
np.save("game_test.npy", X)

y_test = np.zeros(X.shape[0])
np.save("game_test_label.npy", y_test)

y_train = np.zeros(X.shape[0])
np.save("game_train_label.npy", y_train)