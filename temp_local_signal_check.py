from kalshi_btc_bot.cli import build_engine
from kalshi_btc_bot.dashboard import load_recent_snapshots, build_live_signal_table
from kalshi_btc_bot.storage.snapshots import SnapshotStore
settings, engine = build_engine(attach_calibration=False, attach_repricing_profile=False)
store = SnapshotStore(str(settings.collector['sqlite_path']))
snapshots = load_recent_snapshots(store, series_ticker=str(settings.collector['series_ticker']), lookback_hours=1, limit=2000)
print('recent_snapshots', len(snapshots))
signal_table = build_live_signal_table(engine=engine, snapshots=snapshots, volatility_window=int(settings.data['volatility_window']), annualization_factor=float(settings.data['annualization_factor']))
print('signal_rows', len(signal_table))
if len(signal_table):
    cols = ['market_ticker','minutes_to_expiry','yes_ask_cents','decision','action','side','edge','quality_score','reason_bucket','reason','spread_cents','model_probability','market_probability']
    keep = [c for c in cols if c in signal_table.columns]
    print(signal_table[keep].head(20).to_string(index=False))
    in_band = signal_table[(signal_table['yes_ask_cents'] >= 25) & (signal_table['yes_ask_cents'] <= 60)] if 'yes_ask_cents' in signal_table.columns else signal_table.iloc[0:0]
    print('in_band_rows', len(in_band))
    if len(in_band):
        print(in_band[keep].head(20).to_string(index=False))
