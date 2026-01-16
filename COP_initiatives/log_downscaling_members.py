import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

from ..historical_data import read_process_historical_data as hist_data

def LogResults(df_hist_IMAGE_Countries, Sector, Year):
    # create log file to check for which countries there was historical data available
    df_log = df_hist_IMAGE_Countries.copy()
    df_log["ICI"] = Sector
    mask = df_log["Year"]==Year
    df_log = df_log[mask]
    df_log = df_log.loc[:, ["ISO3", "Country_name", "ICI", "Value"]]
    df_log.rename(columns={"Country_name":"Country"}, inplace=True)

    return df_log

def SaveResults(df_ISO_countries_downscaling:pd.DataFrame, dir:str, hist_year:int):
    # process ISO codes used for downscaling + add weight based on 2020 total Kyoto emissions
    # 1. determine ISO codes of historical data used for downscaling
    print("- Saving log file for downscaling members...")
    ISO_codes = df_ISO_countries_downscaling["ISO3"].unique()
    pd_ISOC_codes = pd.DataFrame(ISO_codes, columns=["ISO3"])
    # 2. calculate weight country based on Kyoto emissions
    print("- Calculating weights based on {hist_year} Kyoto emissions...")
    df_Kyoto = hist_data.ReadProcessHistoricalData("Emissions|Kyoto Gases", hist_year)
    mask = (df_Kyoto["Year"]==2020) & (df_Kyoto["ISO3"].isin(ISO_codes)) & (~(df_Kyoto["ISO3"]=="WLD"))
    df_Kyoto_2020 = df_Kyoto.loc[mask, ["ISO3", "Value"]]
    df_Kyoto_2020["Weight"] = 100*df_Kyoto_2020["Value"]/(df_Kyoto_2020["Value"].sum())
    df_Kyoto_2020.drop(["Value"], axis=1, inplace=True)
    # 3. Create downscaling log file showing which country/ISO are used for each initiative
    log_ISO_downscaling = pd.merge(pd_ISOC_codes, df_ISO_countries_downscaling, on="ISO3", how="left")
    log_ISO_downscaling = pd.merge(df_ISO_countries_downscaling, df_Kyoto_2020, on="ISO3", how="left")
    log_ISO_downscaling["Value"] = np.where(log_ISO_downscaling["Value"]==0, 0, 1)
    log_ISO_downscaling = log_ISO_downscaling.pivot_table(index=["ISO3", "Country", "Weight"], columns="ICI", values="Value", aggfunc="first")
    log_ISO_downscaling = log_ISO_downscaling.sort_values(by="Weight", ascending=False)
    log_ISO_downscaling.reset_index(inplace=True)
    log_ISO_downscaling = log_ISO_downscaling.fillna("NA")
    now = datetime.now()

    if Path(dir).exists()==False:
        Path(dir).mkdir(parents=True, exist_ok=True)
    log_ISO_downscaling.to_excel(f"{dir}/ISO_countries_downscaling_{now:%Y_%m_%d__%H_%M}.xlsx")
