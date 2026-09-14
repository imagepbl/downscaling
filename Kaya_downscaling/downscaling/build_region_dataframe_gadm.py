"""Build a long dataframe (ISO3, Country name, Region code) for the IAMconsortium
R5/R9/R10 region definitions, taking ISO3 from the GADM level-0 layer (GID_0).

The region YAML is read straight from the common-definitions repo, so it always
reflects the current 'main'. The alias table was verified against GADM's official
country roster, so the "unmatched" report should come back empty; it stays as a
guard against future name changes on either side.

Requires: pyyaml, pandas, geopandas (+ pyogrio recommended).
    pip install pyyaml pandas geopandas pyogrio
"""
import re
import json
import unicodedata
import urllib.request
import ssl
import certifi
from pathlib import Path
import pandas as pd
import geopandas as gpd
import yaml

from tools.general_functions import PRINT_COLORS, apply_root_json

# import debugpy
# debugpy.listen(5678)
# print("Waiting for debugger attach...")
# debugpy.wait_for_client()

groupings = {"R5", "R9", "R10"}

# In the repo file, 'common', 'R5', 'R9' and 'R10' are top-level siblings in a
# YAML list. This finder yields the wanted groups whether they sit at the top
# level or nested inside a 'common' wrapper (and tolerates a dict top level).
def _iter_region_groups(node, wanted=groupings):
    items = node if isinstance(node, list) else [node]
    for entry in items:
        if not isinstance(entry, dict):
            continue
        for name, body in entry.items():
            if name in wanted and isinstance(body, list):
                yield name, body
            elif name == "common" and isinstance(body, list):
                yield from _iter_region_groups(body, wanted=groupings)

def _norm(name):
    """Accent-, case- and punctuation-insensitive key for name matching."""
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"\(.*?\)", "", s.casefold())   # drop "(Dutch part)" etc.
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", s)).strip()
               
def download_IAMC_regions():

    project_dir = Path.cwd()
    print(f"{PRINT_COLORS['green']}Project directory: {project_dir}{PRINT_COLORS['end']}")
    regions_dir = project_dir / "data"/ "processed" / "models"
    regions_dir.mkdir(parents=True, exist_ok=True)
    GADM_dir = project_dir / "data" / "processed" / "GADM"

    # --- 1. Region YAML (local copy) ---------------------------------------------
    dir_common = project_dir / "data" / "input" / "models"
    path_common_yaml = dir_common / "common.yaml"
    print(f"{PRINT_COLORS['green']}Reading IAMC region YAML from {path_common_yaml}{PRINT_COLORS['end']}")
    data = yaml.safe_load(path_common_yaml.read_text(encoding="utf-8"))

    # --- 2. GADM level 0: country name + code, no geometry (fast) ----------------
    settings_file = project_dir / "downscaling" / "settings_data_locations.json"
    with open(settings_file, "r") as f:
        data_files = json.load(f)
    data_files = apply_root_json(data_files, data_files["data_root"])
    dir_GADM_geopackage = Path(data_files["GADM"]["dir_GADM_geopackage"])
    gadm_gpkg_path = dir_GADM_geopackage / "gadm_410-levels.gpkg"
    print(f"{PRINT_COLORS['green']}Reading GADM level-0 countries from {gadm_gpkg_path}{PRINT_COLORS['end']}")
    adm0 = gpd.read_file(gadm_gpkg_path, layer="ADM_0", ignore_geometry=True)  # cols: GID_0, COUNTRY
    adm0 = adm0[~adm0["GID_0"].str.startswith("Z0")]   # drop regions with Z0<number>
    # add North Macedonia (ISO "MKD"), Hong Kong (ISO "HKG", and Macao (ISO "MAC") if that ISO is not included
    if "MKD" not in adm0["GID_0"].values:
        adm0 = pd.concat([adm0, pd.DataFrame({"GID_0": ["MKD"], "COUNTRY": ["North Macedonia"]})], ignore_index=True)
    if "HKG" not in adm0["GID_0"].values:
        adm0 = pd.concat([adm0, pd.DataFrame({"GID_0": ["HKG"], "COUNTRY": ["Hong Kong"]})], ignore_index=True)
    if "MAC" not in adm0["GID_0"].values:
        adm0 = pd.concat([adm0, pd.DataFrame({"GID_0": ["MAC"], "COUNTRY": ["Macao"]})], ignore_index=True)

    # YAML name -> GADM COUNTRY name, only where normalisation alone won't bridge it.
    # (Cabo Verde, Czechia, Timor-Leste, Palestine, Micronesia etc. match directly.)
    ALIAS = {
        "Russian Federation": "Russia", "Viet Nam": "Vietnam", "Brunei Darussalam": "Brunei",
        "Congo": "Republic of the Congo", "Eswatini": "Swaziland", #"North Macedonia": "Macedonia",
        "United States Virgin Islands": "Virgin Islands, U.S.",
        "Sint Maarten (Dutch part)": "Sint Maarten",
    }

    # --- 3. Flatten the R5/R9/R10 groupings into (Country name, Region code) rows -
    print(f"{PRINT_COLORS['green']}Building IAMC region dataframe{PRINT_COLORS['end']}")
    code_keys = ("navigate", "ar6")

    # Create a list of records for each region
    records = []
    region_records = []
    for group_name, group_body in _iter_region_groups(data):
        for number, region in enumerate(group_body, start=1):
            if not isinstance(region, dict):
                continue
            (region_name, region_body), = region.items()
            region_body = region_body or {}
            code = next((region_body[k] for k in code_keys if region_body.get(k)), region_name)
            region_records.append({"category": group_name, "region": code, "number": number})
            for country in (region_body.get("countries") or []):
                records.append({"Country name": country, "Region code": code})

    # Save the region numbers to CSV files
    df_region_numbers = pd.DataFrame(region_records)
    for category, group in df_region_numbers.groupby("category"):
        # remove 'region' ending with 'OWO' if it exists
        group = group[~group["region"].str.endswith("OWO")]
        region_numbers_path = regions_dir / f"IAMC_region_numbers_{category}.csv"
        region_numbers_path.parent.mkdir(parents=True, exist_ok=True)
        group[["region", "number"]].to_csv(region_numbers_path, index=False, sep=";")
        
    if not records:
        raise ValueError("No R5/R9/R10 regions found; check the YAML structure "
                        f"(top-level items: {[list(e)[0] for e in data if isinstance(e, dict)]}).")

    df_IAMC_regions = pd.DataFrame(records)
    print("Distinct region codes:", sorted(df_IAMC_regions["Region code"].unique()))

    # --- 4. Attach ISO3 (= GADM GID_0) via the normalised key --------------------
    print(f"{PRINT_COLORS['green']}Mapping IAMC region countries to GADM ISO3 codes{PRINT_COLORS['end']}")
    # a. Alias + normalise the IAMC names into a match key
    df_IAMC_regions["match_key"] = df_IAMC_regions["Country name"].replace(ALIAS).map(_norm)
    # b. Normalise the GADM names into the same key, keep one row per key
    adm0_keyed = (adm0.assign(match_key=adm0["COUNTRY"].map(_norm))
                .drop_duplicates("match_key", keep="last")
                .loc[:, ["match_key", "GID_0"]])
    # c. Merge on the key and rename GID_0 to ISO3
    df_IAMC_regions = (df_IAMC_regions
                    .merge(adm0_keyed, on="match_key", how="left")
                    .rename(columns={"GID_0": "ISO3"})
                    .drop(columns="match_key"))
    # d. check for unmatched IAMC country names (should be empty; extend ALIAS if GADM names have shifted)
    unmatched = sorted(df_IAMC_regions.loc[df_IAMC_regions["ISO3"].isna(), "Country name"].unique())
    if unmatched:  # should be empty; extend ALIAS if GADM names have shifted
        print("UNMATCHED (extend ALIAS):", unmatched)
        #print("GADM COUNTRY names:\n", "\n".join(sorted(adm0["COUNTRY"])))

    # -- 5. Write the IAMC region dataframe to CSV ---------------------------------
    df_IAMC_regions = df_IAMC_regions[["ISO3", "Country name", "Region code"]]
    country_regions_path = regions_dir / "IAMC_country_to_regions.csv"
    print(f"{PRINT_COLORS['green']}Writing IAMC region dataframe to {country_regions_path}{PRINT_COLORS['end']}")
    df_IAMC_regions.to_csv(country_regions_path, index=False, sep=";")
    print(f"{len(df_IAMC_regions)} rows, {df_IAMC_regions['ISO3'].notna().sum()} with an ISO3 code")

    # split into separate files for R5, R9, R10
    for group in groupings:
        group_path = regions_dir / f"IAMC_country_to_regions_{group}.csv"
        df_group = df_IAMC_regions[df_IAMC_regions["Region code"].str.startswith(group)]
        df_group.to_csv(group_path, index=False, sep=";")

        # compare to iso_to_id_mapping to check for missing ISO3 codes
        iso_to_id_mapping_path = GADM_dir / "iso_to_id_mapping.csv"
        iso_to_id_mapping = pd.read_csv(iso_to_id_mapping_path, sep=";")
        missing_iso3 = set(iso_to_id_mapping["ISO"]) - set(df_group["ISO3"].dropna()) 

        # convert to dataframe and merge with iso_to_id_mapping to get GID_0
        missing_iso3 = pd.DataFrame({"ISO": list(missing_iso3)})
        missing_iso3 = pd.merge(missing_iso3, iso_to_id_mapping, left_on="ISO", right_on="ISO", how="left")
        if not missing_iso3.empty:
            print(f"{PRINT_COLORS['red']}Warning: {len(missing_iso3)} ISO3 codes in iso_to_id_mapping.csv, but not found in {group}:{PRINT_COLORS['end']}")
            print(", ".join(sorted(missing_iso3["ISO"].dropna().astype(str))))
            print(", ".join(sorted(missing_iso3["NAME"].dropna().astype(str))))
        else:
            print(f"{PRINT_COLORS['yellow']}No missing ISO3 codes in {group}{PRINT_COLORS['end']}")

        # save to csv
        missing_iso3_path = regions_dir / f"missing_iso3_{group}.csv"
        missing_iso3.to_csv(missing_iso3_path, index=False, sep=";")

if __name__ == "__main__":
    download_IAMC_regions()
