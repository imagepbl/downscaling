# import ee
# from ee import data as ee_data
import time
from pathlib import Path
# import google.auth

import subprocess
import re

'''
Earth Engine Raster Upload – Quick Reference
1. Authenticate (two separate logins required)
   1) Google Cloud (GCP) is storage + infrastructure --> Cloud Storage (your gs:// bucket), billing, project configuration
      and used by 'gcloud storage cp', gsutil (legacy), and Python when using GCS.
   2) Earth Engine (EE) is analysis platform  --> controls access to 'Earth Engine assets' and EE ingestion tasks,
      and is usedd by 'earthengine upload image'.
EE Python API (ee.Initialize()))
    gcloud auth login
    gcloud auth application-default login
    gcloud config set project <project_id> --> unique-nebula-467816-n2
    gcloud auth application-default set-quota-project <project_id>
    earthengine authenticate
    pixi run earthengine set_project <project_id>
2. Create bucket (once)
    gcloud storage buckets create gs://ee-temp-uploads-467816 --project=<project_id>--location=EU
3. Create EE folders (if needed)
    projects/.../assets/emissions
    projects/.../assets/population
    projects/.../assets/gdp
4. Upload workflow
    LOCAL .tif --> gcloud storage cp --> gs://bucket → earthengine upload image → EE asset
5. Test manually
    gcloud storage cp file.tif gs://ee-temp-uploads-467816/
    earthengine upload image --asset_id=projects/... gs://ee-temp-uploads-467816/file.tif
6. Python essentials
    Use subprocess with: pixi run gcloud storage cp
    Then: earthengine upload image
7. Run
    pixi run python upload_results_ee.py
8. Monitor
    pixi run earthengine task list
9. Cleanup
    gcloud storage rm gs://ee-temp-uploads-467816/file.tif
    Key rules- No local uploads → must use GCS- EE auth ≠ GCloud auth- Use gcloud storage (not gsutil
10. Check
    pixi run earthengine task list
'''
# ── Configuration ─────────────────────────────────────────
POLL_INTERVAL = 30
# ──────────────────────────────────────────────────────────

BUCKET = "gs://gee-temp-uploads-downscaling"

def ensure_ee_authenticated():
    result = subprocess.run(["earthengine", "ls"], capture_output=True)
    if result.returncode != 0:
        print("Earth Engine not authenticated. Run:")
        print("earthengine authenticate")
        exit(1)
    else:
        print("Earth Engine authentication verified.")

import subprocess

import time

def wait_for_task(task_id, poll_interval=10):
    while True:
        result = subprocess.run(["earthengine", "task", "info", task_id], capture_output=True, text=True)
        output = result.stdout
        print(output)
        if "COMPLETED" in output:
            return True
        elif "FAILED" in output:
            raise RuntimeError(f"Task failed: {task_id}")
        time.sleep(poll_interval)

def ensure_folder_exists(folder, use_pixi: bool = False):
    cmd = ["earthengine", "create", "folder", folder]
    if use_pixi:
        cmd = ["pixi", "run"] + cmd
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 and "already exists" not in result.stderr.lower():
        raise RuntimeError(f"Failed to create folder: {result.stderr}")

def delete_asset_if_exists(asset_id):
    result = subprocess.run(["earthengine", "asset", "info", asset_id], capture_output=True)
    if result.returncode == 0:
        print(f"Deleting existing asset: {asset_id}")
        subprocess.run(["earthengine", "rm", asset_id], check=True)

def upload_to_gcs(local_path):
    gcs_uri = f"{BUCKET}/{local_path.name}"
    print(f"Uploading to GCS: {local_path} -> {gcs_uri}")

    subprocess.run(["pixi", "run", "gcloud", "storage", "cp", str(local_path), gcs_uri], check=True)

    return gcs_uri

def upload_to_gee(tif_path, asset_id) -> str:
    gcs_uri = upload_to_gcs(tif_path)
    print(f"Uploading {tif_path.name} to {asset_id}...")
    result = subprocess.run(["earthengine", "upload", "image", f"--asset_id={asset_id}", gcs_uri], check=True, text=True, capture_output=True)
    print("Message GEE:", result.stdout)
    match = re.search(r"ID:\s*(\S+)", result.stdout)
    if not match:
        raise RuntimeError("Could not extract task ID from upload output")

    task_id = match.group(1)
    return task_id

def upload_years(scenario: str, source_version:str, var_type: str, local_tif_folder:Path, years:list, ee_asset_folder:str):
    # var_type is one of "all", "emissions", "population", or "gdp"
    # init checks
    # make sure ensure_ee_authenticated() is called
    if not local_tif_folder.exists():
        print(f"Folder not found: {local_tif_folder}")
        return
    upload_asset_folder_root = f"{ee_asset_folder}/{source_version}"
    ensure_folder_exists(upload_asset_folder_root, True)
    upload_asset_folder = f"{ee_asset_folder}/{source_version}/{scenario}"
    ensure_folder_exists(upload_asset_folder, True)

    # determine which files to upload and filter by variable type
    all_tif_files = list(local_tif_folder.glob("*.tif"))
    selected_years_tif_files = [f for f in all_tif_files if any(str(year) in f.stem for year in years)]
    selected_years_tif_files = sorted(selected_years_tif_files)
    if var_type != "all":
        selected_tif_files = [f for f in selected_years_tif_files if var_type in f.stem.lower()]
    else:
        selected_tif_files = selected_years_tif_files

    # upload each selected file
    if not selected_tif_files:
        print(f"No .tif files found for years: {years}")
    else:
        task_ids = []
        for tif_path in selected_tif_files:
            asset_id = f"{upload_asset_folder}/{tif_path.stem}"
            delete_asset_if_exists(asset_id)
            task_id = upload_to_gee(tif_path, asset_id)
            task_ids.append(task_id)
        print(f"Uploading {len(task_ids)} tasks")
        # Wait for all tasks to complete, and check them one-by-one”
        for i, task_id in enumerate(task_ids, 1):
            print(f"Waiting for task {i}/{len(task_ids)}: {task_id}")
            wait_for_task(task_id)
        print("All uploads completed.")

if __name__ == "__main__":
    # To create bucket: gsutil mb -p unique-nebula-467816-n2 gs://ee-temp-uploads
    print("If this fails, run first for authtication in terminal: 'earthengine authenticate' or 'pixi run earthengine set_project <project_id>'")
    ensure_ee_authenticated()
    scenario = "SSP2-1150F"
    #tif_path = Path("K:/PythonWork/downscaling/data/processed/2UP_GHSL_2024_M3_Murakami_version_2021_1_EDGAR_2024/IMAGE_ELV-SSP2-1150F/Population_ELV-SSP2-1150F_2020.tif")
    tif_path = Path("K:/PythonWork/downscaling/data/processed/2UP_GHSL_2024_M3_Murakami_version_2021_1_EDGAR_2024/IMAGE_ELV-SSP2-1150F/Emissions_CO2_Excl_shipping_aviation_AFOLU_harmonised_IMAGE_ELV-SSP2-1150F_2020.tif")
    asset_id = f"{EE_ASSET_FOLDER}/{scenario}_{tif_path.stem}"
    upload_to_gee(tif_path, asset_id)
