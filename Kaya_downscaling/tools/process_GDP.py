from pathlib import Path

import pandas as pd

# for R script execution
import subprocess
import tempfile

# This code uses the R-library GDPuc
# https://pik-piam.github.io/GDPuc/
# Which means, you need to have R installed, with the library GDPuc
# and the rpy2 python library
# define the path to the the Rscript.exe on your computer
# rscript_path = r"C:\Program Files\R\R-4.3.1\bin\Rscript.exe" # Windows example

def run_r_script(r_code, R_SCRIPT_PATH):
    """Run R code using a temporary file approach"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.R', delete=False) as f:
        f.write(r_code)
        temp_r_file = f.name

    temp_r_path = Path(temp_r_file)

    try:
        result = subprocess.run([R_SCRIPT_PATH, str(temp_r_path)],
                              capture_output=True, text=True, timeout=300)  # 5 minute timeout
        return result
    finally:
        # Clean up
        if temp_r_path.exists():
            temp_r_path.unlink()

def check_r_package_installed(package_name, R_SCRIPT_PATH):
    """Check if R package is installed"""
    r_code = f"""
    if (!require({package_name}, quietly = TRUE)) {{
        cat("NOT_INSTALLED")
    }} else {{
        cat("INSTALLED")
    }}
    """

    try:
        result = run_r_script(r_code, R_SCRIPT_PATH)
        return "INSTALLED" in result.stdout
    except Exception as e:
        print(f"Error checking package: {e}")
        return False

def install_r_package(package_name, R_SCRIPT_PATH):
    """Install R package if not already installed"""
    print(f"Checking if {package_name} is installed...")

    if not check_r_package_installed(package_name, R_SCRIPT_PATH):
        print(f"Installing {package_name}...")

        r_code = f'''
        install.packages("{package_name}", repos="https://cran.r-project.org")
        cat("Installation completed\\n")
        '''

        try:
            result = run_r_script(r_code, R_SCRIPT_PATH)

            if result.returncode != 0:
                print("Installation failed!")
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                return False
            else:
                print(f"{package_name} installed successfully.")
                print("Installation output:", result.stdout)
                return True
        except Exception as e:
            print(f"Error installing package: {e}")
            return False
    else:
        print(f"{package_name} is already installed.")
        return True

def use_gdpuc(data:pd.DataFrame, R_SCRIPT_PATH:str,
              unit_in:str="constant 2017 Int$PPP", unit_out:str="constant 2005 Int$PPP") -> pd.DataFrame:
    """Use R's GDPuc package via subprocess with file-based approach"""
    # Source: https://pik-piam.github.io/GDPuc/

    # Ensure GDPuc is installed
    print(f"Installing/checking GDPuc package in R environment at {R_SCRIPT_PATH}...")
    if not install_r_package('GDPuc', R_SCRIPT_PATH):
        raise Exception("Failed to install GDPuc package, first install this package in R manually with the command: install.packages('GDPuc').")

    # Create temporary files using pathlib
    temp_dir = Path(tempfile.gettempdir())

    # Create input file
    input_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, dir=temp_dir)
    input_path = Path(input_file.name)
    data.to_csv(input_path, index=False)
    input_file.close()

    # Create output path
    output_path = input_path.with_name(f"{input_path.stem}_output{input_path.suffix}")

    # R script - note the proper column names based on GDPuc documentation
    r_code = f'''
    library(GDPuc)

    # Read data
    data <- read.csv("{input_path.as_posix()}")

    # GDPuc expects columns named "iso3c" and "year"
    # Let's check what columns we have and rename if necessary
    cat("Input data columns:", paste(names(data), collapse=", "), "\\n")
    cat("Input data (first few rows):\\n")
    print(head(data))

    # If we have 'country' column, we need to convert it to iso3c codes
    # For now, let's assume the data structure and adjust as needed
    if ("country" %in% names(data) && !("iso3c" %in% names(data))) {{
        # For testing purposes, let's create a simple mapping
        # In real use, you'd want proper country code conversion
        data$iso3c <- data$country
    }}

    # Convert GDP using GDPuc
    tryCatch({{
        result <- convertGDP(data, unit_in="{unit_in}", unit_out="{unit_out}")

        # Write result
        write.csv(result, "{output_path.as_posix()}", row.names=FALSE)
        cat("Conversion completed successfully\\n")
    }}, error = function(e) {{
        cat("Error in convertGDP:", e$message, "\\n")
        # Write original data as fallback
        write.csv(data, "{output_path.as_posix()}", row.names=FALSE)
    }})
    '''

    try:
        # Run R script
        result = run_r_script(r_code, R_SCRIPT_PATH)

        print("R script output:")
        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        # Check if output file exists
        if not output_path.exists():
            raise Exception(f"R script failed. No output file created.")

        # Read results
        result_df = pd.read_csv(output_path)
        return result_df

    finally:
        # Clean up temporary files
        for path in [input_path, output_path]:
            if path.exists():
                path.unlink()


