import subprocess
import os

scripts_to_run = ["baroclinic_instability.py", "held_suarez.py", "weather_forecast_on_era5.py"]
expected_outputs = {
    "baroclinic_instability.py": ["b.09.raw"],
    "held_suarez.py": ["h.12.raw"],
    "weather_forecast_on_era5.py": ["w.00.raw", "w.01.raw", "w.02.raw"]
}

all_tests_passed = True

for script_name in scripts_to_run:
    print(f"Running {script_name}...")
    try:
        result = subprocess.run(["python3", script_name], capture_output=True, text=True, check=False, timeout=1200) # Added timeout
        if result.returncode != 0:
            print(f"FAILURE: {script_name} failed to execute.")
            print(f"Stdout:\n{result.stdout}")
            print(f"Stderr:\n{result.stderr}")
            all_tests_passed = False
            # Mark expected files as not found for this script
            for raw_file in expected_outputs.get(script_name, []):
                 print(f"INFO: Skipping check for {raw_file} due to script execution failure.")
            continue # Continue to the next script

        for raw_file in expected_outputs.get(script_name, []): # Use .get for safety
            if os.path.exists(raw_file):
                print(f"SUCCESS: {raw_file} found.")
            else:
                print(f"FAILURE: {raw_file} not found for {script_name}.")
                all_tests_passed = False
    except subprocess.TimeoutExpired:
        print(f"FAILURE: {script_name} timed out after 1200 seconds.")
        all_tests_passed = False
        # Mark expected files as not found for this script
        for raw_file in expected_outputs.get(script_name, []):
            print(f"INFO: Skipping check for {raw_file} due to script timeout.")
        continue # Continue to the next script
    except Exception as e:
        print(f"FAILURE: An unexpected error occurred while running {script_name}: {e}")
        all_tests_passed = False
        # Mark expected files as not found for this script
        for raw_file in expected_outputs.get(script_name, []):
            print(f"INFO: Skipping check for {raw_file} due to unexpected error.")
        continue # Continue to the next script


if all_tests_passed:
    print("All raw data generation tests passed!")
else:
    print("Some raw data generation tests failed.")
