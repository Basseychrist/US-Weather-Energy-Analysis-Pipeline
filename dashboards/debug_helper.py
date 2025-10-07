import os
import sys
import json
import yaml
import subprocess
from datetime import datetime

def diagnose_pipeline_issue(project_root=None):
    """Run diagnostics on the pipeline setup and execution environment.
    Returns a dict with diagnostic results and suggestions.
    """
    if not project_root:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "project_root": project_root,
        "directories": {},
        "files": {},
        "environment": {},
        "suggestions": []
    }
    
    # Check directory structure
    for dir_name in ['data', 'data/raw', 'data/processed', 'logs', 'config']:
        dir_path = os.path.join(project_root, *dir_name.split('/'))
        exists = os.path.exists(dir_path)
        results["directories"][dir_name] = {
            "exists": exists,
            "path": dir_path
        }
        if not exists:
            results["suggestions"].append(f"Create missing directory: {dir_path}")
            try:
                os.makedirs(dir_path, exist_ok=True)
                results["directories"][dir_name]["created"] = True
            except Exception as e:
                results["directories"][dir_name]["error"] = str(e)
                results["suggestions"].append(f"Failed to create {dir_path}: {str(e)}")
    
    # Check critical files
    for file_name in ['main.py', 'config/config.yaml', 'config/config.example.yaml']:
        file_path = os.path.join(project_root, *file_name.split('/'))
        exists = os.path.exists(file_path)
        results["files"][file_name] = {
            "exists": exists,
            "path": file_path
        }
        if exists:
            try:
                size = os.path.getsize(file_path)
                results["files"][file_name]["size"] = size
                if size == 0:
                    results["suggestions"].append(f"File {file_name} exists but is empty")
            except Exception as e:
                results["files"][file_name]["error"] = str(e)
    
    # Check configuration
    if results["files"].get("config/config.yaml", {}).get("exists", False):
        try:
            with open(os.path.join(project_root, 'config', 'config.yaml'), 'r') as f:
                config = yaml.safe_load(f)
            
            # Check API keys
            has_noaa_token = bool(config.get('noaa', {}).get('token'))
            has_eia_key = bool(config.get('eia', {}).get('api_key'))
            results["environment"]["has_noaa_token"] = has_noaa_token
            results["environment"]["has_eia_key"] = has_eia_key
            
            if not has_noaa_token:
                results["suggestions"].append("Missing NOAA API token in config/config.yaml")
            if not has_eia_key:
                results["suggestions"].append("Missing EIA API key in config/config.yaml")
            
            # Check cities configuration
            cities = config.get('cities', [])
            results["environment"]["cities_configured"] = len(cities)
            if not cities:
                results["suggestions"].append("No cities configured in config/config.yaml")
        except Exception as e:
            results["environment"]["config_parse_error"] = str(e)
            results["suggestions"].append(f"Error parsing config/config.yaml: {str(e)}")
    
    # Try running the pipeline
    main_py = os.path.join(project_root, 'main.py')
    if os.path.exists(main_py):
        try:
            cmd = [sys.executable, main_py, 'realtime']
            proc = subprocess.run(
                cmd, cwd=project_root, capture_output=True, text=True, timeout=120
            )
            results["pipeline_run"] = {
                "returncode": proc.returncode,
                "stdout_sample": proc.stdout[:500] + ("..." if len(proc.stdout) > 500 else ""),
                "stderr_sample": proc.stderr[:500] + ("..." if len(proc.stderr) > 500 else ""),
                "command": " ".join(cmd)
            }
            
            # Enhanced error detection - check for API errors even if return code is 0
            output_combined = proc.stdout + proc.stderr
            
            # Look for common API errors
            if "400 Client Error" in output_combined:
                results["suggestions"].append("NOAA API returned 400 Client Error - check date ranges and parameters")
                
                # Check for future date requests (a very common issue)
                import re
                date_matches = re.findall(r'(\d{4}-\d{2}-\d{2})', output_combined)
                if date_matches:
                    try:
                        today = datetime.now().date()
                        for date_str in date_matches:
                            date_parts = [int(p) for p in date_str.split('-')]
                            if len(date_parts) == 3:
                                request_date = datetime(date_parts[0], date_parts[1], date_parts[2]).date()
                                if request_date > today:
                                    results["environment"]["future_date_request"] = True
                                    results["suggestions"].append(
                                        f"Attempting to fetch data for future date {date_str} - NOAA has no data from the future!"
                                    )
                    except Exception:
                        pass
                
                if "www.ncdc.noaa.gov" in output_combined:
                    results["environment"]["noaa_api_error"] = True
                    results["suggestions"].append("NOAA API request failed - validate your token and request parameters")
            
            if "401" in output_combined and "Unauthorized" in output_combined:
                if "ncdc.noaa.gov" in output_combined:
                    results["environment"]["noaa_auth_error"] = True
                    results["suggestions"].append("NOAA API authentication failed - check your token")
                if "api.eia.gov" in output_combined:
                    results["environment"]["eia_auth_error"] = True
                    results["suggestions"].append("EIA API authentication failed - check your API key")
            
            if "429" in output_combined:
                results["suggestions"].append("Rate limit exceeded - consider adding delays between API requests")
            
            if "503" in output_combined or "502" in output_combined:
                results["suggestions"].append("API service unavailable - the data provider's server may be down")
            
            # Check for date-related issues
            if "date" in output_combined.lower() and "invalid" in output_combined.lower():
                results["suggestions"].append("Invalid date format or range detected - check your date parameters")
            
            if proc.returncode != 0:
                results["suggestions"].append(f"Pipeline failed with return code {proc.returncode}")
                if "ModuleNotFoundError" in proc.stderr:
                    results["suggestions"].append("Missing Python modules. Try: pip install -r requirements.txt")
                if "KeyError" in proc.stderr:
                    results["suggestions"].append("Configuration key error - check config.yaml format")
            
            # Check output file
            output_path = os.path.join(project_root, 'data', 'processed', 'weather_energy_data.csv')
            output_exists = os.path.exists(output_path)
            results["files"]["output_csv"] = {
                "exists": output_exists,
                "path": output_path
            }
            if output_exists:
                results["files"]["output_csv"]["size"] = os.path.getsize(output_path)
                if os.path.getsize(output_path) == 0:
                    results["suggestions"].append("Output CSV exists but is empty")
            else:
                results["suggestions"].append("Pipeline did not create output CSV file")
                
            # If output file doesn't exist but return code is 0, it's a partial success/silent failure
            if proc.returncode == 0 and not output_exists:
                results["suggestions"].append(
                    "Pipeline returned success but did not create output file - check logs for warnings"
                )
        except subprocess.TimeoutExpired:
            results["pipeline_run"] = {"error": "Pipeline timed out after 120 seconds"}
            results["suggestions"].append("Pipeline execution timed out - possibly hanging")
        except Exception as e:
            results["pipeline_run"] = {"error": str(e)}
            results["suggestions"].append(f"Error running pipeline: {str(e)}")
    
    # Add an API connection test section
    results["api_tests"] = {}
    try:
        # Test NOAA API connection if token exists
        if results["environment"].get("has_noaa_token"):
            # Don't make actual API calls here, just report that test should be run manually
            results["api_tests"]["noaa"] = {
                "message": "Found NOAA token, but manual testing recommended"
            }
            results["suggestions"].append(
                "Consider testing NOAA API manually with: "
                "curl -H 'token: YOUR_TOKEN' https://www.ncdc.noaa.gov/cdo-web/api/v2/datasets"
            )
        
        # Test EIA API connection if key exists
        if results["environment"].get("has_eia_key"):
            results["api_tests"]["eia"] = {
                "message": "Found EIA API key, but manual testing recommended"
            }
            results["suggestions"].append(
                "Consider testing EIA API manually"
            )
    except Exception as e:
        results["api_tests"]["error"] = str(e)
    
    # Add a solution section for common issues
    if not results["suggestions"]:
        results["solutions"] = ["No issues detected. If problems persist, try generating sample data."]
    else:
        results["solutions"] = []
        
        # Future date detection and solution
        if results["environment"].get("future_date_request"):
            results["solutions"].append(
                "🔍 ISSUE DETECTED: Attempting to fetch data from the future! \n"
                "SOLUTION: \n"
                "1. Check your system date/time - make sure it's set correctly \n"
                "2. Modify the pipeline to only request historical data (up to yesterday) \n"
                "3. If testing/developing, use the sample data generator instead of live API calls"
            )
        
        # NOAA 400 error solutions
        elif any("NOAA API" in s and "400" in s for s in results["suggestions"]):
            results["solutions"].append(
                "For NOAA API 400 errors: \n"
                "1. Check date ranges (should be within available data window, not in future) \n"
                "2. Verify station IDs exist \n"
                "3. Ensure datasetid and datatypeid parameters are valid"
            )
        
        # Authentication issues
        if any("authentication failed" in s for s in results["suggestions"]):
            results["solutions"].append(
                "For API authentication issues: "
                "1. Verify token/key is correct and not expired "
                "2. Check for whitespace or quote characters in the token "
                "3. Ensure the token is properly set in config.yaml"
            )
            
        # Missing output file
        if any("did not create output file" in s for s in results["suggestions"]):
            results["solutions"].append(
                "For missing output file despite successful run: "
                "1. Check if API returned empty responses "
                "2. Verify data processing logic correctly saves results "
                "3. Try the sample data generator to validate dashboard functionality"
            )
    
    return results

def generate_sample_data(project_root=None):
    """Generate sample data for demo purposes when actual pipeline fails"""
    import pandas as pd
    import numpy as np
    
    if not project_root:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Ensure output directory exists
    output_dir = os.path.join(project_root, 'data', 'processed')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create sample data
    cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']
    
    # Date range: 180 days
    dates = pd.date_range(end=datetime.now(), periods=180)
    
    # Create empty dataframe
    rows = []
    
    # Generate data for each city and date
    for city in cities:
        for date in dates:
            # Temperature varies by city and has seasonal pattern
            base_temp = {
                'New York': 55, 'Los Angeles': 70, 'Chicago': 50,
                'Houston': 75, 'Phoenix': 85
            }[city]
            
            # Add seasonal variation (sine wave with 365-day period)
            day_of_year = date.dayofyear
            seasonal_factor = np.sin(2 * np.pi * day_of_year / 365)
            
            # Different seasonal amplitude by city
            seasonal_amplitude = {
                'New York': 20, 'Los Angeles': 10, 'Chicago': 25,
                'Houston': 15, 'Phoenix': 20
            }[city]
            
            # Calculate temperature with seasonal variation and random noise
            temp_avg = base_temp + seasonal_amplitude * seasonal_factor + np.random.normal(0, 2)
            temp_min = temp_avg - np.random.uniform(5, 15)
            temp_max = temp_avg + np.random.uniform(5, 15)
            
            # Energy demand: Base + temperature effect + weekly pattern + random
            base_demand = {
                'New York': 500, 'Los Angeles': 400, 'Chicago': 350,
                'Houston': 300, 'Phoenix': 250
            }[city]
            
            # Energy increases with temperature extremes (U-shaped)
            temp_factor = 0.1 * (temp_avg - 65)**2
            
            # Weekly pattern - higher on weekdays
            weekday_factor = 0.8 if date.weekday() >= 5 else 1.0
            
            # Calculate energy demand
            energy = base_demand + temp_factor + np.random.normal(0, 20)
            energy *= weekday_factor
            
            # Add occasional missing values
            if np.random.random() < 0.02:  # 2% chance of missing temp_max
                temp_max = np.nan
            if np.random.random() < 0.02:  # 2% chance of missing temp_min
                temp_min = np.nan
            if np.random.random() < 0.01:  # 1% chance of missing energy
                energy = np.nan
            
            # Create row
            rows.append({
                'date': date.date(),
                'city': city,
                'state': {
                    'New York': 'New York', 'Los Angeles': 'California',
                    'Chicago': 'Illinois', 'Houston': 'Texas', 'Phoenix': 'Arizona'
                }[city],
                'temp_max_f': temp_max,
                'temp_min_f': temp_min,
                'temp_avg_f': temp_avg,
                'energy_demand_gwh': energy,
                'lat': {
                    'New York': 40.7128, 'Los Angeles': 34.0522, 'Chicago': 41.8781,
                    'Houston': 29.7604, 'Phoenix': 33.4484
                }[city],
                'lon': {
                    'New York': -74.0060, 'Los Angeles': -118.2437, 'Chicago': -87.6298,
                    'Houston': -95.3698, 'Phoenix': -112.0740
                }[city]
            })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(rows)
    output_path = os.path.join(output_dir, 'weather_energy_data.csv')
    df.to_csv(output_path, index=False)
    
    return {'created': True, 'path': output_path, 'rows': len(df)}

if __name__ == "__main__":
    # Run diagnostics when script is executed directly
    results = diagnose_pipeline_issue()
    print(json.dumps(results, indent=2))
    
    # If diagnostics indicate problems, generate sample data
    if results.get("suggestions"):
        print("\nGenerating sample data for demo purposes...")
        sample_results = generate_sample_data()
        print(f"Sample data created: {sample_results['path']} ({sample_results['rows']} rows)")
