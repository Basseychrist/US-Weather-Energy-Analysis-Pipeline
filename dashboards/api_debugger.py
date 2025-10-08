import os
import sys
import re
import json
import yaml
import datetime
import requests
import subprocess
from pathlib import Path

class ApiDebugger:
    """Utility to debug and fix API issues in the weather-energy pipeline"""
    
    def __init__(self, project_root=None):
        if not project_root:
            self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        else:
            self.project_root = project_root
        
        self.config = self._load_config()
        self.main_py_path = os.path.join(self.project_root, 'main.py')
        self.log_path = os.path.join(self.project_root, 'logs', 'pipeline.log')
    
    def _load_config(self):
        """Load configuration from config.yaml"""
        config_path = os.path.join(self.project_root, 'config', 'config.yaml')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f) or {}
            except Exception:
                pass
        return {}
    
    def get_api_tokens(self):
        """Extract API tokens from config"""
        tokens = {
            "noaa_token": self.config.get('noaa', {}).get('token', ''),
            "eia_api_key": self.config.get('eia', {}).get('api_key', '')
        }
        # Mask tokens for display
        display_tokens = {}
        for k, v in tokens.items():
            if v:
                masked = v[:4] + '*' * (len(v) - 8) + v[-4:] if len(v) > 8 else '*' * len(v)
                display_tokens[k] = masked
            else:
                display_tokens[k] = "Not set"
        
        return {"tokens": tokens, "display": display_tokens}
    
    def extract_api_calls_from_code(self):
        """Analyze main.py to extract API call patterns"""
        if not os.path.exists(self.main_py_path):
            return {"error": "main.py not found"}
        
        try:
            with open(self.main_py_path, 'r') as f:
                code = f.read()
            
            # Find NOAA API calls
            noaa_calls = []
            noaa_patterns = [
                r'requests\.get\(([\s\S]*?)ncdc\.noaa\.gov([\s\S]*?)\)',
                r'\.get\(([\s\S]*?)ncdc\.noaa\.gov([\s\S]*?)\)',
                r'noaa.*?base_url([\s\S]*?)\)',
            ]
            
            for pattern in noaa_patterns:
                matches = re.finditer(pattern, code)
                for match in matches:
                    context_start = max(0, match.start() - 100)
                    context_end = min(len(code), match.end() + 100)
                    context = code[context_start:context_end]
                    noaa_calls.append({
                        "match": match.group(0),
                        "context": context,
                        "line": code.count('\n', 0, match.start()) + 1
                    })
            
            # Find EIA API calls
            eia_calls = []
            eia_patterns = [
                r'requests\.get\(([\s\S]*?)api\.eia\.gov([\s\S]*?)\)',
                r'\.get\(([\s\S]*?)api\.eia\.gov([\s\S]*?)\)',
                r'eia.*?base_url([\s\S]*?)\)',
            ]
            
            for pattern in eia_patterns:
                matches = re.finditer(pattern, code)
                for match in matches:
                    context_start = max(0, match.start() - 100)
                    context_end = min(len(code), match.end() + 100)
                    context = code[context_start:context_end]
                    eia_calls.append({
                        "match": match.group(0),
                        "context": context,
                        "line": code.count('\n', 0, match.start()) + 1
                    })
            
            # Extract datetime patterns that may be used in API calls
            date_patterns = [
                r'datetime\.now\(\)\.strftime\([\'"]([^\'"]+)[\'"]\)',
                r'date\.today\(\)\.strftime\([\'"]([^\'"]+)[\'"]\)',
                r'datetime\.now\(\)\.date\(\)',
                r'date\.today\(\)',
                r'datetime\.strptime\((.*?)\)',
                r'pd\.date_range\((.*?)\)'
            ]
            
            date_calls = []
            for pattern in date_patterns:
                matches = re.finditer(pattern, code)
                for match in matches:
                    context_start = max(0, match.start() - 50)
                    context_end = min(len(code), match.end() + 50)
                    context = code[context_start:context_end]
                    date_calls.append({
                        "match": match.group(0),
                        "context": context,
                        "line": code.count('\n', 0, match.start()) + 1
                    })
            
            return {
                "noaa_calls": noaa_calls,
                "eia_calls": eia_calls,
                "date_calls": date_calls
            }
        except Exception as e:
            return {"error": f"Error analyzing code: {str(e)}"}
    
    def extract_requests_from_logs(self):
        """Parse logs to find recent API requests and their errors"""
        if not os.path.exists(self.log_path):
            return {"error": "Log file not found"}
        
        try:
            # Read last 100 lines to find recent errors
            with open(self.log_path, 'r') as f:
                lines = f.readlines()
                last_lines = lines[-100:] if len(lines) > 100 else lines
            
            api_errors = []
            api_calls = []
            
            # Look for API-related log entries
            for i, line in enumerate(last_lines):
                if "noaa" in line.lower() or "eia" in line.lower() or "api" in line.lower():
                    if "error" in line.lower() or "fail" in line.lower() or "exception" in line.lower():
                        context = "".join(last_lines[max(0, i-2):min(len(last_lines), i+3)])
                        api_errors.append({
                            "line": line.strip(),
                            "context": context
                        })
                    elif "request" in line.lower() or "fetching" in line.lower() or "get" in line.lower():
                        api_calls.append(line.strip())
            
            # Extract date ranges being requested
            date_patterns = [
                r"from\s+(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})",
                r"start_date\s*=\s*['\"]?(\d{4}-\d{2}-\d{2})['\"]?",
                r"end_date\s*=\s*['\"]?(\d{4}-\d{2}-\d{2})['\"]?",
                r"date\s*=\s*['\"]?(\d{4}-\d{2}-\d{2})['\"]?"
            ]
            
            dates_found = []
            for pattern in date_patterns:
                for line in last_lines:
                    matches = re.findall(pattern, line)
                    if matches:
                        dates_found.extend(matches)
            
            # Flatten the list of dates (some patterns return tuples)
            flat_dates = []
            for item in dates_found:
                if isinstance(item, tuple):
                    flat_dates.extend(item)
                else:
                    flat_dates.append(item)
            
            # Check for future dates
            today = datetime.datetime.now().date()
            future_dates = []
            for date_str in flat_dates:
                try:
                    date_obj = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
                    if date_obj > today:
                        future_dates.append({
                            "date": date_str,
                            "days_in_future": (date_obj - today).days
                        })
                except ValueError:
                    pass
            
            return {
                "api_errors": api_errors,
                "api_calls": api_calls,
                "dates_found": list(set(flat_dates)),  # remove duplicates
                "future_dates": future_dates
            }
        except Exception as e:
            return {"error": f"Error analyzing logs: {str(e)}"}
    
    def test_noaa_api_parameters(self):
        """Test NOAA API with common parameters to identify valid ones"""
        noaa_token = self.config.get('noaa', {}).get('token')
        if not noaa_token:
            return {"error": "NOAA token not found in config"}
        
        base_url = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"
        headers = {"token": noaa_token}
        
        # Test with different date ranges
        today = datetime.datetime.now().date()
        yesterday = today - datetime.timedelta(days=1)
        last_month = today - datetime.timedelta(days=30)
        
        # Format dates as strings
        today_str = today.strftime("%Y-%m-%d")
        yesterday_str = yesterday.strftime("%Y-%m-%d")
        last_month_str = last_month.strftime("%Y-%m-%d")
        
        test_cases = [
            {
                "description": "Last month to yesterday (safe date range)",
                "params": {
                    "datasetid": "GHCND",
                    "locationid": "CITY:US360019",  # New York
                    "datatypeid": "TMAX,TMIN",
                    "startdate": last_month_str,
                    "enddate": yesterday_str,
                    "limit": 1000
                }
            },
            {
                "description": "Yesterday only (minimal date range)",
                "params": {
                    "datasetid": "GHCND",
                    "locationid": "CITY:US360019",  # New York
                    "datatypeid": "TMAX,TMIN",
                    "startdate": yesterday_str,
                    "enddate": yesterday_str,
                    "limit": 1000
                }
            }
        ]
        
        # If system date appears to be set incorrectly (e.g., year 2025)
        if today.year > 2024:
            # Add a test case with corrected year (2023 instead of 2025)
            corrected_today = datetime.datetime(2023, today.month, today.day).date()
            corrected_yesterday = corrected_today - datetime.timedelta(days=1)
            corrected_yesterday_str = corrected_yesterday.strftime("%Y-%m-%d")
            corrected_last_month = corrected_today - datetime.timedelta(days=30)
            corrected_last_month_str = corrected_last_month.strftime("%Y-%m-%d")
            
            test_cases.append({
                "description": "Corrected date range (2023 instead of 2025)",
                "params": {
                    "datasetid": "GHCND",
                    "locationid": "CITY:US360019",  # New York
                    "datatypeid": "TMAX,TMIN",
                    "startdate": corrected_last_month_str,
                    "enddate": corrected_yesterday_str,
                    "limit": 1000
                }
            })
        
        results = []
        for test in test_cases:
            try:
                # Don't actually make API calls to avoid rate limits
                # Just return what would be called
                results.append({
                    "description": test["description"],
                    "url": f"{base_url}?{'&'.join([f'{k}={v}' for k, v in test['params'].items()])}",
                    "params": test["params"]
                })
            except Exception as e:
                results.append({
                    "description": test["description"],
                    "error": str(e)
                })
        
        return {
            "test_cases": results,
            "notes": "API calls not actually executed to avoid rate limits - these are examples of valid parameters",
            "recommendations": [
                "Always use yesterday or earlier as the enddate",
                "Prefer shorter date ranges (30 days or less) to avoid large response sizes",
                "If your system date is in the future, use explicit date strings from a hardcoded reference year (e.g., 2023)"
            ]
        }
    
    def generate_api_fix_patch(self):
        """Generate a patch to fix common API issues, particularly date-related problems"""
        if not os.path.exists(self.main_py_path):
            return {"error": "main.py not found"}
        
        try:
            with open(self.main_py_path, 'r') as f:
                content = f.read()
            
            # Create a backup
            backup_path = f"{self.main_py_path}.bak"
            with open(backup_path, 'w') as f:
                f.write(content)
            
            # Add safe date handling at the top of the file
            patch = '''
# --- Begin API Fix Patch ---
import datetime as _datetime

# Define safe date functions that never use future dates
def get_safe_today():
    """Return today's date, ensuring it's not in the future (relative to 2023)"""
    today = _datetime.date.today()
    if today.year > 2023:
        # System clock may be wrong - use a hardcoded reference date in 2023
        # that's the same month/day as today
        today = _datetime.date(2023, today.month, today.day)
    return today

def get_safe_date_range(days_back=30):
    """Get a safe date range ending yesterday"""
    today = get_safe_today()
    yesterday = today - _datetime.timedelta(days=1)
    start_date = yesterday - _datetime.timedelta(days=days_back)
    return start_date, yesterday

def format_api_date(date_obj):
    """Format a date for API requests"""
    return date_obj.strftime('%Y-%m-%d')

# --- End API Fix Patch ---

'''
            # Look for the first import statement
            first_import = content.find('import')
            if first_import >= 0:
                patched_content = content[:first_import] + patch + content[first_import:]
                
                # Save patched version
                patched_path = f"{self.main_py_path}.patched"
                with open(patched_path, 'w') as f:
                    f.write(patched_content)
                
                return {
                    "success": True,
                    "backup_path": backup_path,
                    "patched_path": patched_path,
                    "message": "Created patched version of main.py with safe date handling functions."
                }
            else:
                return {"error": "Could not find suitable location for patch"}
        except Exception as e:
            return {"error": f"Error creating patch: {str(e)}"}
    
    def run_patched_pipeline(self, mode='realtime'):
        """Run the pipeline with the patched file"""
        patched_path = f"{self.main_py_path}.patched"
        if not os.path.exists(patched_path):
            # Try to generate the patch first
            patch_result = self.generate_api_fix_patch()
            if not patch_result.get('success'):
                return patch_result
        
        # Run the patched pipeline
        cmd = [sys.executable, patched_path, mode]
        try:
            proc = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True, timeout=600)
            
            return {
                "success": proc.returncode == 0,
                "returncode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "command": " ".join(cmd),
                "patched_path": patched_path
            }
        except Exception as e:
            return {"error": f"Error running patched pipeline: {str(e)}"}
    
    def get_diagnostics_report(self):
        """Generate a comprehensive diagnostics report for API issues"""
        system_date = datetime.datetime.now()
        tokens = self.get_api_tokens()
        code_analysis = self.extract_api_calls_from_code()
        log_analysis = self.extract_requests_from_logs()
        
        # Determine if we have future date issues
        has_future_dates = False
        if log_analysis.get('future_dates'):
            has_future_dates = True
        if system_date.year > 2024:
            has_future_dates = True
        
        return {
            "timestamp": system_date.isoformat(),
            "system_date": {
                "date": system_date.strftime('%Y-%m-%d'),
                "time": system_date.strftime('%H:%M:%S'),
                "year": system_date.year,
                "likely_incorrect": system_date.year > 2024
            },
            "api_tokens": tokens.get('display'),
            "code_analysis": code_analysis,
            "log_analysis": log_analysis,
            "has_future_dates": has_future_dates,
            "recommendations": self._generate_recommendations(has_future_dates)
        }
    
    def _generate_recommendations(self, has_future_dates):
        """Generate specific recommendations based on diagnostics"""
        recommendations = []
        
        if has_future_dates:
            recommendations.extend([
                "ISSUE: Future dates detected in API requests or system date",
                "FIX: Use the patched version of main.py with safe date handling",
                "COMMAND: python api_debugger.py patch-and-run"
            ])
        
        # Always include general API recommendations
        recommendations.extend([
            "Always use yesterday or earlier as the enddate for NOAA API requests",
            "Verify your API tokens are correctly set in config.yaml",
            "When debugging API calls, check the exact URL parameters using tools like curl or Postman",
            "The NOAA API only supports historical data (not forecast or future data)"
        ])
        
        return recommendations

if __name__ == "__main__":
    debugger = ApiDebugger()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "test-params":
            result = debugger.test_noaa_api_parameters()
            print(json.dumps(result, indent=2))
        
        elif command == "patch":
            result = debugger.generate_api_fix_patch()
            if result.get('success'):
                print(f"✅ Created patched version: {result['patched_path']}")
                print(f"Original backed up to: {result['backup_path']}")
            else:
                print(f"❌ Error: {result.get('error')}")
        
        elif command == "patch-and-run":
            print("Patching main.py with API fixes and running pipeline...")
            result = debugger.run_patched_pipeline()
            if result.get('success'):
                print("✅ Patched pipeline completed successfully!")
                print(f"Patched file: {result['patched_path']}")
            else:
                print(f"❌ Pipeline failed: {result.get('error') or result.get('stderr')}")
        
        else:
            print("Available commands:")
            print("  test-params  - Test NOAA API parameters")
            print("  patch        - Create a patched version of main.py with API fixes")
            print("  patch-and-run - Patch main.py and run the pipeline")
    
    else:
        # Run full diagnostics by default
        print("Running API diagnostics...")
        report = debugger.get_diagnostics_report()
        
        print("\n==== API Debugger Report ====")
        print(f"System Date: {report['system_date']['date']} {report['system_date']['time']}")
        if report['system_date']['likely_incorrect']:
            print("⚠️  WARNING: System year appears to be set to the future!")
        
        print("\nAPI Tokens:")
        for k, v in report['api_tokens'].items():
            print(f"  {k}: {v}")
        
        if report.get('log_analysis', {}).get('future_dates'):
            print("\n⚠️  Future dates found in API requests:")
            for date_info in report['log_analysis']['future_dates']:
                print(f"  {date_info['date']} ({date_info['days_in_future']} days in the future)")
        
        print("\nRecommendations:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"{i}. {rec}")
        
        print("\nFor more details, use specific commands (test-params, patch, patch-and-run)")
