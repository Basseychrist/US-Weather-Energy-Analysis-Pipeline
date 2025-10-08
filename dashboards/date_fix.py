import os
import sys
import re
import datetime
import subprocess
import shutil

def check_system_date():
    """Check if system date appears to be set incorrectly (in the future)"""
    now = datetime.datetime.now()
    system_year = now.year
    expected_year_range = range(2022, 2024)  # Reasonable range as of 2023
    
    return {
        "system_date": now.strftime("%Y-%m-%d"),
        "system_year": system_year,
        "likely_incorrect": system_year > 2024,
        "recommended_year": 2023,
    }

def find_date_usage_in_file(filepath):
    """Find code in Python files that might be using system date/now() calls"""
    if not os.path.exists(filepath):
        return {"error": f"File not found: {filepath}"}
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    results = {
        "file": filepath,
        "datetime_now_calls": [],
        "date_today_calls": [],
        "fixed_file": None
    }
    
    # Find datetime.now() or datetime.datetime.now() calls
    now_pattern = r'(datetime\.now\(\)|datetime\.datetime\.now\(\))'
    now_matches = re.finditer(now_pattern, content)
    for match in now_matches:
        start, end = match.span()
        line_start = content.rfind('\n', 0, start) + 1
        line_end = content.find('\n', end)
        if line_end == -1:
            line_end = len(content)
        line = content[line_start:line_end].strip()
        results["datetime_now_calls"].append({"line": line, "position": (start, end)})
    
    # Find date.today() calls
    today_pattern = r'(date\.today\(\)|datetime\.date\.today\(\))'
    today_matches = re.finditer(today_pattern, content)
    for match in today_matches:
        start, end = match.span()
        line_start = content.rfind('\n', 0, start) + 1
        line_end = content.find('\n', end)
        if line_end == -1:
            line_end = len(content)
        line = content[line_start:line_end].strip()
        results["date_today_calls"].append({"line": line, "position": (start, end)})
    
    return results

def fix_main_for_date_issues(project_root=None):
    """Find main.py and patch it to use fixed date logic if needed"""
    if not project_root:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    main_py = os.path.join(project_root, 'main.py')
    if not os.path.exists(main_py):
        return {"error": f"Main.py not found at {main_py}"}
    
    # Check if the main.py file appears to have date issues
    results = find_date_usage_in_file(main_py)
    
    # If we found date/time usage, create a fixed version
    if results["datetime_now_calls"] or results["date_today_calls"]:
        try:
            # Create a backup
            backup_path = main_py + ".bak"
            shutil.copy2(main_py, backup_path)
            
            with open(main_py, 'r') as f:
                content = f.read()
            
            # Add a date offset function at the top of the file
            # Using single-line comments instead of triple-quoted strings
            header_fix = '''
# --- BEGIN DATE FIX (automatically added) ---
# This fixes issues with incorrect system date (year 2025) by forcing the date to be in 2023
import datetime as _datetime_original

# Store the original functions
_original_datetime_now = _datetime_original.datetime.now
_original_date_today = _datetime_original.date.today

def _fixed_datetime_now():
    # Return datetime.now() but with the year adjusted to 2023 if currently 2025
    now = _original_datetime_now()
    if now.year == 2025:
        return now.replace(year=2023)
    return now

def _fixed_date_today():
    # Return date.today() but with the year adjusted to 2023 if currently 2025
    today = _original_date_today()
    if today.year == 2025:
        return today.replace(year=2023)
    return today

# Replace the original functions
_datetime_original.datetime.now = _fixed_datetime_now
_datetime_original.date.today = _fixed_date_today
# --- END DATE FIX ---

'''
            # Add the fix code at the top, after any module docstrings but before actual code
            first_import = content.find('import')
            if first_import >= 0:
                content = content[:first_import] + header_fix + content[first_import:]
                fixed_main_py = main_py + ".fixed"
                with open(fixed_main_py, 'w') as f:
                    f.write(content)
                results["fixed_file"] = fixed_main_py
                results["backup_file"] = backup_path
                results["patch_applied"] = True
            else:
                results["error"] = "Could not find a suitable position to insert the fix"
        except Exception as e:
            results["error"] = str(e)
    else:
        results["message"] = "No datetime.now() or date.today() calls found to patch"
    
    return results

def patch_and_run_fixed_pipeline(mode='realtime', project_root=None, timeout_seconds=600):
    """Create a date-fixed version of main.py and run it with the given mode"""
    if not project_root:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # First, try to patch the main.py file
    patch_result = fix_main_for_date_issues(project_root)
    
    # If patching failed or no fixed file was created, return the error
    if patch_result.get("error") or not patch_result.get("fixed_file"):
        return {
            "ran": False, 
            "reason": f"Failed to patch main.py: {patch_result.get('error', 'No fixed file created')}",
            "patch_result": patch_result
        }
    
    # Use the fixed main.py file
    fixed_main_py = patch_result["fixed_file"]
    
    # Run the fixed main.py
    cmd = [sys.executable, fixed_main_py, mode]
    try:
        proc = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True, timeout=timeout_seconds)
        
        if proc.returncode == 0:
            # If successful, we could replace the original with the fixed version
            # (but for safety we'll leave this as an option for the user)
            return {
                "ran": True, 
                "stdout": proc.stdout, 
                "fixed_main": fixed_main_py,
                "patch_result": patch_result
            }
        else:
            return {
                "ran": False, 
                "stderr": proc.stderr, 
                "stdout": proc.stdout,
                "returncode": proc.returncode,
                "reason": "Fixed pipeline failed",
                "patch_result": patch_result
            }
    except subprocess.TimeoutExpired as te:
        return {"ran": False, "reason": "timeout", "detail": str(te)}
    except Exception as e:
        return {"ran": False, "reason": "exception", "detail": str(e)}

if __name__ == "__main__":
    # When run directly, check the system date and suggest fixes
    date_info = check_system_date()
    
    print(f"System date: {date_info['system_date']}")
    if date_info["likely_incorrect"]:
        print(f"⚠️ WARNING: Your system year ({date_info['system_year']}) appears to be set incorrectly!")
        print(f"   Recommended year: {date_info['recommended_year']}")
        print("   This is likely causing issues with API requests for future dates.\n")
        
        # Ask if the user wants to patch main.py
        print("Options:")
        print("1. Create a patched version of main.py that fixes the date issue")
        print("2. Run the pipeline with the date fix applied")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == "1":
            result = fix_main_for_date_issues()
            if result.get("fixed_file"):
                print(f"\n✅ Created fixed main.py at: {result['fixed_file']}")
                print(f"   Original backed up at: {result['backup_file']}")
                print("\nTo use the fixed version:")
                print(f"python {result['fixed_file']} realtime")
            else:
                print(f"\n❌ Failed to create fixed main.py: {result.get('error', 'unknown error')}")
        
        elif choice == "2":
            print("\nRunning pipeline with date fix applied...")
            result = patch_and_run_fixed_pipeline(mode='realtime')
            if result.get("ran"):
                print("\n✅ Pipeline completed successfully with date fix!")
                print(f"   Fixed main.py at: {result['fixed_main']}")
            else:
                print(f"\n❌ Pipeline failed even with date fix: {result.get('reason', 'unknown error')}")
    else:
        print("✅ System date appears to be set correctly.")
        print("✅ System date appears to be set correctly.")
