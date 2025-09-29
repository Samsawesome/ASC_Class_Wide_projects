import subprocess
import json
import tempfile
import os
from pathlib import Path
import time
import shutil

class FioRunner:
    def __init__(self, drive_letter='D:', test_size='10G'):
        self.drive_letter = drive_letter.rstrip('\\')
        self.test_size = test_size
        self.test_file = Path(f"{self.drive_letter}\\testfile")
        
    def run_fio(self, config_content, output_format='json'):
        """Execute FIO with given configuration"""
        # First, let's check if FIO is available
        if not shutil.which('fio'):
            print("ERROR: FIO not found in PATH. Please ensure FIO is installed.")
            return None
            
        temp_file = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.fio', delete=False, encoding='utf-8') as f:
                # Replace placeholders in config with proper Windows paths
                config_content = config_content.replace('D:\\testfile', str(self.test_file))
                config_content = config_content.replace('10G', self.test_size)
                
                f.write(config_content)
                temp_file = f.name
            
            print(f"Using FIO config: {temp_file}")
            
            # Use --thread to avoid the warning message
            cmd = ['fio', '--thread', '--output-format=json', temp_file]
            
            # Run FIO with timeout
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=True)
            
            # Clean the output - remove any warning lines before the JSON
            stdout_clean = result.stdout.strip()
            
            # Find the start of JSON (first '{' character)
            json_start = stdout_clean.find('{')
            if json_start == -1:
                print("ERROR: No JSON found in FIO output")
                print(f"FIO stdout: {stdout_clean[:500]}")
                return None
                
            # Extract JSON part only
            json_output = stdout_clean[json_start:]
            
            # Debug: Print first 200 chars of cleaned JSON
            print(f"Cleaned JSON start: {json_output[:200]}...")
            
            if not json_output.strip():
                print("ERROR: No JSON content after cleaning")
                return None
                
            # Try to parse JSON
            try:
                data = json.loads(json_output)
                print("JSON parsed successfully")
                return data
            except json.JSONDecodeError as e:
                print(f"ERROR: Failed to parse cleaned JSON: {e}")
                print(f"JSON content (first 500 chars): {json_output[:500]}")
                # Try to find where the JSON breaks
                try:
                    # Attempt to find valid JSON subset
                    json.loads(json_output[:json_output.rfind('}')+1])
                    print("Managed to parse truncated JSON")
                except:
                    pass
                return None
                
        except subprocess.TimeoutExpired:
            print("ERROR: FIO command timed out (5 minutes)")
            return None
        except subprocess.CalledProcessError as e:
            print(f"ERROR: FIO command failed with exit code {e.returncode}")
            print(f"FIO stderr: {e.stderr}")
            return None
        except Exception as e:
            print(f"ERROR: Unexpected error running FIO: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            # Clean up temp file
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except PermissionError:
                    print(f"Warning: Could not delete temp file {temp_file}, will retry...")
                    time.sleep(1)
                    try:
                        os.unlink(temp_file)
                    except:
                        print(f"Warning: Could not delete temp file {temp_file}")
    
    def precondition_drive(self):
        """Precondition the drive for consistent results"""
        print("Preconditioning drive with sequential writes...")
        precondition_config = """
[global]
ioengine=windowsaio
direct=1
size=10G
filename=D:\\testfile
thread

[precondition]
rw=write
bs=1M
iodepth=32
numjobs=4
"""
        result = self.run_fio(precondition_config)
        if result:
            print("Preconditioning completed successfully")
        else:
            print("Preconditioning failed")
        
    def run_benchmark(self, config_name, num_runs=3):
        """Run a benchmark multiple times and aggregate results"""
        config_path = Path('configs') / f'{config_name}.fio'
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_content = f.read()
        
        all_results = []
        
        for run in range(num_runs):
            print(f"Run {run + 1}/{num_runs} for {config_name}")
            result = self.run_fio(config_content)
            if result:
                # Add run identifier
                for job in result['jobs']:
                    job['run_id'] = run
                all_results.append(result)
                print(f"Run {run + 1} completed successfully")
            else:
                print(f"Run {run + 1} failed")
                # Don't continue if first run fails
                if run == 0:
                    print("Stopping benchmark due to first run failure")
                    return None
            
            # Cool down between runs
            if run < num_runs - 1:
                print("Cooling down for 10 seconds...")
                time.sleep(10)
        
        return all_results if all_results else None