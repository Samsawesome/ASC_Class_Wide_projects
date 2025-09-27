@echo off
echo Testing the FIO JSON fix...
echo.

py -c "
import subprocess
import json
import tempfile
import os

# Test the JSON cleaning approach
test_output = '''fio: this platform does not support process shared mutexes, forcing use of threads. Use the 'thread' option to get rid of this warning.
{
  \"fio version\" : \"fio-3.41\",
  \"timestamp\" : 1758996355,
  \"jobs\" : []
}'''

# Clean the output
stdout_clean = test_output.strip()
json_start = stdout_clean.find('{')
if json_start != -1:
    json_output = stdout_clean[json_start:]
    print('Found JSON at position:', json_start)
    print('Cleaned output (first 100 chars):', json_output[:100])
    
    try:
        data = json.loads(json_output)
        print('✓ JSON parsing successful!')
    except Exception as e:
        print('✗ JSON parsing failed:', e)
else:
    print('✗ No JSON found in test output')
"

echo.
echo Now testing with actual FIO command...
py -c "
import subprocess
import json
import tempfile
import os

# Test actual FIO command
cmd = ['fio', '--thread', '--name=test', '--ioengine=windowsaio', '--filename=D:\\testfile', '--size=100M', '--rw=read', '--bs=4k', '--iodepth=1', '--runtime=2s', '--output-format=json']

try:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    print('FIO command executed')
    
    if result.stdout:
        # Clean the output
        stdout_clean = result.stdout.strip()
        json_start = stdout_clean.find('{')
        
        if json_start != -1:
            json_output = stdout_clean[json_start:]
            print('JSON found, attempting to parse...')
            
            try:
                data = json.loads(json_output)
                print('✓ Actual FIO JSON parsing successful!')
                print('FIO version:', data.get('fio version', 'Unknown'))
            except Exception as e:
                print('✗ Actual FIO JSON parsing failed:', e)
                print('First 200 chars of JSON:', json_output[:200])
        else:
            print('✗ No JSON found in FIO output')
            print('FIO stdout:', result.stdout[:200])
            
except Exception as e:
    print('Error running FIO command:', e)
"

echo.
pause