"""Record the actual runtime/build instead of assuming the original host."""
import json
import os
import platform
import subprocess
from pathlib import Path
import torch

cpu = platform.processor() or platform.machine()
if platform.system() == 'Darwin':
    query = subprocess.run(['/usr/sbin/sysctl', '-n', 'machdep.cpu.brand_string'],
                           capture_output=True, text=True)
    if query.returncode == 0:
        cpu = query.stdout.strip()
report = dict(cpu=cpu, ram_bytes=os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES'),
              platform=platform.platform(), python=platform.python_version(),
              torch=str(torch.__version__), cuda=torch.cuda.is_available(),
              mps=torch.backends.mps.is_available(), torch_build=torch.__config__.show())
Path(__file__).with_name('hardware.json').write_text(json.dumps(report, indent=2) + '\n')
