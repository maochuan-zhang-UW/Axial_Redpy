"""
Send REDPy catfill progress update via macOS Mail.app.
Called every 2 hours by cron.
"""
import subprocess
import os
import re
from datetime import datetime

LOG_FILE = '/Users/mczhang/Documents/GitHub/Axial_Redpy/catfill_dd_2018_2021.log'
TO_EMAIL = 'mczhang8@uw.edu'

def get_progress():
    if not os.path.exists(LOG_FILE):
        return 'Log file not found.', 'unknown', 0, 0, 0

    with open(LOG_FILE, 'r') as f:
        lines = f.readlines()

    # Get last timestamp, repeaters, families, orphans
    last_time = 'unknown'
    repeaters = 0
    families  = 0
    orphans   = 0

    for line in reversed(lines):
        line = line.strip()
        if re.match(r'^\d{4}-\d{2}-\d{2}T', line) and last_time == 'unknown':
            last_time = line
        if 'Number of repeaters' in line and repeaters == 0:
            m = re.search(r':\s*(\d+)', line)
            if m: repeaters = int(m.group(1))
        if 'Number of families' in line and families == 0:
            m = re.search(r':\s*(\d+)', line)
            if m: families = int(m.group(1))
        if 'Number of orphans' in line and orphans == 0:
            m = re.search(r':\s*(\d+)', line)
            if m: orphans = int(m.group(1))
        if last_time != 'unknown' and repeaters and families and orphans:
            break

    # Check if process is still running
    result = subprocess.run(['pgrep', '-f', 'redpy-catfill'],
                            capture_output=True, text=True)
    status = 'RUNNING' if result.returncode == 0 else 'STOPPED'

    # Last few non-file lines of log
    tail_lines = [l.strip() for l in lines[-50:]
                  if l.strip() and not l.startswith('/')][-8:]
    tail = '\n'.join(tail_lines)

    body = f"""REDPy Axial Seamount — Progress Update
{'='*45}
Time sent:        {datetime.now().strftime('%Y-%m-%d %H:%M')}
Process:          {status}

Currently processing:  {last_time}
Repeating EQs found:   {repeaters}
Families found:        {families}
Orphans processed:     {orphans}

--- Last log lines ---
{tail}

Log file: {LOG_FILE}
"""
    subject = f'REDPy [{status}] {last_time[:10]} | {repeaters} repeaters / {families} families'
    return body, subject, repeaters, families, orphans


def send_email(to, subject, body):
    import tempfile
    # Escape backslashes and quotes for AppleScript string literals
    body_esc    = body.replace('\\', '\\\\').replace('"', '\\"')
    subject_esc = subject.replace('\\', '\\\\').replace('"', '\\"')
    script = f'''tell application "Mail"
    set theBody to "{body_esc}"
    set newMsg to make new outgoing message with properties {{subject:"{subject_esc}", content:theBody, visible:false}}
    tell newMsg
        make new to recipient with properties {{address:"{to}"}}
    end tell
    send newMsg
end tell'''
    with tempfile.NamedTemporaryFile(suffix='.applescript', mode='w',
                                     delete=False) as f:
        f.write(script)
        tmp = f.name
    try:
        subprocess.run(['osascript', tmp], check=True)
    finally:
        os.unlink(tmp)


if __name__ == '__main__':
    body, subject, repeaters, families, orphans = get_progress()
    print(f'Sending update: {subject}')
    send_email(TO_EMAIL, subject, body)
    print('Sent.')
