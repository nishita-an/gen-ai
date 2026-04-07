import pathlib, sys
root = pathlib.Path('.venv/Lib/site-packages/streamlit')
print('root=', root)
paths = list(root.rglob('*.py'))
print('count=', len(paths))
for path in paths:
    try:
        text = path.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        continue
    if 'ClientStartEvent' in text or 'capture(' in text:
        print(path)
        for i, line in enumerate(text.splitlines(), 1):
            if 'ClientStartEvent' in line or 'capture(' in line:
                print(f'{i}: {line}')
