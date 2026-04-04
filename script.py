from pathlib import Path

# Read the file that has syntax error
fixed_path = Path('output/ModelBuilding_with_confusion_matrix.py')
code = fixed_path.read_text(encoding='utf-8')

# Fix line 708: unterminated string literal
lines = code.splitlines()

# Find the problematic line
for i, line in enumerate(lines):
    if "print(\"" in line and not line.endswith('"'):
        # Assume it's the one with missing close quote and newline
        lines[i] = line.rstrip() + '"'
        print(f'Fixed line {i+1}: {lines[i]}')

# Also ensure the function header is okay and no unclosed """
result = '\n'.join(lines)
fixed_path.write_text(result, encoding='utf-8')
print('Fixed syntax errors in output/ModelBuilding_with_confusion_matrix.py')