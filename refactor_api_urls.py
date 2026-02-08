
import os

target_dir = "frontend/app"
old_url = "http://localhost:8000"
env_var = "process.env.NEXT_PUBLIC_API_URL"
# We'll use a constant if possible, or just inline string replacement if it's inside backticks.

def replace_in_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    if old_url not in content:
        return
        
    print(f"Refactoring {filepath}...")
    
    # 3 cases: key string, backtick template, or single quote string
    
    # Case 1: Simple string 'http://localhost:8000' -> API_URL
    # We should add a const import or definition? No, that's hard to automate perfectly.
    # Let's replace 'http://localhost:8000' with a template literal logic where possible
    # but that might break syntax if it's not in a template literal.
    
    # Safest bet: Replace the specific string with a variable, but we need to define the variable.
    # OR, we can just assume the user will set valid ENV var.
    
    # Let's try to define a centralized config.
    # But for now, let's just do search/replace.
    
    # If it is in a backtick: `http://localhost:8000/foo`
    # We want `${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/foo`
    
    # If it is in a single quote: 'http://localhost:8000/foo'
    # We want process.env.NEXT_PUBLIC_API_URL + '/foo' (or similar)
    
    # This is tricky with regex. 
    # Let's simple create a config file `frontend/config.ts` exporting API_URL.
    # checking...
    
    new_content = content.replace("http://localhost:8000", "${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}")
    
    # But wait, if it was 'http://localhost:8000', now it is '${...}' which is a string literal containing syntax.
    # That is WRONG if it was a single quote string.
    
    # Better approach:
    # 1. Look for `http://localhost:8000...` (backticks). 
    #    Replace `http://localhost:8000` with `${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}`
    
    # 2. Look for 'http://localhost:8000...' (single quotes).
    #    Switch to backticks? 
    #    Or: (process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000') + '...'
    
    # Let's try to identify lines.
    
    lines = content.split('\n')
    final_lines = []
    
    start_snippet = "const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';"
    
    # Check if we already added it?
    # No, let's just do inline replacement carefully.
    
    for line in lines:
        if old_url in line:
            # Replace logic
            # If line has backticks:
            if "`" in line and old_url in line:
                # Assuming it is inside the backticks
                line = line.replace(old_url, "${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}")
            elif "'" in line and old_url in line:
                # Quote based: 'http://...' -> `http://...` ?
                # Convert to backtick and do the replace?
                # This assumes the line is simple. 
                # e.g. axios.get('http://localhost:8000/foo')
                # -> axios.get(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/foo`)
                
                # Naive Swap: ' -> `  AND replace url
                # But there might be other quotes on the line? 
                pass 
                # Let's just do the backtick ones first, they are common in params.
                # For single quotes, let's handle manually or use a more robust regex if needed.
                # Actually, `sed` failed so manual is better.
                
                # Special handling for single quotes:
                # transform 'http://localhost:8000/xyz' to `${API_URL}/xyz` is risky without parsing.
                # simpler: change 'http://localhost:8000 to (process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000') + '
                # e.g. axios.get('http://localhost:8000/herbs/') 
                # -> axios.get((process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000') + '/herbs/')
                
                parts = line.split(old_url)
                # parts[0] ends with ' hopefully?
                # actually, 'http://localhost:8000/herbs/'
                # parts[0] = "axios.get('"
                # parts[1] = "/herbs/')"
                
                # New line: 
                # axios.get((process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000') + '/herbs/')
                # We need to remove the leading quote from parts[1] (no, API_URL doesn't have it)
                # We need to remove the trailing quote from parts[0]? 
                
                # Check if parts[0] ends with '
                if parts[0].endswith("'"):
                     # Remove '
                     p0 = parts[0][:-1]
                     # Add (env... + '
                     line = p0 + "(process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000') + '" + parts[1]
                else:
                     # Maybe it was just implied?
                     pass
                     
            final_lines.append(line)
        else:
            final_lines.append(line)
            
    with open(filepath, 'w') as f:
        f.write('\n'.join(final_lines))

for root, dirs, files in os.walk(target_dir):
    for file in files:
        if file.endswith(".tsx") or file.endswith(".ts"):
            replace_in_file(os.path.join(root, file))

