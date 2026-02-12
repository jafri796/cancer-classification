import ast, os, sys

ok = 0
fail = 0
for root, dirs, files in os.walk("src"):
    for f in files:
        if f.endswith(".py"):
            path = os.path.join(root, f)
            try:
                ast.parse(open(path).read())
                ok += 1
            except SyntaxError as e:
                print(f"FAIL: {path}: {e}")
                fail += 1

for f in os.listdir("scripts"):
    if f.endswith(".py"):
        path = os.path.join("scripts", f)
        try:
            ast.parse(open(path).read())
            ok += 1
        except SyntaxError as e:
            print(f"FAIL: {path}: {e}")
            fail += 1

print(f"Checked {ok + fail} files: {ok} OK, {fail} FAIL")
sys.exit(1 if fail else 0)
