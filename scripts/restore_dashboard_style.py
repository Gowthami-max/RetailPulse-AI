from pathlib import Path

app_path = Path("frontend/app.py")
backup_path = Path("frontend/app_backup.py")

app = app_path.read_text(encoding="utf-8")
backup = backup_path.read_text(encoding="utf-8")

start_marker = 'st.markdown("""'
end_marker = '@st.cache_data(show_spinner="Loading RetailPulse datasets...")'

start = backup.find(start_marker)
end = backup.find(end_marker)

if start == -1 or end == -1 or end <= start:
    raise RuntimeError("Could not locate original styling in backup.")

style_block = backup[start:end].rstrip()

if "PLOT_LAYOUT = dict(" not in style_block:
    raise RuntimeError("Plotly configuration missing from backup.")

if "PLOT_LAYOUT = dict(" in app:
    print("Styling already exists. No changes made.")
    raise SystemExit

insert_marker = "REQUIRED_FILES = ["

if insert_marker not in app:
    raise RuntimeError("Could not locate insertion point.")

app = app.replace(
    insert_marker,
    style_block + "\n\n" + insert_marker,
    1
)

app_path.write_text(app, encoding="utf-8")

print("SUCCESS: Original CSS styling restored.")
print("SUCCESS: Original Plotly configuration restored.")
print("PostgreSQL + Redis integration preserved.")