"""Download UCI Diabetes 130-US hospitals data to data/raw/diabetic_data.csv."""
import io, pathlib, sys, urllib.request, zipfile
URL = "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"
out = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw"
out.mkdir(parents=True, exist_ok=True)
try:
    data = urllib.request.urlopen(URL, timeout=60).read()
    zipfile.ZipFile(io.BytesIO(data)).extractall(out)
    print("Saved to", out)
except Exception as e:
    print("Download failed:", e)
    print("Manual: get diabetic_data.csv from https://archive.ics.uci.edu/dataset/296 (or Kaggle 'diabetes-130-us-hospitals')")
    print("and place it at", out / "diabetic_data.csv")
    sys.exit(1)
