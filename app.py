
import os
import json
import threading
import time
import uvicorn

from dotenv import load_dotenv
load_dotenv()

os.environ.setdefault("API_URL", "http://localhost:8000")

gcp_creds_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
if gcp_creds_json:
    creds_path = "/tmp/gcp_credentials.json"
    with open(creds_path, "w") as f:
        json.dump(json.loads(gcp_creds_json), f)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path


def run_fastapi():
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, log_level="info")


threading.Thread(target=run_fastapi, daemon=True).start()
time.sleep(3)

exec(open("app/ui.py").read())
