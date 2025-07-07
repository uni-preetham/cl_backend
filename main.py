from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
from datetime import datetime
from utils.text_extraction import extract_text
from utils.keyword_extraction import extract_keywords
from utils.scoring import compute_score_and_missing_keywords

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...), file_type: str = Form(...)):
    file_id = str(uuid.uuid4())
    filename = f"{file_id}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    metadata = {
        "id": file_id,
        "filename": file.filename,
        "upload_time": datetime.utcnow().isoformat(),
        "type": file_type
    }
    return metadata

@app.post("/rank/")
async def rank_resumes(jd_id: str = Form(...), resume_ids: str = Form(...)):
    # resume_ids: comma-separated string
    jd_path = None
    resume_paths = []
    for fname in os.listdir(UPLOAD_DIR):
        if fname.startswith(jd_id):
            jd_path = os.path.join(UPLOAD_DIR, fname)
        for rid in resume_ids.split(","):
            if fname.startswith(rid):
                resume_paths.append(os.path.join(UPLOAD_DIR, fname))
    if not jd_path or not resume_paths:
        return JSONResponse(status_code=404, content={"error": "Files not found"})
    jd_text = extract_text(jd_path)
    resumes_text = [extract_text(rp) for rp in resume_paths]
    jd_keywords = extract_keywords(jd_text)
    results = compute_score_and_missing_keywords(jd_text, resumes_text, jd_keywords)
    # Attach resume IDs
    for i, rid in enumerate(resume_ids.split(",")):
        results[i]["resumeId"] = rid
    # Sort by score
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    for idx, r in enumerate(results):
        r["rank"] = idx + 1
    return results
