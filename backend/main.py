from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fairness_engine import FairnessAnalyzer
from llm_service import get_langchain_fairness_report
from typing import Optional
import pandas as pd
import io
import json
import traceback

app = FastAPI(title="Fairness Auditing API")

# Setup CORS to allow Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

analyzer = FairnessAnalyzer()

@app.get("/")
def read_root():
    return {"status": "Fairness Auditing API is running."}

@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    
    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
        analyzer.load_data(df)
        
        # Data Health metrics for the frontend panel
        missing_by_col = df.isnull().sum()
        missing_report = {col: int(cnt) for col, cnt in missing_by_col.items() if cnt > 0}
        
        return {
            "message": "File uploaded successfully",
            "columns": df.columns.tolist(),
            "row_count": len(df),
            "column_count": len(df.columns),
            "missing_values": missing_report,
            "total_missing": int(df.isnull().sum().sum()),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/audit")
async def audit_dataset(
    target_column: str = Form(...),
    protected_attribute: str = Form(...),
    favorable_class: str = Form(...),
    privileged_group: str = Form(...),
    prediction_column: Optional[str] = Form(None)
):
    try:
        results = analyzer.evaluate_bias(
            target_column=target_column,
            protected_attribute=protected_attribute,
            favorable_class=favorable_class,
            privileged_group=privileged_group,
            prediction_column=prediction_column
        )
        # Let Gemini translate the math into English
        ethics_report = get_langchain_fairness_report(results)
        
        # Keys that are promoted to the top level of the response
        top_level_keys = [
            "statistical_significance",
            "proxy_alerts",
            "model_comparison",
            "chartData",
        ]

        top_level = {k: results[k] for k in top_level_keys if k in results}
        raw_metrics = {k: v for k, v in results.items() if k not in top_level_keys}

        final_response = {
            **top_level,
            "ai_ethics_report": ethics_report.dict() if hasattr(ethics_report, "dict") else ethics_report.model_dump(),
            "raw_metrics": raw_metrics,
        }
        
        return final_response
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/mitigate")
async def mitigate_dataset(
    target_column: str = Form(...),
    protected_attribute: str = Form(...),
    favorable_class: str = Form(...),
    privileged_group: str = Form(...)
):
    try:
        results = analyzer.mitigate_bias(
            target_column=target_column,
            protected_attribute=protected_attribute,
            favorable_class=favorable_class,
            privileged_group=privileged_group,
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
