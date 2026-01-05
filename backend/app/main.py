from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api import herbs, combinations, intent, formulas, diseases, targets

app = FastAPI(
    title="Natural Medicine Discovery AI",
    description="B2B AI-native platform for evidence-backed natural medicine reasoning.",
    version="0.1.0"
)

# CORS (Allow Frontend)
origins = ["http://localhost:3000", "http://localhost:3001"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(herbs.router, prefix="/herbs", tags=["Herbs"])
app.include_router(combinations.router, prefix="/combine", tags=["Combinations"])
app.include_router(intent.router, prefix="/intent", tags=["Intent"])
app.include_router(formulas.router, prefix="/formulas", tags=["Formulas"])
app.include_router(diseases.router, prefix="/diseases", tags=["Diseases"])
app.include_router(targets.router, prefix="/targets", tags=["Targets"])

@app.get("/")
def read_root():
    return {"message": "Natural Medicine Discovery AI is running."}
