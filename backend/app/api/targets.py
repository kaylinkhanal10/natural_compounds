from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def get_targets():
    return {"message": "Targets Explorer coming soon"}
