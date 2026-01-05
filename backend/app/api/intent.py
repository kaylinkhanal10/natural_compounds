from fastapi import APIRouter
from pydantic import BaseModel
from ..services.intent_parser import IntentParser

router = APIRouter()

class IntentRequest(BaseModel):
    text: str

@router.post("/")
def parse_intent(request: IntentRequest):
    parser = IntentParser()
    return parser.parse(request.text)
