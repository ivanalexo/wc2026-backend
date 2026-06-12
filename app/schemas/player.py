from pydantic import BaseModel, ConfigDict

class PlayerResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    team_name: str
    name: str
    position: str | None
    number: int | None
    age: int | None
    nacionality: str | None
    club: str | None
    photo_url: str | None
