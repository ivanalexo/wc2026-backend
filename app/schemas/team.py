from pydantic import BaseModel, ConfigDict, computed_field

class TeamResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    slug: str
    group: str | None
    confederation: str | None
    elo_rating: float | None
    flag_code: str | None

    @computed_field
    @property
    def flag_url(self) -> str | None:
        if self.flag_code:
            return f'https://flagcdn.com/w320/{self.flag_code.lower()}.png'
        return None

class TeamSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    name: str
    slug: str
    flag_code: str | None

    @computed_field
    @property
    def flag_url(self) -> str | None:
        if self.flag_code:
            return f'https://flagcdn.com/w320/{self.flag_code.lower()}.png'
        return None