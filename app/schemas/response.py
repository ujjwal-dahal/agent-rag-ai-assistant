from typing import Optional

from pydantic import BaseModel


class AssistantResponse(BaseModel):

    answer: str

    source_used: bool

    tool_used: Optional[str] = None

    verified: bool

    iterations_used: int