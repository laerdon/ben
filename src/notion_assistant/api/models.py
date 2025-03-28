from typing import List, Optional, Union
from pydantic import BaseModel, Field


class RichText(BaseModel):
    plain_text: str
    annotations: dict = Field(default_factory=dict)
    href: Optional[str] = None


class BlockContent(BaseModel):
    rich_text: List[RichText] = Field(default_factory=list)
    color: Optional[str] = None
    checked: Optional[bool] = None
    items: Optional[List[str]] = None


class NotionBlock(BaseModel):
    id: str
    type: str
    content: BlockContent
    has_children: bool = False
    children: Optional[List["NotionBlock"]] = None


class PageContent(BaseModel):
    title: str
    blocks: List[NotionBlock]
