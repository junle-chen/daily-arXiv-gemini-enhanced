from pydantic.v1 import BaseModel, Field

class Structure(BaseModel):
    # --- 新增这一行 ---
    # 这个描述非常关键，它会作为给 AI 的指令，告诉它这个字段需要什么内容。
    tldr: str = Field(description="A very brief, one-sentence summary of the paper's core contribution (TL;DR).")
    
    motivation: str = Field(description="What problem or motivation does this paper address?")
    method: str = Field(description="What method did the authors propose?")
    result: str = Field(description="What are the experimental results or main findings?")
    conclusion: str = Field(description="The main conclusion of the paper.")
    summary_zh: str = Field(description="Translate the original paper summary into clear, fluent Chinese.")
