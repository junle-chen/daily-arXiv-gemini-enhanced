from pydantic import BaseModel, Field
class Structure(BaseModel):
    tldr: str = Field(description="A one-sentence summary of the paper (TL;DR).")
    motivation: str = Field(description="What problem or motivation does this paper address?")
    method: str = Field(description="What method did the authors propose?")
    result: str = Field(description="What are the experimental results or main findings?")
    conclusion: str = Field(description="The main conclusion of the paper.")
    
    # --- 新增这一行 ---
    # 这个描述会告诉 AI，这个字段是用来存放翻译后的中文摘要的
    summary_zh: str = Field(description="Translate the original paper summary into clear, fluent Chinese.")
