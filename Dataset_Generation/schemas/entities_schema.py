from pydantic import BaseModel
from typing import Optional, List

class Education(BaseModel):
    degree: str 
    institution: str 
    year: Optional[str]

class ResumeEntities(BaseModel):
    name: Optional[str]
   
    skills: List[str]
    experience_years: Optional[int]

    companies: List[str]
    designations: List[str]

    achievements: List[str]
    awards: List[str]
    certificates: List[str]

    projects: List[str]

    education: List[Education]

