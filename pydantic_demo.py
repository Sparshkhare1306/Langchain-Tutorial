from pydantic import BaseModel, Field
from typing import Optional
class Student(BaseModel):
    name: str = 'himesh'
    age: Optional[int] = None
    


new_student = {}

student = Student(**new_student)

print((student))

