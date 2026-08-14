from pydantic import BaseModel,field_validator
class UserProfile(BaseModel):
    name:str
    age:int
    email:str
    @field_validator('age')
    def check_age(cls, value):
        if value < 40:
            raise ValueError("Age must be at least 40 years old.")
        return value
data = '{"name": "John Doe", "age": 41, "email": "john.doe@example.com"}'
user= UserProfile.model_validate_json(data)
print(user)

    