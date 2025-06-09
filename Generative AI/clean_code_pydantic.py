"""
Use Case:
---------
This module defines a robust data model for managing course enrollments using Pydantic.
It ensures that all student and course data is validated, normalized, and consistent before being processed or stored.

What We Plan to Achieve:
------------------------
- Enforce strict validation on student and course fields (e.g., email, age, enrollment date, course URL, progress).
- Automatically normalize certain fields (e.g., student name to uppercase).
- Provide custom and model-level validation (e.g., if progress is 100%, student must be certified).
- Demonstrate model usage with example test data and show how validation errors are reported.
- Serve as a foundation for integrating with APIs, databases, or user interfaces where clean, validated data is critical.
"""



from typing import Annotated, List, Optional
from pydantic import (
    BaseModel, Field, EmailStr, HttpUrl, validator, computed_field, model_validator, ValidationError
)
from datetime import date, datetime

# Define a data model for course enrollment using Pydantic
class CourseEnrollment(BaseModel):
    # Student's full name, max 100 characters
    student_name: Annotated[str, Field(title="Student Full Name", max_length=100)]
    # Student's email address, validated as an email
    student_email: EmailStr
    # Student's age, must be non-negative
    age: Annotated[int, Field(ge=0, title="Age of the Student")]
    # Date of enrollment, must not be in the future
    enrollment_date: Annotated[date, Field(title="Date of Enrollment")]
    # Course ID, between 5 and 10 characters
    course_id: Annotated[str, Field(min_length=5, max_length=10)]
    # Course URL, validated as a URL
    course_url: HttpUrl
    # Progress percentage, between 0 and 100
    progress: Annotated[float, Field(ge=0.0, le=100.0)]
    # Whether the student has received the certificate
    is_certified: Annotated[bool, Field(description="Has the student received the certificate?")]
    # Optional list of tags, max 5 tags
    tags: Annotated[Optional[List[str]], Field(default=None, max_length=5)]

    # Convert student name to uppercase before storing
    @validator('student_name')
    def convert_name_to_upper(cls, value):
        return value.upper()

    # Ensure age is not negative
    @validator('age')
    def age_must_be_non_negative(cls, value):
        if value < 0:
            raise ValueError("Age cannot be negative")
        return value

    # Ensure enrollment date is not in the future
    @validator('enrollment_date')
    def date_cannot_be_in_future(cls, value):
        if value > datetime.today().date():
            raise ValueError("Enrollment date cannot be in the future")
        return value

    # Ensure each tag is no longer than 15 characters
    @validator('tags', each_item=True)
    def tags_must_be_short(cls, tag):
        if len(tag) > 15:
            raise ValueError("Each tag must be 15 characters or fewer")
        return tag

    # Model-level validation: if progress is 100%, student must be certified
    @model_validator(mode='after')
    def check_certification_consistency(self):
        if self.progress == 100.0 and not self.is_certified:
            raise ValueError("If progress is 100%, the student must be certified.")
        return self

    # Computed property: returns status string based on certification
    @computed_field(return_type=str)
    @property
    def status(self) -> str:
        return "Certified" if self.is_certified else "In Progress"
    
    
#-------------------------------------# Example test data to validate the model -------------------------------------
# Uncomment the test data below to validate the model with different scenarios
#--------------------------------------------------------------------------------------------------------------------



# # ✅ Example test data 1
test_data = {
    "student_name": "Mike hussey",
    "student_email": "amit@example.com",
    "age": 25,
    "enrollment_date": "2024-06-01",
    "course_id": "ML101",
    "course_url": "https://example.com/courses/ml101",
    "progress": 85.5,
    "is_certified": True,
    "tags": ["Beginner", "Fast"]
}



# ✅ Example test data 2
# test_data = {
#     "student_name": "Mike hussey",
#     "student_email": "mike@example.com",
#     "age": 25,
#     "enrollment_date": "2024-06-01",
#     "course_id": "ML101",
#     "course_url": "https://example.com/courses/ml101",
#     "progress": 100,
#     "is_certified": False,
#     "tags": ["Beginner", "Fast"]
# }


# ✅ Example test data 3
# test_data = {
#     "student_name": "Mike hussey",
#     "student_email": "mikeexample.com",
#     "age": -25,
#     "enrollment_date": "2026-06-01",
#     "course_id": "ML101",
#     "course_url": "https://example.com/courses/ml101",
#     "progress": 100.0,
#     "is_certified": False,
#     "tags": ["Beginner", "Fast"]
# }



# 🚀 Test the model
try:
    enrollment = CourseEnrollment(**test_data)
    print("✅ Model created successfully:")
    print(enrollment.model_dump_json(indent=2))
except ValidationError as e:
    print("❌ Validation error:")
    print(e.json(indent=2))