import os
import google.generativeai as genai


# Fetch the API key from the environment variable
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# --------------------------------------------------------------
# Basics of function calling
# --------------------------------------------------------------


def calculate_area(length: float, width: float) -> float:
    """Returns the area of a rectangle given its length and width."""
    return length * width


model = genai.GenerativeModel(
    model_name="gemini-1.0-pro-latest", tools=[calculate_area]
)
model


chat = model.start_chat(enable_automatic_function_calling=True)

response = chat.send_message("I have a tile of length 5 and width 3. What is the area?")
response.text

# Fetch the chat history
for content in chat.history:
    part = content.parts[0]
    print(content.role, "->", type(part).to_dict(part))
    print("-" * 80)


# --------------------------------------------------------------
# Parallel function calling
# --------------------------------------------------------------


def apply_soap(amount: float) -> bool:
    """Applies the specified amount of soap to the car. Maximum amount is 5 liters if not specified

    Args:
      amount: The amount of soap in liters for washing.

    Returns: True if the soap was applied successfully.
    """

    print(f"Applying {amount} liters of soap.")
    return True


def rinse_car(water_pressure: int) -> bool:
    """Rinses the car with water at the specified pressure. Maximum pressure is 50 psi if not specified

    Args:
      water_pressure: The water pressure level for rinsing.

    Returns: True if the car was rinsed successfully.
    """
    print(f"Rinsing car at {water_pressure} psi.")
    return True


def dry_car(method: str) -> str:
    """Dries the car using the specified method. The method can be air drying, towel drying if not specified

    Args:
      method: The method to use for drying the car

    Returns: The method used to dry the car.
    """
    print(f"Drying car using {method}.")
    return method


def wax_car(type: str) -> bool:
    """Applies the specified type of wax to the car. The type can be liquid, paste, Spray if not specified

    Args:
      type: The type of wax to apply.

    Returns: True if the wax was applied successfully.
    """
    print(f"Applying {type} wax.")
    return True


car_fns = [apply_soap, rinse_car, dry_car, wax_car]

model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest", tools=car_fns)


chat = model.start_chat()

# Iteration 1
response = chat.send_message("I want my car washed with soap only")

# Iteration 2
response = chat.send_message(
    "I want full wash of my car and it should not be wet after wash."
)

# Iteration 3
response = chat.send_message("I want my car washed and it should shine like new one")

# Iteration 4
response = chat.send_message(
    "I want my car washed and it should shine like new one. Also list the quantity or the material used in each stage"
)

# Output
response


for part in response.parts:
    if fn := part.function_call:
        args = ", ".join(f"{key}={val}" for key, val in fn.args.items())
        print(f"{fn.name}({args})")


# --------------------------------------------------------------
# Simulate the responses from the specified tools.
# --------------------------------------------------------------

responses = {
    "apply_soap": 1.0,
    "rinse_car": 15,
    "dry_car": "Air Dried",
    "wax_car": "Spray wax",
}

# Build the response parts.
response_parts = [
    genai.protos.Part(
        function_response=genai.protos.FunctionResponse(
            name=fn, response={"result": val}
        )
    )
    for fn, val in responses.items()
]

response = chat.send_message(response_parts)
print(response)

for content in chat.history:
    part = content.parts[0]
    print(content.role, "->", type(part).to_dict(part))
    print("-" * 80)


# Complete history of changes
[content.parts[:] for content in chat.history]
