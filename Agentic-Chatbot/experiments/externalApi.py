import os
import openai
from time import sleep

openai.api_key = os.getenv("OPENAI_API_KEY")


assistant = openai.beta.assistants.create(
instructions="Sen bir hava durumu robotusun. Soruları yanıtlamak için sağlanan işlevleri kullanın.",
model="gpt-4o",
tools=[
    {
    "type": "function",
    "function": {
        "name": "get_current_temperature",
        "description": "Get the current temperature for a specific location",
        "parameters": {
        "type": "object",
        "properties": {
            "location": {
            "type": "string",
            "description": "The city and state, e.g., San Francisco, CA"
            },
            "unit": {
            "type": "string",
            "enum": ["Celsius", "Fahrenheit"],
            "description": "The temperature unit to use. Infer this from the user's location."
            }
        },
        "required": ["location", "unit"]
        }
    }
    }
    # {
    # "type": "function",
    # "function": {
    #     "name": "get_rain_probability_elif",
    #     "description": "Get the probability of rain for a specific location",
    #     "parameters": {
    #     "type": "object",
    #     "properties": {
    #         "location": {
    #         "type": "string",
    #         "description": "The city and state, e.g., San Francisco, CA"
    #         }
    #     },
    #     "required": ["location"]
    #     }
    # }
    # }
]
)

thread = openai.beta.threads.create()
message = openai.beta.threads.messages.create(
thread_id=thread.id,
role="user",
content="What's the weather in San Francisco today and the likelihood it'll rain?",
)


run = openai.beta.threads.runs.create_and_poll(
thread_id=thread.id,
assistant_id=assistant.id,
)

if run.status == 'completed':
    messages = openai.beta.threads.messages.list(
    thread_id=thread.id
)
    print(messages)
else:
    print(run.status)

# Define the list to store tool outputs
tool_outputs = []

# Loop through each tool in the required action section
for tool in run.required_action.submit_tool_outputs.tool_calls:
    if tool.function.name == "get_current_temperature":
        tool_outputs.append({
        "tool_call_id": tool.id,
        "output": "57"
    })
    elif tool.function.name == "get_rain_probability":
        tool_outputs.append({
        "tool_call_id": tool.id,
        "output": "0.06"
    })

# Submit all tool outputs at once after collecting them in a list
if tool_outputs:
    try:
            run = openai.beta.threads.runs.submit_tool_outputs_and_poll(
                thread_id=thread.id,
                run_id=run.id,
                tool_outputs=tool_outputs
            )
            print("Tool outputs submitted successfully.")
    except Exception as e:
            print("Failed to submit tool outputs:", e)
else:
    print("No tool outputs to submit.")

if run.status == 'completed':
    messages = openai.beta.threads.messages.list(
        thread_id=thread.id
)
    print(messages)
else:
    print(run.status)