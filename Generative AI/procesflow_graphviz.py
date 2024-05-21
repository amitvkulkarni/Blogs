import io
import os
import google.generativeai as genai
import json
import requests
import webbrowser


# Fetch the API key from the environment variable
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_gemini_response(input_text: str) -> str:
    """Get the response from the Gemini model for the given input text."""
    model = genai.GenerativeModel("gemini-1.5-flash-latest")
    response = model.generate_content(input_text)
    return response.text


def format_dot_code(dot_code: str) -> str:
    """Format the DOT code to ensure it displays the workflow vertically."""
    formatted_code = dot_code.strip("```dot").strip()
    lines = formatted_code.split("\n")
    for i, line in enumerate(lines):
        if "rankdir" in line:
            lines[i] = "    rankdir=TB;"
    return "\n".join(lines)


def save_and_open_svg(dot_code: str, output_filename: str):
    """Generate and save the SVG image from the DOT code, and open it in a web browser."""
    quickchart_url = "https://quickchart.io/graphviz"
    post_data = {"graph": dot_code}

    try:
        response = requests.post(quickchart_url, json=post_data, verify=False)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "").lower()
        if "image/svg+xml" in content_type:
            svg_content = response.text
            with open(output_filename, "w", encoding="utf-8") as f:
                f.write(svg_content)

            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Flowchart</title>
            </head>
            <body>
                <img src="{output_filename}" alt="Flowchart">
            </body>
            </html>
            """
            html_filename = output_filename.replace(".svg", ".html")
            with open(html_filename, "w", encoding="utf-8") as f:
                f.write(html_content)

            webbrowser.open("file://" + os.path.realpath(html_filename))
        else:
            print("Unexpected response content type:", content_type)

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")


def save_png(dot_code: str, output_filename: str):
    """Generate and save the PNG image from the DOT code."""
    quickchart_url = "https://quickchart.io/graphviz"
    post_data = {"graph": dot_code, "format": "png"}

    try:
        response = requests.post(quickchart_url, json=post_data, verify=False)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "").lower()
        if "image/png" in content_type:
            png_content = response.content
            with open(output_filename, "wb") as f:
                f.write(png_content)
            print(f"Flowchart saved as {output_filename}")
        else:
            print("Unexpected response content type:", content_type)

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")


def main():
    """Main function to generate flowchart from input text using Gemini Flash and Graphviz."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: Google API key not found in environment variables.")
        return

    question = "you are an expert in ecommerce marketing and supply chain. Can you help me with the process flow diagram for a \
            on how companies like amazon and flipkart manage their supply chain?\
            Please use Graphviz DOT Language. Try to make it as detailed as possible with all the steps involved in the process.\
            Add colors to the different stages of the process to make it visually appealing."

    LLM_response = get_gemini_response(question)
    print("Response:\n", LLM_response)

    formatted_dot_code = format_dot_code(LLM_response)

    svg_filename = "flowchart.svg"
    save_and_open_svg(formatted_dot_code, svg_filename)

    png_filename = "flowchart.png"
    save_png(formatted_dot_code, png_filename)


if __name__ == "__main__":
    main()


###########################################################################
# Getting started
# Basic question to test the model
###########################################################################
# question = "Can you take an example of car wash workflow. You can include the arrival, waiting time, various stages, inspection, payment and exit as steps. Please Convert Bullet Points to Graphviz DOT Language"
# question = "Can you take an example of ticket issuing system with queue. You can include various steps involved from entry to exit and detail it as much as possible. Please use Graphviz DOT Language"
# question = "I need to generate the process diagram for call center. \
#             you are an expert in optimizing the process and management. please help with the process flow diagram.\
#             Please use Graphviz DOT Language. Try to make it as \
#             detailed as possible with all the steps involved in the process."

# question = "you are an expert in data science and machine learning. Can you help me with the process flow diagram for a machine learning project? Please use Graphviz DOT Language. Try to make it as detailed as possible with all the steps involved in the process.\
#             You can make it as detailed as possible with all the steps involved in the process."

# question = "you are an expert in banking and financial systems. Can you help me with the process flow diagram for a \
#             banking system espeically focussing on the credit card approval process?\
#             Please use Graphviz DOT Language. Try to make it as detailed as possible with all the steps involved in the process.\
#             Add colors to the different stages of the process to make it visually appealing."

# question = "you are an expert in ecommerce marketing and supply chain. Can you help me with the process flow diagram for a \
#             on how companies like amazon and flipkart manage their supply chain?\
#             Please use Graphviz DOT Language. Try to make it as detailed as possible with all the steps involved in the process.\
#             Add colors to the different stages of the process to make it visually appealing."


################################################################################################
# Here's a detailed explanation of the selected Python code:
################################################################################################

# The code starts by importing necessary modules. io, os, json, requests are standard Python libraries. google.generativeai is a hypothetical module for interacting with Google's generative AI models. fastapi is a modern, fast (high-performance), web framework for building APIs with Python 3.6+ based on standard Python type hints.

# The code fetches a Google API key from an environment variable and configures the genai module with this key.

# The get_gemini_response function is defined. This function takes an input string, uses the genai.GenerativeModel class to create a model named "gemini-1.5-flash-latest", and generates content based on the input string. The generated content is returned as text.

# A series of commented-out questions are provided. These are presumably examples of the kind of input that the get_gemini_response function can handle.

# A question is defined and passed to the get_gemini_response function. The response is printed to the console.

# The response is formatted by removing certain characters and whitespace. The rankdir attribute in the response is modified to display the workflow vertically.

# The requests module is used to send a POST request to the QuickChart API, which can generate images from Graphviz DOT code. The DOT code from the response is included in the request payload.

# If the response from the QuickChart API is an SVG image, the image is saved to a file and an HTML file is created to display the image. The HTML file is opened in the default web browser.

# The process is repeated to save a PNG version of the image. If the response from the QuickChart API is a PNG image, the image is saved to a file.

# This code is designed to generate a flowchart from a description of a process, using a generative AI model to create the flowchart in Graphviz DOT language and the QuickChart API to convert the DOT language into an image.


################################################################################################
# Raw code - Archived
################################################################################################

# import io
# import os
# import google.generativeai as genai
# import json
# import requests
# from fastapi import FastAPI
# import webbrowser


# os.getenv("GOOGLE_API_KEY")
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# def get_gemini_response(input):
#     """Get the response from the Gemini model for the given input text."""

#     model = genai.GenerativeModel("gemini-1.5-flash-latest")
#     response = model.generate_content(input)
#     return response.text


# question = "I am writing a blog on how to use gemini flash LLM model to generate the flow digram using graphviz. As part of this\
#             i want to represnt the process in the form og diagram for easy understanding to the readers.\
#             Please use Graphviz DOT Language. Try to make it as detailed as possible with all the steps involved in the process.\
#             Make the different stages of the process clear and easy to understand. Add some colors to make it visually appealing."

# LLM_response = get_gemini_response(question)

# print("Response:\n", LLM_response)


# # Remove triple backticks and 'dot' text
# formatted_out = LLM_response.strip("```dot")
# # Remove leading and trailing whitespace
# formatted_out = formatted_out.strip()


# # Find the line containing the rankdir attribute
# lines = formatted_out.split("\n")
# for i, line in enumerate(lines):
#     if "rankdir" in line:
#         # Modify the rankdir attribute to display the workflow vertically
#         lines[i] = "    rankdir=TB;"

# # Reconstruct the DOT code with the modified rankdir attribute
# formatted_out = "\n".join(lines)


# quickchart_url = "https://quickchart.io/graphviz"


# dot_code = formatted_out
# # Prepare the payload for QuickChart
# post_data = {"graph": dot_code}

# try:
#     # Send the POST request to QuickChart API
#     response = requests.post(quickchart_url, json=post_data, verify=False)
#     response.raise_for_status()

#     # Check if the response content is SVG
#     content_type = response.headers.get("content-type", "").lower()
#     if "image/svg+xml" in content_type:
#         # Save SVG content to a file
#         svg_content = response.text
#         with open("flowchart.svg", "w", encoding="utf-8") as f:
#             f.write(svg_content)

#         # Open the HTML file in the default web browser
#         html_content = f"""
#         <!DOCTYPE html>
#         <html>
#         <head>
#             <title>Flowchart</title>
#         </head>
#         <body>
#             <img src="flowchart.svg" alt="Flowchart">
#         </body>
#         </html>
#         """
#         with open("flowchart.html", "w", encoding="utf-8") as f:
#             f.write(html_content)

#         webbrowser.open("file://" + os.path.realpath("flowchart.html"))

#     else:
#         print("Unexpected response content type:", content_type)

# except requests.exceptions.RequestException as e:
#     print(f"An error occurred: {e}")


# ##### Saving PNG image

# post_data = {"graph": dot_code, "format": "png"}  # Specify that we want a PNG image

# try:
#     # Send the POST request to QuickChart API
#     response = requests.post(quickchart_url, json=post_data, verify=False)
#     response.raise_for_status()

#     # Check if the response content is PNG
#     content_type = response.headers.get("content-type", "").lower()
#     if "image/png" in content_type:
#         # Save PNG content to a file
#         png_content = response.content
#         with open("flowchart.png", "wb") as f:
#             f.write(png_content)
#         print("Flowchart saved as flowchart.png")
#     else:
#         print("Unexpected response content type:", content_type)

# except requests.exceptions.RequestException as e:
#     print(f"An error occurred: {e}")
