import ast
import astor
import os
from tqdm import tqdm

from auto_tech_writer.method_extractor import MethodExtractor
from auto_tech_writer.llm_response import get_llm_response


class DocstringUpdater(ast.NodeTransformer):
    def __init__(self):

        self.updated_docstrings = []

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        if ast.get_docstring(node):
            return node  # Skip if the docstring already exists

        # Constructing the new docstring
        method_code = self.get_code(self.current_file, node.name)
        docstring = self.create_docstring(node, method_code)
        if docstring:
            node.body.insert(
                0, ast.Expr(value=ast.Constant(value=docstring, lineno=node.lineno))
            )
            self.updated_docstrings.append((node.name, docstring))
        return node

    def create_docstring(self, node, method_code=None):
        """Generate a docstring based on method arguments using LLM for detailed descriptions."""
        if not node.args.args:
            return None

        docstring = '"""\n'
        docstring += "Args:\n"
        print(f"Initializing docstring generation for --> {node.name}")
        # Generate a meaningful description for each argument using LLM
        for arg in tqdm(node.args.args, desc="Generating docstrings"):
            arg_name = arg.arg

            # Ensure that the method_code is dynamically extracted

            """Extract method code from a given Python file."""

            method_code = self.get_code(self.current_file, node.name)
            description = self.get_arg_description(arg_name, method_code)
            docstring += f"\t{arg_name} (type): {description}\n"
            print(f"Docstring generation for --> {arg_name}")

        docstring += '"""\n'
        return docstring

    def get_arg_description(self, arg_name, method_code):
        """Use an LLM to generate a meaningful description for a given argument."""
        prompt = (
            f"Given the following method code, please describe the argument '{arg_name}' "
            f"in a meaningful way.\n\n"
            f"Method code:\n{method_code}\n\n"
            f"Description for '{arg_name}':"
        )

        try:
            # Make the API call to the language model
            response = get_llm_response(prompt)

            # Extract the text from the response
            response_text = response.candidates[0].content.parts[0].text
            # return response_text

            # print(f"****response_text --> {response_text}****")

        except AttributeError as e:
            print(f"AttributeError: {str(e)}")
            return "An error occurred while generating the code explanation."

        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")
            return "An error occurred while generating the code explanation."

        # description = response.choices[0].text.strip()
        return response_text if response_text else f"Description of {arg_name}."

    def update_docstrings_in_file(self, file_path):
        """Update docstrings in a single Python file."""
        self.current_file = file_path
        with open(file_path, "r") as file:
            tree = ast.parse(file.read(), filename=file_path)

        self.visit(tree)

        # Write the updated code back to the file
        with open(file_path, "w") as file:
            file.write(astor.to_source(tree))

    def update_docstrings_in_directory(self, directory):
        """Update docstrings in all Python files in a directory."""
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    self.update_docstrings_in_file(file_path)
        print(f"-" * 50)
        print(f"Docstrings updated in all Python files in codebase '{directory}'.")
        print(f"-" * 50)

    def get_code(self, file_path, method_name):
        """Extract method code from a given Python file."""
        with open(file_path, "r") as file:
            source_code = file.read()

        extractor = MethodExtractor(method_name)
        method_code = extractor.get_method_code(source_code)

        return method_code
