import ast
import astor
import os


class MethodExtractor(ast.NodeVisitor):
    def __init__(self, method_name):
        self.method_name = method_name
        self.method_code = None

    def visit_ClassDef(self, node):
        # Check each method in the class
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == self.method_name:
                # Get the source code of the method
                self.method_code = astor.to_source(item)
                break
        # Continue to visit other classes (if nested)
        self.generic_visit(node)

    def get_method_code(self, source_code):
        tree = ast.parse(source_code)
        self.visit(tree)
        # print(f"****self.method_code --> {self.method_code}****")
        return self.method_code
