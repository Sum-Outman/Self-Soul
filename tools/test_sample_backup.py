"""Test file for bilingual comments tool"""

def calculate_sum(a, b):
    return a + b

def validate_input(data: str) -> bool:
    if not data:
        return False
    return len(data) > 0

class DataProcessor:
    def __init__(self, config):
        self.config = config
    
    def process(self, data):
        result = []
        for item in data:
            if self._filter(item):
                result.append(self._transform(item))
        return result
    
    def _filter(self, item):
        return item is not None
    
    def _transform(self, item):
        return str(item).upper()

@decorator
def decorated_function(x, y=10):
    return x * y

class AdvancedCalculator:
    """A calculator with advanced operations"""
    
    def add(self, x, y):
        return x + y
    
    def multiply(self, x, y):
        return x * y

def nested_function():
    def inner_func():
        return "inner"
    return inner_func()

class OuterClass:
    class InnerClass:
        def method(self):
            return "nested"
    
    def outer_method(self):
        return "outer"

async def async_function():
    await asyncio.sleep(1)
    return "done"

def function_with_type_hints(name: str, age: int = 30) -> dict:
    return {"name": name, "age": age}