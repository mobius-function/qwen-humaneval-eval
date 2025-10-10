import requests

prompt = '''def has_close_elements(numbers, threshold):
    """Check if any two numbers are closer than threshold."""
    # TODO: Implement the function body here
'''

response = requests.post('http://localhost:8000/v1/completions', json={
    'model': 'Qwen/Qwen2.5-Coder-0.5B',
    'prompt': prompt,
    'max_tokens': 100,
    'temperature': 0.2
})

print('RAW OUTPUT:')
print(repr(response.json()['choices'][0]['text']))
