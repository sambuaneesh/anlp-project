import google.generativeai as genai

API_KEY = "api key here"
genai.configure(api_key=API_KEY)

print("Available models that support 'generateContent':\n")
for m in genai.list_models():
  if 'generateContent' in m.supported_generation_methods:
    print(m.name)