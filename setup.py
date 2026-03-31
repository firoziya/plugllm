"""Setup configuration for PlugLLM"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_desc = f.read()

setup(
    name='plugllm',
    version='2.0.1',
    author='Yash Kumar Firoziya',
    author_email='ykfiroziya@gmail.com',
    description='Unified LLM API interface for OpenAI, Gemini, Mistral, Groq, Claude, and 10+ providers',
    long_description=long_desc,
    long_description_content_type='text/markdown',
    url='https://github.com/firoziya/plugllm',
    packages=find_packages(),
    install_requires=[
        "requests>=2.20.0",
    ],
    include_package_data=True,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Programming Language :: Python :: 3.14',
    ],
    python_requires='>=3.8',
    project_urls={
        'Bug Reports': 'https://github.com/firoziya/plugllm/issues',
        'Source': 'https://github.com/firoziya/plugllm',
        'Documentation': 'https://plugllm.firoziyash.life',
    },
    keywords=[
        'llm', 'openai', 'gemini', 'claude', 'groq', 'mistral',
        'ai', 'chatgpt', 'gpt-4', 'language-model', 'api-wrapper', 'firoziyash'
    ],
    license="MIT",
)