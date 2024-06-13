from setuptools import setup, find_packages

setup(
    name='polyeval',
    version='0.1.0',
    author='everfly',
    author_email='tagriver@gmail.com',
    description='A Python library for evaluating RAG outputs of Chinese large language models.',
    long_description_content_type='text/markdown',
    url='https://github.com/evalsone/polyeval',
    packages=find_packages(exclude=['cookbook', 'docs']),
    include_package_data=True,
    exclude_package_data={'': ['.DS_Store']},
    install_requires=[
        'datasets',
        'openai',
        'unionllm',
        'jsonschema',
        'langchain',
        'langchain_community',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.7',
)