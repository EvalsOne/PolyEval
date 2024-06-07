from setuptools import setup, find_packages

setup(
    name='zeval',
    version='0.1.5',
    author='everfly',
    author_email='tagriver@gmail.com',
    description='A Python library for evaluating RAG outputs of Chinese large language models.',
    # long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourgithub/zeval',
    packages=find_packages(),
    install_requires=[
        'datasets',
        'openai',
        'unionllm',
        'jsonschema',
        # 添加其他需要的依赖
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