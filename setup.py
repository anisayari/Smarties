from setuptools import setup, find_packages

setup(name='Smarties',

      version='0.3',

      url='https://github.com/anisayari/Smarties',

      license='GPUL3',

      author='Anis Ayari',

      author_email='anis.ayari.pro@gmail.com',

      description='Manage configuration files',

      packages=find_packages(exclude=['tests']),

      long_description=open('README.md').read(),

      zip_safe=False,
      install_requires=[
          'nltk',
          'gensim',
          'wikipedia',
          'scikit-learn',
          'pandas',
          'numpy',
      ],
      setup_requires=['nltk',
                      'gensim',
                      'wikipedia',
                      'scikit-learn',
                      'pandas',
                      'numpy'],
      test_suite='nose.collector')