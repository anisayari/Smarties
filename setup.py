from setuptools import setup, find_packages

setup(name='Smarties',

      version='1.2',

      url='https://github.com/anisayari/Smarties',

      license='GPL3',

      author='Anis Ayari',

      author_email='anis.ayari.pro@gmail.com',

      description='A Smart AI Text Learner and Classifier',

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