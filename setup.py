from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

try: # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError: # for pip <= 9.0.3
    from pip.req import parse_requirements

install_reqs = parse_requirements("requirements.txt", session=False)

try:
    requirements = [str(ir.req) for ir in install_reqs]
except AttributeError:
    requirements = [str(ir.requirement) for ir in install_reqs]

setup(
    name='cta-model',
    version='0.0.1',
    ext_modules=cythonize(
        [
            Extension("cta_model.*", ['cta_model/*/*.py']),
        ],
        build_dir="build",
        compiler_directives=dict(
            always_allow_keywords=True,
            language_level=3
        ),
    ),
    cmdclass=dict(
        build_ext=build_ext
    ),
    packages=[],
    include_package_data=True,
    license='MIT',
    author='author',
    author_email='author@seek-data.com',
    description='cta model',
    # install_requires=requirements,
)
