# This SConstruct is for faster parallel builds.
# Use "setup.py" for normal installation.
#
# module     -- build shared library object, srreal_ext.so
# develop    -- srreal_ext.so under diffpy/srreal/ directory

import os

# copy system environment variables related to compilation
DefaultEnvironment(ENV={
        'PATH' : os.environ['PATH'],
        'PYTHONPATH' : os.environ.get('PYTHONPATH', ''),
        'LD_LIBRARY_PATH' : os.environ.get('LD_LIBRARY_PATH', ''),
        'CPATH' : os.environ.get('CPATH', ''),
        'LIBRARY_PATH' : os.environ.get('LIBRARY_PATH', ''),
    }
)


# Create construction environment
env = DefaultEnvironment().Clone()

# Variables definitions below work only with 0.98 or later.
env.EnsureSConsVersion(0, 98)

# Customizable compile variables
vars = Variables('sconsvars.py')

vars.Add(EnumVariable('build',
    'compiler settings', 'debug',
    allowed_values=('debug', 'fast')))
vars.Add(BoolVariable('profile',
    'build with profiling information', False))
vars.Update(env)
env.Help(vars.GenerateHelpText(env))

# Declare external libraries.
good_python_flags = lambda n : (
    n not in ('-g', '-Wstrict-prototypes'))
env.ParseConfig("python-config --cflags")
env.Replace(CCFLAGS=filter(good_python_flags, env['CCFLAGS']))
env.Replace(CPPDEFINES='')
env.AppendUnique(LIBS='libdiffpy')
env.AppendUnique(LINKFLAGS=['-Wl,-O1', '-Wl,-Bsymbolic-functions'])

# g++ options
fast_optimflags = ['-ffast-math']

# Configure build variants
if env['build'] == 'debug':
    env.AppendUnique(CCFLAGS='-g')
elif env['build'] == 'fast':
    env.AppendUnique(CCFLAGS=['-O3'] + fast_optimflags)
    env.AppendUnique(CPPDEFINES='NDEBUG')

if env['profile']:
    env.AppendUnique(CCFLAGS='-pg')
    env.AppendUnique(LINKFLAGS='-pg')

builddir = env.Dir('build/' + env['build'])
Export('env')

env.SConscript('srrealmodule/SConscript', variant_dir=builddir)

# vim: ft=python
