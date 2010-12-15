# This SConstruct is for faster parallel builds.
# Use "setup.py" for normal installation.
#
# module     -- build shared library object, srreal_ext.so
# develop    -- srreal_ext.so under diffpy/srreal/ directory

import os

def subdictionary(d, keyset):
    return dict([kv for kv in d.items() if kv[0] in keyset])

# copy system environment variables related to compilation
DefaultEnvironment(ENV=subdictionary(os.environ, [
    'PATH', 'PYTHONPATH',
    'CPATH', 'CPLUS_INCLUDE_PATH',
    'LD_LIBRARY_PATH', 'LIBRARY_PATH',
    ])
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

# Use Intel C++ compiler when it is available
icpc = env.WhereIs('icpc')
if icpc:
    env.Tool('intelc', topdir=icpc[:icpc.rfind('/bin')])

# Declare external libraries.
good_python_flags = lambda n : (
    n not in ('-g', '-Wstrict-prototypes'))
env.ParseConfig("python-config --cflags")
env.Replace(CCFLAGS=filter(good_python_flags, env['CCFLAGS']))
env.Replace(CPPDEFINES='')
env.AppendUnique(LIBS=['libdiffpy'])

# Compiler specific options
if icpc:
    # options for Intel C++ compiler on hpc dev-intel07
    env.AppendUnique(CCFLAGS=['-w1', '-fp-model', 'precise'])
    env.PrependUnique(LIBS=['imf'])
    fast_optimflags = ['-fast', '-no-ipo']
else:
    # g++ options
    env.AppendUnique(CCFLAGS=['-Wall'])
    fast_optimflags = ['-ffast-math']

# Configure build variants
if env['build'] == 'debug':
    env.AppendUnique(CCFLAGS='-g')
elif env['build'] == 'fast':
    env.AppendUnique(CCFLAGS=['-O3'] + fast_optimflags)
    env.AppendUnique(CPPDEFINES='NDEBUG')
    env.AppendUnique(LINKFLAGS='-s')

if env['profile']:
    env.AppendUnique(CCFLAGS='-pg')
    env.AppendUnique(LINKFLAGS='-pg')

builddir = env.Dir('build/' + env['build'])
Export('env')

env.SConscript('srrealmodule/SConscript', variant_dir=builddir)

test = env.Alias('test', env.Alias('develop'), 'alltests.py tests')
AlwaysBuild(test)

# vim: ft=python
