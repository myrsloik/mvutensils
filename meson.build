project('MVGPUtensils', 'cpp',
    default_options: [
        'buildtype=release', 'warning_level=2', 'b_lto=true', 'b_ndebug=if-release', 'cpp_std=c++20',
    ],
    license: 'GPL-2.0-or-later',
    license_files: 'LICENSE',
    meson_version: '>=1.2.3',
    version: '1',
)

cpp = meson.get_compiler('cpp')
py = import('python').find_installation(pure: false)

# VapourSynth headers (API 4.3, for VSVulkan4.h): an explicit path wins, the Python package
# provides them otherwise.
vs_include = get_option('vapoursynth_include')
if vs_include == ''
    vs_include = run_command(py, '-c', 'import vapoursynth as vs; print(vs.get_include())', check: true).stdout().strip()
endif
incdirs = [include_directories(vs_include)]

# Only the Vulkan headers are needed: the plugin never links the loader, the VapourSynth core
# hands it every entry point already resolved.
deps = []
vulkan_include = get_option('vulkan_include')
if vulkan_include != ''
    incdirs += include_directories(vulkan_include)
else
    vulkan_dep = dependency('vulkan', required: false)
    if vulkan_dep.found()
        deps += vulkan_dep.partial_dependency(compile_args: true, includes: true)
    elif not cpp.has_header('vulkan/vulkan_core.h')
        error('Vulkan headers not found; install them or pass -Dvulkan_include=path/to/include')
    endif
endif

add_project_arguments('-DPACKAGE_VERSION="@0@"'.format(meson.project_version()), language: ['cpp'])
add_project_arguments('-DVS_USE_API_43', language: ['cpp'])
# The MSVC runtime deprecates getenv (the profiler's MVGPU_PROFILE); a no-op elsewhere
add_project_arguments('-D_CRT_SECURE_NO_WARNINGS', language: ['cpp'])

# The kernels, compiled at runtime by the core, are embedded as raw string literals. Listing them
# as inputs makes an edited kernel regenerate the header, and the header in the compiler's depfile
# rebuilds what includes it.
shader_sources = files(
    'src/shaders/compensate.comp',
    'src/shaders/degrain.comp',
    'src/shaders/degrain_common.glsl',
    'src/shaders/degrain_count.comp',
    'src/shaders/degrain_weights.comp',
    'src/shaders/flow_blur.comp',
    'src/shaders/flow_common.glsl',
    'src/shaders/flow_fetch.comp',
    'src/shaders/flow_inter.comp',
    'src/shaders/flow_prep.comp',
    'src/shaders/mask_blocks.comp',
    'src/shaders/mask_common.glsl',
    'src/shaders/mask_resize.comp',
    'src/shaders/median.comp',
    'src/shaders/pyr_common.glsl',
    'src/shaders/pyr_pass.comp',
    'src/shaders/pyr_reduce.comp',
    'src/shaders/pyr_seed.comp',
    'src/shaders/pyr_top.comp',
    'src/shaders/recalc.comp',
    'src/shaders/refine_apply.comp',
    'src/shaders/refine_common.glsl',
    'src/shaders/refine_fallback.comp',
    'src/shaders/refine_flag.comp',
    'src/shaders/refine_halfpel.comp',
    'src/shaders/refine_init.comp',
    'src/shaders/refine_pass.comp',
    'src/shaders/refine_qpel.comp',
    'src/shaders/sadf_common.glsl',
    'src/shaders/seed_build.comp',
    'src/shaders/seed_scatter.comp',
    'src/shaders/super.comp',
    'src/shaders/super_common.glsl',
    'src/shaders/super_qpel.comp',
)
shader_header = custom_target('shader_sources_h',
    input: shader_sources,
    output: 'ShaderSources.h',
    command: [py, files('src/shaders/embed_shaders.py'), '@OUTPUT@', '@INPUT@'],
)

# The plugin installs into the Python package's plugin directory, where the package's VapourSynth
# autoloads from; -Dplugindir overrides it for a plain meson install into some other VapourSynth.
plugindir = get_option('plugindir')
if plugindir == ''
    plugindir = py.get_install_dir() / 'vapoursynth/plugins'
endif

main_sources = files(
    'src/Analyse.cpp',
    'src/Compensate.cpp',
    'src/Degrain.cpp',
    'src/EntryPoint.cpp',
    'src/FilterShared.cpp',
    'src/Flow.cpp',
    'src/FlowInterpolate.cpp',
    'src/Mask.cpp',
    'src/Recalculate.cpp',
    'src/SCDetection.cpp',
    'src/Super.cpp',
    'src/SuperLayout.cpp',
    'src/VectorConvert.cpp',
    'src/VulkanContext.cpp',
)

shared_module('mvgputensils',
    main_sources, shader_header,
    include_directories: incdirs,
    dependencies: deps,
    gnu_symbol_visibility: 'hidden',
    install: true,
    install_dir: plugindir,
    name_prefix: '',
)

# The search's CPU reference, for test/check_reference.py; built on request (meson compile
# mvgpu_reference), never installed
executable('mvgpu_reference', 'test/reference/reference.cpp',
    build_by_default: false,
    install: false,
)
