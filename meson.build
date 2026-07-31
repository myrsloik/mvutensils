project('MVUtensils', 'cpp',
    default_options: [
        'buildtype=release', 'warning_level=2', 'b_lto=true', 'b_ndebug=if-release', 'cpp_std=c++20',
    ],
    license: 'GPL-2.0-or-later',
    license_files: 'LICENSE',
    meson_version: '>=1.2.3',
    version: '6',
)

py = import('python').find_installation(pure: false)

incdir = include_directories(
    run_command(py, '-c', 'import vapoursynth as vs; print(vs.get_include())', check: true).stdout().strip(),
)

deps = [
    dependency('zimg', version: '>=3.0.5'),
    dependency('fftw3f', version: '>=3.3.5')
]

add_project_arguments('-DPACKAGE_VERSION="@0@"'.format(meson.project_version()), language: ['cpp'])

add_project_arguments('-fno-math-errno', language: ['cpp'])

install_dir = py.get_install_dir() / 'vapoursynth/plugins/mvutensils'

# VapourSynth variant-selection manifest; installed alongside the DLLs so the autoloader
# can pick the right ISA build from this subdirectory.
install_data('manifest.vs', install_dir: install_dir)

common_kwargs = {
    'gnu_symbol_visibility': 'hidden',
    'include_directories': incdir,
}

# Per-ISA hand-written kernels. They are runtime-dispatched, so even a baseline build contains them.
avx2_sources = files(
    'src/Overlap_AVX2.cpp',
    'src/SADFunctions_AVX2.cpp',
    'src/FlowShared_AVX2.cpp',
)
avx512_sources = files(
    'src/SADFunctions_AVX512.cpp',
    'src/FlowShared_AVX512.cpp',
)

# Float-pixel SAD baseline: built with -ffast-math (the clang/gcc equivalent of MSVC /fp:fast) so the
# scalar reduction reassociates and auto-vectorizes. Isolated in its own TU so fast-math touches nothing
# else. On x86 the AVX2/AVX-512 hand kernels (in the avx2/avx512 libs, NOT fast-math) override it.
floatsad_sources = files('src/SADFunctions_Float.cpp')

main_sources = files(
    'src/Analyse.cpp',
    'src/Compensate.cpp',
    'src/CopyCode.cpp',
    'src/CPU.cpp',
    'src/Degrains.cpp',
    'src/DepanAnalyse.cpp',
    'src/DepanCompensate.cpp',
    'src/DepanEstimate.cpp',
    'src/DepanShared.cpp',
    'src/DepanStabilise.cpp',
    'src/EntryPoint.cpp',
    'src/Flow.cpp',
    'src/FlowBlur.cpp',
    'src/FlowFPS.cpp',
    'src/FlowInter.cpp',
    'src/Mask.cpp',
    'src/MaskResize.cpp',
    'src/MotionBlockPyramid.cpp',
    'src/Recalculate.cpp',
    'src/SCDetection.cpp',
    'src/Super.cpp',
    'src/SuperPyramid.cpp',
    'src/Overlap.cpp',
    'src/SADFunctions.cpp',
)

if host_machine.cpu_family() == 'x86_64'
    add_project_arguments('-DMVTOOLS_X86', language: ['cpp'])

    # ISA ladder. The effective -march of a translation unit is the HIGHEST of (its own minimum
    # level, the variant's level); -march=znver4 sits at the top so it trumps every -march below it.
    # -mprefer-vector-width=512 and -ffast-math are orthogonal tuning flags and kept on their TUs in
    # every variant.
    march_levels = {
        '0': [],                                                     # baseline (generic x86-64, SSE2)
        '1': ['-march=x86-64-v3'],                                   # AVX2
        '2': ['-march=x86-64-v4'],                                   # AVX-512
        '3': ['-march=znver4', '-mno-sse4a', '-mno-avx512bf16'],  # Zen4 whole-program, highest level
    }
    avx2_min_level = 1
    avx512_min_level = 2

    # The three whole-library variants built on x64; all three are always compiled and installed.
    # They share one VapourSynth plugin id (com.vapoursynth.mvutensils), so only one can be active
    # under the `mvu` namespace per session -- load the wanted build explicitly rather than letting
    # all three autoload from the same directory.
    variants = [
        { 'name': 'mvutensils',        'tag': 'baseline', 'level': 0 },
        { 'name': 'mvutensils.avx2',   'tag': 'avx2',     'level': 1 },
        { 'name': 'mvutensils.zn4',    'tag': 'znver4',   'level': 3 },
    ]

    foreach v : variants
        g = v['level']
        tag = v['tag']

        avx2_level = (avx2_min_level > g) ? avx2_min_level : g
        avx512_level = (avx512_min_level > g) ? avx512_min_level : g

        vlibs = []

        vlibs += static_library('avx2_kernels_' + tag,
            avx2_sources,
            cpp_args: march_levels[avx2_level.to_string()],
            kwargs: common_kwargs,
        )

        vlibs += static_library('avx512_kernels_' + tag,
            avx512_sources,
            cpp_args: march_levels[avx512_level.to_string()] + ['-mprefer-vector-width=512'],
            kwargs: common_kwargs,
        )

        vlibs += static_library('floatsad_' + tag,
            floatsad_sources,
            cpp_args: march_levels[g.to_string()] + ['-ffast-math'],
            kwargs: common_kwargs,
        )

        shared_module(v['name'],
            main_sources,
            cpp_args: march_levels[g.to_string()],
            dependencies: deps,
            link_with: vlibs,
            install: true,
            install_dir: install_dir,
            name_prefix: '',
            kwargs: common_kwargs,
        )
    endforeach
else
    if host_machine.cpu_family() == 'aarch64'
        add_project_arguments('-DMVTOOLS_ARM', language: ['cpp'])
    endif

    floatsad = static_library('floatsad',
        floatsad_sources,
        cpp_args: '-ffast-math',
        kwargs: common_kwargs,
    )

    shared_module('mvutensils',
        main_sources,
        dependencies: deps,
        link_with: floatsad,
        install: true,
        install_dir: install_dir,
        name_prefix: '',
        kwargs: common_kwargs,
    )
endif
