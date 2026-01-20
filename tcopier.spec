# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, collect_dynamic_libs

block_cipher = None

# Collect files for en_core_web_sm
datas_sm = collect_data_files('en_core_web_sm')
hiddenimports_sm = collect_submodules('en_core_web_sm')
hiddenimports_bbs = collect_submodules("bbstrader")
binaries_bbs = collect_dynamic_libs("bbstrader")
binaries_sm = collect_dynamic_libs('en_core_web_sm')

explicit_hidden_imports = [
    'bbstrader.api.metatrader_client', 
    'MetaTrader5'
]



a = Analysis(
    ['bbstrader/metatrader/_copier.py'],
    pathex=[],
    binaries=binaries_sm + binaries_bbs,
    datas=[('bbstrader/assets', 'assets')] + datas_sm,
    hiddenimports=hiddenimports_sm + hiddenimports_bbs + explicit_hidden_imports,
    hookspath=['hooks'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='tcopier',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False, # --windowed
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='bbstrader/assets/bbstrader.ico',
)
