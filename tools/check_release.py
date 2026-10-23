#!/usr/bin/env python3
"""Validate the GitHub/Python release mapping and actual distribution files."""
import argparse
import ast
from email.parser import BytesParser
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def versions(source):
    return {node.targets[0].id: ast.literal_eval(node.value)
            for node in ast.parse(source).body if isinstance(node, ast.Assign)}


def metadata_matches(raw, release):
    metadata = BytesParser().parsebytes(raw)
    require(metadata['Name'] == release['package_name'], 'distribution name mismatch')
    require(metadata['Version'] == release['package_version'], 'distribution version mismatch')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dist', type=Path)
    args = parser.parse_args()
    release = json.loads((ROOT / 'release.json').read_text())
    expected = {'__version__': release['package_version'], '__release__': release['source_release']}
    require(versions((ROOT / 'python/src/symft/_version.py').read_text()) == expected,
            'release.json does not match the Python source version')
    require((ROOT / release['notes']).is_file(), 'release notes are missing')
    ref = os.environ.get('RELEASE_REF', '')
    if ref.startswith('refs/tags/'):
        require(ref == 'refs/tags/' + release['tag'], 'tag does not match release.json')
    if os.environ.get('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'], 'a') as output:
            values = {'tag': release['tag'], 'version': release['package_version'],
                      'notes': release['notes'],
                      'wheel-count': 20 if os.environ.get('FULL_MATRIX') == 'true' else 4}
            for key, value in values.items():
                print(f'{key}={value}', file=output)
    if args.dist is None:
        print(f"PASS {release['tag']} -> {release['package_name']} {release['package_version']} ({release['source_release']})")
        return

    from packaging.utils import parse_wheel_filename
    wheel_count = int(os.environ.get('EXPECTED_WHEELS', '20'))
    require(wheel_count in (4, 20), 'unsupported release matrix')
    pythons = ['cp39', 'cp310', 'cp311', 'cp312', 'cp313'] if wheel_count == 20 else ['cp313']
    platforms = ['linux_x86_64', 'linux_aarch64', 'macos_arm64', 'win_amd64']
    required_matrix = {(python, platform) for python in pythons for platform in platforms}
    found_matrix = set()
    wheels = sorted(args.dist.glob('*.whl'))
    sdists = sorted(args.dist.glob('*.tar.gz'))
    require(len(wheels) == wheel_count and len(sdists) == 1, 'incomplete wheel/sdist matrix')
    for path in wheels:
        name, version, _, tags = parse_wheel_filename(path.name)
        require(str(name) == release['package_name'] and str(version) == release['package_version'],
                'wheel filename has incorrect identity: ' + path.name)
        pairs = set()
        for tag in tags:
            require(tag.abi == tag.interpreter, 'unexpected wheel ABI: ' + path.name)
            platform = tag.platform
            if platform.startswith('manylinux'):
                if platform.endswith('_aarch64'):
                    platform = 'linux_aarch64'
                elif platform.endswith('_x86_64'):
                    platform = 'linux_x86_64'
            elif platform.startswith('macosx') and platform.endswith('_arm64'):
                platform = 'macos_arm64'
            pairs.add((tag.interpreter, platform))
        require(len(pairs) == 1 and pairs <= required_matrix and not pairs & found_matrix,
                'unexpected or duplicate wheel platform: ' + path.name)
        found_matrix.update(pairs)
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            metadata = [name for name in names if name.endswith('.dist-info/METADATA')]
            require(len(metadata) == 1, 'missing/ambiguous wheel metadata')
            metadata_matches(archive.read(metadata[0]), release)
            require(versions(archive.read('symft/_version.py').decode()) == expected,
                    'wheel contains the wrong source version')
            require(all(name in names for name in ('symft/py.typed', 'symft/_native.pyi')),
                    'wheel is missing type information')
            require(any(name.startswith('symft/_native.') and name.endswith(('.so', '.pyd')) for name in names),
                    'wheel is missing its native extension')
    require(found_matrix == required_matrix, 'wheel matrix does not match supported platforms')
    with tarfile.open(sdists[0]) as archive:
        prefix = f"{release['package_name']}-{release['package_version']}/"
        metadata_matches(archive.extractfile(prefix + 'PKG-INFO').read(), release)
        require(versions(archive.extractfile(prefix + 'src/symft/_version.py').read().decode()) == expected,
                'sdist contains the wrong source version')
        for name in ('LICENSE', 'cpp/src/core/common.cpp', 'cpp/src/cuda/cuda_runtime.cu'):
            require(archive.getmember(prefix + name).isfile(), 'sdist is missing ' + name)
    files = wheels + sdists
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    manifest = dict(release, source_commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        workflow_run_id=os.environ.get('GITHUB_RUN_ID'),
        wheel_build={'native': False, 'cuda': False, 'python_tags': pythons, 'platforms': platforms},
        sha256=hashes)
    manifest_path = args.dist / 'release-manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    files.append(manifest_path)
    (args.dist / 'SHA256SUMS').write_text(''.join(
        f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n'
        for path in sorted(files)))
    print(f'PASS {len(wheels)} wheels, one sdist, source identity, metadata, matrix and SHA256 manifest')


if __name__ == '__main__':
    main()
