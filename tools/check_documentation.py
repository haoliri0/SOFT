#!/usr/bin/env python3
"""Check repo-local Markdown links, release hygiene, and curated measurements."""
import ast
import re
from pathlib import Path
import subprocess
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def prose(text):
    lines, fence = [], None
    for line in text.splitlines():
        match = re.match(r'^\s*(`{3,}|~{3,})', line)
        if match:
            marker = match.group(1)
            if fence is None:
                fence = marker[0]
            elif marker[0] == fence:
                fence = None
            continue
        if fence is None:
            lines.append(line)
    require(fence is None, 'unclosed Markdown code fence')
    return '\n'.join(lines)


def anchors(text):
    result, seen = set(), {}
    for line in prose(text).splitlines():
        heading = re.match(r'^#{1,6}\s+(.+?)(?:\s+#+)?$', line)
        if heading:
            slug = re.sub(r'[^\w\- ]', '', heading.group(1).lower()).replace(' ', '-')
            count = seen.get(slug, 0)
            seen[slug] = count + 1
            result.add(slug if count == 0 else f'{slug}-{count}')
    return result


def main():
    names = subprocess.check_output(
        ['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], cwd=ROOT
    ).decode().split('\0')
    paths = sorted({Path(name) for name in names if name})
    binary_suffixes = {'.so', '.pyd', '.dylib', '.o', '.a', '.obj', '.lib', '.exe', '.cubin', '.ptx'}
    for path in paths:
        require(path.suffix not in binary_suffixes, 'build artifact in commit candidates: ' + str(path))
        require(not any(part.startswith('build-') or part in ('jit-cache', '.jit-cache', '__pycache__')
                        for part in path.parts), 'generated directory in commit candidates: ' + str(path))
    documents = [ROOT / path for path in paths if path.suffix.lower() == '.md']
    checked_links = 0
    for path in documents:
        body = path.read_text()
        require('/home/lirl/' not in body, 'machine-specific path in public docs: ' + str(path))
        for match in re.finditer(r'!?\[[^\]\n]*\]\(([^)\n]+)\)', prose(body)):
            target = match.group(1).strip().strip('<>')
            url = urlsplit(target)
            if url.scheme or url.netloc:
                continue
            require(not url.path.startswith('/'), 'absolute local document link: ' + target)
            file = (path.parent / unquote(url.path)).resolve() if url.path else path
            require(file.is_relative_to(ROOT), 'local link escapes repository: ' + target)
            require(file.exists(), f'{path.relative_to(ROOT)}: missing {target}')
            if url.fragment and file.suffix.lower() == '.md':
                require(unquote(url.fragment) in anchors(file.read_text()),
                        f'{path.relative_to(ROOT)}: missing anchor {target}')
            checked_links += 1
    require((ROOT / 'README.md').read_text().startswith('# SOFT\n'), 'project title is not SOFT')
    version_tree = ast.parse((ROOT / 'python/src/symft/_version.py').read_text())
    versions = {node.targets[0].id: ast.literal_eval(node.value) for node in version_tree.body
                if isinstance(node, ast.Assign)}
    require(versions == {'__version__': '2026.10.8', '__release__': 'symft_26_10_08'}, 'release identity mismatch')
    for name in ('README.md', 'CHANGELOG.md', 'python/README.md', 'docs/PROJECT.md'):
        require(versions['__release__'] in (ROOT / name).read_text(), 'release missing from ' + name)
    guide = (ROOT / 'python/README.md').read_text()
    for option in ('cpu_backend', 'cpu_real_gauge', 'cpu_hoist_detectors'):
        require(option in guide, 'missing CPU API option: ' + option)
        require(option in (ROOT / 'python/src/symft/_native.pyi').read_text(), 'missing type hint: ' + option)
    subprocess.run([sys.executable, str(ROOT / 'benchmark/validation/check_results.py')], check=True)
    subprocess.run([sys.executable, str(ROOT / 'benchmark/validation/check_cpu_comparison.py')], check=True)
    print(f'PASS {len(documents)} Markdown files, {checked_links} local links/anchors, API names, and artifact hygiene')
    print('External URLs and rendered GitHub layout are not checked by this script.')


if __name__ == '__main__':
    main()
