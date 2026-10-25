"""Verify and materialize a source bundle without unsafe tar extraction."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import tarfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def safe_name(name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or not p.parts or str(p) != name:
        raise ValueError(f'Unsafe archive member: {name}')
    return p


def verify_tree(root, manifest):
    root = Path(root).resolve()
    marker = root / '.rsi-bundle.json'
    if not marker.is_file() or marker.is_symlink() or json.loads(marker.read_text()) != manifest:
        raise ValueError('Bundle metadata differs from verified manifest')
    observed = set()
    for base, dirs, files in os.walk(root, followlinks=False):
        for name in files + [d for d in dirs if (Path(base)/d).is_symlink()]:
            relative = str((Path(base)/name).relative_to(root))
            if relative != '.rsi-bundle.json':
                observed.add(relative)
    if observed != set(manifest['files']):
        raise ValueError('Unexpected or missing files in extracted bundle')
    for name, expected in manifest['files'].items():
        safe_name(name)
        path = root / name
        if name in manifest.get('symlinks', {}):
            if not path.is_symlink() or os.readlink(path) != manifest['symlinks'][name]:
                raise ValueError(f'Symlink changed: {name}')
            data = ('symlink:' + os.readlink(path)).encode()
        else:
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ValueError(f'Unexpected symlink: {name}')
            data = path.read_bytes()
        if sha(data) != expected:
            raise ValueError(f'Bundle file changed: {name}')


def materialize(archive, manifest_path, destination):
    archive, destination = Path(archive), Path(destination)
    manifest = json.loads(Path(manifest_path).read_text())
    if sha(archive.read_bytes()) != manifest['archive_sha256']:
        raise ValueError('Archive SHA256 mismatch')
    marker = destination / '.rsi-bundle.json'
    if destination.exists():
        if not marker.exists() or json.loads(marker.read_text()) != manifest:
            raise ValueError('Refusing nonmatching existing destination')
        verify_tree(destination, manifest)
        return manifest
    destination.mkdir(parents=True)
    # Inspect the complete table before writing; no hardlinks/devices/fifos.
    with tarfile.open(archive) as tar:
        entries = tar.getmembers()
        names = [m.name for m in entries]
        if len(names) != len(set(names)) or set(names) != set(manifest['files']):
            raise ValueError('Archive/member manifest mismatch')
        for m in entries:
            safe_name(m.name)
            if m.issym():
                target = posixpath.normpath(posixpath.join(posixpath.dirname(m.name), m.linkname))
                safe_name(target)
                if m.linkname.startswith('/') or manifest.get('symlinks', {}).get(m.name) != m.linkname:
                    raise ValueError('Invalid symlink target')
            elif not m.isfile():
                raise ValueError(f'Unsupported member type: {m.name}')
        # Write regular files before symlinks, refusing aliasing through parents.
        for m in entries:
            if not m.isfile():
                continue
            path = destination / m.name
            path.parent.mkdir(parents=True, exist_ok=True)
            data = tar.extractfile(m).read()
            if sha(data) != manifest['files'][m.name]:
                raise ValueError(f'Member hash mismatch: {m.name}')
            path.write_bytes(data)
            path.chmod(0o444)
        for m in entries:
            if m.issym():
                path=destination/m.name
                path.parent.mkdir(parents=True,exist_ok=True)
                path.symlink_to(m.linkname)
    marker.write_text(json.dumps(manifest,sort_keys=True)+'\n')
    marker.chmod(0o444)
    verify_tree(destination,manifest)
    for base, dirs, _ in os.walk(destination,followlinks=False):
        Path(base).chmod(0o555)
    return manifest
