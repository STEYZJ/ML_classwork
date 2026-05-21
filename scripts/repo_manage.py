#!/usr/bin/env python3
"""Repository metadata, version, branch, and GitHub About helper."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
META_PATH = ROOT / ".github" / "repo-meta.json"
README_PATH = ROOT / "README.md"
VERSION_PATH = ROOT / "VERSION"
README_START = "<!-- repo-meta:start -->"
README_END = "<!-- repo-meta:end -->"


class RepoManageError(RuntimeError):
    pass


def git(args: list[str], *, check: bool = True, capture: bool = True) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )
    if check and result.returncode != 0:
        message = result.stderr.strip() if result.stderr else "git command failed"
        raise RepoManageError(f"git {' '.join(args)}: {message}")
    return result.stdout.strip() if capture and result.stdout else ""


def load_meta() -> dict[str, Any]:
    try:
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RepoManageError(f"metadata file is missing: {META_PATH}") from exc
    except json.JSONDecodeError as exc:
        raise RepoManageError(f"metadata file is not valid JSON: {exc}") from exc

    required = [
        ("repository", "name"),
        ("repository", "url"),
        ("repository", "description"),
        ("repository", "topics"),
        ("version", "current"),
        ("version", "tag_prefix"),
        ("branches", "default"),
        ("branches", "allowed_prefixes"),
    ]
    for section, key in required:
        if section not in meta or key not in meta[section]:
            raise RepoManageError(f"missing metadata key: {section}.{key}")
    return meta


def save_meta(meta: dict[str, Any]) -> None:
    META_PATH.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def version_from_meta(meta: dict[str, Any]) -> str:
    return str(meta["version"]["current"]).strip()


def tag_for_version(meta: dict[str, Any], version: str | None = None) -> str:
    version = version or version_from_meta(meta)
    return f"{meta['version'].get('tag_prefix', 'v')}{version}"


def format_readme_block(meta: dict[str, Any]) -> str:
    repo = meta["repository"]
    version = version_from_meta(meta)
    topics = " ".join(f"`{topic}`" for topic in repo["topics"])
    return "\n".join(
        [
            README_START,
            f"- 当前版本：`{version}`",
            f"- 稳定分支：`{meta['branches']['default']}`",
            f"- 远程仓库：`{repo['url']}`",
            f"- GitHub About：{repo['description']}",
            f"- Topics：{topics}",
            README_END,
        ]
    )


def update_version_file(meta: dict[str, Any], *, check: bool) -> bool:
    expected = version_from_meta(meta) + "\n"
    current = VERSION_PATH.read_text(encoding="utf-8") if VERSION_PATH.exists() else ""
    if current == expected:
        return False
    if check:
        raise RepoManageError("VERSION does not match .github/repo-meta.json")
    VERSION_PATH.write_text(expected, encoding="utf-8")
    return True


def update_readme(meta: dict[str, Any], *, check: bool) -> bool:
    readme = README_PATH.read_text(encoding="utf-8")
    block = format_readme_block(meta)
    if README_START not in readme or README_END not in readme:
        raise RepoManageError("README metadata markers are missing")

    pattern = re.compile(
        re.escape(README_START) + r".*?" + re.escape(README_END),
        flags=re.S,
    )
    updated = pattern.sub(block, readme, count=1)
    if updated == readme:
        return False
    if check:
        raise RepoManageError("README metadata block is out of date")
    README_PATH.write_text(updated, encoding="utf-8")
    return True


def validate_branch(meta: dict[str, Any], branch: str | None) -> None:
    if not branch:
        branch = git(["branch", "--show-current"], check=False)
    if not branch:
        return
    allowed = [meta["branches"]["default"], *meta["branches"]["allowed_prefixes"]]
    if branch == meta["branches"]["default"]:
        return
    if any(branch.startswith(prefix) for prefix in meta["branches"]["allowed_prefixes"]):
        return
    raise RepoManageError(
        f"branch '{branch}' is not allowed; use one of: {', '.join(allowed)}"
    )


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value or "update"


def create_branch(meta: dict[str, Any], kind: str, title: str, *, push: bool) -> str:
    prefix = f"{kind}/"
    if prefix not in meta["branches"]["allowed_prefixes"]:
        raise RepoManageError(
            f"unsupported branch kind '{kind}'; allowed kinds: "
            + ", ".join(p.rstrip("/") for p in meta["branches"]["allowed_prefixes"])
        )
    branch = f"{prefix}{date.today():%Y%m%d}-{slugify(title)}"
    git(["switch", "-c", branch], capture=False)
    if push:
        git(["push", "-u", "origin", branch], capture=False)
    return branch


def next_calendar_version(current: str) -> str:
    base = date.today().strftime("%Y.%m.%d")
    if current == base:
        return f"{base}.1"
    if current.startswith(base + "."):
        suffix = current.removeprefix(base + ".")
        if suffix.isdigit():
            return f"{base}.{int(suffix) + 1}"
    return base


def bump_version(meta: dict[str, Any], version: str | None) -> str:
    new_version = version or next_calendar_version(version_from_meta(meta))
    if not re.fullmatch(r"\d{4}\.\d{2}\.\d{2}(?:\.\d+)?", new_version):
        raise RepoManageError("version must look like YYYY.MM.DD or YYYY.MM.DD.N")
    meta["version"]["current"] = new_version
    save_meta(meta)
    update_version_file(meta, check=False)
    update_readme(meta, check=False)
    return new_version


def github_api(token: str, method: str, url: str, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "ML_classwork-repo-manage",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            if response.status >= 300:
                raise RepoManageError(f"GitHub API returned HTTP {response.status}")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RepoManageError(f"GitHub API HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RepoManageError(f"GitHub API request failed: {exc.reason}") from exc


def sync_about(meta: dict[str, Any]) -> None:
    token = os.environ.get("REPO_ADMIN_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        raise RepoManageError(
            "REPO_ADMIN_TOKEN is required to sync GitHub About metadata"
        )

    repo = meta["repository"]
    api_root = f"https://api.github.com/repos/{repo['name']}"
    github_api(
        token,
        "PATCH",
        api_root,
        {
            "description": repo["description"],
            "homepage": repo.get("homepage", repo["url"]),
            "has_issues": True,
            "has_projects": False,
            "has_wiki": False,
        },
    )
    github_api(token, "PUT", f"{api_root}/topics", {"names": repo["topics"]})


def tag_version(meta: dict[str, Any], *, push: bool) -> str:
    update_version_file(meta, check=True)
    tag = tag_for_version(meta)
    exists = git(["rev-parse", "-q", "--verify", f"refs/tags/{tag}"], check=False)
    if not exists:
        git(["tag", "-a", tag, "-m", f"Release {tag}"], capture=False)
    if push:
        remote_exists = git(
            ["ls-remote", "--tags", "origin", f"refs/tags/{tag}"],
            check=False,
        )
        if not remote_exists:
            git(["push", "origin", tag], capture=False)
    return tag


def command_check(args: argparse.Namespace) -> None:
    meta = load_meta()
    update_version_file(meta, check=True)
    update_readme(meta, check=True)
    validate_branch(meta, args.branch)
    print("repository metadata check passed")


def command_update(_: argparse.Namespace) -> None:
    meta = load_meta()
    changed = [
        update_version_file(meta, check=False),
        update_readme(meta, check=False),
    ]
    print("repository metadata updated" if any(changed) else "repository metadata is current")


def command_bump(args: argparse.Namespace) -> None:
    version = bump_version(load_meta(), args.version)
    print(f"version set to {version}")


def command_branch(args: argparse.Namespace) -> None:
    branch = create_branch(load_meta(), args.kind, args.title, push=args.push)
    print(f"created branch {branch}")


def command_sync_about(_: argparse.Namespace) -> None:
    sync_about(load_meta())
    print("GitHub About metadata synced")


def command_tag(args: argparse.Namespace) -> None:
    tag = tag_version(load_meta(), push=args.push)
    print(f"version tag ready: {tag}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check", help="validate metadata, version, README, branch")
    check.add_argument("--branch", help="branch name to validate")
    check.set_defaults(func=command_check)

    update = subparsers.add_parser("update", help="rewrite VERSION and README metadata")
    update.set_defaults(func=command_update)

    bump = subparsers.add_parser("bump", help="bump calendar version")
    bump.add_argument("--version", help="explicit version, e.g. 2026.05.21.1")
    bump.set_defaults(func=command_bump)

    branch = subparsers.add_parser("branch", help="create a managed branch")
    branch.add_argument("kind", help="branch kind: work, docs, experiment, fix, release")
    branch.add_argument("title", help="short branch title")
    branch.add_argument("--push", action="store_true", help="push branch and set upstream")
    branch.set_defaults(func=command_branch)

    sync_about_parser = subparsers.add_parser("sync-about", help="sync GitHub About")
    sync_about_parser.set_defaults(func=command_sync_about)

    tag = subparsers.add_parser("tag", help="create a version tag from VERSION")
    tag.add_argument("--push", action="store_true", help="push tag to origin")
    tag.set_defaults(func=command_tag)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except RepoManageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
